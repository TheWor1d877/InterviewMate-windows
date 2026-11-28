"""语音采集的启停与状态。

见《5-语音输入与流式ASR》「F8 交互」「部分结果与最终结果」「状态显示」
与「性能」四节。

这个模块只做编排，不管渲染也不管按键绑定：

- 界面层绑定 F8 之后调 :meth:`AudioManager.toggle`，
  动作结果是 ``started`` / ``stopped`` / ``empty`` / ``error`` 四个标识之一，
  界面按这个标识更新状态栏；
- 部分结果走部分结果回调，只作临时显示，不提交；
- 最终文本走最终文本回调，接上去之后自动提交给问题处理模块，
  见《5-语音输入与流式ASR》「F8 交互」一节；
- 同一句最终文本不提交两次，见《8-集成与验收》「防止重复调用」一节。
  去重记忆按录音段算：新的一段录音开始就清掉，与识别状态一样从头开始。

采集与推理都不占住事件循环：采集在采集线程里读音频，
推理放在线程里跑，这里只做编排。设备枚举也放进线程执行器，
并按 ``audio.init_timeout`` 限时，见《8-集成与验收》「异步任务」
与「超时管理」两节。

来源标签按《5-语音输入与流式ASR》「音频来源」一节处理：
麦克风与系统回环分开采集时各自保留标签，
这个模块按标签分开记文本，两路文字不混成一句。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from app.audio.audio_devices import (
    SOURCE_AUTO,
    SOURCE_MICROPHONE,
    SOURCE_SYSTEM_MONITOR,
)
from app.audio.funasr_engine import (
    ASR_FINAL_LATENCY_METRIC,
    DEFAULT_SAMPLE_RATE,
    STATE_ERROR,
    STATE_LOADING,
    STATE_READY,
)
from app.config import current_config
from app.core.exceptions import AudioDeviceError
from app.core.question_session import normalize_text
from app.utils.logger import log_event
from app.utils.timing import default_recorder

__all__ = [
    "RESULT_STARTED",
    "RESULT_STOPPED",
    "RESULT_EMPTY",
    "RESULT_ERROR",
    "AUDIO_STATE_ON",
    "AUDIO_STATE_OFF",
    "DEFAULT_INIT_TIMEOUT",
    "FINAL_LATENCY_SOURCE",
    "AsrEngine",
    "FinalRefiner",
    "AudioManager",
    "Capture",
    "CaptureFactory",
    "AsrEngineFactory",
    "TextCallback",
]


# 动作结果标识，见《5-语音输入与流式ASR》「F8 交互」一节
RESULT_STARTED: str = "started"
RESULT_STOPPED: str = "stopped"
RESULT_EMPTY: str = "empty"
RESULT_ERROR: str = "error"

# 采集状态，取值与状态栏的 AUDIO 一致，见《2-骨架与配置》状态表
AUDIO_STATE_ON: str = "on"
AUDIO_STATE_OFF: str = "off"

# 音频设备初始化超时的默认秒数：进程里没加载过配置时用它，
# 见《8-集成与验收》「超时管理」一节
DEFAULT_INIT_TIMEOUT: float = 10.0

# 最终文本耗时的来源标签：这个指标由本模块记录，
# 口径是停止入口到最终文本产生，见《2-骨架与配置》性能指标一节
FINAL_LATENCY_SOURCE: str = "audio_manager"


@runtime_checkable
class AsrEngine(Protocol):
    """ASR 引擎的调用约定。

    默认实现是 :class:`~app.audio.funasr_engine.FunASREngine`，
    单元测试注入假引擎，不做真实推理。
    """

    @property
    def state(self) -> str:
        """引擎状态，取 ``loading``、``ready`` 或 ``error``。"""
        ...

    async def ensure_loaded(self) -> None:
        """加载模型。已经加载过就直接返回。"""
        ...

    async def feed(self, block: Any) -> str:
        """喂一块音频，返回这次解码出的增量文本。"""
        ...

    async def flush(self) -> str:
        """冲刷，返回剩下的文本。"""
        ...


@runtime_checkable
class FinalRefiner(Protocol):
    """最终文本重识别器的调用约定。

    默认实现是 :class:`~app.audio.sensevoice_engine.SenseVoiceEngine`：
    整段录音重识别加拼音级热词纠正，见《5-语音输入与流式ASR》
    「重识别与热词纠正」一节。单元测试注入假实现。
    """

    async def recognize(self, audio: Any, sample_rate: int) -> str:
        """整段重识别一段音频，返回热词纠正后的最终文本。"""
        ...

    async def correct_text(self, text: str) -> str:
        """只做热词纠正，不碰音频。重识别降级到流式文本时用。"""
        ...


@runtime_checkable
class Capture(Protocol):
    """采集器的调用约定。

    单路采集与混合采集用 ``get_block`` 取块；
    分离采集用 ``get_microphone_block`` 与 ``get_system_monitor_block``，
    见 :class:`~app.audio.audio_source.SeparatedCapture`。
    """

    @property
    def is_running(self) -> bool:
        """采集是不是开着。"""
        ...

    async def start(self) -> None:
        """开始采集。"""
        ...

    async def stop(self) -> None:
        """停止采集。"""
        ...


# 采集器工厂：参数是来源名，返回采集器
CaptureFactory = Callable[[str], Capture]

# ASR 引擎工厂：参数是来源标签，返回这一路用的引擎
AsrEngineFactory = Callable[[str], AsrEngine]

# 文本回调：参数是文本与来源标签，同步函数与协程函数都收
TextCallback = Callable[[str, str], Awaitable[None] | None]


@dataclass
class _Stream:
    """一条语音流的识别状态。

    :param source: 来源标签。
    :param engine: 这一条流用的 ASR 引擎。
    :param parts: 已经解码出的增量文本，按顺序存；
        拼起来是当前这句的累计文本，见
        :meth:`AudioManager.partial_text`。
    :param blocks: 这一路喂给模型的块数。
    :param loaded: 引擎是不是已经加载过。
    """

    source: str
    engine: AsrEngine
    parts: list[str] = field(default_factory=list)
    blocks: int = 0
    loaded: bool = False


class AudioManager:
    """语音采集的启停与状态。

    程序启动后不监听麦克风，第一次按 F8 才 :meth:`start`；
    再次按 F8 走 :meth:`stop`，把最终文本送出去自动提交，
    见《5-语音输入与流式ASR》「F8 交互」一节。

    :param capture_factory: 采集器工厂，参数是来源名，返回采集器。
    :param asr_engine: ASR 引擎。
    :param on_final_text: 最终文本回调，参数是最终文本与来源标签。
    :param source: 音频来源配置，取值见
        :data:`~app.audio.audio_devices.AUDIO_SOURCES`。
    :param on_partial_text: 部分结果回调，参数是这段语音的累计文本与来源标签。
        临时显示用，不提交，见《5-语音输入与流式ASR》「部分结果与最终结果」一节。
    :param asr_engine_factory: 多路来源时每路各建一个引擎的工厂。
        没有它时多路来源共用注入的引擎，识别状态会串流，
        见《5-语音输入与流式ASR》「流式推理」一节。
    :param partial_text_cumulative: 部分结果回调传累计文本还是增量文本。
        默认传增量：界面层把收到的文字自己累加成临时显示那一行
        （见 :meth:`app.ui.app.InterviewMateApp.show_voice_text`），
        两边都累加会把同一段文字接成两遍。
        界面层改成替换显示之后，这里打开开关，
        管理器就把当前这段语音的累计文本传过去。
    :param final_refiner: 最终文本重识别器，None 时不重识别，
        最终文本就是流式结果。给了的话停止采集时优先用整段录音
        重识别加热词纠正产出最终文本，重识别失败退回流式文本
        （同样过热词纠正），见《5-语音输入与流式ASR》
        「重识别与热词纠正」一节。
    :param init_timeout: 音频设备初始化超时秒数，None 时取进程里加载好的
        配置里的 ``audio.init_timeout``，见《8-集成与验收》「超时管理」一节。
    """

    def __init__(
        self,
        capture_factory: CaptureFactory,
        asr_engine: AsrEngine,
        on_final_text: TextCallback,
        *,
        source: str = SOURCE_AUTO,
        on_partial_text: TextCallback | None = None,
        asr_engine_factory: AsrEngineFactory | None = None,
        partial_text_cumulative: bool = False,
        final_refiner: FinalRefiner | None = None,
        init_timeout: float | None = None,
    ) -> None:
        self._capture_factory = capture_factory
        self._engine = asr_engine
        self._on_final_text = on_final_text
        self._source = source
        self._on_partial_text = on_partial_text
        self._engine_factory = asr_engine_factory
        self._partial_text_cumulative = bool(partial_text_cumulative)
        self._final_refiner = final_refiner
        self._init_timeout = init_timeout

        self._capture: Capture | None = None
        self._streams: dict[str, _Stream] = {}
        self._primary_source: str | None = None
        self._running = False
        self._asr_state: str = getattr(asr_engine, "state", STATE_LOADING) or STATE_LOADING
        # 启停互斥：两个停止流程不交叉
        self._lock = asyncio.Lock()
        # 喂模型的互斥锁：共用一个引擎时两路来源不同时喂
        self._feed_lock = asyncio.Lock()
        self._pump_tasks: list[asyncio.Task[None]] = []
        self._shared_engine_warned = False
        # 最近一次已经提交出去的最终文本，规范化之后存。
        # 下一句规范化后与它一样就不重复提交，
        # 见《8-集成与验收》「防止重复调用」一节。
        # 这个记忆按录音段算：开始新的一段录音就清掉，
        # 与识别状态一起从头开始
        self._last_final_normalized: str = ""

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def source(self) -> str:
        """音频来源配置。"""
        return self._source

    @property
    def is_running(self) -> bool:
        """采集是不是开着。"""
        return self._running

    @property
    def asr_state(self) -> str:
        """ASR 状态，取 ``loading``、``ready`` 或 ``error``。"""
        return self._asr_state

    @property
    def audio_state(self) -> str:
        """采集状态，取 ``on`` 或 ``off``。"""
        return AUDIO_STATE_ON if self._running else AUDIO_STATE_OFF

    @property
    def init_timeout(self) -> float:
        """音频设备初始化超时秒数。

        构造时没给就从进程里加载好的配置里取 ``audio.init_timeout``，
        配置还没加载时用 :data:`DEFAULT_INIT_TIMEOUT`，
        见《8-集成与验收》「超时管理」一节。
        """
        if self._init_timeout is not None:
            return float(self._init_timeout)
        config = current_config()
        if config is not None:
            return float(config.audio.init_timeout)
        return DEFAULT_INIT_TIMEOUT

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    async def toggle(self) -> str:
        """按一次 F8 的入口。

        没在采集时开始采集：加载模型 → 开流 → 进入采集状态；
        已在采集时停止采集：关流 → 冲刷最终文本 → 送出去自动提交。

        :returns: 这次动作的结果标识：

            - ``started``：开始采集成功；
            - ``stopped``：停止采集并拿到最终文本；
            - ``empty``：停止采集但没识别到内容，不发模型请求；
            - ``error``：模型加载失败或开流失败。
        """
        # 计时起点在入口、拿锁之前，锁空等的时间也算停止耗时
        started_at = time.perf_counter()
        async with self._lock:
            if self._running:
                return await self._stop_locked(started_at)
            await self._start_locked()
            return RESULT_STARTED if self._running else RESULT_ERROR

    async def start(self) -> None:
        """加载 ASR 模型并开始采集。

        模型加载失败不抛异常：ASR 状态标成 ``error``，记一条日志，
        程序照常往下走，见《5-语音输入与流式ASR》「FunASR 模型」一节。

        已经在采集时直接返回。
        """
        async with self._lock:
            await self._start_locked()

    async def stop(self) -> str:
        """停止采集，冲刷最终文本并送出去。

        没在采集时直接返回；重复调用是空操作，
        不会重复冲刷，也不会重复提交。

        ``asr_final_latency`` 在这里记：起点是入口、拿锁之前，
        终点是第一段非空最终文本规范化之后，flush 解码包含在内，
        见《2-骨架与配置》性能指标一节。

        :returns: 这次动作的结果标识，与 :meth:`toggle` 一致：
            ``stopped`` 拿到最终文本，``empty`` 没识别到内容；
            没在采集时是空操作，同样返回 ``stopped``。
        """
        # 计时起点在入口、拿锁之前，锁空等的时间也算停止耗时
        started_at = time.perf_counter()
        async with self._lock:
            return await self._stop_locked(started_at)

    async def _start_locked(self) -> None:
        """开始采集。调用方已经拿着锁。

        模型加载失败或开流失败时记一条日志就返回，
        不抛异常，程序照常往下走。
        """
        if self._running:
            return
        self._asr_state = STATE_LOADING
        try:
            await self._engine.ensure_loaded()
        except Exception as exc:
            # 模型加载失败不挡住程序启动，状态标成 error
            self._asr_state = STATE_ERROR
            log_event(
                "audio.asr.load_failed",
                f"ASR 模型加载失败：{exc}",
                level="ERROR",
                error=str(exc),
            )
            return
        self._asr_state = STATE_READY

        try:
            capture = await self._create_capture()
            await capture.start()
        except Exception as exc:
            log_event(
                "audio.capture.start_failed",
                f"开始采集失败：{exc}",
                level="ERROR",
                error=str(exc),
                code=getattr(exc, "code", ""),
                source=self._source,
            )
            return

        self._capture = capture
        # 新的一段录音从头开始，上一段的识别状态不留下来
        self._streams = {}
        # 去重记忆按录音段算：新的一段录音从头开始，
        # 上一段提交过的句子不再拿来比对
        self._last_final_normalized = ""
        self._primary_source = None
        self._running = True
        self._pump_tasks = [
            asyncio.create_task(self._pump(getter, source_hint))
            for getter, source_hint in self._capture_getters(capture)
        ]
        log_event(
            "audio.manager.started",
            "开始语音采集",
            source=self._source,
            streams=len(self._pump_tasks),
        )

    async def _create_capture(self) -> Capture:
        """造出采集器。

        设备枚举是阻塞调用，放到线程执行器里跑，事件循环不被占住，
        见《8-集成与验收》「异步任务」一节；枚举限时 ``audio.init_timeout`` 秒，
        超时报 ``E_AUDIO_DEVICE``，见《8-集成与验收》「超时管理」一节。

        :returns: 采集器。
        :raises AudioDeviceError: 枚举超时，或枚举失败。
        """
        timeout = self.init_timeout
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._capture_factory, self._source), timeout
            )
        except asyncio.TimeoutError as exc:
            raise AudioDeviceError(f"音频设备初始化超时（{timeout:g} 秒）") from exc

    async def _stop_locked(self, started_at: float | None = None) -> str:
        """停止采集、冲刷最终文本并送出去。调用方已经拿着锁。

        :param started_at: 停止入口的时刻，用 :func:`time.perf_counter` 取，
            用来记 ``asr_final_latency``；None 时从现在开始算。
        :returns: ``stopped`` 拿到最终文本，``empty`` 没识别到内容。
        """
        if started_at is None:
            started_at = time.perf_counter()
        if not self._running:
            # 没在采集时是空操作，不会重复冲刷与重复提交
            return RESULT_STOPPED
        self._running = False
        capture = self._capture
        self._capture = None

        await self._stop_pumps()
        if capture is not None:
            try:
                await capture.stop()
            except Exception as exc:
                log_event(
                    "audio.capture.stop_failed",
                    f"停止采集失败：{exc}",
                    level="WARNING",
                    error=str(exc),
                )
        # 队列里没喂完的块先补喂，最后一句才不会被截断
        await self._drain(capture)

        finals = await self._flush_streams()
        recognized = False
        submitted = False
        # 一次停止只记一条 asr_final_latency：多路都出了文本时按第一段算
        latency_recorded = False
        for source, text, recording in finals:
            text = await self._refine_final(source, text, recording)
            text = text.strip()
            if not text:
                continue
            normalized = normalize_text(text)
            if not normalized:
                # 规范化后是空串：只有空白与标点，不是题目，不提交
                continue
            recognized = True
            if not latency_recorded:
                latency_recorded = True
                # 终点在最终文本规范化之后、送出之前；没产生文本就不记，
                # 异常与取消走不到这里，不会留下样本
                default_recorder().record(
                    ASR_FINAL_LATENCY_METRIC,
                    (time.perf_counter() - started_at) * 1000.0,
                    source=FINAL_LATENCY_SOURCE,
                )
            if normalized == self._last_final_normalized:
                # 同一句话不提交两次，见《8-集成与验收》「防止重复调用」一节
                log_event(
                    "audio.asr.duplicate",
                    "最终结果与上一次相同，不重复提交",
                    source=source,
                )
                continue
            self._last_final_normalized = normalized
            submitted = True
            await self._invoke(self._on_final_text, text, source)
        if not recognized:
            # 没识别到内容就不发模型请求，界面按 empty 标识提示失败
            log_event("audio.asr.empty", "没有识别到内容", level="WARNING")
            return RESULT_EMPTY
        if not submitted:
            log_event(
                "audio.asr.duplicate_only",
                "识别到的内容与上一次相同，没有重复提交",
                level="INFO",
            )
        log_event("audio.manager.stopped", "停止语音采集", source=self._source)
        return RESULT_STOPPED

    # ------------------------------------------------------------------
    # 取块与识别
    # ------------------------------------------------------------------
    @staticmethod
    def _capture_getters(
        capture: Capture,
    ) -> list[tuple[Callable[[], Awaitable[Any]], str]]:
        """取采集器的取块方法与对应的来源标签。

        分离采集有两路，各取各的；单路与混合采集只有一路。
        """
        microphone = getattr(capture, "get_microphone_block", None)
        loopback = getattr(capture, "get_system_monitor_block", None)
        if microphone is not None and loopback is not None:
            return [
                (microphone, SOURCE_MICROPHONE),
                (loopback, SOURCE_SYSTEM_MONITOR),
            ]
        return [(capture.get_block, str(getattr(capture, "source", "")))]

    async def _pump(
        self,
        getter: Callable[[], Awaitable[Any]],
        source_hint: str,
    ) -> None:
        """取块、喂给 ASR，把增量文本送出去。

        这个任务只做取块与喂模型，不发网络请求。
        """
        while True:
            try:
                block = await getter()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    "audio.capture.read_failed",
                    f"读取音频块失败：{exc}",
                    level="WARNING",
                    error=str(exc),
                )
                return
            await self._handle_block(block, source_hint)

    async def _handle_block(self, block: Any, source_hint: str) -> None:
        """喂一块音频，把解码出的文本送出去。

        部分结果回调收到增量还是累计文本，
        由 :attr:`AudioManager.partial_text_cumulative` 决定：
        界面层自己累加就传增量，界面层替换显示就传累计，
        见那个参数的说明。
        当前这段语音的累计文本始终由管理器维护，
        :meth:`partial_text` 随时能取。

        :param block: 音频块，来源标签在块的 ``source`` 上。
        :param source_hint: 块上没有标签时用的来源。
        """
        source = str(getattr(block, "source", None) or source_hint)
        stream = self._stream_for(source)
        if stream is None:
            return
        try:
            # 共用一个引擎时两路来源不同时喂，引擎内部状态才不会乱
            async with self._feed_lock:
                if not stream.loaded:
                    await stream.engine.ensure_loaded()
                    stream.loaded = True
                text = await stream.engine.feed(block)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 解码失败：对外报 error，状态栏能看出 ASR 出问题了
            self._asr_state = STATE_ERROR
            log_event(
                "audio.asr.feed_failed",
                f"音频识别失败：{exc}",
                level="ERROR",
                error=str(exc),
                source=source,
            )
            return
        # 引擎的状态同步到管理器，解码恢复之后状态也跟着恢复
        self._sync_asr_state(stream.engine)
        if not text:
            return
        stream.parts.append(text)
        stream.blocks += 1
        if self._on_partial_text is not None:
            # 部分结果只作临时显示，不提交。
            # 传增量还是传累计由 partial_text_cumulative 决定：
            # 界面层自己累加时传增量，界面层改成替换显示时传累计，
            # 见这个参数的说明。
            payload = "".join(stream.parts) if self._partial_text_cumulative else text
            await self._invoke(self._on_partial_text, payload, source)

    def partial_text(self, source: str) -> str:
        """这一路来源当前语音流已经识别出来的累计文本。

        累计值在一条语音流内持续维护，开始新的一条语音流时清零，
        见《5-语音输入与流式ASR》「流式推理」一节的当前句子缓冲。

        :param source: 来源标签。
        :returns: 累计文本，这路还没有语音时是空串。
        """
        stream = self._streams.get(source)
        if stream is None:
            return ""
        return "".join(stream.parts)

    def _sync_asr_state(self, engine: AsrEngine) -> None:
        """把引擎的状态同步到管理器。

        引擎自己知道自己是 loading、ready 还是 error，
        管理器对外报的状态跟着它走。

        :param engine: 这一路来源用的引擎。
        """
        state = getattr(engine, "state", None)
        if isinstance(state, str) and state:
            self._asr_state = state

    def _stream_for(self, source: str) -> _Stream | None:
        """取这一路来源的识别流，没有就建一条。

        :param source: 来源标签。
        :returns: 识别流，建不出来时是 None。
        """
        stream = self._streams.get(source)
        if stream is not None:
            return stream
        engine, loaded = self._engine_for(source)
        if engine is None:
            return None
        stream = _Stream(source=source, engine=engine, loaded=loaded)
        self._streams[source] = stream
        return stream

    def _engine_for(self, source: str) -> tuple[AsrEngine | None, bool]:
        """给这一路来源挑 ASR 引擎。

        第一个出现的来源用注入的引擎，已经加载过；
        后来的来源用工厂各建一个，这样每条语音流有自己的 cache，
        状态不会串流，见《5-语音输入与流式ASR》「流式推理」一节。
        没有工厂时只能共用注入的引擎，这时记一条警告。

        :param source: 来源标签。
        :returns: 引擎与它是不是已经加载过。
        """
        if self._primary_source is None:
            self._primary_source = source
            return self._engine, True
        if self._engine_factory is not None:
            try:
                return self._engine_factory(source), False
            except Exception as exc:
                log_event(
                    "audio.asr.engine_create_failed",
                    f"创建 ASR 引擎失败：{exc}",
                    level="ERROR",
                    error=str(exc),
                    source=source,
                )
                return None, False
        if not self._shared_engine_warned:
            self._shared_engine_warned = True
            log_event(
                "audio.asr.shared_engine",
                "多路来源共用同一个 ASR 引擎，识别状态会串流",
                level="WARNING",
                source=source,
            )
        return self._engine, True

    async def _flush_streams(self) -> list[tuple[str, str, Any]]:
        """冲刷各路识别流。

        冲刷之前先把整段录音缓存取走交给重识别：
        ``FunASREngine.take_recording`` 必须在 ``flush`` 之前调，
        冲刷会把录音缓存清掉。录音归喂得最多的一路
        （与冲刷尾巴的归属一致），其余路没有录音可给。

        :returns: 每路来源的最终文本，文本、来源标签与整段录音一对；
            引擎没有录音缓存（假引擎或不支持重识别）时录音是 None。
        """
        by_engine: dict[int, list[_Stream]] = {}
        for stream in self._streams.values():
            by_engine.setdefault(id(stream.engine), []).append(stream)

        results: list[tuple[str, str, Any]] = []
        for streams in by_engine.values():
            recording = self._take_recording(streams[0].engine)
            remainder = ""
            try:
                remainder = await streams[0].engine.flush()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 冲刷失败：对外报 error，状态栏能看出 ASR 出问题了
                self._asr_state = STATE_ERROR
                log_event(
                    "audio.asr.flush_failed",
                    f"冲刷识别结果失败：{exc}",
                    level="ERROR",
                    error=str(exc),
                )
            # 共用一个引擎时，冲刷出来的尾巴归到喂得最多的一路
            owner = max(streams, key=lambda item: item.blocks)
            for stream in streams:
                text = "".join(stream.parts)
                if stream is owner:
                    text += remainder
                results.append(
                    (stream.source, text, recording if stream is owner else None)
                )
        return results

    def _take_recording(self, engine: AsrEngine) -> Any:
        """取走引擎的整段录音缓存。

        引擎没有 ``take_recording`` 方法（假引擎、mock）时返回 None，
        重识别跳过，最终文本退回流式结果加热词纠正。

        :param engine: 这一路来源用的引擎。
        :returns: float32 的一维波形，引擎不支持时是 None。
        """
        take = getattr(engine, "take_recording", None)
        if take is None:
            return None
        try:
            return take()
        except Exception as exc:
            log_event(
                "audio.recording.take_failed",
                f"取整段录音缓存失败，重识别退回流式结果：{exc}",
                level="WARNING",
                error=str(exc),
            )
            return None

    async def _refine_final(self, source: str, text: str, recording: Any) -> str:
        """产出一路来源的最终文本：重识别优先，流式结果兜底。

        有整段录音且重识别器可用时，用 SenseVoice 重识别整段音频
        加热词纠正；重识别失败、结果为空或没有录音时，退回流式文本
        （同样过一遍热词纠正），口径与重识别路径一致。

        :param source: 来源标签，日志里标明是哪一路降级。
        :param text: 流式识别的最终文本。
        :param recording: 整段录音缓存，None 时表示没有录音可用。
        :returns: 最终文本。
        """
        refiner = self._final_refiner
        if refiner is None:
            return text
        if recording is not None and getattr(recording, "size", 0) > 0:
            try:
                refined = await refiner.recognize(recording, DEFAULT_SAMPLE_RATE)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    "audio.refine.failed",
                    f"重识别失败，退回流式文本：{exc}",
                    level="WARNING",
                    error=str(exc),
                    source=source,
                )
            else:
                refined = refined.strip()
                if refined:
                    return refined
                log_event(
                    "audio.refine.empty",
                    "重识别结果为空，退回流式文本",
                    level="WARNING",
                    source=source,
                )
        else:
            log_event(
                "audio.refine.no_recording",
                "这一路没有整段录音缓存，最终文本保持流式结果",
                level="DEBUG",
                source=source,
            )
        try:
            return await refiner.correct_text(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_event(
                "audio.refine.correct_failed",
                f"流式文本热词纠正失败，保持原文：{exc}",
                level="WARNING",
                error=str(exc),
                source=source,
            )
            return text

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    async def _stop_pumps(self) -> None:
        """收掉取块的任务。"""
        tasks = self._pump_tasks
        self._pump_tasks = []
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _drain(self, capture: Capture | None) -> None:
        """把关流前队列里剩下的块喂给模型。

        采集线程往队列里塞块，关流时可能还有没喂的，
        补喂一遍，最后一句才不会被截断。

        :param capture: 刚停掉的采集器。
        """
        if capture is None:
            return
        for queue in self._capture_queues(capture):
            while True:
                try:
                    block = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    break
                await self._handle_block(block, str(getattr(block, "source", "")))

    @staticmethod
    def _capture_queues(capture: Capture) -> list[Any]:
        """取采集器底下的块队列。

        单路与混合采集有一个队列；分离采集有麦克风与系统回环两个。
        """
        queue = getattr(capture, "queue", None)
        if queue is not None:
            return [queue]
        queues: list[Any] = []
        for name in ("microphone", "loopback"):
            side = getattr(capture, name, None)
            side_queue = getattr(side, "queue", None)
            if side_queue is not None:
                queues.append(side_queue)
        return queues

    @staticmethod
    async def _invoke(callback: TextCallback, text: str, source: str) -> None:
        """调文本回调，同步函数与协程函数都收。

        回调抛异常不影响采集流程，记一条日志。
        """
        try:
            result = callback(text, source)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            log_event(
                "audio.callback.failed",
                f"语音回调执行失败：{exc}",
                level="WARNING",
                error=str(exc),
            )
