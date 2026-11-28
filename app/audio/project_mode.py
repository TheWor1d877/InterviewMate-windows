"""项目模式的双流转写控制器：麦克风记考生、系统回环记面试官。

与普通语音输入（:mod:`app.audio.audio_manager`）互相独立：项目模式
自包含管理两条采集流，不复用 AudioManager 的 toggle 逻辑，避免两套
状态互相牵动。

设计要点：

- 双流各自一个独立的 :class:`~app.audio.funasr_engine.FunASREngine`
  实例。流式模型内部有 cache 状态，两路混喂同一实例会把识别状态串流；
- 采集直接用 :class:`~app.audio.audio_capture.AudioCapture`：一路默认
  麦克风（沿用设备选择逻辑），一路系统回环。回环设备优先取默认扬声器
  的回环端点，环境变量 ``INTERVIEWMATE_LOOPBACK_DEVICE`` 给出端点编号
  时覆盖默认选择（用户耳机场景需要）；
- 按句落库：句末标点、单句字数上限、单句时长上限任一触发，就把当前句
  写进对话日志；部分结果经事件总线发 ``dialogue.partial``，完整句子发
  ``dialogue.sentence`` 与 ``dialogue.sentence_logged``；
- F8 截取窗口：按下开始窗口，窗口期间落库的面试官句子在松开时逐句
  标记成问题，返回拼接的问题文本给集成层触发回答；
- 角色只按物理来源区分，不做说话人分离、不做回声对冲：说话串了由
  人工重开一轮（对话日志 ``clear``）处理；
- 一路故障不拖垮另一路：引擎加载失败、开流失败、块解码失败都记日志后
  该路跳过或继续，另一路照常跑。两路都起不来时 :meth:`start` 记日志
  返回，``is_running`` 保持假，不抛异常。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from app.audio.audio_capture import AudioCapture
from app.audio.audio_devices import (
    SOURCE_MICROPHONE,
    SOURCE_SYSTEM_MONITOR,
    select_source,
)
from app.audio.funasr_engine import FunASREngine
from app.core.event_bus import (
    DialoguePartial,
    DialogueQuestionMarked,
    DialogueSentence,
    DialogueSentenceLogged,
    DialogueSessionStarted,
    Event,
    ProjectModeStarted,
    ProjectModeStopped,
    now_ts,
)
from app.core.exceptions import AudioDeviceError
from app.dialogue.dialogue_log import ROLE_INTERVIEWER, ROLE_SELF, DialogueLog
from app.platform import get_platform
from app.utils.logger import REPO_ROOT, log_event

if TYPE_CHECKING:
    from app.platform.types import AudioDevice

__all__ = [
    "LOOPBACK_DEVICE_ENV",
    "DEFAULT_SENTENCE_PUNCTUATIONS",
    "DEFAULT_MAX_SENTENCE_CHARS",
    "DEFAULT_MAX_SENTENCE_SECONDS",
    "EngineFactory",
    "ProjectModeController",
]

# 当前句的整段音频缓存点数上限：与单句时长上限同量级再放宽一档，
# 防止无标点的长串把内存吃满；超过就丢最早的音频
MAX_SENTENCE_AUDIO_SECONDS: float = 30.0


# 回环端点编号的环境变量：用户耳机是默认扬声器时，端点编号随切换漂移，
# 用它指定确定的回环端点
LOOPBACK_DEVICE_ENV: str = "INTERVIEWMATE_LOOPBACK_DEVICE"

# 句末标点集合：出现任一标点就切句落库
DEFAULT_SENTENCE_PUNCTUATIONS: str = "。！？；…?!;"

# 单句字数上限：超过就整句落库，防止无标点的长串一直挂着
DEFAULT_MAX_SENTENCE_CHARS: int = 60

# 单句时长上限（秒）：按喂进去的块数近似累计，超过就整句落库
DEFAULT_MAX_SENTENCE_SECONDS: float = 20.0


# 引擎工厂：参数是角色（「我」/「面试官」），返回这一路用的 ASR 引擎
EngineFactory = Callable[[str], Any]


@dataclass
class _RoleStream:
    """一路角色的识别状态。

    :param role: 说话角色。
    :param source: 采集来源标签，写在每块音频上。
    :param capture: 这一路的采集器。
    :param engine: 这一路独占的 ASR 引擎。
    :param buffer: 当前句累计文本，落库后清空。
    :param sentence_blocks: 当前句已经喂进去的块数，近似累计句时长。
    """

    role: str
    source: str
    capture: AudioCapture
    engine: Any
    buffer: str = field(default="")
    sentence_blocks: int = field(default=0)
    # 当前句的整段音频缓存：句文本触发落库时按比例切走对应音频，
    # 交给 SenseVoice 重识别，见《5-语音输入与流式ASR》「重识别与热词纠正」
    sentence_audio: list["np.ndarray"] = field(default_factory=list)
    sentence_audio_samples: int = field(default=0)


class ProjectModeController:
    """双流转写控制器：麦克风一路记「我」，回环一路记「面试官」。

    :param config: 进程配置，取 ``audio`` 与 ``asr`` 两节的采集与模型参数。
    :param dialogue_log: 对话日志，完整句子都写进它。
    :param engine_factory: 引擎工厂，参数是角色，返回该路用的 ASR 引擎。
        不给时按配置建两个独立的 :class:`FunASREngine`。
    :param backend: 采集后端，不给时用平台默认实现（Windows 的
        WASAPI 后端）。
    :param event_bus: 事件总线，不给时不发事件（集成层组装时传入，
        界面靠它收临时文字）。
    :param sentence_punctuations: 句末标点集合，None 时用默认集合。
    :param max_sentence_chars: 单句字数上限。
    :param max_sentence_seconds: 单句时长上限（秒），按块数近似累计。
    :param refiner: 最终文本重识别器，None 时不重识别，落库文本就是
        流式结果。给了的话每句落库前用整段句音频重识别加热词纠正，
        失败退回流式文本（同样过热词纠正），
        见《5-语音输入与流式ASR》「重识别与热词纠正」一节。
    """

    def __init__(
        self,
        *,
        config: Any,
        dialogue_log: DialogueLog,
        engine_factory: EngineFactory | None = None,
        backend: Any | None = None,
        event_bus: Any | None = None,
        sentence_punctuations: str | None = None,
        max_sentence_chars: int = DEFAULT_MAX_SENTENCE_CHARS,
        max_sentence_seconds: float = DEFAULT_MAX_SENTENCE_SECONDS,
        refiner: Any | None = None,
    ) -> None:
        self._config = config
        self._dialogue_log = dialogue_log
        self._engine_factory = engine_factory
        self._backend = backend if backend is not None else get_platform().audio
        self._event_bus = event_bus
        self._refiner = refiner
        self._sentence_punctuations = (
            sentence_punctuations
            if sentence_punctuations is not None
            else DEFAULT_SENTENCE_PUNCTUATIONS
        )
        self._max_sentence_chars = max(1, int(max_sentence_chars))
        self._max_sentence_seconds = max(0.0, float(max_sentence_seconds))
        self._sample_rate = int(config.audio.sample_rate)
        self._channels = int(config.audio.channels)
        self._block_duration_ms = int(config.audio.chunk_duration_ms)

        self._streams: list[_RoleStream] = []
        self._pump_tasks: list[asyncio.Task[None]] = []
        self._running = False
        # F8 截取窗口：开着时落库的面试官句子先收集，松开时标记成问题
        self._window_open = False
        self._window_sentences: list[str] = []

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def session_id(self) -> str | None:
        """当前对话会话 ID，会话没开时是 None。"""
        return self._dialogue_log.session_id

    def is_running(self) -> bool:
        """双流转写是不是开着。"""
        return self._running

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """开始双流转写。

        会话未开就先开一个；然后选设备（麦克风 + 回环）、懒加载两路
        各自的引擎、开两条采集流。一路起不来就跳过该路（记日志），
        另一路照常跑；两路都起不来时记日志返回，不抛异常。

        重复调用是空操作。
        """
        if self._running:
            return
        if not self._dialogue_log.is_active():
            session_id = self._dialogue_log.start_session()
            await self._publish(
                DialogueSessionStarted(ts=now_ts(), session_id=session_id)
            )

        mic_device, loopback_device = await self._select_devices()
        self._streams = []
        for role, source, device in (
            (ROLE_SELF, SOURCE_MICROPHONE, mic_device),
            (ROLE_INTERVIEWER, SOURCE_SYSTEM_MONITOR, loopback_device),
        ):
            # 回环没选到端点编号时跳过这一路：不能拿默认输入设备当回环
            if role == ROLE_INTERVIEWER and device is None:
                continue
            stream = await self._open_role(role, source, device)
            if stream is not None:
                self._streams.append(stream)

        if not self._streams:
            log_event(
                "project_mode.start_failed",
                "项目模式启动失败：两路采集都没起来",
                level="ERROR",
            )
            return

        self._running = True
        self._pump_tasks = [
            asyncio.create_task(self._pump(stream)) for stream in self._streams
        ]
        log_event(
            "project_mode.started",
            "项目模式双流转写已开始",
            session_id=self._dialogue_log.session_id or "",
            streams=len(self._streams),
            mic_device=mic_device,
            loopback_device=loopback_device,
        )
        await self._publish(
            ProjectModeStarted(
                ts=now_ts(), session_id=self._dialogue_log.session_id or ""
            )
        )

    async def stop(self) -> None:
        """停止双流转写。

        截取窗口还开着时先关掉（窗口里收集的面试官句子补标记）；
        收掉取块任务、关流，把队列里没喂完的块补喂，冲刷两路引擎，
        各自的尾巴作为最后一句落库。对话日志文件保留，会话不封存。

        没在运行时是空操作。
        """
        if not self._running:
            return
        self._running = False
        if self._window_open:
            await self.close_question_window()

        tasks = self._pump_tasks
        self._pump_tasks = []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        for stream in self._streams:
            with contextlib.suppress(Exception):
                await stream.capture.stop()
        # 队列里没喂完的块先补喂，最后一句才不会被截断
        for stream in self._streams:
            await self._drain(stream)
            await self._flush_role(stream)
        self._streams = []

        log_event(
            "project_mode.stopped",
            "项目模式双流转写已停止",
            session_id=self._dialogue_log.session_id or "",
        )
        await self._publish(
            ProjectModeStopped(
                ts=now_ts(), session_id=self._dialogue_log.session_id or ""
            )
        )

    # ------------------------------------------------------------------
    # F8 截取窗口
    # ------------------------------------------------------------------
    async def open_question_window(self) -> None:
        """F8 按下：开始截取窗口。

        窗口期间落库的面试官句子先收集起来，松开时逐句标记成问题。
        没在运行时忽略；窗口已开时是空操作。
        """
        if not self._running:
            log_event(
                "project_mode.question_window_ignored",
                "截取窗口被忽略：双流转写没在运行",
                level="WARNING",
                reason="not_running",
            )
            return
        if self._window_open:
            return
        self._window_open = True
        self._window_sentences = []
        log_event(
            "project_mode.question_window_opened",
            "截取窗口开始",
            session_id=self._dialogue_log.session_id or "",
        )

    async def close_question_window(self) -> str:
        """F8 松开：把窗口内的面试官句子标记成问题，返回问题文本。

        标记动作是逐句往对话日志追加 ``**[问题]**`` 行，返回值是拼接好
        的问题文本（换行分隔），集成层拿去触发回答。

        窗口没开时是空操作，返回空串。

        :returns: 窗口内面试官句子按时间序拼接的问题文本。
        """
        if not self._window_open:
            log_event(
                "project_mode.question_window_ignored",
                "关闭截取窗口被忽略：窗口没开着",
                level="WARNING",
                reason="window_not_open",
            )
            return ""
        self._window_open = False
        sentences = self._window_sentences
        self._window_sentences = []
        for text in sentences:
            self._dialogue_log.mark_question(text)
        marked = "\n".join(sentences)
        if marked:
            await self._publish(DialogueQuestionMarked(ts=now_ts(), text=marked))
        log_event(
            "project_mode.question_window_closed",
            "截取窗口结束",
            questions=len(sentences),
            session_id=self._dialogue_log.session_id or "",
        )
        return marked

    # ------------------------------------------------------------------
    # 设备选择
    # ------------------------------------------------------------------
    async def _select_devices(self) -> tuple[int | None, int | None]:
        """挑出麦克风与回环两路要开的设备编号。

        设备枚举是阻塞调用，放到线程里跑，事件循环不被占住。

        :returns: ``(麦克风编号, 回环编号)``，选不出来的一路是 None。
        """
        try:
            devices = await asyncio.to_thread(self._backend.list_devices)
        except Exception as exc:
            log_event(
                "project_mode.device_enumerate_failed",
                f"枚举音频设备失败，麦克风退回默认输入、回环跳过：{exc}",
                level="ERROR",
                error=str(exc),
            )
            return None, None
        mic_index: int | None = None
        try:
            mic_index = int(select_source(SOURCE_MICROPHONE, devices).device.index)
        except AudioDeviceError:
            # 麦克风没选出来时退回后端默认输入设备，不给 None 也开不了流
            log_event(
                "project_mode.microphone_fallback",
                "没选到默认麦克风，退回后端默认输入设备",
                level="WARNING",
            )
        return mic_index, self._loopback_index(devices)

    def _loopback_index(self, devices: "list[AudioDevice]") -> int | None:
        """挑回环端点编号，环境变量指定的编号优先。

        :param devices: 设备表。
        :returns: 回环端点编号，没有可用端点时是 None。
        """
        env_value = os.environ.get(LOOPBACK_DEVICE_ENV, "").strip()
        if env_value:
            index = self._parse_loopback_env(env_value)
            if index is not None:
                log_event(
                    "project_mode.loopback_env_selected",
                    "使用环境变量指定的回环端点",
                    device=index,
                )
                return index
        try:
            return int(select_source(SOURCE_SYSTEM_MONITOR, devices).device.index)
        except AudioDeviceError as exc:
            log_event(
                "project_mode.loopback_missing",
                f"没有可用的系统回环端点，面试官一路跳过：{exc}",
                level="WARNING",
            )
            return None

    def _parse_loopback_env(self, env_value: str) -> int | None:
        """解析环境变量里的回环端点编号。

        :param env_value: 环境变量原文。
        :returns: 非负端点编号，解析不出时是 None（并记一条警告）。
        """
        try:
            index = int(env_value)
        except ValueError:
            log_event(
                "project_mode.loopback_env_invalid",
                f"{LOOPBACK_DEVICE_ENV} 不是整数，回退到默认回环端点",
                level="WARNING",
                value=env_value,
            )
            return None
        if index < 0:
            log_event(
                "project_mode.loopback_env_invalid",
                f"{LOOPBACK_DEVICE_ENV} 不能是负数，回退到默认回环端点",
                level="WARNING",
                value=env_value,
            )
            return None
        return index

    # ------------------------------------------------------------------
    # 一路的组装
    # ------------------------------------------------------------------
    async def _open_role(
        self, role: str, source: str, device: int | None
    ) -> _RoleStream | None:
        """起一路：建引擎、加载模型、开采集流。

        :param role: 这一路的说话角色。
        :param source: 采集来源标签。
        :param device: 设备编号，麦克风是 None 时用后端默认输入设备。
        :returns: 识别流，这一路起不来时是 None（记日志后跳过）。
        """
        engine = self._build_engine(role)
        if engine is None:
            return None
        try:
            await engine.ensure_loaded()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_event(
                "project_mode.engine_load_failed",
                f"「{role}」一路的 ASR 引擎加载失败，该路跳过：{exc}",
                level="ERROR",
                error=str(exc),
                role=role,
            )
            return None

        capture = AudioCapture(
            self._backend,
            device=device,
            source=source,
            sample_rate=self._sample_rate,
            channels=self._channels,
            block_duration_ms=self._block_duration_ms,
        )
        try:
            await capture.start()
        except Exception as exc:
            log_event(
                "project_mode.stream_start_failed",
                f"「{role}」一路的开流失败，该路跳过：{exc}",
                level="ERROR",
                error=str(exc),
                role=role,
                device=device,
            )
            return None
        return _RoleStream(role=role, source=source, capture=capture, engine=engine)

    def _build_engine(self, role: str) -> Any | None:
        """给这一路建一个独立的 ASR 引擎。

        :param role: 这一路的说话角色。
        :returns: 引擎，建不出来时是 None（记日志后跳过）。
        """
        try:
            if self._engine_factory is not None:
                return self._engine_factory(role)
            return self._default_engine()
        except Exception as exc:
            log_event(
                "project_mode.engine_create_failed",
                f"「{role}」一路创建 ASR 引擎失败，该路跳过：{exc}",
                level="ERROR",
                error=str(exc),
                role=role,
            )
            return None

    def _default_engine(self) -> FunASREngine:
        """按配置建一个独立的 FunASR 引擎实例。

        模型目录与主链路 ``select_asr_engine`` 保持一致：显式给仓库内的
        ``models/``，支持 ``INTERVIEWMATE_MODELS`` 覆盖；不给的话引擎会
        回落到 ``~/models``，那里通常没有模型，加载只会报
        「model '...' is not registered」。

        :returns: 引擎实例。流式模型内部有 cache 状态，
            每一路必须有自己的实例。
        """
        asr = self._config.asr
        return FunASREngine(
            model_dir=os.environ.get("INTERVIEWMATE_MODELS") or str(REPO_ROOT / "models"),
            asr_model=asr.model,
            device=asr.device,
            vad_enabled=bool(asr.vad_enabled),
            init_timeout=float(asr.init_timeout),
        )

    # ------------------------------------------------------------------
    # 取块与按句落库
    # ------------------------------------------------------------------
    async def _pump(self, stream: _RoleStream) -> None:
        """取块、喂给这一路的引擎，把完整句子落库。

        这个任务只做取块与喂模型，不发网络请求。解码失败记日志后
        继续下一块，这条流的其余块照常识别。
        """
        while True:
            try:
                block = await stream.capture.get_block()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    "project_mode.read_failed",
                    f"读取音频块失败，这一路停止取块：{exc}",
                    level="WARNING",
                    error=str(exc),
                    role=stream.role,
                )
                return
            await self._handle_block(stream, block)

    async def _handle_block(self, stream: _RoleStream, block: Any) -> None:
        """喂一块音频，把部分结果发出去，判断要不要落库。

        部分结果发的是当前句累计文本，界面收到后替换显示当前那一行；
        句末标点、字数上限或时长上限触发时整句落库。

        :param stream: 这一路的识别流。
        :param block: 一块音频。
        """
        self._cache_block_audio(stream, block)
        try:
            text = await stream.engine.feed(block)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 解码失败记日志继续：这条流剩下的块照常喂，另一路不受影响
            log_event(
                "project_mode.feed_failed",
                f"音频识别失败，本块跳过：{exc}",
                level="ERROR",
                error=str(exc),
                role=stream.role,
            )
            return
        stream.sentence_blocks += 1
        if text:
            stream.buffer += text
        if stream.buffer:
            await self._publish(
                DialoguePartial(
                    ts=now_ts(), role=stream.role, text=stream.buffer
                )
            )
        sentence, rest = self._take_sentence(stream)
        if sentence is not None:
            # 句文本与音频按字符比例切分：标点占字符不占音频，
            # 边界是近似值，重识别不受分块影响，误差可接受
            audio = self._take_sentence_audio(stream, len(sentence), len(stream.buffer))
            await self._commit_sentence(stream, sentence, audio)
            stream.buffer = rest
            stream.sentence_blocks = 0
            return
        # 时长窗口：当前句挂得太久就整句落库（块数 × 块长近似时长）
        elapsed_ms = stream.sentence_blocks * self._block_duration_ms
        if stream.buffer and elapsed_ms >= self._max_sentence_seconds * 1000.0:
            audio = self._take_sentence_audio(stream, len(stream.buffer), len(stream.buffer))
            await self._commit_sentence(stream, stream.buffer, audio)
            stream.buffer = ""
            stream.sentence_blocks = 0

    def _cache_block_audio(self, stream: _RoleStream, block: Any) -> None:
        """把一块音频追加进当前句的整段缓存，超上限丢最早的。

        :param stream: 这一路的识别流。
        :param block: 一块音频，取它的 ``samples``；没有样本的块跳过。
        """
        samples = getattr(block, "samples", None)
        if samples is None:
            return
        wave = np.asarray(samples, dtype="float32").reshape(-1)
        if wave.size == 0:
            return
        stream.sentence_audio.append(wave)
        stream.sentence_audio_samples += int(wave.size)
        limit = int(MAX_SENTENCE_AUDIO_SECONDS * self._sample_rate)
        excess = stream.sentence_audio_samples - limit
        while excess > 0 and stream.sentence_audio:
            head = stream.sentence_audio[0]
            if head.size > excess:
                stream.sentence_audio[0] = head[excess:]
                dropped = excess
            else:
                dropped = int(head.size)
                stream.sentence_audio.pop(0)
            stream.sentence_audio_samples -= dropped
            excess -= dropped

    def _take_sentence_audio(
        self, stream: _RoleStream, sentence_chars: int, total_chars: int
    ) -> "np.ndarray | None":
        """从当前句的音频缓存里按字符比例切走这句对应的音频。

        句文本长度占缓冲文本长度的比例近似等于这句音频占总音频的
        比例：句与句之间有停顿、标点不占音频，边界是近似值，
        落库的句子由整段重识别重新出文本，分块误差不影响识别质量。

        :param stream: 这一路的识别流。
        :param sentence_chars: 这句的字符数。
        :param total_chars: 缓冲文本的总字符数。
        :returns: 这句对应的波形，缓存为空时是 None。
        """
        if stream.sentence_audio_samples <= 0:
            return None
        if total_chars <= 0 or sentence_chars >= total_chars:
            # 整段都归这句
            wave = self._concat_sentence_audio(stream)
            return wave if wave.size > 0 else None
        ratio = min(1.0, max(0.0, sentence_chars / total_chars))
        take = int(round(stream.sentence_audio_samples * ratio))
        take = max(1, min(take, stream.sentence_audio_samples))
        parts: list["np.ndarray"] = []
        remaining = take
        while remaining > 0 and stream.sentence_audio:
            head = stream.sentence_audio[0]
            if head.size > remaining:
                parts.append(head[:remaining])
                stream.sentence_audio[0] = head[remaining:]
                remaining = 0
            else:
                parts.append(head)
                remaining -= int(head.size)
                stream.sentence_audio.pop(0)
        stream.sentence_audio_samples -= take
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)

    def _concat_sentence_audio(self, stream: _RoleStream) -> "np.ndarray":
        """取走当前句音频缓存的整段波形，缓存清空。"""
        if not stream.sentence_audio:
            return np.zeros(0, dtype="float32")
        if len(stream.sentence_audio) == 1:
            wave = stream.sentence_audio[0]
        else:
            wave = np.concatenate(stream.sentence_audio)
        stream.sentence_audio.clear()
        stream.sentence_audio_samples = 0
        return wave

    def _take_sentence(self, stream: _RoleStream) -> tuple[str | None, str]:
        """按句末标点或字数上限从当前句缓冲里切出一个完整句。

        :param stream: 这一路的识别流。
        :returns: ``(完整句或 None, 剩余文本)``。标点在中间时，
            标点及其之前的内容是一句，之后的内容留给下一句。
        """
        buffer = stream.buffer
        if not buffer:
            return None, ""
        cut = -1
        for index, char in enumerate(buffer):
            if char in self._sentence_punctuations:
                cut = index
        if cut >= 0:
            return buffer[: cut + 1], buffer[cut + 1 :]
        if len(buffer) >= self._max_sentence_chars:
            return buffer, ""
        return None, ""

    async def _commit_sentence(
        self, stream: _RoleStream, sentence: str, audio: "np.ndarray | None" = None
    ) -> None:
        """把一个完整句落库并发句子事件。

        有重识别器时，这句文本先用整段句音频重识别加热词纠正，
        失败退回流式文本（同样过热词纠正），
        见《5-语音输入与流式ASR》「重识别与热词纠正」一节。

        截取窗口开着且这一路是面试官时，句子同时收进窗口，
        松开窗口时统一标记成问题。

        :param stream: 这一路的识别流。
        :param sentence: 流式识别出的完整句文本。
        :param audio: 这句对应的整段音频，None 或空时跳过重识别。
        """
        text = sentence.strip()
        if not text:
            return
        text = await self._reidentify_sentence(stream.role, text, audio)
        self._dialogue_log.append(stream.role, text)
        if self._window_open and stream.role == ROLE_INTERVIEWER:
            self._window_sentences.append(text)
        await self._publish(
            DialogueSentence(ts=now_ts(), role=stream.role, text=text)
        )
        await self._publish(
            DialogueSentenceLogged(
                ts=now_ts(),
                role=stream.role,
                text=text,
                session_id=self._dialogue_log.session_id or "",
            )
        )

    async def _reidentify_sentence(
        self, role: str, text: str, audio: "np.ndarray | None"
    ) -> str:
        """重识别一句文本：整段句音频优先，流式文本兜底。

        重识别失败、结果为空或没有句音频时，退回流式文本，
        两条路径都过一遍热词纠正，口径一致。

        :param role: 说话角色，日志里标明是哪一路降级。
        :param text: 流式识别出的句子文本。
        :param audio: 这句对应的整段音频，None 或空时跳过重识别。
        :returns: 落库用的最终文本。
        """
        refiner = self._refiner
        if refiner is None:
            return text
        if audio is not None and audio.size > 0:
            try:
                refined = await refiner.recognize(audio, self._sample_rate)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    "project_mode.reidentify_failed",
                    f"句子重识别失败，退回流式文本：{exc}",
                    level="WARNING",
                    error=str(exc),
                    role=role,
                )
            else:
                refined = refined.strip()
                if refined:
                    return refined
                log_event(
                    "project_mode.reidentify_empty",
                    "句子重识别结果为空，退回流式文本",
                    level="WARNING",
                    role=role,
                )
        try:
            return await refiner.correct_text(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_event(
                "project_mode.correct_failed",
                f"流式句子热词纠正失败，保持原文：{exc}",
                level="WARNING",
                error=str(exc),
                role=role,
            )
            return text

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    async def _drain(self, stream: _RoleStream) -> None:
        """把队列里没喂完的块补喂给引擎。

        关流之后 pump 已收，队列里可能还剩块，补喂一遍，
        最后一句才不会被截断。

        :param stream: 这一路的识别流。
        """
        while True:
            try:
                block = stream.capture.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            except Exception:
                return
            await self._handle_block(stream, block)

    async def _flush_role(self, stream: _RoleStream) -> None:
        """冲刷这一路的引擎，尾巴作为最后一句落库。

        :param stream: 这一路的识别流。
        """
        try:
            remainder = await stream.engine.flush()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_event(
                "project_mode.flush_failed",
                f"冲刷识别结果失败：{exc}",
                level="WARNING",
                error=str(exc),
                role=stream.role,
            )
            remainder = ""
        if remainder:
            stream.buffer += remainder
        if stream.buffer.strip():
            # 尾巴对应的音频是缓存里剩下的整段
            audio = self._concat_sentence_audio(stream)
            await self._commit_sentence(stream, stream.buffer, audio)
            stream.buffer = ""

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    async def _publish(self, event: Event) -> None:
        """往事件总线发一个事件，总线没接时是空操作。"""
        bus = self._event_bus
        if bus is None:
            return
        await bus.publish(event)