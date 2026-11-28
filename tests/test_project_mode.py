"""项目模式双流转写控制器的测试。

用假引擎（按预定增量出文本）与假后端（设备表可控、开流可失败）覆盖：

- 双流启动/停止生命周期，会话自动开启，重复 start 幂等；
- 两条采集流各自的块喂各自引擎实例，互不串流；
- 句末标点、单句字数上限、单句时长窗口三种按句落库触发；
- 停止时补喂队列尾巴并冲刷引擎，最后一句落库；
- F8 截取窗口：窗口内面试官句标记成问题、考生句不标记、
  窗口关闭后不再标记、未运行时窗口忽略、停止时窗口自动关闭；
- 一路引擎异常不拖垮另一路；
- 回环设备选择：环境变量覆盖、非法环境变量回退、默认扬声器优先、
  回环开流失败降级单路；
- 两路都起不来时记日志返回，不抛异常。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from app.audio.audio_capture import AudioBlock
from app.audio.audio_devices import SOURCE_MICROPHONE, SOURCE_SYSTEM_MONITOR
from app.audio.funasr_engine import STATE_ERROR, STATE_LOADING, STATE_READY
from app.audio.project_mode import LOOPBACK_DEVICE_ENV, ProjectModeController
from app.core.event_bus import EventBus
from app.core.exceptions import AudioDeviceError
from app.dialogue.dialogue_log import (
    QUESTION_MARK_PREFIX,
    ROLE_INTERVIEWER,
    ROLE_SELF,
    DialogueLog,
)
from app.platform.types import AudioDevice

# 块长：与配置里的 chunk_duration_ms 一致，16000 Hz 下是 200 毫秒
BLOCK_SAMPLES = 3200


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
class FakeEngine:
    """假 ASR 引擎：按预定列表依次返回增量文本，可注入加载/喂块错误。"""

    def __init__(
        self,
        *,
        feed_results: list[str] | None = None,
        load_error: Exception | None = None,
        feed_error: Exception | None = None,
        flush_text: str = "",
    ) -> None:
        self.state: str = STATE_LOADING
        self.feed_results = list(feed_results or [])
        self.load_error = load_error
        self.feed_error = feed_error
        self.flush_text = flush_text
        self.load_calls = 0
        self.feed_calls: list[Any] = []
        self.flush_calls = 0

    async def ensure_loaded(self) -> None:
        """加载模型，可注入失败。"""
        self.load_calls += 1
        if self.load_error is not None:
            self.state = STATE_ERROR
            raise self.load_error
        self.state = STATE_READY

    async def feed(self, block: Any) -> str:
        """喂一块音频，按预定列表返回增量文本。"""
        self.feed_calls.append(block)
        if self.feed_error is not None:
            raise self.feed_error
        if self.feed_results:
            return self.feed_results.pop(0)
        return ""

    async def flush(self) -> str:
        """冲刷，返回预定的尾巴文本。"""
        self.flush_calls += 1
        return self.flush_text


class FakeRefiner:
    """假重识别器：预置重识别文本与纠正文本，可注入错误。"""

    def __init__(
        self,
        *,
        text: str = "",
        corrected: str | None = None,
        error: Exception | None = None,
        correct_error: Exception | None = None,
    ) -> None:
        self.text = text
        self.corrected = corrected
        self.error = error
        self.correct_error = correct_error
        self.recognize_calls: list[tuple[Any, int]] = []
        self.correct_calls: list[str] = []

    async def recognize(self, audio: Any, sample_rate: int) -> str:
        """整段重识别，记下音频与采样率。"""
        self.recognize_calls.append((audio, sample_rate))
        if self.error is not None:
            raise self.error
        return self.text

    async def correct_text(self, text: str) -> str:
        """热词纠正，记下输入文本。"""
        self.correct_calls.append(text)
        if self.correct_error is not None:
            raise self.correct_error
        return self.corrected if self.corrected is not None else text


class FakeStream:
    """假音频流：记下停止动作。"""

    def __init__(self) -> None:
        self.stopped = False

    @property
    def is_active(self) -> bool:
        """流是不是还开着。"""
        return not self.stopped

    def stop(self) -> None:
        """关掉流。"""
        self.stopped = True


class FakeBackend:
    """假采集后端：设备表可控，回环开流与全部开流都可注入失败。"""

    name = "fake"
    required_packages: tuple[str, ...] = ()

    def __init__(
        self,
        devices: list[AudioDevice] | None = None,
        *,
        fail_loopback: bool = False,
        fail_all: bool = False,
    ) -> None:
        self.devices = list(devices or [])
        self.fail_loopback = fail_loopback
        self.fail_all = fail_all
        self.start_calls: list[SimpleNamespace] = []
        self.streams: list[FakeStream] = []

    def list_devices(self) -> list[AudioDevice]:
        """返回注入的设备表。"""
        return list(self.devices)

    def record(self, **_kwargs: Any) -> "np.ndarray":
        """不被项目模式用到。"""
        raise NotImplementedError

    def start_stream(
        self,
        *,
        device: int | None,
        sample_rate: int,
        channels: int,
        block_duration_ms: int,
        on_block: Any,
    ) -> FakeStream:
        """开流：先记调用参数，再按注入的开关决定失败。"""
        self.start_calls.append(
            SimpleNamespace(
                device=device,
                sample_rate=sample_rate,
                channels=channels,
                block_duration_ms=block_duration_ms,
            )
        )
        if self.fail_all:
            raise AudioDeviceError("开流失败：全部失败开关打开")
        if self.fail_loopback and device is not None and device >= 10000:
            raise AudioDeviceError(f"开流失败：回环设备 {device}")
        stream = FakeStream()
        self.streams.append(stream)
        return stream


class _Collector:
    """记下总线投递给某事件类型的全部事件。"""

    def __init__(self, bus: EventBus, event_type: str) -> None:
        self.items: list[Any] = []
        bus.subscribe(event_type, self)

    def __call__(self, event: Any) -> None:
        self.items.append(event)


# --------------------------------------------------------------------------
# 组装工具
# --------------------------------------------------------------------------
def _make_config() -> SimpleNamespace:
    """按项目模式的取参面造一份配置。"""
    audio = SimpleNamespace(
        enabled=True,
        source="auto",
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=200,
        init_timeout=2,
    )
    asr = SimpleNamespace(
        enabled=True,
        model="paraformer-zh-streaming",
        device="cpu",
        cpu_fallback=True,
        vad_enabled=True,
        preload=False,
        init_timeout=2,
    )
    return SimpleNamespace(audio=audio, asr=asr)


def _device(index: int, *, loopback: bool = False, default: bool = False) -> AudioDevice:
    """造一行设备表。"""
    return AudioDevice(
        name=f"dev-{index}",
        index=index,
        is_input=True,
        is_loopback=loopback,
        default_samplerate=48000.0,
        channels=2,
        is_default=default,
    )


def _default_devices() -> list[AudioDevice]:
    """默认设备表：一台麦克风 + 两台回环，编号 10001 是默认扬声器。"""
    return [
        _device(0),
        _device(10000, loopback=True),
        _device(10001, loopback=True, default=True),
    ]


def _make_engines(
    *,
    self_engine: FakeEngine | None = None,
    interviewer_engine: FakeEngine | None = None,
) -> dict[str, FakeEngine]:
    """建两路假引擎，按角色取用。"""
    return {
        ROLE_SELF: self_engine if self_engine is not None else FakeEngine(),
        ROLE_INTERVIEWER: (
            interviewer_engine
            if interviewer_engine is not None
            else FakeEngine()
        ),
    }


def _make_controller(
    tmp_path: Path,
    *,
    devices: list[AudioDevice] | None = None,
    engines: dict[str, FakeEngine] | None = None,
    backend: FakeBackend | None = None,
    bus: EventBus | None = None,
    **kwargs: Any,
) -> tuple[ProjectModeController, DialogueLog, FakeBackend, dict[str, FakeEngine]]:
    """组装一个注入假引擎与假后端的控制器。"""
    dialogue_log = DialogueLog(log_dir=tmp_path / "dialogues")
    if backend is None:
        backend = FakeBackend(devices=_default_devices() if devices is None else devices)
    if engines is None:
        engines = _make_engines()
    controller = ProjectModeController(
        config=_make_config(),
        dialogue_log=dialogue_log,
        engine_factory=engines.__getitem__,
        backend=backend,
        event_bus=bus,
        **kwargs,
    )
    return controller, dialogue_log, backend, engines


def _stream_of(
    controller: ProjectModeController, role: str
) -> Any:
    """取某一路的识别流。"""
    for stream in controller._streams:  # noqa: SLF001 - 测试读内部状态
        if stream.role == role:
            return stream
    raise AssertionError(f"没有 {role} 这一路")


def _block(source: str) -> AudioBlock:
    """造一块 200 毫秒的静音音频。"""
    return AudioBlock(
        samples=np.zeros(BLOCK_SAMPLES, dtype="float32"),
        source=source,
        sample_rate=16000,
        captured_at=0.0,
    )


def _push(controller: ProjectModeController, role: str) -> None:
    """往某一路的块队列里塞一块静音音频。"""
    stream = _stream_of(controller, role)
    stream.capture.queue.push(_block(stream.source))


async def _spin(count: int = 30) -> None:
    """让出控制权若干次，等 pump 任务把队列里的块消化完。"""
    for _ in range(count):
        await asyncio.sleep(0)


# --------------------------------------------------------------------------
# 生命周期
# --------------------------------------------------------------------------
async def test_start_stop_lifecycle(tmp_path: Path) -> None:
    """start 起两路并自动开会话，stop 关两路、日志文件保留。"""
    controller, dialogue_log, backend, engines = _make_controller(tmp_path)
    bus = EventBus()
    controller._event_bus = bus  # noqa: SLF001 - 测试注入总线
    started = _Collector(bus, "project_mode.started")
    stopped = _Collector(bus, "project_mode.stopped")

    await controller.start()

    assert controller.is_running()
    assert dialogue_log.is_active()
    assert controller.session_id == dialogue_log.session_id
    assert len(controller._streams) == 2  # noqa: SLF001
    assert len(backend.start_calls) == 2
    assert engines[ROLE_SELF].state == STATE_READY
    assert engines[ROLE_INTERVIEWER].state == STATE_READY
    assert len(started.items) == 1

    await controller.stop()

    assert not controller.is_running()
    for stream in controller._streams:  # noqa: SLF001
        assert not stream.capture.is_running
    # 两条底层流都关了
    assert len(backend.streams) == 2
    assert all(stream.stopped for stream in backend.streams)
    assert dialogue_log.is_active()  # 日志文件保留，会话不封存
    assert len(stopped.items) == 1


async def test_start_twice_opens_one_session(tmp_path: Path) -> None:
    """重复 start 是空操作：不重开会话、不重复开流。"""
    controller, dialogue_log, _backend, _engines = _make_controller(tmp_path)

    await controller.start()
    first_session = controller.session_id
    await controller.start()

    assert controller.is_running()
    assert controller.session_id == first_session
    files = list(dialogue_log._log_dir.glob("*.md"))  # noqa: SLF001
    assert len(files) == 1
    assert len(controller._streams) == 2  # noqa: SLF001


async def test_no_devices_start_fails_gracefully(tmp_path: Path) -> None:
    """两路都起不来时记日志返回，不抛异常，is_running 是假。"""
    backend = FakeBackend(devices=[], fail_all=True)
    controller, _dialogue_log, _b, _e = _make_controller(tmp_path, backend=backend)

    await controller.start()

    assert not controller.is_running()


# --------------------------------------------------------------------------
# 双流与按句落库
# --------------------------------------------------------------------------
async def test_dual_stream_feeds_own_engine(tmp_path: Path) -> None:
    """两条采集流各自的块喂各自引擎实例，互不串流。"""
    controller, _log, _b, engines = _make_controller(tmp_path)
    await controller.start()

    _push(controller, ROLE_SELF)
    _push(controller, ROLE_INTERVIEWER)
    await _spin()

    self_calls = engines[ROLE_SELF].feed_calls
    interviewer_calls = engines[ROLE_INTERVIEWER].feed_calls
    assert len(self_calls) == 1 and self_calls[0].source == SOURCE_MICROPHONE
    assert self_calls[0].samples.size == BLOCK_SAMPLES
    assert (
        len(interviewer_calls) == 1
        and interviewer_calls[0].source == SOURCE_SYSTEM_MONITOR
    )


async def test_punctuation_sentence_logged_and_events(tmp_path: Path) -> None:
    """句末标点触发落库，partial/sentence/sentence_logged 事件依次发出。"""
    bus = EventBus()
    partial = _Collector(bus, "dialogue.partial")
    sentence = _Collector(bus, "dialogue.sentence")
    logged = _Collector(bus, "dialogue.sentence_logged")
    engines = _make_engines(
        self_engine=FakeEngine(feed_results=["你好", "。"]),
    )
    controller, dialogue_log, _b, _e = _make_controller(tmp_path, engines=engines, bus=bus)

    await controller.start()
    _push(controller, ROLE_SELF)
    _push(controller, ROLE_SELF)
    await _spin()
    await controller.stop()

    assert "[我] 你好。" in dialogue_log.full_text
    # partial 每块发一条，文本是当前句累计
    assert [item.text for item in partial.items] == ["你好", "你好。"]
    assert partial.items[0].role == ROLE_SELF
    assert len(sentence.items) == 1
    assert sentence.items[0].role == ROLE_SELF
    assert sentence.items[0].text == "你好。"
    assert len(logged.items) == 1
    assert logged.items[0].session_id == controller.session_id


async def test_char_limit_sentence_logged(tmp_path: Path) -> None:
    """无标点的长串超过字数上限时整句落库。"""
    engines = _make_engines(self_engine=FakeEngine(feed_results=["abcdefgh"]))
    controller, dialogue_log, _b, _e = _make_controller(
        tmp_path, engines=engines, max_sentence_chars=5
    )

    await controller.start()
    _push(controller, ROLE_SELF)
    await _spin()
    await controller.stop()

    assert "[我] abcdefgh" in dialogue_log.full_text


async def test_time_window_sentence_logged(tmp_path: Path) -> None:
    """无标点的短句挂过时长窗口后整句落库。"""
    engines = _make_engines(self_engine=FakeEngine(feed_results=["嗯", ""]))
    controller, dialogue_log, _b, _e = _make_controller(
        tmp_path, engines=engines, max_sentence_seconds=0.3
    )

    await controller.start()
    _push(controller, ROLE_SELF)
    await _spin()
    assert dialogue_log.full_text == ""  # 200ms 还没到窗口
    _push(controller, ROLE_SELF)
    await _spin()

    assert "[我] 嗯" in dialogue_log.full_text
    await controller.stop()


async def test_stop_flushes_final_sentence(tmp_path: Path) -> None:
    """停止时冲刷引擎，缓冲里没落库的尾巴作为最后一句落库。"""
    engines = _make_engines(self_engine=FakeEngine(feed_results=["没有标点的尾巴"]))
    controller, dialogue_log, _b, _e = _make_controller(tmp_path, engines=engines)

    await controller.start()
    _push(controller, ROLE_SELF)
    await _spin()
    assert dialogue_log.full_text == ""  # 还没触发落库

    await controller.stop()

    assert "[我] 没有标点的尾巴" in dialogue_log.full_text
    assert engines[ROLE_SELF].flush_calls == 1


# --------------------------------------------------------------------------
# F8 截取窗口
# --------------------------------------------------------------------------
async def test_question_window_marks_interviewer_only(tmp_path: Path) -> None:
    """窗口内的面试官句标记成问题，考生句不标记，关窗后不再标记。"""
    bus = EventBus()
    marked = _Collector(bus, "dialogue.question_marked")
    engines = _make_engines(
        self_engine=FakeEngine(feed_results=["嗯。"]),
        interviewer_engine=FakeEngine(feed_results=["什么是闭包？", "后续话题。"]),
    )
    controller, dialogue_log, _b, _e = _make_controller(tmp_path, engines=engines, bus=bus)

    await controller.start()
    await controller.open_question_window()
    _push(controller, ROLE_INTERVIEWER)
    await _spin()
    _push(controller, ROLE_SELF)
    await _spin()

    question_text = await controller.close_question_window()

    assert question_text == "什么是闭包？"
    assert "[面试官] 什么是闭包？" in dialogue_log.full_text
    assert "[我] 嗯。" in dialogue_log.full_text
    assert dialogue_log.full_text.count(QUESTION_MARK_PREFIX) == 1
    assert len(marked.items) == 1 and marked.items[0].text == "什么是闭包？"

    # 关窗之后的面试官句照常落库，但不再标记
    _push(controller, ROLE_INTERVIEWER)
    await _spin()
    assert "[面试官] 后续话题。" in dialogue_log.full_text
    assert dialogue_log.full_text.count(QUESTION_MARK_PREFIX) == 1


async def test_window_ignored_when_not_running(tmp_path: Path) -> None:
    """没在运行时窗口操作被忽略，close 返回空串。"""
    controller, _log, _b, _e = _make_controller(tmp_path)

    await controller.open_question_window()
    assert await controller.close_question_window() == ""


async def test_stop_closes_open_window(tmp_path: Path) -> None:
    """停止时截取窗口还开着，窗口里收集的面试官句补标记。"""
    engines = _make_engines(
        interviewer_engine=FakeEngine(feed_results=["什么是闭包？"])
    )
    controller, dialogue_log, _b, _e = _make_controller(tmp_path, engines=engines)

    await controller.start()
    await controller.open_question_window()
    _push(controller, ROLE_INTERVIEWER)
    await _spin()

    await controller.stop()

    assert dialogue_log.full_text.count(QUESTION_MARK_PREFIX) == 1


# --------------------------------------------------------------------------
# 故障隔离
# --------------------------------------------------------------------------
async def test_mic_engine_error_does_not_kill_loopback(tmp_path: Path) -> None:
    """一路引擎持续解码失败，另一路照常识别落库，控制器不倒。"""
    engines = _make_engines(
        self_engine=FakeEngine(feed_error=RuntimeError("解码炸了")),
        interviewer_engine=FakeEngine(feed_results=["面试官的问题？"]),
    )
    controller, dialogue_log, _b, _e = _make_controller(tmp_path, engines=engines)

    await controller.start()
    _push(controller, ROLE_SELF)
    _push(controller, ROLE_INTERVIEWER)
    await _spin()

    assert controller.is_running()
    assert "[面试官] 面试官的问题？" in dialogue_log.full_text
    assert "[我]" not in dialogue_log.full_text

    await controller.stop()


# --------------------------------------------------------------------------
# 回环设备选择
# --------------------------------------------------------------------------
async def test_loopback_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量指定的回环端点编号覆盖默认选择。"""
    monkeypatch.setenv(LOOPBACK_DEVICE_ENV, "10005")
    backend = FakeBackend(devices=_default_devices())
    controller, _log, _b, _e = _make_controller(tmp_path, backend=backend)

    await controller.start()
    await controller.stop()

    devices_used = [call.device for call in backend.start_calls]
    assert devices_used.count(10005) == 1


async def test_loopback_env_invalid_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """环境变量不是整数时记警告并回退到默认回环端点。"""
    monkeypatch.setenv(LOOPBACK_DEVICE_ENV, "not-a-number")
    backend = FakeBackend(devices=_default_devices())
    controller, _log, _b, _e = _make_controller(tmp_path, backend=backend)

    await controller.start()
    await controller.stop()

    devices_used = [call.device for call in backend.start_calls]
    assert devices_used.count(10001) == 1


async def test_loopback_default_selection(tmp_path: Path) -> None:
    """没给环境变量时回环一路用默认扬声器的回环端点。"""
    backend = FakeBackend(devices=_default_devices())
    controller, _log, _b, _e = _make_controller(tmp_path, backend=backend)

    await controller.start()
    await controller.stop()

    devices_used = [call.device for call in backend.start_calls]
    # 10001 是 is_default 的回环端点；10000 不是默认的不该被选
    assert devices_used.count(10001) == 1
    assert devices_used.count(10000) == 0


async def test_loopback_open_failure_degrades_to_mic(tmp_path: Path) -> None:
    """回环开流失败时降级成单路：麦克风一路照常运行。"""
    backend = FakeBackend(devices=_default_devices(), fail_loopback=True)
    controller, dialogue_log, _b, _e = _make_controller(tmp_path, backend=backend)
    engines = _e
    engines[ROLE_SELF] = FakeEngine(feed_results=["我自己说的话。"])

    await controller.start()

    assert controller.is_running()
    assert len(controller._streams) == 1  # noqa: SLF001
    assert controller._streams[0].role == ROLE_SELF  # noqa: SLF001

    _push(controller, ROLE_SELF)
    await _spin()
    assert "[我] 我自己说的话。" in dialogue_log.full_text
    await controller.stop()


async def test_loopback_missing_degrades_to_mic(tmp_path: Path) -> None:
    """设备表里没有回环端点时降级成单路。"""
    backend = FakeBackend(devices=[_device(0)])
    controller, _log, _b, _e = _make_controller(tmp_path, backend=backend)

    await controller.start()

    assert controller.is_running()
    assert len(controller._streams) == 1  # noqa: SLF001
    await controller.stop()

# --------------------------------------------------------------------------
# 句级重识别（SenseVoice + 热词纠正）
# --------------------------------------------------------------------------
async def test_refiner_replaces_sentence_text(tmp_path: Path) -> None:
    """有句音频时落库文本用重识别结果，dialogue_log 与事件都用它。"""
    bus = EventBus()
    sentence = _Collector(bus, "dialogue.sentence")
    engines = _make_engines(
        self_engine=FakeEngine(feed_results=["你好", "。"]),
    )
    refiner = FakeRefiner(text="重识别的句子。")
    controller, dialogue_log, _b, _e = _make_controller(
        tmp_path, engines=engines, bus=bus, refiner=refiner
    )

    await controller.start()
    _push(controller, ROLE_SELF)
    _push(controller, ROLE_SELF)
    await _spin()
    await controller.stop()

    assert "[我] 重识别的句子。" in dialogue_log.full_text
    assert sentence.items[0].text == "重识别的句子。"
    # 重识别收到的是这句的整段音频（两块 3200 点）与 16k 采样率
    audio, sample_rate = refiner.recognize_calls[0]
    assert audio.size == 2 * BLOCK_SAMPLES
    assert sample_rate == 16000


async def test_refiner_error_falls_back_to_corrected_streaming(
    tmp_path: Path,
) -> None:
    """重识别失败退回流式文本，流式文本同样过热词纠正后落库。"""
    engines = _make_engines(
        self_engine=FakeEngine(feed_results=["你好", "。"]),
    )
    refiner = FakeRefiner(error=RuntimeError("重识别崩了"), corrected="纠正的句子。")
    controller, dialogue_log, _b, _e = _make_controller(
        tmp_path, engines=engines, refiner=refiner
    )

    await controller.start()
    _push(controller, ROLE_SELF)
    _push(controller, ROLE_SELF)
    await _spin()
    await controller.stop()

    assert "[我] 纠正的句子。" in dialogue_log.full_text
    assert refiner.recognize_calls and refiner.correct_calls == ["你好。"]


async def test_no_refiner_keeps_streaming_text(tmp_path: Path) -> None:
    """没配重识别器时落库文本就是流式结果，行为与从前一致。"""
    engines = _make_engines(
        self_engine=FakeEngine(feed_results=["你好", "。"]),
    )
    controller, dialogue_log, _b, _e = _make_controller(tmp_path, engines=engines)

    await controller.start()
    _push(controller, ROLE_SELF)
    _push(controller, ROLE_SELF)
    await _spin()
    await controller.stop()

    assert "[我] 你好。" in dialogue_log.full_text


async def test_refiner_empty_result_falls_back(tmp_path: Path) -> None:
    """重识别结果为空串时退回流式文本，句子不丢。"""
    engines = _make_engines(
        self_engine=FakeEngine(feed_results=["你好", "。"]),
    )
    refiner = FakeRefiner(text="", corrected="纠正的句子。")
    controller, dialogue_log, _b, _e = _make_controller(
        tmp_path, engines=engines, refiner=refiner
    )

    await controller.start()
    _push(controller, ROLE_SELF)
    _push(controller, ROLE_SELF)
    await _spin()
    await controller.stop()

    assert "[我] 纠正的句子。" in dialogue_log.full_text