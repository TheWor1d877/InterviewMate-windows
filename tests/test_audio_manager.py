"""语音采集管理器与 F8 控制器的测试。

覆盖《5-语音输入与流式ASR》「F8 交互」「部分结果与最终结果」
「状态显示」几节要求的行为：

- 第一次按 F8 加载模型并开始采集，``is_running`` 变真；
- 识别过程中的部分结果只走临时显示回调，不触发最终提交；
- 再次按 F8 停止采集，把最终文本交给最终提交回调；
- 最终文本为空时不调用提交回调，返回空标识；
- ASR 模型加载失败时不抛异常，状态为 error，程序继续；
- 重复调用 stop 不会交叉，第二次是空操作；
- 来源标签不同的块不混在一起；
- 正在停止时又按一次 F8，这次按键被忽略。
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator

import pytest

from app.audio.audio_capture import AudioBlock, AudioCapture
from app.audio.audio_manager import (
    AUDIO_STATE_OFF,
    AUDIO_STATE_ON,
    RESULT_EMPTY,
    RESULT_ERROR,
    RESULT_STARTED,
    RESULT_STOPPED,
    AudioManager,
)
from app.audio.funasr_engine import (
    ASR_FINAL_LATENCY_METRIC,
    STATE_ERROR,
    STATE_LOADING,
    STATE_READY,
)
from app.core.exceptions import AsrLoadError
from app.inputs.hotkey_controller import RESULT_IGNORED, HotkeyController

from app.platform.windows.audio import WasapiAudioBackend
from app.utils.timing import MetricRecorder, default_recorder, reset_default_recorder

np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
class _FakeCapture:
    """假采集器：块从一个队列里出。"""

    def __init__(self, source: str = "microphone") -> None:
        self.source = source
        self.queue: asyncio.Queue[AudioBlock] = asyncio.Queue()
        self.started = False
        self.stopped = False

    @property
    def is_running(self) -> bool:
        """采集是不是开着。"""
        return self.started

    async def start(self) -> None:
        """开始采集。"""
        self.started = True

    async def stop(self) -> None:
        """停止采集。"""
        self.started = False
        self.stopped = True

    async def get_block(self) -> AudioBlock:
        """等下一块。"""
        return await self.queue.get()

    def push(self, source: str = "microphone") -> None:
        """塞一块音频进去。"""
        self.queue.put_nowait(_block(source))


class _FakeEngine:
    """假 ASR 引擎：记下调用，按块返回增量文本，冲刷返回最后一段。"""

    def __init__(
        self,
        *,
        load_error: Exception | None = None,
        final_text: str = "",
        feed_text: Callable[[Any], str] | None = None,
    ) -> None:
        self.state: str = STATE_LOADING
        self.load_error = load_error
        self.final_text = final_text
        self.feed_text = feed_text
        self.load_calls = 0
        self.feed_calls: list[Any] = []
        self.flush_calls = 0
        # 冲刷时等这个闸门，用来制造"正在停止"的窗口
        self.flush_gate: asyncio.Event | None = None
        self.flush_started = asyncio.Event()

    async def ensure_loaded(self) -> None:
        """加载模型。"""
        self.load_calls += 1
        if self.load_error is not None:
            self.state = STATE_ERROR
            raise self.load_error
        self.state = STATE_READY

    async def feed(self, block: Any) -> str:
        """喂一块音频，返回增量文本。"""
        self.feed_calls.append(block)
        if self.feed_text is not None:
            return self.feed_text(block)
        return ""

    async def flush(self) -> str:
        """冲刷，返回最后的文本。"""
        self.flush_calls += 1
        self.flush_started.set()
        if self.flush_gate is not None:
            await self.flush_gate.wait()
        text = self.final_text
        self.final_text = ""
        return text


class _Recorder:
    """记下回调收到的文本与来源标签。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, source: str) -> None:
        """收下一次回调。"""
        self.calls.append((text, source))

    @property
    def texts(self) -> list[str]:
        """只取文本。"""
        return [text for text, _ in self.calls]


@dataclass
class _Harness:
    """一套替身与被测对象。"""

    manager: AudioManager
    capture: _FakeCapture
    engine: _FakeEngine
    partial: _Recorder
    final: _Recorder
    source_calls: list[str] = field(default_factory=list)


def _block(source: str = "microphone") -> AudioBlock:
    """造一块 200 毫秒的音频块。"""
    return AudioBlock(
        samples=np.zeros(3200, dtype="float32"),
        source=source,
        sample_rate=16000,
        captured_at=0.0,
    )


async def _wait_until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    """等条件成立，等不到就报错。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("等待条件超时")


def _make_harness(
    *,
    engine: _FakeEngine | None = None,
    capture: _FakeCapture | None = None,
    asr_engine_factory: Callable[[str], Any] | None = None,
    partial_text_cumulative: bool = False,
) -> _Harness:
    """造一套替身与被测的管理器。"""
    engine = engine if engine is not None else _FakeEngine()
    capture = capture if capture is not None else _FakeCapture()
    partial = _Recorder()
    final = _Recorder()
    source_calls: list[str] = []

    def capture_factory(source: str) -> _FakeCapture:
        """假采集器工厂。"""
        source_calls.append(source)
        return capture

    manager = AudioManager(
        capture_factory,
        engine,
        final,
        source="auto",
        on_partial_text=partial,
        asr_engine_factory=asr_engine_factory,
        partial_text_cumulative=partial_text_cumulative,
    )
    return _Harness(
        manager=manager,
        capture=capture,
        engine=engine,
        partial=partial,
        final=final,
        source_calls=source_calls,
    )


@pytest.fixture()
def isolated_recorder() -> Iterator[MetricRecorder]:
    """用例前后清空进程级记录器，样本不串到别的用例。"""
    reset_default_recorder()
    try:
        yield default_recorder()
    finally:
        reset_default_recorder()


# --------------------------------------------------------------------------
# 启停
# --------------------------------------------------------------------------
async def test_first_toggle_loads_model_and_starts_capture() -> None:
    """第一次按 F8 加载模型并开始采集。"""
    harness = _make_harness()
    assert harness.manager.is_running is False
    assert harness.manager.audio_state == AUDIO_STATE_OFF

    result = await harness.manager.toggle()

    assert result == RESULT_STARTED
    assert harness.manager.is_running is True
    assert harness.manager.audio_state == AUDIO_STATE_ON
    assert harness.manager.asr_state == STATE_READY
    assert harness.capture.started is True
    assert harness.engine.load_calls == 1
    assert harness.source_calls == ["auto"]


async def test_manager_does_not_capture_before_first_toggle() -> None:
    """程序启动后不监听麦克风。"""
    harness = _make_harness()
    assert harness.source_calls == []
    assert harness.capture.started is False
    assert harness.engine.load_calls == 0


async def test_second_toggle_stops_capture_and_submits_final_text() -> None:
    """再次按 F8 停止采集，把最终文本交给最终提交回调。"""
    harness = _make_harness(engine=_FakeEngine(final_text="面试官的问题"))

    assert await harness.manager.toggle() == RESULT_STARTED
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))

    assert await harness.manager.toggle() == RESULT_STOPPED
    assert harness.manager.is_running is False
    assert harness.manager.audio_state == AUDIO_STATE_OFF
    assert harness.capture.stopped is True
    assert harness.final.calls == [("面试官的问题", "microphone")]


async def test_partial_result_only_goes_to_partial_callback() -> None:
    """部分结果只更新临时显示，不触发最终提交。"""
    harness = _make_harness(
        engine=_FakeEngine(feed_text=lambda block: f"部分[{block.source}]")
    )

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.partial.calls))

    assert harness.partial.calls == [("部分[microphone]", "microphone")]
    # 采集过程中不触发最终提交
    assert harness.final.calls == []

    # 停止之后部分结果合成最终文本，只提交一次
    await harness.manager.stop()
    assert harness.final.calls == [("部分[microphone]", "microphone")]


async def test_empty_final_text_skips_submit_callback() -> None:
    """最终文本为空时不调用提交回调，返回空标识。"""
    harness = _make_harness(engine=_FakeEngine(final_text=""))

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))

    result = await harness.manager.toggle()

    assert result == RESULT_EMPTY
    assert harness.final.calls == []


async def test_empty_final_text_skips_submit_callback_without_any_block() -> None:
    """一句都没识别到内容时不调用提交回调。"""
    harness = _make_harness(engine=_FakeEngine(final_text=""))

    await harness.manager.toggle()
    result = await harness.manager.toggle()

    assert result == RESULT_EMPTY
    assert harness.final.calls == []


async def test_model_load_failure_keeps_program_running() -> None:
    """ASR 模型加载失败时不抛异常，状态为 error，程序继续。"""
    harness = _make_harness(engine=_FakeEngine(load_error=AsrLoadError("模型不在")))

    result = await harness.manager.toggle()

    assert result == RESULT_ERROR
    assert harness.manager.asr_state == STATE_ERROR
    assert harness.manager.is_running is False
    assert harness.manager.audio_state == AUDIO_STATE_OFF
    # 模型没加载起来就不开流
    assert harness.capture.started is False
    assert harness.source_calls == []
    assert harness.final.calls == []


async def test_repeated_stop_is_noop() -> None:
    """重复调用 stop 不会交叉，第二次是空操作。"""
    harness = _make_harness(engine=_FakeEngine(final_text="一句话"))

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))
    await harness.manager.toggle()

    assert harness.engine.flush_calls == 1
    assert len(harness.final.calls) == 1

    # 第二次 stop 是空操作：不重复冲刷，也不重复提交
    await harness.manager.stop()
    assert harness.engine.flush_calls == 1
    assert len(harness.final.calls) == 1
    assert harness.manager.is_running is False


async def test_concurrent_stop_does_not_cross() -> None:
    """两个停止流程不交叉，只提交一次。"""
    harness = _make_harness(engine=_FakeEngine(final_text="一句话"))

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))

    await asyncio.gather(harness.manager.stop(), harness.manager.stop())

    assert harness.engine.flush_calls == 1
    assert len(harness.final.calls) == 1


async def test_second_capture_has_no_residue_from_first() -> None:
    """连续两段录音之间没有互相干扰。"""
    harness = _make_harness(
        engine=_FakeEngine(feed_text=lambda block: f"[{block.source}]")
    )

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))
    await harness.manager.toggle()
    assert harness.final.calls == [("[microphone]", "microphone")]

    harness.final.calls.clear()
    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: len(harness.engine.feed_calls) == 2)
    await harness.manager.toggle()

    # 第二段的识别结果里没有第一段的残留文字
    assert harness.final.calls == [("[microphone]", "microphone")]


# --------------------------------------------------------------------------
# 来源标签
# --------------------------------------------------------------------------
async def test_blocks_from_different_sources_are_not_mixed() -> None:
    """来源标签不同的块不混在一起。"""
    harness = _make_harness(
        engine=_FakeEngine(feed_text=lambda block: f"文字[{block.source}]")
    )

    await harness.manager.toggle()
    harness.capture.push("microphone")
    await _wait_until(lambda: len(harness.partial.calls) == 1)
    harness.capture.push("system_monitor")
    await _wait_until(lambda: len(harness.partial.calls) == 2)

    # 两路文字各走各的回调，没有并成一句
    assert harness.partial.calls == [
        ("文字[microphone]", "microphone"),
        ("文字[system_monitor]", "system_monitor"),
    ]
    for text, _source in harness.partial.calls:
        assert not ("microphone" in text and "system_monitor" in text)
    # 块带着自己的标签进模型，没有被合并
    assert [block.source for block in harness.engine.feed_calls] == [
        "microphone",
        "system_monitor",
    ]
    assert harness.final.calls == []


async def test_separate_engines_keep_final_text_per_source() -> None:
    """每路来源各有引擎时，最终文本也按来源分开。"""

    def engine_factory(source: str) -> _FakeEngine:
        """按来源建一个假引擎。"""
        return _FakeEngine(feed_text=lambda block: f"文字[{block.source}]")

    harness = _make_harness(
        engine=_FakeEngine(feed_text=lambda block: f"文字[{block.source}]"),
        asr_engine_factory=engine_factory,
    )

    await harness.manager.toggle()
    harness.capture.push("microphone")
    await _wait_until(lambda: len(harness.partial.calls) == 1)
    harness.capture.push("system_monitor")
    await _wait_until(lambda: len(harness.partial.calls) == 2)
    await harness.manager.stop()

    assert sorted(text for text, _ in harness.final.calls) == [
        "文字[microphone]",
        "文字[system_monitor]",
    ]
    assert sorted(source for _, source in harness.final.calls) == [
        "microphone",
        "system_monitor",
    ]


# --------------------------------------------------------------------------
# 与真实采集层对接
# --------------------------------------------------------------------------
class _FakeInputStream:
    """假 sounddevice InputStream。"""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self) -> None:
        """开始采集。"""
        self.started = True

    def stop(self) -> None:
        """停止采集。"""
        self.started = False

    def close(self) -> None:
        """关掉流。"""
        self.closed = True


async def test_manager_drives_windows_audio_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """管理器能驱动 Windows 采集后端：块进队列、识别、出最终文本。"""
    fake = ModuleType("sounddevice")
    created: dict[str, Any] = {}
    info = {
        "name": "麦克风阵列 (Realtek(R) Audio)",
        "index": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    }
    fake.default = SimpleNamespace(device=(0, 0))  # type: ignore[attr-defined]
    fake.query_devices = lambda device=None, kind=None: dict(info)  # type: ignore[attr-defined]

    def input_stream(**kwargs: Any) -> _FakeInputStream:
        """记下开流参数，返回假流。"""
        created.update(kwargs)
        return _FakeInputStream(**kwargs)

    fake.InputStream = input_stream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    capture = AudioCapture(WasapiAudioBackend(), 0, source="microphone")
    engine = _FakeEngine(feed_text=lambda block: "识别到的字")
    partial = _Recorder()
    final = _Recorder()

    manager = AudioManager(
        lambda source: capture,
        engine,
        final,
        source="microphone",
        on_partial_text=partial,
    )

    assert await manager.toggle() == RESULT_STARTED
    # 模拟一次 200 毫秒的立体声回调，采集层下混并重采样成 3200 点
    created["callback"](np.zeros((9600, 2), dtype="float32"), 9600, None, None)
    await _wait_until(lambda: bool(partial.calls))
    assert partial.calls == [("识别到的字", "microphone")]

    assert await manager.toggle() == RESULT_STOPPED
    assert final.calls == [("识别到的字", "microphone")]
    assert manager.is_running is False


# --------------------------------------------------------------------------
# 临时识别文字与状态
# --------------------------------------------------------------------------
async def test_partial_callback_receives_cumulative_text() -> None:
    """累计模式下，部分结果回调收到的是这段语音的累计文本。

    临时显示那一行要是一句可读的话，只给最后一个增量会只剩两三个字。
    """
    increments = iter(["你", "好", "呀"])
    harness = _make_harness(
        engine=_FakeEngine(feed_text=lambda block: next(increments)),
        partial_text_cumulative=True,
    )

    await harness.manager.toggle()
    for _ in range(3):
        harness.capture.push()
    await _wait_until(lambda: len(harness.partial.calls) == 3)

    assert harness.partial.texts == ["你", "你好", "你好呀"]


async def test_partial_callback_receives_increments_by_default() -> None:
    """默认传增量：界面层把收到的文字自己累加成临时显示那一行。

    界面层现在是累加的，两边都累加会把同一段文字接成两遍。
    """
    increments = iter(["你", "好", "呀"])
    harness = _make_harness(
        engine=_FakeEngine(feed_text=lambda block: next(increments))
    )

    await harness.manager.toggle()
    for _ in range(3):
        harness.capture.push()
    await _wait_until(lambda: len(harness.partial.calls) == 3)

    assert harness.partial.texts == ["你", "好", "呀"]
    # 累计文本仍然在管理器这侧维护着
    assert harness.manager.partial_text("microphone") == "你好呀"


async def test_partial_accumulation_resets_on_new_stream() -> None:
    """新的一段语音开始时，累计文本从头算。"""
    texts = ["第一段", "第二段"]
    counter = {"index": 0}

    def feed_text(block: Any) -> str:
        """按次数返回不同的增量。"""
        index = counter["index"]
        counter["index"] += 1
        return texts[index % len(texts)]

    harness = _make_harness(
        engine=_FakeEngine(feed_text=feed_text),
        partial_text_cumulative=True,
    )

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: len(harness.partial.calls) == 1)
    assert harness.manager.partial_text("microphone") == "第一段"
    await harness.manager.stop()

    harness.partial.calls.clear()
    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: len(harness.partial.calls) == 1)

    # 第二段的临时显示只有第二段自己的文字，没有第一段的残留
    assert harness.partial.texts == ["第二段"]
    assert harness.manager.partial_text("microphone") == "第二段"


async def test_asr_state_reports_error_after_decode_failure() -> None:
    """解码失败后管理器对外报的 ASR 状态是 error。"""

    class _FailingEngine(_FakeEngine):
        """喂一块就抛错的假引擎。"""

        async def feed(self, block: Any) -> str:
            """假装解码失败。"""
            self.state = STATE_ERROR
            raise RuntimeError("模型崩了")

    harness = _make_harness(engine=_FailingEngine())

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: harness.manager.asr_state == STATE_ERROR)

    assert harness.manager.asr_state == STATE_ERROR
    # 采集本身不受影响
    assert harness.manager.is_running is True


async def test_asr_state_reports_error_after_flush_failure() -> None:
    """冲刷失败后管理器对外报的 ASR 状态是 error。"""

    class _FailingFlushEngine(_FakeEngine):
        """冲刷会抛错的假引擎。"""

        async def flush(self) -> str:
            """假装冲刷失败。"""
            self.state = STATE_ERROR
            raise RuntimeError("冲刷崩了")

    harness = _make_harness(engine=_FailingFlushEngine())

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))

    result = await harness.manager.toggle()

    assert result == RESULT_EMPTY
    assert harness.manager.asr_state == STATE_ERROR


async def test_asr_state_follows_engine_after_recovery() -> None:
    """解码恢复之后，管理器报的状态跟着引擎回到 ready。"""

    class _RecoveringEngine(_FakeEngine):
        """第一次喂失败，之后正常的假引擎。"""

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def feed(self, block: Any) -> str:
            """第一次抛错，之后返回增量。"""
            if not self.failed:
                self.failed = True
                self.state = STATE_ERROR
                raise RuntimeError("模型崩了")
            # 真引擎解码成功后会回到 ready
            self.state = STATE_READY
            return "识别到的字"

    harness = _make_harness(engine=_RecoveringEngine())

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: harness.manager.asr_state == STATE_ERROR)

    harness.capture.push()
    await _wait_until(lambda: bool(harness.partial.calls))

    assert harness.manager.asr_state == STATE_READY


# --------------------------------------------------------------------------
# F8 控制器
# --------------------------------------------------------------------------
async def test_hotkey_toggle_starts_then_stops() -> None:
    """按 F8：没在采集时开始采集，已在采集时停止并提交。"""
    harness = _make_harness(engine=_FakeEngine(final_text="面试官的问题"))
    controller = HotkeyController(harness.manager)

    assert await controller.handle_voice_toggle() == RESULT_STARTED
    assert harness.manager.is_running is True

    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))

    assert await controller.handle_voice_toggle() == RESULT_STOPPED
    assert harness.manager.is_running is False
    assert harness.final.calls == [("面试官的问题", "microphone")]


async def test_hotkey_ignored_while_stop_in_flight() -> None:
    """正在停止时又按了一次 F8，这次按键被忽略，不交叉。"""
    harness = _make_harness(engine=_FakeEngine(final_text="面试官的问题"))
    controller = HotkeyController(harness.manager)

    await controller.handle_voice_toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))

    # 冲刷卡住，制造"正在停止"的窗口
    gate = asyncio.Event()
    harness.engine.flush_gate = gate
    stopping = asyncio.create_task(controller.handle_voice_toggle())
    await asyncio.wait_for(harness.engine.flush_started.wait(), timeout=2.0)

    # 停止还没结束时又按了一次：这次按键被忽略
    assert await controller.handle_voice_toggle() == RESULT_IGNORED

    gate.set()
    assert await stopping == RESULT_STOPPED
    # 两个停止流程没有交叉，只提交一次
    assert harness.engine.flush_calls == 1
    assert len(harness.final.calls) == 1


# --------------------------------------------------------------------------
# asr_final_latency
# --------------------------------------------------------------------------
async def test_stop_with_final_text_records_one_final_latency(
    isolated_recorder: MetricRecorder,
) -> None:
    """有最终文本的 stop 只记一条 asr_final_latency，来源是 audio_manager。"""
    harness = _make_harness(engine=_FakeEngine(final_text="面试官的问题"))

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))
    await harness.manager.stop()

    samples = isolated_recorder.samples(ASR_FINAL_LATENCY_METRIC)
    assert len(samples) == 1
    assert samples[0].source == "audio_manager"
    assert samples[0].value_ms >= 0

    # 停了之后再调一次是空操作，不再记样本
    await harness.manager.stop()
    assert len(isolated_recorder.samples(ASR_FINAL_LATENCY_METRIC)) == 1


async def test_stop_without_final_text_records_no_final_latency(
    isolated_recorder: MetricRecorder,
) -> None:
    """空结果的 stop 不记 asr_final_latency。"""
    harness = _make_harness(engine=_FakeEngine(final_text=""))

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))

    assert await harness.manager.toggle() == RESULT_EMPTY
    assert isolated_recorder.samples(ASR_FINAL_LATENCY_METRIC) == []


async def test_toggle_stop_also_records_final_latency(
    isolated_recorder: MetricRecorder,
) -> None:
    """按 F8 停止走同一条停止流程，同样记一条。"""
    harness = _make_harness(engine=_FakeEngine(final_text="一句话"))

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))
    assert await harness.manager.toggle() == RESULT_STOPPED

    samples = isolated_recorder.samples(ASR_FINAL_LATENCY_METRIC)
    assert len(samples) == 1
    assert samples[0].source == "audio_manager"


async def test_each_stop_records_its_own_final_latency(
    isolated_recorder: MetricRecorder,
) -> None:
    """两次停止各记一条，不攒成一条。"""
    harness = _make_harness()

    for round_index in range(1, 3):
        harness.engine.final_text = f"第{round_index}句"
        await harness.manager.toggle()
        harness.capture.push()
        await _wait_until(lambda: len(harness.engine.feed_calls) == round_index)
        await harness.manager.stop()

        samples = isolated_recorder.samples(ASR_FINAL_LATENCY_METRIC)
        assert len(samples) == round_index
        assert all(sample.source == "audio_manager" for sample in samples)

    stats = isolated_recorder.stats(ASR_FINAL_LATENCY_METRIC)
    assert stats is not None
    assert stats.count == 2


async def test_cancelled_stop_leaves_no_final_latency(
    isolated_recorder: MetricRecorder,
) -> None:
    """停止流程被取消时不留下样本。"""
    harness = _make_harness(engine=_FakeEngine(final_text="一句话"))

    await harness.manager.toggle()
    harness.capture.push()
    await _wait_until(lambda: bool(harness.engine.feed_calls))

    # 冲刷卡住，在拿到最终文本之前取消整个停止流程
    harness.engine.flush_gate = asyncio.Event()
    stopping = asyncio.create_task(harness.manager.stop())
    await asyncio.wait_for(harness.engine.flush_started.wait(), timeout=2.0)
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping

    assert isolated_recorder.samples(ASR_FINAL_LATENCY_METRIC) == []
