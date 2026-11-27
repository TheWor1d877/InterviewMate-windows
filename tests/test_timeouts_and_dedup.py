"""超时管理与 ASR 最终结果去重的测试。

覆盖《8-集成与验收》「超时管理」「防止重复调用」两节：

- 三个超时配置项能加载，值非法（零、负数）时报错；
- 音频设备初始化超时能中断，报 ``E_AUDIO_DEVICE``；
- ASR 初始化超时能中断，报 ``E_ASR_LOAD``；
- 截图接口超时能中断，报 ``E_SCREENSHOT``；
- 同一句 ASR 最终结果只提交一次，不重复触发模型请求；
- 不同的 ASR 最终结果都提交；
- 空文本不提交。

另外验超时与「其它失败」在日志里分得开：超时那条消息里写明超时多少秒。

测试不真加载 FunASR 模型，也不真开音频设备：模型工厂、采集器与截图后端
都是假的，超时靠注入的慢调用制造，见下面的测试替身。
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

import app.config
from app.audio.audio_manager import (
    RESULT_EMPTY,
    RESULT_ERROR,
    RESULT_STOPPED,
    AudioManager,
)
from app.audio.funasr_engine import (
    DEVICE_CPU,
    STATE_ERROR,
    STATE_READY,
    FunASREngine,
)
from app.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from app.core.exceptions import (
    AsrLoadError,
    AudioDeviceError,
    ConfigError,
    ScreenshotError,
)
from app.platform.linux.audio import SoundDeviceAudioBackend
from app.platform.linux.screenshot import GnomeScreenshotBackend
from app.platform.types import Screenshot
from app.vision.image_processor import ProcessedImage
from app.vision.screenshot_manager import ScreenshotManager

# 截图采集失败时写进状态栏提示位的错误码，见《2-骨架与配置》错误码一节
SCREENSHOT_ERROR_CODE = "E_SCREENSHOT"

# 音频设备初始化失败时用的错误码，见《2-骨架与配置》错误码一节
AUDIO_DEVICE_ERROR_CODE = "E_AUDIO_DEVICE"

# ASR 模型加载失败时用的错误码，见《2-骨架与配置》错误码一节
ASR_LOAD_ERROR_CODE = "E_ASR_LOAD"

# 三个超时项在配置里的位置，见《8-集成与验收》「超时管理」一节
TIMEOUT_FIELDS = (
    ("audio", "init_timeout"),
    ("asr", "init_timeout"),
    ("screenshot", "timeout"),
)


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
@dataclass
class _Block:
    """假音频块，只需要一个来源标签。"""

    source: str
    samples: Any = None
    sample_rate: int = 16000
    captured_at: float = 0.0


class _FakeSide:
    """假采集的一路：块从一个队列里出。"""

    def __init__(self, source: str) -> None:
        self.source = source
        self.queue: asyncio.Queue[_Block] = asyncio.Queue()

    async def get(self) -> _Block:
        """等下一块。"""
        return await self.queue.get()

    def push(self) -> None:
        """塞一块音频进去。"""
        self.queue.put_nowait(_Block(source=self.source))


class _TwoSourceCapture:
    """假采集器：麦克风与系统回环两路。

    两路都在时，管理器会开两个取块任务，识别结果按来源标签分开，
    见《5-语音输入与流式ASR》「音频来源」一节。
    """

    source = "mixed"

    def __init__(self) -> None:
        self.microphone = _FakeSide("microphone")
        self.monitor = _FakeSide("system_monitor")
        self.started = False

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

    async def get_microphone_block(self) -> _Block:
        """等麦克风这一路的下一块。"""
        return await self.microphone.get()

    async def get_system_monitor_block(self) -> _Block:
        """等系统回环这一路的下一块。"""
        return await self.monitor.get()


class _FakeEngine:
    """假 ASR 引擎：按来源标签返回固定文本，冲刷返回空串。"""

    def __init__(self, texts: dict[str, str] | None = None, final_text: str = "") -> None:
        self.texts = dict(texts or {})
        self.final_text = final_text
        self.state = STATE_READY
        self.feed_calls: list[Any] = []
        self.flush_calls = 0

    async def ensure_loaded(self) -> None:
        """加载模型，假引擎直接就绪。"""
        self.state = STATE_READY

    async def feed(self, block: Any) -> str:
        """喂一块音频，返回这一路对应的文本。"""
        self.feed_calls.append(block)
        return self.texts.get(str(getattr(block, "source", "")), "")

    async def flush(self) -> str:
        """冲刷，返回最后一段文本。"""
        self.flush_calls += 1
        text = self.final_text
        self.final_text = ""
        return text


class _Recorder:
    """记下最终文本回调收到的内容。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, source: str) -> None:
        """收下一次回调。"""
        self.calls.append((text, source))

    @property
    def texts(self) -> list[str]:
        """只取文本。"""
        return [text for text, _ in self.calls]


class _SlowModelFactory:
    """假模型工厂：等一个闸门才返回，用来制造模型加载超时的场景。"""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.calls = 0

    def __call__(self, *, model: str, model_path: str, device: str) -> Any:
        """等闸门放行再返回假模型。"""
        self.calls += 1
        self.gate.wait(5.0)
        return object()


class _SlowScreenshotBackend:
    """假截图后端：采集一直卡住，用来制造截图接口超时的场景。"""

    name = "slow"
    required_packages: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.capture_calls = 0

    async def probe(self, timeout: float = 3.0) -> str:
        """探测，假后端一直可用。"""
        return "假后端可用"

    async def capture(self, *, target: str = "screen", monitor: int = 0) -> Screenshot:
        """采集一直卡住。"""
        self.capture_calls += 1
        await asyncio.sleep(5)
        return Screenshot(data=b"", width=0, height=0)


class _FakeProcessor:
    """假图片处理器。"""

    def process(self, raw: bytes) -> ProcessedImage:
        """返回一张假图片。"""
        return ProcessedImage(data=b"processed", mime="image/png", width=1, height=1)

    def to_data_url(self, image: ProcessedImage) -> str:
        """返回固定的 data URL。"""
        return "data:image/png;base64,AAAA"


class _FakeQuestionManager:
    """假问题管理器，记录提交次数。"""

    def __init__(self) -> None:
        self.submissions: list[tuple[str, str | None]] = []

    async def submit_text(self, text: str, image_data_url: str | None = None) -> None:
        """记录一次提交。"""
        self.submissions.append((text, image_data_url))


def _slow_query_devices(*args: Any, **kwargs: Any) -> list[Any]:
    """卡住不返回的设备枚举，用来制造超时。"""
    time.sleep(0.5)
    return []


# --------------------------------------------------------------------------
# 夹具与辅助
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_loaded_config() -> Iterator[None]:
    """每个用例后把进程里加载好的配置还原回去。

    ``load_config`` 会记下进程内最近一次加载成功的配置，
    别的模块从这里取超时值；用例里改过的配置不能留在进程里。
    """
    saved = app.config._loaded_config
    try:
        yield
    finally:
        app.config._loaded_config = saved


@pytest.fixture
def project_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """收集 interviewmate 记录器上的日志。

    临时打开 propagate，让记录走到 pytest 的 handler 上。
    """
    project_logger = logging.getLogger("interviewmate")
    old_propagate = project_logger.propagate
    project_logger.propagate = True
    caplog.set_level(logging.INFO, logger="interviewmate")
    try:
        yield caplog
    finally:
        project_logger.propagate = old_propagate


def _events(caplog: pytest.LogCaptureFixture, event: str) -> list[logging.LogRecord]:
    """挑出指定事件名的日志记录。"""
    return [r for r in caplog.records if getattr(r, "event", "") == event]


def _config_data() -> dict[str, Any]:
    """以仓库里的默认配置文件作为测试基底。"""
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    return data


def _load_with(tmp_path: Path, section: str, key: str, value: Any) -> AppConfig:
    """把配置基底里的一项改成给定值，再加载一份配置。"""
    data = _config_data()
    data[section][key] = value
    target = tmp_path / "config.yaml"
    target.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return load_config(str(target))


async def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    """等条件成立，等不到就报错。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("等待条件成立超时")


# --------------------------------------------------------------------------
# 配置项
# --------------------------------------------------------------------------
def test_config_yaml_declares_the_three_timeouts() -> None:
    """三个超时项在配置文件里都写着，取值与加载出来的一致。"""
    data = _config_data()
    config = load_config()

    assert data["audio"]["init_timeout"] == config.audio.init_timeout
    assert data["asr"]["init_timeout"] == config.asr.init_timeout
    assert data["screenshot"]["timeout"] == config.screenshot.timeout
    assert config.audio.init_timeout > 0
    assert config.asr.init_timeout > 0
    assert config.screenshot.timeout > 0


@pytest.mark.parametrize("section,key", TIMEOUT_FIELDS)
@pytest.mark.parametrize("bad_value", [0, -1, -60])
def test_timeout_values_must_be_positive(
    tmp_path: Path, section: str, key: str, bad_value: int
) -> None:
    """超时值是零或负数时配置校验失败。"""
    with pytest.raises(ConfigError):
        _load_with(tmp_path, section, key, bad_value)


def test_old_config_without_timeout_fields_still_loads(tmp_path: Path) -> None:
    """老的手写配置里没有这三个超时项也能加载，取默认值。"""
    data = _config_data()
    data["audio"].pop("init_timeout")
    data["asr"].pop("init_timeout")
    data["screenshot"].pop("timeout")
    target = tmp_path / "config.yaml"
    target.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    config = load_config(str(target))

    assert config.audio.init_timeout > 0
    assert config.asr.init_timeout > 0
    assert config.screenshot.timeout > 0


def test_config_timeouts_reach_the_code_paths() -> None:
    """三个超时项都接到了真正用它们的地方。"""
    config = load_config()

    assert SoundDeviceAudioBackend().init_timeout == float(config.audio.init_timeout)
    assert FunASREngine().init_timeout == float(config.asr.init_timeout)
    assert GnomeScreenshotBackend().timeout == float(config.screenshot.timeout)


# --------------------------------------------------------------------------
# 音频设备初始化超时
# --------------------------------------------------------------------------
def test_audio_device_enumeration_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设备枚举卡住时超时中断，报 E_AUDIO_DEVICE。"""
    module = ModuleType("sounddevice")
    module.query_devices = _slow_query_devices  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    backend = SoundDeviceAudioBackend(init_timeout=0.05)
    with pytest.raises(AudioDeviceError) as excinfo:
        backend.list_devices()

    assert excinfo.value.code == AUDIO_DEVICE_ERROR_CODE
    assert "超时" in excinfo.value.message
    assert "0.05" in excinfo.value.message


def test_audio_stream_open_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """开流时设备查询卡住，超时中断，报 E_AUDIO_DEVICE。"""
    module = ModuleType("sounddevice")
    module.query_devices = _slow_query_devices  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    backend = SoundDeviceAudioBackend(init_timeout=0.05)
    with pytest.raises(AudioDeviceError) as excinfo:
        backend.start_stream(
            device=None,
            sample_rate=16000,
            channels=1,
            block_duration_ms=200,
            on_block=lambda samples: None,
        )

    assert excinfo.value.code == AUDIO_DEVICE_ERROR_CODE
    assert "超时" in excinfo.value.message
    assert "0.05" in excinfo.value.message


async def test_audio_manager_stops_when_device_init_times_out(
    project_logs: pytest.LogCaptureFixture,
) -> None:
    """设备枚举卡住时采集起不来，日志里写明超时秒数与错误码。"""

    def slow_factory(source: str) -> Any:
        """卡住不返回的设备枚举。"""
        time.sleep(0.5)
        return _TwoSourceCapture()

    manager = AudioManager(
        slow_factory,
        _FakeEngine(),
        _Recorder(),
        init_timeout=0.05,
    )

    result = await manager.toggle()

    assert result == RESULT_ERROR
    assert manager.is_running is False

    records = _events(project_logs, "audio.capture.start_failed")
    assert records, "设备初始化超时时没有记日志"
    fields = getattr(records[0], "fields", {}) or {}
    assert AUDIO_DEVICE_ERROR_CODE in str(fields.get("code", ""))
    assert "超时" in records[0].getMessage()
    assert "0.05" in records[0].getMessage()


# --------------------------------------------------------------------------
# ASR 初始化超时
# --------------------------------------------------------------------------
async def test_asr_model_load_times_out() -> None:
    """模型加载卡住时超时中断，报 E_ASR_LOAD。"""
    factory = _SlowModelFactory()
    engine = FunASREngine(
        model_factory=factory,
        device=DEVICE_CPU,
        vad_enabled=False,
        init_timeout=0.05,
    )

    with pytest.raises(AsrLoadError) as excinfo:
        await engine.ensure_loaded()

    assert excinfo.value.code == ASR_LOAD_ERROR_CODE
    assert "超时" in excinfo.value.message
    assert "0.05" in excinfo.value.message
    assert engine.state == STATE_ERROR


async def test_asr_load_timeout_keeps_the_background_load(
    project_logs: pytest.LogCaptureFixture,
) -> None:
    """超时之后后台那个加载任务还在跑，没有重复加载，也没留下悬挂任务。"""
    factory = _SlowModelFactory()
    engine = FunASREngine(
        model_factory=factory,
        device=DEVICE_CPU,
        vad_enabled=False,
        init_timeout=0.05,
    )

    with pytest.raises(AsrLoadError):
        await engine.ensure_loaded()

    # 超时了但加载线程还活着：模型没有重复加载
    assert engine.is_loading is True
    assert factory.calls == 1

    # 放行后台加载，等它自己结束
    factory.gate.set()
    await _wait_until(lambda: not engine.is_loading)

    # 加载已经完成，引擎就绪，模型仍然只加载了一次
    await engine.ensure_loaded()
    assert engine.state == STATE_READY
    assert factory.calls == 1

    records = _events(project_logs, "asr.model.load_failed")
    assert records, "加载超时时没有记日志"
    assert "超时" in records[0].getMessage()


# --------------------------------------------------------------------------
# 截图接口超时
# --------------------------------------------------------------------------
async def test_screenshot_capture_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实抓屏卡住时超时中断，报 E_SCREENSHOT。"""
    backend = GnomeScreenshotBackend(timeout=0.05)

    async def _hang() -> Any:
        """连不上总线，一直挂着。"""
        await asyncio.sleep(5)
        return None

    monkeypatch.setattr(backend, "_open_bus", _hang)

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert excinfo.value.code == SCREENSHOT_ERROR_CODE
    assert "超时" in excinfo.value.message
    assert "0.05" in excinfo.value.message


async def test_screenshot_manager_reports_capture_timeout(tmp_path: Path) -> None:
    """截图接口超时时状态栏提示 E_SCREENSHOT，程序继续跑。"""
    config = _load_with(tmp_path, "screenshot", "timeout", 1)
    backend = _SlowScreenshotBackend()
    hints: list[str] = []

    async def _publish(hint: str) -> None:
        """写状态栏提示位。"""
        hints.append(hint)

    question_manager = _FakeQuestionManager()
    manager = ScreenshotManager(
        config,
        backend,
        _FakeProcessor(),
        question_manager,
        hint_publisher=_publish,
    )

    result = await manager.capture_and_submit()

    assert result is False
    assert backend.capture_calls == 1
    assert question_manager.submissions == []
    assert hints, "截图超时时没有写提示位"
    assert SCREENSHOT_ERROR_CODE in hints[-1]
    assert "超时" in hints[-1]
    assert "1 秒" in hints[-1]


# --------------------------------------------------------------------------
# ASR 最终结果去重
# --------------------------------------------------------------------------
async def test_same_final_text_is_submitted_once() -> None:
    """同一句最终结果只提交一次，不重复触发模型请求。"""
    capture = _TwoSourceCapture()
    engine = _FakeEngine(
        {
            "microphone": "请讲讲 LRU 缓存",
            "system_monitor": "请讲讲 LRU 缓存",
        }
    )
    recorder = _Recorder()
    manager = AudioManager(lambda source: capture, engine, recorder)

    await manager.toggle()
    capture.microphone.push()
    capture.monitor.push()
    await _wait_until(lambda: len(engine.feed_calls) == 2)
    result = await manager.toggle()

    assert result == RESULT_STOPPED
    assert recorder.texts == ["请讲讲 LRU 缓存"]


async def test_same_final_text_with_different_punctuation_is_submitted_once() -> None:
    """规范化之后相同的最终结果也只提交一次。"""
    capture = _TwoSourceCapture()
    engine = _FakeEngine(
        {
            "microphone": "请讲讲 LRU 缓存",
            "system_monitor": "请讲讲，LRU 缓存。",
        }
    )
    recorder = _Recorder()
    manager = AudioManager(lambda source: capture, engine, recorder)

    await manager.toggle()
    capture.microphone.push()
    capture.monitor.push()
    await _wait_until(lambda: len(engine.feed_calls) == 2)
    await manager.toggle()

    assert recorder.texts == ["请讲讲 LRU 缓存"]


async def test_different_final_texts_are_all_submitted() -> None:
    """不同的句子都提交。"""
    capture = _TwoSourceCapture()
    engine = _FakeEngine(
        {
            "microphone": "请讲讲 LRU 缓存",
            "system_monitor": "再讲讲线程池",
        }
    )
    recorder = _Recorder()
    manager = AudioManager(lambda source: capture, engine, recorder)

    await manager.toggle()
    capture.microphone.push()
    capture.monitor.push()
    await _wait_until(lambda: len(engine.feed_calls) == 2)
    result = await manager.toggle()

    assert result == RESULT_STOPPED
    assert recorder.texts == ["请讲讲 LRU 缓存", "再讲讲线程池"]


async def test_empty_final_text_is_not_submitted() -> None:
    """空文本不提交，也不触发模型请求。"""
    capture = _TwoSourceCapture()
    engine = _FakeEngine({"microphone": "", "system_monitor": ""})
    recorder = _Recorder()
    manager = AudioManager(lambda source: capture, engine, recorder)

    await manager.toggle()
    capture.microphone.push()
    capture.monitor.push()
    await _wait_until(lambda: len(engine.feed_calls) == 2)
    result = await manager.toggle()

    assert result == RESULT_EMPTY
    assert recorder.calls == []


async def test_punctuation_only_final_text_is_not_submitted() -> None:
    """只有标点的文本规范化之后是空串，不提交。"""
    capture = _TwoSourceCapture()
    engine = _FakeEngine({"microphone": "。。。", "system_monitor": "。。。"})
    recorder = _Recorder()
    manager = AudioManager(lambda source: capture, engine, recorder)

    await manager.toggle()
    capture.microphone.push()
    capture.monitor.push()
    await _wait_until(lambda: len(engine.feed_calls) == 2)
    result = await manager.toggle()

    assert result == RESULT_EMPTY
    assert recorder.calls == []
