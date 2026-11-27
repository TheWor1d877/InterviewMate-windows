"""音频采集层的测试。

覆盖《5-语音输入与流式ASR》「音频来源」与「采集参数」两节要求的行为：

- 设备枚举把名字带 ``.monitor`` 的设备标成系统回环；
- 来源选择同类设备里优先系统默认设备，``auto`` 两类齐全时混成一路，
  只有一路时用那一路，两个都没有时报 ``E_AUDIO_DEVICE``；
- 采集器 start 之后 ``is_running`` 为真，stop 之后为假；
- 队列满时丢弃最旧的块并写日志；
- 麦克风与系统回环的块各带自己的来源标签。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import pytest

from app.audio.audio_capture import AudioBlock, AudioCapture, BoundedAudioQueue
from app.audio.audio_devices import (
    SOURCE_MICROPHONE,
    SOURCE_MIXED,
    SOURCE_SYSTEM_MONITOR,
    select_source,
    split_devices,
)
from app.audio.audio_source import MixedCapture, SeparatedCapture, create_capture
from app.core.exceptions import AudioDeviceError
from app.platform.linux.audio import SoundDeviceAudioBackend
from app.platform.types import AudioDevice

np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
def _device(
    index: int,
    name: str,
    *,
    loopback: bool = False,
    is_input: bool = True,
    is_default: bool = False,
) -> AudioDevice:
    """造一个设备，默认是输入设备。"""
    return AudioDevice(
        name=name,
        index=index,
        is_input=is_input,
        is_loopback=loopback,
        default_samplerate=48000.0,
        channels=2,
        is_default=is_default,
    )


class _FakeStream:
    """假音频流，记下有没有被关掉。"""

    def __init__(self) -> None:
        self.active = True

    @property
    def is_active(self) -> bool:
        """流是不是还开着。"""
        return self.active

    def stop(self) -> None:
        """关掉流。"""
        self.active = False


class _FakeAudioBackend:
    """假音频后端：记下开流参数，按设备推块。"""

    name = "fake-audio"
    required_packages = ()

    def __init__(self, devices: list[AudioDevice] | None = None) -> None:
        self.devices = list(devices or [])
        self.streams: list[dict[str, Any]] = []

    def list_devices(self) -> list[AudioDevice]:
        """列出设备。"""
        return list(self.devices)

    def record(self, **kwargs: Any) -> Any:
        """采一段音频，本测试用不到。"""
        raise NotImplementedError

    def start_stream(
        self,
        *,
        device: int | None,
        sample_rate: int,
        channels: int,
        block_duration_ms: int,
        on_block: Any,
    ) -> _FakeStream:
        """开一路假流。"""
        self.streams.append(
            {
                "device": device,
                "sample_rate": sample_rate,
                "channels": channels,
                "block_duration_ms": block_duration_ms,
                "on_block": on_block,
                "handle": _FakeStream(),
            }
        )
        return self.streams[-1]["handle"]

    def emit(self, device: int | None, samples: Any) -> None:
        """模拟采集回调线程推一块音频。"""
        for stream in self.streams:
            if stream["device"] == device:
                stream["on_block"](samples)


class _ListHandler(logging.Handler):
    """把日志记录攒起来的 handler，用来看清 log_event 传了什么。"""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """收下这条记录。"""
        self.records.append(record)


@pytest.fixture()
def captured_logs() -> list[logging.LogRecord]:
    """把项目日志记录攒起来，测试里检查事件名与字段。"""
    logger = logging.getLogger("interviewmate")
    handler = _ListHandler()
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _block(source: str, marker: float) -> AudioBlock:
    """造一块带标记值的音频块。"""
    return AudioBlock(
        samples=np.full(3200, marker, dtype="float32"),
        source=source,
        sample_rate=16000,
        captured_at=marker,
    )


# --------------------------------------------------------------------------
# 设备枚举
# --------------------------------------------------------------------------
def test_list_devices_marks_monitor_as_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """名字带 .monitor 的设备标成回环，其它输入设备不是。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(
        sd,
        "query_devices",
        lambda: [
            {
                "name": "alsa_output.pci-0000_06_00.6.analog-stereo.monitor",
                "index": 12,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "alsa_input.pci-0000_06_00.6.analog-stereo",
                "index": 13,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "扬声器",
                "index": 3,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
        ],
    )
    devices = SoundDeviceAudioBackend().list_devices()
    assert [device.index for device in devices] == [12, 13]
    assert devices[0].is_loopback is True
    assert devices[1].is_loopback is False


def test_list_devices_skips_output_only_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有输入声道的设备不进设备表。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(
        sd,
        "query_devices",
        lambda: [
            {
                "name": "扬声器",
                "index": 3,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            }
        ],
    )
    assert SoundDeviceAudioBackend().list_devices() == []


class _FakeInputStream:
    """假 InputStream，记下开流参数。"""

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


def _patch_sounddevice_stream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    native_rate: float,
    native_channels: int,
) -> dict[str, Any]:
    """把 sounddevice 的设备查询与 InputStream 换成假的，记下开流参数。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(
        sd,
        "query_devices",
        lambda device=None, kind=None: {
            "name": "alsa_output.pci-0000_06_00.6.analog-stereo.monitor",
            "index": 12,
            "max_input_channels": native_channels,
            "max_output_channels": 0,
            "default_samplerate": native_rate,
        },
    )
    created: dict[str, Any] = {}

    def fake_input_stream(**kwargs: Any) -> _FakeInputStream:
        created.update(kwargs)
        return _FakeInputStream(**kwargs)

    monkeypatch.setattr(sd, "InputStream", fake_input_stream)
    return created


def test_start_stream_resamples_native_rate_to_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设备原生 48000 Hz 时，按 200 毫秒开流并重采样成 16000 Hz 单声道。"""
    created = _patch_sounddevice_stream(
        monkeypatch, native_rate=48000.0, native_channels=2
    )
    blocks: list[Any] = []
    stream = SoundDeviceAudioBackend().start_stream(
        device=12,
        sample_rate=16000,
        channels=1,
        block_duration_ms=200,
        on_block=blocks.append,
    )
    # 流按设备原生采样率开，块长是 200 毫秒
    assert created["samplerate"] == 48000.0
    assert created["blocksize"] == 9600
    assert created["channels"] == 2
    assert created["dtype"] == "float32"

    # 手动调一次回调，模拟 200 毫秒的原生数据
    created["callback"](np.zeros((9600, 2), dtype="float32"), 9600, None, None)
    assert len(blocks) == 1
    assert blocks[0].shape == (3200,)
    assert blocks[0].dtype == np.float32
    stream.stop()


def test_start_stream_downmixes_stereo_to_mono(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """立体声设备下混成单声道。"""
    created = _patch_sounddevice_stream(
        monkeypatch, native_rate=48000.0, native_channels=2
    )
    blocks: list[Any] = []
    stream = SoundDeviceAudioBackend().start_stream(
        device=12,
        sample_rate=16000,
        channels=1,
        block_duration_ms=200,
        on_block=blocks.append,
    )
    stereo = np.zeros((9600, 2), dtype="float32")
    stereo[:, 0] = 1.0
    created["callback"](stereo, 9600, None, None)
    assert len(blocks) == 1
    assert blocks[0].shape == (3200,)
    assert pytest.approx(0.5, abs=1e-6) == float(blocks[0].mean())
    stream.stop()


def test_start_stream_caps_virtual_device_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设备报出上百个输入声道时只按立体声开流，避免 PortAudio 崩溃。"""
    created = _patch_sounddevice_stream(
        monkeypatch, native_rate=48000.0, native_channels=128
    )
    stream = SoundDeviceAudioBackend().start_stream(
        device=12,
        sample_rate=16000,
        channels=1,
        block_duration_ms=200,
        on_block=lambda samples: None,
    )
    assert created["channels"] == 2
    stream.stop()


def test_start_stream_resamples_44100_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设备原生 44100 Hz 时同样重采样成 16000 Hz。"""
    created = _patch_sounddevice_stream(
        monkeypatch, native_rate=44100.0, native_channels=1
    )
    blocks: list[Any] = []
    stream = SoundDeviceAudioBackend().start_stream(
        device=12,
        sample_rate=16000,
        channels=1,
        block_duration_ms=200,
        on_block=blocks.append,
    )
    assert created["samplerate"] == 44100.0
    assert created["blocksize"] == 8820
    created["callback"](np.zeros((8820, 1), dtype="float32"), 8820, None, None)
    assert len(blocks) == 1
    assert blocks[0].shape == (3200,)
    stream.stop()


def test_start_stream_wraps_failure_into_audio_device_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """开流失败转成 AudioDeviceError。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(
        sd,
        "query_devices",
        lambda device=None, kind=None: {
            "name": "x.monitor",
            "index": 12,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
    )

    def fake_input_stream(**kwargs: Any) -> Any:
        raise RuntimeError("设备被占用")

    monkeypatch.setattr(sd, "InputStream", fake_input_stream)
    with pytest.raises(AudioDeviceError) as excinfo:
        SoundDeviceAudioBackend().start_stream(
            device=12,
            sample_rate=16000,
            channels=1,
            block_duration_ms=200,
            on_block=lambda samples: None,
        )
    assert excinfo.value.code == "E_AUDIO_DEVICE"


# --------------------------------------------------------------------------
# 来源选择
# --------------------------------------------------------------------------
def test_select_source_auto_mixes_default_microphone_and_loopback() -> None:
    """auto 两类都齐全时混成一路，两路都取系统默认设备。"""
    devices = [
        _device(1, "麦克风"),
        _device(2, "默认麦克风", is_default=True),
        _device(3, "alsa_output.pci-0000_06_00.6.analog-stereo.monitor", loopback=True),
        _device(4, "默认回环.monitor", loopback=True, is_default=True),
    ]
    selection = select_source("auto", devices)
    assert selection.source == SOURCE_MIXED
    assert [device.index for device in selection.devices] == [2, 4]


def test_select_source_auto_only_loopback() -> None:
    """只有回环时 auto 选中默认回环。"""
    devices = [
        _device(2, "普通回环.monitor", loopback=True),
        _device(3, "默认回环.monitor", loopback=True, is_default=True),
    ]
    selection = select_source("auto", devices)
    assert selection.source == SOURCE_SYSTEM_MONITOR
    assert selection.device.index == 3


def test_select_source_auto_falls_back_to_microphone() -> None:
    """只有麦克风时 auto 选中默认麦克风。"""
    devices = [
        _device(1, "麦克风"),
        _device(2, "默认麦克风", is_default=True),
    ]
    selection = select_source("auto", devices)
    assert selection.source == SOURCE_MICROPHONE
    assert selection.device.index == 2


def test_select_source_microphone_prefers_default_device() -> None:
    """麦克风里优先默认设备，不看列表顺序。"""
    devices = [
        _device(1, "麦克风"),
        _device(2, "默认麦克风", is_default=True),
        _device(3, "回环.monitor", loopback=True),
    ]
    selection = select_source(SOURCE_MICROPHONE, devices)
    assert selection.source == SOURCE_MICROPHONE
    assert selection.device.index == 2


def test_select_source_system_monitor_prefers_default_device() -> None:
    """回环里优先默认设备，不看列表顺序。"""
    devices = [
        _device(1, "麦克风"),
        _device(2, "普通回环.monitor", loopback=True),
        _device(3, "默认回环.monitor", loopback=True, is_default=True),
    ]
    selection = select_source(SOURCE_SYSTEM_MONITOR, devices)
    assert selection.source == SOURCE_SYSTEM_MONITOR
    assert selection.device.index == 3


def test_select_source_auto_without_device_raises() -> None:
    """两个都不通时报 E_AUDIO_DEVICE。"""
    with pytest.raises(AudioDeviceError) as excinfo:
        select_source("auto", [])
    assert excinfo.value.code == "E_AUDIO_DEVICE"


def test_select_source_microphone_without_device_raises() -> None:
    """指定麦克风但没有麦克风时报错。"""
    with pytest.raises(AudioDeviceError) as excinfo:
        select_source("microphone", [_device(2, "x.monitor", loopback=True)])
    assert excinfo.value.code == "E_AUDIO_DEVICE"


def test_select_source_mixed_needs_both_devices() -> None:
    """mixed 两路都要有，缺一路报错。"""
    with pytest.raises(AudioDeviceError):
        select_source("mixed", [_device(1, "麦克风")])
    selection = select_source(
        "mixed",
        [
            _device(1, "麦克风"),
            _device(2, "x.monitor", loopback=True),
        ],
    )
    assert selection.source == SOURCE_MIXED
    assert [device.index for device in selection.devices] == [1, 2]


def test_select_source_rejects_unknown_source() -> None:
    """来源名不合法时报错。"""
    with pytest.raises(AudioDeviceError):
        select_source("speaker", [_device(1, "麦克风")])


def test_split_devices_separates_microphone_and_loopback() -> None:
    """设备表分得开麦克风与回环。"""
    microphones, loopbacks = split_devices(
        [
            _device(1, "麦克风"),
            _device(2, "x.monitor", loopback=True),
            _device(3, "扬声器", is_input=False),
        ]
    )
    assert [device.index for device in microphones] == [1]
    assert [device.index for device in loopbacks] == [2]


# --------------------------------------------------------------------------
# 采集器启停
# --------------------------------------------------------------------------
async def test_capture_start_and_stop_toggles_is_running() -> None:
    """start 之后 is_running 为真，stop 之后为假。"""
    backend = _FakeAudioBackend()
    capture = AudioCapture(backend, 1, source=SOURCE_MICROPHONE)
    assert capture.is_running is False
    await capture.start()
    assert capture.is_running is True
    await capture.stop()
    assert capture.is_running is False


async def test_capture_start_passes_capture_parameters() -> None:
    """开流参数按《5》的采集参数一节传下去。"""
    backend = _FakeAudioBackend()
    capture = AudioCapture(backend, 12, source=SOURCE_SYSTEM_MONITOR)
    await capture.start()
    stream = backend.streams[0]
    assert stream["device"] == 12
    assert stream["sample_rate"] == 16000
    assert stream["channels"] == 1
    assert stream["block_duration_ms"] == 200
    assert stream["handle"].active is True
    await capture.stop()
    assert stream["handle"].active is False


async def test_capture_stop_without_start_is_noop() -> None:
    """没采集时 stop 不报错。"""
    backend = _FakeAudioBackend()
    capture = AudioCapture(backend, 1, source=SOURCE_MICROPHONE)
    await capture.stop()
    assert capture.is_running is False
    assert backend.streams == []


async def test_capture_start_wraps_backend_failure() -> None:
    """开流失败转成 AudioDeviceError。"""

    class _FailingBackend(_FakeAudioBackend):
        def start_stream(self, **kwargs: Any) -> Any:
            raise RuntimeError("设备打不开")

    capture = AudioCapture(_FailingBackend(), 1, source=SOURCE_MICROPHONE)
    with pytest.raises(AudioDeviceError) as excinfo:
        await capture.start()
    assert excinfo.value.code == "E_AUDIO_DEVICE"
    assert capture.is_running is False


async def test_capture_receives_block_from_callback_thread() -> None:
    """采集回调线程推的块能进队列。"""
    backend = _FakeAudioBackend()
    capture = AudioCapture(backend, 1, source=SOURCE_MICROPHONE)
    await capture.start()

    def worker() -> None:
        backend.emit(1, np.zeros(3200, dtype="float32"))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    block = await asyncio.wait_for(capture.get_block(), timeout=2.0)
    assert block.source == SOURCE_MICROPHONE
    assert block.samples.shape == (3200,)
    await capture.stop()


# --------------------------------------------------------------------------
# 有界队列
# --------------------------------------------------------------------------
async def test_bounded_queue_drops_oldest_block_and_logs(
    captured_logs: list[logging.LogRecord],
) -> None:
    """队列满时丢弃最旧的块，丢弃计数写日志。"""
    queue = BoundedAudioQueue(2)
    queue.push(_block(SOURCE_MICROPHONE, 1.0))
    queue.push(_block(SOURCE_MICROPHONE, 2.0))
    assert queue.dropped_count == 0

    assert queue.push(_block(SOURCE_MICROPHONE, 3.0)) is True
    assert queue.dropped_count == 1
    assert queue.size == 2

    # 最旧的块被丢了，剩下的是第 2 块与第 3 块
    assert queue.get_nowait().captured_at == 2.0
    assert queue.get_nowait().captured_at == 3.0

    dropped = [
        record
        for record in captured_logs
        if getattr(record, "event", "") == "audio.capture.block_dropped"
    ]
    assert dropped, "丢弃时没有写日志"
    fields = getattr(dropped[-1], "fields", {})
    assert fields["dropped"] == 1
    assert fields["queue_capacity"] == 2


async def test_bounded_queue_rejects_bad_capacity() -> None:
    """容量不是正数时报错。"""
    with pytest.raises(ValueError):
        BoundedAudioQueue(0)


# --------------------------------------------------------------------------
# 分离采集与混合采集
# --------------------------------------------------------------------------
async def test_separated_capture_blocks_carry_own_source_labels() -> None:
    """麦克风与系统回环的块各带自己的来源标签。"""
    backend = _FakeAudioBackend()
    capture = SeparatedCapture(
        backend,
        microphone=_device(1, "麦克风"),
        monitor=_device(2, "alsa_output.pci-0000_06_00.6.analog-stereo.monitor", loopback=True),
    )
    await capture.start()
    backend.emit(1, np.zeros(3200, dtype="float32"))
    backend.emit(2, np.ones(3200, dtype="float32"))

    mic_block = await asyncio.wait_for(capture.get_microphone_block(), timeout=2.0)
    monitor_block = await asyncio.wait_for(
        capture.get_system_monitor_block(), timeout=2.0
    )
    assert mic_block.source == SOURCE_MICROPHONE
    assert monitor_block.source == SOURCE_SYSTEM_MONITOR
    assert float(mic_block.samples.max()) == 0.0
    assert float(monitor_block.samples.max()) == 1.0
    await capture.stop()
    assert capture.is_running is False


async def test_mixed_capture_produces_one_mixed_stream() -> None:
    """混成一路时只有一条流，标签是 mixed。"""
    backend = _FakeAudioBackend()
    capture = MixedCapture(
        backend,
        microphone=_device(1, "麦克风"),
        monitor=_device(2, "x.monitor", loopback=True),
    )
    await capture.start()
    backend.emit(1, np.full(3200, 0.5, dtype="float32"))
    backend.emit(2, np.full(3200, 0.25, dtype="float32"))

    block = await asyncio.wait_for(capture.get_block(), timeout=2.0)
    assert block.source == SOURCE_MIXED
    assert np.allclose(block.samples, 0.75)
    await capture.stop()
    assert capture.is_running is False


async def test_mixed_capture_stops_when_one_source_fails() -> None:
    """第二路开不起来时第一路也收掉。"""
    backend = _FakeAudioBackend()

    class _FailingBackend(_FakeAudioBackend):
        def start_stream(self, **kwargs: Any) -> Any:
            if kwargs.get("device") == 2:
                raise RuntimeError("回环打不开")
            return super().start_stream(**kwargs)

    capture = MixedCapture(
        _FailingBackend(),
        microphone=_device(1, "麦克风"),
        monitor=_device(2, "x.monitor", loopback=True),
    )
    with pytest.raises(AudioDeviceError):
        await capture.start()
    assert capture.microphone.is_running is False


def test_create_capture_builds_capture_per_source() -> None:
    """来源选择的结果对应到对应的采集器。"""
    backend = _FakeAudioBackend()
    microphone = _device(1, "麦克风")
    monitor = _device(2, "x.monitor", loopback=True)

    single = create_capture(
        backend,
        select_source("microphone", [microphone, monitor]),
    )
    assert isinstance(single, AudioCapture)
    assert single.source == SOURCE_MICROPHONE
    assert single.device == 1

    mixed = create_capture(
        backend,
        select_source("mixed", [microphone, monitor]),
    )
    assert isinstance(mixed, MixedCapture)
