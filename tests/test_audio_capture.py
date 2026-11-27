"""音频采集层的测试。

覆盖《5-语音输入与流式ASR》「音频来源」与「采集参数」两节要求的行为：

- Windows 后端把 ``sounddevice`` 的输入设备当麦克风，把 ``soundcard``
  的 WASAPI loopback 设备标成系统回环，默认麦克风与默认扬声器端点都标默认；
- 来源选择同类设备里优先系统默认设备，``auto`` 两类齐全时混成一路，
  只有一路时用那一路，两个都没有时报 ``E_AUDIO_DEVICE``；
- 采集器 start 之后 ``is_running`` 为真，stop 之后为假；
- 队列满时丢弃最旧的块并写日志；
- 麦克风与系统回环的块各带自己的来源标签。

声卡全部用假模块替掉：``sounddevice`` 与 ``soundcard`` 通过
``monkeypatch.setitem(sys.modules, ...)`` 注入，测试不碰真实设备。
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
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
from app.platform.windows.audio import LOOPBACK_INDEX_BASE, WasapiAudioBackend
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


MICROPHONE_NAME = "麦克风阵列 (Realtek(R) Audio)"
USB_MICROPHONE_NAME = "麦克风 (USB Audio)"
SPEAKER_NAME = "扬声器 (Realtek(R) Audio)"
HDMI_SPEAKER_NAME = "扬声器 (HDMI)"


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


class _FakeInputStream:
    """假 sounddevice InputStream，记下开流参数。"""

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


class _FakeLoopbackDevice:
    """假 soundcard 回环设备。

    :param device_id: WASAPI 端点 ID，与扬声器的 ID 相同。
    :param name: 设备名。
    :param channels: 端点声道数。
    :param frames: 每次读块返回的数据，可以是 ``(帧数, 声道数) -> array``。
    """

    def __init__(
        self,
        device_id: str,
        name: str,
        *,
        channels: int = 2,
        frames: Any = None,
        isloopback: bool = True,
        open_error: BaseException | None = None,
    ) -> None:
        self.id = device_id
        self.name = name
        self.channels = channels
        self.isloopback = isloopback
        self.frames = frames
        self.open_error = open_error
        self.recorder_calls: list[dict[str, Any]] = []
        self.record_calls: list[dict[str, Any]] = []

    def recorder(
        self,
        samplerate: float,
        channels: int | None = None,
        blocksize: int | None = None,
    ) -> "_FakeRecorder":
        """返回假 recorder，记下开流参数。"""
        if self.open_error is not None:
            raise self.open_error
        self.recorder_calls.append(
            {"samplerate": samplerate, "channels": channels, "blocksize": blocksize}
        )
        return _FakeRecorder(self)

    def record(self, numframes: int, samplerate: float, channels: int | None = None) -> Any:
        """一次读完，测试里用不到。"""
        raise NotImplementedError


class _FakeRecorder:
    """假 soundcard recorder，``record`` 按请求帧数出数据。"""

    def __init__(self, owner: _FakeLoopbackDevice) -> None:
        self.owner = owner
        self.closed = False

    def __enter__(self) -> "_FakeRecorder":
        """进入采集。"""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """退出采集。"""
        self.closed = True

    def record(self, numframes: int | None = None) -> Any:
        """读一块音频，稍等一下让停止信号能生效。"""
        frames = int(numframes or 1)
        self.owner.record_calls.append({"numframes": frames})
        if self.owner.frames is not None:
            data = self.owner.frames(frames, self.owner.channels)
        else:
            data = np.zeros((frames, self.owner.channels), dtype="float32")
        time.sleep(0.002)
        return data


def _install_fake_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
    *,
    devices: list[dict[str, Any]] | None = None,
    default: tuple[int, int] = (0, 2),
    input_info: dict[str, Any] | None = None,
) -> ModuleType:
    """把 sounddevice 换成假模块。

    :param devices: ``query_devices()`` 返回的设备表。
    :param default: ``default.device`` 的输入与输出索引。
    :param input_info: ``query_devices(index, "input")`` 返回的单个设备。
    :returns: 假模块。
    """
    fake = ModuleType("sounddevice")
    fake.devices = list(devices or [])  # type: ignore[attr-defined]
    fake.default = SimpleNamespace(device=default)  # type: ignore[attr-defined]
    fake.created = {}  # type: ignore[attr-defined]

    def query_devices(device: Any = None, kind: Any = None) -> Any:
        """按查询方式返回设备表或单个设备。"""
        if device is None and kind is None:
            return list(fake.devices)  # type: ignore[attr-defined]
        if input_info is not None:
            return dict(input_info)
        if isinstance(device, int):
            return dict(fake.devices[device])  # type: ignore[attr-defined]
        return dict(fake.devices[default[0]])  # type: ignore[attr-defined]

    def input_stream(**kwargs: Any) -> _FakeInputStream:
        """记下开流参数，返回假流。"""
        fake.created.update(kwargs)  # type: ignore[attr-defined]
        return _FakeInputStream(**kwargs)

    fake.query_devices = query_devices  # type: ignore[attr-defined]
    fake.InputStream = input_stream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def _install_fake_soundcard(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loopbacks: list[_FakeLoopbackDevice] | None = None,
    real_microphones: list[_FakeLoopbackDevice] | None = None,
    default_speaker_id: str | None = None,
) -> ModuleType:
    """把 soundcard 换成假模块。

    :param loopbacks: ``all_microphones(include_loopback=True)`` 里的回环设备。
    :param real_microphones: 同一个调用里混进来的真实麦克风，后端应当跳过。
    :param default_speaker_id: 默认扬声器端点的 ID。
    :returns: 假模块。
    """
    fake = ModuleType("soundcard")
    fake.loopbacks = list(loopbacks or [])  # type: ignore[attr-defined]
    fake.real_microphones = list(real_microphones or [])  # type: ignore[attr-defined]
    fake.default_speaker_id = default_speaker_id  # type: ignore[attr-defined]

    def all_microphones(include_loopback: bool = False) -> list[Any]:
        """按 include_loopback 返回回环设备与真实麦克风。"""
        if not include_loopback:
            return list(fake.real_microphones)  # type: ignore[attr-defined]
        return list(fake.loopbacks) + list(fake.real_microphones)  # type: ignore[attr-defined]

    def default_speaker() -> SimpleNamespace:
        """返回默认扬声器端点，只需要 ID。"""
        return SimpleNamespace(id=fake.default_speaker_id)  # type: ignore[attr-defined]

    def get_microphone(device_id: str, include_loopback: bool = False) -> Any:
        """按 WASAPI ID 找设备。"""
        for device in all_microphones(include_loopback):
            if device.id == device_id:
                return device
        raise RuntimeError(f"找不到设备：{device_id}")

    fake.all_microphones = all_microphones  # type: ignore[attr-defined]
    fake.default_speaker = default_speaker  # type: ignore[attr-defined]
    fake.get_microphone = get_microphone  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "soundcard", fake)
    return fake


def _sounddevice_table(
    *,
    microphone_rate: float = 48000.0,
    microphone_channels: int = 2,
) -> list[dict[str, Any]]:
    """造一张 Windows 风格的 sounddevice 设备表。"""
    return [
        {
            "name": MICROPHONE_NAME,
            "index": 0,
            "max_input_channels": microphone_channels,
            "max_output_channels": 0,
            "default_samplerate": microphone_rate,
        },
        {
            "name": USB_MICROPHONE_NAME,
            "index": 1,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
        },
        {
            "name": SPEAKER_NAME,
            "index": 2,
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
        },
    ]


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
def test_list_devices_separates_microphones_and_loopbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """麦克风走 sounddevice，回环走 soundcard，默认设备各标一个。"""
    _install_fake_sounddevice(monkeypatch, devices=_sounddevice_table())
    real_microphone = _FakeLoopbackDevice(
        "mic-usb", USB_MICROPHONE_NAME, isloopback=False
    )
    loopbacks = [
        _FakeLoopbackDevice("spk-hdmi", HDMI_SPEAKER_NAME),
        _FakeLoopbackDevice("spk-default", SPEAKER_NAME),
    ]
    _install_fake_soundcard(
        monkeypatch,
        loopbacks=loopbacks,
        real_microphones=[real_microphone],
        default_speaker_id="spk-default",
    )

    devices = WasapiAudioBackend().list_devices()

    microphones = [device for device in devices if not device.is_loopback]
    detected = [device for device in devices if device.is_loopback]
    assert [device.index for device in microphones] == [0, 1]
    assert microphones[0].is_default is True
    assert microphones[1].is_default is False

    # soundcard 混在同一个列表里的真实麦克风不重复进设备表
    assert [device.index for device in detected] == [
        LOOPBACK_INDEX_BASE,
        LOOPBACK_INDEX_BASE + 1,
    ]
    assert all(device.is_input for device in detected)
    default_loopback = next(device for device in detected if device.is_default)
    assert default_loopback.name == f"{SPEAKER_NAME}（回环）"
    assert default_loopback.channels == 2
    assert default_loopback.default_samplerate == 48000.0


def test_list_devices_skips_output_only_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有输入声道的设备不进设备表。"""
    _install_fake_sounddevice(
        monkeypatch,
        devices=[
            {
                "name": SPEAKER_NAME,
                "index": 2,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            }
        ],
    )
    _install_fake_soundcard(monkeypatch, loopbacks=[])
    assert WasapiAudioBackend().list_devices() == []


def test_list_devices_without_soundcard_keeps_microphones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """soundcard 不可用时只少了回环，麦克风照常枚举。"""
    _install_fake_sounddevice(monkeypatch, devices=_sounddevice_table())
    monkeypatch.setitem(sys.modules, "soundcard", None)

    devices = WasapiAudioBackend().list_devices()

    assert [device.index for device in devices] == [0, 1]
    assert all(not device.is_loopback for device in devices)


def test_list_devices_without_sounddevice_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sounddevice 不可导入时报 E_AUDIO_DEVICE。"""
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(AudioDeviceError) as excinfo:
        WasapiAudioBackend().list_devices()
    assert excinfo.value.code == "E_AUDIO_DEVICE"


# --------------------------------------------------------------------------
# 麦克风采集
# --------------------------------------------------------------------------
def _patch_microphone_stream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    native_rate: float,
    native_channels: int,
) -> dict[str, Any]:
    """把 sounddevice 的设备查询与 InputStream 换成假的，记下开流参数。"""
    info = {
        "name": MICROPHONE_NAME,
        "index": 0,
        "max_input_channels": native_channels,
        "max_output_channels": 0,
        "default_samplerate": native_rate,
    }
    fake = _install_fake_sounddevice(
        monkeypatch,
        devices=_sounddevice_table(),
        input_info=info,
    )
    return fake.created  # type: ignore[attr-defined]


def test_start_stream_resamples_native_rate_to_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设备原生 48000 Hz 时，按 200 毫秒开流并重采样成 16000 Hz 单声道。"""
    created = _patch_microphone_stream(
        monkeypatch, native_rate=48000.0, native_channels=2
    )
    blocks: list[Any] = []
    stream = WasapiAudioBackend().start_stream(
        device=0,
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
    created = _patch_microphone_stream(
        monkeypatch, native_rate=48000.0, native_channels=2
    )
    blocks: list[Any] = []
    stream = WasapiAudioBackend().start_stream(
        device=0,
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
    created = _patch_microphone_stream(
        monkeypatch, native_rate=48000.0, native_channels=128
    )
    stream = WasapiAudioBackend().start_stream(
        device=0,
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
    created = _patch_microphone_stream(
        monkeypatch, native_rate=44100.0, native_channels=1
    )
    blocks: list[Any] = []
    stream = WasapiAudioBackend().start_stream(
        device=1,
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


def test_start_stream_wraps_microphone_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """麦克风开流失败转成 AudioDeviceError。"""
    fake = _install_fake_sounddevice(
        monkeypatch,
        devices=_sounddevice_table(),
        input_info=_sounddevice_table()[0],
    )

    def fake_input_stream(**kwargs: Any) -> Any:
        raise RuntimeError("设备被占用")

    fake.InputStream = fake_input_stream  # type: ignore[attr-defined]
    with pytest.raises(AudioDeviceError) as excinfo:
        WasapiAudioBackend().start_stream(
            device=0,
            sample_rate=16000,
            channels=1,
            block_duration_ms=200,
            on_block=lambda samples: None,
        )
    assert excinfo.value.code == "E_AUDIO_DEVICE"


# --------------------------------------------------------------------------
# WASAPI loopback 采集
# --------------------------------------------------------------------------
def _install_loopback(
    monkeypatch: pytest.MonkeyPatch,
    *,
    channels: int = 2,
    frames: Any = None,
    open_error: BaseException | None = None,
) -> tuple[WasapiAudioBackend, AudioDevice, _FakeLoopbackDevice]:
    """装好麦克风与一路回环，返回后端、回环设备行与假设备。"""
    _install_fake_sounddevice(monkeypatch, devices=_sounddevice_table())
    loopback = _FakeLoopbackDevice(
        "spk-default",
        SPEAKER_NAME,
        channels=channels,
        frames=frames,
        open_error=open_error,
    )
    _install_fake_soundcard(
        monkeypatch,
        loopbacks=[loopback],
        default_speaker_id="spk-default",
    )
    backend = WasapiAudioBackend()
    device = next(device for device in backend.list_devices() if device.is_loopback)
    return backend, device, loopback


def test_loopback_stream_reads_native_rate_and_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回环按 48000 Hz 立体声读块，下混后重采样成 3200 点。"""

    def stereo_one_side(frames: int, channels: int) -> Any:
        """只有左声道有信号，下混后是 0.5。"""
        data = np.zeros((frames, channels), dtype="float32")
        data[:, 0] = 1.0
        return data

    backend, device, loopback = _install_loopback(monkeypatch, frames=stereo_one_side)
    blocks: list[Any] = []
    done = threading.Event()

    def on_block(samples: Any) -> None:
        """收块并置位。"""
        blocks.append(samples)
        done.set()

    stream = backend.start_stream(
        device=device.index,
        sample_rate=16000,
        channels=1,
        block_duration_ms=200,
        on_block=on_block,
    )
    try:
        assert done.wait(2.0), "回环一路没有出块"
        assert loopback.recorder_calls == [
            {"samplerate": 48000.0, "channels": 2, "blocksize": 9600}
        ]
        assert blocks[0].shape == (3200,)
        assert blocks[0].dtype == np.float32
        assert float(blocks[0].mean()) == pytest.approx(0.5, abs=1e-6)
        assert stream.is_active is True
    finally:
        stream.stop()
    assert stream.is_active is False


def test_loopback_stream_caps_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回环端点报 8 声道时只按立体声开流。"""
    backend, device, loopback = _install_loopback(monkeypatch, channels=8)
    stream = backend.start_stream(
        device=device.index,
        sample_rate=16000,
        channels=1,
        block_duration_ms=200,
        on_block=lambda samples: None,
    )
    try:
        assert loopback.recorder_calls == [
            {"samplerate": 48000.0, "channels": 2, "blocksize": 9600}
        ]
    finally:
        stream.stop()


def test_loopback_stream_wraps_open_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """recorder 打不开时转成 AudioDeviceError。"""
    backend, device, _loopback = _install_loopback(
        monkeypatch, open_error=RuntimeError("端点被占用")
    )
    with pytest.raises(AudioDeviceError) as excinfo:
        backend.start_stream(
            device=device.index,
            sample_rate=16000,
            channels=1,
            block_duration_ms=200,
            on_block=lambda samples: None,
        )
    assert excinfo.value.code == "E_AUDIO_DEVICE"


def test_start_stream_rejects_unknown_loopback_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回环编号找不到设备时报 E_AUDIO_DEVICE。"""
    _install_fake_sounddevice(monkeypatch, devices=_sounddevice_table())
    _install_fake_soundcard(monkeypatch, loopbacks=[])
    with pytest.raises(AudioDeviceError) as excinfo:
        WasapiAudioBackend().start_stream(
            device=LOOPBACK_INDEX_BASE,
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
        _device(1, MICROPHONE_NAME),
        _device(2, "默认麦克风", is_default=True),
        _device(3, f"{HDMI_SPEAKER_NAME}（回环）", loopback=True),
        _device(4, f"{SPEAKER_NAME}（回环）", loopback=True, is_default=True),
    ]
    selection = select_source("auto", devices)
    assert selection.source == SOURCE_MIXED
    assert [device.index for device in selection.devices] == [2, 4]


def test_select_source_auto_only_loopback() -> None:
    """只有回环时 auto 选中默认回环。"""
    devices = [
        _device(2, f"{HDMI_SPEAKER_NAME}（回环）", loopback=True),
        _device(3, f"{SPEAKER_NAME}（回环）", loopback=True, is_default=True),
    ]
    selection = select_source("auto", devices)
    assert selection.source == SOURCE_SYSTEM_MONITOR
    assert selection.device.index == 3


def test_select_source_auto_falls_back_to_microphone() -> None:
    """只有麦克风时 auto 选中默认麦克风。"""
    devices = [
        _device(1, MICROPHONE_NAME),
        _device(2, "默认麦克风", is_default=True),
    ]
    selection = select_source("auto", devices)
    assert selection.source == SOURCE_MICROPHONE
    assert selection.device.index == 2


def test_select_source_microphone_prefers_default_device() -> None:
    """麦克风里优先默认设备，不看列表顺序。"""
    devices = [
        _device(1, MICROPHONE_NAME),
        _device(2, "默认麦克风", is_default=True),
        _device(3, f"{SPEAKER_NAME}（回环）", loopback=True),
    ]
    selection = select_source(SOURCE_MICROPHONE, devices)
    assert selection.source == SOURCE_MICROPHONE
    assert selection.device.index == 2


def test_select_source_system_monitor_prefers_default_device() -> None:
    """回环里优先默认设备，不看列表顺序。"""
    devices = [
        _device(1, MICROPHONE_NAME),
        _device(2, f"{HDMI_SPEAKER_NAME}（回环）", loopback=True),
        _device(3, f"{SPEAKER_NAME}（回环）", loopback=True, is_default=True),
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
        select_source("microphone", [_device(2, f"{SPEAKER_NAME}（回环）", loopback=True)])
    assert excinfo.value.code == "E_AUDIO_DEVICE"


def test_select_source_mixed_needs_both_devices() -> None:
    """mixed 两路都要有，缺一路报错。"""
    with pytest.raises(AudioDeviceError):
        select_source("mixed", [_device(1, MICROPHONE_NAME)])
    selection = select_source(
        "mixed",
        [
            _device(1, MICROPHONE_NAME),
            _device(2, f"{SPEAKER_NAME}（回环）", loopback=True),
        ],
    )
    assert selection.source == SOURCE_MIXED
    assert [device.index for device in selection.devices] == [1, 2]


def test_select_source_rejects_unknown_source() -> None:
    """来源名不合法时报错。"""
    with pytest.raises(AudioDeviceError):
        select_source("speaker", [_device(1, MICROPHONE_NAME)])


def test_split_devices_separates_microphone_and_loopback() -> None:
    """设备表分得开麦克风与回环。"""
    microphones, loopbacks = split_devices(
        [
            _device(1, MICROPHONE_NAME),
            _device(2, f"{SPEAKER_NAME}（回环）", loopback=True),
            _device(3, SPEAKER_NAME, is_input=False),
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
        microphone=_device(1, MICROPHONE_NAME),
        monitor=_device(2, f"{SPEAKER_NAME}（回环）", loopback=True),
    )
    await capture.start()
    backend.emit(1, np.zeros(3200, dtype="float32"))
    backend.emit(2, np.ones(3200, dtype="float32"))

    mic_block = await asyncio.wait_for(capture.get_microphone_block(), timeout=2.0)
    loopback_block = await asyncio.wait_for(
        capture.get_system_monitor_block(), timeout=2.0
    )
    assert mic_block.source == SOURCE_MICROPHONE
    assert loopback_block.source == SOURCE_SYSTEM_MONITOR
    assert float(mic_block.samples.max()) == 0.0
    assert float(loopback_block.samples.max()) == 1.0
    await capture.stop()
    assert capture.is_running is False


async def test_mixed_capture_produces_one_mixed_stream() -> None:
    """混成一路时只有一条流，标签是 mixed。"""
    backend = _FakeAudioBackend()
    capture = MixedCapture(
        backend,
        microphone=_device(1, MICROPHONE_NAME),
        monitor=_device(2, f"{SPEAKER_NAME}（回环）", loopback=True),
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
        microphone=_device(1, MICROPHONE_NAME),
        monitor=_device(2, f"{SPEAKER_NAME}（回环）", loopback=True),
    )
    with pytest.raises(AudioDeviceError):
        await capture.start()
    assert capture.microphone.is_running is False


def test_create_capture_builds_capture_per_source() -> None:
    """来源选择的结果对应到对应的采集器。"""
    backend = _FakeAudioBackend()
    microphone = _device(1, MICROPHONE_NAME)
    loopback = _device(2, f"{SPEAKER_NAME}（回环）", loopback=True)

    single = create_capture(
        backend,
        select_source("microphone", [microphone, loopback]),
    )
    assert isinstance(single, AudioCapture)
    assert single.source == SOURCE_MICROPHONE
    assert single.device == 1

    mixed = create_capture(
        backend,
        select_source("mixed", [microphone, loopback]),
    )
    assert isinstance(mixed, MixedCapture)