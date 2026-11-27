"""平台门面的测试。

覆盖《9-平台差异与Windows适配》「目录结构」与「实施顺序」一节立起来的门面：

- :func:`app.platform.get_platform` 在本机返回 Linux 实现，四个能力后端与
  传输层都在；
- :class:`~app.platform.types.AudioDevice` 与 :class:`~app.platform.types.Screenshot`
  两个数据结构的字段行为；
- Windows 侧的占位后端可导入，方法体抛 :class:`NotImplementedError`；
  进程存活探测已实现，见 :mod:`tests.test_windows_process`；
- Linux 侧 AF_UNIX socket 传输层的一次完整往返。

不真连音频设备，也不连 D-Bus：音频侧用假设备表与假 ``sounddevice`` 模块
属性替身，Windows 侧直接构造占位类，不切换 ``sys.platform``。
"""

from __future__ import annotations

import asyncio
import shutil
import stat
import subprocess
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from app.platform import get_platform
from app.platform.linux.audio import SoundDeviceAudioBackend
from app.platform.linux.compiler import GppCompiler
from app.platform.linux.process import LinuxProcessBackend
from app.platform.linux.screenshot import GnomeScreenshotBackend
from app.platform.linux.transport import UnixSocketTransport
from app.platform.types import AudioDevice, Platform, Screenshot
from app.platform.windows import build_platform as build_windows_platform
from app.platform.windows.audio import WasapiAudioBackend
from app.platform.windows.compiler import MsvcCompiler
from app.platform.windows.process import PsutilProcessBackend
from app.platform.windows.screenshot import MssScreenshotBackend
from app.platform.windows.transport import TcpLoopbackTransport

# ----------------------------------------------------------------------
# 门面
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="只在 Linux 上断言 Linux 平台实现",
)
def test_get_platform_returns_linux_implementation() -> None:
    """本机拿到的是 Linux 实现，四个能力后端与传输层都在。"""
    platform = get_platform()
    assert platform.name == "linux"
    assert platform.screenshot is not None
    assert platform.audio is not None
    assert platform.process is not None
    assert platform.compiler is not None
    assert platform.transport is not None
    assert isinstance(platform, Platform)


def test_get_platform_returns_stable_object() -> None:
    """同一进程反复调用拿到的是同一个对象。"""
    assert get_platform() is get_platform()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="只在 Linux 上断言 Linux 平台实现",
)
def test_linux_platform_transport_is_unix_socket(tmp_path: Path) -> None:
    """Linux 的传输层是 AF_UNIX socket，端点路径就是传入的路径。"""
    endpoint = tmp_path / "interviewmate.sock"
    transport = get_platform().transport(endpoint)
    assert isinstance(transport, UnixSocketTransport)
    assert transport.endpoint_path == endpoint


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------


def test_audio_device_fields() -> None:
    """AudioDevice 的字段按传入值取，取值不变。"""
    device = AudioDevice(
        name="系统回环",
        index=4,
        is_input=True,
        is_loopback=True,
        default_samplerate=48000.0,
        channels=2,
    )
    assert device.name == "系统回环"
    assert device.index == 4
    assert device.is_input is True
    assert device.is_loopback is True
    assert device.default_samplerate == 48000.0
    assert device.channels == 2
    assert device.is_default is False


def test_audio_device_is_frozen() -> None:
    """AudioDevice 是冻结的，字段改不了。"""
    device = AudioDevice(
        name="麦克风",
        index=0,
        is_input=True,
        is_loopback=False,
        default_samplerate=16000.0,
        channels=1,
    )
    with pytest.raises(FrozenInstanceError):
        device.name = "别的设备"  # type: ignore[misc]


def test_screenshot_fields() -> None:
    """Screenshot 存的是图片字节与像素尺寸。"""
    shot = Screenshot(data=b"\x89PNG", width=1920, height=1080)
    assert shot.data == b"\x89PNG"
    assert shot.width == 1920
    assert shot.height == 1080
    assert shot == Screenshot(data=b"\x89PNG", width=1920, height=1080)


def test_screenshot_is_frozen() -> None:
    """Screenshot 是冻结的，字段改不了。"""
    shot = Screenshot(data=b"\x89PNG", width=1920, height=1080)
    with pytest.raises(FrozenInstanceError):
        shot.width = 800  # type: ignore[misc]


# ----------------------------------------------------------------------
# Windows 侧占位后端
# ----------------------------------------------------------------------


def test_windows_platform_exposes_all_capabilities() -> None:
    """Windows 平台的门面与 Linux 侧同构。"""
    platform = build_windows_platform()
    assert platform.name == "win32"
    assert isinstance(platform.screenshot, MssScreenshotBackend)
    assert isinstance(platform.audio, WasapiAudioBackend)
    assert isinstance(platform.process, PsutilProcessBackend)
    assert isinstance(platform.compiler, MsvcCompiler)
    assert isinstance(platform, Platform)


async def test_windows_screenshot_backend_is_placeholder() -> None:
    """Windows 截图后端的两个方法都还没实现。"""
    backend = MssScreenshotBackend()
    assert backend.name == "mss"
    assert backend.required_packages == ("mss",)
    with pytest.raises(NotImplementedError):
        await backend.probe()
    with pytest.raises(NotImplementedError):
        await backend.capture()


def test_windows_audio_backend_is_placeholder() -> None:
    """Windows 音频后端的两个方法都还没实现。"""
    backend = WasapiAudioBackend()
    assert backend.required_packages == ("sounddevice", "soundcard", "numpy")
    with pytest.raises(NotImplementedError):
        backend.list_devices()
    with pytest.raises(NotImplementedError):
        backend.record(sample_rate=16000, channels=1, seconds=1.0)


def test_windows_process_backend_detects_alive_process() -> None:
    """Windows 进程存活探测已实现：非法 PID 一律判死，不抛异常。"""
    backend = PsutilProcessBackend()
    assert backend.name == "psutil"
    assert backend.required_packages == ("psutil",)
    assert backend.is_alive(0) is False
    assert backend.is_alive(-1) is False


def test_windows_transport_endpoints(tmp_path: Path) -> None:
    """Windows 传输层的端口文件与 token 文件路径按约定派生。"""
    endpoint = tmp_path / "interviewmate.port"
    transport = TcpLoopbackTransport(endpoint)
    assert transport.endpoint_path == endpoint
    assert transport.token_path == tmp_path / "interviewmate.token"
    assert transport.required_packages == ()


def test_windows_compiler_command() -> None:
    """Windows 侧的编译命令已经定下来。"""
    assert MsvcCompiler().command() == "cl.exe"


# ----------------------------------------------------------------------
# Linux 侧后端
# ----------------------------------------------------------------------


def test_linux_backends_expose_capability_names() -> None:
    """Linux 侧四个能力后端的名字与依赖包。"""
    assert GnomeScreenshotBackend().name == "gnome"
    assert GnomeScreenshotBackend().required_packages == ("dbus_next",)
    assert SoundDeviceAudioBackend().required_packages == ("sounddevice",)
    assert LinuxProcessBackend().required_packages == ()
    assert GppCompiler().command() == "g++"


def test_linux_process_backend_detects_alive_process() -> None:
    """当前进程活着，PID 小于等于 0 一律算不活着。"""
    backend = LinuxProcessBackend()
    assert backend.is_alive(1) is True
    assert backend.is_alive(0) is False
    assert backend.is_alive(-1) is False


def test_linux_audio_lists_only_input_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """设备表只留输入声道数大于 0 的设备，字段按 sounddevice 给的值映射。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(
        sd,
        "query_devices",
        lambda: [
            {
                "name": "系统回环",
                "index": 2,
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
    assert len(devices) == 1
    device = devices[0]
    assert device.name == "系统回环"
    assert device.index == 2
    assert device.is_input is True
    assert device.is_loopback is False
    assert device.default_samplerate == 48000.0
    assert device.channels == 2


def test_linux_audio_marks_system_default_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wpctl 不可用时按 sounddevice 的默认设备标 is_default。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(shutil, "which", lambda command: None)
    monkeypatch.setattr(
        sd,
        "query_devices",
        lambda: [
            {
                "name": "麦克风 A",
                "index": 11,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "麦克风 B",
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
            {
                "name": "扬声器.monitor",
                "index": 20,
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
            {
                "name": "耳机.monitor",
                "index": 21,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
        ],
    )
    # 默认输入是索引 13，默认输出是索引 3（设备名“扬声器”）
    monkeypatch.setattr(sd, "default", types.SimpleNamespace(device=(13, 3)))

    devices = SoundDeviceAudioBackend().list_devices()
    by_index = {device.index: device for device in devices}
    assert by_index[13].is_default is True
    assert by_index[11].is_default is False
    assert by_index[20].is_default is True
    assert by_index[21].is_default is False
    # 默认输出设备本身不是输入设备，不进设备表
    assert 3 not in by_index


def test_linux_audio_default_probe_failure_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wpctl 与 sounddevice 都拿不到默认设备时 is_default 全 False，且不抛。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(shutil, "which", lambda command: None)
    monkeypatch.setattr(
        sd,
        "query_devices",
        lambda: [
            {
                "name": "麦克风",
                "index": 13,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "扬声器.monitor",
                "index": 20,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
        ],
    )

    class _BrokenDefault:
        """取 device 就抛异常的替身。"""

        @property
        def device(self) -> tuple[int, int]:
            raise RuntimeError("拿不到默认设备")

    monkeypatch.setattr(sd, "default", _BrokenDefault())

    devices = SoundDeviceAudioBackend().list_devices()
    assert [device.index for device in devices] == [13, 20]
    assert all(device.is_default is False for device in devices)


# ----------------------------------------------------------------------
# 默认设备解析：wpctl 优先，sounddevice 兜底
# ----------------------------------------------------------------------
_SINK_NODE_NAME: str = "alsa_output.usb-dac.analog-stereo"
_SOURCE_NODE_NAME: str = "alsa_input.usb-dac.analog-stereo"


def _fake_wpctl_devices() -> list[dict[str, Any]]:
    """假设备表：内置声卡回环与 USB 声卡回环各一个，加两台输入设备。

    ``default`` 与 ``pipewire`` 是按索引取默认设备时会命中的别名设备，
    通道数故意写得很大，和真机上的 PortAudio 别名一致。
    """
    return [
        {
            "name": "default",
            "index": 8,
            "max_input_channels": 128,
            "max_output_channels": 128,
            "default_samplerate": 44100.0,
        },
        {
            "name": "alsa_output.pci-analog-stereo.monitor",
            "index": 14,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
        {
            "name": "alsa_input.pci-analog-stereo",
            "index": 15,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
        {
            "name": _SINK_NODE_NAME + ".monitor",
            "index": 16,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
        {
            "name": _SOURCE_NODE_NAME,
            "index": 17,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
    ]


def _install_fake_wpctl(
    monkeypatch: pytest.MonkeyPatch,
    outputs: dict[str, str],
    *,
    returncode: int = 0,
) -> list[tuple[str, ...]]:
    """装一个假的 ``wpctl``：``which`` 认这个命令，``run`` 按目标回固定输出。

    :param outputs: 目标（``@DEFAULT_AUDIO_SINK@`` 等）到 stdout 的映射。
    :param returncode: 假命令的退出码。
    :returns: 收到的命令行列表，按调用顺序。
    """
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: "/usr/bin/wpctl" if command == "wpctl" else None,
    )

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """记下命令行，按目标返回预设输出。"""
        calls.append(tuple(command))
        assert kwargs.get("timeout") == 2.0
        return subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=outputs.get(command[-1], ""),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_linux_audio_marks_default_devices_via_wpctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wpctl 给出默认 sink 与 source 的节点名时，按名字标默认输入与默认回环。

    ``sd.default.device`` 故意指向别名设备，证明 wpctl 的结果优先。
    """
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(sd, "query_devices", _fake_wpctl_devices)
    monkeypatch.setattr(sd, "default", types.SimpleNamespace(device=(8, 8)))
    calls = _install_fake_wpctl(
        monkeypatch,
        {
            "@DEFAULT_AUDIO_SINK@": (
                "id 55, type PipeWire:Interface:Node\n"
                '    * node.name = "' + _SINK_NODE_NAME + '"\n'
                "    node.description = \"USB DAC\"\n"
            ),
            "@DEFAULT_AUDIO_SOURCE@": (
                "id 56, type PipeWire:Interface:Node\n"
                '  * node.name = "' + _SOURCE_NODE_NAME + '"\n'
            ),
        },
    )

    devices = SoundDeviceAudioBackend().list_devices()
    by_index = {device.index: device for device in devices}
    marked = sorted(index for index, device in by_index.items() if device.is_default)

    assert marked == [16, 17]
    # 问的是 PipeWire 的两个默认目标，每个一次
    assert [command[-1] for command in calls] == [
        "@DEFAULT_AUDIO_SOURCE@",
        "@DEFAULT_AUDIO_SINK@",
    ]


def test_linux_audio_wpctl_names_not_matched_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wpctl 给的节点名在设备表里匹配不到时，退回 sd.default 的索引方案。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(sd, "query_devices", _fake_wpctl_devices)
    monkeypatch.setattr(sd, "default", types.SimpleNamespace(device=(17, 9)))
    _install_fake_wpctl(
        monkeypatch,
        {
            # 这两个名字都不在假设备表里
            "@DEFAULT_AUDIO_SINK@": '  * node.name = "alsa_output.hdmi-stereo"\n',
            "@DEFAULT_AUDIO_SOURCE@": '  * node.name = "alsa_input.hdmi-stereo"\n',
        },
    )

    devices = SoundDeviceAudioBackend().list_devices()
    by_index = {device.index: device for device in devices}
    marked = sorted(index for index, device in by_index.items() if device.is_default)

    # 默认输出索引 9 不在设备表里，只剩默认输入索引 17
    assert marked == [17]


def test_linux_audio_wpctl_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wpctl 超时、退出码非零或输出里没有节点名时，都退回 sd.default 方案。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(sd, "query_devices", _fake_wpctl_devices)
    monkeypatch.setattr(sd, "default", types.SimpleNamespace(device=(15, 10)))
    _install_fake_wpctl(
        monkeypatch,
        {
            "@DEFAULT_AUDIO_SINK@": '  * node.name = "' + _SINK_NODE_NAME + '"\n',
            "@DEFAULT_AUDIO_SOURCE@": '  * node.name = "' + _SOURCE_NODE_NAME + '"\n',
        },
        returncode=1,
    )

    devices = SoundDeviceAudioBackend().list_devices()
    by_index = {device.index: device for device in devices}
    marked = sorted(index for index, device in by_index.items() if device.is_default)

    # 退出码非零时当作没问出来，设备 10 不在表里，只剩默认输入 15
    assert marked == [15]


def test_linux_audio_wpctl_timeout_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wpctl 超时按拿不到处理，退回 sd.default 方案，不往外抛。"""
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(sd, "query_devices", _fake_wpctl_devices)
    monkeypatch.setattr(sd, "default", types.SimpleNamespace(device=(15, 10)))
    monkeypatch.setattr(shutil, "which", lambda command: "/usr/bin/wpctl")

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        """直接超时。"""
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)

    devices = SoundDeviceAudioBackend().list_devices()
    by_index = {device.index: device for device in devices}
    marked = sorted(index for index, device in by_index.items() if device.is_default)

    assert marked == [15]


def test_linux_audio_record_uses_sounddevice_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """采集走 sounddevice 模块的属性，返回 float32 数组。"""
    np = pytest.importorskip("numpy")
    sd = pytest.importorskip("sounddevice")
    seen: dict[str, object] = {}

    def fake_rec(frames: int, **kwargs: object) -> object:
        seen["frames"] = frames
        seen.update(kwargs)
        return np.zeros((frames, int(kwargs["channels"])), dtype="float32")

    monkeypatch.setattr(sd, "rec", fake_rec)
    monkeypatch.setattr(sd, "wait", lambda *args, **kwargs: None)

    samples = SoundDeviceAudioBackend().record(
        sample_rate=16000, channels=1, seconds=0.5
    )
    assert seen["frames"] == 8000
    assert seen["samplerate"] == 16000
    assert seen["channels"] == 1
    assert seen["dtype"] == "float32"
    assert samples.shape == (8000, 1)
    assert samples.dtype == np.float32


async def test_linux_screenshot_probe_reports_missing_dbus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dbus-next 不可导入时抛 ScreenshotError，文案与既有实现一致。"""
    from app.core.exceptions import ScreenshotError

    # sys.modules 里放 None，导入必然失败，结果不依赖本机有没有 D-Bus 会话总线
    monkeypatch.setitem(sys.modules, "dbus_next", None)
    with pytest.raises(ScreenshotError) as excinfo:
        await GnomeScreenshotBackend().probe(timeout=0.1)
    assert "dbus-next 不可导入" in excinfo.value.message


async def test_linux_screenshot_capture_reports_bus_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """连不上 D-Bus 会话总线时截图抛 ScreenshotError。"""
    from app.core.exceptions import ScreenshotError

    backend = GnomeScreenshotBackend()

    async def _fail() -> None:
        raise ScreenshotError("连不上 D-Bus 会话总线")

    monkeypatch.setattr(backend, "_open_bus", _fail)
    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()
    assert "连不上 D-Bus 会话总线" in excinfo.value.message


# ----------------------------------------------------------------------
# Linux 侧传输层
# ----------------------------------------------------------------------


async def test_unix_socket_transport_roundtrip(tmp_path: Path) -> None:
    """起服务、连上、收发一行、停服务，socket 文件最后被删掉。"""
    endpoint = tmp_path / "interviewmate.sock"
    transport = UnixSocketTransport(endpoint)
    received: list[bytes] = []

    async def on_connect(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                # 占用探测的连接连上就断开，没有请求可回
                return
            received.append(line)
            writer.write(b"pong\n")
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            # 对端提前断开，忽略
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    assert transport.prepare_endpoint() is True
    await transport.start(on_connect)
    try:
        assert endpoint.exists()
        # 监听起来之后端点算被活进程占用
        assert transport.is_endpoint_in_use() is True
        # 权限收紧到只有属主可读写
        mode = stat.S_IMODE(endpoint.stat().st_mode)
        assert mode == 0o600

        # 客户端是同步 socket，阻塞在 recv 上时事件循环要还能跑，
        # 所以放到线程里做，和命令行走同步客户端的情形一样
        def exchange() -> bytes:
            """连上去发一行、读一行。"""
            client = transport.connect(timeout=2.0)
            try:
                client.sendall(b"ping\n")
                return client.recv(4096)
            finally:
                client.close()

        data = await asyncio.to_thread(exchange)

        assert data == b"pong\n"
        assert received == [b"ping\n"]
    finally:
        await transport.stop()

    assert not endpoint.exists()


def test_unix_socket_transport_prepare_endpoint_removes_stale_file(
    tmp_path: Path,
) -> None:
    """端点文件存在但连不上时是残留文件，删掉后可以重建。"""
    endpoint = tmp_path / "interviewmate.sock"
    endpoint.write_text("残留", encoding="utf-8")
    transport = UnixSocketTransport(endpoint)
    assert transport.is_endpoint_in_use() is False
    assert transport.prepare_endpoint() is True
    assert not endpoint.exists()


def test_unix_socket_transport_cleanup_is_idempotent(tmp_path: Path) -> None:
    """端点文件不存在时清理也不报错。"""
    endpoint = tmp_path / "interviewmate.sock"
    transport = UnixSocketTransport(endpoint)
    transport.cleanup_endpoint()
    endpoint.write_text("残留", encoding="utf-8")
    transport.cleanup_endpoint()
    assert not endpoint.exists()
