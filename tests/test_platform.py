"""平台门面的测试（Windows-only）。

覆盖《2-骨架与配置》目录结构约定一节立起来的门面：

- :func:`app.platform.get_platform` 在本机返回 Windows 实现，五个能力后端与
  传输层都在；非 Windows 平台抛 :class:`NotImplementedError`；
- :class:`~app.platform.types.AudioDevice` 与 :class:`~app.platform.types.Screenshot`
  两个数据结构的字段行为；
- Windows 侧后端按当前实现（``mss`` / WASAPI / ``psutil`` / ``cl.exe``）暴露
  名字与依赖包，进程存活探测可用；
- Windows 侧 TCP loopback 传输层的一次完整往返；
- 客户端 :func:`app.runtime.socket_client.send_request` 在「连不上」与
  「token 不匹配」时抛 ``SocketError``，错误码 ``E_SOCKET``。

不真抓屏、不真开音频设备：传输层用本机回环端口，音频与截图后端只做能力声明。
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.core.exceptions import SocketError
from app.platform import get_platform
from app.platform.types import AudioDevice, Platform, Screenshot
from app.platform.windows import build_platform as build_windows_platform
from app.platform.windows.audio import WasapiAudioBackend
from app.platform.windows.compiler import MsvcCompiler
from app.platform.windows.process import PsutilProcessBackend
from app.platform.windows.screenshot import MssScreenshotBackend
from app.platform.windows.transport import TcpLoopbackTransport
from app.runtime.socket_client import send_request
from app.runtime.socket_server import SocketServer

# ----------------------------------------------------------------------
# 门面
# ----------------------------------------------------------------------


def test_get_platform_returns_windows_implementation() -> None:
    """本机拿到的是 Windows 实现，五个能力后端与传输层都在。"""
    platform = get_platform()
    assert platform.name == "win32"
    assert isinstance(platform, Platform)
    assert isinstance(platform.screenshot, MssScreenshotBackend)
    assert isinstance(platform.audio, WasapiAudioBackend)
    assert isinstance(platform.process, PsutilProcessBackend)
    assert isinstance(platform.compiler, MsvcCompiler)
    endpoint = Path("runtime/interviewmate.port")
    assert isinstance(platform.transport(endpoint), TcpLoopbackTransport)


def test_get_platform_returns_stable_object() -> None:
    """同一进程反复调用拿到的是同一个对象。"""
    assert get_platform() is get_platform()


def test_get_platform_rejects_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非 Windows 平台抛 NotImplementedError，消息里点明当前平台。"""
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(NotImplementedError) as excinfo:
        get_platform()
    assert "darwin" in str(excinfo.value)
    assert "Windows" in str(excinfo.value)


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
# Windows 侧后端
# ----------------------------------------------------------------------


def test_windows_platform_exposes_all_capabilities() -> None:
    """Windows 平台门面把五个能力后端都装好。"""
    platform = build_windows_platform()
    assert platform.name == "win32"
    assert isinstance(platform.screenshot, MssScreenshotBackend)
    assert isinstance(platform.audio, WasapiAudioBackend)
    assert isinstance(platform.process, PsutilProcessBackend)
    assert isinstance(platform.compiler, MsvcCompiler)
    assert isinstance(platform, Platform)


def test_windows_backends_expose_capability_names() -> None:
    """Windows 侧各后端的能力名与依赖包与当前实现一致。"""
    assert MssScreenshotBackend().name == "mss"
    assert MssScreenshotBackend().required_packages == ("mss",)
    assert WasapiAudioBackend().name == "wasapi"
    assert WasapiAudioBackend().required_packages == (
        "sounddevice",
        "soundcard",
        "numpy",
    )
    assert PsutilProcessBackend().name == "psutil"
    assert PsutilProcessBackend().required_packages == ("psutil",)
    assert MsvcCompiler().name == "msvc"
    assert MsvcCompiler().required_packages == ()


def test_windows_process_backend_detects_alive_process() -> None:
    """Windows 进程存活探测已实现：当前进程活着，非法 PID 一律判死。"""
    backend = PsutilProcessBackend()
    assert backend.is_alive(os.getpid()) is True
    assert backend.is_alive(0) is False
    assert backend.is_alive(-1) is False


def test_windows_compiler_command() -> None:
    """Windows 侧编译命令是 ``cl.exe``，找不到时按 ``g++`` 兜底。"""
    command = MsvcCompiler().command()
    assert isinstance(command, str) and command
    assert command.lower().endswith(("cl.exe", "g++.exe", "g++"))


def test_windows_transport_endpoints(tmp_path: Path) -> None:
    """Windows 传输层的端口文件与 token 文件路径按约定派生。"""
    endpoint = tmp_path / "interviewmate.port"
    transport = TcpLoopbackTransport(endpoint)
    assert transport.name == "tcp-loopback"
    assert transport.endpoint_path == endpoint
    assert transport.token_path == tmp_path / "interviewmate.token"
    assert transport.required_packages == ()


# ----------------------------------------------------------------------
# Windows 侧传输层往返
# ----------------------------------------------------------------------


async def _echo_server(
    transport: TcpLoopbackTransport, received: list[bytes]
) -> None:
    """注册一个先读一行再回 ``pong`` 的回调，模拟协议层。"""

    async def on_connect(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            received.append(line)
            writer.write(b"pong\n")
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    await transport.start(on_connect)


async def test_windows_transport_roundtrip(tmp_path: Path) -> None:
    """起服务、连上、发 token 后收发一行、停服务，两个文件最后都被删掉。"""
    endpoint = tmp_path / "interviewmate.port"
    transport = TcpLoopbackTransport(endpoint)
    received: list[bytes] = []

    assert transport.prepare_endpoint() is True
    await _echo_server(transport, received)
    try:
        assert endpoint.exists()
        assert transport.token_path.exists()
        assert transport.is_endpoint_in_use() is True

        def exchange() -> bytes:
            """connect 发掉 token 后照旧发 JSON 行、读一行。"""
            client = transport.connect(timeout=2.0)
            try:
                client.sendall(b"ping\n")
                return client.recv(4096)
            finally:
                client.close()

        data = await asyncio.to_thread(exchange)
        assert data == b"pong\n"
        # token 握手在传输层被吃掉，协议层只看到业务请求
        assert received == [b"ping\n"]
    finally:
        await transport.stop()

    assert not endpoint.exists()
    assert not transport.token_path.exists()


def test_windows_transport_prepare_endpoint_removes_stale_file(
    tmp_path: Path,
) -> None:
    """端口文件存在但连不上时是残留文件，端口文件与 token 文件一起清掉。"""
    endpoint = tmp_path / "interviewmate.port"
    endpoint.write_text("59999\n", encoding="utf-8")
    token_path = tmp_path / "interviewmate.token"
    token_path.write_text("旧 token\n", encoding="utf-8")
    transport = TcpLoopbackTransport(endpoint)
    assert transport.is_endpoint_in_use() is False
    assert transport.prepare_endpoint() is True
    assert not endpoint.exists()
    assert not token_path.exists()


def test_windows_transport_cleanup_is_idempotent(tmp_path: Path) -> None:
    """端点文件与 token 文件不存在时清理也不报错。"""
    endpoint = tmp_path / "interviewmate.port"
    transport = TcpLoopbackTransport(endpoint)
    transport.cleanup_endpoint()
    endpoint.write_text("残留\n", encoding="utf-8")
    (tmp_path / "interviewmate.token").write_text("残留\n", encoding="utf-8")
    transport.cleanup_endpoint()
    assert not endpoint.exists()
    assert not (tmp_path / "interviewmate.token").exists()


# ----------------------------------------------------------------------
# E_SOCKET 客户端层断言：连不上与 token 不匹配
# ----------------------------------------------------------------------


async def test_send_request_raises_e_socket_when_unreachable(tmp_path: Path) -> None:
    """端口文件在但没人监听（连不上）时 send_request 抛 E_SOCKET。"""
    endpoint = tmp_path / "interviewmate.port"
    endpoint.write_text("1\n", encoding="utf-8")
    (tmp_path / "interviewmate.token").write_text("token\n", encoding="utf-8")

    with pytest.raises(SocketError) as excinfo:
        await asyncio.to_thread(send_request, "status", socket_path=endpoint)
    assert excinfo.value.code == "E_SOCKET"


async def test_send_request_raises_e_socket_on_token_mismatch(tmp_path: Path) -> None:
    """token 文件与服务端不匹配时被拒，send_request 抛 E_SOCKET。"""
    endpoint = tmp_path / "interviewmate.port"
    server = SocketServer(
        socket_path=endpoint,
        handlers={"status": lambda request: {"running": "running"}},
    )
    await server.start()
    try:
        # token 文件被换成与服务端 token 不匹配的陈旧内容
        (tmp_path / "interviewmate.token").write_text(
            "stale-token\n", encoding="utf-8"
        )
        with pytest.raises(SocketError) as excinfo:
            await asyncio.to_thread(send_request, "status", socket_path=endpoint)
        assert excinfo.value.code == "E_SOCKET"
    finally:
        await server.stop()
