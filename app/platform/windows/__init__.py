"""Windows 平台门面：mss / WASAPI / psutil / TCP+token。

Windows 侧五项能力在这里：

- 截图：``mss``，见 :mod:`app.platform.windows.screenshot`；
- 音频：WASAPI loopback，见 :mod:`app.platform.windows.audio`；
- 传输：TCP loopback（端口文件 + token），见 :mod:`app.platform.windows.transport`；
- 进程存活探测：``psutil``，见 :mod:`app.platform.windows.process`；
- 编译器：``cl.exe``，见 :mod:`app.platform.windows.compiler`；
- 全局热键：:class:`~app.platform.windows.hotkeys.AhkHotkeyProvider`，按 AutoHotkey
  进程与 hotkeys.ahk 脚本路径探测可用性。

:func:`get_platform` 返回进程内共用的同一个实现，
:func:`build_platform` 每次新建一个，两者的能力完全一致。
"""

from __future__ import annotations

from pathlib import Path

from app.platform.types import Platform, Transport
from app.platform.windows.audio import WasapiAudioBackend
from app.platform.windows.compiler import MsvcCompiler
from app.platform.windows.hotkeys import AhkHotkeyProvider
from app.platform.windows.process import PsutilProcessBackend
from app.platform.windows.screenshot import MssScreenshotBackend
from app.platform.windows.transport import TcpLoopbackTransport

__all__ = ["WindowsPlatform", "build_platform", "get_platform"]


class WindowsPlatform:
    """Windows 平台实现：组装 mss、WASAPI、psutil、TCP+token 与编译器五项能力。"""

    # 平台名
    name: str = "win32"

    def __init__(self) -> None:
        self.screenshot = MssScreenshotBackend()
        self.audio = WasapiAudioBackend()
        self.process = PsutilProcessBackend()
        self.compiler = MsvcCompiler()
        self.hotkeys = AhkHotkeyProvider()


    @property
    def required_packages(self) -> tuple[str, ...]:
        """平台用到的全部第三方包名。"""
        packages: list[str] = []
        for backend in (
            self.screenshot,
            self.audio,
            self.process,
            self.compiler,
        ):
            for package in backend.required_packages:
                if package not in packages:
                    packages.append(package)
        return tuple(packages)

    def transport(self, endpoint_path: Path) -> Transport:
        """按端点路径造一个 TCP loopback 传输层。

        :param endpoint_path: 端口文件的路径。
        """
        return TcpLoopbackTransport(endpoint_path)


# 进程内共用的平台实现
_platform: Platform = WindowsPlatform()


def build_platform() -> Platform:
    """新建一个 Windows 平台实现。"""
    return WindowsPlatform()


def get_platform() -> Platform:
    """返回进程内共用的 Windows 平台实现。"""
    return _platform
