"""Windows 平台实现。

《9-平台差异与Windows适配》「目录结构」一节里的 Windows 侧五个能力在这里：

- 截图：``mss``，见 :mod:`app.platform.windows.screenshot`；
- 音频：WASAPI loopback，见 :mod:`app.platform.windows.audio`；
- 传输：TCP loopback，见 :mod:`app.platform.windows.transport`；
- 进程存活探测：``psutil``，见 :mod:`app.platform.windows.process`；
- 编译器：``cl.exe``，见 :mod:`app.platform.windows.compiler`；
- 全局热键：本轮用 :class:`~app.inputs.global_hotkeys.NullHotkeyProvider` 占位。

本轮五个能力模块都是接口占位，方法体抛 :class:`NotImplementedError`，
真正的 Windows 后端按《9-平台差异与Windows适配》「实施顺序」一节排在
Linux 侧 M2 到 M8 之后做。全局热键没有占位实现，直接用
:class:`~app.inputs.global_hotkeys.NullHotkeyProvider`，``available()`` 恒为 False。

:func:`get_platform` 返回进程内共用的同一个实现，
:func:`build_platform` 每次新建一个，两者的能力完全一致。
"""

from __future__ import annotations

from pathlib import Path

from app.inputs.global_hotkeys import NullHotkeyProvider
from app.platform.types import Platform, Transport
from app.platform.windows.audio import WasapiAudioBackend
from app.platform.windows.compiler import MsvcCompiler
from app.platform.windows.process import PsutilProcessBackend
from app.platform.windows.screenshot import MssScreenshotBackend
from app.platform.windows.transport import TcpLoopbackTransport

__all__ = ["WindowsPlatform", "build_platform", "get_platform"]


class WindowsPlatform:
    """Windows 平台实现（本轮只有接口占位）。"""

    # 平台名
    name: str = "win32"

    def __init__(self) -> None:
        self.screenshot = MssScreenshotBackend()
        self.audio = WasapiAudioBackend()
        self.process = PsutilProcessBackend()
        self.compiler = MsvcCompiler()
        self.hotkeys = NullHotkeyProvider()

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
