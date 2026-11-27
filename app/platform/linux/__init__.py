"""Linux 平台实现。

《9-平台差异与Windows适配》「目录结构」一节里的 Linux 侧五个能力在这里，
外加全局热键：

- 截图：GNOME Shell 扩展静默抓屏，回退 XDG Portal，见 :mod:`app.platform.linux.screenshot`；
- 音频：sounddevice，见 :mod:`app.platform.linux.audio`；
- 传输：AF_UNIX socket，见 :mod:`app.platform.linux.transport`；
- 进程存活探测：``os.kill``，见 :mod:`app.platform.linux.process`；
- 编译器：``g++``，见 :mod:`app.platform.linux.compiler`；
- 全局热键：读 ``/dev/input``，见 :mod:`app.platform.linux.hotkeys`。

:func:`get_platform` 返回进程内共用的同一个实现，
:func:`build_platform` 每次新建一个，两者的能力完全一致。
"""

from __future__ import annotations

from pathlib import Path

from app.platform.linux.audio import SoundDeviceAudioBackend
from app.platform.linux.compiler import GppCompiler
from app.platform.linux.hotkeys import EvdevHotkeyProvider
from app.platform.linux.process import LinuxProcessBackend
from app.platform.linux.screenshot import GnomeScreenshotBackend
from app.platform.linux.transport import UnixSocketTransport
from app.platform.types import Platform, Transport

__all__ = ["LinuxPlatform", "build_platform", "get_platform"]


class LinuxPlatform:
    """Linux 平台实现。

    各能力后端在构造时各建一个：它们都不持有连接与设备句柄，
    每次调用都能拿到同样的行为。全局热键后端只在 ``start()`` 时才打开设备。
    """

    # 平台名
    name: str = "linux"

    def __init__(self) -> None:
        self.screenshot = GnomeScreenshotBackend()
        self.audio = SoundDeviceAudioBackend()
        self.process = LinuxProcessBackend()
        self.compiler = GppCompiler()
        self.hotkeys = EvdevHotkeyProvider()

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
        """按端点路径造一个 AF_UNIX socket 传输层。

        :param endpoint_path: socket 文件路径。
        """
        return UnixSocketTransport(endpoint_path)


# 进程内共用的平台实现
_platform: Platform = LinuxPlatform()


def build_platform() -> Platform:
    """新建一个 Linux 平台实现。"""
    return LinuxPlatform()


def get_platform() -> Platform:
    """返回进程内共用的 Linux 平台实现。"""
    return _platform
