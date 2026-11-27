"""平台能力接口定义。

《2-骨架与配置》目录结构约定一节把平台相关能力收进 :mod:`app.platform`：
上层拿到的接口在各平台同构，平台相关的内容只落在
:mod:`app.platform.windows` 子包里。

这里只放数据结构与 Protocol，不放实现：

- :class:`AudioDevice` 是音频设备表的一行，字段对应《接口契约》
  音频后端（WASAPI）一节的音频设备枚举；
- :class:`Screenshot` 是一次截图的结果，内容在内存里，不落盘；
- 七个 Protocol 分别描述截图、音频、传输、进程、编译器、全局热键与平台门面本身。

每个 Protocol 都有 ``required_packages`` 属性，值是该能力真正需要的第三方包名；
不需要额外包时是空元组。各后端按需声明自己的依赖，差别就体现在这个属性上。
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from app.inputs.global_hotkeys import GlobalHotkeyProvider

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "AudioDevice",
    "AudioBlockCallback",
    "AudioStream",
    "Screenshot",
    "ScreenshotTarget",
    "AudioBackend",
    "ScreenshotBackend",
    "Transport",
    "ProcessBackend",
    "CompilerBackend",
    "GlobalHotkeyProvider",
    "Platform",
]


# 截图目标：screen 是整屏（多显示器时拼成一张），primary 是主屏幕，
# monitor 是指定的那块屏。取值与配置里的 screenshot.target 一致。
ScreenshotTarget = Literal["screen", "primary", "monitor"]


@dataclass(frozen=True)
class AudioDevice:
    """本机的一个音频设备。

    字段与《接口契约》音频后端（WASAPI）一节的音频设备枚举一一对应：

    - ``name`` 是设备名，Windows 上的设备名由 WASAPI 枚举给出；
    - ``index`` 是设备在本机枚举结果里的真实索引，不是枚举顺序；
    - ``is_input`` 表示这是输入设备；
    - ``is_loopback`` 表示这是系统回环设备（Windows 的 WASAPI loopback）；
    - ``default_samplerate`` 是设备默认采样率；
    - ``channels`` 是设备的输入声道数；
    - ``is_default`` 表示这是系统当前的默认设备，默认输入设备与默认输出
      sink 的回环设备都会置真。放在字段列表末尾并带默认值，其它平台后端
      不用改。
    """

    name: str
    index: int
    is_input: bool
    is_loopback: bool
    default_samplerate: float
    channels: int
    is_default: bool = False


# 音频块回调：收到的是一块已经下混并重采样好的波形，
# 形状是 ``(采样点数,)``，dtype 是 float32。
AudioBlockCallback = Callable[["np.ndarray"], None]


@runtime_checkable
class AudioStream(Protocol):
    """一路已经打开的音频流。

    块由打开流时给的回调送来，采集线程只做读取与入队。
    """

    @property
    def is_active(self) -> bool:
        """流是不是还开着。"""
        ...

    def stop(self) -> None:
        """关掉流。已经关掉的流再调一次不报错。"""
        ...


@dataclass(frozen=True)
class Screenshot:
    """一次截图的结果。

    图片内容在内存里，不落盘：字段 ``data`` 是图片字节，``width`` 与 ``height``
    是像素尺寸。Windows 侧用 ``mss`` 直接取内存里的位图，交给上层的形态一致。
    """

    data: bytes
    width: int
    height: int


@runtime_checkable
class AudioBackend(Protocol):
    """音频设备枚举与采集能力。"""

    # 该能力需要的第三方包名，不需要额外包时是空元组
    required_packages: tuple[str, ...]

    def list_devices(self) -> list[AudioDevice]:
        """列出本机的音频输入设备。

        :returns: 设备表，字段见 :class:`AudioDevice`。
        :raises AudioDeviceError: 底层音频库不可导入，或枚举失败。
        """
        ...

    def record(
        self,
        *,
        sample_rate: int,
        channels: int,
        seconds: float,
    ) -> "np.ndarray":
        """采一段音频。

        :param sample_rate: 采样率。
        :param channels: 声道数。
        :param seconds: 采集时长。
        :returns: float32 数组，形状是 ``(采样点, 声道数)``。
        :raises AudioDeviceError: 采集失败。
        """
        ...

    def start_stream(
        self,
        *,
        device: int | None,
        sample_rate: int,
        channels: int,
        block_duration_ms: int,
        on_block: AudioBlockCallback,
    ) -> AudioStream:
        """开始回调式采集。

        流打开后，采集线程每次读 ``block_duration_ms`` 毫秒，
        通过 ``on_block`` 把这一块送出来。采集线程只做读取与入队，
        不发网络请求，也不做推理，见《5-语音输入与流式ASR》「采集参数」一节。

        每块的形状是 ``(int(sample_rate * block_duration_ms / 1000),)`` 的一维
        float32 数组，采样率是 ``sample_rate``，声道数是 1：
        设备原生采样率与声道数由后端自己下混与重采样。

        :param device: 设备索引，None 表示用后端默认的输入设备。
        :param sample_rate: 目标采样率，采集层按这个值出块。
        :param channels: 目标声道数，当前只支持 1。
        :param block_duration_ms: 每次回调读的毫秒数。
        :param on_block: 收到一块音频时调的回调。
        :returns: 这一路流的句柄，:meth:`AudioStream.stop` 用来关流。
        :raises AudioDeviceError: 设备不可用，或开流失败。
        """
        ...


@runtime_checkable
class ScreenshotBackend(Protocol):
    """截图能力。"""

    # 后端名，对应配置里 ``providers.screenshot`` 的真实实现名
    name: str

    # 该能力需要的第三方包名，不需要额外包时是空元组
    required_packages: tuple[str, ...]

    async def probe(self, timeout: float = 3.0) -> str:
        """检查截图接口能不能用。

        只查接口在不在，不真截图，所以不会在用户屏幕上弹对话框。

        :param timeout: 超时秒数。
        :returns: 一句中文说明。
        :raises ScreenshotError: 截图接口不可用。
        """
        ...

    async def capture(
        self,
        *,
        target: ScreenshotTarget = "screen",
        monitor: int = 0,
    ) -> Screenshot:
        """做一次截图。

        :param target: 截图目标，见 :data:`ScreenshotTarget`。
        :param monitor: 显示器序号，0 起数，只在 ``target`` 是 ``monitor`` 时有意义。
        :returns: 截图结果，字段见 :class:`Screenshot`。
        :raises ScreenshotError: 截图失败或被拒绝。
        """
        ...


@runtime_checkable
class Transport(Protocol):
    """进程通信的传输层。"""

    # 该能力需要的第三方包名，不需要额外包时是空元组
    required_packages: tuple[str, ...]

    @property
    def endpoint_path(self) -> Path:
        """通信端点的存放路径。"""
        ...

    async def start(
        self,
        on_connect: Callable[
            [asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]
        ],
    ) -> None:
        """开始监听。

        :param on_connect: 有连接进来时调用的回调。
        :raises OSError: 监听创建失败。
        """
        ...

    async def stop(self) -> None:
        """停止监听并收掉端点文件。"""
        ...

    def connect(self, *, timeout: float) -> socket.socket:
        """连到服务端，返回已连接的 socket 对象。

        返回同步的 socket，调用方直接用它发请求读响应。

        :param timeout: 连接超时秒数。
        :raises OSError: 连不上。
        """
        ...

    def is_endpoint_in_use(self) -> bool:
        """端点是不是被活进程占用。

        :returns: 被活进程占用返回 True，端点不存在或只是残留文件返回 False。
        """
        ...

    def prepare_endpoint(self) -> bool:
        """把端点准备好，返回能不能继续启动。

        :returns: 可以重建端点返回 True，被活进程占用返回 False。
        """
        ...

    def cleanup_endpoint(self) -> None:
        """删掉端点文件，文件不存在也不报错。"""
        ...

    def restrict_permissions(self) -> None:
        """收紧端点文件的权限，只允许属主访问。"""
        ...


@runtime_checkable
class ProcessBackend(Protocol):
    """进程存活探测能力。"""

    # 该能力需要的第三方包名，不需要额外包时是空元组
    required_packages: tuple[str, ...]

    def is_alive(self, pid: int) -> bool:
        """判断 PID 对应的进程是不是还活着。

        :param pid: 进程 PID。
        """
        ...


@runtime_checkable
class CompilerBackend(Protocol):
    """代码编译能力。"""

    # 该能力需要的第三方包名，不需要额外包时是空元组
    required_packages: tuple[str, ...]

    def command(self) -> str:
        """返回编译命令。"""
        ...


@runtime_checkable
class Platform(Protocol):
    """平台门面，向上层提供各能力的实现。

    平台在进程启动时选定一次，上层任何模块都不判断 ``sys.platform``，
    判断只发生在 :func:`app.platform.get_platform`。
    """

    # 平台名，Windows 侧是 ``win32``
    name: str

    # 平台用到的全部第三方包名，不需要额外包时是空元组
    required_packages: tuple[str, ...]

    # 五个能力后端
    screenshot: ScreenshotBackend
    audio: AudioBackend
    process: ProcessBackend
    compiler: CompilerBackend

    # 全局热键，见 :mod:`app.inputs.global_hotkeys`
    hotkeys: GlobalHotkeyProvider

    def transport(self, endpoint_path: Path) -> Transport:
        """按端点路径造一个传输层。

        :param endpoint_path: 通信端点的存放路径。
        """
        ...
