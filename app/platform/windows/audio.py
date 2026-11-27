"""Windows 侧的音频后端：WASAPI 占位。

见《9-平台差异与Windows适配》「Windows 音频采集」一节：Windows 侧麦克风枚举
用 ``sounddevice``（与 Linux 侧同一个包），系统回环音频用 ``soundcard`` 的
WASAPI loopback，因为 PortAudio 不暴露 loopback 设备枚举。

《9-平台差异与Windows适配》「实施顺序」一节把 Windows 侧的实现排在 Linux 侧
M2 到 M8 之后，音频排在截图之后。本模块现在只有接口占位：方法体抛
:class:`NotImplementedError`。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.platform.types import AudioBlockCallback, AudioDevice, AudioStream

if TYPE_CHECKING:
    import numpy as np

__all__ = ["WasapiAudioBackend"]


class WasapiAudioBackend:
    """基于 WASAPI 的音频后端（本轮只有接口占位）。"""

    # 后端名
    name: str = "wasapi"

    # 该能力需要的第三方包名
    required_packages: tuple[str, ...] = ("sounddevice", "soundcard", "numpy")

    def list_devices(self) -> list[AudioDevice]:
        """列出本机的音频输入设备。

        :raises NotImplementedError: 始终抛异常。
        """
        raise NotImplementedError(
            "Windows 侧的音频设备枚举尚未实现，"
            "见《9-平台差异与Windows适配》的实施顺序"
        )

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
        :raises NotImplementedError: 始终抛异常。
        """
        raise NotImplementedError(
            "Windows 侧的音频采集尚未实现，见《9-平台差异与Windows适配》的实施顺序"
        )

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

        :raises NotImplementedError: 始终抛异常。
        """
        raise NotImplementedError(
            "Windows 侧的音频流式采集尚未实现，"
            "见《9-平台差异与Windows适配》的实施顺序"
        )
