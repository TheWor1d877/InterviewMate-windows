"""麦克风与系统回环的分离采集。

见《5-语音输入与流式ASR》「音频来源」一节：麦克风与系统音频分开采集时，
各自保留来源标签，避免把用户自己说的话当成面试官的问题。

这里提供三种采集器：

- :class:`SeparatedCapture`：麦克风与系统回环各采一路，两路分别产出来，
  标签分别是 ``microphone`` 与 ``system_monitor``；
- :class:`MixedCapture`：两路混成一路，标签是 ``mixed``；
- 单路来源直接用 :class:`~app.audio.audio_capture.AudioCapture`。

:func:`create_capture` 按来源选择的结果造出对应的那一个。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

from app.audio.audio_capture import (
    DEFAULT_BLOCK_DURATION_MS,
    DEFAULT_CHANNELS,
    DEFAULT_QUEUE_CAPACITY,
    DEFAULT_SAMPLE_RATE,
    AudioBlock,
    AudioCapture,
    BoundedAudioQueue,
)
from app.audio.audio_devices import (
    SOURCE_MICROPHONE,
    SOURCE_MIXED,
    SOURCE_SYSTEM_MONITOR,
    SourceSelection,
)
from app.platform.types import AudioBackend, AudioDevice
from app.utils.logger import log_event

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "SeparatedCapture",
    "MixedCapture",
    "create_capture",
]


class _DualSourceCapture:
    """麦克风与系统回环两路采集的公共部分。

    :param backend: 平台音频后端。
    :param microphone: 麦克风设备。
    :param monitor: 系统回环设备。
    """

    def __init__(
        self,
        backend: AudioBackend,
        *,
        microphone: AudioDevice,
        monitor: AudioDevice,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        block_duration_ms: int = DEFAULT_BLOCK_DURATION_MS,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
    ) -> None:
        self._microphone = AudioCapture(
            backend,
            microphone.index,
            source=SOURCE_MICROPHONE,
            sample_rate=sample_rate,
            channels=channels,
            block_duration_ms=block_duration_ms,
            queue_capacity=queue_capacity,
        )
        self._monitor = AudioCapture(
            backend,
            monitor.index,
            source=SOURCE_SYSTEM_MONITOR,
            sample_rate=sample_rate,
            channels=channels,
            block_duration_ms=block_duration_ms,
            queue_capacity=queue_capacity,
        )
        self._sample_rate = sample_rate

    @property
    def microphone(self) -> AudioCapture:
        """麦克风那一路。"""
        return self._microphone

    @property
    def monitor(self) -> AudioCapture:
        """系统回环那一路。"""
        return self._monitor

    @property
    def is_running(self) -> bool:
        """两路是不是都在采。"""
        return self._microphone.is_running and self._monitor.is_running

    async def start(self) -> None:
        """开始采集两路音频。

        :raises AudioDeviceError: 设备不可用，或开流失败。
        """
        await self._microphone.start()
        try:
            await self._monitor.start()
        except Exception:
            # 第二路开不起来时把第一路收掉，不留半开的状态
            await self._microphone.stop()
            raise

    async def stop(self) -> None:
        """停止采集两路音频。"""
        await self._monitor.stop()
        await self._microphone.stop()


class SeparatedCapture(_DualSourceCapture):
    """麦克风与系统回环分开采集。

    两路各有一个块队列，块各带自己的来源标签：
    ``microphone`` 那一路是用户自己说的话，``system_monitor`` 那一路是
    系统播放的声音。标签跟到每一块上，上层按标签分开处理，
    不把用户自己说的话当成面试官的问题。
    """

    async def get_microphone_block(self) -> AudioBlock:
        """等麦克风那一路的下一块。"""
        return await self._microphone.get_block()

    async def get_system_monitor_block(self) -> AudioBlock:
        """等系统回环那一路的下一块。"""
        return await self._monitor.get_block()


class MixedCapture(_DualSourceCapture):
    """麦克风与系统回环混成一路的采集。

    两路各采一块，按采样点相加并限幅，成一块标签是 ``mixed`` 的音频。
    混成一路之后允许用 VAD 切分，也可以接说话人区分，
    见《5-语音输入与流式ASR》「音频来源」一节。
    """

    def __init__(
        self,
        backend: AudioBackend,
        *,
        microphone: AudioDevice,
        monitor: AudioDevice,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        block_duration_ms: int = DEFAULT_BLOCK_DURATION_MS,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
    ) -> None:
        super().__init__(
            backend,
            microphone=microphone,
            monitor=monitor,
            sample_rate=sample_rate,
            channels=channels,
            block_duration_ms=block_duration_ms,
            queue_capacity=queue_capacity,
        )
        self._queue_capacity = queue_capacity
        self._queue = BoundedAudioQueue(queue_capacity)
        self._task: asyncio.Task[None] | None = None

    @property
    def queue(self) -> BoundedAudioQueue:
        """混合后的块队列。"""
        return self._queue

    @property
    def is_running(self) -> bool:
        """两路是不是都在采。"""
        return self._task is not None and super().is_running

    async def start(self) -> None:
        """开始采集两路音频并混合。

        :raises AudioDeviceError: 设备不可用，或开流失败。
        """
        if self._task is not None:
            return
        self._queue = BoundedAudioQueue(self._queue_capacity)
        await super().start()
        self._task = asyncio.get_running_loop().create_task(self._mix_loop())
        log_event(
            "audio.mixed.started",
            "开始混合采集音频",
            sample_rate=self._sample_rate,
            queue_capacity=self._queue_capacity,
        )

    async def stop(self) -> None:
        """停止采集并收掉混合任务。"""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await super().stop()

    async def get_block(self) -> AudioBlock:
        """等下一块混合后的音频。"""
        return await self._queue.get()

    async def _mix_loop(self) -> None:
        """取两路的块混成一块，推进混合队列。"""
        while True:
            mic_block = await self._microphone.get_block()
            monitor_block = await self._monitor.get_block()
            self._queue.push(self._mix(mic_block, monitor_block))

    def _mix(self, mic_block: AudioBlock, monitor_block: AudioBlock) -> AudioBlock:
        """把两块的波形相加并限幅，成一块混合音频。"""
        import numpy as np

        length = min(mic_block.samples.size, monitor_block.samples.size)
        mixed = (
            mic_block.samples[:length].astype("float32")
            + monitor_block.samples[:length].astype("float32")
        )
        np.clip(mixed, -1.0, 1.0, out=mixed)
        return AudioBlock(
            samples=mixed,
            source=SOURCE_MIXED,
            sample_rate=self._sample_rate,
            captured_at=time.monotonic(),
        )


def create_capture(
    backend: AudioBackend,
    selection: SourceSelection,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    block_duration_ms: int = DEFAULT_BLOCK_DURATION_MS,
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
) -> AudioCapture | SeparatedCapture | MixedCapture:
    """按来源选择的结果造采集器。

    ``microphone`` 与 ``system_monitor`` 是单路采集，
    ``mixed`` 是两路混成一路。

    :param backend: 平台音频后端。
    :param selection: 来源选择结果。
    :returns: 采集器。
    :raises ValueError: 选择结果里的设备数量不对。
    """
    if selection.source == SOURCE_MIXED:
        if len(selection.devices) < 2:
            raise ValueError("mixed 来源需要麦克风与系统回环两个设备")
        return MixedCapture(
            backend,
            microphone=selection.devices[0],
            monitor=selection.devices[1],
            sample_rate=sample_rate,
            channels=channels,
            block_duration_ms=block_duration_ms,
            queue_capacity=queue_capacity,
        )
    if not selection.devices:
        raise ValueError("来源选择结果里没有设备")
    return AudioCapture(
        backend,
        selection.devices[0].index,
        source=selection.source,
        sample_rate=sample_rate,
        channels=channels,
        block_duration_ms=block_duration_ms,
        queue_capacity=queue_capacity,
    )
