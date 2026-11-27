"""音频块的流式采集。

见《5-语音输入与流式ASR》「采集参数」一节：

- 采样率 16000 Hz、单声道、每次回调读 200 毫秒；
- 200 毫秒是采集回调每次读取的量，不是送进模型的块长；
- 音频块进入一个有界异步队列，队列满时丢弃最旧的块，丢弃计数写日志；
- 音频回调线程只做读取和入队，不发网络请求。

这个模块只管一路音频：开流、关流、块队列。麦克风与系统回环分开采集
与混成一路的编排放在 :mod:`app.audio.audio_source`。

开流那步是阻塞调用，放到线程里跑并按开流时限限时，不占住事件循环，
见《8-集成与验收》异步任务与超时管理两节。
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.config import current_config
from app.core.exceptions import AudioDeviceError
from app.platform.types import AudioBackend, AudioStream
from app.utils.logger import log_event

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "AudioBlock",
    "BoundedAudioQueue",
    "AudioCapture",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_BLOCK_DURATION_MS",
    "DEFAULT_QUEUE_CAPACITY",
    "DEFAULT_INIT_TIMEOUT",
]


# 采集参数，见《5-语音输入与流式ASR》「采集参数」一节
DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_CHANNELS: int = 1
DEFAULT_BLOCK_DURATION_MS: int = 200

# 音频块队列的默认容量，200 毫秒一块时够放 10 秒
DEFAULT_QUEUE_CAPACITY: int = 50

# 开流时限的兜底秒数：后端没给时限、进程里也没加载过配置时用这个值，
# 见《8-集成与验收》超时管理一节
DEFAULT_INIT_TIMEOUT: float = 10.0


@dataclass(frozen=True, eq=False)
class AudioBlock:
    """一块音频。

    :param samples: 一维 float32 波形，形状是 ``(采样点数,)``，
        采样率已经是目标采样率。
    :param source: 来源标签，取 ``microphone`` 或 ``system_monitor``，
        混成一路时是 ``mixed``。
    :param sample_rate: 采样率。
    :param captured_at: 采集时刻的单调时钟秒数。
    """

    samples: "np.ndarray"
    source: str
    sample_rate: int
    captured_at: float


class BoundedAudioQueue:
    """有界的音频块队列。

    队列满时丢掉最旧的块，丢弃计数记在 :attr:`dropped_count` 上，
    并写一条日志，事件名 ``audio.capture.block_dropped``，
    见《5-语音输入与流式ASR》「采集参数」一节。

    这个类不做线程同步：入队动作在事件循环线程里执行，
    跨线程的投递由 :class:`AudioCapture` 用
    :meth:`asyncio.loop.call_soon_threadsafe` 交回来。
    """

    def __init__(self, capacity: int = DEFAULT_QUEUE_CAPACITY) -> None:
        """
        :param capacity: 队列容量，必须大于 0。
        :raises ValueError: 容量不是正数。
        """
        if capacity <= 0:
            raise ValueError(f"队列容量必须大于 0，收到 {capacity}")
        self._queue: asyncio.Queue[AudioBlock] = asyncio.Queue(maxsize=capacity)
        self._dropped = 0

    @property
    def capacity(self) -> int:
        """队列容量。"""
        return self._queue.maxsize

    @property
    def dropped_count(self) -> int:
        """累计丢弃的块数。"""
        return self._dropped

    @property
    def size(self) -> int:
        """当前队列里的块数。"""
        return self._queue.qsize()

    def push(self, block: AudioBlock) -> bool:
        """入队一块音频，队列满了先丢最旧的块。

        :param block: 要入队的音频块。
        :returns: 这次入队有没有丢块。
        """
        dropped = False
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._dropped += 1
            dropped = True
            log_event(
                "audio.capture.block_dropped",
                "音频队列已满，丢弃最旧的块",
                level="WARNING",
                dropped=self._dropped,
                queue_capacity=self._queue.maxsize,
                source=block.source,
            )
        self._queue.put_nowait(block)
        return dropped

    def get_nowait(self) -> AudioBlock:
        """取一块音频，队列空时抛 :class:`asyncio.QueueEmpty`。"""
        return self._queue.get_nowait()

    async def get(self) -> AudioBlock:
        """等下一块音频。"""
        return await self._queue.get()


class AudioCapture:
    """一路音频的流式采集。

    采集回调线程只做读取与入队，不发网络请求，也不做推理。

    :param backend: 平台音频后端，负责真正开流。
    :param device: 设备索引，None 表示用后端默认的输入设备。
    :param source: 来源标签，写在每一块上。
    :param sample_rate: 目标采样率。
    :param channels: 目标声道数，当前只支持 1。
    :param block_duration_ms: 每次回调读的毫秒数。
    :param queue_capacity: 音频块队列的容量。
    """

    def __init__(
        self,
        backend: AudioBackend,
        device: int | None = None,
        *,
        source: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        block_duration_ms: int = DEFAULT_BLOCK_DURATION_MS,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
    ) -> None:
        self._backend = backend
        self._device = device
        self._source = source
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_duration_ms = block_duration_ms
        self._queue_capacity = queue_capacity
        self._queue = BoundedAudioQueue(queue_capacity)
        self._stream: AudioStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self._running = False

    @property
    def source(self) -> str:
        """这一路的来源标签。"""
        return self._source

    @property
    def device(self) -> int | None:
        """这一路用的设备索引。"""
        return self._device

    @property
    def sample_rate(self) -> int:
        """目标采样率。"""
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        """采集是不是开着。"""
        return self._running

    @property
    def queue(self) -> BoundedAudioQueue:
        """音频块队列。"""
        return self._queue

    @property
    def dropped_count(self) -> int:
        """累计丢弃的块数。"""
        return self._queue.dropped_count

    async def start(self) -> None:
        """开始采集。

        已经在采集时直接返回。

        开流是阻塞调用，放到线程里跑，不占住事件循环，
        同时按开流时限限时：超时报 E_AUDIO_DEVICE，
        见《8-集成与验收》异步任务与超时管理两节。

        :raises AudioDeviceError: 设备不可用，开流失败，或开流超时。
        """
        if self._running:
            return
        # 每次采集换一个队列，丢弃计数也从头算
        self._queue = BoundedAudioQueue(self._queue_capacity)
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        timeout = self._init_timeout()
        try:
            stream = await asyncio.wait_for(
                asyncio.to_thread(
                    self._backend.start_stream,
                    device=self._device,
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    block_duration_ms=self._block_duration_ms,
                    on_block=self._on_block,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            self._loop = None
            raise AudioDeviceError(f"开启音频流超时（{timeout:g} 秒）") from exc
        except AudioDeviceError:
            self._loop = None
            raise
        except Exception as exc:
            self._loop = None
            raise AudioDeviceError(f"开启音频流失败：{exc}") from exc
        self._stream = stream
        self._running = True
        log_event(
            "audio.capture.started",
            "开始采集音频",
            source=self._source,
            device=self._device,
            sample_rate=self._sample_rate,
            block_duration_ms=self._block_duration_ms,
            queue_capacity=self._queue_capacity,
        )

    def _init_timeout(self) -> float:
        """开流时限（秒）。

        优先取后端自己的时限：平台后端构造时没给，就从进程里加载好的配置里
        取 ``audio.init_timeout``。后端没有这个属性时退回配置里的值，
        再退回 :data:`DEFAULT_INIT_TIMEOUT`，见《8-集成与验收》超时管理一节。
        """
        timeout = getattr(self._backend, "init_timeout", None)
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
            if timeout > 0:
                return float(timeout)
        config = current_config()
        if config is not None:
            return float(config.audio.init_timeout)
        return DEFAULT_INIT_TIMEOUT

    async def stop(self) -> None:
        """停止采集。

        没在采集时直接返回。
        """
        if not self._running:
            return
        self._running = False
        stream = self._stream
        self._stream = None
        self._loop = None
        if stream is not None:
            try:
                # 关流要等采集线程退出，放到线程里做，别占住事件循环
                await asyncio.to_thread(stream.stop)
            except Exception as exc:
                log_event(
                    "audio.capture.stop_failed",
                    f"停止音频流失败：{exc}",
                    level="WARNING",
                    source=self._source,
                )
        log_event(
            "audio.capture.stopped",
            "停止采集音频",
            source=self._source,
            dropped=self._queue.dropped_count,
        )

    async def get_block(self) -> AudioBlock:
        """等下一块音频。"""
        return await self._queue.get()

    def _on_block(self, samples: "np.ndarray") -> None:
        """采集回调线程的入口：打包成块，然后入队。

        这里不发网络请求，也不做推理。跨线程时把入队动作交回事件循环线程。
        """
        loop = self._loop
        if loop is None:
            return
        block = AudioBlock(
            samples=samples,
            source=self._source,
            sample_rate=self._sample_rate,
            captured_at=time.monotonic(),
        )
        queue = self._queue
        if threading.get_ident() == self._loop_thread:
            queue.push(block)
            return
        try:
            loop.call_soon_threadsafe(queue.push, block)
        except RuntimeError:
            # 事件循环已经关了，采集线程还在推块，这块直接丢掉
            pass
