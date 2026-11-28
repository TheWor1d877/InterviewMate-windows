"""loopback-test 一键自检：播放合成音频、从系统回环采集、喂 ASR 并给出结论。

用途与 :mod:`app.audio.asr_replay` 不同：replay 把 wav 文件直接喂给引擎，
完全绕过声卡，用来分清「模型问题」与「采集链路问题」；本模块验证的是
「系统声音 → 回环采集 → ASR」整条真实链路：

1. 读一段合成音频（默认 ``tests/assets/asr-test-question.wav``，
   由 ``scripts/windows/make-test-audio.cmd`` 离线合成）；
2. 经系统默认输出设备外放，播放时长即 wav 时长；
3. 同时从 WASAPI loopback 端点采集系统声音（平台音频后端的回环流，
   优先默认扬声器的回环端点）；
4. 采集到的 16000 Hz 单声道波形按 200 毫秒块喂给注入的 ASR 引擎，
   ``feed`` 的增量文本按序拼接、最后 ``flush``，拼法与采集链路一致；
5. 汇总电平（峰值/RMS，判断是否真的录到了声音）、识别文本与结论：
   识别出非空文本，或至少录到明显电平，都算链路通；全程静音时提示
   检查系统音量与默认输出设备。

引擎、播放器与采集器都按 :class:`LoopbackTestEngine`、
:class:`LoopbackTestPlayer` 与 :class:`LoopbackTestRecorder` 三个最小约定
注入：CLI（``app.cli`` 的 ``loopback-test`` 子命令）负责装配真实实现
（``FunASREngine`` / :class:`SoundDevicePlayer` / :class:`LoopbackCaptureRecorder`），
单元测试注入假件，不做真实播放与采集。

播放与采集的并发协调：先启动采集，再把播放放进线程跑
（``sounddevice`` 的 ``play`` + ``wait`` 是阻塞调用，事件循环不被占住）；
播放结束后再等 :data:`LOOPBACK_TAIL_SECONDS` 秒采集尾巴，最后停流收数据。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.audio.asr_replay import (
    REPLAY_BLOCK_MS,
    REPLAY_SAMPLE_RATE,
    iter_pcm_blocks,
    load_wav_int16_mono,
    pcm_int16_to_float32,
)
from app.core.exceptions import AudioDeviceError
from app.platform.types import AudioBackend
from app.utils.logger import log_event

__all__ = [
    "LOOPBACK_TAIL_SECONDS",
    "LOOPBACK_SILENCE_PEAK",
    "LoopbackTestResult",
    "LoopbackTestEngine",
    "LoopbackTestPlayer",
    "LoopbackTestRecorder",
    "SoundDevicePlayer",
    "LoopbackCaptureRecorder",
    "run_loopback_test",
    "_resample_to_rate",
]


# 播放结束后多录的尾巴秒数：系统混音与回环端点都有一点延迟，
# 多录一点再停流，结尾的字不会被截掉
LOOPBACK_TAIL_SECONDS: float = 0.8

# 「没录到声音」的峰值阈值。回环是系统内部的电气回环，不经过空气与
# 麦克风，正常链路的峰值接近播放波形的峰值；静音或设备不对时接近 0
LOOPBACK_SILENCE_PEAK: float = 0.01


class LoopbackTestEngine(Protocol):
    """回环自检引擎的最小调用约定。

    :class:`~app.audio.funasr_engine.FunASREngine` 满足这个约定，
    测试里注入的假引擎也按这个约定实现。
    """

    async def ensure_loaded(self) -> None:
        """加载模型，已经加载过就直接返回。"""
        ...

    async def feed(self, block: Any) -> str:
        """喂一块音频，返回这次解码出的增量文本。"""
        ...

    async def flush(self) -> str:
        """冲刷，返回最终文本。"""
        ...


class LoopbackTestPlayer(Protocol):
    """回环自检播放器的最小调用约定。"""

    def play(self, samples: "np.ndarray", sample_rate: int) -> None:
        """经系统默认输出设备播放一段波形，返回时播放已经结束。"""
        ...


class LoopbackTestRecorder(Protocol):
    """回环自检采集器的最小调用约定。"""

    def start(self) -> None:
        """开始采集，块由采集线程送进来攒着。"""
        ...

    def stop(self) -> "np.ndarray":
        """停止采集，返回攒到的 16000 Hz 单声道 float32 波形。"""
        ...


@dataclass
class LoopbackTestResult:
    """一次回环自检的汇总结果。

    :ivar peak: 采集波形的峰值，0 起的归一化幅值。
    :ivar rms: 采集波形的均方根。
    :ivar captured_seconds: 采集到的时长。
    :ivar played_seconds: 播放的时长（wav 时长）。
    :ivar text: 识别出的完整文本。
    :ivar expected: 调用方给的期望文案，没给是 None。
    :ivar expected_match: 期望文案与识别文本去空白后是否一致；
        没给期望文案时是 None。仅参考，不参与链路是否通过的判定。
    :ivar ok: 链路是否算通。
    :ivar conclusion: 给用户看的中文结论。
    """

    peak: float
    rms: float
    captured_seconds: float
    played_seconds: float
    text: str
    expected: str | None = None
    expected_match: bool | None = None
    ok: bool = False
    conclusion: str = ""


class SoundDevicePlayer:
    """用 sounddevice 经系统默认输出设备播放波形。

    ``play`` + ``wait`` 是阻塞调用，由 :func:`run_loopback_test` 放到
    线程里跑，不占事件循环。``sounddevice`` 在方法体内按需导入，
    没有音频依赖的环境也能导入本模块。

    有的输出设备拒绝按非默认采样率开流（PortAudio 报 ``Invalid sample
    rate``），播放前先查输出设备的默认采样率，不一致就线性插值重采样
    再播：共享模式的渲染流本来就会被系统混音，回环采集后仍重采样回
    16000 Hz 喂引擎，链路不受影响。
    """

    def play(self, samples: "np.ndarray", sample_rate: int) -> None:
        """经系统默认输出设备播放一段波形，阻塞到播放结束。

        :param samples: float32 的一维波形（单声道）。
        :param sample_rate: 波形的采样率。
        :raises AudioDeviceError: sounddevice 不可导入或播放失败。
        """
        try:
            import sounddevice as sd
        except Exception as exc:
            raise AudioDeviceError(f"sounddevice 不可导入：{exc}") from exc
        data = np.asarray(samples, dtype="float32").reshape(-1)
        try:
            info = sd.query_devices(None, "output")
            device_rate = int(float(info.get("default_samplerate", 0.0) or 0.0))
        except Exception:
            # 查不到设备默认采样率时按原采样率播，行为与不加这层兜底一致
            device_rate = 0
        if device_rate > 0 and device_rate != int(sample_rate):
            data = _resample_to_rate(data, int(sample_rate), device_rate)
            sample_rate = device_rate
        try:
            sd.play(data, samplerate=sample_rate)
            sd.wait()
        except Exception as exc:
            raise AudioDeviceError(f"播放音频失败：{exc}") from exc


def _resample_to_rate(
    samples: "np.ndarray", src_rate: int, dst_rate: int
) -> "np.ndarray":
    """把一段波形按线性插值重采样到目标采样率。

    两端对齐到首尾采样点，与采集层
    :func:`app.platform.windows.audio._resample_to` 的做法一致。

    :param samples: 一维波形。
    :param src_rate: 原采样率。
    :param dst_rate: 目标采样率。
    :returns: 新的一维 float32 数组，入参不被修改。
    """
    source = np.asarray(samples, dtype="float32").reshape(-1)
    if source.size == 0:
        return np.zeros(0, dtype="float32")
    if int(src_rate) == int(dst_rate):
        return np.array(source, dtype="float32", copy=True)
    out_len = int(round(source.size * int(dst_rate) / int(src_rate)))
    if out_len <= 0:
        return np.zeros(0, dtype="float32")
    if out_len == 1:
        return np.full(1, float(source[0]), dtype="float32")
    # 输入下标按比例铺开，两端对齐到首尾采样点
    positions = np.arange(out_len, dtype="float64") * (source.size - 1) / (out_len - 1)
    resampled = np.interp(positions, np.arange(source.size, dtype="float64"), source)
    return np.asarray(resampled, dtype="float32")


class LoopbackCaptureRecorder:
    """把平台音频后端的一路回环流包成 :class:`LoopbackTestRecorder`。

    用 :meth:`~app.platform.types.AudioBackend.start_stream` 打开指定
    回环设备编号的采集流，采集线程出的块（已下混并重采样到目标采样率）
    攒在内存里，:meth:`stop` 停流并把攒到的块拼成一段波形。

    :param backend: 平台音频后端，Windows 侧是 WASAPI 后端。
    :param device: 回环设备编号，取设备表里 ``is_loopback`` 设备的索引。
    :param sample_rate: 目标采样率，模型按 16000 Hz 走。
    :param block_duration_ms: 每块的毫秒数，与采集链路一致用 200。
    """

    def __init__(
        self,
        backend: AudioBackend,
        *,
        device: int,
        sample_rate: int,
        block_duration_ms: int = REPLAY_BLOCK_MS,
    ) -> None:
        self._backend = backend
        self._device = int(device)
        self._sample_rate = int(sample_rate)
        self._block_duration_ms = int(block_duration_ms)
        self._chunks: list["np.ndarray"] = []
        self._stream: Any = None

    def start(self) -> None:
        """打开回环采集流，块开始进 :meth:`_on_block`。

        :raises AudioDeviceError: 回环设备不可用，开流失败，或开流超时。
        """
        self._chunks = []
        self._stream = self._backend.start_stream(
            device=self._device,
            sample_rate=self._sample_rate,
            channels=1,
            block_duration_ms=self._block_duration_ms,
            on_block=self._on_block,
        )

    def _on_block(self, samples: "np.ndarray") -> None:
        """采集线程每出一块就攒一块。"""
        self._chunks.append(np.asarray(samples, dtype="float32").reshape(-1))

    def stop(self) -> "np.ndarray":
        """停流并把攒到的块拼成一段波形。

        :returns: float32 的一维波形；一块都没攒到时是空数组。
        """
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
        if not self._chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(self._chunks)


def _build_conclusion(text: str, peak: float, ok: bool) -> str:
    """按识别文本与电平生成给用户看的结论。

    :param text: 识别出的文本。
    :param peak: 采集波形的峰值。
    :param ok: 链路是否算通。
    :returns: 中文结论。
    """
    if text:
        return "回环链路验证通过：系统声音经回环采集后识别出了文本"
    if ok:
        return (
            f"回环链路基本可用：录到了声音（峰值 {peak:.4f}）但没有识别出文本，"
            "可检查 ASR 模型状态与音频清晰度"
        )
    return (
        f"回环链路验证未通过：没有录到明显声音（峰值 {peak:.4f}），"
        "请检查系统音量、默认输出设备是否静音，以及默认输出设备是否是"
        "真实的扬声器（虚拟声卡的回环录不到它人声）"
    )


async def run_loopback_test(
    wav_path: str | Path,
    engine: LoopbackTestEngine,
    player: LoopbackTestPlayer,
    recorder: LoopbackTestRecorder,
    *,
    expected: str | None = None,
    tail_seconds: float = LOOPBACK_TAIL_SECONDS,
    silence_peak: float = LOOPBACK_SILENCE_PEAK,
) -> LoopbackTestResult:
    """跑一次回环自检，返回汇总结果。

    顺序是：读 wav → 加载引擎 → 启动采集 → 外放 → 等尾巴 → 停流收数据
    → 算电平 → 按块喂引擎 → 冲刷 → 汇总结论。播放与采集并发：
    播放阻塞在线程里跑，采集块由采集线程攒着，两边不需要互相等。

    :param wav_path: wav 文件路径。
    :param engine: 注入的 ASR 引擎，见 :class:`LoopbackTestEngine`。
    :param player: 注入的播放器，见 :class:`LoopbackTestPlayer`。
    :param recorder: 注入的采集器，见 :class:`LoopbackTestRecorder`。
    :param expected: 期望文案，给了就做一次去空白对比，仅参考不影响结论。
    :param tail_seconds: 播放结束后多录的尾巴秒数。
    :param silence_peak: 「没录到声音」的峰值阈值。
    :returns: 汇总结果，见 :class:`LoopbackTestResult`。
    :raises ValueError: wav 无法解析。
    :raises AudioDeviceError: 采集或播放失败。
    """
    pcm_int16 = load_wav_int16_mono(wav_path)
    wave_float = pcm_int16_to_float32(pcm_int16)
    played_seconds = wave_float.size / REPLAY_SAMPLE_RATE
    log_event(
        "audio.loopback_test.start",
        "开始回环链路自检",
        path=str(wav_path),
        played_seconds=round(played_seconds, 3),
        expected=expected,
    )

    await engine.ensure_loaded()

    recorder.start()
    try:
        # 阻塞播放放到线程里，事件循环不被占住，采集线程照常出块
        await asyncio.to_thread(player.play, wave_float, REPLAY_SAMPLE_RATE)
        log_event(
            "audio.loopback_test.played",
            "播放完成",
            seconds=round(played_seconds, 3),
        )
        if tail_seconds > 0:
            await asyncio.sleep(tail_seconds)
    finally:
        captured = recorder.stop()

    if captured.size:
        peak = float(np.max(np.abs(captured)))
        rms = float(np.sqrt(np.mean(np.square(captured))))
    else:
        peak = 0.0
        rms = 0.0
    captured_seconds = captured.size / REPLAY_SAMPLE_RATE
    log_event(
        "audio.loopback_test.captured",
        "回环采集完成",
        samples=int(captured.size),
        seconds=round(captured_seconds, 3),
        peak=round(peak, 4),
        rms=round(rms, 4),
    )

    # 采集到的波形按 200 毫秒块喂引擎，增量按序拼接、最后冲刷，
    # 拼法与采集链路（audio_manager）和回放测试（asr_replay）一致。
    # 喂的一定是采集到的波形而不是源文件：喂源文件就绕过了回环链路，
    # 识别结果再好也验证不了「系统声音→回环采集」这一段
    parts: list[str] = []
    blocks = 0
    for block in iter_pcm_blocks(captured):
        if block.size == 0:
            continue
        blocks += 1
        text = await engine.feed(block)
        if text:
            parts.append(text)
    final = await engine.flush()
    text = "".join(parts) + final

    expected_match: bool | None = None
    if expected is not None:
        expected_match = text.strip() == expected.strip()
    ok = bool(text.strip()) or peak >= silence_peak
    conclusion = _build_conclusion(text, peak, ok)
    log_event(
        "audio.loopback_test.done",
        "回环链路自检完成",
        blocks=blocks,
        text=text,
        ok=ok,
        peak=round(peak, 4),
        rms=round(rms, 4),
    )
    return LoopbackTestResult(
        peak=peak,
        rms=rms,
        captured_seconds=captured_seconds,
        played_seconds=played_seconds,
        text=text,
        expected=expected,
        expected_match=expected_match,
        ok=ok,
        conclusion=conclusion,
    )