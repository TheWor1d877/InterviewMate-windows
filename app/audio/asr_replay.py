"""ASR 回放测试：把一个 wav 文件按块喂给注入的 ASR 引擎。

用户反馈识别不准时，要先分清是「模型问题」还是「音频链路问题」：真实
采集链路上音频从麦克风或系统回环进来，设备选择、下混、重采样、分块任何
一环都可能引入问题。回放测试绕过采集环节，直接把一段已知的 wav 喂给
引擎，识别结果不好时就能断定问题出在模型或引擎侧。

测试音频用 ``scripts/windows/make-test-audio.cmd`` 合成（Windows 自带的
SAPI 语音，离线不联网），默认文案是「解释一下什么是GDS」，默认输出
``tests/assets/asr-test-question.wav``。

处理流程与采集链路的参数保持一致：

- 读 wav：标准库 ``wave`` 解析，支持 8/16/24/32 位 PCM；
- 下混：多声道按各声道求平均成单声道；
- 重采样：numpy 线性插值到 16000 Hz；
- 分块：按 200 毫秒（3200 点）切块，与采集层 ``block_duration_ms`` 一致；
- 喂引擎：int16 PCM 归一化成 float32 再喂，``feed`` 的增量文本按序拼接，
  最后 ``flush`` 拿最终文本，拼法与 :class:`~app.audio.audio_manager`
  对采集流的拼法一致。

引擎由调用方注入：CLI 集成时传 ``FunASREngine`` 实例，单元测试注入假
引擎。本模块不构造引擎。引擎只需要两个异步方法：``feed(block)`` 返回
增量文本、``flush()`` 返回最终文本，:class:`AsrReplayEngine` 是这个
最小约定。

CLI（``app.cli``）的 ``asr-replay`` 子命令负责加载配置、构造引擎并调用
:func:`replay_wav`；本模块只管回放本身。
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np

from app.utils.logger import log_event

__all__ = [
    "AsrReplayEngine",
    "REPLAY_SAMPLE_RATE",
    "REPLAY_BLOCK_MS",
    "REPLAY_BLOCK_SAMPLES",
    "load_wav_int16_mono",
    "iter_pcm_blocks",
    "pcm_int16_to_float32",
    "replay_wav",
]


# 回放目标采样率：模型按 16000 Hz 走，
# 见《5-语音输入与流式ASR》「采集参数」一节
REPLAY_SAMPLE_RATE: int = 16000

# 回放块长：与采集层一致，200 毫秒一块
REPLAY_BLOCK_MS: int = 200

# 每块的点数：16000 × 0.2 = 3200
REPLAY_BLOCK_SAMPLES: int = REPLAY_SAMPLE_RATE * REPLAY_BLOCK_MS // 1000


class AsrReplayEngine(Protocol):
    """回放引擎的最小调用约定。

    :class:`~app.audio.funasr_engine.FunASREngine` 满足这个约定，
    测试里注入的假引擎也按这个约定实现。
    """

    async def feed(self, block: Any) -> str:
        """喂一块音频，返回这次解码出的增量文本。"""
        ...

    async def flush(self) -> str:
        """冲刷，返回最终文本。"""
        ...


def _decode_wav_frames(raw: bytes, sample_width: int) -> "np.ndarray":
    """把 ``wave`` 的原始帧字节解成 [-1, 1) 的 float32 波形。

    8 位 PCM 是无符号的，中点 128 对应 0；24 位按小端补码符号扩展；
    32 位按 int32 处理（``wave`` 模块只认 PCM 编码）。

    :param raw: ``wave.readframes`` 返回的字节。
    :param sample_width: 位宽（字节），1/2/3/4。
    :returns: float32 的一维数组（帧 × 声道平铺）。
    :raises ValueError: 位宽不支持。
    """
    if sample_width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (samples - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        if not raw:
            return np.zeros(0, dtype="float32")
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = bytes_[:, 0] | (bytes_[:, 1] << 8) | (bytes_[:, 2] << 16)
        values = (values << 8) >> 8  # 符号扩展到 32 位
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"不支持的 wav 位宽：{sample_width} 字节")


def _resample_linear(samples: "np.ndarray", src_rate: float, dst_rate: int) -> "np.ndarray":
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
    if float(src_rate) == float(dst_rate):
        return np.array(source, dtype="float32", copy=True)
    out_len = int(round(source.size * float(dst_rate) / float(src_rate)))
    if out_len <= 0:
        return np.zeros(0, dtype="float32")
    if out_len == 1:
        return np.full(1, float(source[0]), dtype="float32")
    # 输入下标按比例铺开，两端对齐到首尾采样点
    positions = np.arange(out_len, dtype="float64") * (source.size - 1) / (out_len - 1)
    resampled = np.interp(positions, np.arange(source.size, dtype="float64"), source)
    return np.asarray(resampled, dtype="float32")


def load_wav_int16_mono(
    path: str | Path,
    *,
    target_rate: int = REPLAY_SAMPLE_RATE,
) -> "np.ndarray":
    """读一个 wav 文件，转成目标采样率的单声道 int16 PCM。

    多声道按各声道求平均下混；采样率与目标不一致时线性插值重采样；
    量化乘满量程 32768，与解码时的除法对称，写读一次不损失幅值。

    :param path: wav 文件路径。
    :param target_rate: 目标采样率，默认 16000 Hz。
    :returns: int16 的一维 PCM。
    :raises ValueError: wav 无法解析（位宽不支持，或不是 PCM 编码）。
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    samples = _decode_wav_frames(raw, sample_width)
    if channels > 1:
        # 字节不是完整帧时截掉尾巴，reshape 才不会报错
        frames = samples.size // channels
        samples = samples[: frames * channels].reshape(-1, channels).mean(axis=1)
    samples = _resample_linear(samples, frame_rate, target_rate)
    return np.clip(np.rint(samples * 32768.0), -32768.0, 32767.0).astype(np.int16)


def iter_pcm_blocks(
    pcm: "np.ndarray",
    block_samples: int = REPLAY_BLOCK_SAMPLES,
) -> Iterator["np.ndarray"]:
    """把 int16 PCM 按固定点数切块，最后一块不足点数时也产出。

    :param pcm: int16 的一维 PCM。
    :param block_samples: 每块的点数，默认 200 毫秒（3200 点）。
    :returns: 逐块产出；块是入参的切片视图，不复制数据。
    :raises ValueError: ``block_samples`` 不是正数。
    """
    if block_samples <= 0:
        raise ValueError(f"块长必须是正数：{block_samples}")
    source = np.asarray(pcm).reshape(-1)
    for start in range(0, source.size, block_samples):
        yield source[start : start + block_samples]


def pcm_int16_to_float32(pcm: "np.ndarray") -> "np.ndarray":
    """把 int16 PCM 归一化成 [-1, 1) 的 float32 波形。

    引擎按 float32 收波形，直接把 int16 数组转 float32 会把样点值
    当成声压（±32767），必须先除满量程。

    :param pcm: int16 的一维 PCM。
    :returns: float32 的一维波形。
    """
    return np.asarray(pcm, dtype="float32") / 32768.0


async def replay_wav(
    engine: AsrReplayEngine,
    path: str | Path,
    *,
    block_ms: int = REPLAY_BLOCK_MS,
    sample_rate: int = REPLAY_SAMPLE_RATE,
) -> str:
    """把一个 wav 文件回放给注入的 ASR 引擎，返回识别出的完整文本。

    完整文本是各块 ``feed`` 的增量文本按序拼接，再接上 ``flush`` 的
    最终文本，与采集链路的拼法一致。引擎要先用真实引擎时得加载好模型
    （``FunASREngine`` 先 ``await engine.ensure_loaded()``）。

    :param engine: 注入的 ASR 引擎，见 :class:`AsrReplayEngine`。
    :param path: wav 文件路径。
    :param block_ms: 每块毫秒数，默认 200。
    :param sample_rate: 目标采样率，默认 16000 Hz。
    :returns: 识别出的完整文本，一段音频都没有时是空串。
    :raises ValueError: wav 无法解析或块长非法。
    """
    pcm = load_wav_int16_mono(path, target_rate=sample_rate)
    block_samples = max(1, int(sample_rate * block_ms // 1000))
    blocks = list(iter_pcm_blocks(pcm, block_samples))
    log_event(
        "asr.replay.start",
        "开始回放 wav 到 ASR",
        path=str(path),
        sample_rate=sample_rate,
        block_ms=block_ms,
        samples=int(pcm.size),
        blocks=len(blocks),
    )
    parts: list[str] = []
    for index, block in enumerate(blocks, start=1):
        text = await engine.feed(pcm_int16_to_float32(block))
        log_event(
            "asr.replay.block",
            "回放块解码完成",
            level="DEBUG",
            block=index,
            blocks=len(blocks),
            text=text,
        )
        if text:
            parts.append(text)
    final = await engine.flush()
    text = "".join(parts) + final
    log_event("asr.replay.done", "回放识别完成", text=text)
    return text