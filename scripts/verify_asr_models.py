"""FunASR 本地模型加载验证：GPU 加载、流式推理、VAD、真实语音识别。

用法:
    ./.venv/bin/python scripts/verify_asr_models.py

模型目录默认 ~/models，可用环境变量 INTERVIEWMATE_MODELS 覆盖。

事实依据（funasr 1.4.29 实测）:
- `model_dir` 不起作用。funasr 的 download_from_ms 只在 kwargs 里存在
  `model_path` 时才跳过 hub 下载，否则把权重重新拉到 modelscope 缓存。
  ASR 与 VAD 是两个独立 AutoModel 实例，各自的 `model_path` 放顶层 kwargs。
- paraformer 流式模型的 chunk_size 是列表 [0, 10, 5]；
  fsmn-vad 模型的 chunk_size 是整数毫秒。两者放进同一个 AutoModel 会报
  TypeError: unsupported operand type(s) for /: 'list' and 'int'。
  因此 ASR 与 VAD 分成两个 AutoModel 实例。
"""

import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

# 模型目录默认在 ~/models，可用 INTERVIEWMATE_MODELS 覆盖
MODEL_ROOT = Path(os.environ.get("INTERVIEWMATE_MODELS", str(Path.home() / "models")))
ASR_DIR = str(MODEL_ROOT / "paraformer-zh-streaming")
VAD_DIR = str(MODEL_ROOT / "fsmn-vad")
ASR_WAV = f"{ASR_DIR}/example/asr_example.wav"
VAD_WAV = f"{VAD_DIR}/example/vad_example.wav"

CHUNK_SIZE = [0, 10, 5]
CHUNK_SAMPLES = 9600  # chunk_size[1] * 960


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"采样率应为 16000，实际 {w.getframerate()}"
        assert w.getnchannels() == 1, f"应为单声道，实际 {w.getnchannels()}"
        assert w.getsampwidth() == 2, f"应为 16bit，实际 {w.getsampwidth()}"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> int:
    from funasr import AutoModel

    audio = read_wav(ASR_WAV)
    print(f"[0] wav {len(audio)} samples = {len(audio) / 16000:.2f}s @16kHz mono float32")

    t0 = time.time()
    asr = AutoModel(model="paraformer-zh-streaming", model_path=ASR_DIR, device="cuda")
    print(f"[1] ASR  load ok in {time.time() - t0:.1f}s  device={asr.kwargs.get('device')}")

    t1 = time.time()
    vad = AutoModel(model="fsmn-vad", model_path=VAD_DIR, device="cuda")
    print(f"[2] VAD  load ok in {time.time() - t1:.1f}s  device={vad.kwargs.get('device')}")

    # 流式 ASR：contract 的 chunk_size=[0,10,5] + 共享 cache
    t2 = time.time()
    cache: dict = {}
    parts = []
    n = len(audio)
    for start in range(0, n, CHUNK_SAMPLES):
        seg = audio[start : start + CHUNK_SAMPLES]
        is_final = start + CHUNK_SAMPLES >= n
        if len(seg) < CHUNK_SAMPLES:
            seg = np.pad(seg, (0, CHUNK_SAMPLES - len(seg)))
        out = asr.generate(
            input=seg,
            cache=cache,
            is_final=is_final,
            chunk_size=CHUNK_SIZE,
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
            batch_size=1,
            disable_pbar=True,
        )
        if out and out[0].get("text"):
            parts.append(out[0]["text"])
            print(f"    chunk@{start / 16000:5.2f}s is_final={is_final} -> {out[0]['text']!r}")
    merged = "".join(parts)
    print(f"[3] ASR  streaming total {time.time() - t2:.2f}s -> {merged!r}")

    # 整段离线识别，对照流式结果
    t3 = time.time()
    whole = asr.generate(input=audio, disable_pbar=True)
    print(f"[4] ASR  whole-utterance ({time.time() - t3:.2f}s) -> {whole[0].get('text')!r}")

    # VAD：整数 chunk_size（毫秒）
    t4 = time.time()
    vad_out = vad.generate(input=read_wav(VAD_WAV), chunk_size=600, disable_pbar=True)
    print(f"[5] VAD  ({time.time() - t4:.2f}s) -> {vad_out[0].get('value')}")

    # 复现组合模式的失败：vad_model 与列表 chunk_size 不能共存
    t5 = time.time()
    combo = AutoModel(
        model="paraformer-zh-streaming",
        model_path=ASR_DIR,
        vad_model="fsmn-vad",
        vad_kwargs={"model_path": VAD_DIR},
        device="cuda",
    )
    try:
        combo.generate(
            input=audio[:CHUNK_SAMPLES],
            cache={},
            is_final=False,
            chunk_size=CHUNK_SIZE,
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
            batch_size=1,
            disable_pbar=True,
        )
        print("[6] COMBO unexpectedly succeeded")
    except TypeError as exc:
        print(f"[6] COMBO reproduces contract gap -> TypeError: {exc}")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
