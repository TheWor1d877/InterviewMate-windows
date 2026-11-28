"""SenseVoice 重识别引擎的测试。

覆盖《5-语音输入与流式ASR》「重识别与热词纠正」一节的行为：

- recognize：整段识别 → 清 rich text 标签 → 热词纠正；
- generate 参数带 use_itn=True 与 language=zh；
- 空音频不调模型，直接返回空串；
- 模型异常抛 AsrDecodeError、加载异常抛 AsrLoadError，都不吞；
- auto 设备 CUDA 失败回退 CPU；
- 模型实例整个生命周期只建一次；
- correct_text：只做热词纠正（重识别降级路径用）；
- int16 PCM 字节输入自动归一化。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.audio.hotword_corrector import HotwordCorrector
from app.audio.sensevoice_engine import (
    DEVICE_CUDA,
    SenseVoiceEngine,
    strip_rich_text,
)
from app.core.exceptions import AsrDecodeError, AsrLoadError

np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
class _FakeSenseVoiceModel:
    """假模型：记下每次 generate 的参数，返回预置文本，可注入错误。"""

    def __init__(self) -> None:
        self.text: str = ""
        self.error: Exception | None = None
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> list[dict[str, str]]:
        """假装推理一次，返回预置文本。"""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return [{"key": "fake", "text": self.text}]


class _FakeFactory:
    """假模型工厂：记下创建请求，可指定某些设备创建失败。"""

    def __init__(
        self,
        model: _FakeSenseVoiceModel | None = None,
        *,
        fail_devices: tuple[str, ...] = (),
    ) -> None:
        self.model = model if model is not None else _FakeSenseVoiceModel()
        self.fail_devices = fail_devices
        self.requests: list[dict[str, str]] = []
        self.builds = 0

    def __call__(self, *, model_path: str, vad_model_path: str, device: str) -> Any:
        """按参数建一个假模型。"""
        self.requests.append(
            {
                "model_path": model_path,
                "vad_model_path": vad_model_path,
                "device": device,
            }
        )
        self.builds += 1
        if device in self.fail_devices:
            raise RuntimeError(f"{device} 初始化失败")
        return self.model


def _engine(
    factory: _FakeFactory,
    *,
    corrector: HotwordCorrector | None = None,
    **kwargs: Any,
) -> SenseVoiceEngine:
    """按假工厂造一个引擎，热词表显式注入，不依赖仓库配置。"""
    options: dict[str, Any] = {
        "model_dir": "models",
        "model_factory": factory,
        "corrector": (
            corrector if corrector is not None else HotwordCorrector(["队头阻塞"])
        ),
    }
    options.update(kwargs)
    return SenseVoiceEngine(**options)


# --------------------------------------------------------------------------
# 文本清理
# --------------------------------------------------------------------------
def test_strip_rich_text() -> None:
    """rich text 标签用 <|...|> 正则清理，首尾空白一并去掉。"""
    assert (
        strip_rich_text("<|zh|><|NEUTRAL|>你好世界<|Speech|><|END|>")
        == "你好世界"
    )
    assert strip_rich_text("没有标签") == "没有标签"
    assert strip_rich_text("") == ""


# --------------------------------------------------------------------------
# 识别
# --------------------------------------------------------------------------
async def test_recognize_strips_tags_and_corrects_hotwords() -> None:
    """整段识别 → 清标签 → 热词纠正，产出最终文本。"""
    factory = _FakeFactory()
    factory.model.text = "<|zh|><|NEUTRAL|>什么是对头阻塞<|END|>"
    engine = _engine(factory)

    text = await engine.recognize(np.zeros(3200, dtype="float32"))

    assert text == "什么是队头阻塞"


async def test_generate_params_use_itn_and_language() -> None:
    """推理调用带 use_itn=True、language=zh 与实验一致的参数组合。"""
    factory = _FakeFactory()
    factory.model.text = "文本"
    engine = _engine(factory)
    await engine.recognize(np.zeros(3200, dtype="float32"))

    call = factory.model.calls[0]
    assert call["use_itn"] is True
    assert call["language"] == "zh"
    assert call["cache"] == {}
    assert call["merge_vad"] is True
    assert call["batch_size_s"] == 60


async def test_empty_audio_skips_model() -> None:
    """空音频直接返回空串，不调模型。"""
    factory = _FakeFactory()
    engine = _engine(factory)

    text = await engine.recognize(np.zeros(0, dtype="float32"))

    assert text == ""
    assert factory.model.calls == []


async def test_model_error_raises_asr_decode() -> None:
    """模型推理失败抛 AsrDecodeError，不吞异常，状态置成 error。"""
    factory = _FakeFactory()
    factory.model.error = RuntimeError("推理崩了")
    engine = _engine(factory)

    with pytest.raises(AsrDecodeError):
        await engine.recognize(np.zeros(3200, dtype="float32"))
    assert engine.state == "error"


async def test_load_failure_raises_asr_load_and_sets_error() -> None:
    """模型加载失败抛 AsrLoadError，状态置成 error，由调用方降级。"""
    factory = _FakeFactory(fail_devices=("cuda", "cpu"))
    engine = _engine(factory)

    with pytest.raises(AsrLoadError):
        await engine.recognize(np.zeros(3200, dtype="float32"))
    assert engine.state == "error"


async def test_auto_device_falls_back_to_cpu() -> None:
    """auto 设备 CUDA 初始化失败回退 CPU，识别照常完成。"""
    factory = _FakeFactory(fail_devices=(DEVICE_CUDA,))
    engine = _engine(factory, device="auto")

    text = await engine.recognize(np.zeros(3200, dtype="float32"))

    assert engine.device == "cpu"
    assert factory.requests[0]["device"] == "cuda"
    assert factory.requests[1]["device"] == "cpu"
    assert text == factory.model.text


async def test_model_built_once() -> None:
    """模型实例整个生命周期只建一次，多次识别复用同一个实例。"""
    factory = _FakeFactory()
    engine = _engine(factory)

    await engine.recognize(np.zeros(3200, dtype="float32"))
    await engine.recognize(np.zeros(3200, dtype="float32"))

    assert factory.builds == 1
    assert engine.is_ready is True


async def test_bytes_input_normalized() -> None:
    """int16 PCM 字节输入按满量程归一化成 float32。"""
    factory = _FakeFactory()
    engine = _engine(factory)

    await engine.recognize(np.full(3200, 32767, dtype="int16").tobytes())

    wave = factory.model.calls[0]["input"]
    assert wave.dtype == np.float32
    assert float(wave[0]) == pytest.approx(32767.0 / 32768.0, abs=1e-6)


async def test_correct_text_applies_hotwords() -> None:
    """correct_text 只做热词纠正，不碰音频，降级路径与重识别口径一致。"""
    engine = _engine(_FakeFactory())

    text = await engine.correct_text("什么是对头阻塞")

    assert text == "什么是队头阻塞"


def test_default_corrector_loaded() -> None:
    """不显式给纠正器时从默认路径加载热词表。"""
    engine = _engine(_FakeFactory(), corrector=None)
    assert isinstance(engine.corrector, HotwordCorrector)