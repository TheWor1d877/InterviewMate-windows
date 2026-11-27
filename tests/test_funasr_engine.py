"""FunASR 引擎的测试。

用假的模型工厂注入，不做真实推理。覆盖《5-语音输入与流式ASR》
「FunASR 模型」「流式推理」「部分结果与最终结果」与《接口契约》
「FunASR 音频块参数」几节要求的行为：

- 200 毫秒的块攒够 600 毫秒才调一次模型；
- 推理参数按契约传，``cache`` 同一条流内共用、不同流之间不共享；
- 冲刷用 ``is_final`` 为 true 的那次调用，冲刷完 ``cache`` 被丢弃；
- 部分结果与最终结果分开返回；
- VAD 关闭时不创建 VAD 实例；
- 设备 ``auto`` 在 CUDA 失败时回退 CPU 并记日志；
- 解码失败抛 ``AsrDecodeError``，错误码是 ``E_ASR_DECODE``。
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import pytest

from app.audio.audio_capture import AudioBlock
from app.audio.funasr_engine import (
    ASR_FINAL_LATENCY_METRIC,
    ASR_LATENCY_METRIC,
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    MAX_PENDING_SAMPLES,
    MODEL_CHUNK_SAMPLES,
    STATE_ERROR,
    STATE_READY,
    FunASREngine,
)
from app.core.exceptions import AsrDecodeError, AsrLoadError
from app.utils.timing import MetricRecorder, default_recorder, reset_default_recorder

np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
class _FakeModel:
    """假模型：记下每次 generate 的参数，按次数返回片段文本。"""

    def __init__(self, *, model: str, model_path: str, device: str) -> None:
        self.model = model
        self.model_path = model_path
        self.device = device
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> list[dict[str, str]]:
        """假装解码一次，返回这一段的文本。"""
        self.calls.append(kwargs)
        return [{"key": "fake", "text": f"{self.model}#{len(self.calls)}"}]


class _FakeVadModel(_FakeModel):
    """假 VAD 模型：返回固定的语音区间。"""

    def generate(self, **kwargs: Any) -> list[dict[str, Any]]:
        """假装做一次端点检测。"""
        self.calls.append(kwargs)
        return [{"key": "fake", "value": [[0, 1200], [1500, 3200]]}]


class _CacheWritingModel(_FakeModel):
    """会往 cache 里写状态的假模型。

    真模型会把编码器与解码器的状态写进 cache，
    这个替身照着做，用来检查缓存有没有被丢掉。
    """

    def generate(self, **kwargs: Any) -> list[dict[str, str]]:
        """假装解码一次，往 cache 里写状态。"""
        self.calls.append(kwargs)
        cache = kwargs.get("cache")
        if isinstance(cache, dict):
            cache["encoder"] = {"state": len(self.calls)}
            cache["decoder"] = {"state": len(self.calls)}
        return [{"key": "fake", "text": f"{self.model}#{len(self.calls)}"}]


class _FailingFinalModel(_CacheWritingModel):
    """最后一次调用会抛错的假模型。"""

    def generate(self, **kwargs: Any) -> list[dict[str, str]]:
        """往 cache 里写状态，is_final 为真时抛错。"""
        if kwargs.get("is_final"):
            cache = kwargs.get("cache")
            if isinstance(cache, dict):
                cache["encoder"] = {"state": len(self.calls) + 1}
            raise RuntimeError("冲刷时模型崩了")
        return super().generate(**kwargs)


class _CacheWritingFactory:
    """造会往 cache 里写状态的假模型的工厂。"""

    def __init__(self, *, fail_on_final: bool = False) -> None:
        self.requests: list[dict[str, str]] = []
        self.models: list[_CacheWritingModel] = []
        self.fail_on_final = fail_on_final

    def __call__(self, *, model: str, model_path: str, device: str) -> _CacheWritingModel:
        """建一个会写 cache 的假模型。"""
        self.requests.append({"model": model, "model_path": model_path, "device": device})
        fake: _CacheWritingModel
        if model == "fsmn-vad":
            fake = _CacheWritingModel(model=model, model_path=model_path, device=device)
        elif self.fail_on_final:
            fake = _FailingFinalModel(model=model, model_path=model_path, device=device)
        else:
            fake = _CacheWritingModel(model=model, model_path=model_path, device=device)
        self.models.append(fake)
        return fake

    @property
    def asr_model(self) -> _CacheWritingModel:
        """ASR 那个假模型。"""
        for model in self.models:
            if model.model != "fsmn-vad":
                return model
        raise AssertionError("ASR 模型还没建起来")


class _FakeFactory:
    """假模型工厂：记下每次创建请求，可以指定某些设备或模型创建失败。"""

    def __init__(
        self,
        *,
        fail_devices: tuple[str, ...] = (),
        fail_models: tuple[str, ...] = (),
    ) -> None:
        self.requests: list[dict[str, str]] = []
        self.models: list[_FakeModel] = []
        self.fail_devices = fail_devices
        self.fail_models = fail_models

    def __call__(self, *, model: str, model_path: str, device: str) -> _FakeModel:
        """按参数建一个假模型。"""
        self.requests.append({"model": model, "model_path": model_path, "device": device})
        if device in self.fail_devices:
            raise RuntimeError(f"{device} 初始化失败")
        if model in self.fail_models:
            raise RuntimeError(f"{model} 加载失败")
        fake: _FakeModel
        if model == "fsmn-vad":
            fake = _FakeVadModel(model=model, model_path=model_path, device=device)
        else:
            fake = _FakeModel(model=model, model_path=model_path, device=device)
        self.models.append(fake)
        return fake

    @property
    def asr_model(self) -> _FakeModel:
        """ASR 那个假模型。"""
        for model in self.models:
            if model.model != "fsmn-vad":
                return model
        raise AssertionError("ASR 模型还没建起来")

    @property
    def asr_requests(self) -> list[dict[str, str]]:
        """ASR 的创建请求。"""
        return [request for request in self.requests if request["model"] != "fsmn-vad"]


class _ListHandler(logging.Handler):
    """把日志记录攒起来的 handler，用来看清 log_event 传了什么。"""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """收下这条记录。"""
        self.records.append(record)


@pytest.fixture()
def captured_logs() -> list[logging.LogRecord]:
    """把项目日志记录攒起来，测试里检查事件名与字段。"""
    logger = logging.getLogger("interviewmate")
    handler = _ListHandler()
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


@pytest.fixture()
def isolated_recorder() -> Iterator[MetricRecorder]:
    """用例前后清空进程级记录器，样本不串到别的用例。"""
    reset_default_recorder()
    try:
        yield default_recorder()
    finally:
        reset_default_recorder()


def _block(count: int = 3200, value: float = 0.1) -> AudioBlock:
    """造一块 200 毫秒的音频块（16000 Hz 单声道 3200 点）。"""
    return AudioBlock(
        samples=np.full(count, value, dtype="float32"),
        source="microphone",
        sample_rate=16000,
        captured_at=0.0,
    )


def _engine(factory: _FakeFactory, **kwargs: Any) -> FunASREngine:
    """按假工厂造一个引擎。"""
    options: dict[str, Any] = {"model_factory": factory}
    options.update(kwargs)
    return FunASREngine(**options)


# --------------------------------------------------------------------------
# 攒块与推理参数
# --------------------------------------------------------------------------
async def test_feed_waits_for_600ms_before_calling_model() -> None:
    """200 毫秒的块攒够 600 毫秒才调一次模型。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()

    # 两块 200 毫秒还差 400 毫秒，不调模型
    assert await engine.feed(_block()) == ""
    assert await engine.feed(_block()) == ""
    assert factory.asr_model.calls == []

    # 第三块凑够 600 毫秒，调一次模型
    text = await engine.feed(_block())
    assert len(factory.asr_model.calls) == 1
    assert text == "paraformer-zh-streaming#1"


async def test_feed_passes_contract_parameters() -> None:
    """推理参数按《接口契约》「FunASR 流式推理调用」一节传。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()
    await engine.feed(_block())
    await engine.feed(_block())
    await engine.feed(_block())

    call = factory.asr_model.calls[0]
    assert call["chunk_size"] == [0, 10, 5]
    assert call["encoder_chunk_look_back"] == 4
    assert call["decoder_chunk_look_back"] == 1
    assert call["batch_size"] == 1
    assert call["is_final"] is False
    # 一次调用只传一段 600 毫秒的波形
    assert call["input"].shape == (MODEL_CHUNK_SAMPLES,)
    assert call["input"].dtype == np.float32


async def test_feed_converts_block_to_float32_waveform() -> None:
    """音频块进模型前转成 float32 的一维波形。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()
    await engine.feed(_block())
    await engine.feed(_block())
    await engine.feed(_block())

    waveform = factory.asr_model.calls[0]["input"]
    assert waveform.ndim == 1
    assert waveform.dtype == np.float32
    assert waveform.size == MODEL_CHUNK_SAMPLES


async def test_feed_warns_on_sample_rate_mismatch(
    captured_logs: list[logging.LogRecord],
) -> None:
    """块的采样率不是 16000 Hz 时记一条日志。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()

    block = AudioBlock(
        samples=np.full(3200, 0.1, dtype="float32"),
        source="microphone",
        sample_rate=48000,
        captured_at=0.0,
    )
    await engine.feed(block)

    warnings = [
        record
        for record in captured_logs
        if getattr(record, "event", "") == "asr.block.sample_rate_mismatch"
    ]
    assert warnings, "采样率不对时没有写日志"
    assert getattr(warnings[-1], "fields", {})["sample_rate"] == 48000


# --------------------------------------------------------------------------
# cache 管理
# --------------------------------------------------------------------------
async def test_cache_shared_within_stream_and_not_between_streams() -> None:
    """同一条流内共用一个 cache，不同流之间不共享。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()

    # 第一条流：三次喂音频攒出一块，冲刷再调一次
    for _ in range(3):
        await engine.feed(_block())
    await engine.flush()

    # 第二条流
    for _ in range(3):
        await engine.feed(_block())

    calls = factory.asr_model.calls
    assert len(calls) == 3
    first_stream = [call["cache"] for call in calls[:2]]
    second_stream = [call["cache"] for call in calls[2:]]
    assert first_stream[0] is first_stream[1]
    assert second_stream[0] is not first_stream[0]


async def test_flush_uses_is_final_and_drops_cache() -> None:
    """冲刷用 is_final 为 true 的那次调用，冲刷完 cache 被丢弃。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()
    for _ in range(3):
        await engine.feed(_block())

    stream_cache = factory.asr_model.calls[-1]["cache"]
    await engine.flush()

    final_call = factory.asr_model.calls[-1]
    assert final_call["is_final"] is True
    assert final_call["cache"] is stream_cache

    # 下一条语音流重新建一个 cache
    for _ in range(3):
        await engine.feed(_block())
    new_cache = factory.asr_model.calls[-1]["cache"]
    assert new_cache is not stream_cache


async def test_flush_without_audio_returns_empty_text() -> None:
    """一条音频都没有时冲刷返回空串，也不调模型。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()

    assert await engine.flush() == ""
    assert factory.asr_model.calls == []


# --------------------------------------------------------------------------
# 耗时指标
# --------------------------------------------------------------------------
async def test_flush_records_no_final_latency_sample(
    isolated_recorder: MetricRecorder,
) -> None:
    """冲刷不再记 asr_final_latency：这段耗时由 AudioManager 记。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()
    for _ in range(3):
        await engine.feed(_block())
    before = len(isolated_recorder.samples(ASR_FINAL_LATENCY_METRIC))

    await engine.flush()

    after = len(isolated_recorder.samples(ASR_FINAL_LATENCY_METRIC))
    assert before == 0
    assert after == before
    # 冲刷确实调了模型，不是没跑到
    assert factory.asr_model.calls[-1]["is_final"] is True


async def test_feed_records_asr_latency_and_flush_does_not(
    isolated_recorder: MetricRecorder,
) -> None:
    """asr_latency 口径不变：只在喂块解码时记，冲刷不记。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()
    for _ in range(3):
        await engine.feed(_block())

    samples = isolated_recorder.samples(ASR_LATENCY_METRIC)
    assert len(samples) == 1
    assert samples[0].value_ms >= 0

    await engine.flush()

    assert len(isolated_recorder.samples(ASR_LATENCY_METRIC)) == 1


async def test_flush_drops_cache_after_normal_flush() -> None:
    """正常冲刷后缓存被丢弃，调用计数归零。"""
    factory = _CacheWritingFactory()
    engine = FunASREngine(model_factory=factory, device=DEVICE_CPU)
    await engine.ensure_loaded()
    for _ in range(3):
        await engine.feed(_block())

    # 模型往 cache 里写了状态
    assert engine.has_cached_state is True
    assert engine.stream_calls == 1

    await engine.flush()

    assert engine.has_cached_state is False
    assert engine.stream_calls == 0
    assert engine.pending_samples == 0


async def test_flush_drops_cache_even_when_decode_fails() -> None:
    """冲刷时解码抛错，缓存照样丢掉，异常照旧抛出。

    缓存留在引擎里的话，下一条语音流会复用它，
    上一条语音的状态就串到下一条，
    见《5-语音输入与流式ASR》「流式推理」一节。
    """
    factory = _CacheWritingFactory(fail_on_final=True)
    engine = FunASREngine(model_factory=factory, device=DEVICE_CPU)
    await engine.ensure_loaded()
    for _ in range(3):
        await engine.feed(_block())

    # 模型往 cache 里写了状态
    assert engine.has_cached_state is True

    with pytest.raises(AsrDecodeError) as excinfo:
        await engine.flush()
    assert excinfo.value.code == "E_ASR_DECODE"

    # 冲刷失败也要清理：缓存丢掉，调用计数归零
    assert engine.has_cached_state is False
    assert engine.stream_calls == 0
    assert engine.pending_samples == 0


async def test_next_stream_gets_a_new_cache_after_failed_flush() -> None:
    """冲刷失败之后，下一条语音流拿到的是新缓存。"""
    factory = _CacheWritingFactory(fail_on_final=True)
    engine = FunASREngine(model_factory=factory, device=DEVICE_CPU)
    await engine.ensure_loaded()
    for _ in range(3):
        await engine.feed(_block())

    failed_cache = factory.asr_model.calls[-1]["cache"]
    with pytest.raises(AsrDecodeError):
        await engine.flush()

    # 下一条语音流重新建一个缓存
    for _ in range(3):
        await engine.feed(_block())

    new_cache = factory.asr_model.calls[-1]["cache"]
    assert new_cache is not failed_cache


# --------------------------------------------------------------------------
# 部分结果与最终结果
# --------------------------------------------------------------------------
async def test_partial_and_final_results_are_returned_separately() -> None:
    """部分结果与最终结果分开返回，整句由调用方拼接。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()

    partial = ""
    for _ in range(3):
        partial += await engine.feed(_block())
    final = await engine.flush()

    assert partial == "paraformer-zh-streaming#1"
    assert final == "paraformer-zh-streaming#2"
    assert partial + final == "paraformer-zh-streaming#1paraformer-zh-streaming#2"


async def test_feed_returns_empty_string_when_nothing_decoded() -> None:
    """模型没解出文本时返回空串。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()
    assert await engine.feed(_block()) == ""
    assert await engine.feed(_block()) == ""
    assert await engine.feed(_block()) == "paraformer-zh-streaming#1"


# --------------------------------------------------------------------------
# VAD
# --------------------------------------------------------------------------
async def test_vad_disabled_does_not_create_vad_instance() -> None:
    """VAD 关闭时不创建 VAD 实例。"""
    factory = _FakeFactory()
    engine = _engine(factory, vad_enabled=False)
    await engine.ensure_loaded()

    assert [request["model"] for request in factory.requests] == ["paraformer-zh-streaming"]
    assert engine.has_vad is False


async def test_vad_enabled_creates_its_own_instance() -> None:
    """VAD 开着时单独建一个实例。"""
    factory = _FakeFactory()
    engine = _engine(factory, vad_enabled=True)
    await engine.ensure_loaded()

    assert [request["model"] for request in factory.requests] == [
        "paraformer-zh-streaming",
        "fsmn-vad",
    ]
    assert engine.has_vad is True


async def test_detect_speech_returns_intervals_from_model() -> None:
    """端点检测的区间按模型给的原样返回。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()

    intervals = await engine.detect_speech(_block(count=16000))
    assert intervals == [(0, 1200), (1500, 3200)]

    vad_calls = [model for model in factory.models if model.model == "fsmn-vad"]
    assert vad_calls[0].calls[0]["chunk_size"] == 600


async def test_detect_speech_without_vad_raises() -> None:
    """VAD 没开着时不做端点检测。"""
    factory = _FakeFactory()
    engine = _engine(factory, vad_enabled=False)
    await engine.ensure_loaded()

    with pytest.raises(ValueError):
        await engine.detect_speech(_block(count=16000))


# --------------------------------------------------------------------------
# 设备选择与回退
# --------------------------------------------------------------------------
async def test_auto_device_falls_back_to_cpu_and_logs(
    captured_logs: list[logging.LogRecord],
) -> None:
    """auto 在 CUDA 失败时回退 CPU 并记日志，状态仍是 ready。"""
    factory = _FakeFactory(fail_devices=(DEVICE_CUDA,))
    engine = _engine(factory, device=DEVICE_AUTO)
    await engine.ensure_loaded()

    assert engine.device == DEVICE_CPU
    assert engine.state == STATE_READY
    assert engine.is_ready is True
    # 先试 CUDA 失败，再试 CPU 成功
    assert [request["device"] for request in factory.asr_requests] == [
        DEVICE_CUDA,
        DEVICE_CPU,
    ]

    fallbacks = [
        record
        for record in captured_logs
        if getattr(record, "event", "") == "asr.model.device_fallback"
    ]
    assert fallbacks, "回退时没有写日志"
    fields = getattr(fallbacks[-1], "fields", {})
    assert fields["device"] == DEVICE_CUDA
    assert fields["fallback_device"] == DEVICE_CPU


async def test_auto_device_uses_cuda_when_it_works() -> None:
    """auto 先试 CUDA，CUDA 能用就不回退。"""
    factory = _FakeFactory()
    engine = _engine(factory, device=DEVICE_AUTO)
    await engine.ensure_loaded()

    assert engine.device == DEVICE_CUDA
    assert [request["device"] for request in factory.asr_requests] == [DEVICE_CUDA]


async def test_explicit_cpu_device_never_tries_cuda() -> None:
    """显式指定 CPU 时不试 CUDA。"""
    factory = _FakeFactory()
    engine = _engine(factory, device=DEVICE_CPU)
    await engine.ensure_loaded()

    assert engine.device == DEVICE_CPU
    assert [request["device"] for request in factory.asr_requests] == [DEVICE_CPU]


async def test_model_path_is_used_not_model_dir() -> None:
    """本地模型目录用 model_path 传。"""
    factory = _FakeFactory()
    engine = _engine(factory, model_dir="/tmp/models")
    await engine.ensure_loaded()

    request = factory.requests[0]
    assert request["model_path"] == "/tmp/models/paraformer-zh-streaming"
    assert "model_dir" not in request


# --------------------------------------------------------------------------
# 加载与解码失败
# --------------------------------------------------------------------------
async def test_load_failure_raises_asr_load_error() -> None:
    """模型加载失败抛 AsrLoadError，状态是 error。"""
    factory = _FakeFactory(fail_models=("paraformer-zh-streaming",))
    engine = _engine(factory, device=DEVICE_CPU)

    with pytest.raises(AsrLoadError) as excinfo:
        await engine.ensure_loaded()
    assert excinfo.value.code == "E_ASR_LOAD"
    assert engine.state == STATE_ERROR


async def test_vad_load_failure_keeps_the_asr_instance() -> None:
    """VAD 建不起来时 ASR 的实例不丢，下次只补建 VAD。"""
    factory = _FakeFactory(fail_models=("fsmn-vad",))
    engine = _engine(factory, device=DEVICE_CPU)

    with pytest.raises(AsrLoadError) as excinfo:
        await engine.ensure_loaded()
    assert excinfo.value.code == "E_ASR_LOAD"
    assert engine.state == STATE_ERROR

    # 再来一次：ASR 不重建，只补建 VAD
    factory.fail_models = ()
    await engine.ensure_loaded()

    assert len(factory.asr_requests) == 1
    assert engine.state == STATE_READY


async def test_decode_failure_raises_asr_decode_error() -> None:
    """解码失败抛 AsrDecodeError，错误码是 E_ASR_DECODE。"""

    class _FailingModel(_FakeModel):
        """一调就炸的假模型。"""

        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            """假装解码失败。"""
            raise RuntimeError("模型崩了")

    class _FailingFactory(_FakeFactory):
        """只造会失败的 ASR 模型的工厂。"""

        def __call__(self, *, model: str, model_path: str, device: str) -> _FakeModel:
            """建一个会失败的假模型。"""
            if model == "fsmn-vad":
                return super().__call__(model=model, model_path=model_path, device=device)
            self.requests.append(
                {"model": model, "model_path": model_path, "device": device}
            )
            return _FailingModel(model=model, model_path=model_path, device=device)

    engine = _engine(_FailingFactory(), device=DEVICE_CPU)
    await engine.ensure_loaded()

    # 攒够 600 毫秒触发推理，推理失败
    with pytest.raises(AsrDecodeError) as excinfo:
        for _ in range(3):
            await engine.feed(_block())
    assert excinfo.value.code == "E_ASR_DECODE"
    assert engine.state == STATE_ERROR


# --------------------------------------------------------------------------
# 实例复用
# --------------------------------------------------------------------------
async def test_ensure_loaded_reuses_the_same_instances() -> None:
    """两个实例整个进程生命周期内只初始化一次。"""
    factory = _FakeFactory()
    engine = _engine(factory)

    await engine.ensure_loaded()
    await engine.ensure_loaded()
    await engine.feed(_block())

    assert len(factory.requests) == 2
    assert len(factory.models) == 2


async def test_ensure_loaded_restores_state_after_error() -> None:
    """模型实例还在时再次加载只恢复状态，不重建实例。"""
    factory = _FakeFactory()
    engine = _engine(factory, device=DEVICE_CPU)
    await engine.ensure_loaded()

    engine._state = STATE_ERROR
    await engine.ensure_loaded()

    assert engine.state == STATE_READY
    assert len(factory.requests) == 2


# --------------------------------------------------------------------------
# 缓存长度上限
# --------------------------------------------------------------------------
async def test_stream_cache_is_trimmed_when_over_limit(
    captured_logs: list[logging.LogRecord],
) -> None:
    """一条语音流超过长度上限时，最早的缓存被丢掉并记日志。

    见《5-语音输入与流式ASR》「性能」一节：限制音频缓存长度，
    避免长会议把内存吃满。
    """
    factory = _CacheWritingFactory()
    engine = FunASREngine(
        model_factory=factory,
        device=DEVICE_CPU,
        max_stream_seconds=1,
    )
    await engine.ensure_loaded()

    # 上限 1 秒：喂 8 块 200 毫秒就是 1.6 秒，超了
    for _ in range(8):
        await engine.feed(_block())

    trims = [
        record
        for record in captured_logs
        if getattr(record, "event", "") == "asr.stream.trimmed"
    ]
    assert trims, "超过上限时没有记日志"
    assert getattr(trims[-1], "fields", {})["limit_seconds"] == 1

    # 缓存被换掉了：前面的调用与后面的调用不是同一个字典
    calls = factory.asr_model.calls
    first_cache = calls[0]["cache"]
    assert any(call["cache"] is not first_cache for call in calls)


async def test_short_stream_keeps_the_same_cache() -> None:
    """正常长度的语音流不裁剪缓存，同一条流内共用一个字典。"""
    factory = _CacheWritingFactory()
    engine = FunASREngine(
        model_factory=factory,
        device=DEVICE_CPU,
        max_stream_seconds=60,
    )
    await engine.ensure_loaded()

    for _ in range(9):
        await engine.feed(_block())

    calls = factory.asr_model.calls
    assert len(calls) == 3
    assert calls[0]["cache"] is calls[1]["cache"] is calls[2]["cache"]


async def test_pending_buffer_is_trimmed_when_over_limit(
    captured_logs: list[logging.LogRecord],
) -> None:
    """待解码缓冲超过上限时，最早的音频被丢掉并记日志。"""
    factory = _FakeFactory()
    engine = _engine(factory)
    await engine.ensure_loaded()

    # 一次喂一块 30000 点的音频，超过待解码缓冲的上限
    await engine.feed(_block(count=30000))

    trims = [
        record
        for record in captured_logs
        if getattr(record, "event", "") == "asr.stream.pending_trimmed"
    ]
    assert trims, "待解码缓冲超过上限时没有记日志"
    fields = getattr(trims[-1], "fields", {})
    assert fields["dropped_samples"] == 30000 - MAX_PENDING_SAMPLES
    assert fields["limit_samples"] == MAX_PENDING_SAMPLES
