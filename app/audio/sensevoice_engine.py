"""SenseVoiceSmall 离线重识别引擎。

见《5-语音输入与流式ASR》「重识别与热词纠正」一节：流式识别保留实时
预览，最终文本由本引擎对整段音频重新识别产出，再经拼音级热词纠正
（:mod:`app.audio.hotword_corrector`）后返回给调用方。

设计依据（``scripts/experiments/asr_compare_funasr.py`` 的实验结论）：

- SenseVoiceSmall 整段重识别（``use_itn=True``）质量远高于流式
  paraformer-zh-streaming：145.7 秒音频 0.9 秒出结果，纯汉字 CER 0.0023；
- 与流式引擎同时驻留显存约 3.5 GB，8 GB 显存可容纳；
- 加载失败只告警降级：调用方捕获异常后退回流式文本，主进程不阻断；
- 整段识别无内部状态（每次推理新建 ``cache``），一个进程共享一个实例，
  识别调用在锁内串行执行。

几条硬约束，与 :mod:`app.audio.funasr_engine` 同源：

- 模型目录沿用 ``models/`` 约定，环境变量 ``INTERVIEWMATE_MODELS`` 可覆盖；
- ``AutoModel`` 的 ``model`` 直接传本地目录路径，funasr 会读目录内
  ``config.yaml`` 确定注册名；不写 ``model_dir``，避免权重被拉进
  modelscope 缓存；
- ``use_itn=True`` 打开文本规整，输出带 rich text 标签（``<|zh|>``、
  ``<|NEUTRAL|>``、``<|Speech|>`` 等），统一用 ``<|...|>`` 正则清掉；
- 设备取 ``auto`` 时先试 CUDA，失败回退 CPU 并记日志。

模型加载与推理都是阻塞调用，放到线程里跑，事件循环不被占住。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.audio.funasr_engine import (
    DEFAULT_SAMPLE_RATE,
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    STATE_ERROR,
    STATE_LOADING,
    STATE_READY,
)
from app.audio.hotword_corrector import HotwordCorrector, load_default_corrector
from app.config import current_config
from app.core.exceptions import AsrDecodeError, AsrLoadError
from app.utils.logger import log_event

__all__ = [
    "SenseVoiceEngine",
    "ModelFactory",
    "DEFAULT_SENSEVOICE_MODEL_NAME",
    "DEFAULT_SENSEVOICE_VAD_MODEL_NAME",
    "RICH_TEXT_RE",
    "DEFAULT_VAD_SEGMENT_MS",
    "SENSEVOICE_BATCH_SIZE_S",
    "SENSEVOICE_MERGE_LENGTH_S",
    "strip_rich_text",
]


# 默认模型目录名，与仓库 models/ 下的目录一致
DEFAULT_SENSEVOICE_MODEL_NAME: str = "SenseVoiceSmall"

# VAD 目录名：整段识别用 VAD 切长音频，与流式引擎共用同一份 fsmn-vad
DEFAULT_SENSEVOICE_VAD_MODEL_NAME: str = "fsmn-vad"

# VAD 单段时长上限（毫秒），实验取值
DEFAULT_VAD_SEGMENT_MS: int = 30000

# 整段推理的固定参数，见 scripts/experiments/asr_compare_funasr.py
SENSEVOICE_BATCH_SIZE_S: int = 60
SENSEVOICE_MERGE_LENGTH_S: int = 30

# rich text 标签：<|zh|>、<|NEUTRAL|>、<|Speech|> 这类成对竖线标记
RICH_TEXT_RE: re.Pattern[str] = re.compile(r"<\|[^|]*\|>")


class ModelFactory(Protocol):
    """模型工厂的调用约定。

    工厂拿到模型目录、VAD 目录与设备，返回一个带 ``generate`` 方法的
    模型对象。默认实现是 funasr 的 ``AutoModel``；测试里注入假模型，
    单元测试不做真实推理。
    """

    def __call__(
        self, *, model_path: str, vad_model_path: str, device: str
    ) -> Any:
        """按参数建一个模型实例。"""
        ...


def strip_rich_text(text: str) -> str:
    """清掉 SenseVoice 输出里的 rich text 标签。

    :param text: 模型原始输出。
    :returns: 去掉 ``<|...|>`` 标签并去除首尾空白的文本。
    """
    return RICH_TEXT_RE.sub("", text).strip()


def _extract_text(result: Any) -> str:
    """从 ``generate`` 的返回值里取文本。

    funasr 返回 ``[{"key": ..., "text": ...}]`` 这种形状，
    这里同时兼容字典、字典列表与空返回。

    :param result: ``generate`` 的返回值。
    :returns: 这一次识别出的文本，没有就是空串。
    """
    if isinstance(result, dict):
        items = [result]
    elif isinstance(result, (list, tuple)):
        items = [item for item in result if isinstance(item, dict)]
    else:
        return ""
    parts: list[str] = []
    for item in items:
        text = item.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _to_wave(audio: Any) -> "np.ndarray":
    """把输入规整成 16000 Hz float32 的一维波形。

    :param audio: 波形数组，或 int16 PCM 的字节串。
    :returns: float32 一维数组。
    """
    if isinstance(audio, (bytes, bytearray, memoryview)):
        # int16 PCM 字节：按小端解成 float32 并归一化
        return np.frombuffer(bytes(audio), dtype="<i2").astype("float32") / 32768.0
    array = np.asarray(audio, dtype="float32")
    if array.ndim > 1:
        # 多声道按最后一维取平均，下混成单声道
        array = array.mean(axis=-1)
    return np.ascontiguousarray(array.reshape(-1), dtype="float32")


class SenseVoiceEngine:
    """SenseVoiceSmall 离线重识别引擎。

    对外两个动作：:meth:`ensure_loaded` 加载模型、:meth:`recognize`
    整段识别并做热词纠正。识别失败抛 :class:`AsrDecodeError`，
    加载失败抛 :class:`AsrLoadError`，都不吞异常，
    降级（退回流式文本）由调用方决定。

    :param model_dir: 本地模型目录，默认仓库根 ``models/``，
        目录下的模型按 ``<目录名>`` 命名。
    :param model_name: 重识别模型目录名，默认 ``SenseVoiceSmall``。
    :param vad_model_name: VAD 目录名，默认 ``fsmn-vad``。
    :param device: 设备选择，取 ``auto``、``cuda`` 或 ``cpu``。
        ``auto`` 先试 CUDA，初始化失败回退 CPU。
    :param model_factory: 模型工厂，默认用 funasr 的 ``AutoModel``。
        测试里注入假模型，单元测试不做真实推理。
    :param init_timeout: 模型加载的超时秒数。None 时取进程里加载好的
        配置里的 ``asr.init_timeout``。
    :param corrector: 热词纠正器，None 时从默认路径
        （``config/hotwords.txt``）加载。
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        model_name: str = DEFAULT_SENSEVOICE_MODEL_NAME,
        vad_model_name: str = DEFAULT_SENSEVOICE_VAD_MODEL_NAME,
        device: str = DEVICE_AUTO,
        model_factory: ModelFactory | None = None,
        init_timeout: float | None = None,
        corrector: HotwordCorrector | None = None,
    ) -> None:
        import os

        from app.config import REPO_ROOT

        if model_dir is not None:
            self._model_dir = Path(model_dir)
        else:
            # 与主链路 models/ 约定一致，INTERVIEWMATE_MODELS 可覆盖
            self._model_dir = Path(
                os.environ.get("INTERVIEWMATE_MODELS") or str(REPO_ROOT / "models")
            )
        self._model_name = model_name
        self._vad_model_name = vad_model_name
        self._device_request = device
        self._model_factory = model_factory
        self._init_timeout = init_timeout
        self._corrector = corrector if corrector is not None else load_default_corrector()

        self._model: Any = None
        self._device: str | None = None
        self._state: str = STATE_LOADING
        # 加载任务句柄：加载是阻塞调用，跑在线程里，超时后不重复加载
        self._load_task: asyncio.Task[None] | None = None
        # 识别互斥：引擎实例跨语音链路与项目模式共用，推理串行执行
        self._recognize_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """模型是不是已经加载好，可以识别了。"""
        return self._state == STATE_READY

    @property
    def state(self) -> str:
        """引擎状态，取 ``loading``、``ready`` 或 ``error``。"""
        return self._state

    @property
    def device(self) -> str | None:
        """实际用的设备，加载成功后是 ``cuda`` 或 ``cpu``。"""
        return self._device

    @property
    def model_dir(self) -> Path:
        """本地模型目录。"""
        return self._model_dir

    @property
    def corrector(self) -> HotwordCorrector:
        """热词纠正器。"""
        return self._corrector

    @property
    def init_timeout(self) -> float:
        """模型加载的超时秒数。

        构造时没给就从进程里加载好的配置里取 ``asr.init_timeout``，
        配置还没加载时用默认 60 秒。
        """
        if self._init_timeout is not None:
            return float(self._init_timeout)
        config = current_config()
        if config is not None:
            return float(config.asr.init_timeout)
        return 60.0

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    async def ensure_loaded(self) -> None:
        """加载模型。

        已经加载过就直接返回，``AutoModel`` 实例整个进程生命周期内
        只创建一次。加载失败抛 :class:`AsrLoadError`，状态置成
        ``error``；调用方按降级处理，主进程不阻断。

        加载是阻塞调用，放在线程里跑，事件循环不被占住；
        超过 ``asr.init_timeout`` 秒按加载失败处理。

        :raises AsrLoadError: 模型加载失败，或加载超时。
        """
        if self._model is not None:
            self._state = STATE_READY
            return
        self._state = STATE_LOADING
        model_path = self._model_dir / self._model_name
        log_event(
            "sensevoice.model.loading",
            "开始加载重识别模型",
            device=self._device_request,
            model=self._model_name,
            model_path=str(model_path),
        )
        if not model_path.is_dir():
            log_event(
                "sensevoice.model.path_missing",
                f"本地模型目录不存在：{model_path}（模型 {self._model_name}）",
                level="WARNING",
                model=self._model_name,
                model_path=str(model_path),
            )
        timeout = self.init_timeout
        task = self._load_task
        if task is None or task.done():
            task = asyncio.ensure_future(asyncio.to_thread(self._build_model))
            self._load_task = task
        try:
            # shield：超时只中断这次等待，不动后台那个加载任务
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except asyncio.TimeoutError as exc:
            self._state = STATE_ERROR
            log_event(
                "sensevoice.model.load_failed",
                f"重识别模型加载超时（{timeout:g} 秒）",
                level="ERROR",
                model=self._model_name,
            )
            raise AsrLoadError(f"加载重识别模型超时（{timeout:g} 秒）") from exc
        except Exception as exc:
            self._state = STATE_ERROR
            log_event(
                "sensevoice.model.load_failed",
                f"重识别模型加载失败：{exc}",
                level="ERROR",
                error=str(exc),
                model=self._model_name,
            )
            raise AsrLoadError(f"加载重识别模型失败：{exc}") from exc
        self._state = STATE_READY
        log_event(
            "sensevoice.model.ready",
            "重识别模型加载完成",
            device=self._device,
            model=self._model_name,
        )

    def _build_model(self) -> None:
        """建模型实例，``auto`` 时 CUDA 失败回退 CPU。

        :raises AsrLoadError: 所有候选设备都失败。
        """
        candidates = (
            [DEVICE_CUDA, DEVICE_CPU] if self._device_request == DEVICE_AUTO
            else [self._device_request]
        )
        last_error: Exception | None = None
        for index, candidate in enumerate(candidates):
            try:
                self._model = self._build_model_instance(candidate)
            except Exception as exc:
                last_error = exc
                if index + 1 >= len(candidates):
                    break
                fallback = candidates[index + 1]
                log_event(
                    "sensevoice.model.device_fallback",
                    f"{candidate} 初始化失败，回退到 {fallback}：{exc}",
                    level="WARNING",
                    device=candidate,
                    fallback_device=fallback,
                    model=self._model_name,
                    reason=str(exc),
                )
                continue
            self._device = candidate
            return
        raise AsrLoadError(
            f"加载重识别模型 {self._model_name} 失败：{last_error}"
        ) from last_error

    def _build_model_instance(self, device: str) -> Any:
        """按设备建一个模型实例。

        ``model`` 直接传本地目录路径，funasr 读目录内 ``config.yaml``
        确定注册名；VAD 模型一并传入（目录不存在时 funasr 会报错，
        加载失败由降级路径兜住）。

        :param device: 目标设备。
        :returns: 模型实例。
        """
        model_path = self._model_dir / self._model_name
        vad_model_path = self._model_dir / self._vad_model_name
        if self._model_factory is not None:
            return self._model_factory(
                model_path=str(model_path),
                vad_model_path=str(vad_model_path),
                device=device,
            )
        # funasr 的导入很慢，放到这里按需导入
        from funasr import AutoModel

        return AutoModel(
            model=str(model_path),
            vad_model=str(vad_model_path),
            vad_kwargs={"max_single_segment_time": DEFAULT_VAD_SEGMENT_MS},
            device=device,
            trust_remote_code=True,
            disable_pbar=True,
        )

    # ------------------------------------------------------------------
    # 整段识别
    # ------------------------------------------------------------------
    async def recognize(
        self, audio: Any, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> str:
        """整段识别一段音频，返回热词纠正后的最终文本。

        波形规整 → 整段推理 → 清 rich text 标签 → 热词纠正。
        识别异常照抛（:class:`AsrDecodeError`），由调用方降级；
        空音频直接返回空串，不调模型。

        :param audio: 波形（float32 数组，多声道自动下混），
            或 int16 PCM 的字节串。
        :param sample_rate: 音频采样率，模型按 16000 Hz 走，
            其他采样率只记日志提醒，不做重采样。
        :returns: 最终文本，模型没识别出内容时是空串。
        :raises AsrDecodeError: 模型调用失败。
        :raises AsrLoadError: 模型还没加载且加载失败。
        """
        if sample_rate != DEFAULT_SAMPLE_RATE:
            log_event(
                "sensevoice.sample_rate_mismatch",
                f"音频采样率是 {sample_rate}，模型按 {DEFAULT_SAMPLE_RATE} 走",
                level="WARNING",
                sample_rate=sample_rate,
                expected=DEFAULT_SAMPLE_RATE,
            )
        await self.ensure_loaded()
        wave = _to_wave(audio)
        if wave.size == 0:
            return ""
        async with self._recognize_lock:
            # 推理是阻塞调用，放到线程里跑，事件循环不被占住
            raw = await asyncio.to_thread(self._generate, wave)
        text = strip_rich_text(raw)
        return self._corrector.correct_text(text)

    def _generate(self, wave: "np.ndarray") -> str:
        """跑一次整段推理，返回模型原始输出文本。

        :param wave: float32 的一维波形。
        :returns: 模型原始输出（可能带 rich text 标签）。
        :raises AsrDecodeError: 模型调用失败。
        """
        try:
            result = self._model.generate(
                input=wave,
                cache={},
                language="zh",
                use_itn=True,
                batch_size_s=SENSEVOICE_BATCH_SIZE_S,
                merge_vad=True,
                merge_length_s=SENSEVOICE_MERGE_LENGTH_S,
                disable_pbar=True,
            )
        except Exception as exc:
            self._state = STATE_ERROR
            log_event(
                "sensevoice.recognize.failed",
                f"重识别失败：{exc}",
                level="ERROR",
                error=str(exc),
            )
            raise AsrDecodeError(f"重识别失败：{exc}") from exc
        self._state = STATE_READY
        return _extract_text(result)

    async def correct_text(self, text: str) -> str:
        """只做热词纠正，不碰音频。

        重识别降级到流式文本时用这个方法：流式结果同样过一遍热词纠正，
        口径与重识别路径一致。

        :param text: 流式识别文本。
        :returns: 纠正后的文本。
        """
        async with self._recognize_lock:
            return self._corrector.correct_text(text)