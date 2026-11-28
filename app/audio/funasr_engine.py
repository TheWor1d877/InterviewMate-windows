"""FunASR 本地流式识别引擎。

见《5-语音输入与流式ASR》「FunASR 模型」「流式推理」「语音端点检测」
与「部分结果与最终结果」四节。

几条硬约束：

- ASR 与 VAD 是两个独立的 ``AutoModel`` 实例，本地模型目录用 ``model_path`` 传。
  不能写 ``model_dir``：funasr 只在 kwargs 里存在 ``model_path`` 时才跳过 hub 下载，
  ``model_dir`` 不参与这个判断，权重会被重新拉进 modelscope 缓存。
- 两个实例整个进程生命周期内只初始化一次，之后一直复用同一个实例。
- 采集回调出的 200 毫秒块攒满 200 毫秒（3200 点）就送模型，
  一次调用只传一段波形，``batch_size`` 固定为 1。
- 每条语音流一个 ``cache`` 字典，同一条流内所有块共用；
  冲刷完丢掉这个字典，下一条语音流重新建一个。
- 设备取 ``auto`` 时先试 CUDA，显存不足或初始化失败回退 CPU 并记日志，
  回退后引擎状态仍然是 ready。

模型加载与推理都是阻塞调用，这里放到线程里跑，事件循环不被占住。
模型加载还有一层超时：超过 ``asr.init_timeout`` 秒就当加载失败，
见《8-集成与验收》「超时管理」一节。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from app.config import current_config
from app.core.exceptions import AsrDecodeError, AsrLoadError
from app.utils.logger import log_event
from app.utils.timing import default_recorder, timed

if TYPE_CHECKING:
    from app.audio.audio_capture import AudioBlock

__all__ = [
    "FunASREngine",
    "ModelFactory",
    "MODEL_REGISTRY_NAMES",
    "resolve_model_names",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_ASR_MODEL_NAME",
    "DEFAULT_VAD_MODEL_NAME",
    "DEFAULT_SAMPLE_RATE",
    "MODEL_CHUNK_DURATION_MS",
    "MODEL_CHUNK_SAMPLES",
    "MAX_STREAM_DURATION_SECONDS",
    "MAX_PENDING_SAMPLES",
    "MAX_RECORDING_SAMPLES",
    "STREAM_CHUNK_SIZE",
    "ENCODER_CHUNK_LOOK_BACK",
    "DECODER_CHUNK_LOOK_BACK",
    "BATCH_SIZE",
    "VAD_CHUNK_MS",
    "DEVICE_AUTO",
    "DEVICE_CUDA",
    "DEVICE_CPU",
    "DEFAULT_INIT_TIMEOUT",
    "STATE_LOADING",
    "STATE_READY",
    "STATE_ERROR",
]


# 本地模型目录，默认 ~/models，见《5-语音输入与流式ASR》「FunASR 模型」一节
DEFAULT_MODEL_DIR: Path = Path.home() / "models"

# 默认模型名：ASR 用 paraformer 流式检查点，VAD 用 fsmn-vad
DEFAULT_ASR_MODEL_NAME: str = "paraformer-zh-streaming"
DEFAULT_VAD_MODEL_NAME: str = "fsmn-vad"

# 本地模型目录名（ModelScope 仓库名）→ FunASR 注册表名。
#
# 这两个名字不是一回事：目录名是仓库名，注册表名才是
# funasr.register.tables 里登记的类名，也就是模型目录 ``config.yaml``
# 首行 ``model:`` 的取值。把目录名原样传给 ``AutoModel(model=...)``
# 会报 ``model '...' is not registered``。
MODEL_REGISTRY_NAMES: dict[str, str] = {
    "paraformer-zh-streaming": "ParaformerStreaming",
    "fsmn-vad": "FsmnVADStreaming",
}

# 反过来查：注册表名 → 本地目录名，允许配置直接写注册表名
_MODEL_DIR_NAMES: dict[str, str] = {
    registry: dir_name for dir_name, registry in MODEL_REGISTRY_NAMES.items()
}

# 采样率，采集层与模型都按 16000 Hz 走，
# 见《5-语音输入与流式ASR》「采集参数」一节
DEFAULT_SAMPLE_RATE: int = 16000

# 送进模型的块长：200 毫秒 = 16000 × 0.2 = 3200 点，
# 见《接口契约》「FunASR 音频块参数」一节。
# 原为 600 毫秒（9600 点）：聚合等待占出字延迟约 87%（见 WIN-MILESTONES M9），
# 下调到 200 毫秒换取更快出字，推理频次 ×3 后 GPU 占空比仍约 10%，
# 模型内部的 chunk_size 窗口不受影响，见《接口契约》「FunASR 音频块参数」一节
MODEL_CHUNK_DURATION_MS: int = 200
MODEL_CHUNK_SAMPLES: int = 3200

# 一条语音流的最长时长上限。超过就丢掉最早的缓存重新建一个，
# 见《5-语音输入与流式ASR》「性能」一节的缓存长度限制
MAX_STREAM_DURATION_SECONDS: int = 300

# 待解码缓冲的点数上限。攒不够 200 毫秒时音频先放这里，
# 超过上限就把最早的丢掉
MAX_PENDING_SAMPLES: int = MODEL_CHUNK_SAMPLES * 2

# 整段录音缓存的点数上限，与语音流上限同源（300 秒）：
# 为 SenseVoice 重识别缓存当前录音会话的整段 PCM，
# 见《5-语音输入与流式ASR》「重识别与热词纠正」一节。
# 超过上限丢最早的音频，重识别只覆盖最近上限时长内的内容
MAX_RECORDING_SAMPLES: int = MAX_STREAM_DURATION_SECONDS * DEFAULT_SAMPLE_RATE

# 流式推理的固定参数组合，见《5-语音输入与流式ASR》「流式推理」一节
STREAM_CHUNK_SIZE: list[int] = [0, 10, 5]
ENCODER_CHUNK_LOOK_BACK: int = 4
DECODER_CHUNK_LOOK_BACK: int = 1
BATCH_SIZE: int = 1

# VAD 的 chunk_size 取整数毫秒，实测 600，
# 见《接口契约》「FunASR 流式推理调用」一节
VAD_CHUNK_MS: int = 600

# 设备取值，见《5-语音输入与流式ASR》「FunASR 模型」一节
DEVICE_AUTO: str = "auto"
DEVICE_CUDA: str = "cuda"
DEVICE_CPU: str = "cpu"

# ASR 初始化超时的默认秒数：进程里没加载过配置时用它，
# 见《8-集成与验收》「超时管理」一节
DEFAULT_INIT_TIMEOUT: float = 60.0

# 引擎状态：加载中、就绪、失败
STATE_LOADING: str = "loading"
STATE_READY: str = "ready"
STATE_ERROR: str = "error"

# 指标名，见《2-骨架与配置》性能指标一节
ASR_LATENCY_METRIC: str = "asr_latency"
ASR_FINAL_LATENCY_METRIC: str = "asr_final_latency"


class ModelFactory(Protocol):
    """模型工厂的调用约定。

    工厂拿到模型名、本地目录与设备，返回一个带 ``generate`` 方法的模型对象。
    默认实现是 funasr 的 ``AutoModel``；测试里注入假模型，
    这样单元测试不用真加载权重、不做真实推理。
    """

    def __call__(self, *, model: str, model_path: str, device: str) -> Any:
        """按参数建一个模型实例。"""
        ...


def _to_float32_1d(samples: Any) -> "np.ndarray":
    """转成 float32 的一维波形。

    采集层给的已经是 16000 Hz 单声道 float32，
    这里再兜一层：多声道先下混成单声道，再压成一维。

    :param samples: 波形，可以是数组也可以是序列。
    :returns: float32 的一维数组。
    """
    array = np.asarray(samples, dtype="float32")
    if array.ndim > 1:
        # 多声道按最后一维取平均，下混成单声道
        array = array.mean(axis=-1)
    return np.ascontiguousarray(array.reshape(-1), dtype="float32")


def _extract_text(result: Any) -> str:
    """从 ``generate`` 的返回值里取文本。

    funasr 返回的是 ``[{"key": ..., "text": ...}]`` 这种形状，
    这里同时兼容字典、字典列表与空返回。

    :param result: ``generate`` 的返回值。
    :returns: 这一段解码出的文本，没有就是空串。
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


def _extract_intervals(result: Any) -> list[tuple[int, int]]:
    """从 VAD 的返回值里取语音区间。

    区间边界由模型给出，这里原样返回，不自行修正。

    :param result: ``generate`` 的返回值，语音区间在 ``value`` 字段里。
    :returns: 毫秒区间的列表。
    """
    if isinstance(result, dict):
        items = [result]
    elif isinstance(result, (list, tuple)):
        items = [item for item in result if isinstance(item, dict)]
    else:
        return []
    intervals: list[tuple[int, int]] = []
    for item in items:
        values = item.get("value")
        if not isinstance(values, (list, tuple)):
            continue
        for pair in values:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                intervals.append((int(pair[0]), int(pair[1])))
    return intervals


def resolve_model_names(model_name: str) -> tuple[str, str]:
    """把模型名解析成「注册表名」与「本地目录名」。

    ``asr.model`` 允许两种写法，两种都能用：

    - 目录名，例如 ``paraformer-zh-streaming``。旧的默认值，兼容保留；
    - 注册表名，例如 ``ParaformerStreaming``。

    都不认识的取值原样返回两遍，目录名与注册表名都按它自己算，
    这样用户放一个自定义目录也能用。

    :param model_name: 配置或调用方给的模型名。
    :returns: ``(注册表名, 本地目录名)``。
    """
    if model_name in MODEL_REGISTRY_NAMES:
        return MODEL_REGISTRY_NAMES[model_name], model_name
    if model_name in _MODEL_DIR_NAMES:
        return model_name, _MODEL_DIR_NAMES[model_name]
    return model_name, model_name


def _drop_load_result(task: "asyncio.Future[None]") -> None:
    """加载任务结束时把异常取出来。

    超时之后没人再等这个任务，异常留在任务里会在任务被回收时打一条
    「异常没人取」的警告。这里提前取出来，警告不会出现；
    下次等同一个任务时异常照样抛出来。

    :param task: 加载任务。
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    log_event(
        "asr.model.load_abandoned",
        f"后台的模型加载最后失败：{exc}",
        level="WARNING",
        error=str(exc),
    )


class FunASREngine:
    """本地流式 ASR 引擎。

    对外三个动作：:meth:`ensure_loaded` 加载模型、:meth:`feed` 喂音频块、
    :meth:`flush` 停止时冲刷。:meth:`feed` 返回部分结果，
    :meth:`flush` 返回最终结果，两者分开，调用方按顺序拼接成整句。

    :param model_dir: 本地模型目录，默认 ``~/models``，
        目录下的模型按 ``<模型名>`` 命名。
    :param asr_model: ASR 模型名，默认 ``paraformer-zh-streaming``。
    :param vad_model: VAD 模型名，默认 ``fsmn-vad``。
    :param device: 设备选择，取 ``auto``、``cuda`` 或 ``cpu``。
        ``auto`` 先试 CUDA，初始化失败回退 CPU。
    :param vad_enabled: 是否启用 VAD。为假时不创建 VAD 实例。
    :param model_factory: 模型工厂，默认用 funasr 的 ``AutoModel``。
        测试里注入假模型，单元测试不做真实推理。
    :param max_stream_seconds: 一条语音流的最长时长上限，单位秒。
        超过就丢掉最早的缓存重新建一个，长会议不会把内存吃满。
    :param init_timeout: 模型加载的超时秒数。None 时取进程里加载好的配置里的
        ``asr.init_timeout``，见《8-集成与验收》「超时管理」一节。
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        asr_model: str = DEFAULT_ASR_MODEL_NAME,
        vad_model: str = DEFAULT_VAD_MODEL_NAME,
        device: str = DEVICE_AUTO,
        vad_enabled: bool = True,
        model_factory: ModelFactory | None = None,
        max_stream_seconds: int = MAX_STREAM_DURATION_SECONDS,
        init_timeout: float | None = None,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        self._asr_model_name = asr_model
        self._vad_model_name = vad_model
        self._device_request = device
        self._vad_enabled = bool(vad_enabled)
        self._model_factory = model_factory
        self._max_stream_seconds = max(1, int(max_stream_seconds))
        self._max_stream_samples = self._max_stream_seconds * DEFAULT_SAMPLE_RATE
        self._init_timeout = init_timeout
        # 模型加载任务：加载是阻塞调用，跑在线程里，这里留着任务句柄，
        # 超时之后不用重复加载，见 _load_with_timeout
        self._load_task: asyncio.Task[None] | None = None

        self._asr: Any = None
        self._vad: Any = None
        self._device: str | None = None
        # 模型还没加载成功之前是 loading
        self._state: str = STATE_LOADING
        # 当前语音流的 cache 字典，同一条流内所有块共用这一个
        self._cache: dict[str, Any] = {}
        # 还没攒够 200 毫秒的音频点
        self._pending: list["np.ndarray"] = []
        self._pending_size: int = 0
        # 待解码缓冲最早一块的到来时刻，解码触发时用来报缓冲等待
        self._pending_since: float | None = None
        # 整段录音缓存：当前录音会话的全部 PCM，松开按键后取走交给
        # SenseVoice 重识别，见《5-语音输入与流式ASR》「重识别与热词纠正」
        self._recording: list["np.ndarray"] = []
        self._recording_samples: int = 0
        # 当前语音流已经调过几次模型，用来判断冲刷时要不要再调一次
        self._calls: int = 0
        # 当前语音流累计进来的音频点数，用来判断要不要裁剪缓存
        self._stream_samples: int = 0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """模型是不是已经加载好，可以喂音频了。"""
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
    def vad_enabled(self) -> bool:
        """VAD 是不是开着。"""
        return self._vad_enabled

    @property
    def has_vad(self) -> bool:
        """VAD 实例是不是已经建好。"""
        return self._vad is not None

    @property
    def pending_samples(self) -> int:
        """待解码缓冲里的点数。"""
        return self._pending_size

    @property
    def stream_calls(self) -> int:
        """当前语音流已经调过几次模型。"""
        return self._calls

    @property
    def has_cached_state(self) -> bool:
        """当前语音流的缓存是不是非空。

        冲刷之后这里应该是假：缓存丢掉之后下一条语音流重新建一个。
        """
        return bool(self._cache)

    @property
    def init_timeout(self) -> float:
        """模型加载的超时秒数。

        构造时没给就从进程里加载好的配置里取 ``asr.init_timeout``，
        配置还没加载时用 :data:`DEFAULT_INIT_TIMEOUT`，
        见《8-集成与验收》「超时管理」一节。
        """
        if self._init_timeout is not None:
            return float(self._init_timeout)
        config = current_config()
        if config is not None:
            return float(config.asr.init_timeout)
        return DEFAULT_INIT_TIMEOUT

    @property
    def is_loading(self) -> bool:
        """后台还有没有没结束的模型加载任务。

        加载超时之后加载线程可能还在跑，这里能看出还有没有加载在进行，
        见《8-集成与验收》「超时管理」一节。
        """
        task = self._load_task
        return task is not None and not task.done()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    async def ensure_loaded(self) -> None:
        """加载模型。

        已经加载过就直接返回，两个 ``AutoModel`` 实例整个进程生命周期内
        只创建一次。加载失败抛 :class:`AsrLoadError`，
        状态置成 ``error``，不挡住程序启动。

        加载是阻塞调用，放在线程里跑，事件循环不被占住，
        见《8-集成与验收》「异步任务」一节；超过 ``asr.init_timeout`` 秒
        就中断这次等待，按加载失败处理，
        见《8-集成与验收》「超时管理」一节。

        :raises AsrLoadError: 模型加载失败，或加载超时。
        """
        if self._instances_ready():
            # 实例还在，只把状态恢复成 ready
            self._state = STATE_READY
            return

        self._state = STATE_LOADING
        log_event(
            "asr.model.loading",
            "开始加载 ASR 模型",
            device=self._device_request,
            model=self._asr_model_name,
            model_path=str(self._model_dir / self._asr_model_name),
            vad_enabled=self._vad_enabled,
        )
        try:
            await self._load_with_timeout()
        except AsrLoadError as exc:
            self._state = STATE_ERROR
            log_event(
                "asr.model.load_failed",
                f"ASR 模型加载失败：{exc.message}",
                level="ERROR",
                error=exc.message,
                device=self._device_request,
                model=self._asr_model_name,
            )
            raise
        except Exception as exc:
            self._state = STATE_ERROR
            log_event(
                "asr.model.load_failed",
                f"ASR 模型加载失败：{exc}",
                level="ERROR",
                error=str(exc),
                device=self._device_request,
                model=self._asr_model_name,
            )
            raise AsrLoadError(f"加载 ASR 模型失败：{exc}") from exc

        self._state = STATE_READY
        log_event(
            "asr.model.ready",
            "ASR 模型加载完成",
            device=self._device,
            model=self._asr_model_name,
            vad_enabled=self._vad_enabled,
            vad_loaded=self._vad is not None,
        )

    async def _load_with_timeout(self) -> None:
        """在超时时间内把模型加载出来。

        模型加载是阻塞调用，放到线程里跑，事件循环不被占住，
        见《8-集成与验收》「异步任务」一节。超时按 ``asr.init_timeout`` 算，
        见《8-集成与验收》「超时管理」一节。

        超时了线程也停不下来（Python 杀不掉线程）：加载任务留着，
        下次 :meth:`ensure_loaded` 接着等同一个任务，不重复加载；
        任务结束时的异常在这里取出来，不留悬挂任务。

        :raises AsrLoadError: 加载超时，或加载本身失败。
        """
        timeout = self.init_timeout
        task = self._load_task
        if task is None or task.done():
            task = asyncio.ensure_future(asyncio.to_thread(self._build_models))
            task.add_done_callback(_drop_load_result)
            self._load_task = task
        try:
            # shield：超时只中断这次等待，不动后台那个加载任务
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except asyncio.TimeoutError as exc:
            raise AsrLoadError(f"加载 ASR 模型超时（{timeout:g} 秒）") from exc

    def _build_models(self) -> None:
        """建 ASR 与 VAD 两个实例。

        已经建好的实例直接复用，不重建：VAD 建不起来时，
        ASR 的实例也不丢，下次只补建 VAD。

        :raises AsrLoadError: 模型建不起来。
        """
        if self._asr is None:
            self._asr, self._device = self._load_asr_model()
        if self._vad_enabled and self._vad is None:
            self._vad = self._load_vad_model(self._device or DEVICE_CPU)

    def _load_asr_model(self) -> tuple[Any, str]:
        """建 ASR 实例，``auto`` 时 CUDA 失败就回退 CPU。

        :returns: 模型实例与实际用的设备。
        :raises AsrLoadError: 所有候选设备都失败。
        """
        candidates = self._device_candidates()
        for index, candidate in enumerate(candidates):
            try:
                model = self._build_model(self._asr_model_name, candidate)
            except Exception as exc:
                has_next = index + 1 < len(candidates)
                if not has_next:
                    raise AsrLoadError(
                        f"加载 ASR 模型 {self._asr_model_name} 失败：{exc}"
                    ) from exc
                fallback = candidates[index + 1]
                log_event(
                    "asr.model.device_fallback",
                    f"{candidate} 初始化失败，回退到 {fallback}：{exc}",
                    level="WARNING",
                    device=candidate,
                    fallback_device=fallback,
                    model=self._asr_model_name,
                    reason=str(exc),
                )
                continue
            return model, candidate
        raise AsrLoadError(f"加载 ASR 模型 {self._asr_model_name} 失败")

    def _load_vad_model(self, device: str) -> Any:
        """建 VAD 实例，设备不可用时同样回退 CPU。

        :param device: ASR 实例实际用的设备。
        :returns: VAD 模型实例。
        :raises AsrLoadError: VAD 实例建不起来。
        """
        try:
            return self._build_model(self._vad_model_name, device)
        except Exception as exc:
            if device != DEVICE_CUDA:
                raise AsrLoadError(
                    f"加载 VAD 模型 {self._vad_model_name} 失败：{exc}"
                ) from exc
            log_event(
                "asr.model.device_fallback",
                f"{device} 初始化失败，VAD 回退到 {DEVICE_CPU}：{exc}",
                level="WARNING",
                device=device,
                fallback_device=DEVICE_CPU,
                model=self._vad_model_name,
                reason=str(exc),
            )
            try:
                return self._build_model(self._vad_model_name, DEVICE_CPU)
            except Exception as cpu_exc:
                raise AsrLoadError(
                    f"加载 VAD 模型 {self._vad_model_name} 失败：{cpu_exc}"
                ) from cpu_exc

    def _device_candidates(self) -> list[str]:
        """按设备选择给出候选设备，按尝试顺序排。

        ``auto`` 先试 CUDA 再试 CPU；显式指定的设备只有它自己。
        """
        if self._device_request == DEVICE_AUTO:
            return [DEVICE_CUDA, DEVICE_CPU]
        return [self._device_request]

    def _build_model(self, model_name: str, device: str) -> Any:
        """建一个模型实例。

        ``model_name`` 可以写目录名也可以写注册表名，先用
        :func:`resolve_model_names` 拆成两样再分别用：注册表名给 ``model``，
        目录名拼进 ``model_path``。

        本地目录用 ``model_path`` 传，不能写 ``model_dir``：
        funasr 只在 kwargs 里存在 ``model_path`` 时才跳过 hub 下载，
        ``model_dir`` 不参与这个判断，权重会被重新拉进 modelscope 缓存。

        ASR 实例不带 ``vad_model``：带了之后 ``generate`` 会走
        ``inference_with_vad``，把列表形式的 ``chunk_size`` 原样透传给 VAD，
        而两种模型的 ``chunk_size`` 类型互斥，会抛 TypeError。
        """
        registry_model, dir_name = resolve_model_names(model_name)
        model_path = self._model_dir / dir_name
        if not model_path.is_dir():
            # 目录不在时 funasr 读不到本地 config.yaml，注册名不会被纠正，
            # 最后只会报 model '...' is not registered。先把实际路径写出来，
            # 免得只看到一句「未注册」不知道模型该放哪。
            log_event(
                "asr.model.path_missing",
                f"本地模型目录不存在：{model_path}（模型 {model_name}）",
                level="WARNING",
                model=model_name,
                registry_model=registry_model,
                model_path=str(model_path),
            )
        if self._model_factory is not None:
            return self._model_factory(
                model=registry_model, model_path=str(model_path), device=device
            )
        # funasr 的导入很慢，放到这里按需导入
        from funasr import AutoModel

        return AutoModel(model=registry_model, model_path=str(model_path), device=device)

    def _instances_ready(self) -> bool:
        """模型实例是不是已经建好了。"""
        if self._asr is None:
            return False
        if self._vad_enabled and self._vad is None:
            return False
        return True

    def _check_sample_rate(self, block: Any) -> None:
        """块的采样率不是 16000 Hz 时记一条日志。

        块长按 16000 Hz 算出来的，采样率不对时 3200 点就不是 200 毫秒，
        这里只记日志提醒，不改块的内容。
        """
        sample_rate = getattr(block, "sample_rate", None)
        if sample_rate is None or sample_rate == DEFAULT_SAMPLE_RATE:
            return
        log_event(
            "asr.block.sample_rate_mismatch",
            f"音频块采样率是 {sample_rate}，模型按 {DEFAULT_SAMPLE_RATE} 走",
            level="WARNING",
            sample_rate=sample_rate,
            expected=DEFAULT_SAMPLE_RATE,
        )

    # ------------------------------------------------------------------
    # 流式推理
    # ------------------------------------------------------------------
    async def feed(self, block: "AudioBlock | Any") -> str:
        """喂一块 200 毫秒的音频，返回这次解码出的增量文本。

        块里不够 200 毫秒就先攒着，攒够 3200 点才调一次模型。
        返回的是本次调用解码出的片段，不保证是完整句子，
        整句由多次结果按顺序拼接得到。

        :param block: 音频块，取它的 ``samples``；
            直接给波形数组也行。
        :returns: 本次解码出的增量文本，没有就是空串。
        :raises AsrDecodeError: 解码失败。
        """
        if not self._instances_ready():
            await self.ensure_loaded()

        self._check_sample_rate(block)
        samples = _to_float32_1d(_block_samples(block))
        # 缓冲为空时记下最早一块的到来时刻，解码触发时报缓冲等待
        if self._pending_size == 0:
            self._pending_since = time.perf_counter()
        self._pending.append(samples)
        self._pending_size += int(samples.size)
        self._stream_samples += int(samples.size)
        # 同一块音频追加进整段录音缓存，松开按键后交给重识别
        self._recording.append(samples)
        self._recording_samples += int(samples.size)
        # 缓存与待解码缓冲都有上限，超了就把最早的丢掉
        self._trim_pending()
        self._trim_stream()
        self._trim_recording()

        parts: list[str] = []
        if self._pending_size >= MODEL_CHUNK_SAMPLES:
            # 缓冲攒满 200 毫秒触发解码，报一下这段音频攒了多久：
            # 出字延迟里大头是这段聚合等待，留个观测量方便后续调阈值
            buffered_wait_ms = (
                (time.perf_counter() - self._pending_since) * 1000.0
                if self._pending_since is not None
                else 0.0
            )
            log_event(
                "asr.decode.buffer_wait",
                "待解码缓冲攒满，触发一次解码",
                level="DEBUG",
                pending_samples=self._pending_size,
                buffered_wait_ms=round(buffered_wait_ms, 1),
                device=self._device or "",
            )
            timer = timed(
                ASR_LATENCY_METRIC,
                source=self._device or "",
                recorder=default_recorder(),
            )
            with timer:
                while self._pending_size >= MODEL_CHUNK_SAMPLES:
                    chunk = self._take(MODEL_CHUNK_SAMPLES)
                    text = await asyncio.to_thread(self._decode, chunk, False)
                    if text:
                        parts.append(text)
            # 缓冲没取空时剩下的音频从现在重新起算等待
            self._pending_since = (
                time.perf_counter() if self._pending_size > 0 else None
            )
        return "".join(parts)

    async def flush(self) -> str:
        """停止采集时冲刷，返回剩下的文本。

        用 ``is_final`` 为 true 的那次调用把剩余缓存解出来。
        这条流一个模型调用都没过时直接返回空串，不调模型；
        调过但缓存里没剩音频时，这次调用带的是空波形，
        作用是把模型内部的尾块冲出来。

        冲刷完丢掉这个 ``cache`` 字典，下一条语音流重新建一个，
        上一条语音的状态不会串到下一条。
        解码抛错时同样丢掉：清理放在 finally 里，
        异常照旧向上抛，缓存不会留给下一条语音流。

        :returns: 剩下的文本，没有就是空串。
        :raises AsrDecodeError: 解码失败。
        """
        if not self._instances_ready():
            return ""

        text = ""
        try:
            if self._calls > 0 or self._pending_size > 0:
                # 这里的解码耗时算在 AudioManager 的 asr_final_latency 里，
                # 本引擎不单独记这个指标，见《2-骨架与配置》性能指标一节
                remainder = self._take_all()
                text = await asyncio.to_thread(self._decode, remainder, True)
        finally:
            # 冲刷完丢掉 cache，下一条语音流重新建一个。
            # 解码抛错时同样要清理，否则这条语音的缓存会留给下一条语音流，
            # 上一条语音的状态就串到下一条，
            # 见《5-语音输入与流式ASR》「流式推理」一节。
            self._reset_stream()
        return text

    def _reset_stream(self) -> None:
        """丢掉当前语音流的状态，下一条语音流重新建一个。"""
        self._cache = {}
        self._pending.clear()
        self._pending_size = 0
        self._pending_since = None
        self._calls = 0
        self._stream_samples = 0
        # 录音缓存一并丢掉：正常路径里调用方在冲刷前已把录音取走交给
        # 重识别，这里是解码异常时的兜底，缓存不留给下一条语音流
        self._recording.clear()
        self._recording_samples = 0

    def take_recording(self) -> "np.ndarray":
        """取走当前录音会话的整段 PCM 缓存，下次录音重新积累。

        给 SenseVoice 重识别用：流式解码照常跑，这段缓存独立累积，
        松开按键后调用方在 :meth:`flush` 之前取走它，重识别用整段
        音频重新出文本。冲刷（:meth:`_reset_stream`）时缓存会被清掉，
        所以必须在冲刷之前取。

        :returns: float32 的一维波形，一段音频都没喂过时是空数组。
        """
        if not self._recording:
            return np.zeros(0, dtype="float32")
        if len(self._recording) == 1:
            wave = self._recording[0]
        else:
            wave = np.concatenate(self._recording)
        self._recording.clear()
        self._recording_samples = 0
        return wave

    def _trim_recording(self) -> None:
        """整段录音缓存超过上限时把最早的丢掉。

        与语音流缓存同上限（``max_stream_seconds`` 秒）：长会议不吃内存，
        重识别只覆盖最近上限时长内的音频。丢掉的部分流式解码同样
        已经丢过缓存，两边口径一致。
        """
        if self._recording_samples <= self._max_stream_samples:
            return
        excess = self._recording_samples - self._max_stream_samples
        dropped = 0
        while dropped < excess and self._recording:
            head = self._recording[0]
            need = excess - dropped
            if head.size > need:
                self._recording[0] = head[need:]
                dropped += int(need)
            else:
                dropped += int(head.size)
                self._recording.pop(0)
        self._recording_samples -= dropped
        log_event(
            "asr.recording.trimmed",
            "整段录音缓存超过上限，丢弃最早的音频",
            level="WARNING",
            dropped_samples=dropped,
            limit_samples=self._max_stream_samples,
        )

    def _trim_pending(self) -> None:
        """待解码缓冲超过上限时把最早的丢掉。

        见《5-语音输入与流式ASR》「性能」一节：限制音频缓存长度，
        避免长会议把内存吃满。
        """
        if self._pending_size <= MAX_PENDING_SAMPLES:
            return
        excess = self._pending_size - MAX_PENDING_SAMPLES
        self._take(excess)
        log_event(
            "asr.stream.pending_trimmed",
            "待解码缓冲超过上限，丢弃最早的音频",
            level="WARNING",
            dropped_samples=excess,
            limit_samples=MAX_PENDING_SAMPLES,
        )

    def _trim_stream(self) -> None:
        """一条语音流累计的音频超过上限时丢掉最早的缓存。

        见《5-语音输入与流式ASR》「性能」一节：限制音频缓存长度，
        避免长会议把内存吃满。超过上限就把当前缓存丢掉重新建一个，
        最早的音频不再参与后面的解码。

        待解码的缓冲是最新的音频，留着继续解码，不在这里丢。
        """
        if self._stream_samples <= self._max_stream_samples:
            return
        dropped_ms = self._stream_samples * 1000 // DEFAULT_SAMPLE_RATE
        # 丢掉最早的缓存重新建一个，计数从头算，后面的音频接着来
        self._cache = {}
        self._stream_samples = 0
        log_event(
            "asr.stream.trimmed",
            "语音流超过长度上限，丢弃最早的缓存",
            level="WARNING",
            dropped_ms=dropped_ms,
            limit_seconds=self._max_stream_seconds,
        )

    def _decode(self, chunk: "np.ndarray", is_final: bool) -> str:
        """跑一次流式推理，返回这次解码出的文本片段。

        :param chunk: 一段 float32 的一维波形。
        :param is_final: 是不是最后一段。
        :returns: 这一段解码出的文本。
        :raises AsrDecodeError: 解码失败。
        """

        try:
            result = self._asr.generate(
                input=chunk,
                cache=self._cache,
                is_final=is_final,
                chunk_size=list(STREAM_CHUNK_SIZE),
                encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
                batch_size=BATCH_SIZE,
                disable_pbar=True,
            )
        except Exception as exc:
            self._state = STATE_ERROR
            log_event(
                "asr.decode.failed",
                f"ASR 解码失败：{exc}",
                level="ERROR",
                error=str(exc),
                is_final=is_final,
            )
            raise AsrDecodeError(f"ASR 解码失败：{exc}") from exc
        self._state = STATE_READY
        self._calls += 1

        return _extract_text(result)

    def _take(self, count: int) -> "np.ndarray":
        """从待发缓存里取 count 个点。"""
        parts: list["np.ndarray"] = []
        remaining = count
        while remaining > 0:
            head = self._pending[0]
            if head.size > remaining:
                parts.append(head[:remaining])
                self._pending[0] = head[remaining:]
                remaining = 0
            else:
                parts.append(head)
                remaining -= int(head.size)
                self._pending.pop(0)
        self._pending_size -= count
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)

    def _take_all(self) -> "np.ndarray":
        """取走待发缓存里的全部点。"""
        if not self._pending:
            return np.zeros(0, dtype="float32")
        chunk = np.concatenate(self._pending)
        self._pending.clear()
        self._pending_size = 0
        return chunk

    # ------------------------------------------------------------------
    # 语音端点检测
    # ------------------------------------------------------------------
    async def detect_speech(self, samples: Any) -> list[tuple[int, int]]:
        """用 VAD 找出语音区间。

        区间边界由模型给出，这里原样返回，不自行修正。

        :param samples: 一段波形，可以是音频块，也可以是波形数组。
        :returns: 毫秒区间的列表。
        :raises ValueError: VAD 没开着。
        :raises AsrDecodeError: VAD 调用失败。
        """
        if not self._vad_enabled or self._vad is None:
            raise ValueError("VAD 没开着，做不了语音端点检测")
        if not self._instances_ready():
            await self.ensure_loaded()

        wave = _to_float32_1d(_block_samples(samples))
        try:
            result = await asyncio.to_thread(
                self._vad.generate,
                input=wave,
                chunk_size=VAD_CHUNK_MS,
                disable_pbar=True,
            )
        except Exception as exc:
            log_event(
                "asr.vad.failed",
                f"VAD 端点检测失败：{exc}",
                level="ERROR",
                error=str(exc),
            )
            raise AsrDecodeError(f"VAD 端点检测失败：{exc}") from exc
        return _extract_intervals(result)


def _block_samples(block: Any) -> Any:
    """取音频块里的波形。

    块是 :class:`~app.audio.audio_capture.AudioBlock` 时取它的 ``samples``，
    直接给波形数组时原样返回。
    """
    samples = getattr(block, "samples", None)
    if samples is None:
        return block
    return samples
