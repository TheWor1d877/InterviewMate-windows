"""配置加载与校验。

配置结构见《2-骨架与配置》配置结构一节，配置文件默认是仓库根下的 ``config/config.yaml``。
加载顺序：读 YAML → 应用环境变量覆盖 → 校验取值域。

环境变量约定见《2-骨架与配置》环境变量一节：环境变量覆盖配置文件里的同名项。
本轮映射有对应配置项的五个变量：

- ``MODEL_BASE_URL`` → ``model.base_url``
- ``MODEL_NAME`` → ``model.name``
- ``DIFY_BASE_URL`` → ``dify.base_url``
- ``DIFY_DATASET_ID`` → ``dify.dataset_id``
- ``DIFY_DATASET_API_KEY`` → ``dify.api_key``

必填环境变量的检查不在本模块职责内，由调用方（CLI 的 start 路径）用
:func:`missing_required_env` 自行判断，见《2-骨架与配置》命令行入口一节。

超时项 ``audio.init_timeout``、``asr.init_timeout``、``screenshot.timeout``
管的是哪一步，见《8-集成与验收》「超时管理」一节；三项都给了默认值，
老的手写配置里没有也能加载。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.core.exceptions import ConfigError
from app.platform.types import ScreenshotTarget

__all__ = [
    "AppConfig",
    "AppSettings",
    "UISettings",
    "HotkeySettings",
    "AudioSettings",
    "ASRSettings",
    "ScreenshotSettings",
    "ModelSettings",
    "RagSettings",
    "DifySettings",
    "CodingSettings",
    "PrivacySettings",
    "ProviderSettings",
    "load_config",
    "current_config",
    "set_current_config",
    "missing_required_env",
    "REQUIRED_ENV",
    "OPTIONAL_ENV",
    "RETIRED_MODEL_NAMES",
    "ENV_OVERRIDES",
    "REPO_ROOT",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_ENV_FILE",
]


# 仓库根目录：app/ 的上一级，与当前工作目录无关
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config" / "config.yaml"
DEFAULT_ENV_FILE: Path = REPO_ROOT / ".env"

# 必填环境变量：缺任一项时 CLI 以退出码 3 结束，见《2-骨架与配置》命令行入口一节
REQUIRED_ENV: tuple[str, ...] = ("MODEL_API_KEY",)

# 可选环境变量：当前配置结构里没有对应顶层段，只作为常量暴露
# DIFY_BASE_URL / DIFY_DATASET_ID / DIFY_DATASET_API_KEY 已经有对应的 dify 段，
# 移进 ENV_OVERRIDES，这里暂时是空的
OPTIONAL_ENV: tuple[str, ...] = ()

# 已退役的模型名，见《接口契约》模型标识与基址一节
RETIRED_MODEL_NAMES: tuple[str, ...] = (
    "deepseek-v4-flash-vision-exp",
    "deepseek-v4-flash",
    "deepseek-chat",
    "deepseek-reasoner",
)

# 环境变量到配置项的映射：(环境变量名) -> (配置顶层段, 段内键)
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "MODEL_BASE_URL": ("model", "base_url"),
    "MODEL_NAME": ("model", "name"),
    "DIFY_BASE_URL": ("dify", "base_url"),
    "DIFY_DATASET_ID": ("dify", "dataset_id"),
    "DIFY_DATASET_API_KEY": ("dify", "api_key"),
}


class AppSettings(BaseModel):
    """应用级参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    language: str = Field(min_length=1)
    # 日志级别取值见《2-骨架与配置》日志一节
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class UISettings(BaseModel):
    """终端界面布局参数，左右两栏按比例划分。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    left_ratio: int = Field(gt=0, lt=100)
    right_ratio: int = Field(gt=0, lt=100)
    min_width: int = Field(gt=0)
    syntax_highlighting: bool
    auto_scroll: bool
    show_status_bar: bool
    show_text_input: bool

    @model_validator(mode="after")
    def _check_ratio_sum(self) -> "UISettings":
        # 两栏必须占满整行，见《2-骨架与配置》配置结构一节
        if self.left_ratio + self.right_ratio != 100:
            raise ValueError(
                f"ui.left_ratio + ui.right_ratio 必须等于 100，当前是 "
                f"{self.left_ratio} + {self.right_ratio}"
            )
        return self


class HotkeySettings(BaseModel):
    """Textual 内部快捷键绑定。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    voice_toggle: str = Field(min_length=1)
    screenshot: str = Field(min_length=1)
    new_question: str = Field(min_length=1)
    cancel: str = Field(min_length=1)
    reanalyze: str = Field(min_length=1)
    show_rag_sources: str = Field(min_length=1)
    quit: str = Field(min_length=1)


class AudioSettings(BaseModel):
    """音频采集参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool
    # 取值域按《5》语音输入与流式 ASR补齐，《2》未写全
    source: Literal["auto", "microphone", "system_monitor", "mixed"]
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    chunk_duration_ms: int = Field(gt=0)
    # 音频设备初始化超时（秒）：设备枚举与开流这两步的上限，
    # 超时报 E_AUDIO_DEVICE，见《8-集成与验收》「超时管理」一节。
    # 给默认值而不是必填：老的手写配置里没有这一项也能加载。
    init_timeout: int = Field(default=10, gt=0)


class ASRSettings(BaseModel):
    """本地流式 ASR 参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool
    model: str = Field(min_length=1)
    device: str = Field(min_length=1)
    cpu_fallback: bool
    vad_enabled: bool
    preload: bool
    # ASR 初始化超时（秒）：模型加载（预热与首次识别时的懒加载）的上限，
    # 超时报 E_ASR_LOAD，见《8-集成与验收》「超时管理」一节。
    # 给默认值而不是必填：老的手写配置里没有这一项也能加载。
    init_timeout: int = Field(default=60, gt=0)


class ScreenshotSettings(BaseModel):
    """截图参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool
    # 截图目标：screen 是整屏（多显示器时拼成一张），primary 是主屏幕，
    # monitor 是指定的那块屏（序号看 monitor 字段）。
    # 只收这三个值，窗口与区域截图不提供。必填，不给默认值。
    target: ScreenshotTarget
    # target 是 monitor 时抓第几块屏，0 起数；其他目标下这个值被忽略。
    monitor: int = Field(default=0, ge=0)
    max_edge: int = Field(gt=0)
    # format 的取值域文档未定义，这里只做非空检查
    format: str = Field(min_length=1)
    persist_files: bool
    # 截图接口超时（秒）：真实抓屏那一路（不是接口探测那一路）的上限，
    # 超时报 E_SCREENSHOT，见《8-集成与验收》「超时管理」一节。
    # mss 抓屏不弹确认框，单次抓屏通常在几十毫秒内完成，150 秒是宽松上限。
    # 给默认值而不是必填：老的手写配置里没有这一项也能加载。
    timeout: int = Field(default=150, gt=0)


class ModelSettings(BaseModel):
    """模型接口参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    base_url: str = Field(min_length=1)
    name: str = Field(min_length=1)
    stream: bool
    # thinking 与 reasoning_effort 的取值见《接口契约》模型思考模式与推理强度一节
    thinking: Literal["enabled", "disabled"]
    reasoning_effort: Literal["low", "high", "max"]
    temperature: float = Field(ge=0, le=2)
    max_output_tokens: int = Field(gt=0)
    connect_timeout: int = Field(gt=0)
    read_timeout: int = Field(gt=0)
    max_retries: int = Field(ge=0)

    @field_validator("name")
    @classmethod
    def _reject_retired_models(cls, value: str) -> str:
        # 见《接口契约》模型标识与基址一节
        if value in RETIRED_MODEL_NAMES:
            raise ValueError(
                f"模型名 {value!r} 已退役，不能用于配置。"
                f"被拒绝的模型名：{'、'.join(RETIRED_MODEL_NAMES)}"
            )
        return value


class RagSettings(BaseModel):
    """Dify 检索参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool
    top_k: int = Field(ge=1)
    timeout: int = Field(gt=0)
    max_context_chars: int = Field(gt=0)
    score_threshold_enabled: bool
    # 增补项，见《接口契约》Dify 检索请求体一节
    score_threshold: float = Field(ge=0)
    reranking_enable: bool
    # 强制项目模式：开启后所有题目的 rag_required 恒为 true，见《3-文本输入与流式回答》题型判定一节
    force_project_mode: bool = False


class DifySettings(BaseModel):
    """Dify 检索服务的连接参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    # 三项都不写也能加载，真实取值从配置文件或环境变量来，
    # 见《2-骨架与配置》环境变量一节
    base_url: str = ""
    dataset_id: str = ""
    # api_key 是密钥，不进日志：repr 里不带上它，见《2-骨架与配置》日志一节
    api_key: str = Field(default="", repr=False)


class CodingSettings(BaseModel):
    """算法代码题参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    # 取值与默认值见《7-代码问题模式》提交格式识别一节
    language: str = Field(default="cpp", min_length=1)
    standard: str = Field(default="cpp17", min_length=1)
    submission_mode: Literal["auto", "leetcode", "nowcoder", "acm"] = "auto"
    # 编译校验开关，见《7-代码问题模式》代码验证一节
    compile_validation: bool = False


class PrivacySettings(BaseModel):
    """落盘与脱敏开关。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    persist_audio: bool
    persist_transcripts: bool
    persist_screenshots: bool
    redact_secrets_in_logs: bool


class ProviderSettings(BaseModel):
    """各外部能力的实现选择。

    默认走真实实现（openai/dify/funasr/mss）；测试与本地开发改成
    mock，对应实现不连真实服务、不加载模型、不抓真实屏幕。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    llm: Literal["mock", "openai"]
    retrieval: Literal["mock", "dify"]
    asr: Literal["mock", "funasr"]
    # 截图实现名：mock 不做真实截图，mss 是 Windows 侧的 mss 实现，
    # 见《2-骨架与配置》配置结构一节
    screenshot: Literal["mock", "mss"]


class AppConfig(BaseModel):
    """配置根对象，字段名与配置 YAML 顶层键同名。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    app: AppSettings
    ui: UISettings
    hotkeys: HotkeySettings
    audio: AudioSettings
    asr: ASRSettings
    screenshot: ScreenshotSettings
    model: ModelSettings
    rag: RagSettings
    # Dify 连接参数的默认值来自 DifySettings 自身，配置文件里没写也能加载
    dify: DifySettings = Field(default_factory=DifySettings)
    # 代码题参数的默认值来自 CodingSettings 自身，配置文件里没写也能加载
    coding: CodingSettings = Field(default_factory=CodingSettings)
    privacy: PrivacySettings
    providers: ProviderSettings


# 进程内最近一次加载成功的配置。
# 装配层启动时加载一次配置，平台后端与 ASR 引擎从这里取超时值，
# 不用把配置一路传下去，也不用各自再读一遍配置文件。
_loaded_config: AppConfig | None = None


def current_config() -> AppConfig | None:
    """返回进程内最近一次加载成功的配置。

    还没加载过配置时返回 None，调用方自己决定用哪个默认值。
    只读，不触发任何加载，见 :func:`set_current_config`。
    超时项的用途见《8-集成与验收》「超时管理」一节。
    """
    return _loaded_config


def set_current_config(config: AppConfig | None) -> AppConfig | None:
    """设置进程级当前配置，与读配置文件解耦。

    传一个配置就设成进程级当前配置并返回它；传 None 清除并返回 None。
    读配置文件走 :func:`load_config`，它成功后会自己设置；
    :func:`current_config` 只读，不会触发加载。
    """
    global _loaded_config
    _loaded_config = config
    return _loaded_config


def _load_env_file(env_file: str | None) -> None:
    """加载 .env 文件，已存在的环境变量优先，不被覆盖。

    ``env_file`` 为 None 时加载仓库根的 .env（存在才加载）；
    显式传入时加载该路径。语义见《2-骨架与配置》环境变量一节。
    """
    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    if not path.is_file():
        return
    # override=False：已有环境变量不被 .env 覆盖
    load_dotenv(path, override=False)


def missing_required_env(env_file: str | None = None) -> tuple[str, ...]:
    """返回当前环境里缺失的必填环境变量名。

    先加载 ``env_file`` 指定的环境变量文件（None 时用仓库根的 .env，
    存在才加载，已有环境变量优先），再比对 :data:`REQUIRED_ENV`。
    空字符串按缺失处理。

    :param env_file: .env 路径；None 时自动加载仓库根的 .env（存在才加载）。
    """
    _load_env_file(env_file)
    return tuple(name for name in REQUIRED_ENV if not (os.environ.get(name) or "").strip())


def _read_yaml_mapping(config_path: Path) -> dict[str, Any]:
    """读 YAML 文件并返回顶层映射，任何读取或语法问题都转成 ConfigError。"""
    if not config_path.is_file():
        raise ConfigError(f"配置文件不存在：{config_path}")
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"配置文件读取失败：{config_path}（{exc}）") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 YAML 语法错误：{config_path}（{exc}）") from exc
    if data is None:
        raise ConfigError(f"配置文件为空：{config_path}")
    if not isinstance(data, dict):
        raise ConfigError(
            f"配置文件顶层必须是键值映射：{config_path}（实际是 {type(data).__name__}）"
        )
    return data


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """用环境变量覆盖配置项，覆盖发生在 YAML 解析之后、取值域校验之前。"""
    for env_name, (section, key) in ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is None or not value.strip():
            # 未设置或空值都按「没有覆盖」处理
            continue
        bucket = data.get(section)
        if not isinstance(bucket, dict):
            bucket = {}
            data[section] = bucket
        bucket[key] = value


def _format_validation_error(exc: ValidationError) -> str:
    """把 pydantic 的校验错误整理成一行中文说明。"""
    details: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        if err.get("type") == "extra_forbidden":
            details.append(f"未知配置键 {loc}" if loc else "未知配置键")
            continue
        msg = err.get("msg", "")
        details.append(f"{loc}: {msg}" if loc else msg)
    return "；".join(details)


def load_config(path: str | None = None, *, env_file: str | None = None) -> AppConfig:
    """加载并校验配置。

    :param path: 配置文件路径，按当前工作目录解析；None 时读仓库根的 ``config/config.yaml``。
    :param env_file: .env 路径；None 时自动加载仓库根的 .env（存在才加载）。
    :raises ConfigError: 文件读不到、YAML 语法错、字段类型错、取值非法。
    """
    global _loaded_config
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    _load_env_file(env_file)
    data = _read_yaml_mapping(config_path)
    _apply_env_overrides(data)
    try:
        config = AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"配置校验失败：{_format_validation_error(exc)}") from exc
    # 加载成功才记下来：校验失败时沿用上一次加载成功的配置
    _loaded_config = config
    return config
