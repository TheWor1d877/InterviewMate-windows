"""配置加载与校验的测试。

覆盖《2-骨架与配置》配置结构一节、环境变量一节，以及《接口契约》
DeepSeek 模型标识与基址一节的退役模型名规则。
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml
from pydantic import BaseModel

import app.config
from app.config import (
    DEFAULT_CONFIG_PATH,
    ENV_OVERRIDES,
    OPTIONAL_ENV,
    REQUIRED_ENV,
    RETIRED_DEEPSEEK_MODELS,
    AppConfig,
    AppSettings,
    ASRSettings,
    AudioSettings,
    CodingSettings,
    DeepSeekSettings,
    DifySettings,
    HotkeySettings,
    PrivacySettings,
    ProviderSettings,
    RagSettings,
    ScreenshotSettings,
    UISettings,
    current_config,
    load_config,
    missing_required_env,
    set_current_config,
)
from app.core.exceptions import ConfigError

# 会影响配置结果的环境变量，测试里统一清掉
_OVERRIDE_ENV = (
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DIFY_BASE_URL",
    "DIFY_DATASET_ID",
    "DIFY_DATASET_API_KEY",
)
# 必填环境变量，缺失检查相关的测试单独控制
_API_KEY_ENV = "DEEPSEEK_API_KEY"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """清掉会影响配置的环境变量，并把默认 .env 路径指向不存在的文件。

    开发者在仓库根建 .env 之后（默认配置里没有 DEEPSEEK_API_KEY，
    但可能有 DEEPSEEK_MODEL 等覆盖项），默认配置相关的断言不应受其影响，
    否则 M8 阶段会出现与代码无关的假失败。
    """
    for name in _OVERRIDE_ENV + (_API_KEY_ENV,):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app.config, "DEFAULT_ENV_FILE", tmp_path / "not-exists.env")


@pytest.fixture(autouse=True)
def _restore_current_config() -> Iterator[None]:
    """每个用例结束都把进程级当前配置还原。

    ``load_config`` 与 ``set_current_config`` 都会改进程级配置，
    别的模块从这里取超时值，用例里设过的配置不能留给后面的用例。
    """
    saved = current_config()
    try:
        yield
    finally:
        set_current_config(saved)


def _base_config() -> dict[str, Any]:
    """以仓库里的默认配置文件作为测试基底。"""
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    return data


def _write_config(tmp_path: Path, data: dict[str, Any]) -> str:
    """把配置字典写到临时目录，返回文件路径。"""
    target = tmp_path / "config.yaml"
    target.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return str(target)


def _load_mutated(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> AppConfig:
    """改一份默认配置再加载。"""
    data = _base_config()
    mutate(data)
    return load_config(_write_config(tmp_path, data))


# ---------------------------------------------------------------- 默认配置


def test_default_config_loads() -> None:
    """默认配置能加载出来，各段取值与 config/config.yaml 一致。"""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.app.name == "InterviewMate"
    assert config.app.language == "zh-CN"
    assert config.app.log_level == "INFO"
    assert config.ui.left_ratio == 22
    assert config.ui.right_ratio == 78
    assert config.deepseek.model == "deepseek-flash"
    assert config.deepseek.base_url == "https://api.deepseek.com"
    assert config.deepseek.thinking == "disabled"
    assert config.deepseek.reasoning_effort == "low"
    assert config.rag.score_threshold == 0.0
    assert config.rag.force_project_mode is False
    assert config.providers.llm == "deepseek"
    assert config.providers.retrieval == "dify"
    assert config.providers.asr == "funasr"
    assert config.providers.screenshot == "gnome"


def test_default_config_top_level_keys() -> None:
    """顶层键与配置文件一致，含增补的 providers 段与 dify 段。"""
    assert set(AppConfig.model_fields) == {
        "app",
        "ui",
        "hotkeys",
        "audio",
        "asr",
        "screenshot",
        "deepseek",
        "rag",
        "dify",
        "coding",
        "privacy",
        "providers",
    }


def test_default_config_file_exists() -> None:
    """默认配置文件在仓库根下。"""
    assert DEFAULT_CONFIG_PATH.is_file()
    assert DEFAULT_CONFIG_PATH.parent.name == "config"


def test_load_config_without_api_key_succeeds() -> None:
    """load_config 不检查 DEEPSEEK_API_KEY，缺变量也能加载。"""
    import os

    assert "DEEPSEEK_API_KEY" not in os.environ
    config = load_config()
    assert config.deepseek.model == "deepseek-flash"


def test_default_path_is_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认配置路径按仓库根解析，与当前工作目录无关。"""
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.app.name == "InterviewMate"


def test_explicit_relative_path_resolves_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """显式传的相对路径按当前工作目录解析。"""
    data = _base_config()
    data["app"]["name"] = "相对路径配置"
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    config = load_config("config.yaml")
    assert config.app.name == "相对路径配置"


# ---------------------------------------------------------------- 文件与语法错误


def test_missing_config_file_raises_config_error(tmp_path: Path) -> None:
    """配置文件读不到抛 ConfigError。"""
    with pytest.raises(ConfigError) as excinfo:
        load_config(str(tmp_path / "不存在.yaml"))
    assert "不存在" in str(excinfo.value)


def test_yaml_syntax_error_raises_config_error(tmp_path: Path) -> None:
    """YAML 语法错抛 ConfigError。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("app: [unclosed\n  name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(str(bad))
    assert "YAML" in str(excinfo.value)


def test_empty_config_file_raises_config_error(tmp_path: Path) -> None:
    """空配置文件抛 ConfigError。"""
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(empty))


def test_non_mapping_top_level_raises_config_error(tmp_path: Path) -> None:
    """顶层不是键值映射抛 ConfigError。"""
    bad = tmp_path / "list.yaml"
    bad.write_text("- app\n- ui\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(bad))


def test_missing_section_raises_config_error(tmp_path: Path) -> None:
    """缺整段配置抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data.pop("ui"))


# ---------------------------------------------------------------- 类型错误


def test_log_level_wrong_type_raises_config_error(tmp_path: Path) -> None:
    """log_level 写成数字抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["app"].__setitem__("log_level", 123))


def test_left_ratio_wrong_type_raises_config_error(tmp_path: Path) -> None:
    """ui.left_ratio 写成字符串抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["ui"].__setitem__("left_ratio", "22"))


def test_boolean_wrong_type_raises_config_error(tmp_path: Path) -> None:
    """布尔项写成字符串抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(
            tmp_path, lambda data: data["audio"].__setitem__("enabled", "yes")
        )


# ---------------------------------------------------------------- 未知键


def test_unknown_top_level_key_raises_config_error(tmp_path: Path) -> None:
    """未知顶层键抛 ConfigError，消息里带上键名。"""
    with pytest.raises(ConfigError) as excinfo:
        _load_mutated(tmp_path, lambda data: data.__setitem__("unknown_section", {"a": 1}))
    assert "unknown_section" in str(excinfo.value)


def test_unknown_nested_key_raises_config_error(tmp_path: Path) -> None:
    """段内未知键抛 ConfigError，消息里带上键名。"""
    with pytest.raises(ConfigError) as excinfo:
        _load_mutated(tmp_path, lambda data: data["ui"].__setitem__("unknown_key", 1))
    assert "unknown_key" in str(excinfo.value)


# ---------------------------------------------------------------- 取值域


def test_invalid_thinking_raises_config_error(tmp_path: Path) -> None:
    """deepseek.thinking 取值非法抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["deepseek"].__setitem__("thinking", "always"))


def test_invalid_reasoning_effort_raises_config_error(tmp_path: Path) -> None:
    """deepseek.reasoning_effort 取值非法抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(
            tmp_path, lambda data: data["deepseek"].__setitem__("reasoning_effort", "none")
        )


def test_invalid_screenshot_target_raises_config_error(tmp_path: Path) -> None:
    """screenshot.target 取值非法抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["screenshot"].__setitem__("target", "桌面"))


@pytest.mark.parametrize("target", ["window", "area", "active_window"])
def test_screenshot_target_rejects_portal_only_targets(tmp_path: Path, target: str) -> None:
    """screenshot.target 不收 Portal 的窗口与区域目标。

    XDG Portal 的截图接口还支持 Window、Area、Active Window 三个目标值，
    InterviewMate 一律不请求，见《接口契约》XDG Portal 截图目标一节。
    """
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["screenshot"].__setitem__("target", target))


@pytest.mark.parametrize("target", ["screen", "primary", "monitor"])
def test_screenshot_target_accepts_three_values(tmp_path: Path, target: str) -> None:
    """screenshot.target 收 screen、primary、monitor 三个值。"""
    config = _load_mutated(
        tmp_path, lambda data: data["screenshot"].__setitem__("target", target)
    )
    assert config.screenshot.target == target


def test_screenshot_monitor_defaults_to_zero(tmp_path: Path) -> None:
    """screenshot.monitor 不给时是 0。"""
    config = _load_mutated(tmp_path, lambda data: None)
    assert config.screenshot.monitor == 0


def test_invalid_screenshot_monitor_raises_config_error(tmp_path: Path) -> None:
    """screenshot.monitor 是负数时抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["screenshot"].__setitem__("monitor", -1))


def test_invalid_audio_source_raises_config_error(tmp_path: Path) -> None:
    """audio.source 取值非法抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["audio"].__setitem__("source", "speaker"))


def test_invalid_submission_mode_raises_config_error(tmp_path: Path) -> None:
    """coding.submission_mode 取值非法抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(
            tmp_path, lambda data: data["coding"].__setitem__("submission_mode", "generic")
        )


# ---------------------------------------------------------------- 代码题配置


def test_coding_defaults() -> None:
    """代码题配置的默认值符合《7-代码问题模式》提交格式识别一节。"""
    settings = CodingSettings()

    assert settings.language == "cpp"
    assert settings.standard == "cpp17"
    assert settings.submission_mode == "auto"


def test_default_config_coding_section() -> None:
    """默认配置里的 coding 段与 config/config.yaml 一致。"""
    config = load_config()

    assert config.coding.language == "cpp"
    assert config.coding.standard == "cpp17"
    assert config.coding.submission_mode == "auto"


def test_coding_fields_can_be_overridden(tmp_path: Path) -> None:
    """coding 段三个字段都能改。"""
    config = _load_mutated(
        tmp_path,
        lambda data: data["coding"].update(
            {"language": "java", "standard": "java17", "submission_mode": "nowcoder"}
        ),
    )

    assert config.coding.language == "java"
    assert config.coding.standard == "java17"
    assert config.coding.submission_mode == "nowcoder"


@pytest.mark.parametrize("mode", ["auto", "leetcode", "nowcoder", "acm"])
def test_coding_submission_mode_accepts_four_values(tmp_path: Path, mode: str) -> None:
    """coding.submission_mode 收 auto、leetcode、nowcoder、acm 四个值。"""
    config = _load_mutated(
        tmp_path, lambda data: data["coding"].__setitem__("submission_mode", mode)
    )

    assert config.coding.submission_mode == mode


def test_coding_empty_language_raises_config_error(tmp_path: Path) -> None:
    """coding.language 是空串时抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["coding"].__setitem__("language", ""))


def test_coding_empty_standard_raises_config_error(tmp_path: Path) -> None:
    """coding.standard 是空串时抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["coding"].__setitem__("standard", ""))


def test_coding_unknown_key_raises_config_error(tmp_path: Path) -> None:
    """coding 段内未知键抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["coding"].__setitem__("unknown", 1))


def test_coding_section_defaults_when_missing(tmp_path: Path) -> None:
    """配置文件里没有 coding 段时用 CodingSettings 的默认值。"""
    config = _load_mutated(tmp_path, lambda data: data.pop("coding"))

    assert config.coding.language == "cpp"
    assert config.coding.standard == "cpp17"
    assert config.coding.submission_mode == "auto"


# ---------------------------------------------------------------- Dify 配置


def test_dify_defaults() -> None:
    """Dify 配置的默认是三个空串，不写也能用。"""
    settings = DifySettings()

    assert settings.base_url == ""
    assert settings.dataset_id == ""
    assert settings.api_key == ""


def test_default_config_dify_section() -> None:
    """默认配置里的 dify 段与 config/config.yaml 一致。"""
    config = load_config()

    assert config.dify.base_url == "http://your-dify-host/v1"
    assert config.dify.dataset_id == "your_dataset_id"
    assert config.dify.api_key == "your_dataset_api_key"


def test_dify_fields_can_be_overridden(tmp_path: Path) -> None:
    """dify 段三个字段都能改。"""
    config = _load_mutated(
        tmp_path,
        lambda data: data["dify"].update(
            {
                "base_url": "http://dify.internal/v1",
                "dataset_id": "dataset-1",
                "api_key": "dataset-key",
            }
        ),
    )

    assert config.dify.base_url == "http://dify.internal/v1"
    assert config.dify.dataset_id == "dataset-1"
    assert config.dify.api_key == "dataset-key"


def test_dify_section_defaults_when_missing(tmp_path: Path) -> None:
    """配置文件里没有 dify 段时用 DifySettings 的默认值。"""
    config = _load_mutated(tmp_path, lambda data: data.pop("dify"))

    assert config.dify.base_url == ""
    assert config.dify.dataset_id == ""
    assert config.dify.api_key == ""


def test_dify_unknown_key_raises_config_error(tmp_path: Path) -> None:
    """dify 段内未知键抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["dify"].__setitem__("unknown", 1))


def test_dify_api_key_is_not_in_repr() -> None:
    """api_key 是密钥，不出现在 repr 里，写日志时不会带出去。"""
    settings = DifySettings(base_url="http://dify.internal/v1", api_key="secret-key")

    assert "secret-key" not in repr(settings)


def test_invalid_log_level_raises_config_error(tmp_path: Path) -> None:
    """app.log_level 取值非法抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["app"].__setitem__("log_level", "VERBOSE"))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("llm", "openai"),
        ("retrieval", "elastic"),
        ("asr", "whisper"),
        ("screenshot", "imagemagick"),
    ],
)
def test_invalid_provider_raises_config_error(
    tmp_path: Path, key: str, value: str
) -> None:
    """providers.* 取值非法抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["providers"].__setitem__(key, value))


def test_valid_provider_values_accepted(tmp_path: Path) -> None:
    """providers.* 的合法取值能通过校验。"""
    config = _load_mutated(
        tmp_path,
        lambda data: data["providers"].update(
            {"llm": "deepseek", "retrieval": "dify", "asr": "funasr", "screenshot": "gnome"}
        ),
    )
    assert config.providers.llm == "deepseek"
    assert config.providers.retrieval == "dify"
    assert config.providers.asr == "funasr"
    assert config.providers.screenshot == "gnome"


def test_provider_screenshot_accepts_mss(tmp_path: Path) -> None:
    """providers.screenshot 接受 Windows 侧的 mss 实现名。

    见《9-平台差异与Windows适配》平台能力矩阵：截图实现名两平台不同。
    """
    config = _load_mutated(
        tmp_path, lambda data: data["providers"].__setitem__("screenshot", "mss")
    )
    assert config.providers.screenshot == "mss"


# ---------------------------------------------------------------- 数值合理性


def test_ratio_sum_not_100_raises_config_error(tmp_path: Path) -> None:
    """左右两栏比例之和不是 100 抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(
            tmp_path,
            lambda data: data["ui"].update({"left_ratio": 30, "right_ratio": 78}),
        )


def test_ratio_zero_boundary_raises_config_error(tmp_path: Path) -> None:
    """比例取 0 抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(
            tmp_path,
            lambda data: data["ui"].update({"left_ratio": 0, "right_ratio": 100}),
        )


def test_ratio_100_boundary_raises_config_error(tmp_path: Path) -> None:
    """比例取 100 抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(
            tmp_path,
            lambda data: data["ui"].update({"left_ratio": 100, "right_ratio": 0}),
        )


def test_top_k_zero_raises_config_error(tmp_path: Path) -> None:
    """rag.top_k 为 0 抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["rag"].__setitem__("top_k", 0))


def test_temperature_out_of_range_raises_config_error(tmp_path: Path) -> None:
    """temperature 超出 0..2 抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["deepseek"].__setitem__("temperature", 5.0))


def test_negative_max_retries_raises_config_error(tmp_path: Path) -> None:
    """max_retries 为负抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["deepseek"].__setitem__("max_retries", -1))


def test_non_positive_timeout_raises_config_error(tmp_path: Path) -> None:
    """rag.timeout 为 0 抛 ConfigError。"""
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, lambda data: data["rag"].__setitem__("timeout", 0))


# ---------------------------------------------------------------- 退役模型名


@pytest.mark.parametrize("model_name", RETIRED_DEEPSEEK_MODELS)
def test_retired_model_name_rejected(tmp_path: Path, model_name: str) -> None:
    """四个退役模型名都被拒绝，报错消息同时列出全部名字。"""
    with pytest.raises(ConfigError) as excinfo:
        _load_mutated(tmp_path, lambda data: data["deepseek"].__setitem__("model", model_name))
    message = str(excinfo.value)
    for name in RETIRED_DEEPSEEK_MODELS:
        assert name in message


def test_other_model_name_accepted(tmp_path: Path) -> None:
    """非退役的模型名可以接受。"""
    config = _load_mutated(
        tmp_path, lambda data: data["deepseek"].__setitem__("model", "deepseek-flash-next")
    )
    assert config.deepseek.model == "deepseek-flash-next"


# ---------------------------------------------------------------- 环境变量覆盖


def test_env_overrides_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEEPSEEK_MODEL 覆盖配置文件里的模型名。"""
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-env")
    config = load_config(_write_config(tmp_path, _base_config()))
    assert config.deepseek.model == "from-env"


def test_env_overrides_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEEPSEEK_BASE_URL 覆盖配置文件里的基址。"""
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid/v1")
    config = load_config(_write_config(tmp_path, _base_config()))
    assert config.deepseek.base_url == "https://example.invalid/v1"


def test_env_override_still_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """环境变量覆盖后仍走取值域校验，退役模型名照样报错。"""
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    with pytest.raises(ConfigError) as excinfo:
        load_config(_write_config(tmp_path, _base_config()))
    for name in RETIRED_DEEPSEEK_MODELS:
        assert name in str(excinfo.value)


@pytest.mark.parametrize(
    ("env_name", "field", "value"),
    [
        ("DIFY_BASE_URL", "base_url", "http://dify.internal/v1"),
        ("DIFY_DATASET_ID", "dataset_id", "dataset-from-env"),
        ("DIFY_DATASET_API_KEY", "api_key", "key-from-env"),
    ],
)
def test_env_overrides_dify_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_name: str, field: str, value: str
) -> None:
    """三个 DIFY_* 环境变量覆盖配置文件里 dify 段的同名项。"""
    monkeypatch.setenv(env_name, value)
    config = load_config(_write_config(tmp_path, _base_config()))
    assert getattr(config.dify, field) == value


def test_env_override_targets_are_documented() -> None:
    """只映射有对应配置项的五个环境变量。"""
    assert ENV_OVERRIDES == {
        "DEEPSEEK_BASE_URL": ("deepseek", "base_url"),
        "DEEPSEEK_MODEL": ("deepseek", "model"),
        "DIFY_BASE_URL": ("dify", "base_url"),
        "DIFY_DATASET_ID": ("dify", "dataset_id"),
        "DIFY_DATASET_API_KEY": ("dify", "api_key"),
    }


# ---------------------------------------------------------------- .env 加载


def test_env_file_is_loaded(tmp_path: Path) -> None:
    """显式指定 .env 时，里面的值生效。"""
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_MODEL=from-env-file\n", encoding="utf-8")
    config = load_config(_write_config(tmp_path, _base_config()), env_file=str(env_file))
    assert config.deepseek.model == "from-env-file"


def test_env_file_does_not_override_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已有环境变量不被 .env 覆盖。"""
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_MODEL=from-env-file\n", encoding="utf-8")
    config = load_config(_write_config(tmp_path, _base_config()), env_file=str(env_file))
    assert config.deepseek.model == "from-shell"


def test_missing_env_file_is_ignored(tmp_path: Path) -> None:
    """.env 不存在时不报错。"""
    config = load_config(
        _write_config(tmp_path, _base_config()),
        env_file=str(tmp_path / "not-exists.env"),
    )
    assert config.deepseek.model == "deepseek-flash"


def test_default_env_file_points_to_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 .env 路径固定是仓库根下的 .env。

    fixture 把 DEFAULT_ENV_FILE 换成了临时假路径，这里先撤销该替换，
    再核对常量本身的语义，避免这条语义在测试套件里失去保护。
    """
    monkeypatch.undo()
    assert app.config.DEFAULT_ENV_FILE == app.config.REPO_ROOT / ".env"
    # REPO_ROOT 确实是仓库根
    assert (app.config.REPO_ROOT / "config" / "config.yaml").is_file()


def test_default_env_file_is_auto_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不传 env_file 时，自动加载默认 .env（路径由 DEFAULT_ENV_FILE 指定）。"""
    env_file = tmp_path / "auto.env"
    env_file.write_text("DEEPSEEK_MODEL=from-auto-env\n", encoding="utf-8")
    monkeypatch.setattr(app.config, "DEFAULT_ENV_FILE", env_file)
    config = load_config()
    assert config.deepseek.model == "from-auto-env"


def test_default_env_file_does_not_override_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认 .env 里的值不覆盖已有的环境变量。"""
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-shell")
    env_file = tmp_path / "auto.env"
    env_file.write_text("DEEPSEEK_MODEL=from-auto-env\n", encoding="utf-8")
    monkeypatch.setattr(app.config, "DEFAULT_ENV_FILE", env_file)
    config = load_config()
    assert config.deepseek.model == "from-shell"


def test_local_env_file_does_not_affect_default_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认 .env 路径被换掉后，仓库根的本地 .env 不影响默认配置。

    这是开发者在仓库根建 .env 之后的回归防线：
    默认配置断言不能依赖开发者本地的环境变量。
    """
    monkeypatch.setattr(app.config, "DEFAULT_ENV_FILE", tmp_path / "not-exists.env")
    config = load_config()
    assert config.deepseek.model == "deepseek-flash"
    assert config.providers.llm == "deepseek"


# ---------------------------------------------------------------- 必填环境变量


def test_missing_required_env_reports_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺 DEEPSEEK_API_KEY 时返回该变量名。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert missing_required_env() == ("DEEPSEEK_API_KEY",)


def test_missing_required_env_empty_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有 DEEPSEEK_API_KEY 时返回空元组。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert missing_required_env() == ()


def test_env_constants() -> None:
    """必填与可选环境变量常量符合冻结契约。

    DIFY_BASE_URL / DIFY_DATASET_ID / DIFY_DATASET_API_KEY 有了对应的 dify 段，
    移进 ENV_OVERRIDES，OPTIONAL_ENV 现在是空的。
    """
    assert REQUIRED_ENV == ("DEEPSEEK_API_KEY",)
    assert OPTIONAL_ENV == ()


# ---------------------------------------------------------------- 导出符号


def test_settings_models_are_exported() -> None:
    """各段配置模型都是 pydantic 模型。"""
    for model in (
        AppSettings,
        UISettings,
        HotkeySettings,
        AudioSettings,
        ASRSettings,
        ScreenshotSettings,
        DeepSeekSettings,
        RagSettings,
        CodingSettings,
        PrivacySettings,
        ProviderSettings,
    ):
        assert issubclass(model, BaseModel)


def test_config_error_importable_from_app_config() -> None:
    """from app.config import ConfigError 可用。"""
    from app.config import ConfigError as ImportedConfigError
    from app.core.exceptions import ConfigError as OriginConfigError

    assert ImportedConfigError is OriginConfigError


def test_base_config_is_not_shared_between_loads(tmp_path: Path) -> None:
    """每次加载都返回独立对象。"""
    first = load_config(_write_config(tmp_path, _base_config()))
    second = load_config(_write_config(tmp_path, _base_config()))
    assert first is not second
    assert first.deepseek is not second.deepseek
    assert copy.deepcopy(first).deepseek.model == "deepseek-flash"


# ---------------------------------------------------------------- 进程级当前配置


def test_set_current_config_exposes_the_same_object() -> None:
    """set_current_config 设好之后，current_config 返回同一个对象。"""
    config = load_config()

    returned = set_current_config(config)

    assert returned is config
    assert current_config() is config


def test_set_current_config_none_clears() -> None:
    """传 None 清除进程级配置，current_config 随之返回 None。"""
    load_config()

    assert set_current_config(None) is None
    assert current_config() is None


def test_current_config_is_read_only() -> None:
    """current_config 只读，不触发加载。"""
    set_current_config(None)

    assert current_config() is None


def test_load_config_still_sets_process_config(tmp_path: Path) -> None:
    """load_config 成功加载后仍会设置进程级配置（既有副作用不变）。"""
    set_current_config(None)

    config = load_config(_write_config(tmp_path, _base_config()))

    assert current_config() is config


def test_set_current_config_is_exported() -> None:
    """set_current_config 写进了 __all__。"""
    assert "set_current_config" in app.config.__all__
    assert "current_config" in app.config.__all__
