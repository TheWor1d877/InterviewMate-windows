"""错误码与异常的测试。

错误码表见《2-骨架与配置》错误码一节，共 14 个，类名与错误码一一绑定。
"""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    ERROR_CODES,
    AsrDecodeError,
    AsrLoadError,
    AudioDeviceError,
    ConfigError,
    DeepSeekAuthError,
    DeepSeekStreamError,
    DeepSeekTimeoutError,
    DifyAuthError,
    DifyQueryTooLongError,
    DifyResponseError,
    DifyRequestError,
    DifyTimeoutError,
    InterviewMateError,
    ScreenshotError,
    SocketError,
)

# 类名 -> 错误码，严格对应冻结契约
SUBCLASS_CODES: tuple[tuple[type[InterviewMateError], str], ...] = (
    (ConfigError, "E_CONFIG"),
    (AudioDeviceError, "E_AUDIO_DEVICE"),
    (AsrLoadError, "E_ASR_LOAD"),
    (AsrDecodeError, "E_ASR_DECODE"),
    (ScreenshotError, "E_SCREENSHOT"),
    (DeepSeekAuthError, "E_DEEPSEEK_AUTH"),
    (DeepSeekTimeoutError, "E_DEEPSEEK_TIMEOUT"),
    (DeepSeekStreamError, "E_DEEPSEEK_STREAM"),
    (DifyAuthError, "E_DIFY_AUTH"),
    (DifyTimeoutError, "E_DIFY_TIMEOUT"),
    (DifyResponseError, "E_DIFY_RESPONSE"),
    (DifyRequestError, "E_DIFY_REQUEST"),
    (DifyQueryTooLongError, "E_DIFY_QUERY_TOO_LONG"),
    (SocketError, "E_SOCKET"),
)

ALL_SUBCLASSES = tuple(cls for cls, _ in SUBCLASS_CODES)


def test_error_codes_has_fourteen_elements() -> None:
    """错误码恰好 14 个。"""
    assert len(ERROR_CODES) == 14


def test_error_codes_is_frozenset() -> None:
    """错误码集合是 frozenset。"""
    assert isinstance(ERROR_CODES, frozenset)


def test_error_codes_exact_content() -> None:
    """错误码内容与《2-骨架与配置》错误码表一致。"""
    assert ERROR_CODES == frozenset(
        {
            "E_CONFIG",
            "E_AUDIO_DEVICE",
            "E_ASR_LOAD",
            "E_ASR_DECODE",
            "E_SCREENSHOT",
            "E_DEEPSEEK_AUTH",
            "E_DEEPSEEK_TIMEOUT",
            "E_DEEPSEEK_STREAM",
            "E_DIFY_AUTH",
            "E_DIFY_TIMEOUT",
            "E_DIFY_RESPONSE",
            "E_DIFY_REQUEST",
            "E_DIFY_QUERY_TOO_LONG",
            "E_SOCKET",
        }
    )


def test_base_error_default_code() -> None:
    """基类默认错误码是 E_CONFIG。"""
    assert InterviewMateError.code == "E_CONFIG"
    assert InterviewMateError().code == "E_CONFIG"
    assert InterviewMateError("出错了").code == "E_CONFIG"


def test_base_error_default_message_is_empty() -> None:
    """不传消息时消息为空。"""
    error = InterviewMateError()
    assert error.message == ""
    assert str(error) == ""


def test_base_error_accepts_explicit_code() -> None:
    """显式传 code 时覆盖默认错误码。"""
    error = InterviewMateError("socket 断了", code="E_SOCKET")
    assert error.code == "E_SOCKET"
    assert error.message == "socket 断了"


def test_base_error_rejects_unknown_code() -> None:
    """传不在 ERROR_CODES 里的 code 抛 ValueError。"""
    with pytest.raises(ValueError) as excinfo:
        InterviewMateError("x", code="E_NOT_A_CODE")
    assert "E_NOT_A_CODE" in str(excinfo.value)


def test_base_error_rejects_empty_code() -> None:
    """空字符串不是合法错误码。"""
    with pytest.raises(ValueError):
        InterviewMateError("x", code="")


def test_base_error_is_exception_subclass() -> None:
    """基类继承 Exception。"""
    assert issubclass(InterviewMateError, Exception)


@pytest.mark.parametrize(("cls", "code"), SUBCLASS_CODES)
def test_subclass_code_attribute(cls: type[InterviewMateError], code: str) -> None:
    """每个子类的 code 类属性正确。"""
    assert cls.code == code
    assert cls().code == code


@pytest.mark.parametrize(("cls", "code"), SUBCLASS_CODES)
def test_subclass_is_interviewmate_error(
    cls: type[InterviewMateError], code: str
) -> None:
    """每个子类的实例都是 InterviewMateError。"""
    error = cls("出错了")
    assert isinstance(error, InterviewMateError)
    assert isinstance(error, Exception)
    assert error.code == code


@pytest.mark.parametrize(("cls", "code"), SUBCLASS_CODES)
def test_subclass_str_contains_message(
    cls: type[InterviewMateError], code: str
) -> None:
    """子类实例的 str() 含消息内容。"""
    error = cls("配置文件缺失")
    assert "配置文件缺失" in str(error)
    assert error.message == "配置文件缺失"


@pytest.mark.parametrize(("cls", "code"), SUBCLASS_CODES)
def test_subclass_rejects_unknown_code(
    cls: type[InterviewMateError], code: str
) -> None:
    """子类传非法 code 也抛 ValueError。"""
    with pytest.raises(ValueError):
        cls("x", code="E_UNKNOWN")


@pytest.mark.parametrize(("cls", "code"), SUBCLASS_CODES)
def test_subclass_accepts_known_code(
    cls: type[InterviewMateError], code: str
) -> None:
    """子类可以显式覆盖成另一个合法错误码。"""
    error = cls("x", code="E_SOCKET")
    assert error.code == "E_SOCKET"


def test_subclass_codes_cover_error_codes() -> None:
    """子类错误码集合与 ERROR_CODES 一一对应，不多不少。"""
    assert {cls.code for cls in ALL_SUBCLASSES} == set(ERROR_CODES)


def test_subclass_codes_are_distinct() -> None:
    """14 个子类的错误码互不相同。"""
    codes = [cls.code for cls in ALL_SUBCLASSES]
    assert len(codes) == len(set(codes)) == 14


def test_subclasses_are_distinct_types() -> None:
    """14 个子类互不相同。"""
    assert len(set(ALL_SUBCLASSES)) == 14


def test_error_can_be_raised_and_caught_as_base() -> None:
    """子类异常能被基类捕获。"""
    with pytest.raises(InterviewMateError) as excinfo:
        raise ConfigError("配置缺失")
    assert excinfo.value.code == "E_CONFIG"


def test_config_error_importable_from_app_config() -> None:
    """from app.config import ConfigError 指向同一个类。"""
    from app.config import ConfigError as FromConfig
    from app.core.exceptions import ConfigError as FromExceptions

    assert FromConfig is FromExceptions
