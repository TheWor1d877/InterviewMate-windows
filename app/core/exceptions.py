"""跨模块共享的异常类型与错误码。

错误码表见《2-骨架与配置》错误码一节；状态栏与日志用同一套错误码。
异常类名与错误码一一绑定，子类用类属性固定自己的 code，不允许在运行时改写。

``E_DIFY_REQUEST`` 是 M5 终检新增的码：服务端拒绝请求体时用它，
与 ``E_DIFY_RESPONSE``（响应格式异常）区分开。《2》错误码表待主线同步。
"""

from __future__ import annotations

__all__ = [
    "ERROR_CODES",
    "InterviewMateError",
    "ConfigError",
    "AudioDeviceError",
    "AsrLoadError",
    "AsrDecodeError",
    "ScreenshotError",
    "DeepSeekAuthError",
    "DeepSeekTimeoutError",
    "DeepSeekStreamError",
    "DifyAuthError",
    "DifyTimeoutError",
    "DifyResponseError",
    "DifyRequestError",
    "DifyQueryTooLongError",
    "SocketError",
]


# 错误码全集，与《2-骨架与配置》错误码表对应（E_DIFY_REQUEST 为 M5 终检新增）
ERROR_CODES: frozenset[str] = frozenset(
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


class InterviewMateError(Exception):
    """所有业务异常的基类，携带一个错误码。

    默认错误码是 ``E_CONFIG``；子类各自覆盖 ``code`` 类属性。
    显式传 ``code`` 时以传入值为准，但必须是 :data:`ERROR_CODES` 里的已知错误码，
    否则抛 ``ValueError``，避免把拼错的错误码写进日志。
    """

    code: str = "E_CONFIG"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        if code is not None:
            if code not in ERROR_CODES:
                raise ValueError(f"未知错误码：{code!r}，合法错误码见《2-骨架与配置》错误码表")
            # 实例属性覆盖类属性，只影响这一个实例
            self.code = code
        self.message = message
        super().__init__(message)


class ConfigError(InterviewMateError):
    """配置缺失或校验失败，对应 ``E_CONFIG``。"""

    code = "E_CONFIG"


class AudioDeviceError(InterviewMateError):
    """音频设备不可用，对应 ``E_AUDIO_DEVICE``。"""

    code = "E_AUDIO_DEVICE"


class AsrLoadError(InterviewMateError):
    """ASR 模型加载失败，对应 ``E_ASR_LOAD``。"""

    code = "E_ASR_LOAD"


class AsrDecodeError(InterviewMateError):
    """ASR 解码失败，对应 ``E_ASR_DECODE``。"""

    code = "E_ASR_DECODE"


class ScreenshotError(InterviewMateError):
    """截图接口不可用或被拒绝，对应 ``E_SCREENSHOT``。"""

    code = "E_SCREENSHOT"


class DeepSeekAuthError(InterviewMateError):
    """DeepSeek API Key 无效，对应 ``E_DEEPSEEK_AUTH``。"""

    code = "E_DEEPSEEK_AUTH"


class DeepSeekTimeoutError(InterviewMateError):
    """DeepSeek 请求超时，对应 ``E_DEEPSEEK_TIMEOUT``。"""

    code = "E_DEEPSEEK_TIMEOUT"


class DeepSeekStreamError(InterviewMateError):
    """DeepSeek 流式读取中断，对应 ``E_DEEPSEEK_STREAM``。"""

    code = "E_DEEPSEEK_STREAM"


class DifyAuthError(InterviewMateError):
    """Dify API Key 无效，对应 ``E_DIFY_AUTH``。"""

    code = "E_DIFY_AUTH"


class DifyTimeoutError(InterviewMateError):
    """Dify 检索超时，对应 ``E_DIFY_TIMEOUT``。"""

    code = "E_DIFY_TIMEOUT"


class DifyResponseError(InterviewMateError):
    """Dify 响应格式异常，对应 ``E_DIFY_RESPONSE``。"""

    code = "E_DIFY_RESPONSE"


class DifyRequestError(InterviewMateError):
    """Dify 拒绝请求体，对应 ``E_DIFY_REQUEST``。

    请求体被服务端校验拒绝时用它，与响应格式异常区分开。
    """

    code = "E_DIFY_REQUEST"


class DifyQueryTooLongError(InterviewMateError):
    """检索查询超过长度限制，对应 ``E_DIFY_QUERY_TOO_LONG``。"""

    code = "E_DIFY_QUERY_TOO_LONG"


class SocketError(InterviewMateError):
    """IPC 通信失败，对应 ``E_SOCKET``。"""

    code = "E_SOCKET"
