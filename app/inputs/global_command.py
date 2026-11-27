"""socket 事件转内部动作。

桌面快捷键（``interviewmate.cmd hotkey voice-toggle``）通过 socket 把动作送给主进程，
主进程这边把协议里的 action 映射到内部动作处理器，见《8-集成与验收》全局快捷键一节
与《2-骨架与配置》进程通信一节。

action 与内部动作的对应：

| 协议 action | 内部动作 | 行为 |
|---|---|---|
| ``voice-toggle`` | ``voice_toggle`` | 切换语音采集状态 |
| ``screenshot`` | ``screenshot`` | 触发一次截图 |
| ``stop`` | ``stop`` | 只停语音采集并请求主进程退出，不取消在跑的模型请求 |
| ``status`` | ``status`` | 返回当前状态 |
| ``warmup`` | ``warmup`` | 触发预热 |

具体动作由外面注入的处理器去做，本模块只负责分发：处理器的返回值原样传出，
socket 服务端可以直接拿它当响应 payload。

本模块不碰 socket 本身，编解码在 ``app/runtime/socket_protocol.py``；
也不 import 界面层的实现类，见《2-骨架与配置》目录结构约定一节。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from app.runtime.socket_protocol import ACTIONS, Request
from app.utils.logger import log_event

__all__ = [
    "ACTION_VOICE_TOGGLE",
    "ACTION_SCREENSHOT",
    "ACTION_STOP",
    "ACTION_STATUS",
    "ACTION_WARMUP",
    "INTERNAL_ACTIONS",
    "ACTION_TO_INTERNAL",
    "RESULT_UNKNOWN_ACTION",
    "RESULT_NO_HANDLER",
    "ActionHandler",
    "GlobalCommandDispatcher",
]


# 内部动作名，处理器按这些名字注册
ACTION_VOICE_TOGGLE: str = "voice_toggle"
ACTION_SCREENSHOT: str = "screenshot"
ACTION_STOP: str = "stop"
ACTION_STATUS: str = "status"
ACTION_WARMUP: str = "warmup"

# 协议 action 到内部动作的映射，与《2-骨架与配置》进程通信一节的 action 表一致
ACTION_TO_INTERNAL: dict[str, str] = {
    "voice-toggle": ACTION_VOICE_TOGGLE,
    "screenshot": ACTION_SCREENSHOT,
    "stop": ACTION_STOP,
    "status": ACTION_STATUS,
    "warmup": ACTION_WARMUP,
}

# 全部内部动作名
INTERNAL_ACTIONS: tuple[str, ...] = tuple(ACTION_TO_INTERNAL.values())

# 未知 action 的结果标识
RESULT_UNKNOWN_ACTION: str = "unknown-action"

# action 合法但没有注册处理器的结果标识
RESULT_NO_HANDLER: str = "no-handler"

# 日志字段值里允许出现的字符：字母、数字、下划线、点、冒号、连字符。
# action 是客户端可控的，规整后再写日志，防止它伪造出额外的日志字段，
# 见《2-骨架与配置》日志一节。
_LOG_SAFE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_.:-"
)


class ActionHandler(Protocol):
    """内部动作处理器的形状。

    收一条请求，返回值原样当响应 payload；可以是协程函数，
    分发入口会把协程等完再返回。
    """

    def __call__(self, request: Request) -> Any: ...


class GlobalCommandDispatcher:
    """把 socket 的 action 分发到内部动作处理器。

    :param handlers: 内部动作名到处理器的映射，键取 :data:`INTERNAL_ACTIONS` 里的值。
    """

    def __init__(self, handlers: Mapping[str, ActionHandler] | None = None) -> None:
        self._handlers: dict[str, ActionHandler] = dict(handlers or {})

    @property
    def handlers(self) -> dict[str, ActionHandler]:
        """当前注册的处理器。"""
        return dict(self._handlers)

    def register(self, action: str, handler: ActionHandler) -> None:
        """注册一个内部动作处理器。

        :param action: 内部动作名，取 :data:`INTERNAL_ACTIONS` 里的值。
        :param handler: 处理器。
        """
        self._handlers[action] = handler

    async def dispatch(self, request: Request) -> Any:
        """按请求里的 action 找处理器并调用。

        未知 action 与没有注册处理器时都不抛异常：记一条日志，
        返回带明确结果标识的字典。
        处理器自己抛的异常往上抛，socket 服务端负责转成错误响应，
        见《2-骨架与配置》进程通信一节。

        :param request: 一条请求，``action`` 是协议里的取值。
        :returns: 处理器的返回值，原样传出。
        """
        action = request.action
        internal = ACTION_TO_INTERNAL.get(action)
        if internal is None:
            log_event(
                "command.unknown_action",
                "未知 action，已忽略",
                level="WARNING",
                action=_safe_text(action),
            )
            return {
                "result": RESULT_UNKNOWN_ACTION,
                "action": action,
                "note": f"未知 action：{action}",
            }
        handler = self._handlers.get(internal)
        if handler is None:
            log_event(
                "command.no_handler",
                "action 没有注册处理器",
                level="WARNING",
                action=_safe_text(action),
                internal_action=internal,
            )
            return {
                "result": RESULT_NO_HANDLER,
                "action": action,
                "note": f"action {action} 没有注册处理器",
            }
        result = handler(request)
        if inspect.isawaitable(result):
            result = await result
        return result

    def socket_handlers(self) -> dict[str, Callable[[Request], Any]]:
        """给 socket 服务端用的 handler 映射。

        键是协议里的 action 取值，值都是同一个分发入口，
        服务端按《2-骨架与配置》进程通信一节的 action 表分发。
        """
        return {action: self.dispatch for action in ACTIONS}


def _safe_text(text: str) -> str:
    """把字段值里不在白名单里的字符换成连字符。"""
    return "".join(ch if ch in _LOG_SAFE_CHARS else "-" for ch in text)
