"""socket 协议编解码。

线格式见《2-骨架与配置》进程通信一节冻结的 socket 协议：

- 请求是单行 JSON：``{"action": ..., "request_id": ...}``，``action`` 取值
  :data:`ACTIONS`，``request_id`` 可选；
- 响应是单行 JSON：``{"ok": ..., "error": ..., "payload": {...}}``；
- 客户端连上后发一行请求、读一行响应、断开连接。

编码结果一定是一行，末尾带一个换行符；解码失败一律抛
:class:`app.core.exceptions.SocketError`（错误码 ``E_SOCKET``）。

本模块只做编解码，不碰 socket、不发网络请求，服务端与客户端都依赖它，
因此协议格式只有这一处定义。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.core.exceptions import SocketError

__all__ = [
    "ACTIONS",
    "Request",
    "Response",
    "encode_request",
    "decode_request",
    "encode_response",
    "decode_response",
]


# 合法 action 取值，与《2-骨架与配置》进程通信一节的 action 表逐字一致。
# voice-start 与 voice-stop 是按住语义的一对动作：按下开始采集，
# 松开停止采集并冲刷最终文本提交；voice-toggle 保留向后兼容，
# CLI 手动调用仍可用，界面内不再有 F8 点击切换。
# project-mode-toggle 切换项目模式，可能触发模型懒加载；
# context-view-toggle 切换上下文视图显示/切回；context-clear 清空上下文、重开一轮。
ACTIONS: tuple[str, ...] = (
    "voice-toggle",
    "voice-start",
    "voice-stop",
    "screenshot",
    "project-mode-toggle",
    "context-view-toggle",
    "context-clear",
    "stop",
    "status",
    "warmup",
)

# JSON 序列化用的分隔符：紧凑输出，保证一行
_SEPARATORS = (",", ":")


@dataclass(frozen=True)
class Request:
    """一条请求。

    ``action`` 必须是 :data:`ACTIONS` 里的取值；``request_id`` 可选，
    为 None 时请求行里不写这个键。
    """

    action: str
    request_id: str | None = None


@dataclass(frozen=True)
class Response:
    """一条响应。

    ``ok`` 是布尔值，``error`` 是错误码字符串（成功时为空串），
    ``payload`` 是对象，缺省为空对象。
    """

    ok: bool
    error: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


def encode_request(action: str, request_id: str | None = None) -> str:
    """把请求编码成一行 JSON 文本，末尾带换行符。

    ``request_id`` 为 None 时不写这个键，其余情况原样写入。
    编码不做取值校验，越界的 ``action`` 由解码方报错。

    :returns: 单行 JSON 文本，形如 ``{"action":"status"}\\n``。
    """
    body: dict[str, Any] = {"action": action}
    if request_id is not None:
        body["request_id"] = request_id
    return json.dumps(body, ensure_ascii=False, separators=_SEPARATORS) + "\n"


def encode_response(
    ok: bool,
    error: str = "",
    payload: Mapping[str, Any] | None = None,
) -> str:
    """把响应编码成一行 JSON 文本，末尾带换行符。

    ``payload`` 为 None 时编码成空对象。

    :returns: 单行 JSON 文本，形如 ``{"ok":true,"error":"","payload":{}}\\n``。
    """
    body: dict[str, Any] = {
        "ok": ok,
        "error": error,
        "payload": dict(payload) if payload is not None else {},
    }
    return json.dumps(body, ensure_ascii=False, separators=_SEPARATORS) + "\n"


def decode_request(line: str) -> Request:
    """解析一行请求文本。

    非法输入一律抛 :class:`~app.core.exceptions.SocketError`，
    错误消息写明具体原因与实际类型：

    - 不是合法 JSON；
    - JSON 顶层不是对象；
    - 缺 ``action`` 字段；
    - ``action`` 不是字符串；
    - ``action`` 不在 :data:`ACTIONS` 里；
    - ``request_id`` 存在但不是字符串（值为 null 按缺省处理）。
    """
    obj = _load_object(line)
    if "action" not in obj:
        raise SocketError("请求缺少 action 字段")
    action = obj["action"]
    if not isinstance(action, str):
        raise SocketError(f"请求 action 必须是字符串，实际是 {type(action).__name__}")
    if action not in ACTIONS:
        legal = "、".join(ACTIONS)
        raise SocketError(f"未知 action：{action!r}，合法取值：{legal}")
    request_id = obj.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise SocketError(
            f"请求 request_id 必须是字符串，实际是 {type(request_id).__name__}"
        )
    return Request(action=action, request_id=request_id)


def decode_response(line: str) -> Response:
    """解析一行响应文本。

    非法输入一律抛 :class:`~app.core.exceptions.SocketError`，
    错误消息写明具体原因与实际类型：

    - 不是合法 JSON；
    - JSON 顶层不是对象；
    - 缺 ``ok`` 字段；
    - ``ok`` 不是布尔值；
    - ``error`` 存在但不是字符串；
    - ``payload`` 存在但不是对象。
    """
    obj = _load_object(line)
    if "ok" not in obj:
        raise SocketError("响应缺少 ok 字段")
    ok = obj["ok"]
    if not isinstance(ok, bool):
        raise SocketError(f"响应 ok 必须是布尔值，实际是 {type(ok).__name__}")
    if "error" in obj:
        error = obj["error"]
        if not isinstance(error, str):
            raise SocketError(f"响应 error 必须是字符串，实际是 {type(error).__name__}")
    else:
        error = ""
    if "payload" in obj:
        payload = obj["payload"]
        if not isinstance(payload, dict):
            raise SocketError(f"响应 payload 必须是对象，实际是 {type(payload).__name__}")
    else:
        payload = {}
    return Response(ok=ok, error=error, payload=payload)


def _load_object(line: str) -> dict[str, Any]:
    """把一行文本解析成 JSON 对象，失败时抛 SocketError。"""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SocketError(f"不是合法的 JSON 行：{exc}") from exc
    if not isinstance(obj, dict):
        raise SocketError(f"JSON 顶层必须是对象，实际是 {type(obj).__name__}")
    return obj
