"""主进程侧的进程通信监听服务。

见《2-骨架与配置》进程通信一节：客户端连上后发一行请求、读一行响应、断开连接。

传输层由平台门面提供（:func:`app.platform.get_platform`）：Windows 侧是 TCP
loopback，端口号写进端点文件，token 写进同目录的 ``.token`` 文件做握手。
本模块只管协议与分发，不关心端点细节。

每个连接的流程：读一行 → 按《2-骨架与配置》进程通信一节解析请求 →
找对应 action 的 handler → 调用 → 把 handler 返回的 dict 当作 payload 回一行响应 →
关闭连接。

异常一律不抛给服务端主循环：

- 请求解析失败：回 ``ok=false``、``error="E_SOCKET"``、``payload={"code": "E_SOCKET"}``；
- action 合法但没注册 handler：同上；
- handler 抛异常：回 ``ok=false``、``error`` 填异常自带的错误码（没有就用 ``E_SOCKET``）、
  ``payload={"code": ...}``。

日志走项目日志管道：统一用 :func:`app.utils.logger.log_event` 写，
事件名是点分小写的 ``socket.*``（``socket.start``、``socket.accept``、
``socket.request``、``socket.error``、``socket.stop``）。
不用标准库的 ``logging.getLogger(__name__)``，那条路走不到项目记录器
``interviewmate``，日志进不了日志文件也不过脱敏。

一处安全约束：客户端可控的 ``request_id`` 写日志前先脱敏再规整
（见 :func:`_sanitize_request_id`），日志行是空格分隔的 ``key=value``，
不规整会被当成额外的字段。端点权限由传输层在监听创建后收紧。
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping

from app.core.exceptions import SocketError
from app.platform import get_platform
from app.runtime.process_manager import SOCKET_PATH
from app.runtime.socket_protocol import Request, decode_request, encode_response
from app.utils.logger import log_event, redact_text

__all__ = ["SocketServer"]


# 协议层错误统一用的错误码，见《2-骨架与配置》错误码一节
_SOCKET_ERROR_CODE = "E_SOCKET"

# 只负责触发的快捷键动作：先回执，动作放后台跑，命令行侧不等整条链路。
# project-mode-toggle 可能触发模型懒加载，整条链路慢，也放后台；
# context-view-toggle 与 context-clear 是快操作，保持同步处理，响应原样返回。
_FIRE_AND_FORGET_ACTIONS = frozenset(
    {
        "voice-toggle",
        "voice-start",
        "voice-stop",
        "screenshot",
        "project-mode-toggle",
    }
)

# 日志里允许出现的字符：字母、数字、下划线、点、冒号、连字符
_LOG_SAFE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_.:-"
)

# handler 的类型：接收请求，返回值作为响应 payload，可以是协程函数
Handler = Callable[[Request], Any]


def _sanitize_request_id(request_id: str | None) -> str | None:
    """规整客户端传来的 request_id，防止它伪造日志字段。

    日志行是空格分隔的 ``key=value``，客户端可控的 request_id 里如果带空格或
    ``=``，渲染出来会被当成额外的字段。这里把不在白名单里的字符换成 ``-``。

    替换字符用 ``-`` 不用 ``_``：脱敏规则 ``sk-`` 的负向前瞻不放行下划线相邻，
    换成 ``_`` 会凭空造出这个相邻关系，让密钥绕开脱敏。

    顺序是**先脱敏、后规整**：脱敏先做，密钥此时已成 ``***``，
    随后的规整把 ``*`` 换成 ``-`` 无害。

    只作用于写日志的那个副本，客户端发来的原值不动（它还要用于业务关联）。

    None 与空字符串保持原样，写日志时是 ``-``。
    """
    if request_id is None:
        return None
    return "".join(
        ch if ch in _LOG_SAFE_CHARS else "-" for ch in redact_text(request_id)
    )


class SocketServer:
    """进程通信监听服务。

    传输层由平台门面给出：Windows 侧是 TCP loopback。协议与分发和传输层解耦。

    :param socket_path: 通信端点路径，默认仓库根 ``runtime/interviewmate.port``。
    :param handlers: action 到 handler 的映射，key 是《2-骨架与配置》进程通信一节的
        action 取值，value 接收 :class:`~app.runtime.socket_protocol.Request`，
        返回 dict 当作响应 payload。
    """

    def __init__(
        self,
        socket_path: Path = SOCKET_PATH,
        handlers: Mapping[str, Handler] | None = None,
    ) -> None:
        self._socket_path = Path(socket_path)
        self._handlers: dict[str, Handler] = dict(handlers or {})
        self._transport = get_platform().transport(self._socket_path)
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def socket_path(self) -> Path:
        """当前监听的通信端点路径。"""
        return self._socket_path

    async def start(self) -> None:
        """开始监听。

        启动前先检查端点：被活进程占用时抛
        :class:`~app.core.exceptions.SocketError`（退出码 1 由命令行侧处理），
        残留文件会被删掉重建。

        :raises SocketError: 通信端点被活进程占用，或监听创建失败。
        """
        if not self._transport.prepare_endpoint():
            log_event(
                "socket.error",
                "通信端点已被活进程占用",
                level="WARNING",
                path=str(self._socket_path),
            )
            raise SocketError(f"通信端点已被活进程占用：{self._socket_path}")
        try:
            await self._transport.start(self._handle_client)
        except OSError as exc:
            # 启动失败时端点由传输层清掉，不留半成品
            log_event(
                "socket.error",
                "创建通信监听失败",
                level="ERROR",
                path=str(self._socket_path),
                error=str(exc),
            )
            raise SocketError(f"创建通信监听失败：{self._socket_path}（{exc}）") from exc
        log_event("socket.start", "开始监听 socket", path=str(self._socket_path))

    async def stop(self) -> None:
        """停止监听并删掉通信端点。

        重复调用安全。
        """
        for task in list(self._background_tasks):
            task.cancel()
        # 传输层负责关监听与删端点文件
        await self._transport.stop()
        log_event("socket.stop", "停止监听 socket", path=str(self._socket_path))

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """处理一个连接：读一行请求、回一行响应、关闭连接。"""
        log_event("socket.accept", "接受 socket 连接")
        try:
            line = await reader.readline()
            if not line:
                # 客户端连上就断开，没有请求可回
                return
            text = line.decode("utf-8", errors="replace").strip()
            try:
                request = decode_request(text)
            except SocketError as exc:
                log_event(
                    "socket.error",
                    "请求解析失败",
                    level="WARNING",
                    error=str(exc),
                )
                await self._write_response(
                    writer,
                    ok=False,
                    error=_SOCKET_ERROR_CODE,
                    payload={"code": _SOCKET_ERROR_CODE},
                )
                return
            await self._dispatch(writer, request)
        except (ConnectionResetError, BrokenPipeError):
            # 客户端提前断开，忽略
            pass
        except Exception as exc:
            # 单个连接的异常不影响服务端主循环
            log_event(
                "socket.error",
                "处理 socket 连接时出错",
                level="ERROR",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        finally:
            await self._close_writer(writer)

    async def _dispatch(self, writer: asyncio.StreamWriter, request: Request) -> None:
        """按 action 找 handler 并回响应。"""
        handler = self._handlers.get(request.action)
        if handler is None:
            # action 合法但没有注册 handler
            log_event(
                "socket.error",
                "action 没有注册 handler",
                level="WARNING",
                action=request.action,
                request_id=_sanitize_request_id(request.request_id),
            )
            await self._write_response(
                writer,
                ok=False,
                error=_SOCKET_ERROR_CODE,
                payload={"code": _SOCKET_ERROR_CODE},
            )
            return
        if request.action in _FIRE_AND_FORGET_ACTIONS:
            log_event(
                "socket.request",
                "处理请求",
                action=request.action,
                request_id=_sanitize_request_id(request.request_id),
            )
            await self._write_response(
                writer,
                ok=True,
                error="",
                payload={"accepted": True, "action": request.action},
            )
            self._run_in_background(request, handler)
            return
        try:
            result = handler(request)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            code = self._error_code(exc)
            log_event(
                "socket.error",
                "handler 执行失败",
                level="WARNING",
                action=request.action,
                request_id=_sanitize_request_id(request.request_id),
                code=code,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            await self._write_response(
                writer,
                ok=False,
                error=code,
                payload={"code": code},
            )
            return
        log_event(
            "socket.request",
            "处理请求",
            action=request.action,
            request_id=_sanitize_request_id(request.request_id),
        )
        await self._write_response(writer, ok=True, error="", payload=self._as_payload(result))

    def _run_in_background(self, request: Request, handler: Handler) -> None:
        """把快捷键动作放后台执行，异常只记日志。"""

        async def _run() -> None:
            try:
                result = handler(request)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    "socket.background.failed",
                    "快捷键动作执行失败",
                    level="WARNING",
                    action=request.action,
                    request_id=_sanitize_request_id(request.request_id),
                    code=self._error_code(exc),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        task = asyncio.create_task(_run())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    @staticmethod
    def _error_code(exc: Exception) -> str:
        """取异常自带的错误码，没有就用 ``E_SOCKET``。"""
        code = getattr(exc, "code", None)
        if isinstance(code, str) and code:
            return code
        return _SOCKET_ERROR_CODE

    @staticmethod
    def _as_payload(result: Any) -> dict[str, Any]:
        """把 handler 的返回值整理成响应 payload。

        返回 dict 时原样用；返回 None 时用空对象；返回其它类型时放进 ``result`` 字段。
        """
        if result is None:
            return {}
        if isinstance(result, Mapping):
            return dict(result)
        return {"result": result}

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        *,
        ok: bool,
        error: str,
        payload: Mapping[str, Any],
    ) -> None:
        """写一行响应。"""
        line = encode_response(ok, error, payload)
        writer.write(line.encode("utf-8"))
        await writer.drain()

    @staticmethod
    async def _close_writer(writer: asyncio.StreamWriter) -> None:
        """关闭连接，写不出去时的错误忽略。"""
        try:
            writer.close()
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
