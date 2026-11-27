"""命令行侧的进程通信客户端。

见《2-骨架与配置》进程通信一节：客户端连上后发一行请求、读一行响应、断开连接。

连接由平台传输层负责（:func:`app.platform.get_platform`）：Linux 侧是 AF_UNIX
socket 文件，Windows 侧是 TCP loopback。本模块只管请求与响应的读写。

连不上（端点不存在、连接被拒、超时）与响应解析失败都抛
:class:`app.core.exceptions.SocketError`（错误码 ``E_SOCKET``）。
"""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import SocketError
from app.platform import get_platform
from app.runtime.process_manager import SOCKET_PATH
from app.runtime.socket_protocol import Response, decode_response, encode_request

__all__ = ["send_request", "is_server_alive"]


# 一次连接最多读多少字节，响应是一行，足够用
_READ_CHUNK_SIZE = 4096


def send_request(
    action: str,
    request_id: str | None = None,
    *,
    socket_path: Path = SOCKET_PATH,
    timeout: float = 5.0,
) -> Response:
    """发一条请求并读回响应。

    连上服务端后发一行请求、读一行响应、断开连接。

    :param action: action 取值，见 :data:`app.runtime.socket_protocol.ACTIONS`。
    :param request_id: 可选的请求 ID。
    :param socket_path: 通信端点路径。
    :param timeout: 连接与读写的超时秒数。
    :returns: 解析后的响应。
    :raises SocketError: 连不上服务端、读写超时，或响应不是合法协议内容。
    """
    line = encode_request(action, request_id)
    payload = line.encode("utf-8")
    chunks: list[bytes] = []
    transport = get_platform().transport(socket_path)
    try:
        client = transport.connect(timeout=timeout)
    except OSError as exc:
        raise SocketError(f"IPC 通信失败：{socket_path}（{exc}）") from exc
    try:
        client.sendall(payload)
        while True:
            chunk = client.recv(_READ_CHUNK_SIZE)
            if not chunk:
                # 对端关掉了连接
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    except OSError as exc:
        raise SocketError(f"IPC 通信失败：{socket_path}（{exc}）") from exc
    finally:
        client.close()

    if not chunks:
        raise SocketError(f"主进程没有返回响应：{socket_path}")
    raw = b"".join(chunks).split(b"\n", 1)[0]
    text = raw.decode("utf-8", errors="replace")
    # 响应不合法时 decode_response 会抛 SocketError
    return decode_response(text)


def is_server_alive(socket_path: Path = SOCKET_PATH) -> bool:
    """判断主进程是不是在监听。

    连得上返回 True，连不上返回 False，不抛异常。

    :param socket_path: 通信端点路径。
    """
    return get_platform().transport(socket_path).is_endpoint_in_use()
