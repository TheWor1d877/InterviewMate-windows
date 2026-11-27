"""Windows 侧传输层的单测。

覆盖《8-集成与验收》验收清单 Windows 平台一节：token 正确 / 错误 /
缺失、残留端口文件重建、端口被占，以及 ``E_SOCKET`` 三种失败细分
（端口文件不存在 / 连不上 / token 不匹配）都能触发。
"""

from __future__ import annotations

import asyncio
import socket
import threading
from pathlib import Path

import pytest

from app.platform.windows.transport import TcpLoopbackTransport


async def _echo_server(
    transport: TcpLoopbackTransport,
    received: list[bytes],
) -> None:
    """注册一个先读一行再回 ``pong`` 的回调，模拟协议层。"""

    async def on_connect(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            received.append(line)
            writer.write(b"pong\n")
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    await transport.start(on_connect)


def _recv_or_eof(client: socket.socket) -> bytes:
    """读一次数据；被拒连接在 Windows 上可能先收到 RST，此时也按 EOF 处理。

    服务端拒绝握手时关闭连接，如果接收缓冲还有未读数据，close 会发 RST，
    对端 ``recv`` 抛 ``ConnectionResetError``。产品行为本身正确（拒绝连接），
    这里把 RST 与干净 EOF 视为同一结果。
    """
    try:
        return client.recv(4096)
    except ConnectionResetError:
        return b""


async def _read_or_eof(reader: asyncio.StreamReader) -> bytes:
    """异步读一次；被拒连接的 RST 也按 EOF 处理（见 :func:`_recv_or_eof`）。"""
    try:
        return await reader.read(4096)
    except ConnectionResetError:
        return b""


async def test_tcp_transport_roundtrip(tmp_path: Path) -> None:
    """起服务、连上、发 token 后收发一行、停服务，两个文件最后都被删掉。"""
    endpoint = tmp_path / "interviewmate.port"
    transport = TcpLoopbackTransport(endpoint)
    received: list[bytes] = []

    assert transport.prepare_endpoint() is True
    await _echo_server(transport, received)
    try:
        # 端口文件与 token 文件都写出来了
        assert endpoint.exists()
        assert transport.token_path.exists()
        port = int(endpoint.read_text(encoding="utf-8").strip())
        assert 0 < port <= 65535
        token = transport.token_path.read_text(encoding="utf-8").strip()
        assert len(token) >= 32

        # 监听起来之后端点算被活进程占用
        assert transport.is_endpoint_in_use() is True

        # 客户端是同步 socket，放到线程里做，和命令行走同步客户端的情形一样
        def exchange() -> bytes:
            """connect 发掉 token 后照旧发 JSON 行、读一行。"""
            client = transport.connect(timeout=2.0)
            try:
                client.sendall(b"ping\n")
                return client.recv(4096)
            finally:
                client.close()

        data = await asyncio.to_thread(exchange)

        assert data == b"pong\n"
        # 协议层收到的只有业务请求，token 握手在传输层被吃掉
        assert received == [b"ping\n"]
    finally:
        await transport.stop()

    assert not endpoint.exists()
    assert not transport.token_path.exists()


async def test_tcp_transport_wrong_token_closes_connection(tmp_path: Path) -> None:
    """token 不匹配时服务端直接断开，客户端读不到任何响应。"""
    endpoint = tmp_path / "interviewmate.port"
    transport = TcpLoopbackTransport(endpoint)
    received: list[bytes] = []
    assert transport.prepare_endpoint() is True
    await _echo_server(transport, received)
    try:
        port = int(endpoint.read_text(encoding="utf-8").strip())
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"wrong-token\nping\n")
        await writer.drain()
        # token 不匹配，服务端关连接：读到 EOF
        assert await _read_or_eof(reader) == b""
        writer.close()
        # 协议层没收到任何请求
        assert received == []
    finally:
        await transport.stop()


async def test_tcp_transport_missing_token_closes_connection(tmp_path: Path) -> None:
    """连上不发 token（占用探测式的连接）直接被断开，协议层无感知。"""
    endpoint = tmp_path / "interviewmate.port"
    transport = TcpLoopbackTransport(endpoint)
    received: list[bytes] = []
    assert transport.prepare_endpoint() is True
    await _echo_server(transport, received)
    try:
        port = int(endpoint.read_text(encoding="utf-8").strip())
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        # 不发 token 就发业务请求
        writer.write(b"ping\n")
        await writer.drain()
        assert await _read_or_eof(reader) == b""
        writer.close()
        assert received == []
    finally:
        await transport.stop()


async def test_tcp_transport_stale_token_file_rejected_by_server(
    tmp_path: Path,
) -> None:
    """E_SOCKET 细分三（客户端路径）：token 文件内容与服务端 token 不匹配时被拒。

    走真实 ``connect()``：token 文件被换成陈旧内容（例如上次退出留下的），
    connect 把陈旧 token 发给服务端，服务端握手不通过断开连接，
    客户端 ``recv`` 读到 EOF，协议层无感知。
    """
    endpoint = tmp_path / "interviewmate.port"
    transport = TcpLoopbackTransport(endpoint)
    received: list[bytes] = []
    assert transport.prepare_endpoint() is True
    await _echo_server(transport, received)
    try:
        # token 文件被换成与服务端 token 不匹配的陈旧内容
        transport.token_path.write_text("陈旧的 token\n", encoding="utf-8")

        def exchange() -> bytes:
            """connect 发掉文件里的陈旧 token 后照旧发 JSON 行、读响应。"""
            client = transport.connect(timeout=2.0)
            try:
                try:
                    client.sendall(b"ping\n")
                except (ConnectionResetError, BrokenPipeError, OSError):
                    # 服务端已拒绝并关闭，后续读按 EOF 处理
                    pass
                return _recv_or_eof(client)
            finally:
                client.close()

        data = await asyncio.to_thread(exchange)

        # 服务端拒绝，客户端读到 EOF（上层 send_request 会抛 E_SOCKET）
        assert data == b""
        # 协议层没收到任何请求
        assert received == []
    finally:
        await transport.stop()


def test_tcp_transport_prepare_endpoint_removes_stale_files(
    tmp_path: Path,
) -> None:
    """端口文件存在但连不上时是残留文件，端口文件与 token 文件一起清掉。"""
    endpoint = tmp_path / "interviewmate.port"
    endpoint.write_text("59999\n", encoding="utf-8")
    token_path = tmp_path / "interviewmate.token"
    token_path.write_text("旧 token\n", encoding="utf-8")
    transport = TcpLoopbackTransport(endpoint)
    assert transport.is_endpoint_in_use() is False
    assert transport.prepare_endpoint() is True
    assert not endpoint.exists()
    assert not token_path.exists()


def _accept_and_close(blocker: socket.socket) -> None:
    """把连进来的占用探测连接都 accept 掉并立刻关闭，保持 backlog 干净。"""
    blocker.settimeout(0.2)
    while True:
        try:
            conn, _ = blocker.accept()
        except (socket.timeout, TimeoutError):
            continue
        except OSError:
            return
        conn.close()


def test_tcp_transport_endpoint_in_use_when_port_occupied(
    tmp_path: Path,
) -> None:
    """端口文件指向的端口上有活监听时，端点算被占用，不能重建。"""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(5)
    acceptor = threading.Thread(target=_accept_and_close, args=(blocker,), daemon=True)
    acceptor.start()
    port = blocker.getsockname()[1]
    try:
        endpoint = tmp_path / "interviewmate.port"
        endpoint.write_text(f"{port}\n", encoding="utf-8")
        transport = TcpLoopbackTransport(endpoint)
        assert transport.is_endpoint_in_use() is True
        assert transport.prepare_endpoint() is False
        assert endpoint.exists()
    finally:
        blocker.close()
        acceptor.join(timeout=2.0)


def test_tcp_transport_cleanup_is_idempotent(tmp_path: Path) -> None:
    """端点文件与 token 文件不存在时清理也不报错。"""
    endpoint = tmp_path / "interviewmate.port"
    token_path = tmp_path / "interviewmate.token"
    transport = TcpLoopbackTransport(endpoint)
    transport.cleanup_endpoint()
    endpoint.write_text("残留\n", encoding="utf-8")
    token_path.write_text("残留\n", encoding="utf-8")
    transport.cleanup_endpoint()
    assert not endpoint.exists()
    assert not token_path.exists()


def test_tcp_transport_restrict_permissions_is_noop(tmp_path: Path) -> None:
    """Windows 没有 POSIX 权限位，权限收紧是空操作。"""
    transport = TcpLoopbackTransport(tmp_path / "interviewmate.port")
    transport.restrict_permissions()


def test_tcp_transport_connect_missing_port_file(tmp_path: Path) -> None:
    """E_SOCKET 细分一：端口文件不存在，connect 抛 OSError。"""
    transport = TcpLoopbackTransport(tmp_path / "interviewmate.port")
    with pytest.raises(OSError, match="端口文件"):
        transport.connect(timeout=1.0)


def test_tcp_transport_connect_unreachable_port(tmp_path: Path) -> None:
    """E_SOCKET 细分二：端口文件在但没人监听，connect 抛 OSError。"""
    endpoint = tmp_path / "interviewmate.port"
    endpoint.write_text("1\n", encoding="utf-8")
    (tmp_path / "interviewmate.token").write_text("token\n", encoding="utf-8")
    transport = TcpLoopbackTransport(endpoint)
    with pytest.raises(OSError):
        transport.connect(timeout=1.0)


def test_tcp_transport_connect_invalid_port_file(tmp_path: Path) -> None:
    """端口文件内容不是合法端口号时也按连不上处理。"""
    endpoint = tmp_path / "interviewmate.port"
    endpoint.write_text("不是端口号\n", encoding="utf-8")
    transport = TcpLoopbackTransport(endpoint)
    with pytest.raises(OSError, match="端口号"):
        transport.connect(timeout=1.0)
    assert transport.is_endpoint_in_use() is False


def test_tcp_transport_connect_missing_token_file(tmp_path: Path) -> None:
    """token 文件缺失时 connect 抛 OSError，不算正常连接。"""
    endpoint = tmp_path / "interviewmate.port"
    endpoint.write_text("1\n", encoding="utf-8")
    transport = TcpLoopbackTransport(endpoint)
    with pytest.raises(OSError, match="token"):
        transport.connect(timeout=1.0)