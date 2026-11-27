"""进程运行时（M1 W1d）的测试。

覆盖四块：

- socket 协议编解码：线格式与《2-骨架与配置》进程通信一节冻结的格式逐字一致，
  解码失败一律抛 ``SocketError``（``E_SOCKET``）；
- socket 服务端：正常分发、异步 handler、无 handler、handler 抛异常、协议错误；
- socket 客户端：连不上时抛 ``SocketError``，探测主进程是否在监听；
- 进程管理与健康检查：PID 文件读写、socket 文件残留清理、状态探测与状态文本。

socket 文件与 PID 文件一律放在 ``tmp_path`` 里，服务用完都 ``await server.stop()``，
不留残留文件。阻塞式客户端调用放在线程里跑，避免堵住服务端所在的事件循环。
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.exceptions import DifyTimeoutError, SocketError
from app.runtime.health_checker import HealthChecker, HealthStatus
from app.runtime.process_manager import (
    is_process_alive,
    is_running,
    prepare_socket_path,
    read_pid,
    remove_pid,
    write_pid,
)
from app.runtime.socket_client import is_server_alive, send_request
from app.runtime.socket_protocol import (
    ACTIONS,
    Request,
    Response,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)
from app.runtime.socket_server import SocketServer
from app.utils.logger import setup_logging


# --------------------------------------------------------------------------
# 协议编解码
# --------------------------------------------------------------------------


def test_encode_request_matches_frozen_line_format() -> None:
    """请求行的字面格式与冻结格式一致。"""
    assert encode_request("status", "req-1") == '{"action":"status","request_id":"req-1"}\n'


def test_encode_request_without_request_id_omits_key() -> None:
    """request_id 为 None 时请求行里没有这个键。"""
    line = encode_request("voice-toggle")
    assert line == '{"action":"voice-toggle"}\n'
    assert "request_id" not in line


def test_encode_response_matches_frozen_line_format() -> None:
    """响应行的字面格式与冻结格式一致。"""
    line = encode_response(True, "", {"code": "E_SOCKET"})
    assert line == '{"ok":true,"error":"","payload":{"code":"E_SOCKET"}}\n'


def test_encode_response_payload_defaults_to_empty_object() -> None:
    """payload 缺省时编码成空对象。"""
    assert encode_response(True) == '{"ok":true,"error":"","payload":{}}\n'
    assert encode_response(False, "E_SOCKET") == '{"ok":false,"error":"E_SOCKET","payload":{}}\n'


@pytest.mark.parametrize(
    "line",
    [
        encode_request("status", "req-1"),
        encode_request("stop"),
        encode_response(True, "", {"code": "E_SOCKET"}),
        encode_response(False, "E_SOCKET"),
    ],
)
def test_encoded_lines_are_single_line(line: str) -> None:
    """编码结果是一行，末尾恰好一个换行。"""
    assert line.count("\n") == 1
    assert line.endswith("\n")
    assert line[:-1].strip() != ""


def test_request_round_trip_preserves_fields() -> None:
    """请求编码再解码，字段原样回来。"""
    request = decode_request(encode_request("warmup", "req-9"))
    assert request == Request(action="warmup", request_id="req-9")


def test_response_round_trip_preserves_fields() -> None:
    """响应编码再解码，字段原样回来。"""
    response = decode_response(encode_response(True, "", {"code": "E_SOCKET", "n": 1}))
    assert response == Response(ok=True, error="", payload={"code": "E_SOCKET", "n": 1})


def test_decode_request_accepts_null_request_id_as_default() -> None:
    """request_id 显式为 null 按缺省处理。"""
    request = decode_request('{"action":"status","request_id":null}\n')
    assert request == Request(action="status", request_id=None)


def test_decode_request_rejects_invalid_json() -> None:
    """不是合法 JSON 时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_request("not json\n")
    assert excinfo.value.code == "E_SOCKET"


def test_decode_request_rejects_non_object_json() -> None:
    """JSON 顶层不是对象时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_request("[1, 2]\n")
    assert excinfo.value.code == "E_SOCKET"


def test_decode_request_rejects_missing_action() -> None:
    """缺 action 字段时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_request('{"request_id":"req-1"}\n')
    assert excinfo.value.code == "E_SOCKET"
    assert "action" in excinfo.value.message


def test_decode_request_rejects_non_string_action() -> None:
    """action 不是字符串时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_request('{"action":123}\n')
    assert excinfo.value.code == "E_SOCKET"
    assert "action" in excinfo.value.message


def test_decode_request_rejects_unknown_action() -> None:
    """action 不在 ACTIONS 里时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_request('{"action":"restart"}\n')
    assert excinfo.value.code == "E_SOCKET"
    assert "restart" in excinfo.value.message


def test_decode_request_rejects_non_string_request_id() -> None:
    """request_id 存在但不是字符串时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_request('{"action":"status","request_id":7}\n')
    assert excinfo.value.code == "E_SOCKET"
    assert "request_id" in excinfo.value.message


def test_decode_response_rejects_missing_ok() -> None:
    """缺 ok 字段时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_response('{"error":"","payload":{}}\n')
    assert excinfo.value.code == "E_SOCKET"
    assert "ok" in excinfo.value.message


def test_decode_response_rejects_non_bool_ok() -> None:
    """ok 不是布尔值时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_response('{"ok":"yes","error":"","payload":{}}\n')
    assert excinfo.value.code == "E_SOCKET"
    assert "ok" in excinfo.value.message


def test_decode_response_rejects_non_string_error() -> None:
    """error 存在但不是字符串时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_response('{"ok":true,"error":null,"payload":{}}\n')
    assert excinfo.value.code == "E_SOCKET"
    assert "error" in excinfo.value.message


def test_decode_response_rejects_non_object_payload() -> None:
    """payload 存在但不是对象时抛 SocketError。"""
    with pytest.raises(SocketError) as excinfo:
        decode_response('{"ok":true,"error":"","payload":[1]}\n')
    assert excinfo.value.code == "E_SOCKET"
    assert "payload" in excinfo.value.message


def test_decode_response_accepts_missing_error_and_payload() -> None:
    """error 与 payload 缺省时按默认值处理。"""
    response = decode_response('{"ok":true}\n')
    assert response == Response(ok=True, error="", payload={})


def test_actions_tuple_matches_frozen_values() -> None:
    """action 取值与《2-骨架与配置》进程通信一节一致。"""
    assert ACTIONS == ("voice-toggle", "screenshot", "stop", "status", "warmup")


# --------------------------------------------------------------------------
# 服务端
# --------------------------------------------------------------------------


async def test_server_returns_handler_payload(tmp_path: Path) -> None:
    """起服务后客户端发 status，响应 ok=True 且 payload 是 handler 返回的 dict。"""
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={"status": lambda request: {"running": "running"}})
    await server.start()
    try:
        response = await asyncio.to_thread(send_request, "status", socket_path=socket_path)
    finally:
        await server.stop()
    assert response.ok is True
    assert response.error == ""
    assert response.payload == {"running": "running"}


async def test_server_awaits_async_handler(tmp_path: Path) -> None:
    """handler 是协程函数时服务端会 await 它的返回值。"""

    async def handler(request: Request) -> dict[str, object]:
        await asyncio.sleep(0)
        return {"action": request.action, "echo": "async"}

    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={"status": handler})
    await server.start()
    try:
        response = await asyncio.to_thread(send_request, "status", socket_path=socket_path)
    finally:
        await server.stop()
    assert response.ok is True
    assert response.payload == {"action": "status", "echo": "async"}


async def test_server_returns_empty_payload_for_none_handler_result(tmp_path: Path) -> None:
    """handler 返回 None 时 payload 是空对象。"""
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={"stop": lambda request: None})
    await server.start()
    try:
        response = await asyncio.to_thread(send_request, "stop", socket_path=socket_path)
    finally:
        await server.stop()
    assert response.ok is True
    assert response.payload == {}


async def test_server_unknown_action_returns_e_socket(tmp_path: Path) -> None:
    """action 合法但没注册 handler 时回 ok=False 与 E_SOCKET。"""
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={})
    await server.start()
    try:
        response = await asyncio.to_thread(send_request, "status", socket_path=socket_path)
    finally:
        await server.stop()
    assert response.ok is False
    assert response.error == "E_SOCKET"
    assert response.payload == {"code": "E_SOCKET"}


async def test_server_handler_exception_returns_code_and_server_keeps_serving(tmp_path: Path) -> None:
    """handler 抛异常时回错误码，服务端不崩，下一个请求照常处理。"""

    def failing(request: Request) -> dict[str, object]:
        raise DifyTimeoutError("检索超时")

    socket_path = tmp_path / "s.sock"
    server = SocketServer(
        socket_path=socket_path,
        handlers={"warmup": failing, "status": lambda request: {"running": "running"}},
    )
    await server.start()
    try:
        failed = await asyncio.to_thread(send_request, "warmup", socket_path=socket_path)
        following = await asyncio.to_thread(send_request, "status", socket_path=socket_path)
    finally:
        await server.stop()
    assert failed.ok is False
    assert failed.error == "E_DIFY_TIMEOUT"
    assert failed.payload == {"code": "E_DIFY_TIMEOUT"}
    assert following.ok is True
    assert following.payload == {"running": "running"}


async def test_server_invalid_json_line_returns_error_and_closes_connection(tmp_path: Path) -> None:
    """客户端发非法 JSON 行时回 E_SOCKET，并把连接正常关闭。"""
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={"status": lambda request: {}})
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(b"this is not json\n")
        await writer.drain()
        raw = await reader.readline()
        assert raw.endswith(b"\n")
        response = decode_response(raw.decode("utf-8"))
        assert response.ok is False
        assert response.error == "E_SOCKET"
        assert response.payload == {"code": "E_SOCKET"}
        # 连接被服务端正常关闭，剩下的是 EOF
        assert await reader.read() == b""
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


async def test_server_start_on_occupied_socket_raises_socket_error(tmp_path: Path) -> None:
    """socket 文件被活进程占用时 start 抛 SocketError。"""
    socket_path = tmp_path / "s.sock"
    first = SocketServer(socket_path=socket_path, handlers={})
    await first.start()
    second = SocketServer(socket_path=socket_path, handlers={})
    try:
        with pytest.raises(SocketError) as excinfo:
            await second.start()
        assert excinfo.value.code == "E_SOCKET"
        # 占用者没被顶掉
        assert socket_path.exists()
    finally:
        await first.stop()


async def test_server_stop_removes_socket_file(tmp_path: Path) -> None:
    """stop 之后 socket 文件被删掉。"""
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={})
    await server.start()
    assert socket_path.exists()
    await server.stop()
    assert not socket_path.exists()


async def test_server_socket_file_permissions_are_owner_only(tmp_path: Path) -> None:
    """socket 文件创建后权限收紧到只有属主可读写（0o600）。"""
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={})
    await server.start()
    try:
        mode = socket_path.stat().st_mode & 0o777
        assert mode == 0o600
        assert mode & 0o077 == 0
    finally:
        await server.stop()


async def test_server_sanitizes_request_id_before_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """客户端可控的 request_id 不能伪造日志字段，业务侧拿到的仍是原值。"""
    seen: list[str | None] = []

    def handler(request: Request) -> dict[str, object]:
        seen.append(request.request_id)
        return {"running": "running"}

    log_path = setup_logging("INFO", log_dir=tmp_path / "logs")
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={"status": handler})
    caplog.set_level(logging.INFO, logger="interviewmate")
    project_logger = logging.getLogger("interviewmate")
    project_logger.addHandler(caplog.handler)
    try:
        await server.start()
        forged = "x event=forged level=ERROR"
        response = await asyncio.to_thread(
            send_request, "status", forged, socket_path=socket_path
        )
    finally:
        await server.stop()
        project_logger.removeHandler(caplog.handler)

    assert response.ok is True
    # 规整只作用于写日志的副本，handler 拿到的仍是客户端发来的原值
    assert seen == [forged]

    records = [r for r in caplog.records if r.name == "interviewmate"]
    request_record = next(
        r for r in records if getattr(r, "event", "") == "socket.request"
    )
    logged = getattr(request_record, "request_id", None)
    assert logged == "x-event-forged-level-ERROR"
    assert " " not in logged
    assert "=" not in logged

    # 渲染出来的日志行里没有伪造出来的字段
    lines = log_path.read_text(encoding="utf-8").splitlines()
    request_lines = [line for line in lines if "event=socket.request" in line]
    assert request_lines
    tokens = request_lines[-1].split()
    assert "event=forged" not in tokens
    assert "level=ERROR" not in tokens
    assert "request_id=x-event-forged-level-ERROR" in tokens


async def test_server_redacts_secret_in_request_id_before_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """request_id 里的 sk- 密钥先脱敏再规整，落盘搜不到原文。

    顺序不能反：脱敏规则 ``sk-`` 的负向前瞻不放行下划线相邻，
    先把空白换成 ``_`` 会让密钥绕开脱敏。
    """
    log_path = setup_logging("INFO", log_dir=tmp_path / "logs")
    socket_path = tmp_path / "s.sock"
    server = SocketServer(
        socket_path=socket_path,
        handlers={"status": lambda request: {"running": "running"}},
    )
    caplog.set_level(logging.INFO, logger="interviewmate")
    project_logger = logging.getLogger("interviewmate")
    project_logger.addHandler(caplog.handler)
    try:
        await server.start()
        response = await asyncio.to_thread(
            send_request, "status", "boom sk-SECRETVALUE456", socket_path=socket_path
        )
    finally:
        await server.stop()
        project_logger.removeHandler(caplog.handler)

    assert response.ok is True
    content = log_path.read_text(encoding="utf-8")
    # 密钥原文没有落盘
    assert "SECRETVALUE456" not in content
    lines = [line for line in content.splitlines() if "event=socket.request" in line]
    assert lines
    tokens = lines[-1].split()
    request_id_token = next(t for t in tokens if t.startswith("request_id="))
    # 规整仍然生效：值里没有空格，也没有 = 造成的额外字段
    value = request_id_token[len("request_id="):]
    assert " " not in value
    assert "=" not in value
    # 脱敏在前：密钥已变成星号，规整后是连字符
    assert request_id_token == "request_id=boom----"


async def test_server_request_id_secret_not_in_log_file(tmp_path: Path) -> None:
    """request_id 里的 sk- 密钥先脱敏再规整，落盘搜不到原文。

    顺序不能反：先规整成下划线会造出 ``sk-`` 规则不放行的相邻关系，
    让密钥绕开脱敏；规整用的占位符是连字符，不是下划线。
    """
    log_path = setup_logging("INFO", log_dir=tmp_path / "logs")
    socket_path = tmp_path / "s.sock"
    server = SocketServer(
        socket_path=socket_path,
        handlers={"status": lambda request: {"running": "running"}},
    )
    try:
        await server.start()
        response = await asyncio.to_thread(
            send_request, "status", "boom sk-TOPSECRETKEY", socket_path=socket_path
        )
    finally:
        await server.stop()

    assert response.ok is True
    content = log_path.read_text(encoding="utf-8")
    # 密钥原文没有落盘
    assert "TOPSECRETKEY" not in content
    lines = [line for line in content.splitlines() if "event=socket.request" in line]
    assert lines
    token = next(t for t in lines[-1].split() if t.startswith("request_id="))
    # 脱敏在前，规整在后；规整用连字符
    assert token == "request_id=boom----"
    assert "_" not in token[len("request_id="):]


async def test_server_logs_events_to_project_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """socket 服务端的日志走项目 logger，事件名符合 ``socket.*`` 命名。

    项目记录器是 ``interviewmate``（见 :func:`app.utils.logger.setup_logging`），
    它被设了 ``propagate = False``，根记录器收不到，所以把 pytest 的捕获
    handler 直接挂到项目记录器上。
    """
    socket_path = tmp_path / "s.sock"
    server = SocketServer(
        socket_path=socket_path,
        handlers={"status": lambda request: {"running": "running"}},
    )
    caplog.set_level(logging.INFO, logger="interviewmate")
    project_logger = logging.getLogger("interviewmate")
    project_logger.addHandler(caplog.handler)
    try:
        await server.start()
        response = await asyncio.to_thread(
            send_request, "status", "req-42", socket_path=socket_path
        )
    finally:
        await server.stop()
        project_logger.removeHandler(caplog.handler)

    assert response.ok is True
    records = [r for r in caplog.records if r.name == "interviewmate"]
    assert records, "项目 logger 没有收到 socket 服务端的日志"
    event_names = [getattr(r, "event", "") for r in records]
    assert "socket.accept" in event_names
    assert "socket.request" in event_names
    assert all(name.startswith("socket.") for name in event_names)
    request_record = next(
        r for r in records if getattr(r, "event", "") == "socket.request"
    )
    assert getattr(request_record, "request_id", None) == "req-42"
    assert getattr(request_record, "fields", {}).get("action") == "status"


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------


async def test_send_request_raises_when_socket_file_missing(tmp_path: Path) -> None:
    """socket 文件不存在时 send_request 抛 SocketError。"""
    missing = tmp_path / "missing.sock"
    with pytest.raises(SocketError) as excinfo:
        await asyncio.to_thread(send_request, "status", socket_path=missing)
    assert excinfo.value.code == "E_SOCKET"


async def test_is_server_alive_returns_false_when_socket_file_missing(tmp_path: Path) -> None:
    """socket 文件不存在时 is_server_alive 返回 False，不抛异常。"""
    assert is_server_alive(tmp_path / "missing.sock") is False


async def test_is_server_alive_returns_true_while_server_running(tmp_path: Path) -> None:
    """服务端在监听时 is_server_alive 返回 True。"""
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={})
    await server.start()
    try:
        assert is_server_alive(socket_path) is True
    finally:
        await server.stop()


async def test_is_server_alive_returns_false_after_server_stop(tmp_path: Path) -> None:
    """服务端停掉之后 is_server_alive 返回 False。"""
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={})
    await server.start()
    await server.stop()
    assert is_server_alive(socket_path) is False


# --------------------------------------------------------------------------
# 进程管理
# --------------------------------------------------------------------------


def test_write_pid_then_read_pid_returns_same_pid(tmp_path: Path) -> None:
    """write_pid 写进去的 PID 能被 read_pid 读回来。"""
    pid_path = tmp_path / "interviewmate.pid"
    write_pid(pid_path, 4321)
    assert read_pid(pid_path) == 4321
    assert pid_path.read_text(encoding="utf-8").strip() == "4321"


def test_write_pid_defaults_to_current_process(tmp_path: Path) -> None:
    """write_pid 不传 PID 时写当前进程 PID。"""
    pid_path = tmp_path / "interviewmate.pid"
    write_pid(pid_path)
    assert read_pid(pid_path) == os.getpid()


def test_read_pid_returns_none_for_missing_or_invalid_file(tmp_path: Path) -> None:
    """文件不存在或内容不是合法正整数时 read_pid 返回 None。"""
    assert read_pid(tmp_path / "missing.pid") is None
    broken = tmp_path / "broken.pid"
    broken.write_text("not-a-pid\n", encoding="utf-8")
    assert read_pid(broken) is None
    negative = tmp_path / "negative.pid"
    negative.write_text("-1\n", encoding="utf-8")
    assert read_pid(negative) is None


def test_is_process_alive_true_for_current_process() -> None:
    """当前进程一定活着。"""
    assert is_process_alive(os.getpid()) is True


def test_is_process_alive_false_for_dead_pid() -> None:
    """肯定不存在的 PID 判为不活着。"""
    assert is_process_alive(_dead_pid()) is False


def test_is_process_alive_false_for_non_positive_pid() -> None:
    """PID 小于等于 0 判为不活着。"""
    assert is_process_alive(0) is False
    assert is_process_alive(-1) is False


def test_is_running_true_for_live_pid(tmp_path: Path) -> None:
    """PID 文件指向活进程时 is_running 返回 True。"""
    pid_path = tmp_path / "interviewmate.pid"
    write_pid(pid_path, os.getpid())
    assert is_running(pid_path) is True


def test_is_running_false_for_dead_pid(tmp_path: Path) -> None:
    """PID 文件指向已退出进程时 is_running 返回 False。"""
    pid_path = tmp_path / "interviewmate.pid"
    write_pid(pid_path, _dead_pid())
    assert is_running(pid_path) is False


def test_is_running_false_when_pid_file_missing(tmp_path: Path) -> None:
    """PID 文件不存在时 is_running 返回 False。"""
    assert is_running(tmp_path / "missing.pid") is False


def test_remove_pid_does_not_fail_on_missing_file(tmp_path: Path) -> None:
    """remove_pid 删不存在的文件也不报错。"""
    pid_path = tmp_path / "missing.pid"
    remove_pid(pid_path)
    assert not pid_path.exists()


def test_remove_pid_deletes_existing_file(tmp_path: Path) -> None:
    """remove_pid 把 PID 文件删掉。"""
    pid_path = tmp_path / "interviewmate.pid"
    write_pid(pid_path, 1234)
    remove_pid(pid_path)
    assert not pid_path.exists()


def test_prepare_socket_path_returns_true_when_file_missing(tmp_path: Path) -> None:
    """socket 文件不存在时直接返回 True。"""
    assert prepare_socket_path(tmp_path / "missing.sock") is True


async def test_prepare_socket_path_returns_false_when_server_listening(tmp_path: Path) -> None:
    """有活服务在监听时返回 False，且不删掉 socket 文件。"""
    socket_path = tmp_path / "s.sock"
    server = SocketServer(socket_path=socket_path, handlers={})
    await server.start()
    try:
        assert prepare_socket_path(socket_path) is False
        assert socket_path.exists()
    finally:
        await server.stop()


def test_prepare_socket_path_removes_stale_regular_file(tmp_path: Path) -> None:
    """残留的普通文件会被删掉，并返回 True。"""
    stale = tmp_path / "stale.sock"
    stale.write_text("leftover", encoding="utf-8")
    assert prepare_socket_path(stale) is True
    assert not stale.exists()


# --------------------------------------------------------------------------
# 健康检查
# --------------------------------------------------------------------------


async def test_collect_returns_provider_values() -> None:
    """providers 正常时 collect 拿到对应值。"""
    checker = HealthChecker(
        {
            "audio": lambda: "on",
            "asr": lambda: "ready",
            "dify": lambda: "ready",
            "deepseek": lambda: "streaming",
            "ttft": lambda: 1234,
        }
    )
    status = await checker.collect()
    assert status.audio == "on"
    assert status.asr == "ready"
    assert status.dify == "ready"
    assert status.deepseek == "streaming"
    assert status.ttft_ms == 1234.0


async def test_collect_awaits_async_provider() -> None:
    """provider 是协程函数时 collect 会 await 它。"""

    async def audio_provider() -> str:
        await asyncio.sleep(0)
        return "on"

    checker = HealthChecker({"audio": audio_provider})
    status = await checker.collect()
    assert status.audio == "on"


async def test_collect_falls_back_when_provider_raises() -> None:
    """某个 provider 抛异常时该字段落兜底值，其它字段照常。"""

    def broken() -> str:
        raise RuntimeError("探测失败")

    checker = HealthChecker(
        {
            "audio": broken,
            "asr": lambda: "ready",
            "dify": lambda: "ready",
            "deepseek": lambda: "streaming",
            "ttft": lambda: 88,
        }
    )
    status = await checker.collect()
    assert status.audio == "off"
    assert status.asr == "ready"
    assert status.dify == "ready"
    assert status.deepseek == "streaming"
    assert status.ttft_ms == 88.0


async def test_collect_falls_back_on_invalid_values() -> None:
    """provider 返回取值表之外的值时落兜底值。"""
    checker = HealthChecker(
        {
            "audio": lambda: "loud",
            "asr": lambda: "ready",
            "dify": lambda: "unknown",
            "deepseek": lambda: "streaming",
            "ttft": lambda: "fast",
        }
    )
    status = await checker.collect()
    assert status.audio == "off"
    assert status.asr == "ready"
    assert status.dify == "degraded"
    assert status.deepseek == "streaming"
    assert status.ttft_ms is None


async def test_collect_keeps_running_for_caller_to_decide() -> None:
    """running 不从 providers 取，collect 保留缺省值。"""
    checker = HealthChecker({"running": lambda: "running"})
    status = await checker.collect()
    assert status.running == "stopped"


async def test_format_status_matches_expected_text() -> None:
    """状态文本的格式与《2-骨架与配置》命令行入口一节逐字一致。"""
    checker = HealthChecker()
    status = HealthStatus(
        running="running",
        audio="on",
        asr="ready",
        dify="ready",
        deepseek="streaming",
        ttft_ms=1234.0,
    )
    expected = (
        "InterviewMate: running\n"
        "PID: 4321\n"
        "\n"
        "Audio: on\n"
        "ASR: ready\n"
        "Dify: ready\n"
        "DeepSeek: streaming\n"
        "TTFT: 1234"
    )
    assert checker.format_status(status, pid=4321) == expected


async def test_format_status_without_pid_and_ttft() -> None:
    """没有 PID 与 TTFT 数据时写 -，末尾不额外加空行。"""
    checker = HealthChecker()
    expected = (
        "InterviewMate: stopped\n"
        "PID: -\n"
        "\n"
        "Audio: off\n"
        "ASR: idle\n"
        "Dify: offline\n"
        "DeepSeek: idle\n"
        "TTFT: -"
    )
    text = checker.format_status(HealthStatus())
    assert text == expected
    assert not text.endswith("\n")


async def test_format_status_rounds_ttft_to_integer_milliseconds() -> None:
    """TTFT 有小数时按整数毫秒显示。"""
    checker = HealthChecker()
    status = HealthStatus(ttft_ms=1234.6)
    assert checker.format_status(status).endswith("TTFT: 1235")


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------


def _dead_pid() -> int:
    """返回一个肯定已经不存在的 PID。

    起一个跑完就退出的子进程，它的 PID 随进程一起消失。
    """
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.wait()
    assert process.returncode == 0
    return process.pid
