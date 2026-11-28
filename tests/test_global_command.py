"""全局命令分发的测试。

覆盖《2-骨架与配置》进程通信一节的 action 表与《8-集成与验收》全局快捷键一节：

- 十个 action 各自走到对应的内部动作处理器；
- 处理器的返回值原样传出，socket 服务端可以直接拿它当响应 payload；
- 未知 action 与没有注册处理器时都不抛异常，只返回带结果标识的字典；
- 客户端可控的 action 写日志前先规整，伪造不出额外的日志字段。
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import pytest

from app.inputs.global_command import (
    ACTION_TO_INTERNAL,
    INTERNAL_ACTIONS,
    RESULT_NO_HANDLER,
    RESULT_UNKNOWN_ACTION,
    GlobalCommandDispatcher,
)
from app.runtime.socket_protocol import ACTIONS, Request

# 处理器返回的样例结果
HANDLER_RESULT = {"ok": True}


class RecordingHandler:
    """假处理器，记录收到的请求并返回固定结果。"""

    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.requests: list[Request] = []

    def __call__(self, request: Request) -> Any:
        self.requests.append(request)
        return self.result


class AsyncRecordingHandler:
    """假异步处理器，记录收到的请求并返回固定结果。"""

    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.requests: list[Request] = []

    async def __call__(self, request: Request) -> Any:
        self.requests.append(request)
        return self.result


@pytest.fixture
def project_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """收集 interviewmate 记录器上的日志。

    临时打开 propagate，让记录走到 pytest 的 handler 上。
    """
    project_logger = logging.getLogger("interviewmate")
    old_propagate = project_logger.propagate
    project_logger.propagate = True
    caplog.set_level(logging.INFO, logger="interviewmate")
    try:
        yield caplog
    finally:
        project_logger.propagate = old_propagate


@pytest.mark.parametrize("action", ACTIONS)
async def test_action_dispatches_to_its_handler(action: str) -> None:
    """每个 action 都走到自己对应的内部动作处理器。"""
    handler = RecordingHandler(HANDLER_RESULT)
    dispatcher = GlobalCommandDispatcher({ACTION_TO_INTERNAL[action]: handler})
    request = Request(action=action, request_id="req-1")

    result = await dispatcher.dispatch(request)

    assert handler.requests == [request]
    assert result == HANDLER_RESULT


@pytest.mark.parametrize("action", ACTIONS)
async def test_async_handler_result_is_passed_through(action: str) -> None:
    """异步处理器的结果被等完之后原样传出。"""
    handler = AsyncRecordingHandler(HANDLER_RESULT)
    dispatcher = GlobalCommandDispatcher({ACTION_TO_INTERNAL[action]: handler})

    result = await dispatcher.dispatch(Request(action=action))

    assert result == HANDLER_RESULT
    assert len(handler.requests) == 1


async def test_handler_receives_request_with_request_id() -> None:
    """处理器拿到的是原样的请求，request_id 也在里面。"""
    handler = RecordingHandler()
    dispatcher = GlobalCommandDispatcher({"status": handler})
    request = Request(action="status", request_id="req-42")

    await dispatcher.dispatch(request)

    assert handler.requests[0] is request
    assert handler.requests[0].request_id == "req-42"


async def test_unknown_action_returns_result_identifier(
    project_logs: pytest.LogCaptureFixture,
) -> None:
    """未知 action 不抛异常，记一条日志并返回明确的结果标识。"""
    dispatcher = GlobalCommandDispatcher()

    result = await dispatcher.dispatch(Request(action="teleport"))

    assert isinstance(result, dict)
    assert result["result"] == RESULT_UNKNOWN_ACTION
    assert result["action"] == "teleport"
    assert "teleport" in result["note"]

    events = [
        record
        for record in project_logs.records
        if getattr(record, "event", "") == "command.unknown_action"
    ]
    assert len(events) == 1
    assert events[0].levelno == logging.WARNING


async def test_unknown_action_text_is_sanitized_before_logging(
    project_logs: pytest.LogCaptureFixture,
) -> None:
    """客户端可控的 action 写日志前先规整，伪造不出额外的日志字段。"""
    dispatcher = GlobalCommandDispatcher()

    await dispatcher.dispatch(Request(action="bad action extra=1"))

    events = [
        record
        for record in project_logs.records
        if getattr(record, "event", "") == "command.unknown_action"
    ]
    assert len(events) == 1
    fields = getattr(events[0], "fields", {}) or {}
    assert fields["action"] == "bad-action-extra-1"


async def test_action_without_handler_returns_result_identifier(
    project_logs: pytest.LogCaptureFixture,
) -> None:
    """action 合法但没有注册处理器时也不抛异常，返回明确的结果标识。"""
    dispatcher = GlobalCommandDispatcher({"status": RecordingHandler()})

    result = await dispatcher.dispatch(Request(action="warmup"))

    assert isinstance(result, dict)
    assert result["result"] == RESULT_NO_HANDLER
    assert result["action"] == "warmup"

    events = [
        record
        for record in project_logs.records
        if getattr(record, "event", "") == "command.no_handler"
    ]
    assert len(events) == 1


async def test_register_adds_handler() -> None:
    """后注册的处理器照样能被分发到。"""
    dispatcher = GlobalCommandDispatcher()
    handler = RecordingHandler(HANDLER_RESULT)

    dispatcher.register("screenshot", handler)
    result = await dispatcher.dispatch(Request(action="screenshot"))

    assert result == HANDLER_RESULT
    assert len(handler.requests) == 1
    assert "screenshot" in dispatcher.handlers


async def test_socket_handlers_cover_all_protocol_actions() -> None:
    """给 socket 服务端用的映射覆盖协议里的全部 action。"""
    dispatcher = GlobalCommandDispatcher()

    handlers = dispatcher.socket_handlers()

    assert set(handlers) == set(ACTIONS)
    assert set(INTERNAL_ACTIONS) == set(ACTION_TO_INTERNAL.values())


async def test_socket_handlers_route_to_dispatcher() -> None:
    """socket 服务端拿到的映射走的是同一个分发入口。"""
    handler = RecordingHandler(HANDLER_RESULT)
    dispatcher = GlobalCommandDispatcher({"screenshot": handler})
    handlers = dispatcher.socket_handlers()

    result = await handlers["screenshot"](Request(action="screenshot"))

    assert result == HANDLER_RESULT
    assert len(handler.requests) == 1
