"""事件总线的测试。

覆盖冻结的分发语义：分发键、订阅顺序、重复订阅幂等、取消订阅、
同步与异步 handler、异常隔离、空操作、事件不被修改。

publish 是协程，全部测试写成 ``async def``，由 pytest-asyncio 的 auto 模式驱动。
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
from pydantic import ValidationError

from app.core.event_bus import Event, EventBus, now_ts

# 事件类型名，测试里统一用这些字面量，避免拼写漂移
TYPE_A = "audio.state_changed"
TYPE_B = "screenshot.captured"
TYPE_GENERIC = "test.generic"

# 被测日志记录器的名字
BUS_LOGGER = "app.core.event_bus"


class AudioStateChanged(Event):
    """带 type 字段的事件子类。"""

    type: str = TYPE_A
    state: str = "off"


class ScreenshotCaptured(Event):
    """另一个带 type 字段的事件子类。"""

    type: str = TYPE_B
    path: str = "/tmp/shot.png"


class GenericEvent(Event):
    """通用测试事件，带 type 字段。"""

    type: str = TYPE_GENERIC


class PlainEvent(Event):
    """没有 type 字段的事件子类，按类名分发。"""


async def test_publish_delivers_event_with_all_fields() -> None:
    """订阅后 publish 能收到事件，ts/request_id/session_id 原样送达。"""
    bus = EventBus()
    received: list[Event] = []

    def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(TYPE_GENERIC, handler)
    event = GenericEvent(ts=1234.5, request_id="req-1", session_id="sess-1")
    await bus.publish(event)

    assert len(received) == 1
    got = received[0]
    assert got.ts == 1234.5
    assert got.request_id == "req-1"
    assert got.session_id == "sess-1"


async def test_publish_delivers_the_same_event_object() -> None:
    """投递给 handler 的是同一个事件对象，不是复制品。"""
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe(TYPE_GENERIC, lambda event: received.append(event))

    event = GenericEvent(ts=1.0)
    await bus.publish(event)

    assert len(received) == 1
    assert received[0] is event


async def test_subclass_with_type_field_dispatches_by_type() -> None:
    """带 type 字段的子类事件按 type 字段分发。"""
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe(TYPE_A, lambda event: received.append(event))

    await bus.publish(AudioStateChanged(ts=10.0, state="on"))

    assert len(received) == 1
    assert isinstance(received[0], AudioStateChanged)
    assert received[0].state == "on"


async def test_event_without_type_field_dispatches_by_class_name() -> None:
    """没有 type 字段的事件按类名分发。"""
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe("PlainEvent", lambda event: received.append(event))

    await bus.publish(PlainEvent(ts=20.0))

    assert len(received) == 1
    assert isinstance(received[0], PlainEvent)


async def test_base_event_dispatches_by_base_class_name() -> None:
    """基类 Event 自身没有 type 字段，按类名 ``Event`` 分发。"""
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe("Event", lambda event: received.append(event))

    await bus.publish(Event(ts=5.0))

    assert len(received) == 1
    assert received[0].ts == 5.0


async def test_handlers_called_in_subscription_order() -> None:
    """同一事件类型的多个 handler 按订阅顺序被调用。"""
    bus = EventBus()
    calls: list[str] = []

    bus.subscribe(TYPE_GENERIC, lambda event: calls.append("first"))
    bus.subscribe(TYPE_GENERIC, lambda event: calls.append("second"))
    bus.subscribe(TYPE_GENERIC, lambda event: calls.append("third"))

    await bus.publish(GenericEvent(ts=1.0))

    assert calls == ["first", "second", "third"]


async def test_duplicate_subscribe_delivers_once() -> None:
    """重复 subscribe 同一 handler 只投递一次。"""
    bus = EventBus()
    calls: list[Event] = []

    def handler(event: Event) -> None:
        calls.append(event)

    bus.subscribe(TYPE_GENERIC, handler)
    bus.subscribe(TYPE_GENERIC, handler)

    await bus.publish(GenericEvent(ts=1.0))

    assert len(calls) == 1


async def test_unsubscribe_stops_delivery() -> None:
    """unsubscribe 之后不再投递。"""
    bus = EventBus()
    calls: list[str] = []

    def handler(event: Event) -> None:
        calls.append("called")

    bus.subscribe(TYPE_GENERIC, handler)
    bus.unsubscribe(TYPE_GENERIC, handler)

    await bus.publish(GenericEvent(ts=1.0))

    assert calls == []


async def test_unsubscribe_unregistered_handler_is_silent() -> None:
    """取消没注册过的 handler、没订阅过的事件类型，都不报错也不误删。"""
    bus = EventBus()
    calls: list[str] = []

    def handler(event: Event) -> None:
        calls.append("called")

    def stranger(event: Event) -> None:
        calls.append("stranger")

    # 事件类型从来没被订阅过
    bus.unsubscribe("never-subscribed", handler)
    # 事件类型订阅过，但这个 handler 没注册
    bus.subscribe(TYPE_GENERIC, handler)
    bus.unsubscribe(TYPE_GENERIC, stranger)
    # 原有 handler 仍在，发布时照常收到
    await bus.publish(GenericEvent(ts=1.0))

    assert calls == ["called"]


async def test_async_handler_is_awaited() -> None:
    """异步 handler 被正确 await，publish 返回时其内部副作用已经生效。"""
    bus = EventBus()
    received: list[str] = []

    async def handler(event: Event) -> None:
        await asyncio.sleep(0)
        received.append("done")

    bus.subscribe(TYPE_GENERIC, handler)
    await bus.publish(GenericEvent(ts=1.0))

    assert received == ["done"]


async def test_sync_and_async_handlers_run_in_subscription_order() -> None:
    """同步与异步 handler 混用时仍按订阅顺序执行。"""
    bus = EventBus()
    calls: list[str] = []

    async def async_first(event: Event) -> None:
        await asyncio.sleep(0)
        calls.append("async")

    def sync_second(event: Event) -> None:
        calls.append("sync")

    bus.subscribe(TYPE_GENERIC, async_first)
    bus.subscribe(TYPE_GENERIC, sync_second)

    await bus.publish(GenericEvent(ts=1.0))

    assert calls == ["async", "sync"]


async def test_failing_handler_does_not_stop_others() -> None:
    """某个 handler 抛异常时，其它 handler 仍然被调用，publish 自身不抛异常。"""
    bus = EventBus()
    calls: list[str] = []

    def boom(event: Event) -> None:
        calls.append("boom")
        raise RuntimeError("handler failed")

    def after(event: Event) -> None:
        calls.append("after")

    bus.subscribe(TYPE_GENERIC, boom)
    bus.subscribe(TYPE_GENERIC, after)

    # publish 自身不抛异常，异常被隔离
    await bus.publish(GenericEvent(ts=1.0))

    assert calls == ["boom", "after"]


async def test_failing_handler_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """handler 抛出的异常用标准库 logging 记录下来，且带上异常信息。"""
    bus = EventBus()

    def boom(event: Event) -> None:
        raise ValueError("boom")

    bus.subscribe(TYPE_GENERIC, boom)

    with caplog.at_level(logging.ERROR, logger=BUS_LOGGER):
        await bus.publish(GenericEvent(ts=1.0))

    records = [record for record in caplog.records if record.name == BUS_LOGGER]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info is not None


async def test_publish_without_subscribers_is_noop() -> None:
    """没有订阅者时 publish 是空操作。"""
    bus = EventBus()
    other: list[Event] = []

    # 订阅了别的类型，不该收到
    bus.subscribe("other-type", lambda event: other.append(event))

    await bus.publish(GenericEvent(ts=1.0))
    await bus.publish(Event(ts=1.0))

    assert other == []


async def test_publish_does_not_mutate_event() -> None:
    """publish 前后事件字段完全一致。"""
    bus = EventBus()
    bus.subscribe(TYPE_A, lambda event: None)

    event = AudioStateChanged(ts=99.25, request_id="req-9", session_id="sess-9", state="on")
    before = event.model_dump()

    await bus.publish(event)

    assert event.model_dump() == before
    assert event.ts == 99.25
    assert event.request_id == "req-9"
    assert event.session_id == "sess-9"
    assert event.state == "on"


async def test_unrelated_event_types_do_not_interfere() -> None:
    """订阅 A 只收到 A 的事件。"""
    bus = EventBus()
    got_a: list[Event] = []
    got_b: list[Event] = []

    bus.subscribe(TYPE_A, lambda event: got_a.append(event))
    bus.subscribe(TYPE_B, lambda event: got_b.append(event))

    await bus.publish(AudioStateChanged(ts=1.0))
    assert len(got_a) == 1
    assert got_b == []

    await bus.publish(ScreenshotCaptured(ts=2.0))
    assert len(got_a) == 1
    assert len(got_b) == 1
    assert got_b[0].path == "/tmp/shot.png"


async def test_handler_return_value_is_ignored() -> None:
    """handler 的返回值一律忽略，publish 自身返回 None。"""
    bus = EventBus()
    bus.subscribe(TYPE_GENERIC, lambda event: "返回值")

    assert await bus.publish(GenericEvent(ts=1.0)) is None


def test_now_ts_returns_current_time() -> None:
    """now_ts 返回接近当前时间的浮点数。"""
    before = time.time()
    value = now_ts()
    after = time.time()

    assert isinstance(value, float)
    assert before <= value <= after


def test_event_requires_ts() -> None:
    """ts 必填，没有默认值；request_id 与 session_id 默认是 None。"""
    with pytest.raises(ValidationError):
        Event()

    event = Event(ts=1.0)
    assert event.request_id is None
    assert event.session_id is None


def test_event_subclass_can_add_fields() -> None:
    """子类可以自由增加字段，基类只有 ts/request_id/session_id 三个字段。"""
    assert set(Event.model_fields) == {"ts", "request_id", "session_id"}

    event = AudioStateChanged(ts=1.0, state="on")
    assert event.type == TYPE_A
    assert event.state == "on"
