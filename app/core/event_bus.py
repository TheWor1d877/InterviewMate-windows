"""模块间事件总线。

跨层调用只走事件总线，不直接 import 别的层的实现类，
见《2-骨架与配置》目录结构约定一节。本模块只做模块之间的事件投递：
订阅、取消订阅、发布。

职责边界：本总线不承载模型流式事件。模型返回的 ``framework`` / ``item`` /
``section_done`` / ``done`` 这套 JSONL 事件由《3-文本输入与流式回答》流式事件协议
一节定义的解析器与事件分发器负责，本模块不实现那套事件模型。

分发语义（后续模块依赖，不得随意改动）：

1. 分发键取 ``getattr(event, "type", None)``；取不到时回退成 ``type(event).__name__``。
   具体事件写成 :class:`Event` 的子类，用 ``type`` 字段声明自己的事件类型名，
   例如 ``class AudioStateChanged(Event): type: str = "audio.state_changed"``。
2. handler 可以是同步函数，也可以是异步协程函数。同步的直接调用；异步的 await
   其返回值。handler 的返回值一律忽略。
3. 同一事件类型内的 handler 按订阅顺序依次调用。
4. 同一 ``(event_type, handler)`` 重复订阅是幂等的，只保留一份，
   同一次 publish 不会投递两次。
5. ``unsubscribe`` 移除指定 handler；该 handler 未注册时静默返回，不报错。
6. 异常隔离：某个 handler 抛异常时，不影响其它 handler 继续执行，也不让
   ``publish`` 抛出，失败用标准库 logging 记录。
7. 没有订阅者的事件类型，``publish`` 是空操作。
8. ``publish`` 不修改传入的事件对象。``ts`` 由事件构造方填写，必填，无默认值，
   可以用 :func:`now_ts` 取当前时间。
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Callable

from pydantic import BaseModel

__all__ = ["Event", "EventBus", "now_ts"]


class Event(BaseModel):
    """事件基类。

    只有三个字段：``ts`` 必填，是事件产生时刻的时间戳（秒，浮点）；
    ``request_id`` 与 ``session_id`` 可选，用于把事件挂到请求与会话上下文上。

    子类可以自由增加字段，也可以加一个 ``type: str`` 字段声明自己的事件类型名。
    """

    ts: float
    request_id: str | None = None
    session_id: str | None = None


# handler 签名：接收一个事件，返回值被忽略。异步 handler 返回 awaitable。
Handler = Callable[[Event], Any]


class EventBus:
    """事件总线。

    按事件类型保存 handler 列表，publish 时按订阅顺序依次调用。
    订阅表是普通字典，不做跨线程同步，约定在同一个事件循环里使用。
    """

    def __init__(self) -> None:
        # 事件类型 -> 按订阅顺序排列的 handler 列表
        self._handlers: dict[str, list[Handler]] = {}

    def subscribe(self, event_type: str, handler: Handler) -> None:
        """订阅事件类型。

        同一 ``(event_type, handler)`` 重复订阅是幂等的，只保留一份。
        """
        handlers = self._handlers.setdefault(event_type, [])
        if handler not in handlers:
            handlers.append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        """取消订阅。

        该 handler 未注册时静默返回，不报错。
        """
        handlers = self._handlers.get(event_type)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            # 没注册过，静默返回
            return

    async def publish(self, event: Event) -> None:
        """发布事件，按订阅顺序把事件投递给该类型的所有 handler。

        没有订阅者时是空操作。单个 handler 抛异常不影响其它 handler，
        也不影响 ``publish`` 本身。传入的事件对象不会被修改。
        """
        event_type = self._resolve_event_type(event)
        handlers = self._handlers.get(event_type)
        if not handlers:
            return
        # 复制一份再遍历：handler 内部增删订阅不改变这一轮的分发对象
        for handler in list(handlers):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # 异常隔离：记录失败后继续投递给后面的 handler
                logging.getLogger(__name__).exception(
                    "事件处理器执行失败：event_type=%s handler=%r", event_type, handler
                )

    @staticmethod
    def _resolve_event_type(event: Event) -> str:
        """取事件的分发键。

        优先用事件自己的 ``type`` 字段，取不到时回退成类名。
        """
        event_type = getattr(event, "type", None)
        if isinstance(event_type, str) and event_type:
            return event_type
        return type(event).__name__


def now_ts() -> float:
    """返回当前时间戳（秒，浮点），供构造事件时填 ``ts``。"""
    return time.time()
