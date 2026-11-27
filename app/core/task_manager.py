"""异步任务的统一管理：创建、取消、过期、去重。

这一层是《8-集成与验收》「异步任务」「请求取消」「防止重复调用」「超时管理」
四节的落点，管的是 asyncio 任务本身：

- 按请求标识登记任务，同一标识不重复启动，见《8-集成与验收》防止重复调用一节；
- 取消单个任务、取消全部任务，取消时通知客户端关流，
  见《8-集成与验收》请求取消一节；
- 每个任务可以带超时时限，超过时限判为过期并取消，
  见《8-集成与验收》超时管理一节；
- 列出受管任务的状态，给状态展示与健康检查用。

本模块不认识模型客户端、不认识界面，不发网络请求，也不碰配置文件。
客户端侧的取消通知由外面注入的回调完成：core 层不 import llm 层与界面层的
实现类，只用协议与注入的回调，见《2-骨架与配置》目录结构约定一节。

超时语义（本模块选的是「定时器 + 检查」并用）：

- 登记任务时给了时限，模块就用事件循环的定时器在时限到点时取消任务本身，
  不依赖谁来查询。所以「请求不能放在一个无限等待的任务里」这条真的被满足：
  没人看这个任务，它也会被取消，见《8-集成与验收》超时管理一节。
- 同时保留「记开始时间 + 检查时判定」这条路：查询与列出任务时按已运行时长
  复核一次。协程卡在不让出执行权的阻塞点上时定时器的取消送不进去，
  这条路还能把过期状态标出来。
- 与 ``asyncio.wait_for`` 的差别：``wait_for`` 超时是把内层协程包进一个新任务，
  超时后抛 ``TimeoutError``，外层看到的是异常而不是取消；本模块取消的是任务本身，
  等这个任务的一方收到的是 ``CancelledError``，与
  ``QuestionManager`` 现有的取消语义一致。
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.utils.logger import log_event

__all__ = [
    "ClientCancelCallback",
    "TaskFactory",
    "TaskHandle",
    "TaskInfo",
    "TaskManager",
]


# 协程工厂：调用时返回这次任务要跑的 awaitable。
# 工厂在任务里调用，工厂自己抛的异常留在任务里，不影响登记动作。
TaskFactory = Callable[[], Awaitable[Any]]

# 客户端取消通知回调：参数是请求标识。
# 可以是同步函数，也可以是协程函数；协程会被 await。
# 由外面注入，见《2-骨架与配置》目录结构约定一节。
ClientCancelCallback = Callable[[str], Awaitable[None] | None]


@dataclass(frozen=True)
class TaskHandle:
    """一次注册的结果。

    :param request_id: 请求标识。
    :param task: 受管的 asyncio 任务。
    :param created: 这次是不是新建的任务。同一请求标识重复登记时是假，
        表示拿回的是已经在跑的那个任务，见《8-集成与验收》防止重复调用一节。
    :param timeout: 超时时限（秒），没有时限时是 None。
    """

    request_id: str
    task: asyncio.Task[Any]
    created: bool
    timeout: float | None


@dataclass(frozen=True)
class TaskInfo:
    """受管任务的基本信息，给状态展示与健康检查用。

    :param request_id: 请求标识。
    :param done: 任务是否已经结束。
    :param cancelled: 任务是否被取消（发过取消，或 asyncio 报了取消）。
    :param expired: 任务是否因超过时限被判为过期。
    :param elapsed_ms: 已运行时长（毫秒）。
    :param timeout: 超时时限（秒），没有时限时是 None。
    """

    request_id: str
    done: bool
    cancelled: bool
    expired: bool
    elapsed_ms: float
    timeout: float | None


@dataclass
class _Entry:
    """注册表里的一条记录。"""

    request_id: str
    task: asyncio.Task[Any]
    started_at: float
    timeout: float | None
    expired: bool = False
    cancel_requested: bool = False
    timer: asyncio.TimerHandle | None = None


class TaskManager:
    """受管任务的注册表。

    任务按请求标识登记。同一标识重复登记时拿回已经在跑的那个任务，不重复启动，
    见《8-集成与验收》防止重复调用一节。

    :param on_client_cancel: 客户端取消通知回调，参数是请求标识。
        可选：没有注入时不通知客户端，见《8-集成与验收》请求取消一节。
    :param clock: 时钟函数，返回秒。默认是单调时钟，
        用来算已运行时长与判定过期；测试里可以注入假时钟。
    """

    def __init__(
        self,
        *,
        on_client_cancel: ClientCancelCallback | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_client_cancel = on_client_cancel
        self._clock = clock
        self._entries: dict[str, _Entry] = {}

    # ------------------------------------------------------------------
    # 登记，见《8-集成与验收》异步任务与防止重复调用两节
    # ------------------------------------------------------------------
    def register(
        self,
        request_id: str,
        factory: TaskFactory,
        *,
        timeout: float | None = None,
    ) -> TaskHandle:
        """登记一个任务并返回句柄。

        同一请求标识已经在跑时拿回那个任务，不新建，句柄的 ``created`` 是假。
        上一个任务已经结束时可以重新登记同名请求，这次会新建任务。

        :param request_id: 请求标识，不能是空串。
        :param factory: 协程工厂，返回这次任务要跑的 awaitable。
            工厂在任务里调用，工厂抛的异常留在任务里。
        :param timeout: 超时时限（秒），必须是正数；不设时限时传 None。
            超过时限仍未结束的任务判为过期并取消，
            见《8-集成与验收》超时管理一节。
        :raises ValueError: 请求标识是空串，或时限不是正数。
        """
        if not request_id:
            raise ValueError("请求标识不能为空")
        if timeout is not None:
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError(f"超时时限必须是正的有限数，收到 {timeout!r}")

        existing = self._entries.get(request_id)
        if existing is not None and not existing.task.done():
            # 相同请求标识不重复启动，见《8-集成与验收》防止重复调用一节
            log_event(
                "task.register_dedup",
                "同一请求标识已有任务在跑，不重复启动",
                level="DEBUG",
                request_id=request_id,
            )
            return TaskHandle(
                request_id=request_id,
                task=existing.task,
                created=False,
                timeout=existing.timeout,
            )

        task = asyncio.create_task(
            self._run(factory),
            name=f"interviewmate-task:{request_id}",
        )
        entry = _Entry(
            request_id=request_id,
            task=task,
            started_at=self._clock(),
            timeout=timeout,
        )
        self._entries[request_id] = entry
        # 任务结束就撤掉时限定时器，正常结束的务不会被迟到的定时器误伤
        task.add_done_callback(self._make_done_callback(entry))
        if timeout is not None:
            # 定时器到点直接取消任务，不依赖谁来查询，
            # 见《8-集成与验收》超时管理一节
            loop = asyncio.get_running_loop()
            entry.timer = loop.call_later(timeout, self._on_deadline, entry)
        log_event(
            "task.registered",
            "已登记受管任务",
            level="DEBUG",
            request_id=request_id,
            timeout=timeout,
        )
        return TaskHandle(
            request_id=request_id,
            task=task,
            created=True,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # 取消，见《8-集成与验收》请求取消一节
    # ------------------------------------------------------------------
    async def cancel(self, request_id: str) -> bool:
        """取消一个受管任务。

        幂等：任务已经结束或这个请求标识根本没登记过都不抛异常。
        取消是 asyncio 的协作式取消：任务要在下一个挂起点才会真的结束，
        卡在不让出执行权的阻塞点上时取消送不进去，这是 asyncio 本身的限制。

        取消路径上会调用注入的客户端取消通知回调，关掉对应的响应流；
        没有注入回调时不通知。

        :param request_id: 请求标识。
        :return: 这个请求标识在注册表里时返回真，否则返回假。
        """
        entry = self._entries.get(request_id)
        if entry is None:
            log_event(
                "task.cancel_missing",
                "没有这个受管任务，不报错",
                level="DEBUG",
                request_id=request_id,
            )
            return False
        if not entry.task.done():
            entry.task.cancel()
            entry.cancel_requested = True
        await self._notify_client(request_id)
        log_event("task.cancelled", "已取消受管任务", request_id=request_id)
        return True

    async def cancel_all(self) -> int:
        """取消全部受管任务并清空注册表。

        每个被取消的请求都会通知客户端关流。
        没有任务时是空操作。

        :return: 被取消的任务数。
        """
        count = 0
        for request_id in list(self._entries):
            if await self.cancel(request_id):
                count += 1
        self._entries.clear()
        log_event("task.cancelled_all", "已取消全部受管任务", count=count)
        return count

    # ------------------------------------------------------------------
    # 过期与清扫，见《8-集成与验收》超时管理一节
    # ------------------------------------------------------------------
    def sweep(self) -> int:
        """显式清扫：复核过期状态，移除已结束的条目。

        注册表不清理会一直增长，所以任务结束后要扫一次。
        顺带把过期的任务取消掉。

        :return: 被移除的条目数。
        """
        self._check_expired()
        removed = 0
        for request_id, entry in list(self._entries.items()):
            if entry.task.done():
                del self._entries[request_id]
                removed += 1
        log_event("task.swept", "已清扫受管任务", level="DEBUG", removed=removed)
        return removed

    def list_tasks(self) -> list[TaskInfo]:
        """列出受管任务，顺带把过期的取消掉。

        查询这条路径也是「记开始时间 + 检查时判定」那条路的入口：
        任务卡在不可取消的阻塞点上时，这里还是能把过期状态标出来。

        :return: 受管任务的信息列表，顺序按登记顺序。
        """
        self._check_expired()
        return [self._describe(entry) for entry in self._entries.values()]

    def get(self, request_id: str) -> TaskInfo | None:
        """取一个受管任务的信息，顺带把过期的取消掉。

        :param request_id: 请求标识。
        :return: 任务信息，没有登记过这个请求时返回 None。
        """
        self._check_expired()
        entry = self._entries.get(request_id)
        if entry is None:
            return None
        return self._describe(entry)

    def get_task(self, request_id: str) -> asyncio.Task[Any] | None:
        """取受管的任务对象。

        :param request_id: 请求标识。
        :return: 任务对象，没有登记过这个请求时返回 None。
        """
        entry = self._entries.get(request_id)
        return entry.task if entry is not None else None

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    async def _run(self, factory: TaskFactory) -> Any:
        """跑任务本体。

        工厂在任务里调用，工厂自己抛的异常留在任务里，登记动作不受影响。
        """
        return await factory()

    def _make_done_callback(self, entry: _Entry) -> Callable[[Any], None]:
        """造一个任务结束时的回调，用来撤掉时限定时器。"""

        def _on_done(_task: Any) -> None:
            if entry.timer is not None:
                entry.timer.cancel()
                entry.timer = None

        return _on_done

    def _on_deadline(self, entry: _Entry) -> None:
        """时限到点，把任务判为过期并取消。"""
        if entry.task.done():
            return
        self._mark_expired(entry)

    def _mark_expired(self, entry: _Entry) -> None:
        """标记过期并取消任务。"""
        entry.expired = True
        entry.task.cancel()
        entry.cancel_requested = True
        log_event(
            "task.timeout",
            "任务超过时限，已取消",
            level="WARNING",
            request_id=entry.request_id,
            timeout=entry.timeout,
        )

    def _check_expired(self) -> None:
        """按已运行时长复核一遍过期状态。

        定时器到点时已经取消过一次，这里是兜底：定时器的取消送不进去时
        （任务卡在不让出执行权的阻塞点上），过期状态照样能被标出来。
        """
        now = self._clock()
        for entry in self._entries.values():
            if entry.timeout is None or entry.expired or entry.task.done():
                continue
            if now - entry.started_at >= entry.timeout:
                self._mark_expired(entry)

    def _describe(self, entry: _Entry) -> TaskInfo:
        """把一条记录整理成任务信息。"""
        task = entry.task
        return TaskInfo(
            request_id=entry.request_id,
            done=task.done(),
            cancelled=entry.cancel_requested or task.cancelled(),
            expired=entry.expired,
            elapsed_ms=(self._clock() - entry.started_at) * 1000.0,
            timeout=entry.timeout,
        )

    async def _notify_client(self, request_id: str) -> None:
        """通知客户端取消对应请求。

        没有注入回调时不调用。回调是协程时 await 它；回调失败只记日志，
        不影响取消语义，见《8-集成与验收》请求取消一节。
        """
        callback = self._on_client_cancel
        if callback is None:
            return
        try:
            result = callback(request_id)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            log_event(
                "task.client_cancel_failed",
                f"客户端取消通知失败：{exc}",
                level="WARNING",
                request_id=request_id,
            )
