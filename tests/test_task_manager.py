"""受管任务的测试。

覆盖《8-集成与验收》「异步任务」「请求取消」「防止重复调用」「超时管理」四节：
登记后能按请求标识取消、同一标识不重复启动、取消已结束任务幂等不抛、
超时任务被判过期并取消、清扫移除已结束任务、取消全部后列表为空、
取消时通知注入的客户端回调。

只验证任务管理这一层的行为，不碰模型客户端与界面。
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.task_manager import TaskManager

# 超时用例的时限（秒），留出余量够事件循环走完一次取消
TIMEOUT_SECONDS = 0.05


async def _sleep_forever() -> None:
    """挂住不动的任务体。"""
    await asyncio.sleep(3600)


async def _immediate() -> str:
    """立刻结束的任务体。"""
    return "done"


@pytest.fixture
async def manager():
    """一个没有注入客户端回调的受管任务管理器。

    收尾时取消全部任务并等它们结束，别把挂着的任务留给事件循环关闭。
    """
    tm = TaskManager()
    try:
        yield tm
    finally:
        ids = [info.request_id for info in tm.list_tasks()]
        tasks = [task for task in (tm.get_task(rid) for rid in ids) if task is not None]
        await tm.cancel_all()
        if tasks:
            await asyncio.wait(tasks)


async def test_register_creates_task_and_cancel_stops_it(manager):
    """登记后能按请求标识取消，任务真的被取消。"""
    started = asyncio.Event()

    async def worker() -> None:
        started.set()
        await asyncio.sleep(3600)

    handle = manager.register("req-1", worker)
    assert handle.created is True
    assert handle.task is not None
    await started.wait()

    assert await manager.cancel("req-1") is True
    await asyncio.wait([handle.task])
    assert handle.task.cancelled() is True


async def test_duplicate_request_id_does_not_start_twice(manager):
    """同一请求标识重复登记时拿回已有任务，不重复启动。"""
    calls: list[int] = []
    started = asyncio.Event()

    def factory():
        calls.append(1)

        async def worker() -> None:
            started.set()
            await asyncio.sleep(3600)

        return worker()

    first = manager.register("req-1", factory)
    second = manager.register("req-1", factory)

    assert second.created is False
    assert second.task is first.task
    await started.wait()
    assert len(calls) == 1
    assert len(manager.list_tasks()) == 1


async def test_cancel_finished_task_is_idempotent(manager):
    """取消已结束的任务与不存在的请求都不抛异常。"""
    handle = manager.register("req-1", _immediate)
    await handle.task

    assert await manager.cancel("req-1") is True
    # 再取消一次同一个标识也不抛
    assert await manager.cancel("req-1") is True
    # 没登记过的请求标识也不抛
    assert await manager.cancel("req-unknown") is False


async def test_timeout_marks_expired_and_cancels_task(manager):
    """超过时限的任务被判为过期并被取消。"""
    handle = manager.register("req-1", _sleep_forever, timeout=TIMEOUT_SECONDS)
    await asyncio.sleep(TIMEOUT_SECONDS * 4)
    await asyncio.wait([handle.task])

    assert handle.task.cancelled() is True
    info = manager.get("req-1")
    assert info is not None
    assert info.expired is True


async def test_query_cancels_expired_task():
    """查询时按已运行时长复核，过期的任务被取消。

    这里注入假时钟把时间快进，走的是「记开始时间 + 检查时判定」那条路。
    """
    now = [0.0]
    tm = TaskManager(clock=lambda: now[0])
    handle = tm.register("req-1", _sleep_forever, timeout=10.0)
    now[0] = 11.0

    infos = tm.list_tasks()
    assert len(infos) == 1
    assert infos[0].expired is True

    await asyncio.wait([handle.task])
    assert handle.task.cancelled() is True
    await tm.cancel_all()


async def test_sweep_removes_finished_tasks(manager):
    """清扫能移除已结束的条目，注册表不无限增长。"""
    handle = manager.register("req-1", _immediate)
    await handle.task
    assert len(manager.list_tasks()) == 1

    assert manager.sweep() == 1
    assert manager.list_tasks() == []


async def test_cancel_all_empties_registry(manager):
    """取消全部任务后列表为空，任务也真的被取消。"""
    first = manager.register("req-1", _sleep_forever)
    second = manager.register("req-2", _sleep_forever)

    assert await manager.cancel_all() == 2
    assert manager.list_tasks() == []

    await asyncio.wait([first.task, second.task])
    assert first.task.cancelled() is True
    assert second.task.cancelled() is True


async def test_cancel_calls_injected_client_callback():
    """取消时调用注入的客户端通知回调，请求标识传进去。"""
    notified: list[str] = []
    tm = TaskManager(on_client_cancel=lambda request_id: notified.append(request_id))
    handle = tm.register("req-1", _sleep_forever)

    assert await tm.cancel("req-1") is True
    assert notified == ["req-1"]

    await asyncio.wait([handle.task])
    await tm.cancel_all()


async def test_cancel_supports_async_client_callback():
    """客户端回调是协程时也会被调用。"""
    notified: list[str] = []

    async def on_cancel(request_id: str) -> None:
        notified.append(request_id)

    tm = TaskManager(on_client_cancel=on_cancel)
    handle = tm.register("req-1", _sleep_forever)

    await tm.cancel("req-1")
    assert notified == ["req-1"]

    await asyncio.wait([handle.task])
    await tm.cancel_all()


async def test_cancel_all_notifies_client_for_each_request():
    """取消全部时每个请求都通知客户端。"""
    notified: list[str] = []
    tm = TaskManager(on_client_cancel=lambda request_id: notified.append(request_id))
    first = tm.register("req-1", _sleep_forever)
    second = tm.register("req-2", _sleep_forever)

    await tm.cancel_all()
    assert sorted(notified) == ["req-1", "req-2"]

    await asyncio.wait([first.task, second.task])


async def test_list_tasks_reports_basic_info(manager):
    """列表给出请求标识、是否已结束与已运行时长。"""
    handle = manager.register("req-1", _sleep_forever)

    infos = manager.list_tasks()
    assert len(infos) == 1
    assert infos[0].request_id == "req-1"
    assert infos[0].done is False
    assert infos[0].elapsed_ms >= 0.0
    assert infos[0].timeout is None
    # 任务还挂着，收尾时由 fixture 取消
    assert handle.task is manager.get_task("req-1")


async def test_reregister_after_finish_starts_new_task(manager):
    """上一个任务结束后，同一请求标识可以重新登记。"""
    first = manager.register("req-1", _immediate)
    await first.task

    second = manager.register("req-1", _sleep_forever)
    assert second.created is True
    assert second.task is not first.task
    # 新任务还挂着，收尾时由 fixture 取消
    assert second.task is manager.get_task("req-1")


def test_register_rejects_bad_arguments():
    """空请求标识与非正的时限被拒绝，不建任务。"""
    tm = TaskManager()
    with pytest.raises(ValueError):
        tm.register("", _sleep_forever)
    with pytest.raises(ValueError):
        tm.register("req-1", _sleep_forever, timeout=0)
    with pytest.raises(ValueError):
        tm.register("req-1", _sleep_forever, timeout=-1)
    assert tm.list_tasks() == []
