"""主进程回收：进程发现、本应用判定与批量结束（Windows 等价）。

覆盖《2-骨架与配置》进程通信一节里 ``start`` 前回收旧实例的部分：
只认本仓库的启动命令，别的进程不碰；先 ``terminate()``，等不到退出再 ``kill()``。

Windows 口径：进程探测与回收走 ``psutil``，不再读系统进程表，也不依赖信号机制。
本文件用真实子进程（``python -c "import time; time.sleep(60)"``）覆盖活进程、
死进程、不存在的 PID、``pid <= 1``、当前进程、不误杀，以及两阶段结束；
``kill()`` 那一阶段用替身模拟「忽略 terminate」的顽固进程，保证可确定地覆盖。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator

import psutil
import pytest

import app.runtime.process_manager as pm
from app.runtime.process_manager import (
    TerminateReport,
    find_app_processes,
    is_app_process,
    is_process_alive,
    is_process_running,
    terminate_processes,
)

# 起子进程的工厂：参数是命令行参数
Spawner = Callable[[tuple[str, ...]], subprocess.Popen[bytes]]

# 纯睡眠子进程：命令行里不含 app.cli / app.main，用来验证「不误杀」
_SLEEP_ARGS: tuple[str, ...] = ("-c", "import time; time.sleep(60)")


def _spawn(args: tuple[str, ...]) -> subprocess.Popen[bytes]:
    """起一个测试子进程。"""
    return subprocess.Popen(
        [sys.executable, *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture
def child_factory() -> Iterator[Spawner]:
    """登记并收尾测试里起的子进程，测试结束确保它们都被杀掉。"""
    children: list[subprocess.Popen[bytes]] = []

    def spawn(args: tuple[str, ...]) -> subprocess.Popen[bytes]:
        child = _spawn(args)
        children.append(child)
        # 等 psutil 能读到子进程的命令行，避免刚 fork 就读空
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                if psutil.Process(child.pid).cmdline():
                    break
            except psutil.Error:
                pass
            time.sleep(0.01)
        return child

    yield spawn
    for child in children:
        if child.poll() is None:
            child.kill()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _dead_pid() -> int:
    """返回一个肯定已经退出的 PID：起一个跑完就退的子进程。"""
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.wait()
    return process.pid


def _nonexistent_pid() -> int:
    """返回一个当前不存在的 PID。"""
    pid = 4_000_000
    while psutil.pid_exists(pid):
        pid += 1
    return pid


# ----------------------------------------------------------------------
# 存活探测：活 / 死 / 不存在
# ----------------------------------------------------------------------
def test_alive_for_live_child(child_factory: Spawner) -> None:
    """活着的子进程判为活着、在跑。"""
    child = child_factory(_SLEEP_ARGS)
    assert is_process_alive(child.pid) is True
    assert is_process_running(child.pid) is True


def test_not_alive_for_dead_child() -> None:
    """已经退出的 PID 判为不活着、不在跑。"""
    pid = _dead_pid()
    assert is_process_alive(pid) is False
    assert is_process_running(pid) is False


def test_not_alive_for_nonexistent_pid() -> None:
    """不存在的 PID 判为不活着。"""
    assert is_process_alive(_nonexistent_pid()) is False


# ----------------------------------------------------------------------
# 本应用判定：不误杀
# ----------------------------------------------------------------------
def test_plain_sleeper_is_not_app_process(child_factory: Spawner) -> None:
    """命令行是 sleep、不含 app.cli / app.main 的进程不算本应用。"""
    child = child_factory(_SLEEP_ARGS)
    assert is_app_process(child.pid) is False
    assert child.pid not in find_app_processes()


def test_nonexistent_pid_is_not_app_process() -> None:
    """不存在的 PID 不算本应用。"""
    assert is_app_process(_nonexistent_pid()) is False


def test_find_app_processes_excludes_current_process() -> None:
    """当前进程永远不在扫描结果里。"""
    assert os.getpid() not in find_app_processes()


# ----------------------------------------------------------------------
# 批量结束：terminate 阶段（真实子进程）
# ----------------------------------------------------------------------
def test_terminate_processes_terminates_child(child_factory: Spawner) -> None:
    """terminate() 就能结束的子进程记在 terminated 里。"""
    child = child_factory(_SLEEP_ARGS)
    report = terminate_processes([child.pid], timeout=5.0)
    assert isinstance(report, TerminateReport)
    assert report.terminated == (child.pid,)
    assert report.killed == ()
    assert report.survived == ()
    assert child.wait(timeout=5) is not None


# ----------------------------------------------------------------------
# 批量结束：kill 阶段（替身模拟忽略 terminate 的顽固进程）
# ----------------------------------------------------------------------
def test_terminate_processes_kills_when_terminate_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """进程忽略 terminate() 时，超时后走 kill()，记在 killed 里。"""
    state = {"alive": True}

    class _StubbornProcess:
        """忽略 terminate()、只对 kill() 生效的替身。"""

        def __init__(self, pid: int) -> None:
            self.pid = pid

        def terminate(self) -> None:
            # 模拟 Windows 上没权限或进程不响应：terminate 不生效
            return None

        def kill(self) -> None:
            state["alive"] = False

    target = 4321
    # 让存活判定完全由替身状态决定，避免真碰本机进程
    monkeypatch.setattr(pm, "is_process_alive", lambda pid: state["alive"])
    monkeypatch.setattr(psutil, "Process", _StubbornProcess)

    assert is_process_running(target) is True
    report = terminate_processes([target], timeout=0.2)

    assert report.terminated == ()
    assert report.killed == (target,)
    assert report.survived == ()
    assert is_process_running(target) is False


# ----------------------------------------------------------------------
# 批量结束：跳过不该动的目标
# ----------------------------------------------------------------------
def test_terminate_processes_skips_self_and_invalid() -> None:
    """自己、非正数 PID 与 PID=1 都不动。"""
    report = terminate_processes([os.getpid(), 0, -1, 1], timeout=0.1)
    assert report.terminated == ()
    assert report.killed == ()
    assert report.survived == ()
    assert set(report.skipped) == {os.getpid(), 0, -1, 1}


def test_terminate_processes_skips_dead_target() -> None:
    """已经退出的目标直接跳过，不算结束目标。"""
    pid = _dead_pid()
    report = terminate_processes([pid], timeout=0.1)
    assert report.terminated == ()
    assert report.killed == ()
    assert report.survived == ()
    assert pid in report.skipped