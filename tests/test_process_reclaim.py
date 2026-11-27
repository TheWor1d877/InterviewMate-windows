"""主进程回收：进程发现、本应用判定与批量结束。

覆盖《2-骨架与配置》进程通信一节里 ``start`` 前回收旧实例的部分：
只认本仓库的启动命令，别的进程不碰；先 SIGTERM，等不到退出再 SIGKILL。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from app.runtime.process_manager import (
    REPO_ROOT,
    TerminateReport,
    find_app_processes,
    is_app_process,
    terminate_processes,
)

# 起子进程的工厂：参数是命令行参数与工作目录
Spawner = Callable[..., subprocess.Popen[bytes]]

# 让子进程的命令行看起来像本应用的启动命令，同时不会真的跑起来
_FAKE_APP_ARGS: tuple[str, ...] = (
    "-c",
    "import time\ntime.sleep(60)",
    "app.cli",
    "start",
)


def _spawn(args: tuple[str, ...], *, cwd: Path) -> subprocess.Popen[bytes]:
    """起一个测试子进程。"""
    return subprocess.Popen(
        [sys.executable, *args],
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture
def child_factory() -> Iterator[Spawner]:
    """登记并收尾测试里起的子进程。"""
    children: list[subprocess.Popen[bytes]] = []

    def spawn(args: tuple[str, ...], *, cwd: Path) -> subprocess.Popen[bytes]:
        child = _spawn(args, cwd=cwd)
        children.append(child)
        # 等子进程的 /proc 内容就绪，避免刚 fork 就读空
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if (Path("/proc") / str(child.pid) / "cmdline").read_bytes():
                break
            time.sleep(0.01)
        return child

    yield spawn
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_find_app_processes_finds_repo_app(child_factory: Spawner) -> None:
    """命令行像本应用、工作目录在本仓库的进程会被找出来。"""
    child = child_factory(_FAKE_APP_ARGS, cwd=REPO_ROOT)
    assert is_app_process(child.pid) is True
    assert child.pid in find_app_processes()


def test_is_app_process_rejects_other_cwd(child_factory: Spawner, tmp_path: Path) -> None:
    """命令行像本应用但不是从本仓库启动的，不算本应用。"""
    child = child_factory(_FAKE_APP_ARGS, cwd=tmp_path)
    assert is_app_process(child.pid) is False
    assert child.pid not in find_app_processes()


def test_is_app_process_rejects_current_process() -> None:
    """跑测试的进程不算本应用。"""
    assert is_app_process(os.getpid()) is False
    assert os.getpid() not in find_app_processes()


def test_terminate_processes_terminates_child(child_factory: Spawner) -> None:
    """SIGTERM 就能结束的子进程记在 terminated 里。"""
    child = child_factory(_FAKE_APP_ARGS, cwd=REPO_ROOT)
    report = terminate_processes([child.pid], timeout=5.0)
    assert isinstance(report, TerminateReport)
    assert report.terminated == (child.pid,)
    assert report.killed == ()
    assert report.survived == ()
    assert child.wait(timeout=5) is not None


def test_terminate_processes_skips_self_and_invalid() -> None:
    """自己、非正数 PID 与 init 都不动。"""
    report = terminate_processes([os.getpid(), 0, -1, 1], timeout=0.1)
    assert report.terminated == ()
    assert report.killed == ()
    assert report.survived == ()
    assert set(report.skipped) == {os.getpid(), 0, -1, 1}