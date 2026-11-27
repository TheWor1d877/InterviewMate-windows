"""Windows 侧进程存活探测的测试。

覆盖《接口契约》进程存活探测（psutil）一节：
:class:`~app.platform.windows.process.PsutilProcessBackend` 对活进程、
死 PID、非法 PID、权限不足与 PID 复用的判定行为。

真机只跑活进程与死 PID 两条：``psutil`` 权限与进程身份的异常场景用
``monkeypatch`` 替换 ``psutil.Process`` 模拟，不依赖真机的权限环境，
也不真杀任何进程。
"""

from __future__ import annotations

import os
import subprocess
import sys

import psutil
import pytest

from app.platform.windows.process import PsutilProcessBackend


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
    return process.pid


def _process_raising(exc: Exception) -> type:
    """造一个 ``psutil.Process`` 替身：构造时抛指定异常。"""

    class _BrokenProcess:
        """构造即抛异常的替身。"""

        def __init__(self, pid: int) -> None:
            raise exc

    return _BrokenProcess


def test_is_alive_true_for_current_process() -> None:
    """当前进程一定活着。"""
    assert PsutilProcessBackend().is_alive(os.getpid()) is True


def test_is_alive_false_for_dead_pid() -> None:
    """已经退出的 PID 判为不活着。"""
    assert PsutilProcessBackend().is_alive(_dead_pid()) is False


def test_is_alive_false_for_non_positive_pid() -> None:
    """PID 小于等于 0 一律判为不活着。"""
    backend = PsutilProcessBackend()
    assert backend.is_alive(0) is False
    assert backend.is_alive(-1) is False


def test_is_alive_false_for_missing_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """进程查不到（NoSuchProcess）判为不活着。"""
    exc = psutil.NoSuchProcess(1234)
    monkeypatch.setattr(psutil, "Process", _process_raising(exc))
    assert PsutilProcessBackend().is_alive(1234) is False


def test_is_alive_true_for_access_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """权限不足（AccessDenied）说明进程存在，判为活着。"""
    exc = psutil.AccessDenied(1234)
    monkeypatch.setattr(psutil, "Process", _process_raising(exc))
    assert PsutilProcessBackend().is_alive(1234) is True


def test_is_alive_true_for_reused_pid_of_live_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID 复用的现有行为：只看 PID 数字，不校验进程身份。

    残留 PID 文件里的 PID 被系统复用、指到别的活进程时，``is_running``
    只能答"活着"——复用进程会被当成还活着的主进程。身份校验的防线
    在共享判定链（``is_app_process`` 与 ``stop_stale_instance``），见
    WIN-MILESTONES.md W2.3 的核对结论，本测试固定这个现状防止误读。
    """

    class _ReusedProcess:
        """``is_running`` 恒真的替身，模拟复用后的新进程。"""

        def __init__(self, pid: int) -> None:
            self.pid = pid

        def is_running(self) -> bool:
            return True

    monkeypatch.setattr(psutil, "Process", _ReusedProcess)
    assert PsutilProcessBackend().is_alive(4321) is True


def test_is_alive_false_for_other_psutil_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """其它 psutil 错误按不活着处理，不往外抛。"""
    exc = psutil.ZombieProcess(1234)
    monkeypatch.setattr(psutil, "Process", _process_raising(exc))
    assert PsutilProcessBackend().is_alive(1234) is False