"""PID 文件、通信端点文件与主进程回收。

见《2-骨架与配置》进程通信一节：

- 通信端点默认在 ``runtime/interviewmate.sock``，路径来自配置；
- 主进程启动时如果端点已存在，先连一次确认有没有活进程占用，
  有活进程就以退出码 ``1`` 结束，没有就删掉旧文件重建；
- PID 文件写主进程 PID，``start`` 时检查文件里的 PID 是否对应活进程；
- ``start`` 前先回收上一次留下的主进程：先走 socket 的 ``stop``，
  不成再发 ``SIGTERM``，还在就 ``SIGKILL``。回收只认本仓库的进程，
  见 :func:`is_app_process`。

进程存活探测与端点管理走平台门面 :func:`app.platform.get_platform`：
Linux 侧是 ``os.kill(pid, 0)`` 与 AF_UNIX socket 文件，Windows 侧换成
``psutil`` 与 TCP loopback，见《9-平台差异与Windows适配》。
回收用的进程表与信号是 Linux 侧实现（读 ``/proc``），非 Linux 上
:func:`find_app_processes` 返回空元组、不碰任何进程。

本模块只碰本地文件与本机进程，不发网络请求。
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.platform import get_platform

__all__ = [
    "REPO_ROOT",
    "RUNTIME_DIR_NAME",
    "SOCKET_FILE_NAME",
    "PID_FILE_NAME",
    "SOCKET_PATH",
    "PID_PATH",
    "TerminateReport",
    "is_process_alive",
    "is_process_running",
    "is_app_process",
    "find_app_processes",
    "terminate_processes",
    "read_pid",
    "write_pid",
    "remove_pid",
    "is_running",
    "prepare_socket_path",
]


# 仓库根目录：app/runtime/process_manager.py 往上三级，与当前工作目录无关
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# 运行期产物目录名，放在仓库根下，见《2-骨架与配置》目录结构约定一节
RUNTIME_DIR_NAME: str = "runtime"

# socket 文件名与 PID 文件名
SOCKET_FILE_NAME: str = "interviewmate.sock"
PID_FILE_NAME: str = "interviewmate.pid"

# 默认路径：仓库根/runtime/interviewmate.sock 与仓库根/runtime/interviewmate.pid
SOCKET_PATH: Path = REPO_ROOT / RUNTIME_DIR_NAME / SOCKET_FILE_NAME
PID_PATH: Path = REPO_ROOT / RUNTIME_DIR_NAME / PID_FILE_NAME

# 进程表与 /proc 读取是 Linux 侧实现，非 Linux 上没有这个目录
_PROC_ROOT: Path = Path("/proc")

# 判断进程是不是本仓库主进程用的命令行标记：
# ``python -m app.cli start`` 与 ``python -m app.main``
_APP_CLI: str = "app.cli"
_APP_MAIN: str = "app.main"
_APP_CLI_SUBCOMMAND: str = "start"

# 发完终止信号后轮询存活的间隔与等 SIGKILL 生效的秒数
_TERM_POLL_INTERVAL: float = 0.1
_KILL_WAIT_SECONDS: float = 1.0


@dataclass(frozen=True)
class TerminateReport:
    """一次批量结束进程的结果，各字段都是 PID 元组。

    - ``terminated``：收到 SIGTERM 后在等待时间内退出；
    - ``killed``：等不到退出，发了 SIGKILL；
    - ``skipped``：不算有效目标（进程已不在、PID 非法或就是自己）；
    - ``survived``：发完 SIGKILL 还活着，多半是没权限。
    """

    terminated: tuple[int, ...] = ()
    killed: tuple[int, ...] = ()
    skipped: tuple[int, ...] = ()
    survived: tuple[int, ...] = ()


def _read_cmdline(pid: int) -> tuple[str, ...]:
    """读 ``/proc/<pid>/cmdline`` 拆成参数元组，读不到返回空元组。"""
    try:
        raw = (_PROC_ROOT / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ()
    return tuple(
        part.decode("utf-8", "replace") for part in raw.split(b"\0") if part
    )


def _read_cwd(pid: int) -> Path | None:
    """读 ``/proc/<pid>/cwd``，读不到返回 None。"""
    try:
        return Path(os.readlink(_PROC_ROOT / str(pid) / "cwd"))
    except OSError:
        return None


def _belongs_to_repo(pid: int) -> bool:
    """判断进程是不是从本仓库启动的。

    先看工作目录；从别的目录用 ``./interviewmate`` 启动时目录不在仓库根，
    退到 ``PYTHONPATH`` 里找仓库根（启动脚本会把它加进去）。
    """
    cwd = _read_cwd(pid)
    if cwd is not None and cwd == REPO_ROOT:
        return True
    try:
        raw = (_PROC_ROOT / str(pid) / "environ").read_bytes()
    except OSError:
        return False
    repo = str(REPO_ROOT)
    for item in raw.split(b"\0"):
        if not item.startswith(b"PYTHONPATH="):
            continue
        value = item.split(b"=", 1)[1].decode("utf-8", "replace")
        return repo in value.split(os.pathsep)
    return False


def _is_app_cmdline(args: tuple[str, ...]) -> bool:
    """判断命令行参数是不是本仓库主进程的启动命令。"""
    if _APP_MAIN in args:
        return True
    return _APP_CLI in args and _APP_CLI_SUBCOMMAND in args


def is_app_process(pid: int) -> bool:
    """判断 PID 是不是本仓库的主进程。

    PID 不合法、进程不在、命令行不是本应用的启动命令、
    或者不是从本仓库启动的都算不是。

    :param pid: 进程 PID。
    """
    if pid <= 0 or not is_process_alive(pid):
        return False
    if not _is_app_cmdline(_read_cmdline(pid)):
        return False
    return _belongs_to_repo(pid)


def find_app_processes(*, exclude: Iterable[int] = ()) -> tuple[int, ...]:
    """扫 ``/proc`` 找出本仓库还在跑的主进程。

    非 Linux 上没有 ``/proc``，直接返回空元组，不碰任何进程。

    :param exclude: 额外排除的 PID，当前进程总是被排除。
    """
    if not _PROC_ROOT.is_dir():
        return ()
    skip = set(exclude)
    skip.add(os.getpid())
    found: list[int] = []
    try:
        entries = list(_PROC_ROOT.iterdir())
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in skip:
            continue
        if is_app_process(pid):
            found.append(pid)
    return tuple(sorted(found))


def _is_zombie(pid: int) -> bool:
    """进程是不是僵尸：已经退出，只等父进程收尸。

    ``/proc/<pid>/stat`` 的格式是 ``pid (comm) state ...``，``comm`` 里可能有
    空格和括号，所以按最后一个 ``)`` 切。
    """
    try:
        stat = (_PROC_ROOT / str(pid) / "stat").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False
    _, _, rest = stat.rpartition(")")
    return rest.strip().startswith("Z")


def is_process_running(pid: int) -> bool:
    """进程是不是还在跑。

    僵尸已经退出，只等父进程收尸，算不在跑。

    :param pid: 进程 PID。
    """
    return is_process_alive(pid) and not _is_zombie(pid)


def _wait_dead(pids: list[int], timeout: float) -> list[int]:
    """等一批进程退出，返回超时后还在跑的。"""
    pending = [pid for pid in pids if is_process_running(pid)]
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        time.sleep(_TERM_POLL_INTERVAL)
        pending = [pid for pid in pending if is_process_running(pid)]
    return pending


def terminate_processes(
    pids: Iterable[int], *, timeout: float = 5.0
) -> TerminateReport:
    """先 SIGTERM，等不到退出再 SIGKILL。

    调用方负责挑出该结束的进程；这里只挡明显不该动的目标：
    小于等于 1、就是自己、已经不在（含僵尸）的 PID。

    :param pids: 要结束的 PID 集合，重复值只处理一次。
    :param timeout: SIGTERM 后等退出的秒数。
    """
    targets: list[int] = []
    skipped: list[int] = []
    for pid in dict.fromkeys(pids):
        if pid <= 1 or pid == os.getpid() or not is_process_running(pid):
            skipped.append(pid)
            continue
        targets.append(pid)
    if not targets:
        return TerminateReport(skipped=tuple(skipped))

    pending: list[int] = []
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            # 发信号前刚好退了，按已结束处理
            skipped.append(pid)
            continue
        except OSError:
            # 没权限之类的，留给调用方按还活着处理
            skipped.append(pid)
            continue
        pending.append(pid)

    survivors = _wait_dead(pending, timeout)
    terminated = [pid for pid in pending if pid not in survivors]

    killed: list[int] = []
    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
        killed.append(pid)
    survived = _wait_dead(killed, _KILL_WAIT_SECONDS)

    return TerminateReport(
        terminated=tuple(terminated),
        killed=tuple(killed),
        skipped=tuple(skipped),
        survived=tuple(survived),
    )


def is_process_alive(pid: int) -> bool:
    """判断 PID 对应的进程是否还活着。

    委托给平台：Linux 侧用 ``os.kill(pid, 0)``，Windows 侧用 ``psutil``，
    见《9-平台差异与Windows适配》「Windows 进程存活探测与退出」一节。
    ``pid`` 小于等于 0 一律算不活着。

    :param pid: 进程 PID。
    """
    return get_platform().process.is_alive(pid)


def read_pid(pid_path: Path = PID_PATH) -> int | None:
    """读 PID 文件里的 PID。

    文件不存在、内容不是合法正整数时返回 None。

    :param pid_path: PID 文件路径。
    """
    try:
        text = pid_path.read_text(encoding="utf-8").strip()
    except OSError:
        # 文件不存在、权限不足、路径不是文件等一律按读不到处理
        return None
    try:
        pid = int(text)
    except ValueError:
        return None
    if pid <= 0:
        return None
    return pid


def write_pid(pid_path: Path = PID_PATH, pid: int | None = None) -> None:
    """写 PID 文件。

    ``pid`` 为 None 时写当前进程 PID。父目录不存在会先建出来。

    :param pid_path: PID 文件路径。
    :param pid: 要写入的 PID，None 表示当前进程。
    """
    target = os.getpid() if pid is None else int(pid)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{target}\n", encoding="utf-8")


def remove_pid(pid_path: Path = PID_PATH) -> None:
    """删掉 PID 文件，文件不存在也不报错。

    :param pid_path: PID 文件路径。
    """
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass


def is_running(pid_path: Path = PID_PATH) -> bool:
    """判断 PID 文件对应的主进程是否还活着。

    PID 文件不存在、内容非法、或对应进程已退出都返回 False。

    :param pid_path: PID 文件路径。
    """
    pid = read_pid(pid_path)
    if pid is None:
        return False
    return is_process_alive(pid)


def prepare_socket_path(socket_path: Path = SOCKET_PATH) -> bool:
    """检查通信端点能不能用，返回是否可以继续启动。

    委托给平台传输层，端点探测与残留文件清理的规则见
    :mod:`app.platform.types` 的 ``Transport``：

    - 端点不存在：直接返回 True；
    - 端点存在且连得上：说明有活进程占用，返回 False；
    - 端点存在但连不上：是上次退出留下的残留文件，删掉后返回 True。

    :param socket_path: 通信端点路径。
    :returns: 可以重建端点时返回 True，被活进程占用时返回 False。
    """
    return get_platform().transport(socket_path).prepare_endpoint()
