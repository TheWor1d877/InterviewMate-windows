"""PID 文件、通信端点文件与主进程回收（Windows 口径）。

见《2-骨架与配置》进程通信一节：

- 通信端点默认在 ``runtime/interviewmate.port``，路径来自配置；Windows 侧
  监听 ``127.0.0.1`` 上的随机端口，端口号写进端点文件，token 写进同目录的
  ``interviewmate.token``，客户端每次连接都要先发一行 token；
- 主进程启动时如果端点已存在，先连一次确认有没有活进程占用，
  有活进程就以退出码 ``1`` 结束，没有就删掉旧文件重建；
- PID 文件写主进程 PID，``start`` 时检查文件里的 PID 是否对应活进程；
- ``start`` 前先回收上一次留下的主进程：先走传输层的 ``stop``，
  不成再 ``terminate()``，还在就 ``kill()``。回收只认本仓库的进程，
  见 :func:`is_app_process`。

进程存活探测与端点管理走平台门面 :func:`app.platform.get_platform`：
Windows 侧是 ``psutil`` 与 TCP loopback。进程表读取（命令行、工作目录、
环境变量）一律用 ``psutil``；``psutil`` 在方法体内按需
导入，模块顶层不导入。

本模块只碰本地文件与本机进程，不发网络请求。
"""

from __future__ import annotations

import os
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

# 端点文件名与 PID 文件名。Windows 侧的端点文件里写的是端口号，
# token 文件由传输层按同目录、后缀换 ``.token`` 管理。
SOCKET_FILE_NAME: str = "interviewmate.port"
PID_FILE_NAME: str = "interviewmate.pid"

# 默认路径：仓库根/runtime/interviewmate.port 与仓库根/runtime/interviewmate.pid
SOCKET_PATH: Path = REPO_ROOT / RUNTIME_DIR_NAME / SOCKET_FILE_NAME
PID_PATH: Path = REPO_ROOT / RUNTIME_DIR_NAME / PID_FILE_NAME

# 判断进程是不是本仓库主进程用的命令行标记：
# ``python -m app.cli start`` 与 ``python -m app.main``
_APP_CLI: str = "app.cli"
_APP_MAIN: str = "app.main"
_APP_CLI_SUBCOMMAND: str = "start"

# 发完终止请求后轮询存活的间隔与等 ``kill()`` 生效的秒数
_TERM_POLL_INTERVAL: float = 0.1
_KILL_WAIT_SECONDS: float = 1.0


@dataclass(frozen=True)
class TerminateReport:
    """一次批量结束进程的结果，各字段都是 PID 元组。

    - ``terminated``：收到 ``terminate()`` 后在等待时间内退出；
    - ``killed``：等不到退出，发了 ``kill()``；
    - ``skipped``：不算有效目标（进程已不在、PID 非法或就是自己）；
    - ``survived``：发完 ``kill()`` 还活着，多半是没权限。
    """

    terminated: tuple[int, ...] = ()
    killed: tuple[int, ...] = ()
    skipped: tuple[int, ...] = ()
    survived: tuple[int, ...] = ()


def _norm_path(path: Path | str) -> str:
    """把路径归一成可比较的形式。

    Windows 大小写不敏感，用 ``os.path.normcase(os.path.normpath(...))``
    把分隔符与大小写统一掉，避免同一路径因写法不同比较不相等。
    """
    return os.path.normcase(os.path.normpath(str(path)))


def _ancestor_pids() -> set[int]:
    """返回当前进程的祖先 PID 集合，父进程、祖父进程一直往上。

    ``.venv\\Scripts\\python.exe`` 是 venv 启动器，它会再起一个真正的
    python 进程跑脚本；启动器的命令行与子进程几乎一样、工作目录也在仓库根，
    会被 :func:`find_app_processes` 当成"上次残留的主进程"。
    终止启动器会连带终止当前进程（Windows 上以退出码 15 静默结束），
    所以祖先链上的进程一律不当回收对象。

    读不到父进程时返回已经收集到的部分，不往外抛。
    """
    import psutil

    ancestors: set[int] = set()
    try:
        proc = psutil.Process(os.getpid())
        while True:
            parent = proc.parent()
            if parent is None or parent.pid <= 1 or parent.pid in ancestors:
                break
            ancestors.add(parent.pid)
            proc = parent
    except (psutil.Error, NotImplementedError, AttributeError, OSError):
        return ancestors
    return ancestors


def _read_cmdline(pid: int) -> tuple[str, ...]:
    """读进程命令行参数元组，读不到返回空元组。

    用 ``psutil.Process(pid).cmdline()``；进程不在或没权限时返回空元组。
    """
    import psutil

    try:
        return tuple(psutil.Process(pid).cmdline())
    except (psutil.Error, NotImplementedError):
        return ()


def _read_cwd(pid: int) -> Path | None:
    """读进程工作目录，读不到返回 None。"""
    import psutil

    try:
        return Path(psutil.Process(pid).cwd())
    except (psutil.Error, NotImplementedError):
        return None


def _read_environ(pid: int) -> dict[str, str]:
    """读进程环境变量，读不到返回空 dict。

    用 ``psutil.Process(pid).environ()``；``AccessDenied``、``NotImplementedError``
    等读不到的情况一律返回空 dict，不往外抛。
    """
    import psutil

    try:
        return dict(psutil.Process(pid).environ())
    except (psutil.Error, NotImplementedError):
        return {}


def _belongs_to_repo(pid: int) -> bool:
    """判断进程是不是从本仓库启动的。

    先看工作目录；从别的目录用启动器启动时目录不在仓库根，
    退到 ``PYTHONPATH`` 里找仓库根（启动脚本会把它加进去）。
    路径比较统一走 :func:`_norm_path`，Windows 下大小写不敏感。
    """
    cwd = _read_cwd(pid)
    if cwd is not None and _norm_path(cwd) == _norm_path(REPO_ROOT):
        return True
    environ = _read_environ(pid)
    pythonpath = ""
    for key, value in environ.items():
        if key.upper() == "PYTHONPATH":
            pythonpath = value
            break
    repo = _norm_path(REPO_ROOT)
    for entry in pythonpath.split(os.pathsep):
        if entry and _norm_path(entry) == repo:
            return True
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
    """用 ``psutil.process_iter`` 扫出本仓库还在跑的主进程。

    :param exclude: 额外排除的 PID，当前进程总是被排除。
    """
    import psutil

    skip = set(exclude)
    skip.add(os.getpid())
    # venv 启动器与当前进程命令行相同，杀了它会连带杀掉当前进程
    skip |= _ancestor_pids()
    found: list[int] = []
    for proc in psutil.process_iter(["pid"]):
        try:
            pid = proc.info.get("pid")
        except (psutil.Error, AttributeError):
            continue
        if not isinstance(pid, int) or pid in skip:
            continue
        if is_app_process(pid):
            found.append(pid)
    return tuple(sorted(found))


def is_process_running(pid: int) -> bool:
    """进程是不是还在跑。

    Windows 没有僵尸进程（父进程退出后由系统回收），这里直接委托
    :func:`is_process_alive`。

    :param pid: 进程 PID。
    """
    return is_process_alive(pid)


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
    """先 ``terminate()``，等不到退出再 ``kill()``。

    调用方负责挑出该结束的进程；这里只挡明显不该动的目标：
    小于等于 1、就是自己、当前进程的祖先链、已经不在的 PID。
    祖先链要挡是因为 venv 启动器的命令行与本应用主进程相同，
    终止它会连带终止当前进程。

    注意 Windows 上 ``terminate()`` 即 ``TerminateProcess``，本身就是强制结束，
    没有可捕获的优雅退出，因此多数情况下进程会直接记入 ``terminated``，
    ``killed`` 通常为空；两步结构保留是为了语义一致与极端情况下仍能兜底。

    :param pids: 要结束的 PID 集合，重复值只处理一次。
    :param timeout: ``terminate()`` 后等退出的秒数。
    """
    import psutil

    targets: list[int] = []
    skipped: list[int] = []
    protected = {os.getpid()} | _ancestor_pids()
    for pid in dict.fromkeys(pids):
        if pid <= 1 or pid in protected or not is_process_running(pid):
            skipped.append(pid)
            continue
        targets.append(pid)
    if not targets:
        return TerminateReport(skipped=tuple(skipped))

    pending: list[int] = []
    for pid in targets:
        try:
            psutil.Process(pid).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
            # 发请求前刚好退了或没权限，按已跳过处理
            skipped.append(pid)
            continue
        pending.append(pid)

    survivors = _wait_dead(pending, timeout)
    terminated = [pid for pid in pending if pid not in survivors]

    killed: list[int] = []
    for pid in survivors:
        try:
            psutil.Process(pid).kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
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

    委托给平台：Windows 侧用 ``psutil.Process(pid).is_running()``。
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
