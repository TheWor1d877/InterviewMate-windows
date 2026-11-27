"""端到端测试：真起子进程，走真实 Unix socket。

覆盖《2-骨架与配置》验收一节的五条，以及《2-骨架与配置》进程通信一节的完整走查。

隔离办法
--------
每个子进程先 import 相关模块，把模块属性里的 ``SOCKET_PATH`` 与 ``PID_PATH``
统一改到 pytest 的 ``tmp_path``，再以 ``__main__`` 身份执行 ``app.cli`` 或
``app.main``：等价于 ``python -m app.cli`` / ``python -m app.main``，但 socket 文件与
PID 文件不落在仓库根的 ``runtime/`` 里，用户自己开着的现场不受影响。

这些模块是用 ``from ... import SOCKET_PATH`` 绑定的，只改
:mod:`app.runtime.process_manager` 不够，所以逐个模块改：来源模块
(:mod:`app.runtime.process_manager`、:mod:`app.config`) 改了，新执行的 ``app.cli``
与 ``app.main`` 会拿到改后的值；被缓存复用的 :mod:`app.main` 另外单独改。

伪终端
------
Textual 界面需要 TTY，:func:`app.main.start_process` 在 stdout 不是终端时直接拒绝启动。
这里用标准库 :mod:`pty` 给主进程接上伪终端，并起后台线程持续把输出读出来丢弃，
避免 pty 缓冲区写满把子进程卡死。

不连真实服务
------------
全程用 ``providers`` 四键全 mock 的配置副本，``DEEPSEEK_API_KEY`` 用假值，
不访问 DeepSeek、Dify、真实 ASR 模型或真实截图服务。

M2 装配
-------
M2 的那条闭环（输入框 → mock 流式回答 → 左右两栏渲染）在同一个进程里跑：
用 Textual 的 ``run_test()`` 无头启动界面，装配 mock 客户端的问题管理器，
在输入框里敲一句话按回车，断言两栏与状态栏。
"""

from __future__ import annotations

import asyncio
import fcntl
import itertools
import json
import os
import pty
import re
import signal
import socket
import struct
import subprocess
import sys
import termios
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

import app.config
from app.config import load_config
from app.core.event_bus import EventBus
from app.llm.mock_client import MockClient
from app.main import MainState, assemble_manager, assemble_screenshot_manager
from app.ui.app import InterviewMateApp
from app.ui.panels.input_panel import QuestionInputPanel
from app.ui.panels.question_header import QuestionHeader
from app.ui.status_bar import StatusBar

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = app.config.DEFAULT_CONFIG_PATH

# 退出码表见《2-骨架与配置》命令行入口一节
EXIT_OK: int = 0
EXIT_ERROR: int = 1
EXIT_CONFIG_ERROR: int = 3

# 每个子进程调用的超时上限，秒
CLI_TIMEOUT: float = 60.0
# 主进程启动的等待上限，秒
START_TIMEOUT: float = 60.0
# 主进程退出的等待上限，秒
STOP_TIMEOUT: float = 60.0

# 假的 API key，形状像真的但不是真的，不连真实服务
FAKE_API_KEY: str = "sk-test-not-a-real-key"

# 无头界面用的终端尺寸：宽终端 120 列，左栏按 22% 分到 26 列，框架标题放得下
WIDE_WIDTH = 120
WIDE_HEIGHT = 30

# 等一轮生成结束的上限，秒
GENERATION_TIMEOUT = 20.0

# 日志脱敏探针里用的假令牌
PROBE_TOKEN: str = "e2e-probe-token"

# 会影响配置结果的环境变量，子进程里统一清掉再按需注入
_OVERRIDE_ENV: tuple[str, ...] = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DIFY_BASE_URL",
    "DIFY_DATASET_ID",
    "DIFY_DATASET_API_KEY",
)


# ----------------------------------------------------------------------
# 子进程引导脚本
# ----------------------------------------------------------------------

# 在子进程里做的事：改运行期路径 → 写脱敏探针 → 以 __main__ 身份执行 app.cli 或 app.main。
# 用 runpy 而不是直接调函数，是为了和 ``python -m app.cli`` / ``python -m app.main``
# 走同一份 ``__main__`` 代码路径。
_BOOTSTRAP: str = r'''
import os
import runpy
import sys
import warnings
from pathlib import Path

# runpy 会提示模块已在 sys.modules 里，这次重复执行是本引导脚本有意为之
warnings.filterwarnings(
    "ignore", message=r".*found in sys.modules after import of package.*"
)

runtime_dir = Path(os.environ["IM_TEST_RUNTIME_DIR"])
socket_path = Path(os.environ["IM_TEST_SOCKET_PATH"])
pid_path = Path(os.environ["IM_TEST_PID_PATH"])
env_file = Path(os.environ["IM_TEST_ENV_FILE"])
log_root = Path(os.environ["IM_TEST_LOG_ROOT"])

import app.config
import app.main
import app.runtime.health_checker
import app.runtime.process_manager
import app.runtime.socket_client
import app.runtime.socket_server
import app.utils.logger

# 运行期路径全部指到临时目录：仓库根的 runtime/ 与 logs/ 不动
for _module in (
    app.runtime.process_manager,
    app.runtime.socket_client,
    app.runtime.socket_server,
    app.runtime.health_checker,
    app.main,
):
    _module.SOCKET_PATH = socket_path
    _module.PID_PATH = pid_path

app.config.REPO_ROOT = runtime_dir
app.config.DEFAULT_ENV_FILE = env_file
app.main.REPO_ROOT = runtime_dir
app.utils.logger.REPO_ROOT = log_root

# 日志脱敏探针：把密钥值与 Authorization 头原样交给项目的日志管道，
# 供测试断言日志文件里搜不到原文。没有这一行，脱敏断言就是空断言。
if os.environ.get("IM_TEST_LOG_PROBE") == "1":
    from app.utils.logger import log_event, setup_logging

    setup_logging()
    log_event(
        "e2e.log_probe",
        "Authorization: Bearer " + os.environ.get("IM_TEST_PROBE_TOKEN", ""),
        level="INFO",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    )

module_name = os.environ["IM_TEST_MODULE"]
sys.argv = [module_name] + sys.argv[1:]
try:
    runpy.run_module(module_name, run_name="__main__")
except SystemExit as exc:
    _code = exc.code
    sys.exit(0 if _code is None else _code)
'''


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------

_ANSI_PATTERN = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI 序列
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC 序列
    r"|\x1b[()][0-9A-B]"  # 字符集切换
    r"|\x1b[=>]"  # 键盘模式
)


def _strip_ansi(text: str) -> str:
    """去掉终端控制序列，只留可读文本。"""
    return _ANSI_PATTERN.sub("", text)


def _plain(text: str) -> str:
    """去掉控制序列与所有空白，用于在界面渲染结果里找固定文案。"""
    return re.sub(r"\s+", "", _strip_ansi(text))


def _normalize(text: str) -> str:
    """把 pty 的 CRLF 行尾统一成 LF。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _write_config(dest: Path, **sections: Any) -> Path:
    """写一份配置副本，``sections`` 是要整体替换的顶层段。"""
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for name, value in sections.items():
        data[name] = value
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return dest


def _write_config_with_model(dest: Path, model: str) -> Path:
    """写一份配置副本，只把 ``deepseek.model`` 换成给定值。"""
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    deepseek = dict(data["deepseek"])
    deepseek["model"] = model
    data["deepseek"] = deepseek
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return dest


def _write_config_with_llm(dest: Path, llm: str) -> Path:
    """写一份配置副本，四键显式 mock，再把 ``providers.llm`` 换成给定值。"""
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["providers"] = {
        "llm": llm,
        "retrieval": "mock",
        "asr": "mock",
        "screenshot": "mock",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return dest


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    """杀掉子进程所在的进程组并回收。"""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _socket_reachable(path: Path, timeout: float = 1.0) -> bool:
    """socket 文件在不在、连不连得上。"""
    if not path.exists():
        return False
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(timeout)
        client.connect(str(path))
    except OSError:
        return False
    finally:
        client.close()
    return True


# ----------------------------------------------------------------------
# 伪终端
# ----------------------------------------------------------------------
class _PtyReader:
    """持续读 pty 主端的输出并丢弃，防止 pty 缓冲区写满把子进程卡死。"""

    def __init__(self, master_fd: int) -> None:
        self._fd = master_fd
        self._chunks: list[bytes] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """起读线程。"""
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                try:
                    data = os.read(self._fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                self._chunks.append(data)
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass

    def join(self, timeout: float = 5.0) -> None:
        """等读线程结束。"""
        self._thread.join(timeout=timeout)

    def text(self) -> str:
        """读到的全部输出。"""
        return b"".join(self._chunks).decode("utf-8", errors="replace")


def _spawn_pty(
    args: list[str], env: dict[str, str]
) -> tuple[subprocess.Popen[bytes], _PtyReader]:
    """在伪终端里起一个子进程，返回进程与输出读取器。"""
    master_fd, slave_fd = pty.openpty()
    try:
        # 给一个确定的窗口大小，Textual 按这个尺寸排版
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        proc = subprocess.Popen(
            [sys.executable, "-c", _BOOTSTRAP, *args],
            cwd=str(REPO_ROOT),
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
        )
    finally:
        os.close(slave_fd)
    reader = _PtyReader(master_fd)
    reader.start()
    return proc, reader


# ----------------------------------------------------------------------
# 运行期环境
# ----------------------------------------------------------------------
class _Runtime:
    """一次子进程运行用到的临时路径与环境变量。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.socket_path = root / "interviewmate.sock"
        self.pid_path = root / "interviewmate.pid"
        self.env_file = root / "no-such.env"
        self.log_root = root

    def env(
        self,
        *,
        module: str = "app.cli",
        api_key: str | None = FAKE_API_KEY,
        log_probe: bool = False,
    ) -> dict[str, str]:
        """建子进程的环境变量。

        开发者本机的环境变量与仓库根的 .env 都不影响子进程：
        这里先把相关变量清掉，再按测试需要注入假值。
        """
        env = dict(os.environ)
        for name in _OVERRIDE_ENV:
            env.pop(name, None)
        env.pop("IM_TEST_LOG_PROBE", None)
        env.pop("IM_TEST_PROBE_TOKEN", None)
        if api_key is not None:
            env["DEEPSEEK_API_KEY"] = api_key
        env["IM_TEST_RUNTIME_DIR"] = str(self.root)
        env["IM_TEST_SOCKET_PATH"] = str(self.socket_path)
        env["IM_TEST_PID_PATH"] = str(self.pid_path)
        env["IM_TEST_ENV_FILE"] = str(self.env_file)
        env["IM_TEST_LOG_ROOT"] = str(self.log_root)
        env["IM_TEST_MODULE"] = module
        if log_probe:
            env["IM_TEST_LOG_PROBE"] = "1"
            env["IM_TEST_PROBE_TOKEN"] = PROBE_TOKEN
        env["TERM"] = "xterm-256color"
        env["PYTHONUNBUFFERED"] = "1"
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
        )
        return env


# ----------------------------------------------------------------------
# 跑命令
# ----------------------------------------------------------------------
def _run_cli(
    args: list[str],
    runtime: _Runtime,
    *,
    api_key: str | None = FAKE_API_KEY,
    timeout: float = CLI_TIMEOUT,
) -> tuple[int, str, str]:
    """跑一次命令行子进程，返回 (退出码, stdout, stderr)。

    超时按失败处理：杀掉子进程并让测试失败，不会挂死。
    """
    env = runtime.env(api_key=api_key)
    proc = subprocess.Popen(
        [sys.executable, "-c", _BOOTSTRAP, *args],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        out, err = proc.communicate()
        pytest.fail(f"命令 {args} 超过 {timeout} 秒未结束，输出：\n{out}\n{err}")
    return proc.returncode, out, err


def _run_cli_in_pty(
    args: list[str],
    runtime: _Runtime,
    *,
    api_key: str | None = FAKE_API_KEY,
    timeout: float = CLI_TIMEOUT,
) -> tuple[int, str]:
    """在伪终端里跑一次命令行子进程，返回 (退出码, 合并后的输出)。

    stdout 与 stderr 在 pty 上合并成一路，所以这里只返回一路文本。
    """
    env = runtime.env(api_key=api_key)
    proc, reader = _spawn_pty(args, env)
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        reader.join(timeout=5.0)
        pytest.fail(f"命令 {args} 超过 {timeout} 秒未结束")
    reader.join(timeout=5.0)
    return code, _normalize(reader.text())


def _cli_args(runtime: _Runtime, config: Path, *args: str) -> list[str]:
    """拼命令行参数：子命令 + 显式的配置与 env 文件路径。"""
    return [*args, "--config", str(config), "--env-file", str(runtime.env_file)]


# ----------------------------------------------------------------------
# 主进程
# ----------------------------------------------------------------------
class _MainProcess:
    """跑起来的 InterviewMate 主进程。"""

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        reader: _PtyReader,
        runtime: _Runtime,
        config_path: Path,
    ) -> None:
        self.proc = proc
        self.reader = reader
        self.runtime = runtime
        self.config_path = config_path

    @property
    def socket_path(self) -> Path:
        """主进程监听的 socket 文件路径。"""
        return self.runtime.socket_path

    @property
    def pid_path(self) -> Path:
        """主进程写的 PID 文件路径。"""
        return self.runtime.pid_path

    def output(self) -> str:
        """伪终端里读到的输出。"""
        return _normalize(self.reader.text())

    def cli_args(self, *args: str) -> list[str]:
        """拼指向本主进程的命令行参数。

        必须用主进程自己的运行期路径与配置文件，否则命令行侧找不到它。
        """
        return [
            *args,
            "--config",
            str(self.config_path),
            "--env-file",
            str(self.runtime.env_file),
        ]

    def wait_ready(self, timeout: float = START_TIMEOUT) -> None:
        """等主进程监听 socket 并写好 PID 文件。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                pytest.fail(
                    "主进程提前退出，退出码 "
                    f"{self.proc.returncode}，输出：\n{_plain(self.output())[-2000:]}"
                )
            if self.pid_path.exists() and _socket_reachable(self.socket_path):
                return
            time.sleep(0.1)
        pytest.fail(f"主进程在 {timeout} 秒内没有开始监听 socket")

    def wait_exit(self, timeout: float = STOP_TIMEOUT) -> int:
        """等主进程退出，返回退出码。"""
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"主进程在 {timeout} 秒内没有退出")

    def kill(self) -> None:
        """杀掉主进程并清掉运行期产物。"""
        _kill_process_group(self.proc)
        for path in (self.socket_path, self.pid_path):
            try:
                path.unlink()
            except OSError:
                pass


def _start_main_process(runtime: _Runtime, config: Path) -> _MainProcess:
    """起一个主进程子进程，走伪终端。"""
    runtime.root.mkdir(parents=True, exist_ok=True)
    env = runtime.env(module="app.main", api_key=FAKE_API_KEY)
    proc, reader = _spawn_pty(
        ["--config", str(config), "--env-file", str(runtime.env_file)], env
    )
    return _MainProcess(proc, reader, runtime, config)


# ----------------------------------------------------------------------
# 夹具
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def _process_registry() -> Iterator[list[_MainProcess]]:
    """登记所有起过的主进程，模块结束时统一收尾。"""
    registry: list[_MainProcess] = []
    yield registry
    for proc in reversed(registry):
        proc.kill()


@pytest.fixture
def runtime(tmp_path: Path) -> _Runtime:
    """一次子进程运行用的临时路径集合。"""
    return _Runtime(tmp_path)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """providers 四键全 mock 的配置副本。

    默认配置已切到真实实现；端到端用例要真起主进程，不能连真实服务，
    所以这里显式把四键覆盖成 mock。
    """
    return _write_config(
        tmp_path / "config.yaml",
        providers={
            "llm": "mock",
            "retrieval": "mock",
            "asr": "mock",
            "screenshot": "mock",
        },
    )


@pytest.fixture
def start_main_process(
    tmp_path: Path,
    config_path: Path,
    _process_registry: list[_MainProcess],
) -> Callable[[], _MainProcess]:
    """按需起主进程，每个进程一个独立的临时目录与 socket 路径。"""
    counter = itertools.count(1)

    def _start() -> _MainProcess:
        runtime = _Runtime(tmp_path / f"main-{next(counter)}")
        main = _start_main_process(runtime, config_path)
        _process_registry.append(main)
        main.wait_ready()
        return main

    return _start


def _json_line(out: str) -> dict[str, Any]:
    """取 stdout 里最后一行并解析成 JSON。"""
    lines = [line for line in out.strip().splitlines() if line.strip()]
    assert lines, f"输出里没有 JSON 行：\n{out}"
    payload = json.loads(lines[-1])
    assert isinstance(payload, dict)
    return payload


# ----------------------------------------------------------------------
# 验收一：providers.llm=deepseek 时缺 DEEPSEEK_API_KEY 以退出码 3 结束并指出变量名
# ----------------------------------------------------------------------
def test_start_without_api_key_exits_config_error(
    runtime: _Runtime, tmp_path: Path
) -> None:
    """providers.llm=deepseek 时缺 DEEPSEEK_API_KEY，start 以退出码 3 结束，输出点名这个变量。"""
    config = _write_config_with_llm(tmp_path / "deepseek.yaml", "deepseek")
    code, out, _err = _run_cli(
        _cli_args(runtime, config, "start"), runtime, api_key=None
    )
    assert code == EXIT_CONFIG_ERROR
    assert "DEEPSEEK_API_KEY" in out


# ----------------------------------------------------------------------
# 验收一：providers.llm=mock 时缺 DEEPSEEK_API_KEY 照常启动
# ----------------------------------------------------------------------
def test_start_with_mock_provider_ignores_missing_api_key(
    runtime: _Runtime, config_path: Path
) -> None:
    """providers.llm=mock 时缺 DEEPSEEK_API_KEY 不拦启动，也不提这个变量。"""
    code, out, _err = _run_cli(
        _cli_args(runtime, config_path, "start"), runtime, api_key=None
    )
    # 不是交互式终端，界面起不来，但已经走过必填变量那一步
    assert code == EXIT_ERROR
    assert "DEEPSEEK_API_KEY" not in out
    assert "不是交互式终端" in out


# ----------------------------------------------------------------------
# 验收五：配置文件里出现退役模型名时 start 报配置错误并列出被拒绝的名字
# ----------------------------------------------------------------------
def test_start_with_retired_model_exits_config_error(
    runtime: _Runtime, tmp_path: Path
) -> None:
    """deepseek.model 用了退役模型名时 start 以退出码 3 结束，输出列出被拒绝的名字。"""
    config = _write_config_with_model(tmp_path / "retired.yaml", "deepseek-chat")
    code, out, _err = _run_cli(_cli_args(runtime, config, "start"), runtime)
    assert code == EXIT_CONFIG_ERROR
    assert "deepseek-chat" in out
    # 输出里要能看到被拒绝的模型名清单，不只是自己那一个
    assert "deepseek-v4-flash" in out


# ----------------------------------------------------------------------
# 验收二：主进程不在运行时 status 明确说明进程没在运行
# ----------------------------------------------------------------------
def test_status_without_main_process_reports_not_running(
    runtime: _Runtime, config_path: Path
) -> None:
    """没有主进程时 status 退出码 0，说明进程没在运行，不打印 RUNNING。"""
    code, out, _err = _run_cli(_cli_args(runtime, config_path, "status"), runtime)
    assert code == EXIT_OK
    assert "主进程未运行" in out
    assert "InterviewMate: RUNNING" not in out


# ----------------------------------------------------------------------
# 验收三：主进程不在运行时 hotkey screenshot 返回 ok:false 与 E_SOCKET
# ----------------------------------------------------------------------
def test_hotkey_screenshot_without_main_process_reports_socket_error(
    runtime: _Runtime, config_path: Path
) -> None:
    """没有主进程时 hotkey screenshot 退出码 1，输出是能解析的一行 JSON。"""
    code, out, _err = _run_cli(
        _cli_args(runtime, config_path, "hotkey", "screenshot"), runtime
    )
    assert code == EXIT_ERROR
    payload = _json_line(out)
    assert payload["ok"] is False
    assert payload["error"] == "E_SOCKET"
    assert payload["action"] == "screenshot"


# ----------------------------------------------------------------------
# stop：主进程不在运行时清理残留文件
# ----------------------------------------------------------------------
def test_stop_without_main_process_cleans_stale_files(
    runtime: _Runtime, config_path: Path
) -> None:
    """没有主进程时 stop 退出码 0，说明未运行并清掉残留文件。"""
    runtime.socket_path.write_text("残留的 socket 文件", encoding="utf-8")
    runtime.pid_path.write_text("12345\n", encoding="utf-8")

    code, out, _err = _run_cli(_cli_args(runtime, config_path, "stop"), runtime)

    assert code == EXIT_OK
    assert "主进程未运行" in out
    assert "已清理残留文件" in out
    assert not runtime.socket_path.exists()
    assert not runtime.pid_path.exists()


# ----------------------------------------------------------------------
# 验收二与三：主进程在运行时 status 与 hotkey 走 socket
# ----------------------------------------------------------------------
def test_main_process_lifecycle_over_socket(
    start_main_process: Callable[[], _MainProcess],
) -> None:
    """主进程在运行时的一整条链路。

    真起主进程（走伪终端）→ status 给出完整状态 → hotkey screenshot 与
    hotkey voice-toggle 走 socket 拿到 ok:true → stop 让主进程退出，
    退出后 socket 文件与 PID 文件都不存在。
    """
    main = start_main_process()
    assert main.socket_path.exists()
    assert main.pid_path.exists()

    # status：首行是 RUNNING，六项状态都在
    code, out, _err = _run_cli(main.cli_args("status"), main.runtime)
    assert code == EXIT_OK
    lines = out.strip().splitlines()
    assert lines[0] == "InterviewMate: RUNNING"
    for label in ("Audio:", "ASR:", "Dify:", "DeepSeek:", "TTFT:"):
        assert any(line.startswith(label) for line in lines), f"状态里少了 {label}"

    # hotkey screenshot：走 socket 拿主进程的响应
    code, out, _err = _run_cli(main.cli_args("hotkey", "screenshot"), main.runtime)
    assert code == EXIT_OK
    payload = _json_line(out)
    assert payload["ok"] is True
    assert payload["action"] == "screenshot"
    assert payload["payload"]["provider"] == "mock"

    # hotkey voice-toggle：主进程状态真的变了，再问一次 status 能看出来
    code, out, _err = _run_cli(main.cli_args("hotkey", "voice-toggle"), main.runtime)
    assert code == EXIT_OK
    payload = _json_line(out)
    assert payload["ok"] is True
    assert payload["action"] == "voice-toggle"
    assert payload["payload"]["audio"] == "on"

    code, out, _err = _run_cli(main.cli_args("status"), main.runtime)
    assert code == EXIT_OK
    assert "Audio: on" in out

    # 主进程确实起了 Textual 界面，不只是起了 socket 服务
    assert "关键回答内容" in _plain(main.output())

    # stop：主进程收尾，运行期产物清干净
    code, out, _err = _run_cli(main.cli_args("stop"), main.runtime)
    assert code == EXIT_OK
    assert "主进程已停止" in out

    assert main.wait_exit() == EXIT_OK
    assert not main.socket_path.exists()
    assert not main.pid_path.exists()


# ----------------------------------------------------------------------
# 进程通信：warmup 走 socket
# ----------------------------------------------------------------------
def test_main_process_answers_warmup_over_socket(
    start_main_process: Callable[[], _MainProcess],
) -> None:
    """主进程在运行时 warmup 走 socket，输出 mock 标记，不声称模型已驻留。"""
    main = start_main_process()

    code, out, _err = _run_cli(main.cli_args("warmup"), main.runtime)

    assert code == EXIT_OK
    assert "asr: mock" in out
    assert "deepseek: mock" in out
    assert "dify: mock" in out
    assert "已驻留内存" not in out


# ----------------------------------------------------------------------
# 进程通信：socket 协议的一问一答
# ----------------------------------------------------------------------
def test_socket_protocol_round_trip_over_real_socket(
    start_main_process: Callable[[], _MainProcess],
) -> None:
    """直接按协议连主进程的 socket：发一行请求、读一行响应。"""
    main = start_main_process()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(10.0)
        client.connect(str(main.socket_path))
        client.sendall(b'{"action":"status","request_id":"e2e-1"}\n')
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    finally:
        client.close()

    raw = b"".join(chunks).split(b"\n", 1)[0]
    response = json.loads(raw.decode("utf-8"))
    assert response["ok"] is True
    assert response["error"] == ""
    assert response["payload"]["running"] == "running"
    assert isinstance(response["payload"]["pid"], int)

    # 非法 action 也要回一行 ok:false 与 E_SOCKET，服务端不崩
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(10.0)
        client.connect(str(main.socket_path))
        client.sendall(b'{"action":"no-such-action"}\n')
        raw = client.recv(4096).split(b"\n", 1)[0]
    finally:
        client.close()
    response = json.loads(raw.decode("utf-8"))
    assert response["ok"] is False
    assert response["error"] == "E_SOCKET"


# ----------------------------------------------------------------------
# 进程通信：主进程在运行时再来一次 start，旧实例被停掉、新实例接管
# ----------------------------------------------------------------------
def test_start_replaces_running_instance(
    start_main_process: Callable[[], _MainProcess],
) -> None:
    """主进程在运行时再 start：旧实例先退出，新实例写自己的 PID 并接管端点。"""
    main = start_main_process()
    old_pid = main.proc.pid
    assert main.socket_path.exists()

    env = main.runtime.env(api_key=FAKE_API_KEY)
    second, reader = _spawn_pty(main.cli_args("start"), env)
    try:
        # 旧实例走 socket 的 stop 正常退出
        assert main.proc.wait(timeout=STOP_TIMEOUT) == EXIT_OK

        # 新实例重新写 PID 文件并监听端点
        new_pid: int | None = None
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            if second.poll() is not None:
                pytest.fail(
                    "接管的主进程提前退出，退出码 "
                    f"{second.returncode}，输出：\n{_plain(reader.text())[-2000:]}"
                )
            try:
                candidate = int(main.pid_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                candidate = None
            if (
                candidate is not None
                and candidate != old_pid
                and _socket_reachable(main.socket_path)
            ):
                new_pid = candidate
                break
            time.sleep(0.1)
        assert new_pid is not None, "新实例没有在超时时间内接管 PID 文件与端点"
    finally:
        _kill_process_group(second)
        reader.join(timeout=5.0)


# ----------------------------------------------------------------------
# 验收四：日志文件里搜不到 DEEPSEEK_API_KEY 的值与 Authorization 头的值
# ----------------------------------------------------------------------
def test_log_file_does_not_contain_api_key_or_authorization_value(
    start_main_process: Callable[[], _MainProcess],
    config_path: Path,
    tmp_path: Path,
) -> None:
    """真实子进程写的日志文件里搜不到密钥值与 Authorization 头的令牌。

    子进程启动时先用项目的日志管道把这两样原样写一遍（探针），
    再跑主进程，这样断言不是空断言：脱敏坏掉时这里会失败。
    """
    runtime = _Runtime(tmp_path / "log-probe")
    runtime.root.mkdir(parents=True, exist_ok=True)
    env = runtime.env(module="app.main", api_key=FAKE_API_KEY, log_probe=True)
    proc, reader = _spawn_pty(
        ["--config", str(config_path), "--env-file", str(runtime.env_file)], env
    )
    main = _MainProcess(proc, reader, runtime, config_path)
    try:
        main.wait_ready()
        code, out, _err = _run_cli(
            _cli_args(runtime, config_path, "stop"), runtime
        )
        assert code == EXIT_OK
        assert main.wait_exit() == EXIT_OK
    finally:
        main.kill()

    log_files = sorted((runtime.log_root / "logs").glob("interviewmate-*.log"))
    assert log_files, "子进程没有写出日志文件"
    text = log_files[-1].read_text(encoding="utf-8")

    # 主进程确实写过日志
    assert "event=main.start" in text
    # 密钥值与 Authorization 令牌都不在日志里
    assert FAKE_API_KEY not in text
    assert PROBE_TOKEN not in text
    # 脱敏确实发生过：原文被换成了占位符
    assert "***" in text


# ----------------------------------------------------------------------
# test：只读自检逐项打印结果
# ----------------------------------------------------------------------
def test_test_command_prints_self_check_items(
    runtime: _Runtime, config_path: Path
) -> None:
    """test 逐项打印自检结果，mock 的能力打印 mock。

    本机音频设备与截图接口不一定可用，所以不断言退出码是 0，
    只断言项名与 mock 标记。
    """
    code, out, _err = _run_cli(_cli_args(runtime, config_path, "test"), runtime)

    # 契约：全部通过退出码 0，有失败项退出码 1
    assert code in (EXIT_OK, EXIT_ERROR)
    for item in (
        "配置加载与校验",
        "依赖包导入",
        "音频设备枚举",
        "截图接口探测",
        "ASR 加载",
        "DeepSeek 探测",
        "Dify 探测",
    ):
        assert item in out, f"自检输出里少了 {item}"
    # providers 四键全 mock，对应的三项打印 mock
    assert "[mock] ASR 加载" in out
    assert "[mock] DeepSeek 探测" in out
    assert "[mock] Dify 探测" in out


# ----------------------------------------------------------------------
# M2 装配：输入框提交 → mock 流式回答 → 左右两栏渲染
# ----------------------------------------------------------------------
def _generation_finished(app: InterviewMateApp) -> bool:
    """这一轮是不是已经结束：右栏有内容，状态栏 DEEPSEEK 回到 idle。"""
    if not app.answer_panel.plain_text().strip():
        return False
    return app.query_one(StatusBar).status_snapshot()["deepseek"] == "idle"


async def _wait_for_generation(app: InterviewMateApp, pilot: Any) -> None:
    """等这一轮生成结束，超时直接判失败。"""
    deadline = time.monotonic() + GENERATION_TIMEOUT
    while time.monotonic() < deadline:
        await pilot.pause()
        if _generation_finished(app):
            return
        await asyncio.sleep(0.01)
    pytest.fail("生成没有在预期时间内结束")


async def test_mock_stream_fills_two_columns_in_running_app(
    config_path: Path,
) -> None:
    """无头启动界面，提交一句话，左栏出框架标题、右栏出表达单元、状态栏 DEEPSEEK 回到 idle。"""
    config = load_config(str(config_path))
    app = InterviewMateApp(config=config, bus=EventBus())
    client = MockClient(config)

    async with app.run_test(size=(WIDE_WIDTH, WIDE_HEIGHT)) as pilot:
        await pilot.pause()
        assemble_manager(config, client, app)
        await pilot.pause()

        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        await pilot.press(*"请解释一下 Raft 协议")
        await pilot.press("enter")
        await _wait_for_generation(app, pilot)

        # 左栏：框架标题跟着题目走
        assert "Raft 基本原理" in app.navigation_panel.plain_text()
        # 右栏：表达单元的关键词与表达都在
        answer_text = app.answer_panel.plain_text()
        assert "资源管理" in answer_text
        assert "把资源的申请与释放绑定到对象生命周期上" in answer_text
        # 状态栏：这一轮结束后 DEEPSEEK 回到 idle
        assert app.query_one(StatusBar).status_snapshot()["deepseek"] == "idle"


# ----------------------------------------------------------------------
# M3 截图：socket 动作真的采集，界面里的截图流程只改提示位
# ----------------------------------------------------------------------
def _screenshot_payload(main: _MainProcess, attempts: int = 40) -> dict[str, Any]:
    """反复发 screenshot 动作，直到真的采集成功。

    界面起来之前动作只做接口探测，返回的 captured 是 False，
    所以这里重试到采集成功为止。
    """
    payload: dict[str, Any] = {}
    for _ in range(attempts):
        code, out, _err = _run_cli(
            main.cli_args("hotkey", "screenshot"), main.runtime
        )
        payload = _json_line(out)
        if code == EXIT_OK and payload.get("payload", {}).get("captured") is True:
            return payload
        time.sleep(0.3)
    pytest.fail(f"截图动作没有真正触发采集，最后一次响应：{payload}")


def test_hotkey_screenshot_captures_and_submits_with_mock_backend(
    start_main_process: Callable[[], _MainProcess],
) -> None:
    """默认配置下 socket 发 screenshot 能走通采集 → 处理 → 提交带图请求。"""
    main = start_main_process()

    payload = _screenshot_payload(main)

    assert payload["ok"] is True
    assert payload["action"] == "screenshot"
    assert payload["payload"]["captured"] is True
    assert payload["payload"]["provider"] == "mock"


def test_status_shows_capture_latency_after_screenshot(
    start_main_process: Callable[[], _MainProcess],
) -> None:
    """截图走过一遍后，status 里能看到采集耗时统计。"""
    main = start_main_process()

    _screenshot_payload(main)

    code, out, _err = _run_cli(main.cli_args("status"), main.runtime)
    assert code == EXIT_OK
    assert "指标统计" in out
    assert "capture_latency" in out
    assert "ms" in out


async def test_screenshot_flow_keeps_other_status_fields(
    config_path: Path,
) -> None:
    """界面里走一次截图，状态栏只有提示位变了，其余六项保持原样。"""
    config = load_config(str(config_path))
    app = InterviewMateApp(config=config, bus=EventBus())
    client = MockClient(config)

    async with app.run_test(size=(WIDE_WIDTH, WIDE_HEIGHT)) as pilot:
        await pilot.pause()
        manager = assemble_manager(config, client, app)
        assemble_screenshot_manager(config, app, manager)
        await pilot.pause()

        # 先把状态栏填上，再走截图
        await app.publish_status(
            running="running",
            audio="on",
            asr="ready",
            dify="ready",
            deepseek="idle",
            ttft_ms=120.0,
            hint="生成中",
        )
        await pilot.pause()
        bar = app.query_one(StatusBar)
        before = bar.status_snapshot()

        app.action_screenshot()
        await _wait_for_generation(app, pilot)

        # 提示位：截图期间写「截图中」，结束后清掉
        assert bar.hint == ""
        # 其余六项按快照原样保留，没有被默认值覆盖
        assert bar.status_snapshot() == before
        # 图真的提交了：右栏出了内容
        assert app.answer_panel.plain_text().strip()


async def test_socket_screenshot_shows_question_header(
    config_path: Path,
) -> None:
    """socket 那条路也要在右栏顶部显示题面。

    TUI 内的 F9 与 socket 的 ``screenshot`` 动作是两条触发路径，
    两条都要显示题面，见《4-截图输入与多模态》截图流程一节。
    """
    config = load_config(str(config_path))
    app = InterviewMateApp(config=config, bus=EventBus())
    client = MockClient(config)

    async with app.run_test(size=(WIDE_WIDTH, WIDE_HEIGHT)) as pilot:
        await pilot.pause()
        manager = assemble_manager(config, client, app)
        screenshot_manager = assemble_screenshot_manager(config, app, manager)
        state = MainState()
        state.attach_app(app)
        state.set_screenshot_manager(screenshot_manager)
        await pilot.pause()

        payload = await state.capture_screenshot()
        await _wait_for_generation(app, pilot)

        assert payload["captured"] is True
        assert app.query_one(QuestionHeader).question == "截图提问"
        assert app.answer_panel.plain_text().strip()
