"""端到端测试（Windows）：真起命令行子进程，走真实 TCP loopback IPC。

覆盖《2-骨架与配置》验收一节的几条，以及进程通信一节的完整走查：

- 命令行子进程：``start`` 的配置错误、``status`` 未运行说明、``hotkey`` 的
  ``ok:false`` 与 ``E_SOCKET``、``stop`` 的残留清理、``test`` 的逐项自检；
- IPC：在进程内起真实 :class:`~app.runtime.socket_server.SocketServer`（TCP
  loopback，写端口文件与 token 文件），用客户端 :func:`send_request` 走完
  status / warmup / 非法 action / 端点文件清理；
- 日志脱敏：子进程把密钥与 Authorization 原样交给日志管道，日志文件里搜不到；
- M2 装配：无头启动界面，输入框提交 → mock 流式回答 → 左右两栏渲染；
- M3 截图：mock 后端走「采集 → 处理 → 提交带图请求」。

隔离办法
--------
命令行子进程先 import 相关模块，把模块属性里的 ``SOCKET_PATH`` 与 ``PID_PATH``
统一改到 pytest 的 ``tmp_path``，再以 ``__main__`` 身份执行 ``app.cli``：等价于
``python -m app.cli``，但运行期文件不落在仓库根的 ``runtime/`` 里。

不连真实服务
------------
全程用 ``providers`` 四键全 mock 的配置副本，``DEEPSEEK_API_KEY`` 用假值，
不访问 DeepSeek、Dify、真实 ASR 模型或真实截图服务。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
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
from app.runtime.socket_client import send_request
from app.runtime.socket_server import SocketServer
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

# 在子进程里做的事：改运行期路径 → 写脱敏探针 → 以 __main__ 身份执行 app.cli。
# 用 runpy 而不是直接调函数，是为了和 ``python -m app.cli`` 走同一份 __main__ 代码路径。
_BOOTSTRAP: str = r'''
import os
import runpy
import sys
import warnings
from pathlib import Path

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
_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    """去掉控制序列与所有空白，用于在界面渲染结果里找固定文案。"""
    return re.sub(r"\s+", "", _ANSI_PATTERN.sub("", text))


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


# ----------------------------------------------------------------------
# 运行期环境
# ----------------------------------------------------------------------
class _Runtime:
    """一次子进程运行用到的临时路径与环境变量。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.socket_path = root / "interviewmate.port"
        self.token_path = root / "interviewmate.token"
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
        env["PYTHONUNBUFFERED"] = "1"
        # 子进程输出里有中文，统一用 UTF-8，避免 Windows 默认 cp1252 解码失败
        env["PYTHONIOENCODING"] = "utf-8"
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
        )
        return env


def _run_cli(
    args: list[str],
    runtime: _Runtime,
    *,
    api_key: str | None = FAKE_API_KEY,
    log_probe: bool = False,
    timeout: float = CLI_TIMEOUT,
) -> tuple[int, str, str]:
    """跑一次命令行子进程，返回 (退出码, stdout, stderr)。

    超时按失败处理：杀掉子进程并让测试失败，不会挂死。
    """
    env = runtime.env(api_key=api_key, log_probe=log_probe)
    proc = subprocess.Popen(
        [sys.executable, "-c", _BOOTSTRAP, *args],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(f"命令 {args} 超过 {timeout} 秒未结束，输出：\n{out}\n{err}")
    return proc.returncode, out, err


def _cli_args(runtime: _Runtime, config: Path, *args: str) -> list[str]:
    """拼命令行参数：子命令 + 显式的配置与 env 文件路径。"""
    return [*args, "--config", str(config), "--env-file", str(runtime.env_file)]


def _json_line(out: str) -> dict[str, Any]:
    """取 stdout 里最后一行并解析成 JSON。"""
    lines = [line for line in out.strip().splitlines() if line.strip()]
    assert lines, f"输出里没有 JSON 行：\n{out}"
    payload = json.loads(lines[-1])
    assert isinstance(payload, dict)
    return payload


# ----------------------------------------------------------------------
# 夹具
# ----------------------------------------------------------------------
@pytest.fixture
def runtime(tmp_path: Path) -> _Runtime:
    """一次子进程运行用的临时路径集合。"""
    return _Runtime(tmp_path)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """providers 四键全 mock 的配置副本。"""
    return _write_config(
        tmp_path / "config.yaml",
        providers={
            "llm": "mock",
            "retrieval": "mock",
            "asr": "mock",
            "screenshot": "mock",
        },
    )


# ----------------------------------------------------------------------
# 命令行：start 的配置校验
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


def test_start_with_retired_model_exits_config_error(
    runtime: _Runtime, tmp_path: Path
) -> None:
    """deepseek.model 用了退役模型名时 start 以退出码 3 结束，输出列出被拒绝的名字。"""
    config = _write_config_with_model(tmp_path / "retired.yaml", "deepseek-chat")
    code, out, _err = _run_cli(_cli_args(runtime, config, "start"), runtime)
    assert code == EXIT_CONFIG_ERROR
    assert "deepseek-chat" in out
    assert "deepseek-v4-flash" in out


# ----------------------------------------------------------------------
# 命令行：status / hotkey / stop
# ----------------------------------------------------------------------
def test_status_without_main_process_reports_not_running(
    runtime: _Runtime, config_path: Path
) -> None:
    """没有主进程时 status 退出码 0，说明进程没在运行，不打印 RUNNING。"""
    code, out, _err = _run_cli(_cli_args(runtime, config_path, "status"), runtime)
    assert code == EXIT_OK
    assert "主进程未运行" in out
    assert "InterviewMate: RUNNING" not in out


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


def test_stop_without_main_process_cleans_stale_files(
    runtime: _Runtime, config_path: Path
) -> None:
    """没有主进程时 stop 退出码 0，说明未运行并清掉残留的端口与 PID 文件。"""
    runtime.socket_path.write_text("残留的端点文件", encoding="utf-8")
    runtime.token_path.write_text("残留的 token", encoding="utf-8")
    runtime.pid_path.write_text("12345\n", encoding="utf-8")

    code, out, _err = _run_cli(_cli_args(runtime, config_path, "stop"), runtime)

    assert code == EXIT_OK
    assert "主进程未运行" in out
    assert "已清理残留文件" in out
    assert not runtime.socket_path.exists()
    assert not runtime.token_path.exists()
    assert not runtime.pid_path.exists()


def test_test_command_prints_self_check_items(
    runtime: _Runtime, config_path: Path
) -> None:
    """test 逐项打印自检结果，mock 的能力打印 mock。"""
    code, out, _err = _run_cli(_cli_args(runtime, config_path, "test"), runtime)

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
    assert "[mock] ASR 加载" in out
    assert "[mock] DeepSeek 探测" in out
    assert "[mock] Dify 探测" in out


# ----------------------------------------------------------------------
# 日志脱敏：子进程写的日志文件里搜不到密钥值与 Authorization 头的值
# ----------------------------------------------------------------------
def test_log_file_does_not_contain_api_key_or_authorization_value(
    runtime: _Runtime, config_path: Path
) -> None:
    """真实子进程写的日志文件里搜不到密钥值与 Authorization 头的令牌。"""
    code, _out, _err = _run_cli(
        _cli_args(runtime, config_path, "status"), runtime, log_probe=True
    )
    assert code == EXIT_OK

    log_files = sorted((runtime.log_root / "logs").glob("interviewmate-*.log"))
    assert log_files, "子进程没有写出日志文件"
    text = log_files[-1].read_text(encoding="utf-8")

    assert "event=e2e.log_probe" in text
    assert FAKE_API_KEY not in text
    assert PROBE_TOKEN not in text
    assert "***" in text


# ----------------------------------------------------------------------
# IPC：真实 TCP loopback 端点，客户端 send_request 往返
# ----------------------------------------------------------------------
async def test_ipc_round_trip_over_real_tcp_endpoint(tmp_path: Path) -> None:
    """进程内起真实 SocketServer，走端口文件 + token 完成 IPC 往返与错误处理。"""
    endpoint = tmp_path / "interviewmate.port"

    def handle_status(request: Any) -> dict[str, Any]:
        return {"running": "running", "pid": os.getpid()}

    def handle_warmup(request: Any) -> dict[str, Any]:
        return {"asr": "mock", "deepseek": "mock", "dify": "mock"}

    handlers = {
        "status": handle_status,
        "warmup": handle_warmup,
        "screenshot": lambda request: {"captured": True, "provider": "mock"},
    }
    server = SocketServer(socket_path=endpoint, handlers=handlers)
    await server.start()
    try:
        # 端口文件与 token 文件都写出来了
        assert endpoint.exists()
        assert (tmp_path / "interviewmate.token").exists()

        status = await asyncio.to_thread(
            send_request, "status", socket_path=endpoint
        )
        assert status.ok is True
        assert status.error == ""
        assert status.payload["running"] == "running"
        assert isinstance(status.payload["pid"], int)

        warmup = await asyncio.to_thread(
            send_request, "warmup", socket_path=endpoint
        )
        assert warmup.ok is True
        assert warmup.payload == {"asr": "mock", "deepseek": "mock", "dify": "mock"}

        # 非法 action 回一行 ok:false 与 E_SOCKET，服务端不崩
        bad = await asyncio.to_thread(
            send_request, "no-such-action", socket_path=endpoint
        )
        assert bad.ok is False
        assert bad.error == "E_SOCKET"
    finally:
        await server.stop()

    # 停服务后端口文件与 token 文件都清掉
    assert not endpoint.exists()
    assert not (tmp_path / "interviewmate.token").exists()


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

        assert "Raft 基本原理" in app.navigation_panel.plain_text()
        answer_text = app.answer_panel.plain_text()
        assert "资源管理" in answer_text
        assert "把资源的申请与释放绑定到对象生命周期上" in answer_text
        assert app.query_one(StatusBar).status_snapshot()["deepseek"] == "idle"


# ----------------------------------------------------------------------
# M3 截图：mock 后端走通「采集 → 处理 → 提交带图请求」
# ----------------------------------------------------------------------
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

        assert bar.hint == ""
        assert bar.status_snapshot() == before
        assert app.answer_panel.plain_text().strip()


async def test_socket_screenshot_shows_question_header(
    config_path: Path,
) -> None:
    """socket 那条路也要在右栏顶部显示题面。"""
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
