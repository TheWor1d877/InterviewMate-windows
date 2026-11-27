"""命令行入口与进程装配的测试。

覆盖《2-骨架与配置》命令行入口一节的子命令、退出码表与验收一节：
参数错误退出码 2、主进程没运行时 ``status`` 的说明、``hotkey`` 的
``ok: false`` 与 ``E_SOCKET``、缺 ``DEEPSEEK_API_KEY`` 的退出码 3、
退役模型名的配置错误、``test`` 的逐项自检。

测试不真起界面进程，也不连真实服务：socket 侧用假响应替身，
音频采集用假波形替身。
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

import app.cli
import app.config
import app.main
import app.runtime.socket_client
from app.cli import main as cli_main
from app.core.exceptions import SocketError
from app.main import EXIT_ARG_ERROR, EXIT_CONFIG_ERROR, EXIT_ERROR, EXIT_OK
from app.platform.types import AudioDevice
from app.runtime.socket_protocol import ACTIONS, Request, Response
from app.utils.timing import default_recorder, reset_default_recorder

# 会影响配置结果的环境变量，测试里统一清掉
_OVERRIDE_ENV = ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """清掉会影响配置的环境变量，并把默认 .env 路径指向不存在的文件。

    开发者在自己机器上建过 .env 时，断言不应受其影响。
    """
    for name in _OVERRIDE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app.config, "DEFAULT_ENV_FILE", tmp_path / "not-exists.env")


@pytest.fixture(autouse=True)
def _clean_default_recorder() -> Iterator[None]:
    """每个用例前清空进程级记录器，指标断言不受别的用例影响。"""
    reset_default_recorder()
    yield
    reset_default_recorder()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """providers 四键全 mock 的配置副本。

    默认配置已切到真实实现；本文件里需要启动或装配的用例都按 mock 行为断言，
    所以这里显式把四键覆盖成 mock，断言不变。
    """
    data = yaml.safe_load(
        app.config.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    )
    data["providers"] = {
        "llm": "mock",
        "retrieval": "mock",
        "asr": "mock",
        "screenshot": "mock",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _mock_config() -> app.config.AppConfig:
    """读默认配置再把 providers 四键覆盖成 mock。

    默认配置已切到真实实现，按 mock 行为断言的用例用这份配置，
    不连网络、不加载模型。
    """
    config = app.config.load_config()
    config.providers.llm = "mock"
    config.providers.retrieval = "mock"
    config.providers.asr = "mock"
    config.providers.screenshot = "mock"
    return config


@pytest.fixture
def runtime_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """把 PID 文件与 socket 文件指到 tmp_path，不动仓库里的运行期产物。"""
    socket_path = tmp_path / "interviewmate.port"
    pid_path = tmp_path / "interviewmate.pid"
    for module in (app.cli, app.main):
        monkeypatch.setattr(module, "SOCKET_PATH", socket_path)
        monkeypatch.setattr(module, "PID_PATH", pid_path)
    return socket_path, pid_path


def _write_config(tmp_path: Path, **overrides: Any) -> Path:
    """写一份改了若干顶层段的配置，返回路径。"""
    data = yaml.safe_load(app.config.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for section, value in overrides.items():
        data[section] = value
    path = tmp_path / "patched.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# 参数错误：退出码 2
# ----------------------------------------------------------------------
def test_no_subcommand_returns_argument_error(capsys: pytest.CaptureFixture[str]) -> None:
    """不带子命令时退出码是 2。"""
    assert cli_main([]) == EXIT_ARG_ERROR
    assert "参数错误" in capsys.readouterr().err


def test_unknown_subcommand_returns_argument_error() -> None:
    """未知子命令时退出码是 2。"""
    assert cli_main(["badcmd"]) == EXIT_ARG_ERROR


def test_hotkey_without_action_returns_argument_error() -> None:
    """hotkey 缺子动作时退出码是 2。"""
    assert cli_main(["hotkey"]) == EXIT_ARG_ERROR


def test_unknown_option_returns_argument_error() -> None:
    """参数不合法时退出码是 2。"""
    assert cli_main(["status", "--nope"]) == EXIT_ARG_ERROR


# ----------------------------------------------------------------------
# status
# ----------------------------------------------------------------------
def test_status_without_main_process(
    runtime_paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """主进程没运行时，status 说清楚进程没在运行，退出码 0。"""
    assert cli_main(["status"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "主进程未运行" in out
    assert "RUNNING" not in out
    assert "Audio: off" in out
    assert "TTFT: -" in out


def test_status_with_running_main_process(
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """主进程在运行时，status 走 socket 并按状态文本格式输出。"""
    monkeypatch.setattr(app.cli, "is_server_alive", lambda path=None: True)

    def fake_send_request(action: str, *args: Any, **kwargs: Any) -> Response:
        return Response(
            ok=True,
            payload={
                "running": "running",
                "audio": "off",
                "asr": "idle",
                "dify": "offline",
                "deepseek": "idle",
                "ttft": None,
                "pid": 4242,
            },
        )

    monkeypatch.setattr(app.cli, "send_request", fake_send_request)
    assert cli_main(["status"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "InterviewMate: RUNNING" in out
    assert "PID: 4242" in out
    assert "Audio: off" in out
    assert "ASR: idle" in out
    assert "Dify: offline" in out
    assert "DeepSeek: idle" in out
    assert "TTFT: -" in out
    # 没有指标样本时不打印指标段
    assert "指标统计" not in out


def test_status_with_running_main_process_prints_metrics(
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """主进程带着指标样本时，status 在状态文本后面打印指标统计段。"""
    monkeypatch.setattr(app.cli, "is_server_alive", lambda path=None: True)

    def fake_send_request(action: str, *args: Any, **kwargs: Any) -> Response:
        return Response(
            ok=True,
            payload={
                "running": "running",
                "audio": "off",
                "asr": "idle",
                "dify": "offline",
                "deepseek": "idle",
                "ttft": None,
                "pid": 4242,
                "metrics": [
                    {
                        "name": "capture_latency",
                        "count": 3,
                        "p50_ms": 12.0,
                        "p90_ms": 14.0,
                        "p95_ms": 15.0,
                        "max_ms": 15.0,
                    }
                ],
            },
        )

    monkeypatch.setattr(app.cli, "send_request", fake_send_request)
    assert cli_main(["status"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "指标统计" in out
    assert "capture_latency" in out
    assert "样本 3" in out
    assert "P50 12.0 ms" in out
    assert "P95 15.0 ms" in out
    assert "最大 15.0 ms" in out
    # 指标段在六项状态之后
    assert out.index("TTFT:") < out.index("指标统计")


# ----------------------------------------------------------------------
# hotkey
# ----------------------------------------------------------------------
def test_hotkey_screenshot_without_main_process(
    runtime_paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """主进程没运行时，hotkey 返回 ok:false 与 E_SOCKET，退出码 1。"""
    assert cli_main(["hotkey", "screenshot"]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert '"ok": false' in out
    assert "E_SOCKET" in out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert payload["error"] == "E_SOCKET"
    assert payload["action"] == "screenshot"


def test_hotkey_voice_toggle_without_main_process(
    runtime_paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """voice-toggle 同样走 socket，主进程没运行时返回 ok:false。"""
    assert cli_main(["hotkey", "voice-toggle"]) == EXIT_ERROR
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert payload["error"] == "E_SOCKET"
    assert payload["action"] == "voice-toggle"


def test_hotkey_with_running_main_process(
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """主进程在运行时，hotkey 打印主进程给的响应。"""
    monkeypatch.setattr(app.cli, "is_server_alive", lambda path=None: True)
    monkeypatch.setattr(
        app.cli,
        "send_request",
        lambda action, *args, **kwargs: Response(
            ok=True, payload={"captured": False, "provider": "mock"}
        ),
    )
    assert cli_main(["hotkey", "screenshot"]) == EXIT_OK
    out = capsys.readouterr().out
    assert '"ok": true' in out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["payload"]["captured"] is False


# ----------------------------------------------------------------------
# start
# ----------------------------------------------------------------------
def test_start_without_api_key_exits_config_error(
    tmp_path: Path,
    runtime_paths: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """providers.llm=deepseek 时缺 DEEPSEEK_API_KEY，退出码是 3，输出里点名这个变量。"""
    path = _write_config(
        tmp_path,
        providers={
            "llm": "deepseek",
            "retrieval": "mock",
            "asr": "mock",
            "screenshot": "mock",
        },
    )
    assert cli_main(["start", "--config", str(path)]) == EXIT_CONFIG_ERROR
    out = capsys.readouterr().out
    assert "DEEPSEEK_API_KEY" in out


def test_start_with_mock_provider_ignores_missing_api_key(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """providers.llm=mock 时缺 DEEPSEEK_API_KEY 不拦启动，输出里也不提这个变量。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(app.main, "_stdout_isatty", lambda: False)
    assert cli_main(["start", "--config", str(config_path)]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "DEEPSEEK_API_KEY" not in out
    assert "不是交互式终端" in out


def test_start_with_retired_model_exits_config_error(
    tmp_path: Path,
    runtime_paths: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """配置里用了退役模型名时退出码是 3，输出里列出被拒绝的名字。"""
    path = _write_config(tmp_path, deepseek={"model": "deepseek-chat"})
    assert cli_main(["start", "--config", str(path)]) == EXIT_CONFIG_ERROR
    out = capsys.readouterr().out
    assert "deepseek-chat" in out


def test_start_in_non_interactive_terminal(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """非交互式终端里不启动界面，退出码 1。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test-not-a-real-key")
    monkeypatch.setattr(app.main, "_stdout_isatty", lambda: False)
    assert cli_main(["start", "--config", str(config_path)]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "不是交互式终端" in out


def test_start_clears_stale_pid_of_foreign_process(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PID 文件里是别的活进程时按残留文件处理，不结束它，start 照常继续。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test-not-a-real-key")
    monkeypatch.setattr(app.main, "_stdout_isatty", lambda: True)
    monkeypatch.setattr(app.main, "is_server_alive", lambda path=None: False)
    monkeypatch.setattr(app.main, "find_app_processes", lambda **kwargs: ())
    started: list[Any] = []

    async def fake_run_main(config: app.config.AppConfig) -> int:
        started.append(config)
        return EXIT_OK

    monkeypatch.setattr(app.main, "run_main", fake_run_main)
    _, pid_path = runtime_paths
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    assert cli_main(["start", "--config", str(config_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "不是本应用" in out
    assert not pid_path.exists()
    assert started


def test_start_gives_up_when_old_instance_survives(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """旧实例停不下来时 start 放弃，不去抢端点。"""
    monkeypatch.setattr(app.main, "_stdout_isatty", lambda: True)
    monkeypatch.setattr(app.main, "is_server_alive", lambda path=None: True)
    monkeypatch.setattr(app.main, "STALE_STOP_TIMEOUT", 0.0)
    monkeypatch.setattr(app.main, "find_app_processes", lambda **kwargs: ())

    def refuse_stop(action: str, *args: Any, **kwargs: Any) -> Response:
        raise SocketError("连不上旧实例")

    monkeypatch.setattr(app.main, "send_request", refuse_stop)

    assert cli_main(["start", "--config", str(config_path)]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "放弃启动" in out


def test_start_with_missing_package_exits_error(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """依赖包缺一个时退出码是 1，输出里报出包名。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test-not-a-real-key")
    monkeypatch.setattr(app.main, "check_dependencies", lambda packages: ["sounddevice"])
    assert cli_main(["start", "--config", str(config_path)]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "sounddevice" in out


# ----------------------------------------------------------------------
# stop
# ----------------------------------------------------------------------
def test_stop_without_main_process_cleans_stale_files(
    runtime_paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """主进程没运行时，stop 清理残留文件，退出码 0。"""
    socket_path, pid_path = runtime_paths
    socket_path.write_text("残留", encoding="utf-8")
    pid_path.write_text("12345\n", encoding="utf-8")
    assert cli_main(["stop"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "主进程未运行" in out
    assert "已清理残留文件" in out
    assert not socket_path.exists()
    assert not pid_path.exists()


def test_stop_with_running_main_process(
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """主进程在运行时，stop 发 stop 动作并等它退出，退出码 0。"""
    monkeypatch.setattr(app.cli, "is_server_alive", lambda path=None: True)
    monkeypatch.setattr(
        app.cli,
        "send_request",
        lambda action, *args, **kwargs: Response(ok=True, payload={"stopping": True}),
    )
    monkeypatch.setattr(app.cli, "_wait_for_exit", lambda *args, **kwargs: True)
    assert cli_main(["stop"]) == EXIT_OK
    assert "主进程已停止" in capsys.readouterr().out


# ----------------------------------------------------------------------
# warmup
# ----------------------------------------------------------------------
def test_warmup_without_main_process_does_not_claim_residency(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """主进程没运行时，warmup 只做环境检查，不声称模型已驻留内存。"""
    assert cli_main(["warmup", "--config", str(config_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "已驻留内存" not in out
    assert "已预热" not in out
    assert "mock" in out


# ----------------------------------------------------------------------
# test
# ----------------------------------------------------------------------
def test_test_command_prints_items(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """test 逐项打印自检结果，mock 项打印 mock。"""
    code = cli_main(["test", "--config", str(config_path)])
    out = capsys.readouterr().out
    assert code in (EXIT_OK, EXIT_ERROR)
    assert "配置加载与校验" in out
    assert "依赖包导入" in out
    assert "音频设备枚举" in out
    assert "截图接口探测" in out
    assert "ASR 加载" in out
    assert "DeepSeek 探测" in out
    assert "Dify 探测" in out
    assert "mock" in out


def test_test_command_reports_missing_dependency(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """依赖包缺失时 test 退出码是 1，并带错误码。"""
    monkeypatch.setattr(app.cli, "check_dependencies", lambda packages: ["sounddevice"])
    assert cli_main(["test", "--config", str(config_path)]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "E_CONFIG" in out


# ----------------------------------------------------------------------
# devices 与 audio-test
# ----------------------------------------------------------------------
def test_devices_command(
    runtime_paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """devices 列出本机音频输入设备。"""
    code = cli_main(["devices"])
    out = capsys.readouterr().out
    assert code in (EXIT_OK, EXIT_ERROR)
    assert out.strip() != ""
    if code == EXIT_ERROR:
        assert "E_AUDIO_DEVICE" in out


def test_devices_command_marks_default_devices(
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """devices 只在默认设备的行末加「默认」标记，非默认设备不加。"""
    devices = [
        AudioDevice(
            name="默认麦克风",
            index=3,
            is_input=True,
            is_loopback=False,
            default_samplerate=48000.0,
            channels=2,
            is_default=True,
        ),
        AudioDevice(
            name="普通回环.monitor",
            index=9,
            is_input=True,
            is_loopback=True,
            default_samplerate=48000.0,
            channels=2,
        ),
    ]
    monkeypatch.setattr(app.cli, "list_input_devices", lambda: devices)

    assert cli_main(["devices"]) == EXIT_OK
    out = capsys.readouterr().out
    default_line = next(line for line in out.splitlines() if "[3]" in line)
    other_line = next(line for line in out.splitlines() if "[9]" in line)

    assert default_line.endswith("  默认")
    assert not other_line.endswith("  默认")
    # 表格原有的列语义不变
    assert "类型=系统回环" in other_line
    assert "默认采样率=48000" in default_line


def test_audio_test_reports_levels(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """audio-test 报告采样率、声道数与电平峰值，不真录音也不落盘。"""
    import numpy as np
    import sounddevice as sd

    def fake_rec(*args: Any, **kwargs: Any) -> Any:
        return np.zeros((16000, 1), dtype="float32")

    monkeypatch.setattr(sd, "rec", fake_rec)
    monkeypatch.setattr(sd, "wait", lambda *args, **kwargs: None)
    assert cli_main(["audio-test", "--config", str(config_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "采样率：16000 Hz" in out
    assert "声道数：1" in out
    assert "电平峰值：" in out
    assert "不落盘" in out


def test_audio_test_failure_reports_audio_error(
    config_path: Path,
    runtime_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """采集失败时退出码是 1，错误码是 E_AUDIO_DEVICE。"""
    import sounddevice as sd

    def broken_rec(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("设备被占用")

    monkeypatch.setattr(sd, "rec", broken_rec)
    assert cli_main(["audio-test", "--config", str(config_path)]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "E_AUDIO_DEVICE" in out


# ----------------------------------------------------------------------
# 进程装配
# ----------------------------------------------------------------------
def test_build_handlers_covers_all_actions() -> None:
    """handler 覆盖《2-骨架与配置》进程通信一节的全部 action。"""
    handlers = app.main.build_handlers(app.main.MainState())
    assert set(handlers) == set(ACTIONS)


def test_main_state_status_payload() -> None:
    """主进程状态字段的取值落在《2-骨架与配置》状态字段取值表里。"""
    state = app.main.MainState()
    payload = state.status_payload()
    assert payload["running"] == "running"
    assert payload["audio"] == "off"
    assert payload["asr"] == "idle"
    assert payload["dify"] == "offline"
    assert payload["deepseek"] == "idle"
    assert payload["ttft"] is None
    assert payload["pid"] == os.getpid()
    # 没有样本时 metrics 是空列表
    assert payload["metrics"] == []


def test_main_state_status_payload_carries_metrics() -> None:
    """状态 payload 的 metrics 段带指标名、样本数与百分位。"""
    default_recorder().record("capture_latency", 12.0, source="mock")
    default_recorder().record("capture_latency", 20.0, source="mock")

    payload = app.main.MainState().status_payload()

    assert payload["metrics"] == [
        {
            "name": "capture_latency",
            "count": 2,
            "p50_ms": 12.0,
            "p90_ms": 20.0,
            "p95_ms": 20.0,
            "max_ms": 20.0,
        }
    ]


def test_metrics_payload_only_lists_metrics_with_samples() -> None:
    """metrics 段只含已有样本的指标，顺序按《2-骨架与配置》性能指标一节。"""
    default_recorder().record("total_latency", 5.0, source="mock")
    default_recorder().record("capture_latency", 3.0, source="mock")

    items = app.main.metrics_payload()

    assert [item["name"] for item in items] == ["capture_latency", "total_latency"]
    assert items[0]["count"] == 1
    assert items[1]["count"] == 1


def test_main_state_toggle_voice() -> None:
    """语音开关在 on 与 off 之间来回切。"""
    state = app.main.MainState()
    assert state.toggle_voice()["audio"] == "on"
    assert state.toggle_voice()["audio"] == "off"


# ----------------------------------------------------------------------
# 截图装配与 socket 的 screenshot 动作
# ----------------------------------------------------------------------
class _FakeApp:
    """假界面，只接装配层用到的两个方法。"""

    def __init__(self) -> None:
        self.screenshot_manager: Any = None
        self.hints: list[str] = []

    async def publish_hint(self, hint: str) -> None:
        """记录提示位。"""
        self.hints.append(hint)

    def attach_screenshot_manager(self, screenshot_manager: Any) -> None:
        """记录挂上来的截图管理器。"""
        self.screenshot_manager = screenshot_manager


class _FakeQuestionManager:
    """假问题管理器，记录提交过的文字与图片。"""

    def __init__(self) -> None:
        self.submissions: list[tuple[str, str | None]] = []

    async def submit_text(self, text: str, image_data_url: str | None = None) -> None:
        """记录一次提交。"""
        self.submissions.append((text, image_data_url))


async def test_assemble_screenshot_manager_uses_mock_backend() -> None:
    """providers.screenshot=mock 时装的是假后端，图能走完一次带图提交。"""
    config = _mock_config()
    fake_app = _FakeApp()
    manager = _FakeQuestionManager()

    screenshot_manager = app.main.assemble_screenshot_manager(
        config, fake_app, manager
    )

    assert screenshot_manager.backend_name == "mock"
    assert fake_app.screenshot_manager is screenshot_manager
    assert await screenshot_manager.capture_and_submit() is True
    assert len(manager.submissions) == 1
    text, data_url = manager.submissions[0]
    assert text == ""
    assert data_url is not None
    assert data_url.startswith("data:image/png;base64,")


async def test_socket_screenshot_action_triggers_capture() -> None:
    """socket 的 screenshot 动作真的触发一次截图提交，返回 captured 布尔。"""
    config = _mock_config()
    state = app.main.MainState()
    handlers = app.main.build_handlers(state)
    fake_app = _FakeApp()
    manager = _FakeQuestionManager()
    screenshot_manager = app.main.assemble_screenshot_manager(
        config, fake_app, manager
    )
    state.set_screenshot_manager(screenshot_manager)

    payload = await handlers["screenshot"](Request(action="screenshot"))

    assert payload["captured"] is True
    assert payload["provider"] == "mock"
    assert len(manager.submissions) == 1


async def test_socket_screenshot_action_without_manager_only_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """界面还没起来时 screenshot 动作只做接口探测，不采集。"""

    async def _fake_probe(timeout: float = 3.0) -> app.main.ProbeResult:
        """假的接口探测，不碰 D-Bus。"""
        return app.main.ProbeResult(ok=True, detail="截图接口可用")

    monkeypatch.setattr(app.main, "probe_screenshot_interface", _fake_probe)
    state = app.main.MainState()
    handlers = app.main.build_handlers(state)

    payload = await handlers["screenshot"](Request(action="screenshot"))

    assert payload["captured"] is False
    assert payload["provider"] == "mock"


async def test_probe_model_capability_reports_model_for_mock() -> None:
    """providers.llm=mock 时能力探测说明用的是假流客户端。"""
    config = _mock_config()
    client = app.main.select_client(config)

    result = await app.main.probe_model_capability(client, config)

    assert result.ok is True
    assert "mock" in result.detail


async def test_probe_model_capability_reports_failure_without_exiting() -> None:
    """能力探测失败只给失败结果与错误码，不影响截图流程。"""
    config = app.config.load_config()
    config.providers.llm = "deepseek"

    result = await app.main.probe_model_capability(_FailingClient(), config)

    assert result.ok is False
    assert result.code == "E_DEEPSEEK_STREAM"
    assert not hasattr(result, "supports_image")


class _FailingClient:
    """能力探测一定失败的假客户端。"""

    async def check_health(self) -> Any:
        """探测失败。"""
        raise RuntimeError("连不上模型服务")


async def test_warmup_model_failure_sets_deepseek_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """handle_warmup 探测模型失败时，DEEPSEEK 状态落到 error。"""
    config = _mock_config()
    config.providers.llm = "deepseek"
    state = app.main.MainState()
    state.attach_config(config)
    state.set_model_client(_FailingClient())
    # 预热还会刷新设备列表，这里不碰真实音频后端
    monkeypatch.setattr(app.main, "list_input_devices", lambda: [])

    payload = await state.warmup_payload()
    # 状态回调在事件循环里排了一个发布任务，让它跑完
    await asyncio.sleep(0)

    assert payload["deepseek"] == "error"
    assert state.deepseek_state == "error"


# ----------------------------------------------------------------------
# 平台门面
# ----------------------------------------------------------------------
class _FakeAudioBackend:
    """假音频后端，记下被调了几次。"""

    name = "fake-audio"
    required_packages = ()

    def __init__(self, devices: list[AudioDevice]) -> None:
        self._devices = devices
        self.calls = 0

    def list_devices(self) -> list[AudioDevice]:
        self.calls += 1
        return self._devices


class _FakeScreenshotBackend:
    """假截图后端，记下探测时传进来的超时。"""

    name = "fake-screenshot"
    required_packages = ()

    def __init__(self, detail: str = "截图接口可用") -> None:
        self.detail = detail
        self.timeouts: list[float] = []

    async def probe(self, timeout: float = 3.0) -> str:
        self.timeouts.append(timeout)
        return self.detail


class _FakeSocket:
    """假 socket，只实现 send_request 用到的那几个方法。"""

    def __init__(self, response: bytes) -> None:
        self._response = response
        self.sent = b""
        self.closed = False

    def sendall(self, payload: bytes) -> None:
        self.sent += payload

    def recv(self, size: int) -> bytes:
        data, self._response = self._response[:size], self._response[size:]
        return data

    def close(self) -> None:
        self.closed = True


class _FakeTransport:
    """假传输层，记下连接时传进来的超时。"""

    def __init__(self, endpoint_path: Path) -> None:
        self.endpoint_path = endpoint_path
        self.connect_timeouts: list[float] = []
        self.socket: _FakeSocket | None = None

    def connect(self, *, timeout: float) -> _FakeSocket:
        self.connect_timeouts.append(timeout)
        self.socket = _FakeSocket(
            b'{"ok":true,"error":"","payload":{"ping":"pong"}}\n'
        )
        return self.socket

    def is_endpoint_in_use(self) -> bool:
        return False


class _FakePlatform:
    """假平台门面，只实现被测到的能力。"""

    name = "fake"

    def __init__(
        self,
        audio: _FakeAudioBackend | None = None,
        screenshot: _FakeScreenshotBackend | None = None,
    ) -> None:
        self.audio = audio if audio is not None else _FakeAudioBackend([])
        self.screenshot = (
            screenshot if screenshot is not None else _FakeScreenshotBackend()
        )
        self.transports: list[_FakeTransport] = []

    def transport(self, endpoint_path: Path) -> _FakeTransport:
        transport = _FakeTransport(endpoint_path)
        self.transports.append(transport)
        return transport


def test_probe_audio_devices_uses_platform_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """音频设备枚举走平台门面的音频后端。"""
    backend = _FakeAudioBackend(
        [
            AudioDevice(
                name="假麦克风",
                index=3,
                is_input=True,
                is_loopback=False,
                default_samplerate=48000.0,
                channels=1,
            )
        ]
    )
    platform = _FakePlatform(audio=backend)
    monkeypatch.setattr(app.main, "get_platform", lambda: platform)

    result = app.main.probe_audio_devices()

    assert result.ok is True
    assert "1 个" in result.detail
    assert backend.calls == 1


async def test_probe_screenshot_interface_uses_platform_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """截图接口探测走平台门面的截图后端。"""
    backend = _FakeScreenshotBackend("假截图接口可用")
    platform = _FakePlatform(screenshot=backend)
    monkeypatch.setattr(app.main, "get_platform", lambda: platform)

    result = await app.main.probe_screenshot_interface(2.5)

    assert result.ok is True
    assert result.detail == "假截图接口可用"
    assert backend.timeouts == [2.5]


async def test_probe_screenshot_interface_reports_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """平台截图后端还是占位时，探测结果不抛异常，错误码是 E_SCREENSHOT。"""

    class _NotImplementedScreenshot:
        name = "fake-screenshot"
        required_packages = ()

        async def probe(self, timeout: float = 3.0) -> str:
            raise NotImplementedError("Windows 侧截图后端还没实现")

    platform = _FakePlatform(screenshot=_NotImplementedScreenshot())  # type: ignore[arg-type]
    monkeypatch.setattr(app.main, "get_platform", lambda: platform)

    result = await app.main.probe_screenshot_interface()

    assert result.ok is False
    assert result.code == "E_SCREENSHOT"
    assert "还没实现" in result.detail


def test_send_request_uses_platform_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """客户端连接走平台门面的传输层。"""
    platform = _FakePlatform()
    monkeypatch.setattr(app.runtime.socket_client, "get_platform", lambda: platform)

    response = app.runtime.socket_client.send_request(
        "status", socket_path=tmp_path / "s.port"
    )

    assert response.ok is True
    assert response.payload == {"ping": "pong"}
    assert len(platform.transports) == 1
    assert platform.transports[0].connect_timeouts == [5.0]
    assert platform.transports[0].endpoint_path == tmp_path / "s.port"


def test_is_server_alive_uses_platform_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """主进程存活探测走平台门面的传输层。"""
    platform = _FakePlatform()
    monkeypatch.setattr(app.runtime.socket_client, "get_platform", lambda: platform)

    assert app.runtime.socket_client.is_server_alive(tmp_path / "s.port") is False
    assert len(platform.transports) == 1
    assert platform.transports[0].endpoint_path == tmp_path / "s.port"


def test_required_packages_unchanged_on_windows(config_path: Path) -> None:
    """Windows 上算出来的依赖包是音频与截图后端这一套（去重后）。"""
    config = app.main.load_app_config(str(config_path))

    packages = app.main.required_packages(config)

    # 音频后端要 sounddevice / soundcard（numpy 已在基础包里），截图后端要 mss
    assert packages == app.main.BASE_PACKAGES + ("sounddevice", "soundcard", "mss")


def test_required_packages_include_asr_and_image_packages(tmp_path: Path) -> None:
    """providers 指向真实实现时带上 FunASR 与 Pillow，顺序不变。"""
    path = _write_config(
        tmp_path,
        providers={
            "llm": "mock",
            "retrieval": "mock",
            "asr": "funasr",
            "screenshot": "mss",
        },
    )
    config = app.main.load_app_config(str(path))

    packages = app.main.required_packages(config)

    assert packages == (
        app.main.BASE_PACKAGES
        + ("sounddevice", "soundcard", "mss")
        + app.main.ASR_PACKAGES
        + ("PIL",)
    )


def test_screenshot_provider_mss_matches_windows_backend(tmp_path: Path) -> None:
    """providers.screenshot=mss 能通过配置校验，且与 Windows 平台后端一致。"""
    path = _write_config(
        tmp_path,
        providers={
            "llm": "mock",
            "retrieval": "mock",
            "asr": "mock",
            "screenshot": "mss",
        },
    )

    config = app.main.load_app_config(str(path))

    assert config.providers.screenshot == "mss"
    assert app.main.get_platform().screenshot.name == "mss"


def test_no_module_outside_platform_facade_checks_sys_platform() -> None:
    """除 app/platform/ 外，app/ 下不许出现 sys.platform。

    平台在进程启动时选定一次，判断只发生在
    :func:`app.platform.get_platform`，见《2-骨架与配置》目录结构约定一节。
    """
    app_dir = Path(app.__file__).resolve().parent
    platform_dir = app_dir / "platform"
    offenders: list[str] = []
    for path in sorted(app_dir.rglob("*.py")):
        if platform_dir in path.parents:
            continue
        if "sys.platform" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(app_dir)))
    assert offenders == []
