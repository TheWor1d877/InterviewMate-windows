"""命令行解析与分发。

子命令见《2-骨架与配置》命令行入口一节：``start``、``warmup``、``stop``、
``status``、``test``、``devices``、``audio-test``、``hotkey``。

退出码表见同一节：0 正常、1 运行错误、2 参数错误、3 配置缺失或校验失败。

输出分两路：命令结果与错误提示写 stdout，日志走 stderr（由
:func:`app.utils.logger.setup_logging` 装好），同一份内容同时落
``logs/interviewmate-{日期}.log``。日志级别与脱敏开关取配置文件。

本模块被 import 时不执行任何命令，也不退出进程。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from app.audio.asr_replay import load_wav_int16_mono, pcm_int16_to_float32
from app.audio.funasr_engine import DEFAULT_SAMPLE_RATE
from app.audio.loopback_test import (
    LoopbackCaptureRecorder,
    LoopbackTestResult,
    SoundDevicePlayer,
    run_loopback_test,
)
from app.config import REPO_ROOT, AppConfig, missing_required_env
from app.core.exceptions import (
    AsrLoadError,
    ConfigError,
    InterviewMateError,
    SocketError,
)
from app.main import (
    EXIT_ARG_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_ERROR,
    EXIT_OK,
    check_dependencies,
    list_input_devices,
    load_app_config,
    probe_audio_devices,
    probe_screenshot_interface,
    required_packages,
    resolve_config_path,
    select_client,
    select_final_refiner,
    select_asr_engine,
    start_process,
)
from app.platform import get_platform
from app.runtime.health_checker import HealthChecker, HealthStatus
from app.runtime.process_manager import PID_PATH, SOCKET_PATH, read_pid, remove_pid
from app.runtime.socket_client import is_server_alive, send_request
from app.utils.logger import log_event, setup_logging

__all__ = [
    "build_parser",
    "cmd_start",
    "cmd_warmup",
    "cmd_stop",
    "cmd_status",
    "cmd_test",
    "cmd_devices",
    "cmd_audio_test",
    "cmd_loopback_test",
    "cmd_hotkey",
    "main",
]


# 全局参数的默认值，见《2-骨架与配置》命令行入口一节
DEFAULT_CONFIG_ARG: str = "config/config.yaml"

# hotkey 子动作到 socket action 的映射，取值见《2-骨架与配置》进程通信一节。
# voice-start 与 voice-stop 是 AHK F8 按下/松开用的按住动作；
# voice-toggle 保留向后兼容，CLI 手动调用仍可用；
# project-mode-toggle、context-view-toggle、context-clear 对应
# AHK F9/F10/F11 的一按一触发动作。
_HOTKEY_ACTIONS: dict[str, str] = {
    "voice-toggle": "voice-toggle",
    "voice-start": "voice-start",
    "voice-stop": "voice-stop",
    "screenshot": "screenshot",
    "project-mode-toggle": "project-mode-toggle",
    "context-view-toggle": "context-view-toggle",
    "context-clear": "context-clear",
}

# 等主进程退出的超时秒数
_STOP_TIMEOUT: float = 10.0

# audio-test 采集时长，秒
_AUDIO_TEST_SECONDS: float = 1.0

# loopback-test 默认测试音频：scripts/windows/make-test-audio.cmd 合成的
# 「解释一下什么是GDS」，16000 Hz / 16 bit / 单声道，约 3 秒
DEFAULT_LOOPBACK_WAV: Path = REPO_ROOT / "tests" / "assets" / "asr-test-question.wav"

# 设备类型标签，devices 命令用来区分系统回环与麦克风，
# 见《5-语音输入与流式ASR》音频来源一节
DEVICE_KIND_LOOPBACK: str = "系统回环"
DEVICE_KIND_MICROPHONE: str = "麦克风"


class _ArgumentError(Exception):
    """命令行参数不合法。"""


class _ArgumentParser(argparse.ArgumentParser):
    """把 argparse 的参数错误变成返回值，不让 argparse 自己退出进程。"""

    def error(self, message: str) -> NoReturn:
        """抛 :class:`_ArgumentError`，由 :func:`main` 转成退出码 2。"""
        raise _ArgumentError(f"{self.prog}: {message}")


def _add_common_options(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """加全局参数 ``--config`` 与 ``--env-file``。

    ``--config`` 默认 ``config/config.yaml``；``--env-file`` 默认 None，
    表示用仓库根的 .env。

    ``suppress=True`` 时用 :data:`argparse.SUPPRESS` 做默认值，
    这样子命令后面再写一次 ``--config`` 不会把全局值覆盖掉。
    """
    config_default = argparse.SUPPRESS if suppress else DEFAULT_CONFIG_ARG
    env_default = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "--config",
        default=config_default,
        help=f"配置文件路径，默认 {DEFAULT_CONFIG_ARG}",
    )
    parser.add_argument(
        "--env-file",
        default=env_default,
        help="环境变量文件路径，默认用仓库根的 .env",
    )


def build_parser() -> argparse.ArgumentParser:
    """建命令行解析器。

    子命令见《2-骨架与配置》命令行入口一节。

    :returns: 解析器。
    """
    parser = _ArgumentParser(
        prog="interviewmate",
        description="InterviewMate 命令行入口",
    )
    _add_common_options(parser, suppress=False)
    # 子命令后面的同名参数也认，值相同
    common = argparse.ArgumentParser(add_help=False)
    _add_common_options(common, suppress=True)

    subparsers = parser.add_subparsers(dest="command", metavar="子命令", required=True)
    subparsers.add_parser("start", help="启动主进程", parents=[common])
    subparsers.add_parser("warmup", help="通知主进程预热", parents=[common])
    subparsers.add_parser("stop", help="停止主进程", parents=[common])
    subparsers.add_parser("status", help="打印运行状态", parents=[common])
    subparsers.add_parser("test", help="只读自检", parents=[common])
    subparsers.add_parser("devices", help="列出音频输入设备", parents=[common])
    subparsers.add_parser("audio-test", help="采集一小段音频并报告电平", parents=[common])
    asr_replay_parser = subparsers.add_parser(
        "asr-replay", help="把 wav 文件回放给 ASR 引擎并打印识别文本", parents=[common]
    )
    asr_replay_parser.add_argument("wav", help="wav 文件路径")
    asr_replay_parser.add_argument(
        "--block-ms", type=int, default=200, help="每次喂给引擎的毫秒数，默认 200"
    )
    loopback_parser = subparsers.add_parser(
        "loopback-test",
        help="外放测试音频并从系统回环采集，验证「播放→回环采集→ASR」链路",
        parents=[common],
    )
    loopback_parser.add_argument(
        "wav",
        nargs="?",
        default=str(DEFAULT_LOOPBACK_WAV),
        help=f"wav 文件路径，默认 tests/assets/asr-test-question.wav",
    )
    loopback_parser.add_argument(
        "--expected",
        default=None,
        help="期望文案，用于与识别文本对比（仅参考，不影响退出码）",
    )
    hotkey_parser = subparsers.add_parser(
        "hotkey", help="触发快捷键动作", parents=[common]
    )
    hotkey_subparsers = hotkey_parser.add_subparsers(
        dest="action", metavar="动作", required=True
    )
    for action in _HOTKEY_ACTIONS:
        hotkey_subparsers.add_parser(action, help=action, parents=[common])
    return parser


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _setup_logging_for(config: AppConfig) -> None:
    """按配置里的级别与脱敏开关重装日志管道。

    :param config: 应用配置。
    """
    setup_logging(
        config.app.log_level,
        redact=config.privacy.redact_secrets_in_logs,
    )


def _print_item(state: str, name: str, detail: str = "") -> bool:
    """打印一条自检结果，返回这一项是否算通过。

    ``state`` 取 ``ok``、``FAIL`` 或 ``mock``。
    """
    text = f"[{state}] {name}"
    if detail:
        text = f"{text}：{detail}"
    print(text, flush=True)
    return state == "ok"


def _cleanup_stale_files() -> list[str]:
    """删掉残留的 PID 文件与通信端点文件，返回删掉的文件路径。

    端点文件由平台传输层清：Windows 侧是端口文件与 token 文件，
    见《2-骨架与配置》进程通信一节。
    """
    removed: list[str] = []
    if PID_PATH.exists():
        remove_pid(PID_PATH)
        removed.append(str(PID_PATH))
    if SOCKET_PATH.exists():
        get_platform().transport(SOCKET_PATH).cleanup_endpoint()
        removed.append(str(SOCKET_PATH))
    return removed


def _wait_for_exit(timeout: float = _STOP_TIMEOUT, interval: float = 0.2) -> bool:
    """等主进程退出。

    socket 连不上就算退出。

    :param timeout: 最长等待秒数。
    :param interval: 轮询间隔秒数。
    :returns: 主进程在超时前退出返回 True。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_server_alive(SOCKET_PATH):
            return True
        time.sleep(interval)
    return not is_server_alive(SOCKET_PATH)


async def _probe_model(config: AppConfig) -> tuple[bool, str]:
    """真的探测一次模型服务，返回 (是否通过, 说明)。

    只在 ``providers.llm`` 指向真实实现时调用，会发一次网络请求。
    失败时说明里带错误码，取值见《2-骨架与配置》错误码表。

    :param config: 应用配置。
    """
    client = select_client(config)
    try:
        try:
            capability = await client.check_health()
        except InterviewMateError as exc:
            return False, f"{exc.message} [{exc.code}]"
        except Exception as exc:
            return False, f"{exc} [E_MODEL_STREAM]"
        supports_image = "是" if capability.supports_image else "否"
        return True, f"模型 {capability.model_id}，支持图片：{supports_image}"
    finally:
        try:
            await client.aclose()
        except Exception as exc:
            log_event(
                "test.warn",
                f"关闭模型客户端失败：{exc}",
                level="WARNING",
            )


# ----------------------------------------------------------------------
# 子命令
# ----------------------------------------------------------------------
def cmd_start(args: argparse.Namespace) -> int:
    """启动主进程。

    步骤顺序见《2-骨架与配置》命令行入口一节，实现在
    :func:`app.main.start_process`：读配置校验 → 查依赖包 → 查音频设备（只告警）→
    查截图接口（只告警）→ 建事件循环 → 建 socket 并监听 → 启动界面 → 退出时收尾。

    :param args: 解析后的参数。
    :returns: 退出码。
    """
    return asyncio.run(start_process(args.config, args.env_file))


def cmd_warmup(args: argparse.Namespace) -> int:
    """通知主进程预热。

    主进程在运行时发 ``warmup`` 动作；不在运行时只做环境检查与可用性检查，
    输出里不声称模型已经驻留内存。

    :param args: 解析后的参数。
    :returns: 退出码。
    """
    if is_server_alive(SOCKET_PATH):
        try:
            response = send_request("warmup", socket_path=SOCKET_PATH)
        except SocketError as exc:
            print(f"通知主进程 warmup 失败：{exc}", flush=True)
            log_event("warmup.error", "通知主进程 warmup 失败", level="ERROR", error=str(exc))
            return EXIT_ERROR
        if not response.ok:
            print(f"主进程返回错误：{response.error}", flush=True)
            log_event("warmup.error", "主进程拒绝 warmup", level="ERROR", code=response.error)
            return EXIT_ERROR
        payload = dict(response.payload)
        print("主进程已处理 warmup 请求：", flush=True)
        for key in ("asr", "model", "dify"):
            print(f"  {key}: {payload.get(key, '-')}", flush=True)
        if payload.get("note"):
            print(f"  说明：{payload['note']}", flush=True)
        log_event("warmup.done", "主进程已处理 warmup 请求")
        return EXIT_OK

    # 主进程没运行：只做环境检查与可用性检查
    try:
        config = load_app_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"E_CONFIG：{exc.message}", flush=True)
        log_event("config.error", "配置读取失败", level="ERROR", error=exc.message)
        return EXIT_CONFIG_ERROR
    _setup_logging_for(config)

    print("主进程未运行，只做环境检查与可用性检查", flush=True)
    missing = check_dependencies(required_packages(config))
    if missing:
        print(f"  依赖包：缺少 {'、'.join(missing)}", flush=True)
    else:
        print("  依赖包：都能导入", flush=True)
    audio = probe_audio_devices()
    print(f"  音频设备：{audio.detail}", flush=True)
    screenshot = asyncio.run(probe_screenshot_interface())
    print(f"  截图接口：{screenshot.detail}", flush=True)
    print(f"  ASR：mock（providers.asr={config.providers.asr}）", flush=True)
    print(f"  模型：mock（providers.llm={config.providers.llm}）", flush=True)
    print(f"  Dify：mock（providers.retrieval={config.providers.retrieval}）", flush=True)
    log_event("warmup.done", "主进程未运行，只做环境检查")
    return EXIT_OK


def cmd_stop(args: argparse.Namespace) -> int:
    """停止主进程。

    主进程在运行时发 ``stop`` 动作并等它退出；不在运行时清理残留的 PID 文件与
    通信端点文件。两种情况退出码都是 0。

    :param args: 解析后的参数。
    :returns: 退出码。
    """
    if is_server_alive(SOCKET_PATH):
        try:
            response = send_request("stop", socket_path=SOCKET_PATH)
        except SocketError as exc:
            print(f"通知主进程停止失败：{exc}", flush=True)
            log_event("stop.error", "通知主进程停止失败", level="ERROR", error=str(exc))
            return EXIT_ERROR
        if not response.ok:
            print(f"主进程拒绝停止：{response.error}", flush=True)
            log_event("stop.error", "主进程拒绝停止", level="ERROR", code=response.error)
            return EXIT_ERROR
        if not _wait_for_exit():
            print("主进程没有在超时时间内退出", flush=True)
            log_event("stop.error", "等待主进程退出超时", level="ERROR")
            return EXIT_ERROR
        print("主进程已停止", flush=True)
        log_event("stop.done", "主进程已停止")
        return EXIT_OK

    removed = _cleanup_stale_files()
    message = "主进程未运行，已清理残留文件"
    if removed:
        message = f"{message}：{'、'.join(removed)}"
    print(message, flush=True)
    log_event("stop.done", "主进程未运行，已清理残留文件")
    return EXIT_OK


def _format_ms(value: Any) -> str:
    """把毫秒值渲染成一位小数的毫秒数，取不到时写 ``-``。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.1f} ms"


def _print_metrics(metrics: Any) -> None:
    """打印指标统计段。

    payload 里没有指标段、指标段是空的或者没有可用条目时什么都不打印，
    不输出空标题。指标名、样本数、P50、P95 与最大值都带上，单位是毫秒。

    :param metrics: 状态 payload 里的 ``metrics`` 段。
    """
    if not isinstance(metrics, list):
        return
    items = [item for item in metrics if isinstance(item, dict)]
    if not items:
        return
    print("指标统计：", flush=True)
    for item in items:
        print(
            f"  {item.get('name', '-')}：样本 {item.get('count', 0)}，"
            f"P50 {_format_ms(item.get('p50_ms'))}，"
            f"P95 {_format_ms(item.get('p95_ms'))}，"
            f"最大 {_format_ms(item.get('max_ms'))}",
            flush=True,
        )


def cmd_status(args: argparse.Namespace) -> int:
    """打印运行状态。

    主进程在运行时通过 socket 问它要状态，格式见《2-骨架与配置》命令行入口一节；
    不在运行时读 PID 文件判断，输出里明确说明进程没在运行。
    有指标样本时在状态文本后面追加一段指标统计。两种情况退出码都是 0。

    :param args: 解析后的参数。
    :returns: 退出码。
    """
    app_name = "InterviewMate"
    try:
        config = load_app_config(args.config, args.env_file)
    except ConfigError as exc:
        # 状态命令不靠配置活着，配置读不到就用默认应用名，并把原因告诉用户
        print(f"配置读取失败，状态里用默认应用名：{exc.message}", flush=True)
        log_event(
            "config.warn",
            "状态命令读不到配置，用默认应用名",
            level="WARNING",
            error=exc.message,
        )
    else:
        app_name = config.app.name

    if is_server_alive(SOCKET_PATH):
        try:
            response = send_request("status", socket_path=SOCKET_PATH)
        except SocketError as exc:
            print(f"问主进程要状态失败：{exc}", flush=True)
            log_event("status.error", "问主进程要状态失败", level="ERROR", error=str(exc))
            return EXIT_ERROR
        if not response.ok:
            print(f"主进程返回错误：{response.error}", flush=True)
            log_event("status.error", "主进程返回错误", level="ERROR", code=response.error)
            return EXIT_ERROR
        payload = dict(response.payload)
        ttft = payload.get("ttft")
        pid = payload.get("pid")
        # 首行用大写形式，见《2-骨架与配置》命令行入口一节的状态文本示例
        running = str(payload.get("running", "running")).upper()
        status = HealthStatus(
            running=running,
            audio=str(payload.get("audio", "off")),
            asr=str(payload.get("asr", "idle")),
            dify=str(payload.get("dify", "offline")),
            model=str(payload.get("model", "idle")),
            ttft_ms=float(ttft) if isinstance(ttft, (int, float)) else None,
        )
        text = HealthChecker().format_status(
            status,
            app_name=app_name,
            pid=pid if isinstance(pid, int) else None,
        )
        print(text, flush=True)
        _print_metrics(payload.get("metrics"))
        log_event("status.done", "打印主进程状态", running=status.running)
        return EXIT_OK

    pid = read_pid(PID_PATH)
    if pid is None:
        reason = "没有 PID 文件，通信端点也没有进程在监听"
    else:
        reason = f"PID 文件里的进程 {pid} 已经不在，通信端点也没有进程在监听"
    print(f"主进程未运行：{reason}", flush=True)
    text = HealthChecker().format_status(
        HealthStatus(running="STOPPED"),
        app_name=app_name,
        pid=pid,
    )
    print(text, flush=True)
    log_event("status.done", "主进程未运行")
    return EXIT_OK


def cmd_test(args: argparse.Namespace) -> int:
    """只读自检。

    检查配置能否加载并通过校验、依赖包能否导入、音频设备能否枚举、截图接口能否探测、
    ASR 能否加载，再按 ``providers`` 决定要不要探测模型服务与 Dify。
    ``providers`` 指向 mock 的能力不连真实服务，对应项打印 ``mock``；
    ``providers.llm=openai`` 时真的探测一次模型服务，失败项带错误码。
    全部通过退出码 0，有失败项退出码 1。

    :param args: 解析后的参数。
    :returns: 退出码。
    """
    try:
        config = load_app_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"E_CONFIG：{exc.message}", flush=True)
        log_event("config.error", "配置读取失败", level="ERROR", error=exc.message)
        return EXIT_CONFIG_ERROR
    _setup_logging_for(config)

    failed = False
    _print_item("ok", "配置加载与校验", str(resolve_config_path(args.config)))

    missing = check_dependencies(required_packages(config))
    if missing:
        failed |= not _print_item(
            "FAIL", "依赖包导入", f"缺少 {'、'.join(missing)} [E_CONFIG]"
        )
    else:
        _print_item("ok", "依赖包导入", f"{len(required_packages(config))} 个包都能导入")

    audio = probe_audio_devices()
    if audio.ok:
        _print_item("ok", "音频设备枚举", audio.detail)
    else:
        failed |= not _print_item(
            "FAIL", "音频设备枚举", f"{audio.detail} [{audio.code}]"
        )

    screenshot = asyncio.run(probe_screenshot_interface())
    if screenshot.ok:
        _print_item("ok", "截图接口探测", screenshot.detail)
    else:
        failed |= not _print_item(
            "FAIL", "截图接口探测", f"{screenshot.detail} [{screenshot.code}]"
        )

    if config.providers.asr == "mock":
        _print_item("mock", "ASR 加载", "providers.asr=mock，不连真实服务")
    else:
        failed |= not _print_item(
            "FAIL", "ASR 加载", "真实 ASR provider 尚未接入 [E_CONFIG]"
        )

    if config.providers.llm == "mock":
        _print_item("mock", "模型探测", "providers.llm=mock，不连真实服务")
    else:
        ok, detail = asyncio.run(_probe_model(config))
        if ok:
            _print_item("ok", "模型探测", detail)
        else:
            failed |= not _print_item("FAIL", "模型探测", detail)

    if config.providers.retrieval == "mock":
        _print_item("mock", "Dify 探测", "providers.retrieval=mock，不连真实服务")
    else:
        failed |= not _print_item(
            "FAIL", "Dify 探测", "真实 Dify provider 尚未接入 [E_CONFIG]"
        )

    # 必填环境变量只在 providers.llm 指向真实实现时提示：
    # mock 模式不提示缺 MODEL_API_KEY，也不影响退出码
    if config.providers.llm == "openai":
        missing_env = missing_required_env(args.env_file)
        if missing_env:
            _print_item(
                "信息",
                "必填环境变量",
                f"缺少 {'、'.join(missing_env)}（start 会以退出码 3 结束）",
            )
        else:
            _print_item("信息", "必填环境变量", "齐全")

    log_event("test.done", "自检结束", failed=failed)
    return EXIT_ERROR if failed else EXIT_OK


def cmd_devices(args: argparse.Namespace) -> int:
    """列出本机音频输入设备。

    设备表由当前平台的音频后端给出，方括号里是设备在本机枚举结果里的真实索引，
    见《5-语音输入与流式ASR》音频来源一节。每台设备后面标出类型：
    ``系统回环`` 是 Windows 侧的 WASAPI loopback，
    ``麦克风`` 是普通的输入设备，见《5-语音输入与流式ASR》音频来源一节。
    系统当前默认的设备在行末标 ``默认``，切换默认设备后重新列一次就能确认。
    音频依赖不可导入或枚举失败时退出码 1，错误码 ``E_AUDIO_DEVICE``。

    :param args: 解析后的参数。
    :returns: 退出码。
    """
    try:
        devices = list_input_devices()
    except InterviewMateError as exc:
        print(f"{exc.code}：{exc.message}", flush=True)
        log_event("audio.error", exc.message, level="ERROR", code=exc.code)
        return EXIT_ERROR
    except NotImplementedError as exc:
        # 平台的音频后端还没实现（Windows 侧本轮是占位）
        print(f"E_AUDIO_DEVICE：{exc}", flush=True)
        log_event("audio.error", str(exc), level="ERROR", code="E_AUDIO_DEVICE")
        return EXIT_ERROR
    if not devices:
        print("本机没有音频输入设备", flush=True)
        log_event("audio.done", "本机没有音频输入设备")
        return EXIT_OK
    loopbacks = [device for device in devices if device.is_loopback]
    microphones = [device for device in devices if not device.is_loopback]
    print(
        f"音频输入设备（{len(devices)} 个："
        f"麦克风 {len(microphones)} 个，系统回环 {len(loopbacks)} 个）：",
        flush=True,
    )
    for device in devices:
        kind = DEVICE_KIND_LOOPBACK if device.is_loopback else DEVICE_KIND_MICROPHONE
        # 系统当前默认的设备在行末标出来，换成默认设备后肉眼就能确认
        default_mark = "  默认" if device.is_default else ""
        print(
            f"  [{device.index}] {device.name}"
            f"  类型={kind}"
            f"  输入声道={device.channels}"
            f"  默认采样率={int(device.default_samplerate)}"
            f"{default_mark}",
            flush=True,
        )
    log_event(
        "audio.done",
        "列出音频输入设备",
        count=len(devices),
        microphones=len(microphones),
        loopbacks=len(loopbacks),
    )
    return EXIT_OK


def _missing_audio_packages() -> list[str]:
    """当前平台的音频后端要的第三方包里，导入失败的。

    :returns: 导入失败的包名列表。
    """
    missing: list[str] = []
    for name in get_platform().audio.required_packages:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def cmd_audio_test(args: argparse.Namespace) -> int:
    """采集一小段音频，报告采样率、声道数与电平。

    采集由当前平台的音频后端完成。``privacy.persist_audio`` 为 false 时不写音频文件。
    失败退出码 1，错误码 ``E_AUDIO_DEVICE``。

    :param args: 解析后的参数。
    :returns: 退出码。
    """
    try:
        config = load_app_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"E_CONFIG：{exc.message}", flush=True)
        log_event("config.error", "配置读取失败", level="ERROR", error=exc.message)
        return EXIT_CONFIG_ERROR
    _setup_logging_for(config)

    try:
        import numpy as np
    except Exception as exc:
        print(f"E_AUDIO_DEVICE：音频依赖不可导入：{exc}", flush=True)
        log_event("audio.error", "音频依赖不可导入", level="ERROR", code="E_AUDIO_DEVICE")
        return EXIT_ERROR

    missing = _missing_audio_packages()
    if missing:
        names = "、".join(missing)
        print(f"E_AUDIO_DEVICE：音频依赖不可导入：{names}", flush=True)
        log_event("audio.error", "音频依赖不可导入", level="ERROR", code="E_AUDIO_DEVICE")
        return EXIT_ERROR

    sample_rate = config.audio.sample_rate
    channels = config.audio.channels
    try:
        recording = get_platform().audio.record(
            sample_rate=sample_rate,
            channels=channels,
            seconds=_AUDIO_TEST_SECONDS,
        )
    except Exception as exc:
        print(f"E_AUDIO_DEVICE：采集失败：{exc}", flush=True)
        log_event("audio.error", "采集失败", level="ERROR", code="E_AUDIO_DEVICE")
        return EXIT_ERROR

    samples = np.asarray(recording, dtype="float32")
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0

    print(f"采样率：{sample_rate} Hz", flush=True)
    print(f"声道数：{channels}", flush=True)
    print(f"采集时长：{_AUDIO_TEST_SECONDS:.1f} 秒", flush=True)
    print(f"电平峰值：{peak:.4f}", flush=True)
    print(f"RMS：{rms:.4f}", flush=True)

    if config.privacy.persist_audio:
        path = _write_wav(samples, sample_rate)
        print(f"已保存音频：{path}", flush=True)
    else:
        print("按配置不落盘（privacy.persist_audio=false）", flush=True)
    log_event("audio.done", "音频采集完成", peak=peak, rms=rms)
    return EXIT_OK


def _write_wav(samples: Any, sample_rate: int) -> Path:
    """把采集到的波形写成 WAV 文件。

    :param samples: 波形数组。
    :param sample_rate: 采样率。
    :returns: 文件路径。
    """
    import wave

    import numpy as np

    target_dir = REPO_ROOT / "runtime"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"audio-test-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    channel_count = samples.shape[1] if samples.ndim > 1 else 1
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(channel_count))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return path


def cmd_asr_replay(args: argparse.Namespace) -> int:
    """把一个 wav 文件交给 SenseVoice 重识别，打印最终识别文本。

    与产品链路一致：整段音频重识别（use_itn=True）加拼音级热词纠正，
    产出最终文本，见《5-语音输入与流式ASR》「重识别与热词纠正」一节。
    模型加载约 10 秒。用于区分「识别不准」是模型问题还是音频链路问题：
    同一段音频回放结果稳定而麦克风识别差，问题就在采集链路。
    ``--block-ms`` 对整段识别无效，保留参数只为命令行兼容。

    :param args: 解析后的参数。
    :returns: 退出码。
    """

    try:
        config = load_app_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"E_CONFIG：{exc.message}", flush=True)
        log_event("config.error", "配置读取失败", level="ERROR", error=exc.message)
        return EXIT_CONFIG_ERROR
    _setup_logging_for(config)

    if config.providers.asr != "funasr":
        print("E_ASR_LOAD：asr-replay 需要 providers.asr=funasr", flush=True)
        return EXIT_ERROR

    wav_path = Path(args.wav)
    if not wav_path.is_file():
        print(f"E_ASR_LOAD：wav 文件不存在：{wav_path}", flush=True)
        log_event("asr.replay.failed", "wav 文件不存在", level="ERROR", path=str(wav_path))
        return EXIT_ERROR

    async def _run() -> str:
        refiner = select_final_refiner(config)
        if refiner is None:
            raise AsrLoadError("providers.asr=funasr 时重识别引擎必须可用")
        await refiner.ensure_loaded()
        # 读 wav、下混、重采样到 16000 Hz，归一化成 float32 后整段识别
        pcm = load_wav_int16_mono(wav_path, target_rate=DEFAULT_SAMPLE_RATE)
        return await refiner.recognize(pcm_int16_to_float32(pcm))

    try:
        text = asyncio.run(_run())
    except InterviewMateError as exc:
        print(f"{exc.code}：{exc.message}", flush=True)
        log_event("asr.replay.failed", "回放识别失败", level="ERROR", code=exc.code)
        return EXIT_ERROR

    print(f"识别文本：{text}", flush=True)
    log_event("asr.replay.done", "回放识别完成", text=text)
    return EXIT_OK


def cmd_loopback_test(args: argparse.Namespace) -> int:
    """回环链路一键自检：外放合成音频 + 回环采集 + ASR 识别 + 打印结论。

    经系统默认输出设备外放一段 wav（默认测试音频约 3 秒），同时从默认
    扬声器的 WASAPI loopback 端点采集系统声音，采集到的波形喂给 FunASR
    引擎识别，再打印电平（峰值/RMS）、识别文本与结论。识别出非空文本，
    或至少录到明显电平，链路算通，退出码 0；设备、依赖或引擎失败退出码
    1 并带错误码；全程静音也按验证未通过退出 1。

    与 ``asr-replay`` 互补：replay 绕过声卡，本命令走的是真实回环采集链路。
    模型加载约 10 秒，播放会外放约 wav 时长的声音。

    :param args: 解析后的参数。
    :returns: 退出码。
    """
    try:
        config = load_app_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"E_CONFIG：{exc.message}", flush=True)
        log_event("config.error", "配置读取失败", level="ERROR", error=exc.message)
        return EXIT_CONFIG_ERROR
    _setup_logging_for(config)

    if config.providers.asr != "funasr":
        print("E_ASR_LOAD：loopback-test 需要 providers.asr=funasr", flush=True)
        log_event(
            "audio.loopback_test.failed",
            "providers.asr 不是 funasr",
            level="ERROR",
            code="E_ASR_LOAD",
        )
        return EXIT_ERROR

    wav_path = Path(args.wav)
    if not wav_path.is_file():
        print(f"E_ASR_LOAD：wav 文件不存在：{wav_path}", flush=True)
        log_event(
            "audio.loopback_test.failed",
            "wav 文件不存在",
            level="ERROR",
            path=str(wav_path),
            code="E_ASR_LOAD",
        )
        return EXIT_ERROR

    try:
        devices = list_input_devices()
    except InterviewMateError as exc:
        print(f"E_AUDIO_DEVICE：枚举音频设备失败：{exc.message}", flush=True)
        log_event(
            "audio.loopback_test.failed",
            "枚举音频设备失败",
            level="ERROR",
            code="E_AUDIO_DEVICE",
            error=exc.message,
        )
        return EXIT_ERROR
    except NotImplementedError as exc:
        print(f"E_AUDIO_DEVICE：{exc}", flush=True)
        log_event(
            "audio.loopback_test.failed",
            "平台的音频后端还没实现",
            level="ERROR",
            code="E_AUDIO_DEVICE",
        )
        return EXIT_ERROR

    loopbacks = [device for device in devices if device.is_loopback]
    if not loopbacks:
        print("E_AUDIO_DEVICE：本机没有可用的系统回环设备", flush=True)
        log_event(
            "audio.loopback_test.failed",
            "本机没有系统回环设备",
            level="ERROR",
            code="E_AUDIO_DEVICE",
        )
        return EXIT_ERROR
    # 优先默认扬声器的回环端点：外放走默认输出设备，回环也得是它的
    target = next((device for device in loopbacks if device.is_default), loopbacks[0])
    print(
        f"回环设备：[{target.index}] {target.name}{'（默认）' if target.is_default else ''}",
        flush=True,
    )
    print(f"测试音频：{wav_path}", flush=True)
    print("模型加载约 10 秒；播放会外放约 wav 时长的声音……", flush=True)

    async def _run() -> LoopbackTestResult:
        """装配真实引擎、播放器与回环采集器，跑一次回环自检。"""
        engine = select_asr_engine(config)
        backend = get_platform().audio
        recorder = LoopbackCaptureRecorder(
            backend,
            device=target.index,
            sample_rate=config.audio.sample_rate,
        )
        player = SoundDevicePlayer()
        return await run_loopback_test(
            wav_path, engine, player, recorder, expected=args.expected
        )

    try:
        result = asyncio.run(_run())
    except InterviewMateError as exc:
        print(f"{exc.code}：{exc.message}", flush=True)
        log_event(
            "audio.loopback_test.failed",
            "回环链路自检失败",
            level="ERROR",
            code=exc.code,
            error=exc.message,
        )
        return EXIT_ERROR
    except ValueError as exc:
        print(f"E_ASR_LOAD：wav 无法解析：{exc}", flush=True)
        log_event(
            "audio.loopback_test.failed",
            "wav 无法解析",
            level="ERROR",
            error=str(exc),
            code="E_ASR_LOAD",
        )
        return EXIT_ERROR

    print(f"播放时长：{result.played_seconds:.1f} 秒", flush=True)
    print(f"采集时长：{result.captured_seconds:.1f} 秒", flush=True)
    print(f"电平峰值：{result.peak:.4f}", flush=True)
    print(f"RMS：{result.rms:.4f}", flush=True)
    print(f"识别文本：{result.text if result.text else '（空）'}", flush=True)
    if result.expected is not None:
        match_text = (
            "一致" if result.expected_match else "不一致（仅参考，不影响结论）"
        )
        print(f"期望文案：{result.expected}（自动对比：{match_text}）", flush=True)
    else:
        print("期望文案：未提供（可用 --expected 指定，用于人工对比）", flush=True)
    print(f"结论：{result.conclusion}", flush=True)
    return EXIT_OK if result.ok else EXIT_ERROR


def cmd_hotkey(args: argparse.Namespace) -> int:
    """触发一个快捷键动作。

    主进程不在运行时输出 ``ok: false`` 与错误码 ``E_SOCKET``，退出码 1。

    :param args: 解析后的参数。
    :returns: 退出码。
    """
    action = _HOTKEY_ACTIONS[args.action]
    if not is_server_alive(SOCKET_PATH):
        result = {"ok": False, "error": "E_SOCKET", "action": action, "payload": {}}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        log_event(
            "hotkey.error",
            "主进程未运行，快捷键没有发出去",
            level="ERROR",
            action=action,
            code="E_SOCKET",
        )
        return EXIT_ERROR

    try:
        response = send_request(action, socket_path=SOCKET_PATH)
    except SocketError as exc:
        result = {
            "ok": False,
            "error": "E_SOCKET",
            "action": action,
            "payload": {},
            "detail": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        log_event("hotkey.error", "快捷键通信失败", level="ERROR", action=action)
        return EXIT_ERROR

    result = {
        "ok": response.ok,
        "error": response.error,
        "action": action,
        "payload": dict(response.payload),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    log_event("hotkey.done", "快捷键动作已发出", action=action, ok=response.ok)
    return EXIT_OK if response.ok else EXIT_ERROR


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
_COMMANDS = {
    "start": cmd_start,
    "warmup": cmd_warmup,
    "stop": cmd_stop,
    "status": cmd_status,
    "test": cmd_test,
    "devices": cmd_devices,
    "audio-test": cmd_audio_test,
    "asr-replay": cmd_asr_replay,
    "loopback-test": cmd_loopback_test,
    "hotkey": cmd_hotkey,
}


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。

    :param argv: 参数列表，None 时取 ``sys.argv[1:]``。
    :returns: 退出码，见《2-骨架与配置》退出码表。
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except _ArgumentError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return EXIT_ARG_ERROR
    except SystemExit as exc:
        # --help 之类 argparse 自己处理完就退出，这里只把退出码带回来
        return EXIT_OK if exc.code is None else int(exc.code)

    setup_logging()
    handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"E_CONFIG：{exc.message}", flush=True)
        log_event("config.error", exc.message, level="ERROR", code=exc.code)
        return EXIT_CONFIG_ERROR
    except InterviewMateError as exc:
        print(f"{exc.code}：{exc.message}", flush=True)
        log_event("cli.error", exc.message, level="ERROR", code=exc.code)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
