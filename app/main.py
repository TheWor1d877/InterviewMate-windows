"""主进程入口与模块装配。

《2-骨架与配置》命令行入口一节的 ``start`` 步骤里，属于进程的部分都在这里：
读配置校验、查依赖包、查本机能力、建 socket 服务端、启动 Textual 界面、退出时收尾。
命令行解析与分发在 :mod:`app.cli`。

三块内容：

- 退出码表：见《2-骨架与配置》命令行入口一节，:mod:`app.cli` 复用这里的常量；
- 本机能力检查：音频设备（:func:`list_input_devices`）与平台截图接口
  （:func:`probe_screenshot_interface`），``start`` 的第 3、4 步与 ``test`` 自检都用；
- 进程装配：:func:`run_main` 写 PID 文件、建通信服务端、启动界面、退出时收尾。

M1 阶段没有真实 provider 实现（DeepSeek 客户端、Dify、FunASR、截图分别在
M2/M4/M8 落地），因此这里不造假的 provider 类：状态字段按《2-骨架与配置》状态字段
取值表的缺省值给，截图只做接口能力检查，不落盘存图。

M2 起模型客户端真的接上了：:func:`select_client` 按 ``providers.llm`` 选客户端，
:func:`assemble_manager` 把事件分发器、流式解析器与问题管理器接到界面上，
``start`` 的第 1 步与第 7 步也跟着 ``providers.llm`` 走，见《2-骨架与配置》
命令行入口一节。

M4/M5 起检索与语音两条链路也真的接上了：:func:`assemble_retrieval` 按
``providers.retrieval`` 选检索实现并造检索回调，:func:`assemble_voice` 造语音
采集与 ASR 引擎，socket 的 ``voice-toggle`` 动作走真采集。
检索失败不让问题处理崩掉，语音停止后最终文本自动提交，
见《6-Dify检索与项目知识》检索流程一节与《5-语音输入与流式ASR》F8 交互一节。

界面按 F8 与 socket 的 ``voice-toggle`` 走同一条语音链路，界面按 F3 才弹
检索来源浮层、检索完成时来源只静默存下，代码题判定出来时通知分发器切形态，
启动时按配置预加载 ASR，见《2-骨架与配置》命令行入口一节第 9 步。

M7 起三种输入汇到一条路：:func:`build_input_router` 造输入路由器，
手工输入回车、语音最终文本、截图都交给它，提交回调指向问题管理器，
见《8-集成与验收》三种输入的汇合一节。socket 的五个动作由
:class:`~app.inputs.global_command.GlobalCommandDispatcher` 统一分发，
:func:`build_handlers` 只是它的薄封装。DEEPSEEK 字段由问题管理器的状态回调
更新，不再写死，见《2-骨架与配置》状态字段取值表。

平台相关的能力（截图、音频、进程通信、进程存活探测、编译器）都走
:mod:`app.platform` 门面，:func:`app.platform.get_platform` 是全仓唯一选择平台的
地方，本模块不自己判断平台。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import inspect
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.audio.audio_devices import resolve_source
from app.audio.audio_manager import (
    AUDIO_STATE_OFF,
    AUDIO_STATE_ON,
    RESULT_EMPTY,
    RESULT_STARTED,
    RESULT_STOPPED,
    AsrEngine,
    AudioManager,
    Capture,
    CaptureFactory,
)
from app.audio.audio_source import create_capture
from app.audio.funasr_engine import (
    STATE_ERROR,
    STATE_LOADING,
    STATE_READY,
    FunASREngine,
)
from app.config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    AppConfig,
    load_config,
    missing_required_env,
)
from app.core.event_bus import EventBus
from app.core.exceptions import (
    AudioDeviceError,
    ConfigError,
    InterviewMateError,
    ScreenshotError,
    SocketError,
)
from app.core.question_manager import (
    MODEL_STATE_CONNECTING,
    MODEL_STATE_ERROR,
    MODEL_STATE_IDLE,
    MODEL_STATE_STREAMING,
    QuestionManager,
    RetrievalFunc,
    SourcesCallback,
)
from app.inputs.global_command import (
    ACTION_SCREENSHOT,
    ACTION_STATUS,
    ACTION_STOP,
    ACTION_VOICE_TOGGLE,
    ACTION_WARMUP,
    GlobalCommandDispatcher,
)
from app.inputs.global_hotkeys import (
    GlobalHotkeyProvider,
    HotkeyHandlers,
    HotkeyListener,
)
from app.inputs.hotkey_controller import RESULT_IGNORED, HotkeyController
from app.inputs.input_router import InputRouter
from app.llm.deepseek_client import DeepSeekClient
from app.llm.mock_client import MockClient
from app.llm.prompt_manager import PromptManager
from app.llm.question_classifier_client import DeepSeekQuestionClassifier
from app.llm.stream_parser import StreamParser
from app.output.event_dispatcher import EventDispatcher
from app.platform import get_platform
from app.platform.types import AudioDevice, ScreenshotBackend
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.dify_retriever import DifyRetriever
from app.retrieval.mock_retriever import MockRetriever
from app.retrieval.retrieval_models import RetrievalResult
from app.runtime.process_manager import (
    PID_PATH,
    SOCKET_PATH,
    find_app_processes,
    is_app_process,
    is_process_alive,
    is_process_running,
    read_pid,
    remove_pid,
    terminate_processes,
    write_pid,
)
from app.runtime.socket_client import is_server_alive, send_request
from app.runtime.socket_protocol import Request
from app.runtime.socket_server import SocketServer
from app.utils.logger import (
    LOG_FILE_TEMPLATE,
    current_log_path,
    log_event,
    redirect_console,
    setup_logging,
)
from app.utils.timing import METRIC_NAMES, default_recorder
from app.vision.image_processor import ImageProcessor
from app.vision.mock_screenshot import MockScreenshotBackend
from app.vision.screenshot_manager import ScreenshotManager

__all__ = [
    "EXIT_OK",
    "EXIT_ERROR",
    "EXIT_ARG_ERROR",
    "EXIT_CONFIG_ERROR",
    "BASE_PACKAGES",
    "ASR_PACKAGES",
    "VOICE_TOGGLE_DEBOUNCE_SECONDS",
    "TUI_LOG_FILENAME",
    "ProbeResult",
    "MainState",
    "RetrievalChain",
    "VoiceChain",
    "resolve_config_path",
    "load_app_config",
    "required_packages",
    "check_dependencies",
    "list_input_devices",
    "probe_audio_devices",
    "probe_screenshot_interface",
    "select_client",
    "select_retriever",
    "select_asr_engine",
    "preload_asr",
    "probe_model_capability",
    "assemble_manager",
    "assemble_retrieval",
    "assemble_voice",
    "assemble_screenshot_manager",
    "assemble_global_hotkeys",
    "build_input_router",
    "capture_factory",
    "metrics_payload",
    "close_client",
    "close_retrieval",
    "build_handlers",
    "run_main",
    "start_process",
    "main",
]


# 退出码表见《2-骨架与配置》命令行入口一节
EXIT_OK: int = 0  # 正常
EXIT_ERROR: int = 1  # 运行错误
EXIT_ARG_ERROR: int = 2  # 参数错误
EXIT_CONFIG_ERROR: int = 3  # 配置缺失或校验失败


# 状态字段的取值，见《2-骨架与配置》状态字段取值表。
# ASR 字段多一个 idle：模型还没加载过时是 idle。
ASR_STATE_IDLE: str = "idle"
# DIFY 字段三档：offline 是没连上或不连真实 Dify 服务，degraded 是连了但出问题
DIFY_STATE_OFFLINE: str = "offline"
DIFY_STATE_READY: str = "ready"
DIFY_STATE_DEGRADED: str = "degraded"

# 停止采集但没识别到内容时给状态栏提示位的一句话，
# 见《5-语音输入与流式ASR》F8 交互一节
VOICE_EMPTY_HINT: str = "识别失败，没有听到内容"

# 语音开关的去抖窗口（秒）：界面 F8 与桌面快捷键可能为同一次按键各触发一次，
# 窗口内的重复 toggle 直接丢掉，见 :meth:`MainState.toggle_voice_capture`。
# 用单调时钟计时，不会被系统时间调整影响。
VOICE_TOGGLE_DEBOUNCE_SECONDS: float = 0.4

# TUI 运行时的日志文件名：界面占着终端，日志只追加落盘，不写 stderr。
# 相对路径按仓库根解析，最终文件是 ``logs/interviewmate.log``。
TUI_LOG_FILENAME: str = "interviewmate.log"


# 基础依赖包，取值见《2-骨架与配置》依赖一节。
# providers 指向真实实现时才带上对应的包：providers.asr=mock 时不要求 funasr 与 torch。
BASE_PACKAGES: tuple[str, ...] = (
    "textual",
    "rich",
    "httpx",
    "pydantic",
    "yaml",
    "dotenv",
    "numpy",
)
# pillow 只在真实截图实现里用来压缩与转格式，见《2-骨架与配置》依赖一节。
# 这里写的是导入名，不是发行包名：Pillow 的导入名是 PIL。
IMAGE_PACKAGE: str = "PIL"
ASR_PACKAGES: tuple[str, ...] = ("funasr", "modelscope", "torch", "kaldi_native_fbank")


@dataclass(frozen=True)
class ProbeResult:
    """一次本机能力检查的结果。

    ``ok`` 是检查结论；``detail`` 是给人看的一句话；``code`` 是失败时的错误码，
    取值见《2-骨架与配置》错误码表。
    """

    ok: bool
    detail: str
    code: str = ""


# ----------------------------------------------------------------------
# 配置与依赖
# ----------------------------------------------------------------------
def resolve_config_path(config_path: str | None = None) -> Path:
    """把配置路径解析成绝对路径。

    相对路径按仓库根解析，与当前工作目录无关；None 时用仓库根的
    ``config/config.yaml``。

    :param config_path: 配置文件路径。
    """
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_app_config(
    config_path: str | None = None,
    env_file: str | None = None,
) -> AppConfig:
    """读配置并校验。

    :param config_path: 配置文件路径，None 时用仓库根的 ``config/config.yaml``。
    :param env_file: 环境变量文件路径，None 时用仓库根的 .env。
    :raises ConfigError: 文件读不到、YAML 语法错、字段类型错、取值非法。
    """
    return load_config(str(resolve_config_path(config_path)), env_file=env_file)


def required_packages(config: AppConfig) -> tuple[str, ...]:
    """本次运行要用的依赖包。

    基础包见《2-骨架与配置》依赖一节；音频与截图按配置开关带上，
    这两个能力的第三方包由当前平台的实现给出（Windows 侧是 ``psutil`` 与
    ``mss`` 这一套）；
    ``providers.asr`` 指向真实实现时才要求 FunASR 这一套，
    ``providers.screenshot`` 指向真实实现时才要求 pillow。

    :param config: 应用配置。
    """
    packages = list(BASE_PACKAGES)
    if config.audio.enabled:
        packages.extend(get_platform().audio.required_packages)
    if config.screenshot.enabled:
        packages.extend(get_platform().screenshot.required_packages)
    if config.providers.asr == "funasr":
        packages.extend(ASR_PACKAGES)
    if config.providers.screenshot != "mock":
        packages.append(IMAGE_PACKAGE)
    # 去重并保持顺序：平台给的包可能与基础包重叠（Windows 侧 numpy 算两次）
    unique: list[str] = []
    for name in packages:
        if name not in unique:
            unique.append(name)
    return tuple(unique)


def check_dependencies(packages: Iterable[str]) -> list[str]:
    """检查包能不能导入，返回导入失败的包名。

    :param packages: 包名列表。
    """
    missing: list[str] = []
    for name in packages:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


# ----------------------------------------------------------------------
# 本机能力检查
# ----------------------------------------------------------------------
def list_input_devices() -> list[AudioDevice]:
    """枚举本机音频输入设备。

    委托给当前平台的音频后端：Windows 侧是 WASAPI 枚举，
    字段见 :class:`~app.platform.types.AudioDevice`。

    :raises AudioDeviceError: 音频依赖不可导入，或枚举失败。
    """
    return get_platform().audio.list_devices()


def probe_audio_devices() -> ProbeResult:
    """检查音频设备能不能用。

    失败只给结果不抛异常：``start`` 的第 3 步只告警不退出。

    失败时的错误码是 ``E_AUDIO_DEVICE``。
    """
    try:
        devices = list_input_devices()
    except AudioDeviceError as exc:
        return ProbeResult(ok=False, detail=exc.message, code=exc.code)
    except NotImplementedError as exc:
        # 平台的音频后端还没有实现
        return ProbeResult(ok=False, detail=str(exc), code="E_AUDIO_DEVICE")
    if not devices:
        return ProbeResult(ok=False, detail="本机没有音频输入设备", code="E_AUDIO_DEVICE")
    return ProbeResult(ok=True, detail=f"本机有 {len(devices)} 个音频输入设备")


async def probe_screenshot_interface(timeout: float = 3.0) -> ProbeResult:
    """检查截图接口能不能用。

    委托给当前平台的截图后端（Windows 侧由 ``mss`` 实现）：只查接口在不在，
    不真截图，所以不会在用户屏幕上留下痕迹。开发期 ``providers.screenshot=mock`` 时
    也只做这一步能力检查，不落盘存图。

    失败时的错误码是 ``E_SCREENSHOT``。
    """
    try:
        detail = await get_platform().screenshot.probe(timeout)
    except ScreenshotError as exc:
        return ProbeResult(ok=False, detail=exc.message, code=exc.code)
    except NotImplementedError as exc:
        # 平台的截图后端还没有实现
        return ProbeResult(ok=False, detail=str(exc), code="E_SCREENSHOT")
    return ProbeResult(ok=True, detail=detail)


# ----------------------------------------------------------------------
# 性能指标
# ----------------------------------------------------------------------
def metrics_payload() -> list[dict[str, Any]]:
    """返回各指标的统计列表。

    只包含有样本的指标，按《2-骨架与配置》性能指标一节的指标顺序排。
    元素形如 ``{"name": "capture_latency", "count": 3, "p50_ms": ...,
    "p90_ms": ..., "p95_ms": ..., "max_ms": ...}``；没有样本时是空列表。

    样本落在进程级默认记录器里，见 :func:`app.utils.timing.default_recorder`。
    """
    recorder = default_recorder()
    items: list[dict[str, Any]] = []
    for name in METRIC_NAMES:
        stats = recorder.stats(name)
        if stats is None:
            continue
        items.append(
            {
                "name": stats.name,
                "count": stats.count,
                "p50_ms": stats.p50_ms,
                "p90_ms": stats.p90_ms,
                "p95_ms": stats.p95_ms,
                "max_ms": stats.max_ms,
            }
        )
    return items


def _log_metric_stats() -> None:
    """把各指标的统计打成一条日志。

    事件名 ``metric.stats``，P50、P90、P95 与最大值写进日志字段；
    指标多时一条日志带全部，不一条指标一条日志。没有样本时不记这条日志。
    """
    items = metrics_payload()
    if not items:
        return
    fields: dict[str, Any] = {}
    for item in items:
        name = str(item["name"])
        fields[f"{name}_count"] = item["count"]
        fields[f"{name}_p50_ms"] = round(float(item["p50_ms"]), 3)
        fields[f"{name}_p90_ms"] = round(float(item["p90_ms"]), 3)
        fields[f"{name}_p95_ms"] = round(float(item["p95_ms"]), 3)
        fields[f"{name}_max_ms"] = round(float(item["max_ms"]), 3)
    names = "、".join(str(item["name"]) for item in items)
    log_event("metric.stats", f"指标统计：{names}", **fields)


# ----------------------------------------------------------------------
# 主进程状态与 action 处理
# ----------------------------------------------------------------------
class MainState:
    """主进程当前的状态。

    状态字段的取值见《2-骨架与配置》状态字段取值表。
    DEEPSEEK 字段由问题管理器的状态回调更新：请求发出前是 connecting，
    收到第一个流式片段是 streaming，正常结束回到 idle，出错是 error。

    :param voice_toggle_debounce_seconds: 语音开关的去抖窗口（秒）。
        ``run_main`` 按 :data:`VOICE_TOGGLE_DEBOUNCE_SECONDS` 打开；
        直接构造主进程状态的工具与测试默认是 0，不去抖，
        连续两次调用仍然各执行一次。
    :param voice_toggle_clock: 去抖用的单调时钟，默认 :func:`time.monotonic`。
    """

    def __init__(
        self,
        *,
        voice_toggle_debounce_seconds: float = 0.0,
        voice_toggle_clock: Callable[[], float] | None = None,
    ) -> None:
        self.audio_on: bool = False
        self.asr_state: str = ASR_STATE_IDLE
        self.dify_state: str = DIFY_STATE_OFFLINE
        # DEEPSEEK 字段，取值见《2-骨架与配置》状态字段取值表
        self.deepseek_state: str = MODEL_STATE_IDLE
        self.ttft_ms: float | None = None
        # 最近一次语音识别出来的最终文本，空串表示还没有
        self.last_voice_text: str = ""
        # 产出的最终文本计数：开始与停止录音之间用它判断本次停止有没有新文本，
        # 没有新文本时由主进程释放可能残留的截图，见 ``toggle_voice_capture``
        self.voice_final_seq: int = 0
        self._app: Any = None
        self._config: AppConfig | None = None
        self._stop_requested: bool = False
        self._screenshot_manager: ScreenshotManager | None = None
        self._client: MockClient | DeepSeekClient | None = None
        self._retrieval: RetrievalChain | None = None
        self._voice: VoiceChain | None = None
        self._input_router: InputRouter | None = None
        # 语音开关去抖：窗口与上次 toggle 的时刻（单调时钟，单位秒）
        self._voice_toggle_debounce: float = max(
            0.0, float(voice_toggle_debounce_seconds)
        )
        self._voice_toggle_clock: Callable[[], float] = (
            voice_toggle_clock if voice_toggle_clock is not None else time.monotonic
        )
        self._last_voice_toggle_at: float | None = None

    def set_deepseek_state(self, value: str) -> None:
        """更新 DEEPSEEK 字段。

        问题管理器在请求生命周期里往外报状态，这里把状态记下来，
        见《2-骨架与配置》状态字段取值表。

        :param value: 取值是 idle、connecting、streaming 或 error，
            其它取值不认，状态保持原样。
        """
        if value not in (
            MODEL_STATE_IDLE,
            MODEL_STATE_CONNECTING,
            MODEL_STATE_STREAMING,
            MODEL_STATE_ERROR,
        ):
            log_event(
                "main.unknown_model_state",
                f"不认识的模型状态：{value}",
                level="WARNING",
            )
            return
        self.deepseek_state = value
        self._schedule_status_publish()

    def _schedule_status_publish(self) -> None:
        """把状态发布出去。

        状态回调是同步的，发布是协程：这里排一个任务让它跑。
        没有运行中的事件循环时只记状态，不发布。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.publish_status())

    def attach_app(self, app: Any) -> None:
        """记住 Textual 应用实例，供 ``stop`` 动作调用退出。

        :param app: Textual 应用实例。
        """
        self._app = app

    def attach_config(self, config: AppConfig) -> None:
        """记住应用配置，预热时按 ``providers`` 的取值决定探测哪一项。

        :param config: 应用配置。
        """
        self._config = config

    def set_model_client(self, client: MockClient | DeepSeekClient) -> None:
        """记住模型客户端，预热时用它探测模型。

        :param client: 模型客户端。
        """
        self._client = client

    def set_retrieval_chain(self, retrieval: RetrievalChain) -> None:
        """记住检索链路，预热时用它探测 Dify。

        :param retrieval: 检索链路的装配结果。
        """
        self._retrieval = retrieval

    def set_voice_chain(self, voice: VoiceChain) -> None:
        """记住语音链路，socket 的 ``voice-toggle`` 动作靠它做真采集。

        :param voice: 语音链路的装配结果。
        """
        self._voice = voice

    def set_input_router(self, router: InputRouter) -> None:
        """记住输入路由器，语音采集启停时用它标记录音会话。

        :param router: 三种输入共用的输入路由器。
        """
        self._input_router = router

    def set_screenshot_manager(self, screenshot_manager: ScreenshotManager) -> None:
        """记住截图管理器，界面起来之后由装配层塞进来。

        界面还没起来时收到 ``screenshot`` 动作，只做接口探测不采集，
        见《2-骨架与配置》进程通信一节。

        :param screenshot_manager: 截图管理器。
        """
        self._screenshot_manager = screenshot_manager

    @property
    def stop_requested(self) -> bool:
        """有没有收到过 ``stop`` 动作。

        ``stop`` 可能早于界面起来到达，这时界面还没建起来，
        标志先记下来，界面装好后由 :func:`run_main` 认领。
        """
        return self._stop_requested

    def toggle_voice(self) -> dict[str, Any]:
        """切换语音采集状态，返回切换后的状态。

        只切状态字段，不做真采集：真采集走 :meth:`toggle_voice_capture`，
        那条路要开流、跑模型，只在有事件循环的地方能跑。
        """
        self.audio_on = not self.audio_on
        return {
            "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
            "note": "只切换状态字段，没有做真采集",
        }

    async def toggle_voice_capture(self) -> dict[str, Any]:
        """按一次 F8：开始或停止语音采集。

        走 :class:`~app.inputs.hotkey_controller.HotkeyController`，
        真的开流与关流，见《5-语音输入与流式ASR》F8 交互一节。
        采集在事件循环里跑，不占住调用方。

        开始采集时临时识别文字的累计值清零，下一条语音流从头开始；
        停止采集时那一行清掉，最终文本已经进了右栏题面；
        没识别到内容时状态栏提示位给一句失败提示，不发模型请求，
        见《5-语音输入与流式ASR》F8 交互一节。

        界面 F8 与桌面快捷键可能为同一次按键各触发一次，键盘重复也会连着触发：
        上一次 toggle 之后去抖窗口内到达的重复调用直接丢掉，只记一条
        ``audio.toggle.debounced`` 日志。窗口用单调时钟算，按一次仍然立即开流；
        去抖只挡重复的开关，不捕获切换过程中的异常。

        :returns: 动作结果，字段有 ``audio``、``asr``、``result``、``text`` 与 ``note``。
        """
        debounced = self._debounce_voice_toggle()
        if debounced is not None:
            return debounced
        voice = self._voice
        if voice is None:
            # 语音链路还没装好：没有采集可开关，状态字段不动
            log_event("main.voice.unavailable", "语音链路还没装好，忽略这次语音动作")
            return {
                "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
                "asr": self.asr_state,
                "result": RESULT_IGNORED,
                "text": "",
                "note": "语音链路还没装好，没有采集可开关",
            }
        if voice.engine.state == STATE_LOADING:
            # 模型还在加载，状态栏先显示 loading，加载完再发 ready 或 error，
            # 见《5-语音输入与流式ASR》状态显示一节
            self.asr_state = STATE_LOADING
            await self.publish_status()
        # 记下动作前的最终文本计数：停止录音后它没变，说明本次停止不会走
        # route_voice，残留在路由器里的截图要由主进程释放
        final_seq_before = self.voice_final_seq
        result = await voice.hotkey.handle_voice_toggle()
        await self._apply_voice_result(result, final_seq_before)
        return {
            "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
            "asr": self.asr_state,
            "result": result,
            "text": self.last_voice_text,
            "note": _voice_note(result),
        }

    async def _apply_voice_result(self, result: str, final_seq_before: int) -> None:
        """按动作结果更新状态字段、录音会话与状态栏。

        切换路径与按住路径共用这一段：更新 AUDIO 与 ASR 字段，
        按 ``started`` 通知路由器录音开始，按 ``stopped``/``empty``
        且本次没有新最终文本时释放可能残留的截图，
        再清临时识别文字、空语音时给一句提示，最后发布状态。

        :param result: 这次动作的结果标识，见
            :meth:`~app.audio.audio_manager.AudioManager.toggle`。
        :param final_seq_before: 动作前的最终文本计数，
            动作后它没变说明本次停止不会走 ``route_voice``。
        """
        voice = self._voice
        if voice is None:
            return
        self.audio_on = voice.audio_manager.is_running
        self.asr_state = voice.audio_manager.asr_state
        router = self._input_router
        if router is not None:
            if result == RESULT_STARTED:
                # 录音开始：截图先到时暂存图片，等语音最终文本合并
                router.voice_session_started()
            elif (
                result in (RESULT_STOPPED, RESULT_EMPTY)
                and self.voice_final_seq == final_seq_before
            ):
                # 本次停止没有新最终文本：不会走 route_voice，
                # 这里释放可能残留的截图，空语音不会把旧图留到下一次提问
                await router.voice_finished()
        if result in (RESULT_STARTED, RESULT_STOPPED, RESULT_EMPTY):
            # 开始时累计值清零，下一条语音流从头开始；
            # 停止后临时行清掉，最终文本已经进了右栏题面
            self._clear_voice_text()
        if result == RESULT_EMPTY:
            # 没识别到内容：不发模型请求，状态栏提示位给一句失败提示
            await self.publish_hint(VOICE_EMPTY_HINT)
        await self.publish_status()

    async def start_voice_capture(self) -> dict[str, Any]:
        """按住语音键：开始语音采集，与切换路径共用同一段结果处理。

        已经开着采集时不再开一次流，直接返回忽略结果；
        模型还在加载时状态栏先显示 loading，见
        《5-语音输入与流式ASR》状态显示一节。

        按住路径不走去抖：按键重复时靠 ``is_running`` 判断，
        不会在按住期间反复开流。

        :returns: 动作结果，字段有 ``audio``、``asr``、``result``、``text`` 与 ``note``。
        """
        voice = self._voice
        if voice is None:
            # 语音链路还没装好：没有采集可开关，状态字段不动
            log_event("main.voice.unavailable", "语音链路还没装好，忽略这次语音动作")
            return {
                "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
                "asr": self.asr_state,
                "result": RESULT_IGNORED,
                "text": "",
                "note": "语音链路还没装好，没有采集可开关",
            }
        if voice.audio_manager.is_running:
            log_event(
                "main.voice.start_ignored",
                "已在采集，忽略按住开始",
                level="DEBUG",
            )
            return {
                "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
                "asr": self.asr_state,
                "result": RESULT_IGNORED,
                "text": self.last_voice_text,
                "note": "已在采集，忽略按住开始",
            }
        if voice.engine.state == STATE_LOADING:
            # 模型还在加载，状态栏先显示 loading，加载完再发 ready 或 error
            self.asr_state = STATE_LOADING
            await self.publish_status()
        final_seq_before = self.voice_final_seq
        result = await voice.hotkey.handle_voice_start()
        await self._apply_voice_result(result, final_seq_before)
        return {
            "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
            "asr": self.asr_state,
            "result": result,
            "text": self.last_voice_text,
            "note": _voice_note(result),
        }

    async def stop_voice_capture(self) -> dict[str, Any]:
        """松开语音键：停止语音采集，最终文本照常送出去。

        没在采集时是空操作，直接返回忽略结果，不碰采集器；
        停止时的结果处理与切换路径共用 :meth:`_apply_voice_result`。

        :returns: 动作结果，字段有 ``audio``、``asr``、``result``、``text`` 与 ``note``。
        """
        voice = self._voice
        if voice is None or not voice.audio_manager.is_running:
            log_event(
                "main.voice.stop_ignored",
                "没在采集，忽略按住停止",
                level="DEBUG",
            )
            return {
                "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
                "asr": self.asr_state,
                "result": RESULT_IGNORED,
                "text": self.last_voice_text,
                "note": "没在采集，忽略按住停止",
            }
        final_seq_before = self.voice_final_seq
        result = await voice.hotkey.handle_voice_stop()
        await self._apply_voice_result(result, final_seq_before)
        return {
            "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
            "asr": self.asr_state,
            "result": result,
            "text": self.last_voice_text,
            "note": _voice_note(result),
        }

    def _debounce_voice_toggle(self) -> dict[str, Any] | None:
        """把去抖窗口内的重复 toggle 丢掉。

        上一次 toggle 之后 ``voice_toggle_debounce_seconds`` 内又到的调用
        直接返回一条忽略结果并记 ``audio.toggle.debounced``，不碰语音链路，
        这样同一次按键被界面 F8 与桌面快捷键各触发一次时只真正切换一次。
        被丢掉的调用也会把窗口起点往后推：按住 F8 时键盘重复一直落在窗口内，
        不会每过 400 毫秒又切换一次。窗口外（或窗口为 0）的调用照常执行。

        :returns: 丢弃时返回忽略结果；照常执行时返回 None。
        """
        window = self._voice_toggle_debounce
        now = self._voice_toggle_clock()
        last = self._last_voice_toggle_at
        self._last_voice_toggle_at = now
        if window > 0 and last is not None and (now - last) < window:
            log_event(
                "audio.toggle.debounced",
                "去抖窗口内的重复语音开关，忽略这次调用",
                level="WARNING",
                elapsed_ms=round(max(0.0, now - last) * 1000.0, 3),
                window_ms=round(window * 1000.0, 3),
            )
            return {
                "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
                "asr": self.asr_state,
                "result": RESULT_IGNORED,
                "text": self.last_voice_text,
                "note": "按键重复，忽略这次语音开关",
            }
        return None

    def _clear_voice_text(self) -> None:
        """清掉状态栏下方的临时识别文字，累计值一起清零。

        界面没挂上或没有这个方法时只记日志，不抛异常。
        """
        app = self._app
        if app is None:
            return
        clearer = getattr(app, "clear_voice_text", None)
        if clearer is None:
            log_event("main.voice.text_missing", "界面还没有临时识别文字行")
            return
        try:
            clearer()
        except Exception as exc:
            log_event(
                "main.voice.text_clear_failed",
                f"临时识别文字清不掉：{exc}",
                level="WARNING",
                error=str(exc),
            )

    async def stop_voice(self) -> None:
        """停掉语音采集。

        收尾流程的第一步，见《2-骨架与配置》进程通信一节对 ``stop`` 的描述：
        先停音频采集，再关 HTTP 客户端。
        """
        voice = self._voice
        if voice is None:
            self.audio_on = False
            return
        try:
            await voice.audio_manager.stop()
        except Exception as exc:
            log_event(
                "main.audio.stop_failed",
                f"停止语音采集失败：{exc}",
                level="WARNING",
                error=str(exc),
            )
        self.audio_on = voice.audio_manager.is_running
        self.asr_state = voice.audio_manager.asr_state
        await self.publish_status()

    async def publish_status(self, **fields: Any) -> None:
        """把当前状态发布到界面。

        界面只认完整快照：少一个字段就会被默认值覆盖掉，
        所以这里拿界面上一次的快照打底，只改 AUDIO、ASR、DIFY、DEEPSEEK
        四个字段。提示位归界面管，按界面上的快照原样重发。

        :param fields: 要额外覆盖的字段。
        """
        app = self._app
        if app is None:
            return
        publisher = getattr(app, "publish_status", None)
        if publisher is None:
            return
        payload = self._status_fields()
        payload.update(fields)
        try:
            result = publisher(**payload)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            log_event(
                "main.status_publish_failed",
                f"发布状态更新失败：{exc}",
                level="WARNING",
                error=str(exc),
            )

    async def publish_hint(self, hint: str) -> None:
        """把提示位发布到界面。

        提示位归界面管，界面只认 ``publish_hint`` 这一条路，
        截图链路的提示也走它，见《5-语音输入与流式ASR》状态显示一节。

        :param hint: 提示位上的一句话，空串表示清掉提示。
        """
        app = self._app
        if app is None:
            return
        publisher = getattr(app, "publish_hint", None)
        if publisher is None:
            return
        try:
            result = publisher(hint)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            log_event(
                "main.hint_publish_failed",
                f"发布状态栏提示失败：{exc}",
                level="WARNING",
                error=str(exc),
            )

    async def publish_dify(self, value: str) -> None:
        """更新并发布 DIFY 字段。

        :param value: 取值是 offline、ready 或 degraded，
            见《2-骨架与配置》状态字段取值表。
        """
        self.dify_state = value
        await self.publish_status()

    def _status_fields(self) -> dict[str, Any]:
        """当前六项状态与提示位。"""
        self.ttft_ms = self._latest_ttft_ms()
        fields: dict[str, Any] = {
            "running": "running",
            "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
            "asr": self.asr_state,
            "dify": self.dify_state,
            "deepseek": self.deepseek_state,
            "ttft_ms": self.ttft_ms,
            "hint": "",
        }
        app = self._app
        snapshot = getattr(app, "_status", None) if app is not None else None
        if isinstance(snapshot, dict):
            # 提示位归界面管，按界面上的快照原样重发；
            # DEEPSEEK 归主进程管：问题管理器报的状态才是真的
            base = dict(snapshot)
            base.update(
                {
                    "running": fields["running"],
                    "audio": fields["audio"],
                    "asr": fields["asr"],
                    "dify": fields["dify"],
                    "deepseek": fields["deepseek"],
                    "ttft_ms": fields["ttft_ms"],
                }
            )
            return base
        return fields

    def status_payload(self) -> dict[str, Any]:
        """返回当前状态。

        字段名与《2-骨架与配置》状态字段取值表一致；
        ``metrics`` 是各指标的统计列表，见 :func:`metrics_payload`。
        """
        self.ttft_ms = self._latest_ttft_ms()
        return {
            "running": "running",
            "audio": AUDIO_STATE_ON if self.audio_on else AUDIO_STATE_OFF,
            "asr": self.asr_state,
            "dify": self.dify_state,
            "deepseek": self.deepseek_state,
            "ttft": self.ttft_ms,
            "pid": os.getpid(),
            "metrics": metrics_payload(),
        }

    @staticmethod
    def _latest_ttft_ms() -> float | None:
        """最近一条首字延迟样本（毫秒）。

        值从进程级记录器里取：``default_recorder().samples("ttft")`` 的最后一条。
        没有样本时是 None，见《8-集成与验收》性能统计一节。
        """
        samples = default_recorder().samples("ttft")
        if not samples:
            return None
        return float(samples[-1].value_ms)

    async def warmup_payload(self) -> dict[str, Any]:
        """返回预热请求的结果。

        预热做五件事：预热 ASR、建立 HTTP 连接、探测模型、探测 Dify、
        刷新设备列表，见《2-骨架与配置》命令行入口一节对 ``warmup`` 的描述。
        每一项失败都只记日志，不影响其它项，也不影响程序。

        ``providers`` 指向 mock 的那一项不连真实服务，对应项给 mock 标记。
        """
        asr = await self._warmup_asr()
        deepseek = await self._warmup_model()
        dify = await self._warmup_dify()
        devices = await self._warmup_devices()
        return {
            "asr": asr,
            "deepseek": deepseek,
            "dify": dify,
            "devices": devices,
            "note": f"ASR {asr}、模型 {deepseek}、Dify {dify}，设备列表刷新到 {devices} 台",
        }

    async def _warmup_asr(self) -> str:
        """预热 ASR 模型。

        模型加载失败不抛异常：状态标成 error，程序照常往下走，
        见《5-语音输入与流式ASR》FunASR 模型一节。
        """
        voice = self._voice
        if voice is None:
            return ASR_STATE_IDLE
        if voice.asr_provider == "mock":
            return "mock"
        try:
            await voice.engine.ensure_loaded()
        except Exception as exc:
            log_event(
                "main.warmup.asr_failed",
                f"ASR 预热失败：{exc}",
                level="WARNING",
                error=str(exc),
            )
            return STATE_ERROR
        return voice.engine.state

    async def _warmup_model(self) -> str:
        """探测模型能力，顺便把 HTTP 连接建起来。

        失败只给 error，不抛异常，调用方只告警不退出；
        DEEPSEEK 字段同步落到 error，与启动探测失败的处理一致。
        """
        client = self._client
        config = self._config
        if client is None:
            return "idle"
        if config is not None and config.providers.llm == "mock":
            return "mock"
        try:
            await client.check_health()
        except Exception as exc:
            log_event(
                "main.warmup.model_failed",
                f"模型探测失败：{exc}",
                level="WARNING",
                error=str(exc),
            )
            self.set_deepseek_state(MODEL_STATE_ERROR)
            return "error"
        return "ready"

    async def _warmup_dify(self) -> str:
        """探测 Dify 可用状态，顺便把 HTTP 连接建起来。

        探测失败时状态标成 degraded，见《6-Dify检索与项目知识》故障降级一节。
        """
        retrieval = self._retrieval
        if retrieval is None or retrieval.provider is None:
            return DIFY_STATE_OFFLINE
        if retrieval.provider_name == "mock":
            return "mock"
        try:
            outcome = await retrieval.provider.retrieve_outcome("warmup")
        except Exception as exc:
            log_event(
                "main.warmup.dify_failed",
                f"Dify 探测失败：{exc}",
                level="WARNING",
                error=str(exc),
            )
            await self.publish_dify(DIFY_STATE_DEGRADED)
            return DIFY_STATE_DEGRADED
        if outcome.degraded or not outcome.ok:
            log_event(
                "main.warmup.dify_degraded",
                f"Dify 探测降级：{outcome.error_detail}",
                level="WARNING",
                error_code=outcome.error_code,
            )
            await self.publish_dify(DIFY_STATE_DEGRADED)
            return DIFY_STATE_DEGRADED
        await self.publish_dify(DIFY_STATE_READY)
        return DIFY_STATE_READY

    async def _warmup_devices(self) -> int:
        """刷新本机音频设备列表，返回设备台数。

        枚举失败只记日志，设备列表保持上一次的结果。
        """
        try:
            devices = list_input_devices()
        except Exception as exc:
            log_event(
                "main.warmup.devices_failed",
                f"刷新音频设备列表失败：{exc}",
                level="WARNING",
                error=str(exc),
            )
            return 0
        log_event("main.warmup.devices", "音频设备列表已刷新", count=len(devices))
        return len(devices)

    async def capture_screenshot(self) -> dict[str, Any]:
        """做一次截图并提交。

        界面起来之后截图管理器已经装好，这时真的采集一张图并提交一次带图请求；
        界面还没起来时只查接口能不能用，不采集。

        :raises ScreenshotError: 界面还没起来且截图接口不可用。
        """
        screenshot_manager = self._screenshot_manager
        if screenshot_manager is None:
            probe = await probe_screenshot_interface()
            if not probe.ok:
                raise ScreenshotError(probe.detail)
            return {
                "provider": "mock",
                "captured": False,
                "detail": probe.detail,
                "note": "界面还没起来，只做能力检查，不落盘存图",
            }
        # 题面写到右栏顶部：socket 这条路与 TUI 内 F9 是同一个入口，
        # 见《4-截图输入与多模态》截图流程一节
        app = self._app
        if app is not None:
            app.show_screenshot_question()
        captured = await screenshot_manager.capture_and_submit()
        return {
            "provider": screenshot_manager.backend_name,
            "captured": captured,
            "detail": "截图已提交" if captured else "截图失败，原因见状态栏提示位",
        }

    def request_stop(self) -> None:
        """请求主进程退出。

        界面还没起来时先记标志，界面起来后由 :func:`run_main` 认领退出；
        否则这里会空转，命令行侧会一直等到超时。
        """
        self._stop_requested = True
        app = self._app
        if app is not None:
            app.exit()


def build_handlers(state: MainState) -> dict[str, Callable[[Request], Any]]:
    """建 action 到 handler 的映射。

    action 取值见《2-骨架与配置》进程通信一节：``voice-toggle``、``screenshot``、
    ``stop``、``status``、``warmup``。handler 收到请求，返回值当作响应 payload。

    分发入口只有一条：:class:`~app.inputs.global_command.GlobalCommandDispatcher`。
    这里把五个动作注册进去，返回它给 socket 服务端用的 handler 映射；
    未知动作由它兜住，不抛异常，见《8-集成与验收》全局快捷键一节。

    :param state: 主进程状态。
    """
    async def handle_voice_toggle(request: Request) -> dict[str, Any]:
        """切换语音采集状态。

        走真采集：开始或停止音频采集与识别，
        见《5-语音输入与流式ASR》F8 交互一节。
        """
        return await state.toggle_voice_capture()

    async def handle_screenshot(request: Request) -> dict[str, Any]:
        """触发一次截图并提交。"""
        return await state.capture_screenshot()

    async def handle_stop(request: Request) -> dict[str, Any]:
        """走收尾流程，退出主进程。

        先停掉音频采集，再退出界面，见《2-骨架与配置》进程通信一节对 ``stop`` 的描述。
        """
        await state.stop_voice()
        state.request_stop()
        return {"stopping": True}

    async def handle_status(request: Request) -> dict[str, Any]:
        """返回当前状态。"""
        return state.status_payload()

    async def handle_warmup(request: Request) -> dict[str, Any]:
        """触发预热。"""
        return await state.warmup_payload()

    dispatcher = GlobalCommandDispatcher()
    dispatcher.register(ACTION_VOICE_TOGGLE, handle_voice_toggle)
    dispatcher.register(ACTION_SCREENSHOT, handle_screenshot)
    dispatcher.register(ACTION_STOP, handle_stop)
    dispatcher.register(ACTION_STATUS, handle_status)
    dispatcher.register(ACTION_WARMUP, handle_warmup)
    return dispatcher.socket_handlers()


# ----------------------------------------------------------------------
# 模型客户端与装配
# ----------------------------------------------------------------------
def select_client(config: AppConfig) -> MockClient | DeepSeekClient:
    """按 ``providers.llm`` 选模型客户端。

    ``mock`` 用本地假流客户端，不发网络请求；
    ``deepseek`` 用真实客户端，API Key 取环境变量 ``DEEPSEEK_API_KEY``。

    :param config: 应用配置。
    """
    if config.providers.llm == "deepseek":
        api_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        return DeepSeekClient(config, api_key)
    return MockClient(config)


async def probe_model_capability(
    client: MockClient | DeepSeekClient,
    config: AppConfig,
) -> ProbeResult:
    """探测模型能力，对应 ``start`` 的第 7 步。

    ``providers.llm=mock`` 时不连网络，只说明用的是 mock；
    ``deepseek`` 时真的探测一次，失败只给结果不抛异常，调用方只告警不退出。
    失败时的错误码是 ``E_DEEPSEEK_AUTH``、``E_DEEPSEEK_TIMEOUT``
    或 ``E_DEEPSEEK_STREAM``。

    探测结果只用于展示：detail 里带上模型支不支持 image 模态，
    客户端不因缺少 ``image`` 而拒绝截图，见《4-截图输入与多模态》模型视觉能力一节。

    :param client: 模型客户端。
    :param config: 应用配置。
    """
    if config.providers.llm == "mock":
        return ProbeResult(
            ok=True,
            detail="providers.llm=mock，用的是本地假流客户端，不连真实服务",
        )
    try:
        capability = await client.check_health()
    except InterviewMateError as exc:
        return ProbeResult(ok=False, detail=exc.message, code=exc.code)
    except Exception as exc:
        return ProbeResult(ok=False, detail=str(exc), code="E_DEEPSEEK_STREAM")
    supports_image = "是" if capability.supports_image else "否"
    return ProbeResult(
        ok=True,
        detail=f"模型 {capability.model_id}，支持图片：{supports_image}",
    )


def assemble_manager(
    config: AppConfig,
    client: MockClient | DeepSeekClient,
    app: Any,
    *,
    retrieve: RetrievalFunc | None = None,
    on_sources: SourcesCallback | None = None,
    state: MainState | None = None,
) -> QuestionManager:
    """把问题管理器装配到界面上。

    顺序：事件分发器 → 解析器工厂 → 提示词 → 问题管理器 → 挂到界面上。
    事件分发器要界面上的两个面板，面板要等界面挂载后才拿得到，
    所以这个函数要在界面起来之后调用。

    检索回调与来源回调可选：给了就接上《6-Dify检索与项目知识》检索流程一节
    里的检索，没给就只做检索决策，不发起真实检索。

    代码题通知直接接分发器的 ``set_coding_mode``：判定出代码题时右栏先切形态，
    模型一个协议事件都不给时围栏块也能进代码区，
    见《7-代码问题模式》纯代码块兜底一节。

    给了主进程状态时，问题管理器的模型状态回调接到它身上：
    DEEPSEEK 字段跟着请求生命周期走，不再写死，
    见《2-骨架与配置》状态字段取值表。

    ``providers.llm`` 不是 mock 时装配模型题型判定器并注入问题管理器；
    mock 模式传 None，只走同步规则，不发网络请求，
    见《3-文本输入与流式回答》题型判定一节。

    :param config: 应用配置。
    :param client: 模型客户端。
    :param app: Textual 应用实例。
    :param retrieve: 发起检索的回调，参数是题面，返回上下文文本与来源条目。
    :param on_sources: 把来源条目送给界面的回调。
    :param state: 主进程状态，给了就把模型状态报给它。
    :returns: 装配好的问题管理器。
    """
    dispatcher = EventDispatcher(
        app.navigation_panel,
        app.answer_panel,
        app.show_error,
    )
    prompt_manager = PromptManager(config)

    # 题型与检索需求改由模型判定：mock 模式只走同步规则，不发网络请求；
    # 真实客户端失败或超时时问题管理器回退同步规则，
    # 见《3-文本输入与流式回答》题型判定一节
    llm_classifier: DeepSeekQuestionClassifier | None = None
    if config.providers.llm != "mock":
        llm_classifier = DeepSeekQuestionClassifier(client)  # type: ignore[arg-type]

    def _parser_factory(request_id: str, session_id: str) -> StreamParser:
        """按请求新建解析器，每个请求用各自的解析器。"""
        return StreamParser(request_id, session_id)

    manager = QuestionManager(
        config,
        client,
        _parser_factory,
        dispatcher,
        prompt_manager.build_messages,
        build_multimodal_messages=prompt_manager.build_multimodal_messages,
        retrieve=retrieve,
        on_sources=on_sources,
        on_coding_mode=dispatcher.set_coding_mode,
        on_model_state=state.set_deepseek_state if state is not None else None,
        llm_classifier=llm_classifier,
    )
    app.attach_manager(manager)
    return manager


def build_input_router(manager: QuestionManager) -> InputRouter:
    """造输入路由器：三种输入都从这里进。

    手工输入回车、语音最终文本、截图三种来源都交给它，
    提交回调指向问题管理器的 ``submit_text``，
    见《8-集成与验收》三种输入的汇合一节。

    :param manager: 问题管理器。
    :returns: 装好的输入路由器。
    """

    async def _submit(question: str, image_data_url: str | None) -> None:
        """提交一条题面，可以带一张图。"""
        if image_data_url:
            await manager.submit_text(question, image_data_url)
            return
        await manager.submit_text(question)

    return InputRouter(_submit)


class _RouterSubmitter:
    """把截图管理器的提交转到输入路由器上。

    截图管理器只会调 ``submit_text``，这里转成路由器的 ``route_screenshot``：
    于是截图也走三条输入的同一条路，
    见《8-集成与验收》三种输入的汇合一节。

    文字仍然由问题管理器取会话保留的语音文字，与截图合成一条请求，
    见《4-截图输入与多模态》与语音合并一节。
    """

    def __init__(self, router: InputRouter) -> None:
        self._router = router

    async def submit_text(self, text: str, image_data_url: str | None = None) -> None:
        """按有没有图片分发到路由器。"""
        if image_data_url:
            await self._router.route_screenshot(image_data_url)
            return
        await self._router.route_text(text)


def assemble_screenshot_manager(
    config: AppConfig,
    app: Any,
    manager: QuestionManager,
    *,
    router: InputRouter | None = None,
) -> ScreenshotManager:
    """把截图管理器装配到界面上。

    后端按 ``providers.screenshot`` 选：``mock`` 用假后端，返回固定 PNG；
    其它取值用当前平台的截图后端（Windows 侧是 ``mss``）。

    截图提交走输入路由器：三种输入汇到一条路上，
    见《8-集成与验收》三种输入的汇合一节。没给路由器时这里现造一个。

    截图接口的超时取配置里的 ``screenshot.timeout``，显式给到截图管理器；
    平台后端自己的超时兜底见《8-集成与验收》超时管理一节。

    这个函数要在界面起来之后调用：提示位回调要界面上的状态栏。

    :param config: 应用配置。
    :param app: Textual 应用实例。
    :param manager: 问题管理器，截图处理完交给它提交。
    :param router: 输入路由器，None 时现造一个。
    :returns: 装配好的截图管理器。
    """
    backend: ScreenshotBackend
    if config.providers.screenshot == "mock":
        backend = MockScreenshotBackend()
    else:
        backend = get_platform().screenshot
    processor = ImageProcessor(config.screenshot)
    active_router = router if router is not None else build_input_router(manager)
    screenshot_manager = ScreenshotManager(
        config,
        backend,
        processor,
        _RouterSubmitter(active_router),  # type: ignore[arg-type]
        hint_publisher=app.publish_hint,
        router=active_router,
    )
    app.attach_screenshot_manager(screenshot_manager)
    return screenshot_manager


# ----------------------------------------------------------------------
# 全局热键装配
# ----------------------------------------------------------------------
def _log_hotkey_action_failure(task: asyncio.Task[Any], key: str, phase: str) -> None:
    """全局热键起的状态动作结束后看结果，失败记一条日志。

    取消与正常结束都放过，只有抛过异常才写日志：状态方法自己会把可预期的
    失败变成结果字段，能冒到这里的是没预料到的错误。

    :param task: 回调起的状态动作任务。
    :param key: 触发这个回调的键名，只用于日志。
    :param phase: 按下或松开，只用于日志。
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    log_event(
        "main.hotkeys.action_failed",
        f"全局热键动作失败：{exc}",
        level="WARNING",
        key=key,
        action=phase,
        error=type(exc).__name__,
    )


def _hotkey_action(
    action: Callable[[], Awaitable[Any]],
    key: str,
    phase: str,
) -> Callable[[], None]:
    """把状态上的协程方法包成全局热键用的同步回调。

    回调由监听在事件循环里调，但状态方法要跑好几秒（截图后端最多阻塞
    120 秒、语音停止要冲刷 ASR），在回调里等会堵住后面的按键，
    所以只起一个任务，异常交给 :func:`_log_hotkey_action_failure` 记录，
    绝不冒给监听线程，见 :meth:`~app.inputs.global_hotkeys.HotkeyListener._invoke`。

    :param action: 状态上的协程方法，如 :meth:`MainState.start_voice_capture`。
    :param key: 触发这个回调的键名，只用于日志。
    :param phase: 按下或松开，只用于日志。
    :returns: 同步回调。
    """

    def _run() -> None:
        try:
            awaitable = action()
        except Exception as exc:
            log_event(
                "main.hotkeys.action_failed",
                f"全局热键动作没跑起来：{exc}",
                level="WARNING",
                key=key,
                action=phase,
                error=type(exc).__name__,
            )
            return
        try:
            task = asyncio.create_task(awaitable)
        except RuntimeError as exc:
            # 事件循环已关闭：协程没有机会跑，关掉免得留下未等待的告警
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            log_event(
                "main.hotkeys.action_failed",
                f"全局热键动作没跑起来：{exc}",
                level="WARNING",
                key=key,
                action=phase,
                error=type(exc).__name__,
            )
            return
        task.add_done_callback(
            lambda done: _log_hotkey_action_failure(done, key, phase)
        )

    return _run


def assemble_global_hotkeys(
    config: AppConfig,
    state: MainState,
    app: Any,
    *,
    provider: GlobalHotkeyProvider | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> HotkeyListener | None:
    """把全局热键监听装到主进程上。

    全局热键由 AutoHotkey v2 触发，终端失焦时也能收到按键。F8 按一次
    开始采集，再按一次停止并提交；F9 按一次截屏。界面被标记之后
    自己的 F8 与 F9 动作会短路，同一次按键不会处理两遍，
    见《5-语音输入与流式ASR》F8 交互一节。

    键名取配置里的 ``hotkeys.voice_toggle`` 与 ``hotkeys.screenshot``，
    归一成小写后作为绑定表的键。两项配成同一个键属于配置写错：
    只保留语音绑定并记 WARNING。

    这个函数要在输入路由器与截图管理器挂到界面上之后调用，
    保证按下时链路已经就绪。

    :param config: 应用配置。
    :param state: 主进程状态，动作落在它上面。
    :param app: Textual 应用实例，启动成功后调 ``mark_global_hotkeys_active``。
    :param provider: 全局按键提供者，None 时取当前平台的实现。
    :param loop: 回调要投递到的事件循环，None 时 :meth:`HotkeyListener.start`
        取当前运行的循环。
    :returns: 始终返回构造好的监听对象，是否生效看 ``running``。
    """
    active_provider = provider if provider is not None else get_platform().hotkeys
    voice_key = config.hotkeys.voice_toggle.strip().lower()
    screenshot_key = config.hotkeys.screenshot.strip().lower()
    bindings: dict[str, HotkeyHandlers] = {
        voice_key: HotkeyHandlers(
            on_press=_hotkey_action(state.start_voice_capture, voice_key, "press"),
            on_release=_hotkey_action(state.stop_voice_capture, voice_key, "release"),
        )
    }
    if screenshot_key == voice_key:
        log_event(
            "main.hotkeys.conflict",
            "语音键与截图键配成同一个键，只保留语音绑定",
            level="WARNING",
            key=voice_key,
        )
    else:
        bindings[screenshot_key] = HotkeyHandlers(
            on_press=_hotkey_action(state.capture_screenshot, screenshot_key, "press"),
        )
    listener = HotkeyListener(active_provider, bindings, loop=loop)
    if not listener.start():
        # 提供者不可用或启动出错：界面绑定保持原来的切换语义
        log_event(
            "main.hotkeys.unavailable",
            "全局热键不可用，F8 与 F9 仍由界面处理",
            level="WARNING",
        )
        return listener
    app.mark_global_hotkeys_active()
    log_event(
        "main.hotkeys.active",
        "全局热键已接管按键，界面不再处理 F8 与 F9",
        voice_key=voice_key,
        screenshot_key=screenshot_key,
    )
    return listener


# ----------------------------------------------------------------------
# 检索装配
# ----------------------------------------------------------------------
def select_retriever(config: AppConfig) -> MockRetriever | DifyRetriever | None:
    """按 ``providers.retrieval`` 选检索实现。

    ``mock`` 用本地假检索后端，不发网络请求；
    ``dify`` 用真实检索实现，连接参数取配置里的 ``dify`` 段，
    检索参数取 ``rag`` 段。

    Dify 配置缺项时返回 None：检索退化成「没取到」，
    状态栏的 DIFY 标成 offline，见《6-Dify检索与项目知识》故障降级一节。

    :param config: 应用配置。
    :returns: 检索实现，Dify 配置不全时是 None。
    """
    if config.providers.retrieval != "dify":
        return MockRetriever()
    dify = config.dify
    missing = [
        name
        for name, value in (
            ("dify.base_url", dify.base_url),
            ("dify.dataset_id", dify.dataset_id),
            ("dify.api_key", dify.api_key),
        )
        if not (value or "").strip()
    ]
    if missing:
        names = "、".join(missing)
        log_event(
            "retrieval.config_missing",
            f"Dify 检索配置不全：{names}",
            level="ERROR",
            missing=names,
        )
        return None
    return DifyRetriever(
        dify.base_url.strip(),
        dify.dataset_id.strip(),
        dify.api_key.strip(),
        top_k=config.rag.top_k,
        timeout=float(config.rag.timeout),
        score_threshold_enabled=config.rag.score_threshold_enabled,
        score_threshold=(
            config.rag.score_threshold if config.rag.score_threshold_enabled else None
        ),
        reranking_enable=config.rag.reranking_enable,
    )


@dataclass
class RetrievalChain:
    """检索链路的装配结果。

    :param provider: 检索实现，Dify 配置不全时是 None。
    :param provider_name: 配置里 ``providers.retrieval`` 的取值，mock 或 dify。
    :param retrieve: 检索回调，参数是题面，返回上下文文本与来源条目。
    :param on_sources: 来源条目回调，把来源条目送给界面的检索来源面板。
    """

    provider: MockRetriever | DifyRetriever | None
    provider_name: str
    retrieve: RetrievalFunc
    on_sources: SourcesCallback


def assemble_retrieval(
    config: AppConfig,
    app: Any,
    *,
    state: MainState | None = None,
) -> RetrievalChain:
    """装配检索链路。

    造两样东西：检索回调与来源条目回调。检索回调发起检索、把片段拼成上下文，
    来源条目回调把来源条目送到界面的检索来源面板。

    检索失败或没取到内容时检索回调返回 ``(None, [])``：
    请求照常发出，回答里不声称检索到了项目资料，
    见《6-Dify检索与项目知识》故障降级一节。

    :param config: 应用配置。
    :param app: Textual 应用实例，来源条目送给它。
    :param state: 主进程状态，DIFY 字段由它发布。
    :returns: 装配好的检索链路。
    """
    provider = select_retriever(config)
    provider_name = config.providers.retrieval
    builder = ContextBuilder(max_context_chars=config.rag.max_context_chars)

    async def _publish_dify(value: str) -> None:
        """发布 DIFY 字段。没有主进程状态时只记日志。"""
        if state is None:
            return
        await state.publish_dify(value)

    async def _publish_dify_state(ok: bool) -> None:
        """按检索结果发布 DIFY 字段。

        用假检索后端时不连真实 Dify 服务，状态保持 offline；
        连真实 Dify 时按结果发 ready 或 degraded。
        """
        if provider is None or provider_name == "mock":
            await _publish_dify(DIFY_STATE_OFFLINE)
            return
        await _publish_dify(DIFY_STATE_READY if ok else DIFY_STATE_DEGRADED)

    async def _retrieve(query: str) -> tuple[str | None, list[dict]]:
        """发起检索，返回上下文文本与来源条目。

        :param query: 检索查询，用题面。
        """
        if provider is None:
            # Dify 没配好：按没取到处理，状态栏的 DIFY 标成 offline
            await _publish_dify_state(False)
            return None, []
        try:
            outcome = await provider.retrieve_outcome(query)
        except Exception as exc:
            # 兜底：检索出问题不让问题处理崩掉
            log_event(
                "retrieval.failed",
                f"检索失败：{exc}",
                level="WARNING",
                error=str(exc),
            )
            await _publish_dify_state(False)
            return None, []
        if outcome.degraded or not outcome.ok:
            log_event(
                "retrieval.degraded",
                f"检索降级：{outcome.error_detail}",
                level="WARNING",
                error_code=outcome.error_code,
            )
            await _publish_dify_state(False)
            return None, []
        if not outcome.results:
            # 没检索到相关内容：按没取到处理，请求照常发出
            await _publish_dify_state(True)
            return None, []
        context = builder.build(outcome.results)
        sources = [_source_entry(item) for item in outcome.results]
        await _publish_dify_state(True)
        return (context or None), sources

    def _on_sources(sources: list[dict]) -> None:
        """把来源条目静默存进界面的当前会话。

        检索完成时来源只存下来，不弹浮层；浮层由用户按 F3 时才打开，
        见《6-Dify检索与项目知识》检索来源展示一节。

        界面还没有这个方法时退回只存不显示之外的旧接口，
        面板没挂上时只记日志，不抛异常，也不让装配失败。
        """
        handler = getattr(app, "store_rag_sources", None)
        if handler is None:
            # 旧界面没有只存不显示的入口，退回会弹浮层的那一个：
            # 来源照样存进会话，代价是浮层会跟着弹出来
            handler = getattr(app, "show_rag_sources", None)
        if handler is None:
            log_event(
                "retrieval.sources_panel_missing",
                "界面还没有检索来源入口，来源条目先记日志",
                level="WARNING",
                source_count=len(sources),
            )
            return
        try:
            handler(sources)
        except Exception as exc:
            log_event(
                "retrieval.sources_notify_failed",
                f"检索来源送不到界面：{exc}",
                level="WARNING",
                error=str(exc),
            )

    return RetrievalChain(
        provider=provider,
        provider_name=provider_name,
        retrieve=_retrieve,
        on_sources=_on_sources,
    )


def _source_entry(result: RetrievalResult) -> dict[str, Any]:
    """把一条检索片段变成来源条目。

    字段是 document_name、score、content，
    见《6-Dify检索与项目知识》结果整理一节。
    """
    return {
        "document_name": result.document_name,
        "score": result.score,
        "content": result.content,
    }


async def close_retrieval(retrieval: RetrievalChain | None) -> None:
    """关闭检索实现底下的 HTTP 客户端，主进程收尾时调用。

    关不掉只记日志，不影响退出码。

    :param retrieval: 检索链路。
    """
    if retrieval is None or retrieval.provider is None:
        return
    closer = getattr(retrieval.provider, "aclose", None)
    if closer is None:
        return
    try:
        await closer()
    except Exception as exc:
        log_event(
            "main.warn",
            f"关闭检索客户端失败：{exc}",
            level="WARNING",
        )


# ----------------------------------------------------------------------
# 语音装配
# ----------------------------------------------------------------------
@dataclass
class VoiceChain:
    """语音链路的装配结果。

    :param audio_manager: 语音采集管理器。
    :param hotkey: F8 控制器，socket 的 ``voice-toggle`` 动作走它。
    :param engine: ASR 引擎。
    :param source: 音频来源配置，取值见 ``audio.source``。
    :param asr_provider: 配置里 ``providers.asr`` 的取值，mock 或 funasr。
    :param submit_tasks: 语音文本自动提交的任务，收尾与测试用它认领任务。
    """

    audio_manager: AudioManager
    hotkey: HotkeyController
    engine: AsrEngine
    source: str
    asr_provider: str = "funasr"
    submit_tasks: list[asyncio.Task[None]] = field(default_factory=list)


class _MockAsrEngine:
    """不加载模型的假 ASR 引擎。

    ``providers.asr=mock`` 时用它：开发期与验收用例不在第一次按 F8 时
    真去加载模型，采集照常开关，识别不出文字。
    """

    def __init__(self) -> None:
        self._state: str = STATE_LOADING

    @property
    def state(self) -> str:
        """引擎状态，取 ``loading``、``ready`` 或 ``error``。"""
        return self._state

    async def ensure_loaded(self) -> None:
        """加载模型。假引擎不加载，直接置成 ready。"""
        self._state = STATE_READY

    async def feed(self, block: Any) -> str:
        """喂一块音频。假引擎不识别，返回空串。"""
        return ""

    async def flush(self) -> str:
        """冲刷。假引擎没有文本，返回空串。"""
        return ""


def select_asr_engine(config: AppConfig) -> AsrEngine:
    """按 ``providers.asr`` 选 ASR 引擎。

    ``funasr`` 用本地流式引擎，模型第一次用时加载；
    ``mock`` 用不加载模型的假引擎，开发期不在第一次按 F8 时真去加载模型。

    引擎的初始化超时显式给配置里的 ``asr.init_timeout``，
    进程级配置只当兜底，见《8-集成与验收》超时管理一节。

    :param config: 应用配置。
    """
    if config.providers.asr != "funasr":
        return _MockAsrEngine()
    return FunASREngine(
        asr_model=config.asr.model,
        device=config.asr.device,
        vad_enabled=config.asr.vad_enabled,
        init_timeout=config.asr.init_timeout,
    )


async def preload_asr(
    config: AppConfig,
    engine: AsrEngine,
    state: MainState | None = None,
) -> str:
    """按配置预加载 ASR 模型，对应 ``start`` 的第 9 步。

    加载在后台跑，不阻塞界面起来；加载失败只记日志，ASR 状态标成 ``error``，
    程序照常往下走，见《2-骨架与配置》命令行入口一节第 9 步。

    引擎按 ``providers.asr`` 选真实实现，不用假引擎顶替；
    预加载与语音链路共用同一个实例，模型只加载一次。

    :param config: 应用配置。
    :param engine: ASR 引擎。
    :param state: 主进程状态，ASR 字段由它发布。
    :returns: 预加载之后的 ASR 状态。
    """
    if not config.asr.enabled or not config.asr.preload:
        log_event(
            "main.asr.preload_skipped",
            "配置没有要求预加载 ASR",
            level="DEBUG",
            asr_enabled=config.asr.enabled,
            preload=config.asr.preload,
        )
        return ASR_STATE_IDLE
    if state is not None:
        state.asr_state = STATE_LOADING
        await state.publish_status()
    try:
        await engine.ensure_loaded()
    except Exception as exc:
        log_event(
            "main.asr.preload_failed",
            f"ASR 预加载失败：{exc}",
            level="WARNING",
            error=str(exc),
            code="E_ASR_LOAD",
        )
        if state is not None:
            state.asr_state = STATE_ERROR
            await state.publish_status()
        return STATE_ERROR
    asr_state = getattr(engine, "state", STATE_READY)
    if state is not None:
        state.asr_state = asr_state
        await state.publish_status()
    log_event("main.asr.preloaded", "ASR 预加载完成", asr_state=asr_state)
    return asr_state


def capture_factory(config: AppConfig) -> CaptureFactory:
    """造采集器工厂。

    参数是来源名，返回按来源选好设备的采集器，
    见《5-语音输入与流式ASR》音频来源一节。

    :param config: 应用配置。
    """
    backend = get_platform().audio

    def _factory(source: str) -> Capture:
        """按来源选设备并造出采集器。"""
        selection = resolve_source(backend, source)
        return create_capture(
            backend,
            selection,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            block_duration_ms=config.audio.chunk_duration_ms,
        )

    return _factory


def assemble_voice(
    config: AppConfig,
    app: Any,
    manager: QuestionManager,
    *,
    state: MainState | None = None,
    engine: AsrEngine | None = None,
    factory: CaptureFactory | None = None,
    router: InputRouter | None = None,
) -> VoiceChain:
    """装配语音链路：采集器、ASR 引擎与最终文本的自动提交。

    停止采集后最终文本写进会话并自动提交，不需要用户再按回车，
    见《5-语音输入与流式ASR》F8 交互一节。部分结果只作临时显示，不提交。

    最终文本走输入路由器：截图还在采集时两段材料合成一次提交，
    没有截图在路上时这段语音文字立即提交，纯语音路径不等待，
    见《4-截图输入与多模态》与语音合并一节与《8-集成与验收》三种输入的汇合一节。

    设备初始化的超时显式给配置里的 ``audio.init_timeout``，
    进程级配置只当兜底，见《8-集成与验收》超时管理一节。

    采集在事件循环里跑：开流与喂模型都不占住调用方。

    界面按 F8 与 socket 的 ``voice-toggle`` 走同一条路：把主进程状态的
    ``toggle_voice_capture`` 挂到界面上，正在采集就停止并提交最终文本，
    没在采集就开始采集，见《5-语音输入与流式ASR》F8 交互一节。

    :param config: 应用配置。
    :param app: Textual 应用实例，部分结果送给它。
    :param manager: 问题管理器，最终文本交给它提交。
    :param state: 主进程状态，最终文本记在它身上。
    :param engine: ASR 引擎，None 时按 ``providers.asr`` 选。
    :param factory: 采集器工厂，None 时按当前平台的音频后端造。
    :param router: 输入路由器，None 时现造一个。
    :returns: 装配好的语音链路。
    """
    submit_tasks: list[asyncio.Task[None]] = []
    asr_provider = config.providers.asr
    if engine is None:
        engine = select_asr_engine(config)
    if factory is None:
        factory = capture_factory(config)
    active_router = router if router is not None else build_input_router(manager)

    async def _on_final_text(text: str, source: str) -> None:
        """语音最终文本：显示到右栏题面，写进会话并自动提交。

        提交即发起一次模型请求，不需要用户再按回车；
        右栏顶部的题面与手动输入提交的一样走 ``show_question``，
        见《5-语音输入与流式ASR》F8 交互一节。
        文本为空时不提交，见《5-语音输入与流式ASR》F8 交互一节。
        """
        stripped = text.strip()
        if not stripped:
            return
        if state is not None:
            state.last_voice_text = stripped
            # 每产出一段最终文本就加一，停止录音时用它判断本次会不会走 route_voice
            state.voice_final_seq += 1
        _show_question(app, stripped)
        # 语音文字记进会话：截图随后到达时与图片合成一条请求，
        # 见《4-截图输入与多模态》与语音合并一节
        manager.set_voice_text(stripped)
        _spawn_submit(lambda: _submit_voice(active_router, stripped), submit_tasks)

    def _on_partial_text(text: str, source: str) -> None:
        """部分结果只作临时显示，不提交。

        见《5-语音输入与流式ASR》部分结果与最终结果一节。
        """
        handler = getattr(app, "show_voice_text", None)
        if handler is None:
            log_event(
                "audio.voice_text_missing",
                "界面还没有临时识别文字行，部分结果只记日志",
                level="WARNING",
            )
            return
        try:
            handler(text)
        except Exception as exc:
            log_event(
                "audio.voice_text_failed",
                f"部分结果送不到界面：{exc}",
                level="WARNING",
                error=str(exc),
            )

    audio_manager = AudioManager(
        factory,
        engine,
        _on_final_text,
        source=config.audio.source,
        on_partial_text=_on_partial_text,
        init_timeout=config.audio.init_timeout,
    )
    hotkey = HotkeyController(audio_manager)
    if state is not None:
        # 界面按 F8 接到 socket 的 voice-toggle 同一条路上：
        # 采集启停、状态发布与最终文本提交都走主进程状态这一份实现
        state.set_input_router(active_router)
        attach = getattr(app, "attach_voice_toggle", None)
        if attach is not None:
            attach(state.toggle_voice_capture)
    return VoiceChain(
        audio_manager=audio_manager,
        hotkey=hotkey,
        engine=engine,
        source=config.audio.source,
        asr_provider=asr_provider,
        submit_tasks=submit_tasks,
    )


def _show_question(app: Any, text: str) -> None:
    """把题面显示到右栏顶部。

    语音最终文本与手动输入提交走同一个入口，
    见《5-语音输入与流式ASR》F8 交互一节。
    界面没有这个方法时只记日志，不抛异常，也不让语音提交失败。

    :param app: Textual 应用实例。
    :param text: 题面文本。
    """
    handler = getattr(app, "show_question", None)
    if handler is None:
        log_event(
            "audio.question_missing",
            "界面还没有题面入口，语音题面只记日志",
            level="WARNING",
        )
        return
    try:
        handler(text)
    except Exception as exc:
        log_event(
            "audio.question_failed",
            f"语音题面送不到界面：{exc}",
            level="WARNING",
            error=str(exc),
        )


async def _submit_voice(router: InputRouter, text: str) -> None:
    """把语音最终文本交给输入路由器。

    语音停止后文本交给路由器，不需要用户再按回车（F8 自动提交语义不变），
    见《5-语音输入与流式ASR》F8 交互一节。截图还在采集时两段材料合成
    一次提交；没有截图在路上时这段语音文字立即提交，
    见《4-截图输入与多模态》与语音合并一节与《8-集成与验收》三种输入的汇合一节。

    :param router: 输入路由器。
    :param text: 语音识别出来的最终文字。
    """
    await router.route_voice(text)


def _spawn_submit(
    make_awaitable: Callable[[], Awaitable[None]],
    tasks: list[asyncio.Task[None]],
) -> None:
    """在后台提交题面，不占住语音停止流程。

    :param make_awaitable: 造出这次提交动作的函数，调用它得到 awaitable。
    :param tasks: 任务列表，提交的任务记在里面。
    """

    async def _runner() -> None:
        try:
            await make_awaitable()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_event(
                "main.voice.submit_failed",
                f"语音文本提交失败：{exc}",
                level="WARNING",
                error=str(exc),
            )

    try:
        task = asyncio.create_task(_runner())
    except RuntimeError as exc:
        # 没有事件循环就提交不了，只记日志
        log_event(
            "main.voice.submit_failed",
            f"语音文本提交失败：{exc}",
            level="WARNING",
            error=str(exc),
        )
        return
    # 只留没收尾的任务，列表不一直长
    tasks[:] = [item for item in tasks if not item.done()]
    tasks.append(task)


def _voice_note(result: str) -> str:
    """把语音动作的结果标识翻成一句人话。

    :param result: 结果标识，见 :meth:`app.audio.audio_manager.AudioManager.toggle`。
    """
    notes = {
        "started": "已开始语音采集",
        "stopped": "已停止语音采集，最终文本已自动提交",
        "empty": "已停止语音采集，没有识别到内容",
        "error": "语音采集失败，原因见日志",
        RESULT_IGNORED: "上一次语音动作还没结束，这次忽略",
    }
    return notes.get(result, result)


async def close_client(client: MockClient | DeepSeekClient) -> None:
    """关闭模型客户端，主进程收尾时调用。

    关不掉只记日志，不影响退出码。

    :param client: 模型客户端。
    """
    try:
        await client.aclose()
    except Exception as exc:
        log_event(
            "main.warn",
            f"关闭模型客户端失败：{exc}",
            level="WARNING",
        )


# ----------------------------------------------------------------------
# 进程装配
# ----------------------------------------------------------------------
# start 前回收旧实例：先等它自己退，超时再发信号
STALE_STOP_TIMEOUT: float = 10.0
STALE_TERMINATE_TIMEOUT: float = 5.0


def _wait_for_server_exit(timeout: float, interval: float = 0.2) -> bool:
    """等通信端点上的主进程退出，超时前退出返回 True。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_server_alive(SOCKET_PATH):
            return True
        time.sleep(interval)
    return not is_server_alive(SOCKET_PATH)


def _wait_for_instance_exit(
    pid: int | None, timeout: float, interval: float = 0.2
) -> bool:
    """等旧实例整个退出，超时前退出返回 True。

    有 PID 时等进程结束，比只等 socket 关掉准确：进程收尾时 socket 先关，
    还要继续停采集、关客户端，这段时间不能再给它发信号。
    没有 PID 时退回看 socket。
    """
    if pid is None:
        return _wait_for_server_exit(timeout, interval)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_process_running(pid):
            return True
        time.sleep(interval)
    return not is_process_running(pid)


def _running_instance_pid() -> int | None:
    """问正在运行的主进程要自己的 PID，问不到返回 None。"""
    try:
        response = send_request("status", socket_path=SOCKET_PATH)
    except SocketError:
        return None
    if not response.ok:
        return None
    pid = response.payload.get("pid")
    if isinstance(pid, int) and pid > 0:
        return pid
    return None


def _format_pids(pids: Iterable[int]) -> str:
    """把 PID 集合拼成 ``123、456`` 的形式。"""
    return "、".join(str(pid) for pid in pids)


def _cleanup_runtime_files() -> list[str]:
    """删掉残留的 PID 文件与通信端点文件，返回删掉的文件路径。"""
    removed: list[str] = []
    if PID_PATH.exists():
        remove_pid(PID_PATH)
        removed.append(str(PID_PATH))
    if SOCKET_PATH.exists():
        get_platform().transport(SOCKET_PATH).cleanup_endpoint()
        removed.append(str(SOCKET_PATH))
    return removed


def stop_stale_instance() -> bool:
    """回收上一次留下的主进程，返回能否继续启动。

    顺序：通信端点上还有活进程就先发 ``stop`` 并等它整个退出；没退出的，
    按 PID 文件与 ``psutil`` 扫出的本应用进程发 ``terminate()``，超时再
    ``kill()``；最后清掉残留的 PID 与端点文件。

    PID 文件里的 PID 可能被系统复用，不是本应用的进程一概不结束，
    只把 PID 文件当残留文件处理。回收后端点仍被占用或者还有本应用进程
    活着时返回 False，调用方放弃启动。

    :returns: 可以继续启动返回 True。
    """
    stopped_pid: int | None = None
    graceful = False

    if is_server_alive(SOCKET_PATH):
        stopped_pid = _running_instance_pid()
        where = f"PID {stopped_pid}" if stopped_pid is not None else "PID 未知"
        print(f"检测到上次的主进程还在运行（{where}），先停掉它", flush=True)
        log_event("main.reclaim", "start 前发现上次的主进程", pid=stopped_pid)
        try:
            response = send_request("stop", socket_path=SOCKET_PATH)
        except SocketError as exc:
            response = None
            print(f"通知上次的主进程停止失败：{exc}", flush=True)
            log_event(
                "main.reclaim",
                "通知上次的主进程停止失败",
                level="WARNING",
                error=str(exc),
            )
        if response is not None and not response.ok:
            log_event(
                "main.reclaim",
                "上次的主进程拒绝停止",
                level="WARNING",
                code=response.error,
            )
        graceful = _wait_for_instance_exit(stopped_pid, STALE_STOP_TIMEOUT)
        if graceful:
            print("上次的主进程已正常退出", flush=True)
            log_event("main.reclaim", "上次的主进程已正常退出", pid=stopped_pid)
        else:
            print("上次的主进程没有自己退出，改为发送终止信号", flush=True)

    candidates: set[int] = set()
    pid = read_pid(PID_PATH)
    if pid is not None and is_process_running(pid):
        if graceful and pid == stopped_pid:
            # 正常退出中的收尾还没走完，不再补刀
            pass
        elif is_app_process(pid):
            candidates.add(pid)
        else:
            print(
                f"PID 文件里的进程 {pid} 不是本应用，按残留文件处理，不结束它",
                flush=True,
            )
            log_event(
                "main.reclaim",
                "PID 文件里的进程不是本应用",
                level="WARNING",
                pid=pid,
            )

    for found in find_app_processes():
        if graceful and found == stopped_pid:
            continue
        candidates.add(found)

    if stopped_pid is not None and not graceful and is_process_running(stopped_pid):
        candidates.add(stopped_pid)

    if candidates:
        report = terminate_processes(candidates, timeout=STALE_TERMINATE_TIMEOUT)
        if report.terminated:
            print(
                f"已结束上次的主进程：{_format_pids(report.terminated)}", flush=True
            )
        if report.killed:
            print(
                f"强制结束上次的主进程：{_format_pids(report.killed)}", flush=True
            )
        if report.survived:
            print(f"有进程没结束：{_format_pids(report.survived)}", flush=True)
        log_event(
            "main.reclaim",
            "回收上次的主进程",
            terminated=len(report.terminated),
            killed=len(report.killed),
            survived=len(report.survived),
        )

    if is_server_alive(SOCKET_PATH) or find_app_processes():
        print("上次的主进程还在占用资源，放弃启动", flush=True)
        log_event("main.reclaim", "旧实例没有停下来", level="ERROR")
        return False

    removed = _cleanup_runtime_files()
    if removed:
        print(f"已清理残留文件：{'、'.join(removed)}", flush=True)
        log_event("main.reclaim", "已清理残留文件", removed=len(removed))
    return True


async def run_main(config: AppConfig) -> int:
    """装配主进程并跑起来。

    顺序：写 PID 文件 → 建 socket 服务端并监听 → 初始化模型客户端并探测能力 →
    启动 Textual 界面 → 退出时收尾。
    socket 文件被活进程占用时以退出码 1 结束。

    :param config: 应用配置。
    :returns: 退出码，见《2-骨架与配置》退出码表。
    """
    # 界面依赖只在主进程里导入：命令行侧的其它命令用不到
    from app.ui.app import InterviewMateApp

    # PID 文件里的进程还活着就不重复启动，见《2-骨架与配置》进程通信一节
    pid = read_pid(PID_PATH)
    if pid is not None and is_process_alive(pid):
        print(f"主进程已在运行（PID {pid}），不重复启动", flush=True)
        log_event("main.error", "主进程已在运行", level="ERROR", pid=pid)
        return EXIT_ERROR

    state = MainState(voice_toggle_debounce_seconds=VOICE_TOGGLE_DEBOUNCE_SECONDS)
    state.attach_config(config)
    server = SocketServer(SOCKET_PATH, build_handlers(state))
    log_event("main.start", "主进程启动", pid=os.getpid(), socket=str(SOCKET_PATH))
    try:
        await server.start()
    except SocketError as exc:
        print(f"E_SOCKET：{exc.message}", flush=True)
        log_event(
            "main.error", "socket 监听启动失败", level="ERROR", error=str(exc)
        )
        return EXIT_ERROR

    # 监听起来了再写 PID 文件：启动失败时不碰活进程留下的 PID 文件
    write_pid(PID_PATH)
    app = InterviewMateApp(config=config, bus=EventBus())
    state.attach_app(app)

    # 第 7 步：初始化模型客户端并探测模型能力，失败只告警不退出
    client = select_client(config)
    state.set_model_client(client)
    capability = await probe_model_capability(client, config)
    if capability.ok:
        log_event("main.model", "模型能力探测完成", detail=capability.detail)
    else:
        log_event(
            "main.model.warn",
            "模型能力探测失败",
            level="WARNING",
            code=capability.code,
            detail=capability.detail,
        )
        # 探测失败时状态栏的 DEEPSEEK 落到 error，见《2-骨架与配置》状态字段取值表
        state.set_deepseek_state(MODEL_STATE_ERROR)

    # 检索链路，界面起来之后与问题管理器一起装
    retrieval: RetrievalChain | None = None

    # 全局热键监听，界面起来之后装配，收尾时最先关掉
    hotkey_listener: HotkeyListener | None = None

    # 第 9 步：按配置预加载 ASR。加载在后台跑，不阻塞界面起来，
    # 失败只记日志、ASR 状态标成 error，程序照常往下走，
    # 见《2-骨架与配置》命令行入口一节第 9 步。
    # 引擎在这里造一次，语音链路复用同一个实例，模型只加载一次
    asr_engine = select_asr_engine(config)
    asr_preload_task = asyncio.create_task(preload_asr(config, asr_engine, state))

    async def _assemble(_pilot: Any) -> None:
        """界面起来之后装配问题管理器、检索、语音与截图管理器。

        事件分发器要界面上的导航面板与内容面板，面板要等界面挂载后才拿得到，
        提示位回调也要界面上的状态栏，所以这一步放在界面起来之后做。
        """
        nonlocal retrieval, hotkey_listener
        retrieval = assemble_retrieval(config, app, state=state)
        state.set_retrieval_chain(retrieval)
        manager = assemble_manager(
            config,
            client,
            app,
            retrieve=retrieval.retrieve,
            on_sources=retrieval.on_sources,
            state=state,
        )
        # 三种输入汇到同一个路由器：手工输入、语音、截图都走它，
        # 见《8-集成与验收》三种输入的汇合一节
        router = build_input_router(manager)
        voice = assemble_voice(
            config,
            app,
            manager,
            state=state,
            engine=asr_engine,
            router=router,
        )
        state.set_voice_chain(voice)
        screenshot_manager = assemble_screenshot_manager(
            config,
            app,
            manager,
            router=router,
        )
        state.set_screenshot_manager(screenshot_manager)
        # 手工输入回车也走路由器
        app.attach_input_router(router)
        # 全局热键最后接上：路由与截图管理器都就绪了，按下时链路已经能跑，
        # 见《5-语音输入与流式ASR》F8 交互一节
        hotkey_listener = assemble_global_hotkeys(config, state, app)
        # 界面挂好了，把当前状态发一次：ASR 预加载可能在界面起来之前就结束，
        # 那次状态事件没有订阅者，状态栏要在这里补上
        await state.publish_status()

    try:
        if state.stop_requested:
            # stop 在界面起来之前就到了：不启界面，直接走收尾
            log_event("main.stop", "启动期间收到 stop 请求，不启动界面")
        else:
            # 界面起来后终端归 Textual 所有，第三方库（ASR 加载等）打给
            # stdout/stderr 的提示一律接进日志文件，见 app.utils.logger
            log_target = current_log_path()
            with (
                redirect_console(log_target)
                if log_target is not None
                else contextlib.nullcontext()
            ):
                await app.run_async(auto_pilot=_assemble)
    except Exception as exc:
        print(f"界面运行失败：{exc}", flush=True)
        log_event("main.error", "界面运行失败", level="ERROR", error=str(exc))
        return EXIT_ERROR
    finally:
        # 收尾顺序见《2-骨架与配置》进程通信一节与 WIN-MILESTONES W2.5：
        # 停采集 → 取消模型任务 → 结束 ASR 流 → 关 HTTP → 删端口/token 文件 → 删 PID 文件。
        # 先关全局热键：收尾过程中不再收到按键，重复 stop 安全
        if hotkey_listener is not None:
            hotkey_listener.stop()
        # 停采集并冲刷 ASR 流：AudioManager.stop() 会先停采集再结束识别流，
        # HTTP 客户端才能随后关掉
        await state.stop_voice()
        # ASR 还在加载时收尾：取消模型加载任务，不留悬挂的任务
        if not asr_preload_task.done():
            asr_preload_task.cancel()
            await asyncio.gather(asr_preload_task, return_exceptions=True)
        await close_client(client)
        await close_retrieval(retrieval)
        # 先删端口/token 文件（传输层收掉监听），再删 PID 文件：
        # 命令行侧的 stop 轮询端口是否还在，最后一步落定 PID 文件已清理
        await server.stop()
        remove_pid(PID_PATH)
        # 收尾时把这一轮跑下来的指标统计打成一条日志
        _log_metric_stats()
        log_event("main.stop", "主进程收尾完成")
    return EXIT_OK


def _stdout_isatty() -> bool:
    """stdout 是不是交互式终端。取不到时按不是处理。"""
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


async def start_process(
    config_path: str | None = None,
    env_file: str | None = None,
) -> int:
    """``start`` 的完整流程。

    步骤顺序见《2-骨架与配置》命令行入口一节：
    读配置校验 → 查依赖包 → 查音频设备（只告警）→ 查截图接口（只告警）→
    建事件循环 → 建 socket 并监听 → 初始化模型客户端并探测能力（只告警）→
    启动界面 → 退出时收尾。

    ``providers.llm=deepseek`` 时缺 ``DEEPSEEK_API_KEY`` 以退出码 3 结束；
    ``providers.llm=mock`` 时不查这个变量，缺了照常启动。

    :param config_path: 配置文件路径，None 时用仓库根的 ``config/config.yaml``。
    :param env_file: 环境变量文件路径，None 时用仓库根的 .env。
    :returns: 退出码，见《2-骨架与配置》退出码表。
    """
    try:
        config = load_app_config(config_path, env_file)
    except ConfigError as exc:
        print(f"E_CONFIG：{exc.message}", flush=True)
        log_event("config.error", "配置读取失败", level="ERROR", error=exc.message)
        return EXIT_CONFIG_ERROR

    log_path = setup_logging(
        config.app.log_level,
        filename_template=TUI_LOG_FILENAME,
        mirror_filename_template=LOG_FILE_TEMPLATE,
        console=False,
        redact=config.privacy.redact_secrets_in_logs,
    )

    # 配置指定的截图后端必须与当前平台的后端一致，见《2-骨架与配置》配置结构一节
    platform = get_platform()
    configured_screenshot = config.providers.screenshot
    if (
        configured_screenshot != "mock"
        and configured_screenshot != platform.screenshot.name
    ):
        message = (
            f"配置指定的截图后端 {configured_screenshot} 与当前平台的后端 "
            f"{platform.screenshot.name} 不一致"
        )
        print(f"E_CONFIG：{message}", flush=True)
        log_event(
            "config.error",
            message,
            level="ERROR",
            configured=configured_screenshot,
            platform=platform.screenshot.name,
        )
        return EXIT_CONFIG_ERROR

    # 必填环境变量只在 providers.llm 指向真实实现时检查：
    # mock 模式缺 DEEPSEEK_API_KEY 照常启动，见《2-骨架与配置》命令行入口一节第 1 步
    if config.providers.llm == "deepseek":
        missing_env = missing_required_env(env_file)
        if missing_env:
            names = "、".join(missing_env)
            print(f"缺少必填环境变量：{names}", flush=True)
            log_event(
                "config.error", "缺少必填环境变量", level="ERROR", missing=names
            )
            return EXIT_CONFIG_ERROR

    # 依赖检查要真导入一遍包，funasr 导入时会 print ffmpeg 通知、
    # 还会调 logging.basicConfig()；终端输出已由 setup_logging 关掉，
    # 这里把 print 的提示也接进日志文件，不让它打扰后面的界面
    with redirect_console(log_path):
        missing_packages = check_dependencies(required_packages(config))
    if missing_packages:
        names = "、".join(missing_packages)
        print(f"缺少依赖包：{names}", flush=True)
        log_event("main.error", "缺少依赖包", level="ERROR", missing=names)
        return EXIT_ERROR

    # 平台后端的探针同样要导入第三方库，提示一并接进日志文件
    with redirect_console(log_path):
        audio = probe_audio_devices()
        screenshot = await probe_screenshot_interface()
    if not audio.ok:
        log_event(
            "audio.warn",
            "音频设备不可用",
            level="WARNING",
            code=audio.code,
            detail=audio.detail,
        )

    if not screenshot.ok:
        log_event(
            "screenshot.warn",
            "截图接口不可用",
            level="WARNING",
            code=screenshot.code,
            detail=screenshot.detail,
        )

    if not _stdout_isatty():
        print("当前不是交互式终端，无法启动界面", flush=True)
        log_event(
            "main.error", "当前不是交互式终端，无法启动界面", level="ERROR"
        )
        return EXIT_ERROR

    # 前面的检查都过了才回收旧实例：配置写错或没有终端时不动正在跑的进程
    if not stop_stale_instance():
        print(
            "上次的主进程没有停下来，start 放弃；可先执行 interviewmate.cmd stop",
            flush=True,
        )
        log_event("main.reclaim", "旧实例没停下来，放弃启动", level="ERROR")
        return EXIT_ERROR

    return await run_main(config)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m app.main`` 的入口：跑主进程。

    参数只有 ``--config`` 与 ``--env-file``，与命令行入口的全局参数一致。

    :param argv: 参数列表，None 时取 ``sys.argv[1:]``。
    :returns: 退出码，见《2-骨架与配置》退出码表。
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="启动 InterviewMate 主进程",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="配置文件路径，默认 config/config.yaml",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="环境变量文件路径，默认用仓库根的 .env",
    )
    args = parser.parse_args(argv)
    setup_logging()
    return asyncio.run(start_process(args.config, args.env_file))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
