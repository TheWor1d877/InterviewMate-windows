"""主进程装配的测试。

覆盖 M4/M5 装起来的两条链路：

- 检索：按 ``providers.retrieval`` 选实现，检索成功时上下文进模型请求、
  来源条目送到界面；检索失败或 Dify 没配好时不崩掉，请求照常发，
  状态栏的 DIFY 变 degraded，见《6-Dify检索与项目知识》检索流程与故障降级两节；
- 语音：socket 的 ``voice-toggle`` 动作真的开流与关流，停止后最终文本自动提交，
  部分结果只作临时显示，见《5-语音输入与流式ASR》F8 交互与部分结果和最终结果两节；
- 收尾：``stop`` 动作先停音频采集再退出界面，
  见《2-骨架与配置》进程通信一节；
- 命令行：``devices`` 标出系统回环与麦克风，
  见《9-平台差异与Windows适配》设备枚举一节。

检索、模型客户端、采集器与 ASR 引擎都用假实现，不连真实服务也不开真流。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

import app.cli
import app.config
import app.main
from app.audio.funasr_engine import FunASREngine
from app.config import AppConfig
from app.llm.deepseek_client import DeepSeekClient
from app.llm.question_classifier_client import DeepSeekQuestionClassifier
from app.main import EXIT_OK
from app.platform import get_platform
from app.platform.types import AudioDevice
from app.retrieval.dify_retriever import DifyRetriever, TransportError
from app.runtime.socket_protocol import Request

# 题面样例：命中项目关键词，会触发检索
QUESTION_PROJECT = "讲讲我的项目是怎么做的"


# ----------------------------------------------------------------------
# 假实现
# ----------------------------------------------------------------------
class FakePanel:
    """假面板，方法都是空实现。"""

    def show_framework(self, sections: list[dict]) -> None:
        return None

    def mark_section_done(self, section_id: str) -> None:
        return None

    def append_item(self, section_id: str, keyword: str, sentence: str) -> None:
        return None

    def append_fallback(self, text: str) -> None:
        return None

    def clear(self) -> None:
        return None


class FakeApp:
    """假界面，只实现装配层用到的那几个方法。"""

    def __init__(self) -> None:
        self.navigation_panel = FakePanel()
        self.answer_panel = FakePanel()
        self.errors: list[tuple[str, str]] = []
        self.voice_texts: list[str] = []
        self.rag_sources: list[list[dict]] = []
        self.statuses: list[dict] = []
        self.events: list[str] = []
        self.manager: Any = None
        # 部分结果送到界面时置位，测试用它等界面收到
        self.voice_text_event = asyncio.Event()

    def show_error(self, code: str, detail: str) -> None:
        self.errors.append((code, detail))

    def attach_manager(self, manager: Any) -> None:
        self.manager = manager

    def show_voice_text(self, text: str) -> None:
        self.voice_texts.append(text)
        self.voice_text_event.set()

    def show_rag_sources(self, entries: list[dict]) -> None:
        self.rag_sources.append(list(entries))

    async def publish_status(self, **fields: Any) -> None:
        self.statuses.append(dict(fields))

    def exit(self) -> None:
        self.events.append("app.exit")


class FakeClient:
    """假模型客户端，记录收到的请求消息。"""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def chat_stream(self, messages: list[dict[str, str]], request_id: str):
        self.calls.append(messages)
        yield "回答"

    async def multimodal_stream(self, messages: list[dict], request_id: str):
        yield "回答"

    async def cancel(self, request_id: str) -> None:
        return None


class FakeClassifierClient(FakeClient):
    """带非流式补全的假模型客户端，记录判定请求。"""

    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply = reply
        self.complete_calls: list[dict[str, Any]] = []

    async def complete_json(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        self.complete_calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return self.reply


class FakeManager:
    """假问题管理器，记录语音文字与提交。"""

    def __init__(self) -> None:
        self.voice_texts: list[str] = []
        self.submissions: list[str] = []

    def set_voice_text(self, text: str) -> None:
        self.voice_texts.append(text)

    async def submit_text(self, text: str, image_data_url: str | None = None) -> None:
        self.submissions.append(text)


class FakeBlock:
    """假音频块，只带来源标签。"""

    source = "microphone"


class FakeCapture:
    """假采集器，记录开流与关流。"""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.started = 0
        self.stopped = 0
        self._running = False
        self._blocks = 0

    @property
    def is_running(self) -> bool:
        """采集是不是开着。"""
        return self._running

    async def start(self) -> None:
        self.events.append("capture.start")
        self.started += 1
        self._running = True

    async def stop(self) -> None:
        self.events.append("capture.stop")
        self.stopped += 1
        self._running = False

    async def get_block(self) -> FakeBlock:
        """给一块音频，之后一直等着，等采集停掉时被取消。"""
        self._blocks += 1
        if self._blocks > 1:
            await asyncio.sleep(3600)
        return FakeBlock()


class FakeEngine:
    """假 ASR 引擎，喂块返回部分结果，冲刷返回剩下的文本。"""

    def __init__(self, partial: str = "你好", final: str = "，请介绍一下项目") -> None:
        self.partial = partial
        self.final = final
        self.loaded = 0
        self._state = "loading"
        # 喂过第一块时置位，测试用它等识别真的开始
        self.fed = asyncio.Event()

    @property
    def state(self) -> str:
        """引擎状态。"""
        return self._state

    async def ensure_loaded(self) -> None:
        self.loaded += 1
        self._state = "ready"

    async def feed(self, block: Any) -> str:
        self.fed.set()
        return self.partial

    async def flush(self) -> str:
        return self.final


class FailingTransport:
    """连不上的传输层，检索必然降级。"""

    async def post(self, url: str, payload: dict, headers: dict):
        raise TransportError("连不上 Dify")


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _config(
    retrieval: str = "mock", *, llm: str = "mock", **sections: Any
) -> AppConfig:
    """按 ``providers.retrieval`` 与若干顶层段造一份配置。

    四键先显式设为 mock，只覆盖 ``retrieval`` 与 ``llm``：
    本文件的装配用例都按 mock 行为断言，默认配置已切到真实实现，
    不能让它漏进来。
    """
    data = yaml.safe_load(
        app.config.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    )
    data["providers"] = {
        "llm": llm,
        "retrieval": retrieval,
        "asr": "mock",
        "screenshot": "mock",
    }
    for section, value in sections.items():
        data[section] = value
    return AppConfig.model_validate(data)


def _voice_chain(
    ui: FakeApp,
    manager: FakeManager,
    state: app.main.MainState,
    capture: FakeCapture,
    engine: FakeEngine,
) -> app.main.VoiceChain:
    """装配一条用假采集器与假引擎的语音链路。"""
    voice = app.main.assemble_voice(
        _config(),
        ui,
        manager,
        state=state,
        engine=engine,
        factory=lambda source: capture,
    )
    state.set_voice_chain(voice)
    return voice


@contextmanager
def _collect_log_events():
    """收集项目记录器上的日志，用来断言事件名。"""
    logger = logging.getLogger("interviewmate")
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


# ----------------------------------------------------------------------
# 默认配置的选型：四键默认真实实现，选型不连网络、不加载模型
# ----------------------------------------------------------------------
def test_default_config_uses_real_providers() -> None:
    """默认配置里四个 provider 都是真实取值。"""
    config = app.config.load_config()

    assert config.providers.llm == "deepseek"
    assert config.providers.retrieval == "dify"
    assert config.providers.asr == "funasr"
    assert config.providers.screenshot == "gnome"


async def test_default_config_selects_deepseek_client() -> None:
    """默认配置选出的模型客户端是 DeepSeekClient。

    只构造不探测：构造不发网络请求，也不加载模型。
    """
    config = app.config.load_config()

    client = app.main.select_client(config)

    try:
        assert isinstance(client, DeepSeekClient)
    finally:
        await app.main.close_client(client)


async def test_default_config_selects_dify_retriever() -> None:
    """默认配置选出的检索实现是 DifyRetriever。

    ``dify`` 段的占位值非空，按当前设计就当作已配置，不连网络。
    """
    config = app.config.load_config()

    retriever = app.main.select_retriever(config)

    try:
        assert isinstance(retriever, DifyRetriever)
    finally:
        if retriever is not None:
            await retriever.aclose()


def test_default_config_selects_funasr_engine() -> None:
    """默认配置选出的 ASR 引擎是 FunASREngine，构造不加载模型。"""
    config = app.config.load_config()

    engine = app.main.select_asr_engine(config)

    assert isinstance(engine, FunASREngine)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="gnome 后端只在 Linux 上"
)
def test_default_config_selects_gnome_screenshot_backend() -> None:
    """默认配置的截图后端与 Linux 平台后端一致，都是 gnome。"""
    config = app.config.load_config()

    assert get_platform().screenshot.name == "gnome"
    assert config.providers.screenshot == "gnome"


# ----------------------------------------------------------------------
# 检索实现的选取
# ----------------------------------------------------------------------
def test_select_retriever_picks_mock_implementation() -> None:
    """providers.retrieval=mock 时选假检索后端。"""
    retriever = app.main.select_retriever(_config(retrieval="mock"))

    assert isinstance(retriever, app.main.MockRetriever)


def test_select_retriever_picks_dify_implementation() -> None:
    """providers.retrieval=dify 时选 Dify，连接参数取配置里的值。"""
    config = _config(retrieval="dify")

    retriever = app.main.select_retriever(config)

    assert isinstance(retriever, DifyRetriever)
    assert retriever.base_url == config.dify.base_url.rstrip("/")
    assert retriever.dataset_id == config.dify.dataset_id
    assert retriever.top_k == config.rag.top_k


def test_select_retriever_returns_none_when_dify_config_missing() -> None:
    """Dify 配置缺项时不崩，退化成不可用。"""
    config = _config(
        retrieval="dify",
        dify={"base_url": "", "dataset_id": "", "api_key": ""},
    )

    retriever = app.main.select_retriever(config)

    assert retriever is None


async def test_retrieve_returns_nothing_when_dify_is_unavailable() -> None:
    """Dify 没配好时检索回调返回「没取到」，DIFY 标成 offline。"""
    config = _config(
        retrieval="dify",
        dify={"base_url": "", "dataset_id": "", "api_key": ""},
    )
    state = app.main.MainState()
    chain = app.main.assemble_retrieval(config, FakeApp(), state=state)

    context, sources = await chain.retrieve(QUESTION_PROJECT)

    assert context is None
    assert sources == []
    assert state.dify_state == "offline"


# ----------------------------------------------------------------------
# 检索成功与降级
# ----------------------------------------------------------------------
async def test_retrieval_context_reaches_the_request_and_sources_are_sent() -> None:
    """检索成功时上下文进模型请求，来源条目送到界面的检索来源面板。"""
    ui = FakeApp()
    config = _config(retrieval="mock")
    state = app.main.MainState()
    chain = app.main.assemble_retrieval(config, ui, state=state)
    client = FakeClient()
    manager = app.main.assemble_manager(
        config,
        client,
        ui,
        retrieve=chain.retrieve,
        on_sources=chain.on_sources,
    )

    await manager.submit_text(QUESTION_PROJECT)

    assert len(client.calls) == 1
    content = "\n".join(message["content"] for message in client.calls[0])
    assert "片段 1" in content
    assert ui.rag_sources
    entry = ui.rag_sources[0][0]
    assert set(entry) == {"document_name", "score", "content"}
    assert entry["document_name"]


async def test_retrieval_failure_keeps_the_request_and_marks_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """检索失败时不抛异常，请求照常发，状态栏的 DIFY 变 degraded。"""
    config = _config(retrieval="dify")
    failing = DifyRetriever(
        "http://dify.invalid/v1",
        "dataset",
        "key",
        top_k=config.rag.top_k,
        transport=FailingTransport(),
    )
    monkeypatch.setattr(app.main, "select_retriever", lambda _config: failing)
    ui = FakeApp()
    state = app.main.MainState()
    chain = app.main.assemble_retrieval(config, ui, state=state)
    client = FakeClient()
    manager = app.main.assemble_manager(
        config,
        client,
        ui,
        retrieve=chain.retrieve,
        on_sources=chain.on_sources,
    )

    await manager.submit_text(QUESTION_PROJECT)

    assert len(client.calls) == 1
    # 检索失败时回答里不能声称检索到了项目资料
    content = "\n".join(message["content"] for message in client.calls[0])
    assert "片段" not in content
    assert state.dify_state == "degraded"
    assert ui.rag_sources == []


async def test_sources_callback_survives_missing_panel() -> None:
    """界面还没有检索来源面板时只记日志，不抛异常。"""

    class PanelLessApp:
        """没有检索来源面板的界面。"""

    chain = app.main.assemble_retrieval(_config(retrieval="mock"), PanelLessApp())

    chain.on_sources([{"document_name": "文档", "score": 0.5, "content": "正文"}])


# ----------------------------------------------------------------------
# 语音：voice-toggle 动作与最终文本
# ----------------------------------------------------------------------
async def test_voice_toggle_starts_and_stops_the_capture() -> None:
    """voice-toggle 动作真的开流与关流，不只是切状态字段。"""
    ui = FakeApp()
    manager = FakeManager()
    state = app.main.MainState()
    capture = FakeCapture(ui.events)
    engine = FakeEngine()
    _voice_chain(ui, manager, state, capture, engine)
    state.attach_app(ui)

    payload = await state.toggle_voice_capture()

    assert capture.started == 1
    assert payload["audio"] == "on"
    assert payload["asr"] == "ready"
    assert payload["result"] == "started"

    await engine.fed.wait()
    payload = await state.toggle_voice_capture()

    assert capture.stopped == 1
    assert payload["audio"] == "off"
    assert payload["result"] == "stopped"


async def test_voice_final_text_is_submitted_and_partial_text_is_not() -> None:
    """最终文本自动提交，部分结果只作临时显示、不提交。"""
    ui = FakeApp()
    manager = FakeManager()
    state = app.main.MainState()
    capture = FakeCapture(ui.events)
    engine = FakeEngine()
    voice = _voice_chain(ui, manager, state, capture, engine)

    await state.toggle_voice_capture()
    await engine.fed.wait()
    await ui.voice_text_event.wait()

    # 部分结果只送到临时识别文字行，没有提交
    assert ui.voice_texts == ["你好"]
    assert manager.submissions == []

    await state.toggle_voice_capture()
    for task in voice.submit_tasks:
        await task

    # 语音最终文本立即交给提交入口，没有截图在路上时零等待，
    # 见《4-截图输入与多模态》与语音合并一节
    assert manager.voice_texts == ["你好，请介绍一下项目"]
    assert manager.submissions == ["你好，请介绍一下项目"]
    assert state.last_voice_text == "你好，请介绍一下项目"


async def test_voice_toggle_publishes_audio_and_asr_state() -> None:
    """采集状态与 ASR 状态在动作之后发布出去。"""
    ui = FakeApp()
    manager = FakeManager()
    state = app.main.MainState()
    capture = FakeCapture(ui.events)
    _voice_chain(ui, manager, state, capture, FakeEngine())
    state.attach_app(ui)

    await state.toggle_voice_capture()

    assert ui.statuses
    assert ui.statuses[-1]["audio"] == "on"
    assert ui.statuses[-1]["asr"] == "ready"


# ----------------------------------------------------------------------
# 语音开关去抖：同一次按键被两条路径各触发一次时只切换一次
# ----------------------------------------------------------------------
async def test_voice_toggle_debounces_repeats_within_400ms() -> None:
    """窗口内的第二次 toggle 被丢掉，窗口过去后照常切换。"""
    clock = [1000.0]
    state = app.main.MainState(
        voice_toggle_debounce_seconds=app.main.VOICE_TOGGLE_DEBOUNCE_SECONDS,
        voice_toggle_clock=lambda: clock[0],
    )
    ui = FakeApp()
    manager = FakeManager()
    capture = FakeCapture(ui.events)
    engine = FakeEngine()
    _voice_chain(ui, manager, state, capture, engine)
    state.attach_app(ui)

    with _collect_log_events() as records:
        first = await state.toggle_voice_capture()

        clock[0] += 0.1
        second = await state.toggle_voice_capture()

        # 按住 F8 的键盘重复：每次都在窗口内，窗口起点跟着往后推，
        # 不会每过 400 毫秒又切换一次
        clock[0] += 0.3
        held = await state.toggle_voice_capture()
        clock[0] += 0.3
        held_again = await state.toggle_voice_capture()

    assert app.main.VOICE_TOGGLE_DEBOUNCE_SECONDS == 0.4
    assert first["result"] == "started"
    assert second["result"] == app.main.RESULT_IGNORED
    assert held["result"] == app.main.RESULT_IGNORED
    assert held_again["result"] == app.main.RESULT_IGNORED
    assert capture.started == 1
    assert capture.stopped == 0
    assert state.audio_on is True
    assert any(
        getattr(record, "event", "") == "audio.toggle.debounced"
        for record in records
    )

    # 超过窗口后正常执行：这一次真的停止采集
    await engine.fed.wait()
    clock[0] += 0.401
    third = await state.toggle_voice_capture()

    assert third["result"] == "stopped"
    assert capture.stopped == 1
    assert state.audio_on is False


async def test_voice_toggle_debounce_does_not_swallow_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """切换过程中的异常照常抛出，不被去抖挡掉。"""
    clock = [1000.0]
    state = app.main.MainState(
        voice_toggle_debounce_seconds=app.main.VOICE_TOGGLE_DEBOUNCE_SECONDS,
        voice_toggle_clock=lambda: clock[0],
    )
    ui = FakeApp()
    manager = FakeManager()
    voice = _voice_chain(ui, manager, state, FakeCapture(ui.events), FakeEngine())

    async def _boom() -> str:
        raise RuntimeError("采集器坏了")

    monkeypatch.setattr(voice.hotkey, "handle_voice_toggle", _boom)

    with pytest.raises(RuntimeError):
        await state.toggle_voice_capture()


# ----------------------------------------------------------------------
# 收尾：stop 动作先停采集
# ----------------------------------------------------------------------
async def test_stop_action_stops_the_capture_before_exiting() -> None:
    """stop 动作先把音频采集停掉，再退出界面。"""
    ui = FakeApp()
    manager = FakeManager()
    state = app.main.MainState()
    capture = FakeCapture(ui.events)
    _voice_chain(ui, manager, state, capture, FakeEngine())
    state.attach_app(ui)
    await state.toggle_voice_capture()

    handlers = app.main.build_handlers(state)
    payload = await handlers["stop"](Request(action="stop", request_id="req-1"))

    assert capture.stopped == 1
    assert ui.events[-2:] == ["capture.stop", "app.exit"]
    assert payload == {"stopping": True}


async def test_warmup_reports_mock_markers_when_providers_are_mock() -> None:
    """providers 全 mock 时预热只给 mock 标记。"""
    ui = FakeApp()
    manager = FakeManager()
    state = app.main.MainState()
    state.attach_config(_config())
    state.set_model_client(FakeClient())
    chain = app.main.assemble_retrieval(_config(retrieval="mock"), ui, state=state)
    state.set_retrieval_chain(chain)
    _voice_chain(ui, manager, state, FakeCapture(), FakeEngine())

    payload = await state.warmup_payload()

    assert payload["asr"] == "mock"
    assert payload["deepseek"] == "mock"
    assert payload["dify"] == "mock"


# ----------------------------------------------------------------------
# 命令行：devices 标出系统回环
# ----------------------------------------------------------------------
def test_devices_command_marks_loopback_devices(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """devices 命令在每台设备后面标出是系统回环还是麦克风。"""
    devices = [
        AudioDevice(
            name="alsa_input.pci-0000_01_00.1.analog-stereo",
            index=13,
            is_input=True,
            is_loopback=False,
            default_samplerate=48000.0,
            channels=2,
        ),
        AudioDevice(
            name="alsa_output.pci-0000_01_00.1.hdmi-stereo.monitor",
            index=14,
            is_input=True,
            is_loopback=True,
            default_samplerate=48000.0,
            channels=2,
        ),
    ]
    monkeypatch.setattr(app.cli, "list_input_devices", lambda: devices)

    code = app.cli.main(["devices"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "类型=麦克风" in out
    assert "类型=系统回环" in out
    assert "麦克风 1 个，系统回环 1 个" in out
    monitor_line = [line for line in out.splitlines() if "hdmi-stereo.monitor" in line]
    assert monitor_line and "类型=系统回环" in monitor_line[0]


# ----------------------------------------------------------------------
# 装配层的配置来源
# ----------------------------------------------------------------------
def test_load_app_config_sets_process_config() -> None:
    """装配路径照旧走 load_config：读到的配置就是进程级当前配置。"""
    saved = app.config.current_config()
    try:
        config = app.main.load_app_config()

        assert app.config.current_config() is config
    finally:
        app.config.set_current_config(saved)


async def test_start_process_configures_file_only_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``start`` 的 run 路径把日志改成只追加落盘，不写 stderr。"""
    config = _config()
    calls: list[dict[str, Any]] = []
    log_path = tmp_path / app.main.TUI_LOG_FILENAME

    def _fake_setup(level: str, **kwargs: Any) -> Path:
        calls.append({"level": level, **kwargs})
        log_path.write_text("", encoding="utf-8")
        return log_path

    async def _fake_screenshot_probe() -> Any:
        return app.main.ProbeResult(ok=True, detail="ok")

    async def _fake_run_main(_config: AppConfig) -> int:
        return EXIT_OK

    monkeypatch.setattr(app.main, "load_app_config", lambda *a, **k: config)
    monkeypatch.setattr(app.main, "setup_logging", _fake_setup)
    monkeypatch.setattr(
        app.main, "probe_audio_devices", lambda: app.main.ProbeResult(ok=True, detail="ok")
    )
    monkeypatch.setattr(app.main, "probe_screenshot_interface", _fake_screenshot_probe)
    monkeypatch.setattr(app.main, "_stdout_isatty", lambda: True)
    monkeypatch.setattr(app.main, "run_main", _fake_run_main)

    code = await app.main.start_process()

    assert code == EXIT_OK
    assert calls, "start_process 没有配置日志"
    last = calls[-1]
    assert last["level"] == config.app.log_level
    assert last["filename_template"] == app.main.TUI_LOG_FILENAME == "interviewmate.log"
    assert last["mirror_filename_template"] == app.main.LOG_FILE_TEMPLATE
    assert last["console"] is False
    assert last["redact"] == config.privacy.redact_secrets_in_logs


# ----------------------------------------------------------------------
# 题型判定的装配：mock 不注入，真实 provider 注入模型判定器
# ----------------------------------------------------------------------
async def test_mock_provider_assembles_without_model_classifier() -> None:
    """providers.llm=mock 时问题管理器没有模型判定器，只走规则、不发网络。"""
    ui = FakeApp()
    config = _config(llm="mock")
    client = FakeClient()

    manager = app.main.assemble_manager(config, client, ui)
    await manager.submit_text(QUESTION_PROJECT)

    assert getattr(manager, "_llm_classifier", object()) is None
    # 规则判定照旧：mock 客户端只收到流式请求，没有非流式判定调用
    assert len(client.calls) == 1
    assert not hasattr(client, "complete_json")


async def test_deepseek_provider_assembles_model_classifier() -> None:
    """providers.llm=deepseek 时注入模型判定器，判定结果生效。"""
    ui = FakeApp()
    config = _config(llm="deepseek")
    client = FakeClassifierClient(
        '{"type": "conceptual", "need_rag": false, "reason": "通用知识"}'
    )

    manager = app.main.assemble_manager(config, client, ui)
    await manager.submit_text(QUESTION_PROJECT)

    assert isinstance(manager._llm_classifier, DeepSeekQuestionClassifier)
    session = manager.current_session
    assert session is not None
    # 规则会判 project/true，模型判定把它覆盖成 conceptual/false
    assert session.question_type == "conceptual"
    assert session.rag_required is False
    assert len(client.complete_calls) == 1
    assert len(client.calls) == 1
