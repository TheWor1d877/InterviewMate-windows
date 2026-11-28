"""项目模式集成的测试。

覆盖《5-语音输入与流式ASR》项目模式一节在装配接线之后的行为：

- ``build_handlers`` 注册全部十个动作，三个新动作
  （``project-mode-toggle``、``context-view-toggle``、``context-clear``）
  都有 handler；
- ``project-mode-toggle``：未运行时启动双流转写、界面切到项目模式、
  F10 键位切到上下文视图；运行中停止、界面切回、F10 恢复原绑定；
- ``context-view-toggle``：同步切换上下文视图并返回视图状态；
- ``context-clear``：封存当前对话日志并重开一轮，返回新会话 ID，
  界面的双路临时区与上下文视图跟着刷新；
- voice-start/voice-stop 的模式分派：项目模式运行中按下/松开是截取
  窗口，窗口内的问题文本放后台走问题提交链并带上对话日志上下文；
  普通模式（双流转写没开）走原有语音采集链路，行为零变化；
- 请求拼装：带 ``dialogue_context`` 的提交按「对话日志 → 本轮问题 →
  检索内容 → 指令」的顺序拼进 user 消息；不带时 user 消息与从前完全
  一致，普通模式零变化；
- 装配冒烟：:func:`app.main.assemble_project_mode` 组装对话日志与
  双流转写控制器不报错，数据源挂到界面上。

双流转写控制器与模型客户端都用假实现，不加载 FunASR 模型、
不开真实音频设备。界面用 Textual 的 ``run_test()`` 无头驱动。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import app.main
from app.audio.project_mode import ProjectModeController
from app.core.event_bus import EventBus
from app.core.question_manager import (
    DIALOGUE_CONTEXT_NOTES,
    QuestionManager,
)
from app.dialogue.dialogue_log import DialogueLog
from app.runtime.socket_protocol import ACTIONS, Request
from app.ui.app import InterviewMateApp
from app.ui.panels.context_view import ContextView
from tests.test_ui_skeleton import make_config

# 宽终端的宽度
WIDE_WIDTH = 120

# 项目模式窗口里截取的问题文本
WINDOW_QUESTION = "讲讲你的缓存设计"

# 会触发检索的题面：命中本地规则的项目类关键词，rag_required 为 true
RETRIEVAL_QUESTION = "讲讲我的项目是怎么做的"

# 对话日志里的一句话，带角色标注
LOG_LINE = "15:30:01 [我] 先介绍一下项目背景"


# ----------------------------------------------------------------------
# 假实现
# ----------------------------------------------------------------------
class FakeProjectController:
    """假双流转写控制器：记录启停与截取窗口的调用。"""

    def __init__(self, session_id: str = "20261128-153000") -> None:
        self._running = False
        self._session_id = session_id
        self.started = 0
        self.stopped = 0
        self.windows_opened = 0
        self.windows_closed = 0
        self.close_result = ""

    @property
    def session_id(self) -> str:
        """当前会话 ID。"""
        return self._session_id

    def is_running(self) -> bool:
        """双流转写是否开着。"""
        return self._running

    async def start(self) -> None:
        """开始双流转写。"""
        self.started += 1
        self._running = True

    async def stop(self) -> None:
        """停止双流转写。"""
        self.stopped += 1
        self._running = False

    async def open_question_window(self) -> None:
        """按下截取窗口。"""
        self.windows_opened += 1

    async def close_question_window(self) -> str:
        """松开截取窗口，返回窗口内的问题文本。"""
        self.windows_closed += 1
        return self.close_result


class RecordingUI:
    """假界面：记录模式切换、键位绑定、上下文视图与清空调用。"""

    def __init__(self) -> None:
        self.project_mode_calls: list[tuple[bool, str | None]] = []
        self.cleared = 0
        self.context_toggles = 0
        self.questions: list[str] = []
        self._bindings: dict[str, str] = {"f10": "reanalyze"}

    @property
    def hotkey_bindings(self) -> dict[str, str]:
        """当前生效的键位绑定。"""
        return dict(self._bindings)

    def bind(self, key: str, action: str, show: bool = False) -> None:
        """记录一次键位绑定。"""
        self._bindings[key] = action

    def set_project_mode(self, enabled: bool, session_id: str | None = None) -> None:
        """记录模式切换。"""
        self.project_mode_calls.append((enabled, session_id))

    def clear_dialogue(self) -> None:
        """记录双路临时区清空。"""
        self.cleared += 1

    def action_toggle_context_view(self) -> None:
        """记录上下文视图切换。"""
        self.context_toggles += 1

    @property
    def context_view_open(self) -> bool:
        """上下文视图状态，这里固定返回关着。"""
        return False

    def show_question(self, text: str) -> None:
        """记录右栏题面。"""
        self.questions.append(text)


class RecordingManager:
    """假问题管理器：记录项目模式的问题提交。"""

    def __init__(self) -> None:
        self.submitted = asyncio.Event()
        self.calls: list[tuple[str, str | None, str]] = []

    async def submit_text(
        self,
        text: str,
        image_data_url: str | None = None,
        *,
        dialogue_context: str | None = None,
    ) -> None:
        """记下一次提交。"""
        self.calls.append((text, image_data_url, dialogue_context or ""))
        self.submitted.set()


class FakeClient:
    """假模型客户端：记下收到的请求消息。"""

    name = "mock"

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def chat_stream(
        self, messages: list[dict[str, str]], request_id: str
    ) -> Any:
        """记下消息并给一段固定回答。"""
        self.calls.append(messages)
        yield "回答"

    async def multimodal_stream(
        self, messages: list[dict], request_id: str
    ) -> Any:
        """记下消息并给一段固定回答。"""
        self.calls.append(messages)
        yield "回答"

    async def cancel(self, request_id: str) -> None:
        """什么都不用做。"""


class FakeParser:
    """假解析器：不产出事件。"""

    def __init__(self, request_id: str | None = None) -> None:
        self.request_id = request_id

    def feed(self, text: str) -> list[Any]:
        """没有事件。"""
        return []

    def flush(self) -> list[Any]:
        """没有事件。"""
        return []

    def reset(self) -> None:
        """没有状态可重置。"""


class FakeSink:
    """假事件分发器。"""

    def dispatch(self, event: Any) -> None:
        """忽略事件。"""

    def clear(self) -> None:
        """忽略清空。"""

    def report_error(self, code: str, detail: str) -> None:
        """忽略错误。"""


class FakeCapture:
    """假采集器：普通模式语音链路用，只记开与关。"""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self._running = False

    @property
    def is_running(self) -> bool:
        """采集是否开着。"""
        return self._running

    async def start(self) -> None:
        """开始采集。"""
        self.started += 1
        self._running = True

    async def stop(self) -> None:
        """停止采集。"""
        self.stopped += 1
        self._running = False

    async def get_block(self) -> Any:
        """一直等着，等采集停掉时被取消。"""
        await asyncio.sleep(3600)


class FakeEngine:
    """假 ASR 引擎：识别不出内容，状态直接 ready。"""

    state = "ready"

    async def ensure_loaded(self) -> None:
        """不加载模型。"""

    async def feed(self, block: Any) -> str:
        """识别不出内容。"""
        return ""

    async def flush(self) -> str:
        """没有尾巴。"""
        return ""


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _build_messages(
    question: str,
    question_type: str,
    context: str | None = None,
) -> list[dict[str, str]]:
    """消息构造回调，与普通题的拼装形状一致。"""
    user_text = f"问题：{question.strip()}"
    if context and context.strip():
        user_text = f"{user_text}\n\n背景上下文：\n{context.strip()}"
    return [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": user_text},
    ]


def _build_multimodal_messages(
    question: str,
    image_data_url: str,
    question_type: str,
    context: str | None = None,
) -> list[dict]:
    """带图消息构造回调。"""
    return [{"role": "user", "content": question}]


def _make_manager(client: FakeClient | None = None, retrieve: Any = None) -> tuple[
    QuestionManager, FakeClient
]:
    """造一个用假客户端的问题管理器。"""
    active_client = client if client is not None else FakeClient()
    manager = QuestionManager(
        make_config(),
        active_client,
        lambda request_id, session_id: FakeParser(request_id),
        FakeSink(),
        _build_messages,
        build_multimodal_messages=_build_multimodal_messages,
        retrieve=retrieve,
    )
    return manager, active_client


def _make_state(controller: Any, dialogue_log: DialogueLog, manager: RecordingManager
                ) -> app.main.MainState:
    """造一个挂好项目模式链路的主进程状态。

    问题提交回调模拟装配层的真实回调：题面显示到右栏顶部，
    再带对话日志上下文交给问题管理器。
    """

    async def _submit(text: str) -> None:
        """项目模式的问题提交：题面显示 + 带日志上下文提交。"""
        show = getattr(state._app, "show_question", None)
        if show is not None:
            show(text)
        await manager.submit_text(text, None, dialogue_context=dialogue_log.recent_text())

    state = app.main.MainState()
    state.attach_project_mode(controller, dialogue_log, _submit)
    return state


# ----------------------------------------------------------------------
# handler 注册
# ----------------------------------------------------------------------
def test_build_handlers_covers_all_ten_actions() -> None:
    """build_handlers 注册全部十个协议动作。"""
    state = app.main.MainState()
    handlers = app.main.build_handlers(state)
    for action in ACTIONS:
        assert action in handlers, f"动作 {action} 没有注册 handler"


async def test_project_mode_toggle_starts_and_switches_the_ui() -> None:
    """未运行时切换项目模式：启动双流转写，界面切过去，F10 换成上下文视图。"""
    controller = FakeProjectController()
    dialogue_log = DialogueLog()
    state = _make_state(controller, dialogue_log, RecordingManager())
    ui = RecordingUI()
    state.attach_app(ui)
    handlers = app.main.build_handlers(state)

    payload = await handlers["project-mode-toggle"](
        Request(action="project-mode-toggle", request_id="req-1")
    )

    assert payload["result"] == "started"
    assert payload["running"] is True
    assert payload["session_id"] == "20261128-153000"
    assert controller.started == 1
    assert ui.project_mode_calls == [(True, "20261128-153000")]
    assert ui.hotkey_bindings["f10"] == "toggle_context_view"


async def test_project_mode_toggle_stops_and_restores_f10() -> None:
    """运行中再切一次：停掉双流转写，界面切回来，F10 恢复原绑定。"""
    controller = FakeProjectController()
    dialogue_log = DialogueLog()
    state = _make_state(controller, dialogue_log, RecordingManager())
    ui = RecordingUI()
    state.attach_app(ui)
    handlers = app.main.build_handlers(state)

    await handlers["project-mode-toggle"](Request(action="project-mode-toggle"))
    payload = await handlers["project-mode-toggle"](
        Request(action="project-mode-toggle", request_id="req-2")
    )

    assert payload["result"] == "stopped"
    assert payload["running"] is False
    assert controller.stopped == 1
    assert ui.project_mode_calls[-1] == (False, None)
    assert ui.hotkey_bindings["f10"] == "reanalyze"


async def test_project_mode_toggle_without_the_chain_is_ignored() -> None:
    """链路还没装好时切换动作被忽略，不崩。"""
    state = app.main.MainState()
    handlers = app.main.build_handlers(state)

    payload = await handlers["project-mode-toggle"](
        Request(action="project-mode-toggle")
    )

    assert payload["result"] == "ignored"


async def test_failed_start_does_not_switch_the_ui() -> None:
    """两路采集都没起来时视为启动失败，界面不切模式、键位不动。"""
    class FailedController(FakeProjectController):
        """启动不起来的假控制器。"""

        async def start(self) -> None:
            self.started += 1
            # is_running 保持 False，模拟两路都没起来

    state = _make_state(FailedController(), DialogueLog(), RecordingManager())
    ui = RecordingUI()
    state.attach_app(ui)
    handlers = app.main.build_handlers(state)

    payload = await handlers["project-mode-toggle"](Request(action="project-mode-toggle"))

    assert payload["result"] == "failed"
    assert ui.project_mode_calls == []
    assert ui.hotkey_bindings["f10"] == "reanalyze"


async def test_context_view_toggle_runs_the_app_action() -> None:
    """context-view-toggle 调界面的视图切换 action，返回视图状态。"""
    state = _make_state(FakeProjectController(), DialogueLog(), RecordingManager())
    ui = RecordingUI()
    state.attach_app(ui)
    handlers = app.main.build_handlers(state)

    payload = await handlers["context-view-toggle"](
        Request(action="context-view-toggle", request_id="req-3")
    )

    assert payload["result"] == "toggled"
    assert payload["context_view_open"] is False
    assert ui.context_toggles == 1


async def test_context_clear_resets_the_dialogue_log() -> None:
    """context-clear 封存当前会话并重开一轮，返回新会话 ID，界面跟着刷新。"""
    dialogue_log = DialogueLog()
    state = _make_state(FakeProjectController(), dialogue_log, RecordingManager())
    ui = RecordingUI()
    state.attach_app(ui)
    handlers = app.main.build_handlers(state)

    dialogue_log.append("我", "旧的一轮")
    old_session = dialogue_log.session_id
    payload = await handlers["context-clear"](
        Request(action="context-clear", request_id="req-4")
    )

    assert payload["result"] == "cleared"
    assert payload["session_id"] != old_session
    assert payload["session_id"] == dialogue_log.session_id
    # 日志全文已清空，旧句子不在新会话里
    assert dialogue_log.full_text == ""
    assert ui.cleared == 1


async def test_context_clear_without_the_log_is_ignored() -> None:
    """对话日志还没装好时清空动作被忽略，不崩。"""
    state = app.main.MainState()
    handlers = app.main.build_handlers(state)

    payload = await handlers["context-clear"](Request(action="context-clear"))

    assert payload["result"] == "ignored"


# ----------------------------------------------------------------------
# voice-start / voice-stop 的模式分派
# ----------------------------------------------------------------------
async def test_voice_start_opens_the_question_window_in_project_mode() -> None:
    """项目模式运行中按下 F8：开截取窗口，不碰普通语音采集。"""
    controller = FakeProjectController()
    dialogue_log = DialogueLog()
    state = _make_state(controller, dialogue_log, RecordingManager())
    handlers = app.main.build_handlers(state)
    await handlers["project-mode-toggle"](Request(action="project-mode-toggle"))
    controller.windows_opened = 0

    payload = await handlers["voice-start"](
        Request(action="voice-start", request_id="req-5")
    )

    assert payload["result"] == "started"
    assert controller.windows_opened == 1
    # 普通语音链路的状态没被碰
    assert state.audio_on is False


async def test_voice_stop_submits_the_window_question_with_the_log() -> None:
    """项目模式运行中松开 F8：窗口内的问题标成问题并带日志上下文提交。"""
    controller = FakeProjectController()
    dialogue_log = DialogueLog()
    manager = RecordingManager()
    state = _make_state(controller, dialogue_log, manager)
    ui = RecordingUI()
    state.attach_app(ui)
    handlers = app.main.build_handlers(state)
    await handlers["project-mode-toggle"](Request(action="project-mode-toggle"))

    # 窗口期间对话日志落了一句
    dialogue_log.append("我", LOG_LINE)
    controller.close_result = WINDOW_QUESTION

    payload = await handlers["voice-stop"](
        Request(action="voice-stop", request_id="req-6")
    )
    await asyncio.wait_for(manager.submitted.wait(), timeout=5)

    assert payload["result"] == "stopped"
    assert payload["text"] == WINDOW_QUESTION
    assert manager.calls == [(WINDOW_QUESTION, None, dialogue_log.recent_text())]
    assert LOG_LINE in manager.calls[0][2]
    # 题面显示到右栏顶部
    assert ui.questions == [WINDOW_QUESTION]


async def test_voice_stop_with_an_empty_window_submits_nothing() -> None:
    """窗口里没有面试官语句时不提交，返回的 text 是空串。"""
    controller = FakeProjectController()
    manager = RecordingManager()
    state = _make_state(controller, DialogueLog(), manager)
    handlers = app.main.build_handlers(state)
    await handlers["project-mode-toggle"](Request(action="project-mode-toggle"))

    payload = await handlers["voice-stop"](Request(action="voice-stop"))

    assert payload["result"] == "stopped"
    assert payload["text"] == ""
    assert not manager.submitted.is_set()


async def test_voice_actions_stay_normal_when_project_mode_is_off() -> None:
    """双流转写没开时 F8 走普通语音链路：开流、关流，不碰截取窗口。"""
    controller = FakeProjectController()
    manager = RecordingManager()
    state = _make_state(controller, DialogueLog(), manager)
    ui = RecordingUI()
    state.attach_app(ui)
    capture = FakeCapture()
    engine = FakeEngine()
    voice = app.main.assemble_voice(
        make_config(),
        ui,
        RecordingManager(),
        state=state,
        engine=engine,
        factory=lambda source: capture,
    )
    state.set_voice_chain(voice)
    handlers = app.main.build_handlers(state)

    started = await handlers["voice-start"](Request(action="voice-start"))
    assert started["result"] == "started"
    assert capture.started == 1
    assert controller.windows_opened == 0

    stopped = await handlers["voice-stop"](Request(action="voice-stop"))
    assert stopped["result"] == "empty"
    assert capture.stopped == 1
    assert controller.windows_closed == 0
    assert not manager.submitted.is_set()


# ----------------------------------------------------------------------
# 请求拼装：对话日志上下文
# ----------------------------------------------------------------------
async def test_submit_with_dialogue_context_builds_the_expected_user_message() -> None:
    """带日志上下文的提交按「日志 → 问题 → 检索 → 指令」的顺序拼 user 消息。"""
    client = FakeClient()
    manager, _ = _make_manager(
        client,
        retrieve=lambda query: _retrieve_with_context(),
    )
    log_context = f"{LOG_LINE}\n15:30:05 [面试官] 那讲讲你的缓存设计"

    await manager.submit_text(
        RETRIEVAL_QUESTION, None, dialogue_context=log_context
    )

    assert len(client.calls) == 1
    messages = client.calls[0]
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    # 顺序：日志段 → 本轮问题 → 检索内容 → 两句指令
    assert user.index("对话上下文：") < user.index(LOG_LINE)
    assert user.index(LOG_LINE) < user.index(f"问题：{RETRIEVAL_QUESTION}")
    assert user.index(f"问题：{RETRIEVAL_QUESTION}") < user.index("背景上下文：")
    assert user.index("RAG-CONTEXT") < user.index(DIALOGUE_CONTEXT_NOTES)
    assert DIALOGUE_CONTEXT_NOTES in user
    # 检索 query 用原始问题，不被日志污染
    assert user.count(f"问题：{RETRIEVAL_QUESTION}") == 1


async def _retrieve_with_context() -> tuple[str | None, list[dict]]:
    """假检索：返回一段上下文。"""
    return "RAG-CONTEXT", []


async def test_submit_without_dialogue_context_keeps_the_old_user_message() -> None:
    """普通模式不传日志上下文：user 消息与从前完全一致，零变化。"""
    client = FakeClient()
    manager, _ = _make_manager(
        client,
        retrieve=lambda query: _retrieve_with_context(),
    )

    await manager.submit_text(RETRIEVAL_QUESTION)

    assert len(client.calls) == 1
    messages = client.calls[0]
    user = messages[1]["content"]
    assert user == (
        f"问题：{RETRIEVAL_QUESTION}\n\n背景上下文：\nRAG-CONTEXT"
    )
    assert "对话上下文" not in user
    assert DIALOGUE_CONTEXT_NOTES not in user


async def test_dialogue_context_with_no_retrieval_puts_notes_after_the_question() -> None:
    """没检索到内容时，指令单独成段跟在问题后面，提交照常发出。"""
    client = FakeClient()
    manager, _ = _make_manager(
        client,
        retrieve=lambda query: _retrieve_empty(),
    )
    log_context = LOG_LINE

    await manager.submit_text(RETRIEVAL_QUESTION, None, dialogue_context=log_context)

    assert len(client.calls) == 1
    user = client.calls[0][1]["content"]
    assert "背景上下文" not in user
    assert user.index(LOG_LINE) < user.index(f"问题：{RETRIEVAL_QUESTION}")
    assert user.index(f"问题：{RETRIEVAL_QUESTION}") < user.index(DIALOGUE_CONTEXT_NOTES)
    # 降级路径不阻塞提交：检索为空照样出请求
    assert DIALOGUE_CONTEXT_NOTES in user


async def _retrieve_empty() -> tuple[str | None, list[dict]]:
    """假检索：Dify 离线降级，返回空结果。"""
    return None, []


# ----------------------------------------------------------------------
# 装配冒烟与界面联动
# ----------------------------------------------------------------------
def test_assemble_project_mode_smoke() -> None:
    """装配冒烟：对话日志与双流转写控制器组装不报错，数据源挂到界面。"""
    config = make_config()
    ui = RecordingUI()
    manager = RecordingManager()

    dialogue_log, controller = app.main.assemble_project_mode(
        config, ui, manager,  # type: ignore[arg-type]
    )

    assert isinstance(dialogue_log, DialogueLog)
    assert isinstance(controller, ProjectModeController)
    assert controller.is_running() is False
    # RecordingUI 没有 attach_text_provider：装配不因此失败
    assert not manager.submitted.is_set()


async def test_assemble_project_mode_attaches_the_text_provider() -> None:
    """装配后对话日志全文作为上下文视图的数据源挂到界面上。

    装配在界面挂载之后做（与 ``run_main`` 的时序一致），
    上下文视图打开时取到的就是日志全文。
    """
    config = make_config()
    manager = RecordingManager()
    ui = InterviewMateApp(config=config)

    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        dialogue_log, controller = app.main.assemble_project_mode(
            config, ui, manager,  # type: ignore[arg-type]
        )
        await pilot.pause()
        dialogue_log.append("我", "装配冒烟的一句话")
        await pilot.pause()

        assert isinstance(controller, ProjectModeController)
        assert ui._text_provider is not None
        assert "装配冒烟的一句话" in dialogue_log.full_text
        # 数据源取到的就是界面视图会显示的内容
        assert "装配冒烟的一句话" in ui._text_provider()


async def test_f10_switches_to_the_context_view_in_project_mode() -> None:
    """项目模式激活后按 F10 打开上下文视图，退出后 F10 回到重新分析。"""
    controller = FakeProjectController()
    dialogue_log = DialogueLog()
    manager = RecordingManager()
    state = _make_state(controller, dialogue_log, manager)
    ui = InterviewMateApp(config=make_config())
    state.attach_app(ui)
    handlers = app.main.build_handlers(state)

    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        # 数据源与真实装配一致：对话日志全文作为上下文视图的数据源
        ui.attach_text_provider(lambda: dialogue_log.full_text)
        await handlers["project-mode-toggle"](Request(action="project-mode-toggle"))
        await pilot.pause()
        assert ui.project_mode is True
        assert "f10" in ui.hotkey_bindings
        assert ui.hotkey_bindings["f10"] == "toggle_context_view"

        dialogue_log.append("面试官", "请介绍一下架构")
        await pilot.press("f10")
        await pilot.pause()
        assert ui.context_view_open is True
        assert "请介绍一下架构" in ui.query_one(ContextView).plain_text()

        # 退出项目模式：F10 恢复 reanalyze，视图回到回答视图
        await handlers["project-mode-toggle"](Request(action="project-mode-toggle"))
        await pilot.pause()
        assert ui.project_mode is False
        assert ui.hotkey_bindings["f10"] == "reanalyze"


async def test_real_dialogue_log_flow_through_the_handlers() -> None:
    """真实对话日志走 handler 全流程：落库、截取、清空后新会话。"""
    controller = FakeProjectController()
    dialogue_log = DialogueLog()
    manager = RecordingManager()
    state = _make_state(controller, dialogue_log, manager)
    state.attach_app(RecordingUI())
    handlers = app.main.build_handlers(state)

    await handlers["project-mode-toggle"](Request(action="project-mode-toggle"))
    await handlers["voice-start"](Request(action="voice-start"))
    dialogue_log.append("面试官", WINDOW_QUESTION)
    controller.close_result = WINDOW_QUESTION
    await handlers["voice-stop"](Request(action="voice-stop"))
    await asyncio.wait_for(manager.submitted.wait(), timeout=5)

    # 提交的问题带上了窗口期间的日志全文
    assert manager.calls[0][0] == WINDOW_QUESTION
    assert WINDOW_QUESTION in manager.calls[0][2]

    # 清空：日志封存重开，全文清零
    payload = await handlers["context-clear"](Request(action="context-clear"))
    assert payload["session_id"] == dialogue_log.session_id
    assert dialogue_log.full_text == ""


@pytest.mark.parametrize("action", ["project-mode-toggle", "context-view-toggle", "context-clear"])
async def test_new_actions_never_crash_without_the_chain(action: str) -> None:
    """三个新动作在链路没装好时给出明确的结果标识，不崩。"""
    state = app.main.MainState()
    handlers = app.main.build_handlers(state)

    payload = await handlers[action](Request(action=action))
    assert payload.get("result") in {"ignored", "toggled"}