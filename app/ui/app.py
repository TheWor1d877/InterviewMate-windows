"""Textual 应用。

界面四段结构见《3-文本输入与流式回答》终端界面一节，快捷键见同节快捷键表：
F7 截图并立即提交、回车提交手动文本、Shift+回车换行、
Ctrl+N 创建新问题、F10 重新分析、Esc 取消、F3 查看检索来源、Q 退出。
F8 按住说话由 AHK 全局热键（``scripts/windows/hotkeys.ahk``）承担：
按下开始录音、松开停止并提交最终文本，经 socket 的 ``voice-start`` /
``voice-stop`` 送到主进程，不在界面绑定。

界面这边只做四件事：把状态栏接到事件总线上，把输入框与快捷键接到问题管理器上，
把当前问题显示到右栏顶部，把检索来源与临时识别文字显示出来。
问题管理器由装配层建好后用 :meth:`InterviewMateApp.attach_manager` 挂进来，
截图管理器用 :meth:`InterviewMateApp.attach_screenshot_manager` 挂进来，
挂之前界面照常能起。

检索来源面板是临时浮层，按 F3 打开、列出后按 Esc 或再按 F3 关掉，
见《6-Dify检索与项目知识》检索来源展示一节；临时识别文字显示在状态栏下方的一行，
见《5-语音输入与流式ASR》状态显示一节。

项目模式这一波 UI 只做显示与切换：双路临时转写订阅事件总线的
``dialogue.partial`` / ``dialogue.sentence``（引擎层发布，这里只渲染），
F10 上下文视图用注入的 ``text_provider`` 现取对话日志全文，
状态栏与快捷键提示栏随模式切换。模式的开与关由集成层调
:meth:`set_project_mode`，引擎对象与数据源由集成层注入，界面不 import 引擎模块。
"""

from __future__ import annotations

import asyncio
import inspect
import time

from typing import Any, Awaitable, Callable

from textual import on
from textual.app import App, ComposeResult
from textual.css.query import NoMatches

from app.config import AppConfig, current_config, load_config
from app.core.event_bus import EventBus
from app.ui.layout import MainLayout
from app.ui.panels.answer_panel import AnswerPanel
from app.ui.panels.context_view import ContextView
from app.ui.panels.dialogue_temp import DialogueTempPanel
from app.ui.panels.hotkey_bar import HotkeyBar
from app.ui.panels.input_panel import QuestionInputPanel
from app.ui.panels.navigation_panel import NavigationPanel
from app.ui.panels.question_header import QuestionHeader
from app.ui.panels.rag_sources import RagSourcesPanel
from app.ui.panels.voice_text import VoiceTextPanel
from app.ui.status_bar import StatusBar, StatusUpdate
from app.utils.logger import log_event

__all__ = ["InterviewMateApp"]


# 状态栏订阅的事件类型名，与 StatusUpdate.type 一致
STATUS_EVENT_TYPE = "status.update"

# 项目模式的双路转写事件类型名，引擎层发布，这里只订阅渲染
DIALOGUE_PARTIAL_EVENT = "dialogue.partial"
DIALOGUE_SENTENCE_EVENT = "dialogue.sentence"

# 生成过程中状态栏提示位显示的短提示
GENERATING_HINT = "生成中"

# 界面这边自己撞上的异常用这个错误码，与《2-骨架与配置》错误码表一致
STREAM_ERROR_CODE = "E_MODEL_STREAM"

# 截图提问在右栏顶部显示的题面，没有语音文字时用它
SCREENSHOT_QUESTION_TEXT = "截图提问"

# 上下文视图打开但数据源还没注入时显示的提示
CONTEXT_NO_PROVIDER_HINT = "上下文数据源未接入"


class InterviewMateApp(App):
    """InterviewMate 终端应用。"""

    CSS_PATH = "styles.tcss"
    TITLE = "InterviewMate"
    # 快捷键按配置里的 hotkeys 段在实例化时绑定，见 _bind_hotkeys。
    # 这里只放两个不归配置管的退出键。
    BINDINGS = [
        ("ctrl+c", "quit"),
        ("ctrl+q", "quit"),
    ]

    def __init__(
        self,
        config: AppConfig | None = None,
        bus: EventBus | None = None,
        manager: Any | None = None,

    ) -> None:
        super().__init__()
        # 显式注入的配置只给本次构造用，不写进程级配置；
        # 没注入时优先取进程级当前配置，真没有再读一遍配置文件
        if config is not None:
            self._config = config
        else:
            self._config = current_config()
            if self._config is None:
                self._config = load_config()
        self._bus = bus if bus is not None else EventBus()
        # 问题管理器由装配层挂进来，类型不写死，界面只用四个方法
        self._manager: Any | None = manager
        # 截图管理器由装配层挂进来，类型不写死，界面只用 capture_and_submit
        self._screenshot_manager: Any | None = None
        # 输入路由器由装配层挂进来，手工输入回车走它，只用 route_text
        self._input_router: Any | None = None

        # 当前会话的检索来源，普通字典，字段是 document_name、score、content
        # 来源存进会话后一直留着，模型输出结束也能再按 F3 查
        self._rag_sources: list[dict] = []
        # 最近一次状态更新的快照，改 MODEL 时按快照整体重发
        self._status: dict[str, Any] = {
            "running": "stopped",
            "audio": "off",
            "asr": "idle",
            "dify": "offline",
            "model": "idle",
            "ttft_ms": None,
            "hint": "",
        }
        # 生成轮次编号，只有最新一轮能改状态栏的 MODEL 字段
        self._generation_id = 0
        # 报过错的那一轮编号，别的轮次不受影响
        self._error_generation = -1
        # 按配置绑定的快捷键：键名 -> action 名
        self._hotkey_bindings: dict[str, str] = {}
        self._bind_hotkeys()
        # 全局热键监听接管截图键后置 True，界面这边不再处理截图键；
        # F8 按住说话由 AHK 全局热键处理，本来就不在界面绑定
        self._global_hotkeys_active: bool = False
        # 当前这段语音已经识别出来的累计文本，语音流结束时清零
        self._voice_text: str = ""
        # 项目模式开关，由集成层调 set_project_mode 切换，界面自己不绑键
        self._project_mode: bool = False
        # 上下文视图的日志全文数据源，由集成层注入，界面不 import 引擎模块
        self._text_provider: Callable[[], str] | None = None

    # ------------------------------------------------------------------
    # 快捷键
    # ------------------------------------------------------------------
    def _bind_hotkeys(self) -> None:
        """按配置里的 hotkeys 段绑定快捷键。

        配置里的写法是 ``F8``、``ctrl+n``、``escape`` 这种，转成 Textual 的
        键名要小写。绑定后 :attr:`hotkey_bindings` 里是实际生效的键位。
        """
        hotkeys = self._config.hotkeys
        # F8 按住说话由 AHK 全局热键经 socket 处理，不在界面绑定，
        # 见《5-语音输入与流式ASR》F8 交互一节。
        mapping = (
            (hotkeys.screenshot, "screenshot"),
            (hotkeys.new_question, "new_question"),
            (hotkeys.cancel, "cancel"),
            (hotkeys.reanalyze, "reanalyze"),
            (hotkeys.show_rag_sources, "show_rag_sources"),
            (hotkeys.quit, "quit"),
        )
        for key, action in mapping:
            normalized = key.strip().lower()
            if not normalized:
                continue
            self.bind(normalized, action, show=False)
            self._hotkey_bindings[normalized] = action

    @property
    def hotkey_bindings(self) -> dict[str, str]:
        """当前生效的快捷键，键名到 action 名。"""
        return dict(self._hotkey_bindings)

    def mark_global_hotkeys_active(self) -> None:
        """标记全局热键监听已接管截图键。

        全局热键由 AutoHotkey v2 脚本（``scripts/windows/hotkeys.ahk``）捕获，
        终端失焦时也能触发；脚本调 ``python -m app.cli hotkey ...`` 通知主进程。
        生效后同一次截图键（F7）会被 Textual 绑定与全局链路各处理一次，
        所以界面这边要短路截图动作。F8 按住说话不在界面绑定，
        见《5-语音输入与流式ASR》F8 交互一节。
        """
        self._global_hotkeys_active = True
        log_event(
            "ui.hotkey.global_active",
            "全局热键监听已接管截图键，界面不再处理截图键",
        )

    @property
    def global_hotkeys_active(self) -> bool:
        """全局热键监听是否已接管截图键（F8 不在界面绑定）。"""
        return self._global_hotkeys_active

    # ------------------------------------------------------------------
    # 只读入口
    # ------------------------------------------------------------------
    @property
    def config(self) -> AppConfig:
        """应用使用的配置。"""
        return self._config

    @property
    def bus(self) -> EventBus:
        """应用使用的事件总线。"""
        return self._bus

    @property
    def manager(self) -> Any | None:
        """应用使用的问题管理器，没有挂上时是 None。"""
        return self._manager

    @property
    def navigation_panel(self) -> NavigationPanel:
        """左侧「回答流程」面板，供装配层建分发器用。"""
        return self.query_one(NavigationPanel)

    @property
    def answer_panel(self) -> AnswerPanel:
        """右侧「关键回答内容」面板，供装配层建分发器用。"""
        return self.query_one(AnswerPanel)

    @property
    def question_header(self) -> QuestionHeader:
        """右栏顶部的当前问题标题。"""
        return self.query_one(QuestionHeader)

    # ------------------------------------------------------------------
    # 组装
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        """产出主布局。"""
        yield MainLayout(self._config)

    async def on_mount(self) -> None:
        """订阅状态栏更新与双路转写事件，挂上检索来源浮层，并把焦点给底部输入框。"""
        self._bus.subscribe(STATUS_EVENT_TYPE, self._handle_status_update)
        self._bus.subscribe(DIALOGUE_PARTIAL_EVENT, self._handle_dialogue_partial)
        self._bus.subscribe(DIALOGUE_SENTENCE_EVENT, self._handle_dialogue_sentence)
        panel = RagSourcesPanel(id="rag-sources")
        # 浮层默认隐藏，按 F3 才打开
        panel.display = False
        await self.mount(panel)
        try:
            self.query_one(QuestionInputPanel).focus()
        except NoMatches:
            # 输入框还没挂上，等下一次挂载
            pass

    def attach_manager(self, manager: Any) -> None:
        """装配层建好问题管理器后挂进来。

        :param manager: 问题管理器，界面只用 ``submit_text``、``cancel``、
            ``new_question``、``reanalyze`` 四个方法。
        """
        self._manager = manager
        log_event("ui.manager_attached", "问题管理器已挂到界面上")

    def attach_screenshot_manager(self, screenshot_manager: Any) -> None:
        """装配层建好截图管理器后挂进来。

        :param screenshot_manager: 截图管理器，界面只用 ``capture_and_submit``
            一个方法。
        """
        self._screenshot_manager = screenshot_manager
        log_event("ui.screenshot_manager_attached", "截图管理器已挂到界面上")

    def attach_input_router(self, router: Any) -> None:
        """装配层建好输入路由器后挂进来。

        手工输入回车走路由器：三种输入汇到一条路上，
        见《8-集成与验收》三种输入的汇合一节。
        路由器没挂上时退回直接调用问题管理器，行为与从前一样。
        Ctrl+N 新会话时还要调它的 ``reset_session``，旧会话会合中留的输入作废。

        :param router: 输入路由器，界面用 ``route_text`` 与 ``reset_session`` 两个方法。
        """
        self._input_router = router
        log_event("ui.input_router_attached", "输入路由器已挂到界面上")

    @property
    def screenshot_manager(self) -> Any | None:
        """应用使用的截图管理器，没有挂上时是 None。"""
        return self._screenshot_manager

    # ------------------------------------------------------------------
    # 项目模式
    # ------------------------------------------------------------------
    @property
    def project_mode(self) -> bool:
        """当前是否处于项目模式。"""
        return self._project_mode

    def attach_text_provider(self, text_provider: Callable[[], str]) -> None:
        """装配层建好对话日志数据源后注入进来。

        上下文视图打开与清空刷新时各调一次 ``text_provider()`` 现取全文，
        界面不缓存旧日志，也不 import 对话引擎的实现模块。

        :param text_provider: 返回当前对话日志全文的可调用对象。
        """
        self._text_provider = text_provider
        log_event("ui.text_provider_attached", "上下文数据源已挂到界面上")
        # 上下文视图开着的话，换数据源后立刻刷新一遍
        view = self._context_view()
        if view is not None and view.display:
            self._refresh_context_view()

    def set_project_mode(self, enabled: bool, session_id: str | None = None) -> None:
        """切换项目模式，模式相关的显示跟着变。

        状态栏的 MODE/SESSION、快捷键提示栏的键位、状态栏下方的双路临时转写区
        一起切过去。进入项目模式时把普通模式的单路语音行清掉，两者互斥；
        退出项目模式时双路转写的数据清掉，下次进入不带残留。
        模式的开与关、项目模式键位的实际行为都归集成层管，界面只管显示。

        :param enabled: 是否进入项目模式。
        :param session_id: 当前会话 ID，只在项目模式下显示。
        """
        self._project_mode = bool(enabled)
        mode = "project" if enabled else "normal"
        status_bar = self._status_bar()
        if status_bar is not None:
            status_bar.set_mode(mode)
            status_bar.set_session(session_id if enabled else None)
        try:
            hotkey_bar = self.query_one(HotkeyBar)
        except NoMatches:
            hotkey_bar = None
        if hotkey_bar is not None:
            hotkey_bar.set_mode(mode)
        dialogue_panel = self._dialogue_panel()
        if dialogue_panel is not None:
            dialogue_panel.set_active(enabled)
            if not enabled:
                dialogue_panel.clear()
        if enabled:
            # 单路语音行让位给双路转写，两个临时区不同时显示
            self.clear_voice_text()
        log_event(
            "ui.project_mode",
            f"项目模式{'开启' if enabled else '关闭'}",
        )

    def show_dialogue_partial(self, role: str, text: str) -> None:
        """显示一路还在识别中的临时转写。

        数据始终记录，是否显示由项目模式开关管着。

        :param role: 说话角色，例如 ``我``、``面试官``。
        :param text: 这一路的临时转写。
        """
        panel = self._dialogue_panel()
        if panel is None:
            return
        panel.show_partial(role, text)

    def show_dialogue_sentence(self, role: str, text: str) -> None:
        """显示一路已经定稿的句子。

        :param role: 说话角色，例如 ``我``、``面试官``。
        :param text: 定稿句。
        """
        panel = self._dialogue_panel()
        if panel is None:
            return
        panel.show_sentence(role, text)

    def clear_dialogue(self) -> None:
        """清掉双路临时转写，并让上下文视图跟着刷新。

        F11 重开一轮时由集成层先清对话引擎的数据，再调这个方法刷新视图：
        上下文视图开着的话重新取一次日志全文，拿到的就是清空后的内容。
        """
        panel = self._dialogue_panel()
        if panel is not None:
            panel.clear()
        view = self._context_view()
        if view is None:
            return
        view.clear()
        if view.display:
            self._refresh_context_view()

    def action_toggle_context_view(self) -> None:
        """F10：右栏临时切成上下文视图，再按一次切回回答视图。

        键位绑定留给集成层：配置里 F10 已绑定重新分析（reanalyze），
        现有绑定以现有为准，项目模式激活时由集成层把 F10 切到这个 action。
        """
        view = self._context_view()
        if view is None:
            return
        if view.display:
            self._close_context_view()
        else:
            self._open_context_view()

    @property
    def context_view_open(self) -> bool:
        """上下文视图当前是否打开。"""
        view = self._context_view()
        return view is not None and view.display

    def _open_context_view(self) -> None:
        """打开上下文视图：右栏的问题与回答内容让位，日志全文顶上。"""
        view = self._context_view()
        if view is None:
            return
        self._refresh_context_view()
        view.display = True
        header = self._question_header()
        if header is not None:
            header.display = False
        try:
            self.query_one("#answer-scroll").display = False
        except NoMatches:
            pass
        log_event("ui.context_view", "上下文视图已打开")

    def _close_context_view(self) -> None:
        """关掉上下文视图：回答视图回来，有题目的话题目也回来。"""
        view = self._context_view()
        if view is None:
            return
        view.display = False
        header = self._question_header()
        if header is not None:
            header.display = bool(header.question)
        try:
            self.query_one("#answer-scroll").display = True
        except NoMatches:
            pass
        log_event("ui.context_view", "上下文视图已关闭")

    def _refresh_context_view(self) -> None:
        """从注入的数据源现取一次日志全文显示到上下文视图。"""
        view = self._context_view()
        if view is None:
            return
        provider = self._text_provider
        if provider is None:
            view.show_text(CONTEXT_NO_PROVIDER_HINT)
            return
        try:
            text = provider()
        except Exception as exc:
            log_event(
                "ui.context_view_failed",
                f"取对话日志全文失败：{exc}",
                level="ERROR",
            )
            view.show_text(f"上下文读取失败：{exc}")
            return
        view.show_text(text)

    def _handle_dialogue_partial(self, event: Any) -> None:
        """双路转写临时事件的同步处理函数，把角色与文本写到临时区。

        事件类由引擎层定义，这里只用 ``role`` / ``text`` 两个字段，
        取不到按空值处理，不让界面崩。
        """
        role = str(getattr(event, "role", "") or "")
        text = str(getattr(event, "text", "") or "")
        self.show_dialogue_partial(role, text)

    def _handle_dialogue_sentence(self, event: Any) -> None:
        """双路转写定稿事件的同步处理函数。"""
        role = str(getattr(event, "role", "") or "")
        text = str(getattr(event, "text", "") or "")
        self.show_dialogue_sentence(role, text)

    def _dialogue_panel(self) -> DialogueTempPanel | None:
        """取双路临时转写区，取不到时返回 None。"""
        try:
            return self.query_one(DialogueTempPanel)
        except NoMatches:
            return None

    def _context_view(self) -> ContextView | None:
        """取上下文视图，取不到时返回 None。"""
        try:
            return self.query_one(ContextView)
        except NoMatches:
            return None

    def show_question(self, text: str) -> None:
        """把当前问题显示到右栏顶部。

        截图与语音的题目以后也从这个入口写进来，右栏顶部只认它一个。

        :param text: 题面文本。
        """
        header = self._question_header()
        if header is None:
            # 组件还没挂上，等下一次挂载
            return
        header.show_question(text)

    def clear_question(self) -> None:
        """清空右栏顶部的问题，新建问题时用。"""
        header = self._question_header()
        if header is None:
            return
        header.clear_question()

    def show_error(self, code: str, detail: str) -> None:
        """把错误显示到状态栏提示位，并把 MODEL 置成 error。

        :param code: 错误码，例如 ``E_MODEL_TIMEOUT``。
        :param detail: 一句话说明。
        """
        log_event("ui.error", f"{code} {detail}", level="ERROR", code=code)
        self._error_generation = self._generation_id
        hint = f"{code} {detail}"
        status_bar = self._status_bar()
        if status_bar is not None:
            # 提示位先立刻显示，状态字段随后由状态事件带过去
            status_bar.set_hint(hint)
        self._publish_model_later("error", hint)

    def store_rag_sources(self, entries: list[dict]) -> None:
        """只把检索来源存进当前会话，不打开浮层。

        给装配层的 ``on_sources`` 回调用：检索完成后来源静默存下来，
        只有用户按 F3 时才弹浮层，见《6-Dify检索与项目知识》检索来源展示一节。

        :param entries: 当前问题用到的来源条目，顺序就是显示顺序。
        """
        self._rag_sources = list(entries)

    def show_rag_sources(self, entries: list[dict]) -> None:
        """把检索来源记进当前会话，并打开检索来源面板。

        显示语义，保留给 F3 这条路与既有调用方；
        检索完成时的静默存档走 :meth:`store_rag_sources`。

        来源是普通字典，字段是 document_name、score、content；
        面板不 import 检索层的实现类。

        :param entries: 当前问题用到的来源条目，顺序就是显示顺序。
        """
        self.store_rag_sources(entries)
        self._open_rag_sources()

    def clear_rag_sources(self) -> None:
        """清掉当前会话的检索来源，面板内容一起清空并关掉。

        只有新建问题或显式清理时才清，模型输出结束后来源仍然留着，
        按 F3 还能再查。
        """
        self._rag_sources = []
        panel = self._rag_sources_panel()
        if panel is None:
            return
        panel.clear()
        self._close_rag_sources()

    def show_voice_text(self, text: str) -> None:
        """把临时识别文字显示到状态栏下方的一行。

        这一行显示的是当前这段语音的累计文本：
        送上来的是增量就拼到已有文字后面，送上来的是累计文本就直接替换，
        两种送法都得到一整句可读的话，
        见《5-语音输入与流式ASR》部分结果与最终结果一节。

        一条语音流结束时由 :meth:`clear_voice_text` 清零，
        下一条语音流从头开始，不带上上一条的残留。

        :param text: 这次送来的临时识别文字，空串按清空处理。
        """
        if not text:
            self._voice_text = ""
        elif text.startswith(self._voice_text):
            # 送上来的是这段语音的累计文本，直接替换
            self._voice_text = text
        else:
            # 送上来的是增量，拼到已有文字后面
            self._voice_text += text
        panel = self._voice_text_panel()
        if panel is None:
            # 组件还没挂上，等下一次挂载
            return
        panel.show_text(self._voice_text)

    def clear_voice_text(self) -> None:
        """清掉状态栏下方的临时识别文字。

        累计值一起清零：下一条语音流从头开始，不带上上一条的残留，
        见《5-语音输入与流式ASR》部分结果与最终结果一节。
        """
        self._voice_text = ""
        panel = self._voice_text_panel()
        if panel is None:
            return
        panel.clear()

    # ------------------------------------------------------------------
    # 事件总线
    # ------------------------------------------------------------------
    def _handle_status_update(self, event: StatusUpdate) -> None:
        """状态更新事件的同步处理函数，把新状态刷到状态栏。"""
        # 记下最近一次的状态，改 MODEL 时按快照整体重发，其余字段不受影响
        self._status.update(
            running=event.running,
            audio=event.audio,
            asr=event.asr,
            dify=event.dify,
            model=event.model,
            ttft_ms=event.ttft_ms,
            hint=event.hint,
        )
        status_bar = self._status_bar()
        if status_bar is None:
            # 状态栏还没挂上，忽略这次事件
            return
        status_bar.update_status(
            running=event.running,
            audio=event.audio,
            asr=event.asr,
            dify=event.dify,
            model=event.model,
            ttft_ms=event.ttft_ms,
            hint=event.hint,
        )

    async def publish_status(self, **fields: Any) -> None:
        """构造并发布一条状态更新事件。

        ``fields`` 覆盖 :class:`~app.ui.status_bar.StatusUpdate` 的对应字段。
        """
        event = StatusUpdate(ts=time.time(), **fields)
        await self._bus.publish(event)

    # ------------------------------------------------------------------
    # 界面事件
    # ------------------------------------------------------------------
    @on(QuestionInputPanel.Submitted)
    def _handle_input_submitted(self, event: QuestionInputPanel.Submitted) -> None:
        """输入框提交：题面显示到右栏顶部，再交给输入路由器。

        路由器没挂上时退回直接调用问题管理器，行为与从前一样。
        """
        self.show_question(event.text)
        router = self._input_router
        if router is None:
            self._start_generation("submit_text", event.text)
            return
        self._start_generation("route_text", event.text, target=router)

    # ------------------------------------------------------------------
    # 快捷键动作
    # ------------------------------------------------------------------

    def action_screenshot(self) -> None:
        """截图键（F7）：截图并立即提交。

        截图后端最多阻塞 120 秒（Windows 侧截图失败时会等待到超时后返回错误提示），
        所以整个流程放后台跑，界面不被卡住。
        截图管理器没有挂上时只写日志，不报错。
        """
        if self._global_hotkeys_active:
            log_event(
                "ui.hotkey.global_owned",
                "截图键由全局热键监听处理，界面忽略这次截图",
                level="DEBUG",
            )
            return
        manager = self._screenshot_manager
        if manager is None:
            log_event(
                "ui.hotkey",
                "快捷键截图键：截图管理器没有挂上，忽略这次截图",
                level="WARNING",
            )
            return
        self.show_screenshot_question()
        self._spawn(manager.capture_and_submit(), "capture_and_submit")

    def action_new_question(self) -> None:
        """Ctrl+N：创建新问题。

        右栏顶部的问题、检索来源与语音临时文字一起清掉，
        检索来源是当前会话的内容，新问题开始就不再留着。
        输入路由器也做一次会话级重置：会合中留的语音文字作废，
        旧会话的截图不带到新会话，见《8-集成与验收》请求取消一节。
        """
        self.clear_question()
        self.clear_rag_sources()
        self.clear_voice_text()
        self._call_manager("new_question")
        router = self._input_router
        if router is not None:
            router.reset_session()
            log_event(
                "ui.input_router_reset",
                "已重置输入路由器的会话状态",
                level="DEBUG",
            )

    def action_cancel(self) -> None:
        """Esc：检索来源浮层开着时先关掉浮层，否则取消当前模型任务。"""
        if self._close_rag_sources():
            return
        self._call_manager("cancel")

    def action_reanalyze(self) -> None:
        """F10：重新分析当前问题。"""
        self._start_generation("reanalyze")

    def action_show_rag_sources(self) -> None:
        """F3：打开或关闭检索来源面板。

        面板是临时浮层，列出当前会话的检索来源后再按 F3 或按 Esc 关掉，
        关掉后界面回到原来的两栏布局，见《6-Dify检索与项目知识》检索来源展示一节。
        """
        if self._close_rag_sources():
            return
        self.show_rag_sources(self._rag_sources)
        log_event("ui.hotkey", "快捷键 F3：打开检索来源面板")

    def action_quit(self) -> None:
        """Q、Ctrl+C、Ctrl+Q：退出应用。"""
        log_event("ui.hotkey", "快捷键：退出应用")
        self.exit()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _status_bar(self) -> StatusBar | None:
        """取状态栏，取不到时返回 None。"""
        try:
            return self.query_one(StatusBar)
        except NoMatches:
            return None

    def _question_header(self) -> QuestionHeader | None:
        """取右栏顶部的问题标题，取不到时返回 None。"""
        try:
            return self.query_one(QuestionHeader)
        except NoMatches:
            return None

    def _rag_sources_panel(self) -> RagSourcesPanel | None:
        """取检索来源浮层面板，取不到时返回 None。"""
        try:
            return self.query_one(RagSourcesPanel)
        except NoMatches:
            return None

    def _voice_text_panel(self) -> VoiceTextPanel | None:
        """取状态栏下方的临时识别文字行，取不到时返回 None。"""
        try:
            return self.query_one(VoiceTextPanel)
        except NoMatches:
            return None

    def _open_rag_sources(self) -> None:
        """打开检索来源浮层，列出当前会话的来源。"""
        panel = self._rag_sources_panel()
        if panel is None:
            # 浮层还没挂上，等下一次挂载
            return
        panel.show_sources(self._rag_sources)
        panel.display = True
        panel.focus()

    def _close_rag_sources(self) -> bool:
        """关掉检索来源浮层，浮层开着时返回 True。

        面板内容留着，只是不再显示；下一次按 F3 还能看到同一批来源。
        """
        panel = self._rag_sources_panel()
        if panel is None or not panel.display:
            return False
        panel.display = False
        try:
            # 焦点回到输入框，界面回到原来的两栏布局
            self.query_one(QuestionInputPanel).focus()
        except NoMatches:
            pass
        return True

    def _resolve_manager_method(self, method_name: str) -> Any | None:
        """取问题管理器上的方法。

        管理器没有挂上或没有这个方法时返回 None，并写一条日志。
        """
        manager = self._manager
        if manager is None:
            log_event(
                "ui.manager_missing",
                f"问题管理器没有挂上，忽略调用 {method_name}",
                level="WARNING",
            )
            return None
        method = getattr(manager, method_name, None)
        if method is None:
            log_event(
                "ui.manager_missing",
                f"问题管理器没有方法 {method_name}",
                level="WARNING",
            )
            return None
        return method

    def _resolve_target_method(self, target: Any, method_name: str) -> Any | None:
        """取指定对象上的方法。

        目标没有这个方法时返回 None，并写一条日志。

        :param target: 方法所在的对象。
        :param method_name: 方法名。
        """
        method = getattr(target, method_name, None)
        if method is None:
            log_event(
                "ui.manager_missing",
                f"目标对象没有方法 {method_name}",
                level="WARNING",
            )
            return None
        return method

    def _call_manager(self, method_name: str, *args: Any) -> None:
        """调用问题管理器的方法。

        同步方法直接调用，协程方法放后台跑。管理器没有挂上时只记日志。
        取消当正常收尾接住，其余异常只写日志，不让界面崩。
        """
        method = self._resolve_manager_method(method_name)
        if method is None:
            return
        try:
            result = method(*args)
        except asyncio.CancelledError:
            log_event(
                "ui.manager_cancelled",
                f"调用问题管理器方法 {method_name} 被取消",
                level="DEBUG",
            )
            return
        except Exception as exc:
            log_event(
                "ui.manager_failed",
                f"调用问题管理器方法 {method_name} 失败：{exc}",
                level="ERROR",
            )
            return
        if inspect.isawaitable(result):
            self._spawn(result, method_name)

    def _start_generation(
        self,
        method_name: str,
        *args: Any,
        target: Any | None = None,
    ) -> None:
        """开始一轮生成：MODEL 置 streaming，结束后置回 idle，出错时置 error。

        :param method_name: 方法名。
        :param args: 传给方法的参数。
        :param target: 方法所在的对象，默认是问题管理器；
            手工输入走输入路由器时传它。
        """
        if target is None:
            method = self._resolve_manager_method(method_name)
        else:
            method = self._resolve_target_method(target, method_name)
        if method is None:
            return
        self._generation_id += 1
        generation = self._generation_id

        async def _runner() -> None:
            await self._publish_model("streaming", GENERATING_HINT)
            try:
                result = method(*args)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                log_event(
                    "ui.manager_cancelled",
                    f"问题管理器方法 {method_name} 被取消",
                    level="DEBUG",
                )
                await self._finish_generation(generation)
                return
            except Exception as exc:
                log_event(
                    "ui.manager_failed",
                    f"问题管理器方法 {method_name} 执行失败：{exc}",
                    level="ERROR",
                )
                await self._publish_model(
                    "error", f"{STREAM_ERROR_CODE} {exc}"
                )
                return
            await self._finish_generation(generation)

        self._spawn(_runner(), method_name)

    async def _finish_generation(self, generation: int) -> None:
        """一轮生成的收尾：正常结束把 MODEL 置回 idle。

        只有最新一轮能改状态：上一轮被新问题取消时，状态归新一轮管。
        这一轮报过错的话状态留在 error，不改回 idle。
        """
        if generation != self._generation_id:
            return
        if self._error_generation == generation:
            return
        await self._publish_model("idle", "")

    async def _publish_model(self, model: str, hint: str) -> None:
        """改 MODEL 字段与提示位，其余五个字段按快照原样重发。"""
        fields = dict(self._status)
        fields["model"] = model
        fields["hint"] = hint
        await self.publish_status(**fields)

    async def publish_hint(self, hint: str) -> None:
        """只改提示位，其余五个状态字段按快照原样重发。

        提示位单独改时必须带全快照重发，否则其余字段会被默认值覆盖掉。
        这个方法也作为 ``hint_publisher`` 传给截图管理器。
        """
        fields = dict(self._status)
        fields["hint"] = hint
        await self.publish_status(**fields)

    def show_screenshot_question(self) -> None:
        """把截图提问显示到右栏顶部。

        TUI 内的 F9 与 socket 的 ``screenshot`` 动作是两条触发路径，
        都要在右栏顶部显示题面，见《4-截图输入与多模态》截图流程一节。
        """
        self.show_question(self._screenshot_question_text())

    def _screenshot_question_text(self) -> str:
        """截图提问在右栏顶部显示的题面。

        会话里已经合并了语音文字就用语音文字，否则用固定文案。
        """
        voice_text = getattr(self._manager, "voice_text", "") or ""
        return voice_text.strip() or SCREENSHOT_QUESTION_TEXT

    def _publish_model_later(self, model: str, hint: str) -> None:
        """在后台发布一次状态更新，发布不了就只记日志。"""
        try:
            self.run_worker(
                self._publish_model(model, hint),
                name="status",
                exit_on_error=False,
            )
        except Exception as exc:
            log_event(
                "ui.status_publish_failed",
                f"发布状态更新失败：{exc}",
                level="ERROR",
            )

    def _spawn(self, awaitable: Awaitable[Any], method_name: str) -> None:
        """把管理器的协程放到后台跑。

        取消当正常收尾接住，其余异常只写日志，不让界面崩。
        """

        async def _runner() -> None:
            try:
                await awaitable
            except asyncio.CancelledError:
                log_event(
                    "ui.manager_cancelled",
                    f"问题管理器方法 {method_name} 被取消",
                    level="DEBUG",
                )
            except Exception as exc:
                log_event(
                    "ui.manager_failed",
                    f"问题管理器方法 {method_name} 执行失败：{exc}",
                    level="ERROR",
                )

        try:
            self.run_worker(
                _runner(),
                name=f"manager-{method_name}",
                exit_on_error=False,
            )
        except Exception as exc:
            log_event(
                "ui.manager_failed",
                f"启动后台任务 {method_name} 失败：{exc}",
                level="ERROR",
            )
