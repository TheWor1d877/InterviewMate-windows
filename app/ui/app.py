"""Textual 应用。

界面四段结构见《3-文本输入与流式回答》终端界面一节，快捷键见同节快捷键表：
F8 语音开始与停止、F9 截图并立即提交、回车提交手动文本、Shift+回车换行、
Ctrl+N 创建新问题、F10 重新分析、Esc 取消、F3 查看检索来源、Q 退出。

界面这边只做四件事：把状态栏接到事件总线上，把输入框与快捷键接到问题管理器上，
把当前问题显示到右栏顶部，把检索来源与临时识别文字显示出来。
问题管理器由装配层建好后用 :meth:`InterviewMateApp.attach_manager` 挂进来，
截图管理器用 :meth:`InterviewMateApp.attach_screenshot_manager` 挂进来，
挂之前界面照常能起。

F8 的语音采集由装配层用 :meth:`InterviewMateApp.attach_voice_toggle` 接进来：
界面不 import 语音层与装配层的实现类，回调没接上时按 F8 只记日志、不报错。

检索来源面板是临时浮层，按 F3 打开、列出后按 Esc 或再按 F3 关掉，
见《6-Dify检索与项目知识》检索来源展示一节；临时识别文字显示在状态栏下方的一行，
见《5-语音输入与流式ASR》状态显示一节。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any, Awaitable

from textual import on
from textual.app import App, ComposeResult
from textual.css.query import NoMatches

from app.config import AppConfig, current_config, load_config
from app.core.event_bus import EventBus
from app.ui.layout import MainLayout
from app.ui.panels.answer_panel import AnswerPanel
from app.ui.panels.input_panel import QuestionInputPanel
from app.ui.panels.navigation_panel import NavigationPanel
from app.ui.panels.question_header import QuestionHeader
from app.ui.panels.rag_sources import RagSourcesPanel
from app.ui.panels.voice_text import VoiceTextPanel
from app.ui.status_bar import StatusBar, StatusUpdate
from app.utils.logger import log_event

__all__ = ["InterviewMateApp", "VoiceToggleCallback"]


# 状态栏订阅的事件类型名，与 StatusUpdate.type 一致
STATUS_EVENT_TYPE = "status.update"

# 按一次 F8 的回调：无参数，同步函数或协程函数都行。
# 语义与 socket 的 ``voice-toggle`` 一致：正在采集就停止并提交最终文本，
# 没在采集就开始采集，见《5-语音输入与流式ASR》F8 交互一节。
VoiceToggleCallback = Callable[[], Any]

# 生成过程中状态栏提示位显示的短提示
GENERATING_HINT = "生成中"

# 界面这边自己撞上的异常用这个错误码，与《2-骨架与配置》错误码表一致
STREAM_ERROR_CODE = "E_DEEPSEEK_STREAM"

# 截图提问在右栏顶部显示的题面，没有语音文字时用它
SCREENSHOT_QUESTION_TEXT = "截图提问"


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
        voice_toggle: VoiceToggleCallback | None = None,
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
        # 按一次 F8 的回调，由装配层挂进来，界面不 import 语音层的实现类
        self._voice_toggle: VoiceToggleCallback | None = voice_toggle
        # 当前会话的检索来源，普通字典，字段是 document_name、score、content
        # 来源存进会话后一直留着，模型输出结束也能再按 F3 查
        self._rag_sources: list[dict] = []
        # 最近一次状态更新的快照，改 DEEPSEEK 时按快照整体重发
        self._status: dict[str, Any] = {
            "running": "stopped",
            "audio": "off",
            "asr": "idle",
            "dify": "offline",
            "deepseek": "idle",
            "ttft_ms": None,
            "hint": "",
        }
        # 生成轮次编号，只有最新一轮能改状态栏的 DEEPSEEK 字段
        self._generation_id = 0
        # 报过错的那一轮编号，别的轮次不受影响
        self._error_generation = -1
        # 按配置绑定的快捷键：键名 -> action 名
        self._hotkey_bindings: dict[str, str] = {}
        self._bind_hotkeys()
        # 全局热键监听接管 F8 与 F9 后置 True，界面这边不再处理这两个键
        self._global_hotkeys_active: bool = False
        # 当前这段语音已经识别出来的累计文本，语音流结束时清零
        self._voice_text: str = ""

    # ------------------------------------------------------------------
    # 快捷键
    # ------------------------------------------------------------------
    def _bind_hotkeys(self) -> None:
        """按配置里的 hotkeys 段绑定快捷键。

        配置里的写法是 ``F8``、``ctrl+n``、``escape`` 这种，转成 Textual 的
        键名要小写。绑定后 :attr:`hotkey_bindings` 里是实际生效的键位。
        """
        hotkeys = self._config.hotkeys
        mapping = (
            (hotkeys.voice_toggle, "voice_toggle"),
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
        """标记全局热键监听已接管 F8 与 F9。

        全局监听直接读 ``/dev/input``，终端失焦时也能收到按键，做成按住说话；
        生效后同一次按键会被 Textual 绑定与全局监听各处理一次，
        所以界面这边要短路这两个动作，见《5-语音输入与流式ASR》F8 交互一节。
        全局监听不可用时不要调用本方法，界面绑定保持原来的语义：
        按一下开始、再按一下停止。
        """
        self._global_hotkeys_active = True
        log_event(
            "ui.hotkey.global_active",
            "全局热键监听已接管 F8 与 F9，界面不再处理这两个键",
        )

    @property
    def global_hotkeys_active(self) -> bool:
        """全局热键监听是否已接管 F8 与 F9。"""
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
        """订阅状态栏更新事件，挂上检索来源浮层，并把焦点给底部输入框。"""
        self._bus.subscribe(STATUS_EVENT_TYPE, self._handle_status_update)
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

    def attach_voice_toggle(self, callback: VoiceToggleCallback | None) -> None:
        """装配层把「按一次 F8」的回调挂进来。

        回调的语义与 socket 的 ``voice-toggle`` 一致：正在采集就停止并提交
        最终文本，没在采集就开始采集，见《5-语音输入与流式ASR》F8 交互一节。
        界面不 import 语音层与装配层的实现类，只调用这一个回调。

        :param callback: 无参数的回调，同步函数或协程函数都行。
        """
        self._voice_toggle = callback
        log_event("ui.voice_toggle_attached", "语音采集回调已挂到界面上")

    @property
    def voice_toggle(self) -> VoiceToggleCallback | None:
        """应用使用的语音采集回调，没有挂上时是 None。"""
        return self._voice_toggle

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
        """把错误显示到状态栏提示位，并把 DEEPSEEK 置成 error。

        :param code: 错误码，例如 ``E_DEEPSEEK_TIMEOUT``。
        :param detail: 一句话说明。
        """
        log_event("ui.error", f"{code} {detail}", level="ERROR", code=code)
        self._error_generation = self._generation_id
        hint = f"{code} {detail}"
        status_bar = self._status_bar()
        if status_bar is not None:
            # 提示位先立刻显示，状态字段随后由状态事件带过去
            status_bar.set_hint(hint)
        self._publish_deepseek_later("error", hint)

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
        # 记下最近一次的状态，改 DEEPSEEK 时按快照整体重发，其余字段不受影响
        self._status.update(
            running=event.running,
            audio=event.audio,
            asr=event.asr,
            dify=event.dify,
            deepseek=event.deepseek,
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
            deepseek=event.deepseek,
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
    def action_voice_toggle(self) -> None:
        """F8：语音开始与停止。

        真正做采集启停的回调由装配层注入，语义与 socket 的 ``voice-toggle``
        一致：正在采集就停止并提交最终文本，没在采集就开始采集，
        见《5-语音输入与流式ASR》F8 交互一节。
        回调没有接上时只记日志，不报错。
        """
        if self._global_hotkeys_active:
            log_event(
                "ui.hotkey.global_owned",
                "F8 由全局热键监听处理，界面忽略这次语音动作",
                level="DEBUG",
            )
            return
        callback = self._voice_toggle
        if callback is None:
            log_event(
                "ui.hotkey",
                "快捷键 F8：语音采集回调没有接上，忽略这次语音动作",
                level="WARNING",
            )
            return
        try:
            result = callback()
        except asyncio.CancelledError:
            log_event(
                "ui.voice_toggle_cancelled",
                "语音采集回调被取消",
                level="DEBUG",
            )
            return
        except Exception as exc:
            log_event(
                "ui.voice_toggle_failed",
                f"语音采集回调执行失败：{exc}",
                level="ERROR",
            )
            return
        if inspect.isawaitable(result):
            # 采集在事件循环里跑，放后台执行，界面不被卡住
            self._spawn(result, "voice_toggle")

    def action_screenshot(self) -> None:
        """F9：截图并立即提交。

        截图后端最多阻塞 120 秒（XDG Portal 兜底要等用户在弹窗里确认），
        所以整个流程放后台跑，界面不被卡住。
        截图管理器没有挂上时只写日志，不报错。
        """
        if self._global_hotkeys_active:
            log_event(
                "ui.hotkey.global_owned",
                "F9 由全局热键监听处理，界面忽略这次截图",
                level="DEBUG",
            )
            return
        manager = self._screenshot_manager
        if manager is None:
            log_event(
                "ui.hotkey",
                "快捷键 F9：截图管理器没有挂上，忽略这次截图",
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
        """开始一轮生成：DEEPSEEK 置 streaming，结束后置回 idle，出错时置 error。

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
            await self._publish_deepseek("streaming", GENERATING_HINT)
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
                await self._publish_deepseek(
                    "error", f"{STREAM_ERROR_CODE} {exc}"
                )
                return
            await self._finish_generation(generation)

        self._spawn(_runner(), method_name)

    async def _finish_generation(self, generation: int) -> None:
        """一轮生成的收尾：正常结束把 DEEPSEEK 置回 idle。

        只有最新一轮能改状态：上一轮被新问题取消时，状态归新一轮管。
        这一轮报过错的话状态留在 error，不改回 idle。
        """
        if generation != self._generation_id:
            return
        if self._error_generation == generation:
            return
        await self._publish_deepseek("idle", "")

    async def _publish_deepseek(self, deepseek: str, hint: str) -> None:
        """改 DEEPSEEK 字段与提示位，其余五个字段按快照原样重发。"""
        fields = dict(self._status)
        fields["deepseek"] = deepseek
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

    def _publish_deepseek_later(self, deepseek: str, hint: str) -> None:
        """在后台发布一次状态更新，发布不了就只记日志。"""
        try:
            self.run_worker(
                self._publish_deepseek(deepseek, hint),
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
