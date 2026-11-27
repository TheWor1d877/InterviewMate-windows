"""把模型流式事件投递给界面面板。

分发器不 import 界面类，只按下面这组方法名调用导航面板与内容面板：

- ``navigation.show_framework(sections)``
- ``navigation.mark_section_done(section_id)``
- ``navigation.clear()``
- ``answer.append_item(section_id, keyword, sentence)``
- ``answer.append_fallback(text)``
- ``answer.clear()``

代码题模式另有一组方法名，见《7-代码问题模式》流式事件一节：

- ``navigation.show_steps(steps)``
- ``navigation.append_hint(text)``
- ``answer.show_steps(steps)``
- ``answer.append_code(text)``
- ``answer.enter_code_mode()``
- ``answer.complete_code(language, standard, submission_mode)``

分发器自己记 framework 声明过的 section id。``item`` 的 section_id 不在集合里时，
内容落兜底区，原始内容不丢；``section_done`` 的 section_id 不在集合里时忽略。

截图请求的第一行是路由事件，见《4-截图输入与多模态》：收到时先清空上一题的残留，
再按 ``kind`` 预置右栏形态，``coding`` 进代码栏，``normal`` 进简答栏；
之后 framework 到达时只渲染框架，不改已经定下的形态。

代码题兜底：一次代码题请求结束时（``code_complete`` 或 ``done``）一个代码片段都没收到，
右栏显示一行提示并记一条 WARNING，不静默留白。
"""

from __future__ import annotations

from typing import Any, Callable

from app.output.event_models import (
    CodeCompleteEvent,
    CodeDeltaEvent,
    CodeFrameworkEvent,
    DoneEvent,
    FallbackEvent,
    FrameworkEvent,
    ItemEvent,
    ROUTE_CODING,
    ROUTE_KINDS,
    ROUTE_NORMAL,
    RouteEvent,
    SectionDoneEvent,
    StreamEvent,
)
from app.output.response_normalizer import ResponseNormalizer
from app.utils.logger import log_event

__all__ = ["EventDispatcher", "EMPTY_CODE_NOTICE"]


# 代码题没有返回代码时右栏显示的一行提示
EMPTY_CODE_NOTICE: str = "本次未返回代码，按 F10 重试"


class EventDispatcher:
    """把流式事件送到导航面板与内容面板。

    :param navigation: 左侧导航面板，实现 ``show_framework``、``mark_section_done`` 与 ``clear``。
    :param answer: 右侧内容面板，实现 ``append_item``、``append_fallback`` 与 ``clear``。
    :param error_reporter: 可选的错误上报回调，签名是 ``(code, detail)``；
        不给的时候只写日志。
    """

    def __init__(
        self,
        navigation: Any,
        answer: Any,
        error_reporter: Callable[[str, str], None] | None = None,
    ) -> None:
        self._navigation = navigation
        self._answer = answer
        self._error_reporter = error_reporter
        # framework 声明过的 section id
        self._section_ids: set[str] = set()
        # 是否已经收到 done 事件
        self._done = False
        # 当前是不是代码题模式，由收到代码题事件推导，
        # 也可以由装配层显式告知，见《7-代码问题模式》纯代码块兜底一节
        self._coding_mode = False
        # 这次请求路由事件给出的 kind，还没收到路由事件时是 None。
        # 有值时 framework 只渲染内容，不改右栏形态
        self._route_kind: str | None = None
        # 这次请求收到过非空的代码片段
        self._code_received = False
        # 空代码提示已经报过，同一次请求只报一次
        self._empty_code_reported = False

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def dispatch(self, event: StreamEvent) -> None:
        """按事件类型分派到一个事件。"""
        if isinstance(event, RouteEvent):
            self._handle_route(event)
        elif isinstance(event, CodeFrameworkEvent):
            self._handle_code_framework(event)
        elif isinstance(event, FrameworkEvent):
            self._handle_framework(event)
        elif isinstance(event, ItemEvent):
            self._handle_item(event)
        elif isinstance(event, SectionDoneEvent):
            self._handle_section_done(event)
        elif isinstance(event, DoneEvent):
            self._handle_done(event)
        elif isinstance(event, CodeDeltaEvent):
            self._handle_code_delta(event)
        elif isinstance(event, CodeCompleteEvent):
            self._handle_code_complete(event)
        elif isinstance(event, FallbackEvent):
            self._handle_fallback(event)
        else:
            log_event(
                "output.dispatch.unknown",
                "收到不认识的事件类型",
                level="WARNING",
                request_id=getattr(event, "request_id", None),
                session_id=getattr(event, "session_id", None),
                event_type=getattr(event, "type", "-"),
            )

    def clear(self) -> None:
        """清空两个面板，并清掉自己记的 section 集合与完成状态。

        清空界面意味着下一次请求是全新的，左栏的解题阶梯与提示也一并清掉，
        代码题模式重置回假。
        """
        self._navigation.clear()
        self._answer.clear()
        self._section_ids.clear()
        self._done = False
        self._coding_mode = False
        self._route_kind = None
        self._code_received = False
        self._empty_code_reported = False

    def set_coding_mode(self, enabled: bool) -> None:
        """显式设置当前是不是代码题模式。

        模型完全不按事件协议输出时，一个代码题事件都不会来，
        分发器只能靠装配层在判定出代码题时告知，
        见《7-代码问题模式》纯代码块兜底一节。

        切到代码题模式时右栏先切到代码形态等着，代码随后再追加进来。

        :param enabled: True 表示当前是代码题模式，False 表示回到普通题模式。
        """
        if enabled:
            self._enter_coding_mode()
            return
        self._coding_mode = False

    def report_error(self, code: str, detail: str) -> None:
        """上报一个错误。

        有 ``error_reporter`` 就调它，没有就只写日志。
        """
        log_event(
            "output.dispatch.error",
            "上报错误",
            level="ERROR",
            code=code,
            detail=detail,
        )
        if self._error_reporter is not None:
            self._error_reporter(code, detail)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        """是否已经收到 done 事件。"""
        return self._done

    @property
    def section_ids(self) -> frozenset[str]:
        """当前 framework 声明过的 section id 集合。"""
        return frozenset(self._section_ids)

    @property
    def coding_mode(self) -> bool:
        """当前是不是代码题模式。"""
        return self._coding_mode

    # ------------------------------------------------------------------
    # 各事件的处理
    # ------------------------------------------------------------------
    def _handle_route(self, event: RouteEvent) -> None:
        """按路由事件清空上一题残留并预置右栏形态。

        见《4-截图输入与多模态》：``coding`` 进代码栏、``normal`` 进简答栏。
        形态定下之后，后面的 framework 只渲染框架，不再改形态。
        ``kind`` 不在允许取值里时按 ``normal`` 处理。

        :param event: 路由事件。
        """
        kind = event.kind if event.kind in ROUTE_KINDS else ROUTE_NORMAL
        # 清空上一题残留：两个面板与分发器自己记的状态一起归零
        self._navigation.clear()
        self._answer.clear()
        self._section_ids.clear()
        self._done = False
        self._route_kind = kind
        self._code_received = False
        self._empty_code_reported = False
        if kind == ROUTE_CODING:
            self._coding_mode = True
            self._answer.enter_code_mode()
        else:
            self._coding_mode = False
            enter_normal = getattr(self._answer, "enter_normal_mode", None)
            if callable(enter_normal):
                enter_normal()
        log_event(
            "output.dispatch.route",
            "按路由事件预置右栏形态",
            level="INFO",
            request_id=event.request_id,
            session_id=event.session_id,
            kind=kind,
        )

    def _handle_framework(self, event: FrameworkEvent) -> None:
        """渲染框架，并重新记一遍声明过的 section id。"""
        self._section_ids = {section.id for section in event.sections}
        self._navigation.show_framework(event.sections)

    def _handle_item(self, event: ItemEvent) -> None:
        """渲染一个内容单元；section 未知时落兜底区。"""
        if event.section_id in self._section_ids:
            self._answer.append_item(event.section_id, event.keyword, event.sentence)
            return
        text = ResponseNormalizer.format_item(event.keyword, event.sentence)
        self._answer.append_fallback(text)
        log_event(
            "output.dispatch.unknown_section",
            "item 的 section_id 不在框架里，内容落兜底区",
            level="WARNING",
            request_id=event.request_id,
            session_id=event.session_id,
            section_id=event.section_id,
        )

    def _handle_section_done(self, event: SectionDoneEvent) -> None:
        """标记某一层完成；section 未知时忽略。"""
        if event.section_id in self._section_ids:
            self._navigation.mark_section_done(event.section_id)
            return
        log_event(
            "output.dispatch.unknown_section",
            "section_done 的 section_id 不在框架里，忽略",
            level="DEBUG",
            request_id=event.request_id,
            session_id=event.session_id,
            section_id=event.section_id,
        )

    def _handle_fallback(self, event: FallbackEvent) -> None:
        """把兜底文本送到内容面板，进面板前先整理一遍。

        代码题模式下按《7-代码问题模式》纯代码块兜底一节拆成两路：
        代码正文进右栏代码区，说明文字进左侧提示。
        普通题模式照旧整体落兜底区。
        """
        if not self._coding_mode:
            self._answer.append_fallback(ResponseNormalizer.normalize_fallback(event.text))
            return
        code, note = ResponseNormalizer.extract_code_blocks(event.text)
        if code:
            self._answer.append_code(code)
        if note:
            self._navigation.append_hint(note)

    def _handle_code_framework(self, event: CodeFrameworkEvent) -> None:
        """渲染解题阶梯，并记下当前是代码题模式。

        左栏渲染阶梯，右栏代码上方也展示同一份步骤与思路。
        已经收到路由事件时形态由路由定下，这里只渲染阶梯，不再改形态，
        见《4-截图输入与多模态》。
        """
        if self._route_kind is None:
            self._enter_coding_mode()
        self._navigation.show_steps(event.steps)
        # 右栏代码上方也给一份思路，代码本身不再夹带解释
        show_steps = getattr(self._answer, "show_steps", None)
        if callable(show_steps):
            show_steps(event.steps)

    def _handle_code_delta(self, event: CodeDeltaEvent) -> None:
        """追加一段代码，并记下当前是代码题模式。

        路由已经判定为普通题时整段忽略：右栏的 ``append_code`` 会无条件切到代码形态，
        追进来会把简答视图顶掉，后续普通表达单元也被忽略，见《4-截图输入与多模态》。
        """
        if self._route_kind == ROUTE_NORMAL:
            log_event(
                "output.dispatch.code_delta_in_normal_route",
                "普通题路由下收到代码增量事件，已忽略",
                level="WARNING",
                request_id=event.request_id,
                session_id=event.session_id,
                route_kind=self._route_kind,
            )
            return
        self._enter_coding_mode()
        self._answer.append_code(event.text)
        if event.text.strip():
            self._code_received = True

    def _handle_code_complete(self, event: CodeCompleteEvent) -> None:
        """代码给完了，把语言、标准与提交模式交给内容面板，并检查空代码。

        路由已经判定为普通题时整段忽略：``complete_code`` 同样会把右栏切成代码形态，
        普通题也不存在空代码兜底，见《4-截图输入与多模态》。
        """
        if self._route_kind == ROUTE_NORMAL:
            log_event(
                "output.dispatch.code_complete_in_normal_route",
                "普通题路由下收到代码完成事件，已忽略",
                level="WARNING",
                request_id=event.request_id,
                session_id=event.session_id,
                route_kind=self._route_kind,
            )
            return
        self._answer.complete_code(event.language, event.standard, event.submission_mode)
        self._report_empty_code(event)

    def _handle_done(self, event: DoneEvent) -> None:
        """整体完成，并检查这次代码题有没有返回代码。"""
        self._done = True
        self._report_empty_code(event)
        # 流结束时把追加节流中挂起的刷新立刻补上，保证结束态布局完整
        flush_pending = getattr(self._answer, "flush_pending", None)
        if callable(flush_pending):
            flush_pending()

    def _report_empty_code(self, event: StreamEvent) -> None:
        """代码题一个代码片段都没收到时，右栏给一行提示并记一条 WARNING。

        只对代码题模式生效，同一次请求只报一次；右栏已经有代码时不报。
        见《4-截图输入与多模态》与《7-代码问题模式》：不许静默留白。

        :param event: 触发这次检查的事件，用来带 request_id 与 session_id。
        """
        if not self._coding_mode or self._empty_code_reported:
            return
        accumulated = getattr(self._answer, "code_text", "")
        if self._code_received or (isinstance(accumulated, str) and accumulated.strip()):
            return
        self._empty_code_reported = True
        notice = getattr(self._answer, "show_code_notice", None)
        if callable(notice):
            notice(EMPTY_CODE_NOTICE)
        log_event(
            "coding.empty_code",
            "代码题没有返回代码，右栏给出重试提示",
            level="WARNING",
            request_id=event.request_id,
            session_id=event.session_id,
        )

    def _enter_coding_mode(self) -> None:
        """记下当前是代码题模式，并让右栏先切到代码形态。

        右栏先切形态等着，代码随后再追加进来，
        见《7-代码问题模式》右侧完整代码一节。
        """
        self._coding_mode = True
        self._answer.enter_code_mode()
