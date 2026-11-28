"""普通题的两栏视图。

左侧是「回答流程」，右侧是「关键回答内容」，见《3-文本输入与流式回答》终端界面一节。

右栏是竖着两段：上面是当前问题，下面是回答内容。问题由界面在提交时写进
``QuestionHeader``，没有问题时这一段隐藏。
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll

from app.ui.panels.answer_panel import AnswerPanel
from app.ui.panels.context_view import ContextView
from app.ui.panels.navigation_panel import NavigationPanel
from app.ui.panels.question_header import QuestionHeader

__all__ = ["NormalView"]


class NormalView(Horizontal):
    """普通题两栏：左侧回答流程 + 右侧问题与内容。

    右栏还挂着一个默认隐藏的上下文视图：项目模式按 F10 时右栏临时
    切成对话日志全文，再按一次切回问题与回答内容，两栏结构不变。
    """

    def compose(self) -> ComposeResult:
        """产出左栏与右栏两个部分。"""
        yield NavigationPanel(id="left-panel")
        with Vertical(id="right-column"):
            yield QuestionHeader(id="question-header")
            # 回答内容放进滚动容器，鼠标滚轮才能上下翻；标题固定在上面
            with VerticalScroll(id="answer-scroll"):
                yield AnswerPanel(id="right-panel")
            # 上下文视图默认隐藏，项目模式按 F10 时顶替问题与回答内容
            yield ContextView(id="context-view")
