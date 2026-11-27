"""界面面板。

M2 起，界面拆成几个面板，目录结构见《3-文本输入与流式回答》：

- :class:`~app.ui.panels.navigation_panel.NavigationPanel`：左侧「回答流程」；
- :class:`~app.ui.panels.answer_panel.AnswerPanel`：右侧「关键回答内容」；
- :class:`~app.ui.panels.input_panel.QuestionInputPanel`：底部输入框；
- :class:`~app.ui.panels.normal_view.NormalView`：普通题的两栏组合。

M3 之后新增的界面只放在这个目录下，不再改 M1 留下的界面文件。
"""

from __future__ import annotations

from app.ui.panels.answer_panel import AnswerPanel
from app.ui.panels.input_panel import QuestionInputPanel
from app.ui.panels.navigation_panel import NavigationPanel
from app.ui.panels.normal_view import NormalView

__all__ = [
    "AnswerPanel",
    "NavigationPanel",
    "NormalView",
    "QuestionInputPanel",
]
