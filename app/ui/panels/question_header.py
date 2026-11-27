"""右栏顶部的问题标题。

当前问题显示在右栏最上面，题目加粗，底下一条分隔线把题目与回答内容分开。
没有问题时整块隐藏，不占位置。

组件只管显示：题目由界面在提交时写进来，清空由界面在新建问题时触发。
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static

__all__ = ["QuestionHeader", "QUESTION_LABEL"]


# 题目前面的标签，说明这一行是什么
QUESTION_LABEL = "问题"


class QuestionHeader(Static):
    """右栏顶部的当前问题。"""

    def __init__(self, **kwargs: Any) -> None:
        # 属性要在 super().__init__ 之前赋值，Textual 允许子类这样做
        self._question: str = ""
        super().__init__("", markup=False, **kwargs)
        # 没有问题时整块隐藏，右栏直接显示回答内容
        self.display = False

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def show_question(self, question: str) -> None:
        """显示当前问题，替换上一个问题。

        :param question: 题面文本，首尾空白会被去掉。
        """
        self._question = question.strip()
        self.display = True
        self.refresh(layout=True)

    def clear_question(self) -> None:
        """清空问题，整块隐藏。"""
        self._question = ""
        self.display = False
        self.refresh(layout=True)

    @property
    def question(self) -> str:
        """当前问题文本，没有问题时是空串。"""
        return self._question

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self) -> Text:
        """返回问题行：先是标签，再是题目本身。"""
        text = Text()
        text.append(f"{QUESTION_LABEL} ", style="dim")
        text.append(self._question, style="bold")
        return text
