"""项目模式的上下文视图。

按 F10 临时把右栏切成对话日志全文：每行带角色标注，原样滚动显示，
再按一次切回回答视图，见《9-项目模式》上下文视图一节。

日志全文由界面注入的 ``text_provider`` 现取现显：打开视图与收到清空刷新时
各取一次，视图不缓存旧日志，也不 import 对话引擎的实现模块。
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

__all__ = ["ContextView", "CONTEXT_TITLE", "EMPTY_TEXT"]


# 视图标题
CONTEXT_TITLE = "上下文"

# 没有日志时的提示，不显示空白
EMPTY_TEXT = "（暂无对话记录）"


class ContextView(VerticalScroll):
    """右栏的上下文视图，显示对话日志全文。"""

    def __init__(self, **kwargs: Any) -> None:
        # 最近一次显示的正文，数据源没挂或取失败时是提示文本
        self._body: str = ""
        super().__init__(**kwargs)
        # 默认隐藏，按 F10 打开
        self.display = False

    def compose(self) -> ComposeResult:
        """产出承载日志全文的文本块。"""
        yield Static("", markup=False, id="context-content")

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def show_text(self, text: str) -> None:
        """显示一段日志全文，标题在第一行，滚动回到顶部。

        :param text: 对话日志全文，空白按暂无记录处理。
        """
        self._body = text.strip() or EMPTY_TEXT
        content = Text()
        content.append(CONTEXT_TITLE, style="bold")
        content.append("\n")
        content.append(self._body)
        self.query_one("#context-content", Static).update(content)
        self.refresh()
        self.scroll_home(animate=False)

    def clear(self) -> None:
        """清空日志全文。"""
        self._body = ""
        self.query_one("#context-content", Static).update("")
        self.refresh()

    @property
    def body(self) -> str:
        """当前显示的正文，没有内容时是空串。"""
        return self._body

    def plain_text(self) -> str:
        """返回标题与正文的纯文本，供测试与调试用。"""
        if not self._body:
            return ""
        return f"{CONTEXT_TITLE}\n{self._body}"