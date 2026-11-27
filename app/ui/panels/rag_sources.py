"""检索来源浮层面板。

面板列出当前问题用到的检索片段：文档名、分数、正文前若干行，
按 F3 打开，见《6-Dify检索与项目知识》检索来源展示一节。

面板是临时的浮层，列出后立即返回两栏布局，不改变主界面的两栏结构。
片段由界面从当前会话取出，转换成普通字典传进来，字段是 document_name、
score、content 三项；面板不 import 检索层的实现类。
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static

__all__ = ["RagSourcesPanel", "PREVIEW_LINES"]


# 正文预览最多显示的行数，超出的部分截断并加省略标记
PREVIEW_LINES = 4

# 没有片段时的提示，不显示空白
EMPTY_TEXT = "当前问题没有检索片段"

# 分数保留的小数位
SCORE_DIGITS = 3

# 正文预览每行前面的缩进
PREVIEW_INDENT = "  "

# 正文超出预览行数时的省略标记
ELLIPSIS = "…"


def _entry_field(entry: dict, name: str) -> Any:
    """取片段字典里的一个字段，字段缺失时是空串。

    :param entry: 一条片段。
    :param name: 字段名，取 document_name、score、content。
    """
    if not isinstance(entry, dict):
        return ""
    value = entry.get(name)
    if value is None:
        return ""
    return value


def _format_score(score: Any) -> str:
    """把分数格式化成保留三位的字符串，没有分数时是空串。

    :param score: 分数，可能是浮点数，也可能是 None。
    """
    if score is None or score == "":
        return ""
    try:
        return f"{float(score):.{SCORE_DIGITS}f}"
    except (TypeError, ValueError):
        return str(score)


class RagSourcesPanel(Static):
    """检索来源浮层面板。"""

    TITLE = "检索来源"

    # 面板可聚焦，焦点从输入框移过来之后 Q 键才退出程序
    can_focus = True

    # 浮层自带边框与滚动，片段多的时候在面板内部滚动，不改主界面的两栏结构
    DEFAULT_CSS = """
    RagSourcesPanel {
        width: 100%;
        height: auto;
        max-height: 100%;
        padding: 0 1;
        border: solid $border;
        overflow-y: auto;
        background: $surface;
        color: $text;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        # 属性要在 super().__init__ 之前赋值，Textual 允许子类这样做
        self._entries: list[Any] = []
        super().__init__("", markup=False, **kwargs)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def show_sources(self, entries: list[dict]) -> None:
        """列出当前问题用到的检索片段。

        每一项是一个字典，字段是 document_name、score、content。
        文档名可以为空串，这时不显示文档名那一行；分数可以是 None，
        这时不显示分数那一行；正文为空时只显示文档名与分数。

        :param entries: 片段列表，顺序就是显示顺序。
        """
        self._entries = [
            dict(entry) if isinstance(entry, dict) else entry for entry in entries
        ]
        self.refresh(layout=True)

    def clear(self) -> None:
        """清空片段，回到没有片段的状态。"""
        self._entries.clear()
        self.refresh(layout=True)

    @property
    def entries(self) -> list[Any]:
        """当前列出的片段。"""
        return list(self._entries)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self) -> Text:
        """返回面板内容：第一行是标题，后面是片段列表。"""
        text = Text()
        title = Text(self.TITLE, style="bold")
        self._fit(title)
        text.append_text(title)
        text.append("\n")
        text.append_text(self._build_text())
        return text

    def plain_text(self) -> str:
        """返回片段列表的纯文本，不含标题行，供测试与调试用。"""
        return self._build_text().plain

    def _build_text(self) -> Text:
        """按当前片段拼出面板内容。"""
        text = Text()
        lines = self._build_lines()
        for index, line in enumerate(lines):
            text.append_text(line)
            if index < len(lines) - 1:
                text.append("\n")
        return text

    def _build_lines(self) -> list[Text]:
        """按当前片段拼出各行内容。"""
        if not self._entries:
            empty = Text()
            empty.append(EMPTY_TEXT, style="dim")
            return [empty]

        lines: list[Text] = []
        for index, entry in enumerate(self._entries):
            if index:
                # 两条片段之间空一行，读起来清楚些
                lines.append(Text())
            lines.extend(self._entry_lines(entry))
        return lines

    def _entry_lines(self, entry: dict) -> list[Text]:
        """拼出一个片段的几行：文档名、分数、正文预览。

        :param entry: 一条片段。
        """
        lines: list[Text] = []

        document_name = _entry_field(entry, "document_name")
        if document_name:
            name_line = Text()
            name_line.append(str(document_name), style="bold")
            self._fit(name_line)
            lines.append(name_line)

        score_text = _format_score(_entry_field(entry, "score"))
        if score_text:
            score_line = Text()
            score_line.append("分数 ", style="dim")
            score_line.append(score_text, style="dim")
            self._fit(score_line)
            lines.append(score_line)

        lines.extend(self._preview_lines(_entry_field(entry, "content")))
        return lines

    def _preview_lines(self, content: Any) -> list[Text]:
        """拼出正文预览，最多 PREVIEW_LINES 行，超出的部分加省略标记。

        :param content: 片段正文。
        """
        if not isinstance(content, str):
            return []
        content_lines = content.splitlines()
        if not content_lines:
            return []

        lines: list[Text] = []
        for raw in content_lines[:PREVIEW_LINES]:
            line = Text()
            line.append(PREVIEW_INDENT)
            line.append(raw.strip())
            self._fit(line)
            lines.append(line)
        if len(content_lines) > PREVIEW_LINES:
            mark = Text()
            mark.append(PREVIEW_INDENT)
            mark.append(ELLIPSIS)
            self._fit(mark)
            lines.append(mark)
        return lines

    def _fit(self, line: Text) -> None:
        """按面板内容宽度截断一行，放不下时末尾留省略号。"""
        width = self.size.width or 0
        if width > 0:
            line.truncate(width, overflow="ellipsis")
