"""左侧「回答流程」面板。

面板显示模型给出的回答分层结构，每层一行：序号 + 标题。
当前层（第一个没完成的层）高亮，已完成的层标记完成状态。

代码题走另一种形态：左侧是解题阶梯，每步一行序号 + 标题，有提示时标题下面多一行，
见《7-代码问题模式》左侧解题思路一节。阶梯之后可以追加简短提示，
比如关键假设与代码块前后的说明文字。

形态是锁定的：已经有解题阶梯时，普通层的框架事件被忽略，
左栏已经显示的阶梯不会被后来的普通事件清掉。

面板宽度小时标题按可用宽度截断，不溢出；窄终端下左栏收成紧凑导航，
只显示层号与状态标记。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

from app.utils.logger import log_event

if TYPE_CHECKING:
    # 只在类型标注里用到，运行期不 import 输出层模块
    from app.output.event_models import FrameworkSection

__all__ = ["NavigationPanel", "COMPACT_CONTENT_WIDTH"]


# 面板内容宽度小于这个值时只显示层号，不显示标题
COMPACT_CONTENT_WIDTH = 12

# 状态标记：已完成、当前层、待处理
DONE_MARK = "✓"
CURRENT_MARK = "▶"
PENDING_MARK = " "

# 提示行的前缀与缩进
HINT_PREFIX = "· "
HINT_INDENT = "    "


def _step_field(step: Any, name: str) -> str:
    """取步骤对象上的一个字段，取不到时是空串。

    步骤既可以是带属性的对象，也可以是字典，这里两种都认。

    :param step: 一步解题步骤。
    :param name: 字段名，取 id、title、hint。
    """
    if isinstance(step, dict):
        value = step.get(name, "")
    else:
        value = getattr(step, name, "")
    if value is None:
        return ""
    return str(value)


class NavigationPanel(Static):
    """左侧「回答流程」面板。"""

    TITLE = "回答流程"
    # 解题阶梯形态下的栏标题
    STEPS_TITLE = "解题思路"
    # 面板可聚焦，焦点从输入框移过来之后 Q 键才退出程序
    can_focus = True

    def __init__(self, **kwargs: Any) -> None:
        # 属性要在 super().__init__ 之前赋值，Textual 允许子类这样做
        self._sections: list[FrameworkSection] = []
        # 代码题形态下的解题步骤
        self._steps: list[Any] = []
        # 解题阶梯之后的简短提示
        self._hints: list[str] = []
        self._done: set[str] = set()
        super().__init__("", markup=False, **kwargs)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def show_framework(self, sections: list[FrameworkSection]) -> None:
        """渲染层次列表：序号 + 标题，当前层高亮。

        重新给一份框架就重置完成状态，新框架从头算。
        已经有解题阶梯时这次调用被忽略，左栏已经显示的阶梯不会被清掉。

        :param sections: 模型给出的各层，顺序就是显示顺序。
        """
        if self._steps:
            log_event(
                "ui.navigation.ignore_framework",
                "解题阶梯形态下忽略普通框架事件",
                level="WARNING",
            )
            return
        self._sections = list(sections)
        self._steps = []
        self._hints = []
        self._done = set()
        self.refresh(layout=True)

    def mark_section_done(self, section_id: str) -> None:
        """把该层标记成已完成。

        :param section_id: 层的标识。
        """
        self._done.add(section_id)
        self.refresh(layout=True)

    def show_steps(self, steps: list[Any]) -> None:
        """渲染解题阶梯：序号 + 标题，有提示时标题下面多一行。

        步骤数量不写死，模型给几步就显示几步。重新给一份阶梯就重置完成状态，
        见《7-代码问题模式》左侧解题思路一节。

        :param steps: 解题步骤，每步带 id、title、hint 三个字段。
        """
        self._steps = list(steps)
        self._sections = []
        self._hints = []
        self._done = set()
        self.refresh(layout=True)

    def append_hint(self, text: str) -> None:
        """追加一条简短提示。

        提示排在解题阶梯后面，代码块前后的说明文字与关键假设都走这里。

        :param text: 提示内容，空串与纯空白忽略。
        """
        hint = text.strip()
        if not hint:
            return
        self._hints.append(hint)
        self.refresh(layout=True)

    def clear(self) -> None:
        """清空面板：两种形态的内容都清掉。

        清空界面意味着下一次请求是全新的，层次列表、解题阶梯与提示都归零。
        """
        self._sections = []
        self._steps = []
        self._hints = []
        self._done = set()
        self.refresh(layout=True)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self) -> Text:
        """返回面板内容：第一行是标题，后面是各层或解题阶梯。"""
        text = Text()
        title = Text(self._title(), style="bold")
        self._fit(title)
        text.append_text(title)
        text.append("\n")
        text.append_text(self._build_text())
        return text

    def plain_text(self) -> str:
        """返回各层的纯文本，不含标题行，供测试与调试用。"""
        return self._build_text().plain

    def _title(self) -> str:
        """栏标题：解题阶梯形态用另一个名字。"""
        return self.STEPS_TITLE if self._steps else self.TITLE

    def _build_text(self) -> Text:
        """按当前状态拼出各层内容。"""
        text = Text()
        lines = self._build_lines()
        for index, line in enumerate(lines):
            text.append_text(line)
            if index < len(lines) - 1:
                text.append("\n")
        return text

    def _build_lines(self) -> list[Text]:
        """按当前状态拼出各行内容。"""
        if self._steps:
            lines = self._step_lines()
        else:
            lines = self._section_lines()
        lines.extend(self._hint_lines())
        return lines

    def _section_lines(self) -> list[Text]:
        """拼出普通题的各层行。"""
        lines: list[Text] = []
        current_id = self._current_section_id()
        for index, section in enumerate(self._sections, start=1):
            line = self._section_line(index, section, section.id == current_id)
            self._fit(line)
            lines.append(line)
        return lines

    def _step_lines(self) -> list[Text]:
        """拼出解题阶梯的各行，标题下面可以跟一行提示。"""
        lines: list[Text] = []
        current_id = self._current_step_id()
        for index, step in enumerate(self._steps, start=1):
            line = self._step_line(index, step, _step_field(step, "id") == current_id)
            self._fit(line)
            lines.append(line)
            hint = _step_field(step, "hint")
            if not hint or self._content_width() < COMPACT_CONTENT_WIDTH:
                continue
            hint_line = Text()
            hint_line.append(HINT_INDENT)
            hint_line.append(hint, style="dim")
            self._fit(hint_line)
            lines.append(hint_line)
        return lines

    def _hint_lines(self) -> list[Text]:
        """拼出追加的提示行。"""
        lines: list[Text] = []
        if self._content_width() < COMPACT_CONTENT_WIDTH:
            # 窄栏只留层号与状态标记，提示也不显示
            return lines
        for hint in self._hints:
            line = Text()
            line.append(HINT_PREFIX)
            line.append(hint, style="dim")
            self._fit(line)
            lines.append(line)
        return lines

    def _fit(self, line: Text) -> None:
        """按面板内容宽度截断一行，放不下时末尾留省略号。"""
        width = self._content_width()
        if width > 0:
            line.truncate(width, overflow="ellipsis")

    def _section_line(
        self, index: int, section: FrameworkSection, is_current: bool
    ) -> Text:
        """拼出某一层的行内容。

        :param index: 层序号，从 1 开始。
        :param section: 这一层。
        :param is_current: 这一层是不是当前层。
        """
        if is_current:
            mark, style = CURRENT_MARK, "bold"
        elif section.id in self._done:
            mark, style = DONE_MARK, "dim"
        else:
            # 空样式就是默认样式，Rich 不认 normal 这个名字
            mark, style = PENDING_MARK, ""

        line = Text()
        if self._content_width() < COMPACT_CONTENT_WIDTH:
            # 窄栏：只留状态标记与层号
            line.append(mark, style=style)
            line.append(str(index), style=style)
            return line
        line.append(f"{mark} ", style=style)
        line.append(f"{index}. ", style=style)
        line.append(section.title, style=style)
        return line

    def _step_line(self, index: int, step: Any, is_current: bool) -> Text:
        """拼出某一步的行内容，做法跟层次列表一致。

        :param index: 步骤序号，从 1 开始。
        :param step: 这一步。
        :param is_current: 这一步是不是当前步。
        """
        step_id = _step_field(step, "id")
        if is_current:
            mark, style = CURRENT_MARK, "bold"
        elif step_id in self._done:
            mark, style = DONE_MARK, "dim"
        else:
            mark, style = PENDING_MARK, ""

        line = Text()
        if self._content_width() < COMPACT_CONTENT_WIDTH:
            # 窄栏：只留状态标记与步号
            line.append(mark, style=style)
            line.append(str(index), style=style)
            return line
        line.append(f"{mark} ", style=style)
        line.append(f"{index}. ", style=style)
        line.append(_step_field(step, "title"), style=style)
        return line

    def _current_section_id(self) -> str | None:
        """当前层：第一个没完成的层。没有层时返回 None。"""
        for section in self._sections:
            if section.id not in self._done:
                return section.id
        return None

    def _current_step_id(self) -> str | None:
        """当前步：第一个没完成的步骤。没有步骤时返回 None。"""
        for step in self._steps:
            if _step_field(step, "id") not in self._done:
                return _step_field(step, "id")
        return None

    def _content_width(self) -> int:
        """面板内容区宽度，布局还没算出来时返回 0。"""
        return self.size.width or 0
