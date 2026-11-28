"""项目模式的双路临时转写区。

项目模式开启时，状态栏下方按角色分行显示「我」与「面试官」两路临时转写：
事件总线上的 ``dialogue.partial`` 是某一路还在识别中的临时文字，
``dialogue.sentence`` 是已经定稿的一句话。

临时行随识别持续更新：同一路的临时文字来了就替换该路最新一行；
定稿句来了就把该路的临时行就地落定，没有临时行时直接追加一行。
最新的行加粗高亮，只保留最近 :data:`MAX_LINES` 行，超出从最旧丢弃。

面板挂在状态栏控件里：显示时状态栏按行数撑高，清空或退出项目模式时还原。
普通模式的单路临时文字由 :class:`~app.ui.panels.voice_text.VoiceTextPanel`
显示，两个面板由界面按模式互斥切换，不做同时显示。

组件只管显示：文字由界面订阅事件后写进来，清空由界面在新一轮开始时触发。
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static

__all__ = ["DialogueTempPanel", "MAX_LINES", "UNKNOWN_ROLE", "PARTIAL_MARK"]


# 临时区最多保留的行数，超出的从最旧一行丢弃
MAX_LINES = 3

# 临时文字行尾的省略标记，说明这一路还在识别中
PARTIAL_MARK = "…"

# 角色为空时的占位
UNKNOWN_ROLE = "?"


class DialogueTempPanel(Static):
    """状态栏下方的双路临时转写区。"""

    # 这一行不抢焦点，焦点留给输入框
    can_focus = False

    # 停靠在状态栏底部：有内容时按行数占位，没有内容时整块隐藏
    DEFAULT_CSS = """
    DialogueTempPanel {
        dock: bottom;
        height: auto;
        width: 100%;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        # 条目是 (角色, 文本, 是否临时)，按时间先后排列
        self._entries: list[tuple[str, str, bool]] = []
        # 项目模式开启时才显示
        self._active: bool = False
        # 当前向宿主状态栏借用的行数，隐藏或清空时还回去
        self._borrowed: int = 0
        super().__init__("", markup=False, **kwargs)
        # 没有内容时整块隐藏，不占位置
        self.display = False

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def show_partial(self, role: str, text: str) -> None:
        """更新某一路还在识别中的临时转写。

        这一路的最新一行本来就是临时行就替换它的文本，
        否则追加一行新的临时文字。

        :param role: 说话角色，例如 ``我``、``面试官``。
        :param text: 这一路的临时转写，空白忽略。
        """
        stripped = text.strip()
        if not stripped:
            return
        name = role.strip() or UNKNOWN_ROLE
        last = self._entries[-1] if self._entries else None
        if last is not None and last[0] == name and last[2]:
            self._entries[-1] = (name, stripped, True)
        else:
            self._entries.append((name, stripped, True))
        self._sync_after_change()

    def show_sentence(self, role: str, text: str) -> None:
        """写入一路已经定稿的句子。

        这一路的最新一行是临时行就把它就地落定成定稿句，
        否则直接追加一行定稿句。

        :param role: 说话角色，例如 ``我``、``面试官``。
        :param text: 定稿句，空白忽略。
        """
        stripped = text.strip()
        if not stripped:
            return
        name = role.strip() or UNKNOWN_ROLE
        last = self._entries[-1] if self._entries else None
        if last is not None and last[0] == name and last[2]:
            self._entries[-1] = (name, stripped, False)
        else:
            self._entries.append((name, stripped, False))
        self._sync_after_change()

    def clear(self) -> None:
        """清空临时转写，整块隐藏并把状态栏的高度还回去。"""
        self._entries = []
        self._sync_after_change()

    def set_active(self, active: bool) -> None:
        """项目模式开启时显示，退出后隐藏。

        有内容但未开启时不显示，重新开启时原来的内容接着显示。

        :param active: 是否处于项目模式。
        """
        self._active = bool(active)
        self._sync_after_change()

    @property
    def entries(self) -> list[tuple[str, str, bool]]:
        """当前条目列表的拷贝，每项是 (角色, 文本, 是否临时)。"""
        return list(self._entries)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self) -> Text:
        """返回临时转写各行，最新一行加粗，放不下时按宽度截断。"""
        text = Text()
        lines = self._build_lines()
        for index, line in enumerate(lines):
            text.append_text(line)
            if index < len(lines) - 1:
                text.append("\n")
        return text

    def _build_lines(self) -> list[Text]:
        """按当前条目拼出各行。"""
        lines: list[Text] = []
        total = len(self._entries)
        for index, (role, entry_text, is_partial) in enumerate(self._entries):
            shown = f"[{role}] {entry_text}"
            if is_partial:
                shown += PARTIAL_MARK
            line = Text()
            # 最新一行加粗高亮；空样式就是默认样式，Rich 不认 normal 这个名字
            line.append(shown, style="bold" if index == total - 1 else "")
            width = self.size.width or 0
            if width > 0:
                line.truncate(width, overflow="ellipsis")
            lines.append(line)
        return lines

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _sync_after_change(self) -> None:
        """内容或模式变化后重算显示与占用行数。"""
        if len(self._entries) > MAX_LINES:
            self._entries = self._entries[-MAX_LINES:]
        self.display = self._active and bool(self._entries)
        self._sync_host()
        self.refresh(layout=True)

    def _sync_host(self) -> None:
        """把需要的行数同步给宿主状态栏，增量借还，不覆盖别的面板的占用。"""
        host = self.parent
        if host is None:
            return
        needed = len(self._entries) if self.display else 0
        delta = needed - self._borrowed
        self._borrowed = needed
        if delta == 0:
            return
        height = host.styles.height
        base = height.value if height is not None else host.size.height
        if base is None:
            base = host.size.height
        host.styles.height = base + delta