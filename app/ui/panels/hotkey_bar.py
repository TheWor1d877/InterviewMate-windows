"""底部快捷键提示栏。

输入框下面一行，左边列出当前可用的快捷键，右边固定是退出键。
左边放不下时从右端去掉整项，不留半截，右边的退出键始终保留。

提示项里的键名取自配置里的 ``hotkeys`` 段，改配置这一栏跟着变。
"""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from app.config import HotkeySettings

__all__ = [
    "HotkeyBar",
    "HotkeyHints",
    "HOTKEY_SEPARATOR",
    "EXIT_HINT",
    "build_hotkey_items",
    "key_label",
]


# 相邻两段提示之间的分隔符
HOTKEY_SEPARATOR = " · "

# 右下角固定显示的退出键。配置里的 quit 键是 q，输入框有焦点时 q 是字母，
# 所以这里显示一直可用的 Ctrl+Q。
EXIT_HINT = "Ctrl+Q 退出"


def key_label(raw: str) -> str:
    """把配置里的键名转成显示用的标签。

    :param raw: 配置里的键名，例如 ``ctrl+n``、``escape``、``F10``。
    """
    key = raw.strip().lower()
    if key == "escape":
        return "Esc"
    if key.startswith("ctrl+"):
        return "Ctrl+" + key[len("ctrl+") :].upper()
    if len(key) > 1 and key[0] == "f" and key[1:].isdigit():
        return key.upper()
    return raw.strip().upper()


def build_hotkey_items(hotkeys: HotkeySettings) -> list[tuple[str, str]]:
    """按配置生成当前可用的快捷键提示。

    只列真正能用的键。F8、F9、F3 对应的功能在后面的里程碑里落地，
    落地之后再补进这一栏。

    :param hotkeys: 配置里的快捷键段。
    """
    items: list[tuple[str, str]] = [
        ("Enter", "提交"),
        ("Shift+Enter", "换行"),
        (key_label(hotkeys.new_question), "新问题"),
        (key_label(hotkeys.reanalyze), "重新分析"),
        (key_label(hotkeys.cancel), "取消"),
    ]
    return items


class HotkeyHints(Static):
    """左边的快捷键提示列表。

    宽度不够时从右端去掉整项，保留左边靠前的几项。
    """

    def __init__(self, items: list[tuple[str, str]], **kwargs: Any) -> None:
        # 属性要在 super().__init__ 之前赋值，Textual 允许子类这样做
        self._items: list[tuple[str, str]] = list(items)
        super().__init__("", markup=False, **kwargs)

    def render(self) -> str:
        """返回放得下的那几项提示。"""
        width = self.size.width or 0
        parts: list[str] = []
        used = 0
        for key, description in self._items:
            text = f"{key} {description}"
            extra = cell_len(text) + (len(HOTKEY_SEPARATOR) if parts else 0)
            if width > 0 and used + extra > width:
                break
            parts.append(text)
            used += extra
        return HOTKEY_SEPARATOR.join(parts)


class HotkeyBar(Horizontal):
    """底部快捷键提示栏。"""

    def __init__(
        self,
        items: list[tuple[str, str]],
        exit_hint: str = EXIT_HINT,
        **kwargs: Any,
    ) -> None:
        # 属性要在 super().__init__ 之前赋值，Textual 允许子类这样做
        self._items: list[tuple[str, str]] = list(items)
        self._exit_hint: str = exit_hint
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        """产出左边提示列表与右边的退出键提示。"""
        yield HotkeyHints(self._items, id="hotkey-main")
        yield Static(self._exit_hint, markup=False, id="hotkey-exit")
