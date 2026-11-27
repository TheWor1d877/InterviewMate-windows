"""状态栏下方的临时识别文字行。

语音采集过程中的临时识别文字显示在顶部状态栏下方的一行，不占用左右两栏，
见《5-语音输入与流式ASR》状态显示一节。

这一行由状态栏控件承载：面板挂在状态栏里面并停靠在底部，状态栏在有文字时
多占一行，没有文字时整行隐藏，状态栏回到两行。

组件只管显示：文字由界面在识别出临时结果时写进来，清空由界面在新建问题时触发。
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static

__all__ = ["VoiceTextPanel", "VOICE_LABEL"]


# 临时识别文字前面的标签，说明这一行是什么
VOICE_LABEL = "语音"


class VoiceTextPanel(Static):
    """状态栏下方的临时识别文字行。"""

    # 这一行不抢焦点，焦点留给输入框与检索来源浮层
    can_focus = False

    # 停靠在状态栏底部：有文字时占一行，没有文字时整块隐藏
    DEFAULT_CSS = """
    VoiceTextPanel {
        dock: bottom;
        height: 1;
        width: 100%;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        # 属性要在 super().__init__ 之前赋值，Textual 允许子类这样做
        self._text: str = ""
        # 宿主状态栏原本的高度，有文字时多占一行，清空时还回去
        self._host_height: int | None = None
        super().__init__("", markup=False, **kwargs)
        # 没有文字时整块隐藏，不占位置
        self.display = False

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def show_text(self, text: str) -> None:
        """显示一行临时识别文字。

        空串或纯空白按清空处理：整行隐藏，状态栏回到两行。
        文字超长时按控件宽度截断，不把布局撑开。

        :param text: 临时识别文字。
        """
        stripped = text.strip()
        if not stripped:
            self.clear()
            return
        self._text = stripped
        self.display = True
        self._grow_host()
        self.refresh(layout=True)

    def clear(self) -> None:
        """清空临时识别文字，整行隐藏。"""
        self._text = ""
        self.display = False
        self._restore_host()
        self.refresh(layout=True)

    @property
    def text(self) -> str:
        """当前临时识别文字，没有文字时是空串。"""
        return self._text

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self) -> Text:
        """返回临时识别文字行，放不下时按控件宽度截断。"""
        text = Text()
        text.append(f"{VOICE_LABEL} ", style="dim")
        text.append(self._text)
        width = self.size.width or 0
        if width > 0:
            text.truncate(width, overflow="ellipsis")
        return text

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _grow_host(self) -> None:
        """让宿主状态栏多占一行，给临时识别文字腾出位置。"""
        host = self.parent
        if host is None or self._host_height is not None:
            return
        height = host.styles.height
        base = height.value if height is not None else host.size.height
        self._host_height = base
        host.styles.height = base + 1

    def _restore_host(self) -> None:
        """把宿主状态栏的高度还回去。"""
        host = self.parent
        if host is None or self._host_height is None:
            return
        host.styles.height = self._host_height
        self._host_height = None
