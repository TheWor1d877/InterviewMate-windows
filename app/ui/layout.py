"""布局骨架。

界面是四段结构，见《3-文本输入与流式回答》终端界面一节：
顶部状态栏、中间左右两栏内容区、底部文字输入框、输入框下面的快捷键提示栏。

状态栏下面还有一行临时识别文字，见《5-语音输入与流式ASR》状态显示一节：
这一行挂在状态栏控件里面，有文字时状态栏多占一行，没有文字时不占位置，
左右两栏的结构不变。

左右比例默认 22 比 78；终端宽度小于配置的 ``min_width`` 时左栏收成紧凑导航，
右栏保持足够空间，两栏结构在窄终端下不变。
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.events import Resize

from app.config import AppConfig
from app.ui.panels.hotkey_bar import HotkeyBar, build_hotkey_items
from app.ui.panels.input_panel import QuestionInputPanel
from app.ui.panels.navigation_panel import NavigationPanel
from app.ui.panels.normal_view import NormalView
from app.ui.panels.voice_text import VoiceTextPanel
from app.ui.status_bar import StatusBar

__all__ = [
    "COMPACT_NAV_WIDTH",
    "MainLayout",
]


# 窄终端下左栏收成的紧凑导航宽度（单位：列），见《3-文本输入与流式回答》终端界面一节
COMPACT_NAV_WIDTH = 8


class MainLayout(Vertical):
    """主布局：状态栏 + 左右两栏 + 底部输入区。"""

    def __init__(self, config: AppConfig, **kwargs: Any) -> None:
        self._config = config
        super().__init__(**kwargs)

    @property
    def config(self) -> AppConfig:
        """本布局使用的配置。"""
        return self._config

    def compose(self) -> ComposeResult:
        """依次产出四段结构：状态栏、左右两栏、底部输入区、快捷键提示栏。"""
        yield StatusBar(self._config.app.name, id="status-bar")
        yield NormalView(id="content-columns")
        yield QuestionInputPanel(id="input-area")
        yield HotkeyBar(
            build_hotkey_items(self._config.hotkeys),
            id="hotkey-bar",
        )

    async def on_mount(self) -> None:
        """挂上状态栏下方的临时识别文字行，再按当前宽度算一次两栏宽度。"""
        await self.query_one(StatusBar).mount(VoiceTextPanel(id="voice-text"))
        self.call_after_refresh(self._apply_current_width)

    def on_resize(self, event: Resize) -> None:
        """终端尺寸变化时重新分配两栏宽度。"""
        self._apply_width(event.size.width)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _apply_current_width(self) -> None:
        """按当前自身宽度重新分配两栏。"""
        self._apply_width(self.size.width)

    def _apply_width(self, width: int) -> None:
        """按给定宽度分配左右两栏。

        宽度不小于 ``ui.min_width`` 时按 ``ui.left_ratio`` / ``ui.right_ratio``
        划分；小于时左栏收成 :data:`COMPACT_NAV_WIDTH` 列的紧凑导航，
        右栏占满剩余空间，两栏结构不变。
        """
        if width <= 0:
            return
        try:
            left = self.query_one("#left-panel", NavigationPanel)
            right = self.query_one("#right-column", Vertical)
        except NoMatches:
            # 子部件还没挂上，等下一次 resize 再算
            return

        ui = self._config.ui
        if width < ui.min_width:
            left_width = COMPACT_NAV_WIDTH
        else:
            left_width = int(width * ui.left_ratio / 100)
        right_width = max(width - left_width, 1)

        left.styles.width = left_width
        right.styles.width = right_width
