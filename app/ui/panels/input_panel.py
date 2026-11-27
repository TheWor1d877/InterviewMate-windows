"""底部输入框。

输入框用 Textual 的多行控件 :class:`~textual.widgets.TextArea`：

- 回车提交当前文本，发 :class:`QuestionInputPanel.Submitted` 事件并清空输入框；
- Shift+回车与 Ctrl+回车插入换行，不提交。多数终端发不出 Shift+回车，
  这时回车就是提交，这是终端限制；
- 粘贴多行题面保留全部行，由 TextArea 自己处理，不截断；
- Ctrl+V 读系统剪贴板后插入，鼠标中键读 PRIMARY 选区后插入，
  两条路都走 :func:`app.utils.clipboard.read_clipboard`；
  读不到时经状态栏提示位提示终端原生粘贴键（Ctrl+Shift+V 或右键粘贴）；
  终端原生粘贴发的是 :class:`~textual.events.Paste` 事件，照常进输入框。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea

from app.utils.clipboard import read_clipboard
from app.utils.logger import log_event

__all__ = ["QuestionInputPanel", "INPUT_PLACEHOLDER", "PASTE_HINT"]


# 输入框没有内容时显示的提示文字
INPUT_PLACEHOLDER = "输入问题..."

# 插入换行的键
NEWLINE_KEYS = ("shift+enter", "ctrl+enter")

# 剪贴板读不到时给状态栏提示位的一句话
PASTE_HINT = "剪贴板读不到，请用 Ctrl+Shift+V 或右键粘贴"


class QuestionInputPanel(TextArea):
    """底部输入框。回车提交，Shift+回车与 Ctrl+回车换行。"""

    # Ctrl+V 覆盖 TextArea 自带的粘贴：OSC52 在 GNOME Terminal 上不可用，
    # 改成直接读系统剪贴板后插入
    BINDINGS = [Binding("ctrl+v", "paste_clipboard", "粘贴", show=False)]

    @dataclass
    class Submitted(Message):
        """提交事件。"""

        text: str

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "",
            soft_wrap=True,
            show_line_numbers=False,
            placeholder=INPUT_PLACEHOLDER,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def clear_input(self) -> None:
        """清空输入框内容。"""
        self.text = ""

    async def action_paste_clipboard(self) -> None:
        """Ctrl+V：读系统剪贴板插入到光标处，读不到时提示终端原生粘贴键。"""
        log_event("ui.input.paste", "Ctrl+V 读系统剪贴板", level="DEBUG")
        await self._paste_from_clipboard(primary=False)

    async def _paste_from_clipboard(self, *, primary: bool) -> None:
        """读一次剪贴板并插入；读不到时给提示。

        读剪贴板是子进程调用，放进线程里跑，不阻塞按键处理。

        :param primary: True 读 PRIMARY 选区，鼠标中键走这条。
        """
        text = await asyncio.to_thread(read_clipboard, primary=primary)
        if text:
            self.insert(text)
            return
        log_event(
            "ui.input.paste_empty",
            "剪贴板为空或读取失败，提示终端原生粘贴键",
            level="DEBUG",
        )
        await self._publish_hint(PASTE_HINT)

    async def _publish_hint(self, message: str) -> None:
        """把提示送到状态栏，界面没有这个方法时退回界面通知。"""
        app = self.app
        publisher = getattr(app, "publish_hint", None)
        if publisher is None:
            notifier = getattr(app, "notify", None)
            if callable(notifier):
                notifier(message)
            return
        try:
            await publisher(message)
        except Exception as exc:
            log_event(
                "ui.input.paste_hint_failed",
                f"粘贴提示送不到界面：{exc}",
                level="WARNING",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # 鼠标与按键
    # ------------------------------------------------------------------
    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        """鼠标中键粘贴 PRIMARY 选区，其余按键交给 TextArea。

        终端开启鼠标上报后，中键在应用内没有原生行为，
        这里补上与桌面一致的「中键粘贴选中内容」。

        :param event: 鼠标按下事件。
        """
        if event.button == 2:
            event.stop()
            event.prevent_default()
            await self._paste_from_clipboard(primary=True)
            return
        await super()._on_mouse_down(event)

    async def _on_key(self, event: events.Key) -> None:
        """回车提交，Shift+回车与 Ctrl+回车换行，其余键交给 TextArea。"""
        key = event.key
        if key == "enter":
            # 自己处理，不让 TextArea 把回车当换行插进去
            event.stop()
            event.prevent_default()
            self._submit()
            return
        if key in NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)

    def _submit(self) -> None:
        """提交当前文本：去掉首尾空白，发事件并清空输入框。"""
        text = self.text.strip()
        if not text:
            log_event("ui.input.empty", "输入框为空，不提交", level="DEBUG")
            return
        log_event("ui.input.submit", "输入框提交", text_length=len(text))
        self.post_message(self.Submitted(text=text))
        self.clear_input()
