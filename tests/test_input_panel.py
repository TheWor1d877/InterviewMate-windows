"""底部输入框的粘贴行为测试。

覆盖三条粘贴通道：

- ``Ctrl+V`` 读系统剪贴板插入，读不到时经状态栏提示位提示终端原生粘贴键；
- 鼠标中键读 PRIMARY 选区插入；
- 终端原生粘贴（:class:`~textual.events.Paste` 事件）照常写进输入框，
  不被按键绑定吞掉。
"""

from __future__ import annotations

import asyncio

import pytest
from textual import events

from app.ui.app import InterviewMateApp
from app.ui.panels.input_panel import PASTE_HINT, QuestionInputPanel
from app.ui.status_bar import StatusBar
from tests.test_ui_skeleton import make_config

# 宽终端宽度，与其它界面用例一致
WIDE_WIDTH = 120


async def _wait_for_hint(
    ui: InterviewMateApp, needle: str, timeout: float = 5.0
) -> str:
    """等状态栏提示位出现指定字样，返回提示位当前的文字。"""
    bar = ui.query_one(StatusBar)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if needle in bar.hint:
            return bar.hint
        await asyncio.sleep(0.02)
    return bar.hint


def _fake_clipboard(monkeypatch: pytest.MonkeyPatch, text: str) -> list[bool]:
    """把剪贴板读取换成固定返回值，返回收到的 primary 参数列表。"""
    calls: list[bool] = []

    def fake_read_clipboard(*, primary: bool = False) -> str:
        calls.append(primary)
        return text

    monkeypatch.setattr(
        "app.ui.panels.input_panel.read_clipboard", fake_read_clipboard
    )
    return calls


async def test_ctrl_v_pastes_system_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl+V 读系统剪贴板并插入输入框。"""
    calls = _fake_clipboard(monkeypatch, "粘贴进来的内容")
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()

        await pilot.press("ctrl+v")
        await pilot.pause()

        assert panel.text == "粘贴进来的内容"
        assert calls == [False]


async def test_ctrl_v_falls_back_to_hint_when_clipboard_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """剪贴板读不到时内容不变，状态栏提示终端原生粘贴键。"""
    _fake_clipboard(monkeypatch, "")
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.text = "已经输入的内容"
        panel.focus()
        await pilot.pause()

        await pilot.press("ctrl+v")
        hint = await _wait_for_hint(app, "Ctrl+Shift+V")
        await pilot.pause()

        assert hint == PASTE_HINT
        assert panel.text == "已经输入的内容"


async def test_middle_click_pastes_primary_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """鼠标中键读 PRIMARY 选区并插入。"""
    calls = _fake_clipboard(monkeypatch, "选中的文字")
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()

        event = events.MouseDown(
            widget=panel,
            x=1,
            y=0,
            delta_x=0,
            delta_y=0,
            button=2,
            shift=False,
            meta=False,
            ctrl=False,
        )
        await panel._on_mouse_down(event)
        await pilot.pause()

        assert panel.text == "选中的文字"
        assert calls == [True]


async def test_paste_event_still_fills_the_input() -> None:
    """终端原生粘贴照常写进输入框，多行都保留。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()

        app.post_message(events.Paste("第一行\n第二行"))
        await pilot.pause()

        assert panel.text == "第一行\n第二行"