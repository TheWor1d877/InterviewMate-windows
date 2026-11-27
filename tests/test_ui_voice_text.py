"""状态栏下方临时识别文字行的测试。

覆盖《5-语音输入与流式ASR》状态显示一节：临时识别文字显示在顶部状态栏
下方的一行，不占用左右两栏，没有文字时不占位置。

测试用 Textual 的 ``run_test()`` 无头驱动，配置显式构造，
不依赖开发者本地的 ``config/config.yaml``。
"""

from __future__ import annotations

from rich.cells import cell_len

from app.ui.app import InterviewMateApp
from app.ui.layout import MainLayout
from app.ui.panels.voice_text import VoiceTextPanel
from app.ui.status_bar import StatusBar
from tests.test_ui_skeleton import make_config


# 宽终端的宽度
WIDE_WIDTH = 120

# 状态栏本身占的行数，临时识别文字排在它下面
STATUS_BAR_ROWS = 2


async def test_voice_text_panel_is_hidden_without_text() -> None:
    """没有文字时整行隐藏，不占位置。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(VoiceTextPanel)
        assert panel.display is False
        assert panel.text == ""


async def test_voice_text_panel_shows_text() -> None:
    """给了文字之后这一行显示出来，内容是那段文字。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(VoiceTextPanel)
        panel.show_text("这是一段临时识别文字")
        await pilot.pause()
        assert panel.display is True
        assert panel.text == "这是一段临时识别文字"
        assert "这是一段临时识别文字" in panel.render().plain


async def test_voice_text_panel_empty_string_clears_it() -> None:
    """传空串按清空处理：文字没了，整行也隐藏。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(VoiceTextPanel)
        panel.show_text("临时文字")
        await pilot.pause()
        panel.show_text("")
        await pilot.pause()
        assert panel.display is False
        assert panel.text == ""
        assert "临时文字" not in panel.render().plain


async def test_voice_text_panel_blank_text_clears_it() -> None:
    """只有空白的文字按清空处理。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(VoiceTextPanel)
        panel.show_text("临时文字")
        await pilot.pause()
        panel.show_text("   \t ")
        await pilot.pause()
        assert panel.display is False
        assert panel.text == ""


async def test_voice_text_panel_truncates_long_text() -> None:
    """文字超长时按这一行的宽度截断，不把布局撑开。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(VoiceTextPanel)
        panel.show_text("临时" * 200)
        await pilot.pause()
        await pilot.pause()
        assert panel.size.height == 1
        assert cell_len(panel.render().plain) <= panel.size.width


async def test_layout_keeps_four_sections_with_the_voice_text_row() -> None:
    """主布局还是四段结构，临时识别文字行不占其中一段。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        layout = app.query_one(MainLayout)
        assert len(list(layout.children)) == 4
        assert app.query_one("#voice-text") is not None


async def test_voice_text_row_sits_below_the_status_bar() -> None:
    """临时识别文字排在状态栏下面，不在左右两栏里。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        voice = app.query_one(VoiceTextPanel)
        voice.show_text("临时识别文字")
        await pilot.pause()
        await pilot.pause()
        assert voice.region.y >= bar.region.y + STATUS_BAR_ROWS
        assert voice.region.y + voice.region.height <= bar.region.y + bar.region.height


async def test_show_voice_text_updates_the_row() -> None:
    """界面的 show_voice_text 把文字写到那一行。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_voice_text("识别中的句子")
        await pilot.pause()
        panel = app.query_one(VoiceTextPanel)
        assert panel.display is True
        assert "识别中的句子" in panel.render().plain


async def test_clear_voice_text_clears_the_row() -> None:
    """界面的 clear_voice_text 把那一行清掉。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_voice_text("识别中的句子")
        await pilot.pause()
        app.clear_voice_text()
        await pilot.pause()
        panel = app.query_one(VoiceTextPanel)
        assert panel.display is False
        assert panel.text == ""


async def test_new_question_clears_the_voice_text() -> None:
    """Ctrl+N 新建问题时临时识别文字一起清掉。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_voice_text("识别中的句子")
        await pilot.pause()
        app.action_new_question()
        await pilot.pause()
        panel = app.query_one(VoiceTextPanel)
        assert panel.display is False
        assert panel.text == ""
