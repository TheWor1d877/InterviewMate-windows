"""项目模式 UI 层的测试。

覆盖双路临时转写、F10 上下文视图切换、状态栏模式与会话指示、
快捷键提示栏随模式切换、F11 清空后的视图刷新。

测试全部用 Textual 的 ``run_test()`` 无头驱动，配置在文件内显式构造，
不依赖开发者本地的 ``config/config.yaml``。
"""

from __future__ import annotations

import pytest

from app.core.event_bus import Event, EventBus
from app.ui.app import InterviewMateApp
from app.ui.panels.context_view import ContextView
from app.ui.panels.dialogue_temp import DialogueTempPanel
from app.ui.panels.hotkey_bar import HotkeyBar
from app.ui.panels.question_header import QuestionHeader
from app.ui.panels.voice_text import VoiceTextPanel
from app.ui.status_bar import StatusBar
from tests.test_ui_skeleton import make_config


# 宽终端的宽度
WIDE_WIDTH = 120


class _DialoguePartial(Event):
    """测试用的双路转写临时事件，字段与引擎层契约一致。"""

    type: str = "dialogue.partial"
    role: str = "我"
    text: str = ""


class _DialogueSentence(Event):
    """测试用的双路转写定稿事件。"""

    type: str = "dialogue.sentence"
    role: str = "面试官"
    text: str = ""


# ----------------------------------------------------------------------
# 双路临时转写区
# ----------------------------------------------------------------------
async def test_dialogue_panel_hidden_by_default() -> None:
    """没有内容且不在项目模式时整块隐藏。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(DialogueTempPanel)
        assert panel.display is False
        assert panel.entries == []


async def test_dialogue_panel_shows_two_roles_in_lines() -> None:
    """两路转写按角色分行显示，临时行带省略标记。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(DialogueTempPanel)
        panel.set_active(True)
        panel.show_partial("我", "这个项目用了")
        await pilot.pause()
        assert panel.display is True
        plain = panel.render().plain
        assert "[我] 这个项目用了…" in plain
        panel.show_sentence("面试官", "介绍一下架构")
        await pilot.pause()
        plain = panel.render().plain
        assert "[面试官] 介绍一下架构" in plain
        assert "…" not in plain.splitlines()[-1]


async def test_partial_updates_the_same_role_line() -> None:
    """同一路的临时文字来了就替换该路最新一行，不追加。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(DialogueTempPanel)
        panel.set_active(True)
        panel.show_partial("我", "这个项目")
        panel.show_partial("我", "这个项目用了")
        await pilot.pause()
        assert panel.entries == [("我", "这个项目用了", True)]
        plain = panel.render().plain
        assert "这个项目用了" in plain
        assert "这个项目…" not in plain


async def test_sentence_settles_the_partial_line() -> None:
    """定稿句来了就把同一路的临时行就地落定。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(DialogueTempPanel)
        panel.set_active(True)
        panel.show_partial("我", "这个项目用了")
        panel.show_sentence("我", "这个项目用了 Redis")
        await pilot.pause()
        assert panel.entries == [("我", "这个项目用了 Redis", False)]
        assert "…" not in panel.render().plain


async def test_dialogue_panel_keeps_recent_lines_only() -> None:
    """临时区只保留最近几行，超出从最旧丢弃。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(DialogueTempPanel)
        panel.set_active(True)
        for index in range(5):
            panel.show_sentence("面试官", f"第{index}句")
        await pilot.pause()
        entries = panel.entries
        assert len(entries) <= 3
        assert "第0句" not in panel.render().plain
        assert "第4句" in panel.render().plain


async def test_project_mode_controls_the_dialogue_panel() -> None:
    """进入项目模式后有内容就显示，退出后隐藏且数据清掉。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(DialogueTempPanel)
        app.show_dialogue_partial("我", "识别中的话")
        await pilot.pause()
        # 不在项目模式，有内容也不显示
        assert panel.display is False
        app.set_project_mode(True, session_id="sess-1")
        await pilot.pause()
        assert panel.display is True
        app.set_project_mode(False)
        await pilot.pause()
        assert panel.display is False
        assert panel.entries == []


async def test_project_mode_clears_the_voice_text_row() -> None:
    """进入项目模式时普通模式的单路语音行让位清空。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_voice_text("普通模式的语音行")
        await pilot.pause()
        app.set_project_mode(True)
        await pilot.pause()
        voice = app.query_one(VoiceTextPanel)
        assert voice.display is False
        assert voice.text == ""


# ----------------------------------------------------------------------
# 事件总线链路
# ----------------------------------------------------------------------
async def test_dialogue_partial_event_updates_the_panel() -> None:
    """事件总线上的 dialogue.partial 事件写进双路临时区。"""
    bus = EventBus()
    app = InterviewMateApp(config=make_config(), bus=bus)
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.set_project_mode(True)
        await bus.publish(_DialoguePartial(ts=0.0, role="我", text="正在说的话"))
        await pilot.pause()
        panel = app.query_one(DialogueTempPanel)
        assert panel.entries == [("我", "正在说的话", True)]


async def test_dialogue_sentence_event_updates_the_panel() -> None:
    """事件总线上的 dialogue.sentence 事件写进双路临时区。"""
    bus = EventBus()
    app = InterviewMateApp(config=make_config(), bus=bus)
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.set_project_mode(True)
        await bus.publish(_DialogueSentence(ts=0.0, role="面试官", text="已定稿的句子"))
        await pilot.pause()
        panel = app.query_one(DialogueTempPanel)
        assert panel.entries == [("面试官", "已定稿的句子", False)]


# ----------------------------------------------------------------------
# F10 上下文视图
# ----------------------------------------------------------------------
async def test_context_view_toggles_with_the_action() -> None:
    """action_toggle_context_view 开关上下文视图，右栏回答视图跟着让位。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        view = app.query_one(ContextView)
        assert view.display is False
        app.action_toggle_context_view()
        await pilot.pause()
        assert app.context_view_open is True
        assert view.display is True
        # 回答视图让位
        assert app.query_one("#answer-scroll").display is False
        app.action_toggle_context_view()
        await pilot.pause()
        assert app.context_view_open is False
        assert view.display is False
        assert app.query_one("#answer-scroll").display is True


async def test_context_view_shows_the_provider_text() -> None:
    """打开视图时现取一次注入的日志全文。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.attach_text_provider(
            lambda: "[我] 你好\n[面试官] 请介绍你的项目"
        )
        app.action_toggle_context_view()
        await pilot.pause()
        view = app.query_one(ContextView)
        plain = view.plain_text()
        assert "[我] 你好" in plain
        assert "[面试官] 请介绍你的项目" in plain


async def test_context_view_rereads_on_reopen() -> None:
    """关掉再打开会重新取日志，显示的是最新内容。"""
    app = InterviewMateApp(config=make_config())
    log = ["[我] 第一句"]
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.attach_text_provider(lambda: "\n".join(log))
        app.action_toggle_context_view()
        await pilot.pause()
        log.append("[面试官] 第二句")
        app.action_toggle_context_view()
        await pilot.pause()
        app.action_toggle_context_view()
        await pilot.pause()
        assert "[面试官] 第二句" in app.query_one(ContextView).plain_text()


async def test_context_view_without_provider_does_not_crash() -> None:
    """数据源没注入时打开视图显示提示，界面不崩。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.action_toggle_context_view()
        await pilot.pause()
        assert app.is_running
        assert "上下文数据源未接入" in app.query_one(ContextView).plain_text()


async def test_opening_context_view_hides_the_question_header() -> None:
    """右栏顶部的问题在上下文视图打开时让位，关闭时有题目就回来。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_question("什么是 RAII")
        await pilot.pause()
        header = app.query_one(QuestionHeader)
        assert header.display is True
        app.action_toggle_context_view()
        await pilot.pause()
        assert header.display is False
        app.action_toggle_context_view()
        await pilot.pause()
        assert header.display is True


# ----------------------------------------------------------------------
# 状态栏模式与会话指示
# ----------------------------------------------------------------------
async def test_status_bar_shows_mode_and_session() -> None:
    """项目模式开启后状态栏显示 MODE=项目 与会话 ID。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        text = bar.render()
        assert "MODE=普通" in text
        assert "SESSION=-" in text
        app.set_project_mode(True, session_id="sess-42")
        await pilot.pause()
        text = bar.render()
        assert "MODE=project" not in text
        assert "MODE=项目" in text
        assert "SESSION=sess-42" in text


async def test_leaving_project_mode_restores_status_bar() -> None:
    """退出项目模式后状态栏回到普通模式，会话 ID 不再显示。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.set_project_mode(True, session_id="sess-42")
        await pilot.pause()
        app.set_project_mode(False)
        await pilot.pause()
        text = app.query_one(StatusBar).render()
        assert "MODE=普通" in text
        assert "SESSION=-" in text
        assert app.project_mode is False


async def test_status_bar_rejects_unknown_mode() -> None:
    """状态栏的模式字段只认 normal / project，非法取值抛 ValueError。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        with pytest.raises(ValueError) as excinfo:
            bar.set_mode("coding")
        assert "mode" in str(excinfo.value)


# ----------------------------------------------------------------------
# 快捷键提示栏随模式切换
# ----------------------------------------------------------------------
async def test_hotkey_bar_switches_with_project_mode() -> None:
    """进入项目模式提示栏换成项目模式键位，退出后恢复配置键位。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        bar = app.query_one(HotkeyBar)
        normal_text = bar.query_one("#hotkey-main").render()
        assert "Ctrl+N 新问题" in normal_text
        assert "F8 标记问题并提交" not in normal_text

        app.set_project_mode(True)
        await pilot.pause()
        assert bar.mode == "project"
        project_text = bar.query_one("#hotkey-main").render()
        assert "F7 截图" in project_text
        assert "F8 标记问题并提交" in project_text
        assert "F9 退出项目模式" in project_text
        assert "F10 上下文" in project_text
        assert "F11 重开一轮" in project_text

        app.set_project_mode(False)
        await pilot.pause()
        assert bar.mode == "normal"
        assert "Ctrl+N 新问题" in bar.query_one("#hotkey-main").render()


# ----------------------------------------------------------------------
# F11 清空后的视图刷新
# ----------------------------------------------------------------------
async def test_clear_dialogue_clears_the_panel_and_refreshes_the_view() -> None:
    """clear_dialogue 清掉双路临时区；上下文视图开着时重新取日志。"""
    app = InterviewMateApp(config=make_config())
    log = ["[我] 旧的一轮"]
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.set_project_mode(True)
        app.show_dialogue_sentence("我", "旧的一轮")
        await pilot.pause()
        app.attach_text_provider(lambda: "\n".join(log))
        app.action_toggle_context_view()
        await pilot.pause()
        # 集成层清完引擎数据后调界面刷新
        log.clear()
        app.clear_dialogue()
        await pilot.pause()
        panel = app.query_one(DialogueTempPanel)
        assert panel.entries == []
        assert "暂无对话记录" in app.query_one(ContextView).plain_text()


async def test_clear_dialogue_when_view_closed_keeps_it_closed() -> None:
    """上下文视图没开时 clear_dialogue 只清临时区，不开视图。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.set_project_mode(True)
        app.show_dialogue_partial("我", "识别中")
        await pilot.pause()
        app.clear_dialogue()
        await pilot.pause()
        assert app.query_one(DialogueTempPanel).entries == []
        assert app.context_view_open is False