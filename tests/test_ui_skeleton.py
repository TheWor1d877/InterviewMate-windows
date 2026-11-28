"""UI 骨架的测试。

覆盖《3-文本输入与流式回答》终端界面一节的三段结构、快捷键绑定，
以及《2-骨架与配置》状态字段取值表里的状态取值域。

测试全部用 Textual 的 ``run_test()`` 无头驱动，配置在文件内显式构造，
不依赖开发者本地的 ``config/config.yaml``。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from textual.color import Color
from textual.containers import Horizontal, Vertical, VerticalScroll

from app.config import AppConfig, current_config, set_current_config
from app.core.event_bus import EventBus
from app.ui.app import InterviewMateApp
from app.ui.layout import COMPACT_NAV_WIDTH, MainLayout
from app.ui.panels.answer_panel import AnswerPanel
from app.ui.panels.hotkey_bar import HotkeyBar
from app.ui.panels.input_panel import QuestionInputPanel
from app.ui.panels.navigation_panel import NavigationPanel
from app.ui.panels.normal_view import NormalView
from app.ui.panels.question_header import QuestionHeader
from app.ui.status_bar import (
    ASR_VALUES,
    AUDIO_VALUES,
    MODEL_VALUES,
    DIFY_VALUES,
    RUNNING_VALUES,
    STATUS_LABELS,
    STATUS_ORDER,
    StatusBar,
    StatusUpdate,
)


APP_NAME = "InterviewMate"

# 宽终端与窄终端的宽度，窄终端 80 小于配置的 min_width 90
WIDE_WIDTH = 120
NARROW_WIDTH = 80
MIN_WIDTH = 90


@pytest.fixture(autouse=True)
def _isolate_process_config() -> Iterator[None]:
    """每个用例前后都把进程级配置清掉，用例之间互不影响。"""
    saved = current_config()
    set_current_config(None)
    try:
        yield
    finally:
        set_current_config(saved)


def _config_data(
    hotkeys: dict[str, str] | None = None, **ui_overrides: Any
) -> dict[str, Any]:
    """构造与仓库默认配置等价的数据。

    ``ui_overrides`` 覆盖 ui 段，``hotkeys`` 覆盖 hotkeys 段。
    """
    ui: dict[str, Any] = {
        "left_ratio": 22,
        "right_ratio": 78,
        "min_width": MIN_WIDTH,
        "syntax_highlighting": True,
        "auto_scroll": True,
        "show_status_bar": True,
        "show_text_input": True,
    }
    ui.update(ui_overrides)
    hotkey_values: dict[str, Any] = {
        "voice_toggle": "F8",
        "screenshot": "F9",
        "new_question": "ctrl+n",
        "cancel": "escape",
        "reanalyze": "F10",
        "show_rag_sources": "F3",
        "quit": "q",
    }
    hotkey_values.update(hotkeys or {})
    return {
        "app": {"name": APP_NAME, "language": "zh-CN", "log_level": "INFO"},
        "ui": ui,
        "hotkeys": hotkey_values,
        "audio": {
            "enabled": True,
            "source": "auto",
            "sample_rate": 16000,
            "channels": 1,
            "chunk_duration_ms": 200,
        },
        "asr": {
            "enabled": True,
            "model": "paraformer-zh-stream",
            "device": "auto",
            "cpu_fallback": True,
            "vad_enabled": True,
            "preload": True,
        },
        "screenshot": {
            "enabled": True,
            "target": "screen",
            "max_edge": 1600,
            "format": "png",
            "persist_files": False,
        },
        "model": {
            "base_url": "https://api.deepseek.com",
            "name": "deepseek-flash",
            "stream": True,
            "thinking": "disabled",
            "reasoning_effort": "low",
            "temperature": 0.3,
            "max_output_tokens": 2048,
            "connect_timeout": 5,
            "read_timeout": 60,
            "max_retries": 1,
        },
        "rag": {
            "enabled": True,
            "top_k": 4,
            "timeout": 8,
            "max_context_chars": 8000,
            "score_threshold_enabled": False,
            "score_threshold": 0.0,
            "reranking_enable": False,
        },
        "coding": {
            "language": "cpp",
            "standard": "cpp17",
            "submission_mode": "auto",
            "compile_validation": False,
        },
        "privacy": {
            "persist_audio": False,
            "persist_transcripts": False,
            "persist_screenshots": False,
            "redact_secrets_in_logs": True,
        },
        "providers": {
            "llm": "mock",
            "retrieval": "mock",
            "asr": "mock",
            "screenshot": "mock",
        },
    }


def make_config(
    hotkeys: dict[str, str] | None = None, **ui_overrides: Any
) -> AppConfig:
    """构造测试用的显式配置。"""
    return AppConfig.model_validate(_config_data(hotkeys, **ui_overrides))


class _FakeManager:
    """测试用的假问题管理器，记录界面调了哪些方法。"""

    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.cancelled = 0
        self.new_questions = 0
        self.reanalyzed = 0
        # submit_text 进来时置位，测试据此判断请求已经发出
        self.started = asyncio.Event()
        # 非空时 submit_text 等这个事件再返回，用来把生成过程卡住
        self.gate: asyncio.Event | None = None

    async def submit_text(self, text: str) -> None:
        """记录提交的题面。"""
        self.submitted.append(text)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()

    async def cancel(self) -> None:
        """记录取消。"""
        self.cancelled += 1

    def new_question(self) -> None:
        """记录新建问题。"""
        self.new_questions += 1

    async def reanalyze(self) -> None:
        """记录重新分析。"""
        self.reanalyzed += 1


class _CancellingManager(_FakeManager):
    """提交时被取消的假管理器。"""

    async def submit_text(self, text: str) -> None:
        """抛出取消错误。"""
        raise asyncio.CancelledError


class _FailingManager(_FakeManager):
    """提交时抛普通异常的假管理器。"""

    async def submit_text(self, text: str) -> None:
        """抛出普通异常。"""
        raise RuntimeError("模型挂了")


class _GatedManager:
    """每次提交各等一个门，测试按顺序放行，模拟新问题取消旧请求。"""

    def __init__(self, count: int = 2) -> None:
        self.gates = [asyncio.Event() for _ in range(count)]
        self.started = [asyncio.Event() for _ in range(count)]
        self.calls = 0
        self.submitted: list[str] = []

    async def submit_text(self, text: str) -> None:
        """记录题面，然后等自己的门开。"""
        index = self.calls
        self.calls += 1
        self.submitted.append(text)
        self.started[index].set()
        await self.gates[index].wait()



class _FakeScreenshotManager:
    """测试用的假截图管理器，只记录 capture_and_submit 的调用。"""

    def __init__(self) -> None:
        self.captures = 0

    async def capture_and_submit(self) -> None:
        """记录截图提交。"""
        self.captures += 1


# ----------------------------------------------------------------------
# 启动与结构
# ----------------------------------------------------------------------
async def test_app_starts_without_error() -> None:
    """应用能在无头模式下启动，不抛异常。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.is_running


async def test_app_starts_without_manager() -> None:
    """管理器没有挂上时界面照常起。"""
    app = InterviewMateApp(config=make_config(), manager=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.is_running
        assert app.manager is None


async def test_compose_contains_four_part_types() -> None:
    """compose 出的结构里有状态栏、左栏、右栏、底部输入区四类部件。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(StatusBar) is not None
        assert app.query_one(NavigationPanel) is not None
        assert app.query_one(AnswerPanel) is not None
        assert app.query_one(QuestionInputPanel) is not None


async def test_layout_has_four_sections_in_order() -> None:
    """主布局是「状态栏 + 左右两栏 + 底部输入区 + 快捷键提示栏」四段结构，顺序固定。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        layout = app.query_one(MainLayout)
        children = list(layout.children)
        assert len(children) == 4
        assert isinstance(children[0], StatusBar)
        assert isinstance(children[1], NormalView)
        assert isinstance(children[2], QuestionInputPanel)
        assert isinstance(children[3], HotkeyBar)
        columns = list(children[1].children)
        assert len(columns) == 2
        assert isinstance(columns[0], NavigationPanel)
        assert isinstance(columns[1], Vertical)
        right_column = list(columns[1].children)
        assert isinstance(right_column[0], QuestionHeader)
        # 回答内容包在滚动容器里，鼠标滚轮才能上下翻
        assert isinstance(right_column[1], VerticalScroll)
        assert isinstance(list(right_column[1].children)[0], AnswerPanel)


async def test_panels_start_empty() -> None:
    """两栏与输入框一开始都没有内容。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(NavigationPanel).plain_text() == ""
        assert app.query_one(AnswerPanel).plain_text() == ""
        assert app.query_one(QuestionInputPanel).text == ""


async def test_input_area_has_focus_on_start() -> None:
    """启动后焦点在底部输入框，用户可以直接打字。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.focused, QuestionInputPanel)


async def test_panel_properties_are_exposed() -> None:
    """两个面板属性供装配层建分发器用。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.navigation_panel is app.query_one(NavigationPanel)
        assert app.answer_panel is app.query_one(AnswerPanel)


# ----------------------------------------------------------------------
# 右栏顶部的问题
# ----------------------------------------------------------------------
async def test_question_header_hidden_before_any_question() -> None:
    """还没提问时右栏顶部这一段不占位置。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        header = app.query_one(QuestionHeader)
        assert header.display is False
        assert header.question == ""


async def test_question_header_shows_submitted_question() -> None:
    """回车提交后题目显示在右栏顶部。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        panel.text = "什么是 RAII"
        await pilot.press("enter")
        await pilot.pause()
        header = app.query_one(QuestionHeader)
        assert header.question == "什么是 RAII"
        assert header.display is True
        assert "什么是 RAII" in header.render().plain


async def test_question_header_replaces_the_previous_question() -> None:
    """再提交一个问题，顶部换成新题目。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        panel.text = "第一个问题"
        await pilot.press("enter")
        await pilot.pause()
        panel.text = "第二个问题"
        await pilot.press("enter")
        await pilot.pause()
        header = app.query_one(QuestionHeader)
        assert header.question == "第二个问题"
        assert "第一个问题" not in header.render().plain


async def test_new_question_clears_the_question_header() -> None:
    """Ctrl+N 新建问题时右栏顶部的题目清掉。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        panel.text = "什么是 RAII"
        await pilot.press("enter")
        await pilot.pause()
        app.action_new_question()
        await pilot.pause()
        header = app.query_one(QuestionHeader)
        assert header.question == ""
        assert header.display is False


# ----------------------------------------------------------------------
# 底部快捷键提示栏
# ----------------------------------------------------------------------
async def test_hotkey_bar_lists_available_shortcuts() -> None:
    """提示栏列出能用的快捷键，退出键固定在右边。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        bar = app.query_one(HotkeyBar)
        text = bar.query_one("#hotkey-main").render()
        assert "Enter 提交" in text
        assert "Ctrl+N 新问题" in text
        assert "Esc 取消" in text
        assert "Ctrl+Q 退出" in str(bar.query_one("#hotkey-exit").render())


async def test_hotkey_bar_follows_config_keys() -> None:
    """改了配置里的键名，提示栏跟着变。"""
    app = InterviewMateApp(config=make_config(hotkeys={"new_question": "ctrl+k"}))
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        text = app.query_one(HotkeyBar).query_one("#hotkey-main").render()
        assert "Ctrl+K 新问题" in text


async def test_hotkey_bar_sits_below_the_input() -> None:
    """提示栏排在输入框下面。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        input_region = app.query_one(QuestionInputPanel).region
        bar_region = app.query_one(HotkeyBar).region
        assert bar_region.y >= input_region.y + input_region.height


# ----------------------------------------------------------------------
# 状态栏渲染
# ----------------------------------------------------------------------
async def test_status_bar_renders_app_name_and_six_labels() -> None:
    """状态栏文本包含应用名与六个短标签。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        text = app.query_one(StatusBar).render()
        assert APP_NAME in text
        for label in ("RUNNING", "AUDIO", "ASR", "DIFY", "MODEL", "TTFT"):
            assert label in text


async def test_status_bar_label_order_is_fixed() -> None:
    """八个短标签的顺序是 RUNNING、AUDIO、ASR、DIFY、MODEL、TTFT、MODE、SESSION。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        text = app.query_one(StatusBar).render()
        labels = ("RUNNING", "AUDIO", "ASR", "DIFY", "MODEL", "TTFT", "MODE", "SESSION")
        positions = [text.index(f"{label}=") for label in labels]
        assert positions == sorted(positions)
        assert STATUS_ORDER == (
            "running", "audio", "asr", "dify", "model", "ttft", "mode", "session",
        )
        assert STATUS_LABELS["running"] == "RUNNING"
        assert STATUS_LABELS["audio"] == "AUDIO"
        assert STATUS_LABELS["asr"] == "ASR"
        assert STATUS_LABELS["dify"] == "DIFY"
        assert STATUS_LABELS["model"] == "MODEL"
        assert STATUS_LABELS["ttft"] == "TTFT"
        assert STATUS_LABELS["mode"] == "MODE"
        assert STATUS_LABELS["session"] == "SESSION"


async def test_status_bar_default_text_matches_fixed_format() -> None:
    """默认渲染文本就是冻结的固定格式：普通模式，没有会话。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(StatusBar).render() == (
            "InterviewMate | RUNNING=stopped | AUDIO=off | ASR=idle | "
            "DIFY=offline | MODEL=idle | TTFT=- | MODE=普通 | SESSION=-"
        )


async def test_status_bar_background_is_not_the_accent_color() -> None:
    """状态栏的底色不是主题里的高亮色。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        accent = Color.parse(app.get_css_variables()["accent"])
        assert bar.styles.background != accent


async def test_status_bar_segments_are_styled_separately() -> None:
    """状态行按段着色：应用名加粗，标签与取值不是一个颜色。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        strip = app.query_one(StatusBar).render_line(0)
        segments = list(strip)
        assert segments[0].text == APP_NAME
        assert segments[0].style is not None
        assert segments[0].style.bold is True
        label_styles = [seg.style for seg in segments if seg.text == "RUNNING"]
        value_styles = [seg.style for seg in segments if seg.text == "stopped"]
        assert label_styles and value_styles
        assert label_styles[0].color != value_styles[0].color


async def test_update_status_changes_rendered_text() -> None:
    """update_status 之后状态栏文本反映新值。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        bar.update_status(
            running="running",
            audio="on",
            asr="ready",
            dify="degraded",
            model="streaming",
            ttft_ms=250,
        )
        await pilot.pause()
        text = bar.render()
        assert "RUNNING=running" in text
        assert "AUDIO=on" in text
        assert "ASR=ready" in text
        assert "DIFY=degraded" in text
        assert "MODEL=streaming" in text
        assert "TTFT=250" in text


async def test_status_snapshot_matches_render() -> None:
    """status_snapshot() 与渲染文本一致。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        bar.update_status(running="running", asr="loading", ttft_ms=88)
        await pilot.pause()
        snapshot = bar.status_snapshot()
        assert snapshot == {
            "running": "running",
            "audio": "off",
            "asr": "loading",
            "dify": "offline",
            "model": "idle",
            "ttft": "88",
            "mode": "普通",
            "session": "-",
        }
        text = bar.render()
        for key, value in snapshot.items():
            assert f"{STATUS_LABELS[key]}={value}" in text


@pytest.mark.parametrize(
    ("field", "bad_value", "allowed"),
    [
        ("running", "paused", RUNNING_VALUES),
        ("audio", "muted", AUDIO_VALUES),
        ("asr", "warm", ASR_VALUES),
        ("dify", "down", DIFY_VALUES),
        ("model", "busy", MODEL_VALUES),
    ],
)
async def test_update_status_rejects_illegal_value(
    field: str, bad_value: str, allowed: tuple[str, ...]
) -> None:
    """非法取值抛 ValueError，消息里列出该字段的合法取值。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        with pytest.raises(ValueError) as excinfo:
            bar.update_status(**{field: bad_value})
        message = str(excinfo.value)
        assert field in message
        for value in allowed:
            assert value in message


async def test_update_status_rejects_illegal_ttft() -> None:
    """TTFT 只能是 None 或非负毫秒数。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        with pytest.raises(ValueError) as excinfo:
            bar.update_status(ttft_ms=-5)
        assert "ttft_ms" in str(excinfo.value)


async def test_ttft_none_renders_dash() -> None:
    """TTFT 没有数据时显示 -。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        bar.update_status(ttft_ms=None)
        await pilot.pause()
        assert "TTFT=-" in bar.render()
        assert bar.status_snapshot()["ttft"] == "-"


async def test_ttft_number_renders_integer_milliseconds() -> None:
    """TTFT 有值时显示整数毫秒数。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        bar.update_status(ttft_ms=1234.7)
        await pilot.pause()
        assert "TTFT=1234" in bar.render()
        assert bar.status_snapshot()["ttft"] == "1234"


# ----------------------------------------------------------------------
# 状态栏提示位
# ----------------------------------------------------------------------
async def test_status_bar_second_line_is_empty_without_hint() -> None:
    """没有提示时第二行是空的。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        assert bar.render_line(1).text == ""


async def test_status_bar_second_line_shows_hint() -> None:
    """第二行显示提示短文本。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        bar.update_status(hint="E_MODEL_TIMEOUT 请求超时")
        await pilot.pause()
        assert bar.render_line(1).text == "E_MODEL_TIMEOUT 请求超时"


async def test_show_error_puts_code_and_detail_into_hint() -> None:
    """show_error 把错误码与一句话显示到提示位。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.show_error("E_MODEL_TIMEOUT", "请求超时")
        await pilot.pause()
        bar = app.query_one(StatusBar)
        assert bar.render_line(1).text == "E_MODEL_TIMEOUT 请求超时"


async def test_status_update_carries_hint_field() -> None:
    """状态更新事件带 hint 字段，默认空串。"""
    event = StatusUpdate(ts=0.0)
    assert event.hint == ""
    assert StatusUpdate(ts=0.0, hint="生成中").hint == "生成中"


# ----------------------------------------------------------------------
# 状态栏按宽度自适应
# ----------------------------------------------------------------------
async def test_status_bar_wide_width_shows_all_eight_items() -> None:
    """宽度够时画到屏幕上的状态行与固定格式逐字一致。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        assert bar.render_line(0).text == (
            "InterviewMate | RUNNING=stopped | AUDIO=off | ASR=idle | "
            "DIFY=offline | MODEL=idle | TTFT=- | MODE=普通 | SESSION=-"
        )


async def test_status_bar_narrow_terminal_drops_right_end_items() -> None:
    """窄终端下右端的整项被去掉，不留被硬裁掉一半的字段。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(NARROW_WIDTH, 24)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        text = bar.render_line(0).text
        assert len(text) <= bar.size.width
        assert "TTFT=" not in text
        # 剩下的段落都是完整的，没有半截字段
        assert text.endswith("DIFY=offline")
        assert text == (
            "InterviewMate | RUNNING=stopped | AUDIO=off | ASR=idle | DIFY=offline"
        )


async def test_status_bar_narrow_width_truncates_app_name() -> None:
    """控件窄到连应用名都放不下时，截断应用名。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        bar.styles.width = 10
        await pilot.pause()
        await pilot.pause()
        text = bar.render_line(0).text
        assert bar.size.width == 8
        assert len(text) == bar.size.width
        assert text == "InterviewMate"[: bar.size.width]


# ----------------------------------------------------------------------
# 两栏宽度
# ----------------------------------------------------------------------
async def test_narrow_terminal_keeps_two_columns() -> None:
    """窄终端下仍是两栏，右栏宽度不为 0。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(NARROW_WIDTH, 24)) as pilot:
        await pilot.pause()
        left = app.query_one(NavigationPanel)
        right = app.query_one(AnswerPanel)
        # outer_size 含边框与内边距，即两栏实际占的宽度
        assert left.outer_size.width > 0
        assert right.outer_size.width > 0
        assert left.outer_size.width + right.outer_size.width == NARROW_WIDTH
        assert left.size.width > 0
        assert right.size.width > 0


async def test_narrow_terminal_left_panel_becomes_compact_navigation() -> None:
    """窄终端下左栏收成紧凑导航，右栏占满剩余空间。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(NARROW_WIDTH, 24)) as pilot:
        await pilot.pause()
        left = app.query_one(NavigationPanel)
        right = app.query_one(AnswerPanel)
        assert left.outer_size.width == COMPACT_NAV_WIDTH
        assert right.outer_size.width == NARROW_WIDTH - COMPACT_NAV_WIDTH
        assert right.outer_size.width > left.outer_size.width


async def test_wide_terminal_keeps_left_right_ratio() -> None:
    """宽终端下左右比例接近 22 比 78。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        left = app.query_one(NavigationPanel)
        right = app.query_one(AnswerPanel)
        total = left.outer_size.width + right.outer_size.width
        assert total == WIDE_WIDTH
        assert abs(left.outer_size.width / total - 0.22) <= 0.02
        assert abs(right.outer_size.width / total - 0.78) <= 0.02


async def test_ratio_follows_configured_left_ratio() -> None:
    """左右比例取自配置 ui.left_ratio / ui.right_ratio。"""
    app = InterviewMateApp(config=make_config(left_ratio=30, right_ratio=70))
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        left = app.query_one(NavigationPanel)
        right = app.query_one(AnswerPanel)
        total = left.outer_size.width + right.outer_size.width
        assert abs(left.outer_size.width / total - 0.30) <= 0.02
        assert abs(right.outer_size.width / total - 0.70) <= 0.02


# ----------------------------------------------------------------------
# 事件总线
# ----------------------------------------------------------------------
async def test_publish_status_updates_status_bar() -> None:
    """publish_status 之后状态栏被更新，证明事件总线这条链路通。"""
    bus = EventBus()
    app = InterviewMateApp(config=make_config(), bus=bus)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.publish_status(
            running="running",
            audio="on",
            asr="ready",
            dify="ready",
            model="streaming",
            ttft_ms=321,
        )
        await pilot.pause()
        text = app.query_one(StatusBar).render()
        assert "RUNNING=running" in text
        assert "AUDIO=on" in text
        assert "ASR=ready" in text
        assert "DIFY=ready" in text
        assert "MODEL=streaming" in text
        assert "TTFT=321" in text


async def test_publish_status_event_type_and_defaults() -> None:
    """状态更新事件的类型名与默认值符合冻结约定。"""
    event = StatusUpdate(ts=0.0)
    assert event.type == "status.update"
    assert event.running == "stopped"
    assert event.audio == "off"
    assert event.asr == "idle"
    assert event.dify == "offline"
    assert event.model == "idle"
    assert event.ttft_ms is None


async def test_app_builds_its_own_bus_when_none_given() -> None:
    """bus 为 None 时应用自建一个事件总线，publish_status 仍然可用。"""
    app = InterviewMateApp(config=make_config(), bus=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.bus is not None
        await app.publish_status(asr="loading")
        await pilot.pause()
        assert "ASR=loading" in app.query_one(StatusBar).render()


# ----------------------------------------------------------------------
# 快捷键
# ----------------------------------------------------------------------
async def test_bindings_cover_eight_shortcuts() -> None:
    """快捷键按配置绑定，六个来自 hotkeys 段，两个退出键固定。

    F8 按住说话由 AHK 全局热键处理，不在界面绑定，
    见《5-语音输入与流式ASR》F8 交互一节。
    """
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.hotkey_bindings == {
            "f9": "screenshot",
            "ctrl+n": "new_question",
            "escape": "cancel",
            "f10": "reanalyze",
            "f3": "show_rag_sources",
            "q": "quit",
        }
        assert app.BINDINGS == [
            ("ctrl+c", "quit"),
            ("ctrl+q", "quit"),
        ]


async def test_hotkeys_follow_config() -> None:
    """改了配置里的 hotkeys，生效的键位跟着变。"""
    app = InterviewMateApp(config=make_config(hotkeys={"quit": "ctrl+w"}))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.hotkey_bindings["ctrl+w"] == "quit"
        assert "q" not in app.hotkey_bindings


async def test_binding_actions_have_methods() -> None:
    """每个绑定都对应一个 action_* 方法。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        actions = list(app.hotkey_bindings.values()) + [
            action for _key, action in app.BINDINGS
        ]
        for action in actions:
            method = getattr(app, f"action_{action}", None)
            assert method is not None, f"缺少 action_{action}"
            assert callable(method)


async def test_non_quit_actions_are_callable() -> None:
    """留绑定位置的 action 调用不抛异常。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_screenshot()
        app.action_new_question()
        app.action_cancel()
        app.action_reanalyze()
        app.action_show_rag_sources()
        await pilot.pause()
        assert app.is_running


async def test_f8_is_not_bound_in_the_ui() -> None:
    """F8 不在界面绑定里：按住说话由 AHK 全局热键处理。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "f8" not in app.hotkey_bindings
        assert getattr(app, "action_voice_toggle", None) is None


async def test_screenshot_calls_manager_when_not_marked() -> None:
    """没标记全局热键时，F9 照旧调截图管理器。"""
    manager = _FakeScreenshotManager()
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.attach_screenshot_manager(manager)
        app.action_screenshot()
        await pilot.pause()
        assert manager.captures == 1


async def test_mark_global_hotkeys_active_stops_screenshot() -> None:
    """标记全局热键后，F9 不再调截图管理器。"""
    manager = _FakeScreenshotManager()
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.attach_screenshot_manager(manager)
        app.mark_global_hotkeys_active()
        await pilot.pause()
        app.action_screenshot()
        await pilot.pause()
        assert manager.captures == 0


async def test_global_hotkeys_mark_keeps_f9_binding() -> None:
    """标记全局热键不改绑定，f9 照旧在 hotkey_bindings 里。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        before = app.hotkey_bindings
        app.mark_global_hotkeys_active()
        await pilot.pause()
        assert app.hotkey_bindings == before
        assert "f8" not in app.hotkey_bindings
        assert app.hotkey_bindings["f9"] == "screenshot"


async def test_quit_action_exits_app() -> None:
    """Q 退出：action_quit 能把应用关掉。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_quit()
    assert app.return_code == 0


async def test_ctrl_q_exits_app() -> None:
    """Ctrl+Q 退出应用，不弹帮助页。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+q")
    assert app.return_code == 0


async def test_q_types_letter_when_input_has_focus() -> None:
    """焦点在输入框时 q 是字母，应用不退出。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert app.is_running
        assert panel.text == "q"


async def test_input_has_focus_on_start() -> None:
    """启动后焦点在底部输入框，可以直接打字。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.focused is app.query_one(QuestionInputPanel)


async def test_q_quits_when_focus_left_the_input() -> None:
    """焦点移到面板上之后 q 退出程序。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(NavigationPanel).focus()
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
    assert app.return_code == 0


# ----------------------------------------------------------------------
# 与问题管理器对接
# ----------------------------------------------------------------------
async def test_attach_manager_puts_manager_on_app() -> None:
    """attach_manager 把管理器挂到应用上。"""
    app = InterviewMateApp(config=make_config())
    manager = _FakeManager()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.manager is None
        app.attach_manager(manager)
        assert app.manager is manager


async def test_input_submit_calls_manager_submit_text() -> None:
    """输入框回车提交后调用 manager.submit_text。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        panel.text = "什么是 RAII"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert manager.submitted == ["什么是 RAII"]


async def test_escape_calls_manager_cancel() -> None:
    """Esc 调用 manager.cancel。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_cancel()
        await app.workers.wait_for_complete()
        assert manager.cancelled == 1


async def test_ctrl_n_calls_manager_new_question() -> None:
    """Ctrl+N 调用 manager.new_question。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_new_question()
        await pilot.pause()
        assert manager.new_questions == 1


async def test_f10_calls_manager_reanalyze() -> None:
    """F10 调用 manager.reanalyze。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_reanalyze()
        await app.workers.wait_for_complete()
        assert manager.reanalyzed == 1


async def test_actions_without_manager_do_not_crash() -> None:
    """管理器没有挂上时，动作只记日志，界面不崩。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_new_question()
        app.action_cancel()
        app.action_reanalyze()
        await pilot.pause()
        assert app.is_running


async def test_manager_exception_does_not_crash_app() -> None:
    """管理器抛普通异常时界面不崩。"""
    manager = _FailingManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        panel.text = "什么是 RAII"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert app.is_running


async def test_manager_cancellation_does_not_crash_app() -> None:
    """管理器抛取消错误时界面不崩。"""
    manager = _CancellingManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        panel.text = "什么是 RAII"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert app.is_running


# ----------------------------------------------------------------------
# MODEL 状态跟着生成过程变
# ----------------------------------------------------------------------
async def test_model_is_streaming_while_generating() -> None:
    """提交后等模型输出时 MODEL 是 streaming，提示位显示生成中。"""
    manager = _FakeManager()
    manager.gate = asyncio.Event()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        panel.text = "什么是 RAII"
        await pilot.press("enter")
        # 等请求真的发出去，这时状态已经置成 streaming
        await asyncio.wait_for(manager.started.wait(), timeout=5)
        await pilot.pause()
        assert "MODEL=streaming" in bar.render()
        assert bar.render_line(1).text == "生成中"
        manager.gate.set()
        await app.workers.wait_for_complete()


async def test_model_returns_to_idle_after_generation() -> None:
    """请求正常结束后 MODEL 置回 idle，提示位清空。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        panel.text = "什么是 RAII"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert "MODEL=idle" in bar.render()
        assert bar.render_line(1).text == ""


async def test_model_becomes_error_when_manager_reports_error() -> None:
    """管理器报错时 MODEL 置 error，提示位显示错误码。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        app.show_error("E_MODEL_TIMEOUT", "请求超时")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "MODEL=error" in bar.render()
        assert "E_MODEL_TIMEOUT" in bar.render_line(1).text


async def test_old_generation_does_not_reset_model_status() -> None:
    """上一轮被新问题取消时，状态归最新一轮管，不改回 idle。"""
    manager = _GatedManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        panel.text = "第一个问题"
        await pilot.press("enter")
        await asyncio.wait_for(manager.started[0].wait(), timeout=5)
        panel.text = "第二个问题"
        await pilot.press("enter")
        await asyncio.wait_for(manager.started[1].wait(), timeout=5)
        await pilot.pause()
        assert "MODEL=streaming" in bar.render()
        # 上一轮先结束，状态不能被它改回 idle
        manager.gates[0].set()
        await pilot.pause()
        await pilot.pause()
        assert "MODEL=streaming" in bar.render()
        # 最新一轮结束才回到 idle
        manager.gates[1].set()
        await app.workers.wait_for_complete()
        assert "MODEL=idle" in bar.render()


async def test_model_change_keeps_other_fields() -> None:
    """改 MODEL 时其余五个字段不动。"""
    manager = _FakeManager()
    app = InterviewMateApp(config=make_config(), manager=manager)
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(StatusBar)
        await app.publish_status(
            running="running",
            audio="on",
            asr="ready",
            dify="ready",
            model="streaming",
            ttft_ms=250,
        )
        await pilot.pause()
        app.show_error("E_MODEL_TIMEOUT", "请求超时")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = bar.render()
        assert "RUNNING=running" in text
        assert "AUDIO=on" in text
        assert "ASR=ready" in text
        assert "DIFY=ready" in text
        assert "TTFT=250" in text
        assert "MODEL=error" in text


# ----------------------------------------------------------------------
# 进程级配置
# ----------------------------------------------------------------------
def test_injected_config_is_not_written_to_process_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式注入配置时，构造界面不覆盖进程级配置，也不重复读配置文件。"""
    sentinel = make_config(hotkeys={"quit": "ctrl+w"})
    injected = make_config(left_ratio=30, right_ratio=70)
    set_current_config(sentinel)

    def _forbidden_load_config(*args: Any, **kwargs: Any) -> AppConfig:
        raise AssertionError("显式注入配置时不应调用 load_config")

    monkeypatch.setattr("app.ui.app.load_config", _forbidden_load_config)

    app = InterviewMateApp(config=injected)

    assert app.config is injected
    assert current_config() is sentinel


def test_app_prefers_process_config_when_no_config_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没注入配置时用进程级当前配置，不再调 load_config。"""
    sentinel = make_config()
    set_current_config(sentinel)
    calls: list[int] = []

    def _counted_load_config(*args: Any, **kwargs: Any) -> AppConfig:
        calls.append(1)
        return make_config()

    monkeypatch.setattr("app.ui.app.load_config", _counted_load_config)

    app = InterviewMateApp()

    assert app.config is sentinel
    assert calls == []


def test_app_loads_config_when_no_process_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """进程级配置为空时才走 load_config，加载结果成为界面配置。"""
    set_current_config(None)
    fallback = make_config()
    calls: list[int] = []

    def _fake_load_config(*args: Any, **kwargs: Any) -> AppConfig:
        calls.append(1)
        # 与真实 load_config 一致：加载成功后写进程级配置
        set_current_config(fallback)
        return fallback

    monkeypatch.setattr("app.ui.app.load_config", _fake_load_config)

    app = InterviewMateApp()

    assert calls == [1]
    assert app.config is fallback
    assert current_config() is fallback
