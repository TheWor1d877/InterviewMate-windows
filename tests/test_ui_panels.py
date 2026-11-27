"""界面面板的测试。

覆盖《3-文本输入与流式回答》底部输入框一节与终端界面一节对面板的要求：
左侧「回答流程」、右侧「关键回答内容」、底部输入框。

测试全部用 Textual 的 ``run_test()`` 无头驱动，配置在文件内显式构造，
不依赖开发者本地的 ``config/config.yaml``。
"""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from textual import events
from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.message import Message

from app.output.event_models import FrameworkSection
from app.ui.app import InterviewMateApp
from app.ui.panels.answer_panel import AnswerPanel
from app.ui.panels.input_panel import QuestionInputPanel
from app.ui.panels.navigation_panel import NavigationPanel
from tests.test_ui_skeleton import make_config


# 宽终端与窄终端的宽度，窄终端 80 小于配置的 min_width 90
WIDE_WIDTH = 120
NARROW_WIDTH = 80


def _sections() -> list[FrameworkSection]:
    """构造一份两层框架。"""
    return [
        FrameworkSection(id="s1", title="基本原理"),
        FrameworkSection(id="s2", title="实现机制"),
    ]


def _capture_submitted(
    panel: QuestionInputPanel,
) -> list[QuestionInputPanel.Submitted]:
    """拦下输入框发出的提交事件，返回收到的列表。

    输入框用 :meth:`QuestionInputPanel.post_message` 发事件，这里把发送动作
    包一层，测试就能直接断言事件内容。
    """
    received: list[QuestionInputPanel.Submitted] = []
    original = panel.post_message

    def _post(message: Message) -> None:
        if isinstance(message, QuestionInputPanel.Submitted):
            received.append(message)
        original(message)

    panel.post_message = _post
    return received


# ----------------------------------------------------------------------
# 左侧「回答流程」
# ----------------------------------------------------------------------
async def test_panels_show_their_titles() -> None:
    """两栏各自有标题，与终端界面一节的布局图一致。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        answer = app.query_one(AnswerPanel)
        assert nav.render().plain.splitlines()[0] == "回答流程"
        assert answer.render().plain.splitlines()[0] == "关键回答内容"


async def test_show_framework_renders_section_titles() -> None:
    """show_framework 之后各层标题按顺序渲染出来。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_framework(_sections())
        await pilot.pause()
        text = nav.plain_text()
        assert "基本原理" in text
        assert "实现机制" in text
        assert "1." in text
        assert "2." in text
        # 顺序跟模型给的顺序一致
        assert text.index("基本原理") < text.index("实现机制")


async def test_current_section_is_highlighted() -> None:
    """当前层是第一个没完成的层，它被标记成当前。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_framework(_sections())
        await pilot.pause()
        lines = nav.plain_text().splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("▶")
        assert not lines[1].startswith("▶")


async def test_mark_section_done_marks_section() -> None:
    """mark_section_done 把该层标成完成，当前层往后推。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_framework(_sections())
        nav.mark_section_done("s1")
        await pilot.pause()
        lines = nav.plain_text().splitlines()
        assert lines[0].startswith("✓")
        assert lines[1].startswith("▶")


async def test_show_framework_resets_done_state() -> None:
    """重新给一份框架，完成状态从头算。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_framework(_sections())
        nav.mark_section_done("s1")
        nav.show_framework(_sections())
        await pilot.pause()
        lines = nav.plain_text().splitlines()
        assert lines[0].startswith("▶")


async def test_narrow_terminal_compacts_navigation() -> None:
    """窄终端下左栏收成紧凑导航，标题不溢出。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(NARROW_WIDTH, 24)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_framework(_sections())
        await pilot.pause()
        lines = nav.plain_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            assert cell_len(line) <= nav.size.width
        # 窄栏只留层号，标题不显示
        assert "基本原理" not in nav.plain_text()


async def test_wide_terminal_title_is_truncated_to_panel_width() -> None:
    """面板放不下标题时截断，不溢出。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_framework(
            [FrameworkSection(id="s1", title="这是一个非常非常长的层标题")]
        )
        await pilot.pause()
        for line in nav.plain_text().splitlines():
            assert cell_len(line) <= nav.size.width


# ----------------------------------------------------------------------
# 右侧「关键回答内容」
# ----------------------------------------------------------------------
async def test_append_item_renders_keyword_and_sentence_on_one_line() -> None:
    """append_item 把关键词与句子渲染成同一行的「关键点：正文」。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_item("s1", "RAII", "通过对象生命周期自动管理资源。")
        await pilot.pause()
        lines = answer.plain_text().splitlines()
        assert lines[0] == "RAII：通过对象生命周期自动管理资源。"


async def test_answer_panel_scrolls_with_mouse_wheel() -> None:
    """回答内容放在滚动容器里，内容超出高度后可以滚动。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 24)) as pilot:
        await pilot.pause()
        scroll = app.query_one("#answer-scroll", VerticalScroll)
        answer = app.query_one(AnswerPanel)
        for index in range(30):
            answer.append_item("s1", f"关键点{index}", "正文内容。")
        await pilot.pause()
        await pilot.pause()
        assert scroll.is_scrollable is True
        assert scroll.max_scroll_y > 0


async def test_append_item_keeps_model_output_order() -> None:
    """多个表达单元按追加顺序排列，不重新分组。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_item("s1", "第一点", "第一句")
        answer.append_item("s2", "第二点", "第二句")
        await pilot.pause()
        text = answer.plain_text()
        assert text.index("第一点") < text.index("第二点")


async def test_append_fallback_appends_text() -> None:
    """append_fallback 把兜底文本追加到后面。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_fallback("模型返回了普通文字")
        await pilot.pause()
        assert "模型返回了普通文字" in answer.plain_text()


async def test_clear_empties_answer() -> None:
    """clear 把内容清空。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_item("s1", "RAII", "通过对象生命周期自动管理资源。")
        answer.append_fallback("模型返回了普通文字")
        answer.clear()
        await pilot.pause()
        assert answer.plain_text() == ""


# ----------------------------------------------------------------------
# 底部输入框
# ----------------------------------------------------------------------
async def test_enter_submits_text_and_clears_input() -> None:
    """回车提交当前文本并发事件，输入框内容清空。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        received = _capture_submitted(panel)
        panel.focus()
        await pilot.pause()
        await pilot.press(*"什么是 RAII")
        await pilot.pause()
        assert panel.text == "什么是 RAII"
        await pilot.press("enter")
        await pilot.pause()
        assert [event.text for event in received] == ["什么是 RAII"]
        assert panel.text == ""


async def test_enter_strips_surrounding_whitespace() -> None:
    """提交时去掉首尾空白。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        received = _capture_submitted(panel)
        panel.focus()
        await pilot.pause()
        panel.text = "  什么是 RAII  "
        await pilot.press("enter")
        await pilot.pause()
        assert [event.text for event in received] == ["什么是 RAII"]


async def test_enter_on_empty_input_does_not_submit() -> None:
    """空输入回车不提交。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        received = _capture_submitted(panel)
        panel.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert received == []


async def test_shift_enter_inserts_newline_without_submit() -> None:
    """Shift+回车换行，不提交。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        received = _capture_submitted(panel)
        panel.focus()
        await pilot.pause()
        await pilot.press(*"第一行")
        await pilot.press("shift+enter")
        await pilot.press(*"第二行")
        await pilot.pause()
        assert panel.text == "第一行\n第二行"
        assert received == []


async def test_ctrl_enter_inserts_newline_without_submit() -> None:
    """Ctrl+回车换行，不提交。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        received = _capture_submitted(panel)
        panel.focus()
        await pilot.pause()
        await pilot.press(*"第一行")
        await pilot.press("ctrl+enter")
        await pilot.press(*"第二行")
        await pilot.pause()
        assert panel.text == "第一行\n第二行"
        assert received == []


async def test_paste_keeps_all_lines() -> None:
    """粘贴多行题面保留全部行。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.focus()
        await pilot.pause()
        app.post_message(events.Paste("第一行\n第二行\n第三行"))
        await pilot.pause()
        assert panel.text == "第一行\n第二行\n第三行"


async def test_clear_input_empties_the_panel() -> None:
    """clear_input 把输入框清空。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(QuestionInputPanel)
        panel.text = "还没提交的问题"
        panel.clear_input()
        await pilot.pause()
        assert panel.text == ""


# ----------------------------------------------------------------------
# 代码题形态：左侧解题阶梯，右侧代码块
# ----------------------------------------------------------------------
class _Step:
    """带 id、title、hint 的解题步骤。"""

    def __init__(self, id: str, title: str, hint: str = "") -> None:
        self.id = id
        self.title = title
        self.hint = hint


def _steps() -> list[_Step]:
    """构造一份三步的解题阶梯，中间那步没有提示。"""
    return [
        _Step("p1", "理解题意", "明确状态和约束"),
        _Step("p2", "状态转移"),
        _Step("p3", "复杂度分析"),
    ]


async def test_show_steps_renders_titles_and_hints() -> None:
    """show_steps 渲染出步骤序号与标题，有提示的步骤下面多一行提示。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_steps(_steps())
        await pilot.pause()
        lines = nav.plain_text().splitlines()
        assert "1." in lines[0]
        assert "理解题意" in lines[0]
        assert "明确状态和约束" in lines[1]
        assert "2." in lines[2]
        assert "状态转移" in lines[2]
        assert "3." in lines[3]
        assert "复杂度分析" in lines[3]
        # 只有带提示的步骤多出一行
        assert len(lines) == 4


async def test_show_steps_marks_current_step() -> None:
    """当前步是第一个没完成的步，它被标记成当前。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_steps(_steps())
        await pilot.pause()
        lines = nav.plain_text().splitlines()
        assert lines[0].startswith("▶")
        assert not lines[2].startswith("▶")


async def test_mark_section_done_marks_step_done() -> None:
    """mark_section_done 对解题步骤同样生效，当前步往后推。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_steps(_steps())
        nav.mark_section_done("p1")
        await pilot.pause()
        lines = nav.plain_text().splitlines()
        assert lines[0].startswith("✓")
        assert lines[2].startswith("▶")


async def test_show_steps_resets_done_state() -> None:
    """重新给一份阶梯，完成状态从头算。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_steps(_steps())
        nav.mark_section_done("p1")
        nav.show_steps(_steps())
        await pilot.pause()
        assert nav.plain_text().splitlines()[0].startswith("▶")


async def test_append_hint_appends_hint() -> None:
    """append_hint 在左栏追加一条简短提示，排在解题阶梯后面。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_steps(_steps())
        nav.append_hint("假设输入规模为 1e5")
        await pilot.pause()
        text = nav.plain_text()
        assert "假设输入规模为 1e5" in text
        assert text.index("复杂度分析") < text.index("假设输入规模为 1e5")


async def test_narrow_terminal_compacts_steps() -> None:
    """窄终端下解题阶梯收成紧凑导航，标题与提示都不显示。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(NARROW_WIDTH, 24)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_steps(_steps())
        await pilot.pause()
        text = nav.plain_text()
        assert "理解题意" not in text
        assert "明确状态和约束" not in text
        for line in text.splitlines():
            assert cell_len(line) <= nav.size.width


async def test_append_code_appends_code_in_order() -> None:
    """append_code 多次调用，代码内容是拼接关系，不重写已有内容。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_code("#include <bits/stdc++.h>\n")
        answer.append_code("int main() {\n    return 0;\n}\n")
        await pilot.pause()
        expected = "#include <bits/stdc++.h>\nint main() {\n    return 0;\n}\n"
        assert answer.code_text == expected
        assert expected in answer.plain_text()


async def test_code_view_shows_approach_above_code() -> None:
    """代码上方渲染思路块：每步标题与一句话思路都在，代码仍在。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.show_steps(_steps())
        answer.append_code("int x = 1;\n")
        await pilot.pause()
        text = answer.plain_text()
        assert "思路" in text
        assert "1. 理解题意：明确状态和约束" in text
        assert text.index("理解题意") < text.index("int x = 1;")
        assert "int x = 1;" in text


async def test_clear_empties_approach_block() -> None:
    """clear 把思路块一并清掉。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.show_steps(_steps())
        answer.append_code("int x = 1;\n")
        answer.clear()
        await pilot.pause()
        assert "理解题意" not in answer.plain_text()


async def test_complete_code_shows_completion_marker() -> None:
    """complete_code 之后能看到完成标记，语言、标准与提交模式一起显示。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_code("int x = 1;\n")
        answer.complete_code("cpp", "cpp17", "acm")
        await pilot.pause()
        text = answer.plain_text()
        assert "✓" in text
        assert "完成" in text
        assert "cpp" in text
        assert "cpp17" in text
        assert "acm" in text
        assert answer.code_complete is True


async def test_clear_empties_code_buffer() -> None:
    """clear 把代码缓冲一并清掉。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_code("int x = 1;\n")
        answer.complete_code()
        answer.clear()
        await pilot.pause()
        assert answer.code_text == ""
        assert answer.code_complete is False
        assert answer.plain_text() == ""


async def test_code_form_does_not_mix_with_normal_content() -> None:
    """右栏同一时刻只有一种形态，代码形态不与普通内容混排。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_item("s1", "RAII", "通过对象生命周期自动管理资源。")
        await pilot.pause()
        assert "RAII" in answer.plain_text()
        answer.append_code("int x = 1;\n")
        await pilot.pause()
        assert "RAII" not in answer.plain_text()
        assert "int x = 1;" in answer.plain_text()


async def test_append_code_renders_syntax_highlighted_code() -> None:
    """代码块按 C++ 高亮渲染，关键字带上颜色。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_code("int main() {\n    return 0;\n}\n")
        await pilot.pause()
        region = Region(0, 0, answer.size.width, answer.size.height)
        segments = [seg for strip in answer.render_lines(region) for seg in strip]
        keyword = next(seg for seg in segments if seg.text == "int")
        assert keyword.style is not None
        assert keyword.style.color is not None


# ----------------------------------------------------------------------
# 形态是锁定的：后来的普通事件不能把已经显示的内容藏掉
# ----------------------------------------------------------------------
def _segments(panel: AnswerPanel) -> list[Any]:
    """取面板渲染结果里的所有片段，供样式断言用。"""
    region = Region(0, 0, panel.size.width, panel.size.height)
    return [segment for strip in panel.render_lines(region) for segment in strip]


async def test_code_form_keeps_code_when_item_arrives() -> None:
    """代码形态下追加表达单元，代码仍在右栏可见，普通内容不混进来。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_code("int main() {\n    return 0;\n}\n")
        await pilot.pause()
        answer.append_item("s1", "RAII", "通过对象生命周期自动管理资源。")
        await pilot.pause()
        text = answer.plain_text()
        assert "int main()" in text
        assert "RAII" not in text
        # 渲染出来的仍然是代码块
        assert any(segment.text == "int" for segment in _segments(answer))


async def test_code_form_keeps_code_when_fallback_arrives() -> None:
    """代码形态下追加兜底文本，代码仍然可见。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test() as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_code("int main() {\n    return 0;\n}\n")
        await pilot.pause()
        answer.append_fallback("模型返回了普通文字")
        await pilot.pause()
        text = answer.plain_text()
        assert "int main()" in text
        assert "模型返回了普通文字" not in text


async def test_steps_survive_a_later_framework_event() -> None:
    """已经有解题阶梯时，普通框架事件被忽略，阶梯与完成状态都留着。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        nav = app.query_one(NavigationPanel)
        nav.show_steps(_steps())
        nav.mark_section_done("p1")
        await pilot.pause()
        nav.show_framework([FrameworkSection(id="s1", title="基本原理")])
        await pilot.pause()
        text = nav.plain_text()
        assert "理解题意" in text
        assert "复杂度分析" in text
        assert "基本原理" not in text
        # 完成状态没有被重置
        assert text.splitlines()[0].startswith("✓")


async def test_clear_empties_both_forms() -> None:
    """clear 把两种形态的内容都清空。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        nav = app.query_one(NavigationPanel)
        answer.append_item("s1", "RAII", "通过对象生命周期自动管理资源。")
        answer.append_code("int x = 1;\n")
        answer.complete_code()
        nav.show_steps(_steps())
        nav.append_hint("假设输入规模为 1e5")
        await pilot.pause()
        answer.clear()
        nav.clear()
        await pilot.pause()
        assert answer.plain_text() == ""
        assert answer.code_text == ""
        assert answer.code_language == "cpp"
        assert nav.plain_text() == ""


async def test_complete_code_changes_highlight_language() -> None:
    """complete_code 带来的语言决定高亮语言，已经渲染过的代码按新语言重渲染。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_code("def f():\n    return 1\n")
        await pilot.pause()
        cpp_styles = {
            str(segment.style) for segment in _segments(answer) if segment.text == "def"
        }
        answer.complete_code("python", "python3", "auto")
        await pilot.pause()
        python_styles = {
            str(segment.style) for segment in _segments(answer) if segment.text == "def"
        }
        assert answer.code_language == "python"
        assert cpp_styles != python_styles


async def test_default_code_language_is_cpp() -> None:
    """不指定语言时默认按 C++ 高亮，状态行显示默认的语言与标准。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        answer = app.query_one(AnswerPanel)
        answer.append_code("int main() {\n    return 0;\n}\n")
        answer.complete_code()
        await pilot.pause()
        assert answer.code_language == "cpp"
        text = answer.plain_text()
        assert "cpp" in text
        assert "cpp17" in text
        keyword = next(segment for segment in _segments(answer) if segment.text == "int")
        assert keyword.style is not None
        assert keyword.style.color is not None
