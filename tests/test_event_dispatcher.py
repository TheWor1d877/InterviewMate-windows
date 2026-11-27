"""事件分发器的测试。

覆盖分发器把事件送到导航面板与内容面板的对应方法、
未知 section_id 的处理、完成状态、clear 与错误上报，
以及代码题模式下的三类事件与兜底分流。
截图请求的路由事件见《4-截图输入与多模态》：
收到路由事件时清空上一题残留并按 kind 预置右栏形态，
代码题一个代码片段都没返回时右栏出现一行重试提示。
"""

from __future__ import annotations

import logging

import pytest

from app.llm.stream_parser import StreamParser
from app.output.event_dispatcher import EMPTY_CODE_NOTICE, EventDispatcher
from app.output.event_models import (
    CodeCompleteEvent,
    CodeDeltaEvent,
    CodeFrameworkEvent,
    CodeStep,
    DoneEvent,
    FallbackEvent,
    FrameworkEvent,
    FrameworkSection,
    ItemEvent,
    RouteEvent,
    SectionDoneEvent,
)
from app.output.response_normalizer import ResponseNormalizer


class FakeNavigation:
    """记下导航面板收到的方法调用。"""

    def __init__(self) -> None:
        self.framework_calls: list[list[FrameworkSection]] = []
        self.done_calls: list[str] = []
        self.clear_calls: int = 0

    def show_framework(self, sections: list[FrameworkSection]) -> None:
        """收到框架。"""
        self.framework_calls.append(sections)

    def mark_section_done(self, section_id: str) -> None:
        """收到某一层完成。"""
        self.done_calls.append(section_id)

    def clear(self) -> None:
        """收到清空请求，内容一并清掉。"""
        self.clear_calls += 1
        self.framework_calls.clear()
        self.done_calls.clear()


class FakeAnswer:
    """记下内容面板收到的方法调用。"""

    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []
        self.fallbacks: list[str] = []
        self.clear_calls: int = 0
        self.code_mode_calls: int = 0
        self.normal_mode_calls: int = 0
        self.notices: list[str] = []
        # 累计代码，分发器用它判断这次代码题有没有代码
        self.code_text: str = ""

    def append_item(self, section_id: str, keyword: str, sentence: str) -> None:
        """收到一个内容单元。"""
        self.items.append((section_id, keyword, sentence))

    def append_fallback(self, text: str) -> None:
        """收到一段兜底内容。"""
        self.fallbacks.append(text)

    def enter_code_mode(self) -> None:
        """收到切到代码形态的请求。"""
        self.code_mode_calls += 1

    def enter_normal_mode(self) -> None:
        """收到切到简答形态的请求。"""
        self.normal_mode_calls += 1

    def show_code_notice(self, text: str) -> None:
        """收到代码栏的一行提示。"""
        self.notices.append(text)

    def clear(self) -> None:
        """收到清空请求，内容一并清掉。"""
        self.clear_calls += 1
        self.items.clear()
        self.fallbacks.clear()
        self.notices.clear()
        self.code_text = ""


def make_dispatcher() -> tuple[EventDispatcher, FakeNavigation, FakeAnswer]:
    """装一套分发器与假面板。"""
    navigation = FakeNavigation()
    answer = FakeAnswer()
    return EventDispatcher(navigation, answer), navigation, answer


def test_framework_event_goes_to_navigation() -> None:
    """框架事件送到导航面板的 show_framework。"""
    dispatcher, navigation, _ = make_dispatcher()
    sections = [FrameworkSection(id="s1", title="基本原理")]

    dispatcher.dispatch(FrameworkEvent(sections=sections))

    assert len(navigation.framework_calls) == 1
    assert navigation.framework_calls[0][0].id == "s1"
    assert dispatcher.section_ids == frozenset({"s1"})


def test_item_event_goes_to_answer() -> None:
    """section_id 在框架里时，内容单元送到内容面板。"""
    dispatcher, _, answer = make_dispatcher()
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="基本原理")]))

    dispatcher.dispatch(
        ItemEvent(section_id="s1", keyword="RAII", sentence="通过对象生命周期自动管理资源。")
    )

    assert answer.items == [("s1", "RAII", "通过对象生命周期自动管理资源。")]
    assert answer.fallbacks == []


def test_item_with_unknown_section_id_goes_to_fallback() -> None:
    """section_id 不在框架里时，内容单元落兜底区，内容不丢。"""
    dispatcher, _, answer = make_dispatcher()
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="基本原理")]))

    dispatcher.dispatch(
        ItemEvent(section_id="s9", keyword="RAII", sentence="通过对象生命周期自动管理资源。")
    )

    assert answer.items == []
    assert answer.fallbacks == ["RAII：通过对象生命周期自动管理资源。"]


def test_section_done_event_marks_section() -> None:
    """section_id 在框架里时，层完成送到导航面板。"""
    dispatcher, navigation, _ = make_dispatcher()
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="基本原理")]))

    dispatcher.dispatch(SectionDoneEvent(section_id="s1"))

    assert navigation.done_calls == ["s1"]


def test_section_done_with_unknown_section_id_is_ignored() -> None:
    """section_id 不在框架里时，层完成被忽略。"""
    dispatcher, navigation, _ = make_dispatcher()
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="基本原理")]))

    dispatcher.dispatch(SectionDoneEvent(section_id="s9"))

    assert navigation.done_calls == []


def test_done_event_sets_completion_state() -> None:
    """done 事件记下完成状态。"""
    dispatcher, _, _ = make_dispatcher()
    assert dispatcher.done is False

    dispatcher.dispatch(DoneEvent())

    assert dispatcher.done is True


def test_fallback_event_goes_to_answer() -> None:
    """兜底事件送到内容面板的 append_fallback。"""
    dispatcher, _, answer = make_dispatcher()

    dispatcher.dispatch(FallbackEvent(text="RAII 是通过对象生命周期自动管理资源。"))

    assert answer.fallbacks == ["RAII 是通过对象生命周期自动管理资源。"]


def test_fallback_text_is_normalized_before_append() -> None:
    """兜底文本进面板前先整理一遍：去掉多余空行与首尾空白。"""
    dispatcher, _, answer = make_dispatcher()

    dispatcher.dispatch(FallbackEvent(text="  第一段  \n\n\n\n第二段   "))

    assert answer.fallbacks == ["第一段\n\n第二段"]


def test_clear_resets_panels_and_state() -> None:
    """clear 清空内容面板，并清掉 section 集合与完成状态。"""
    dispatcher, _, answer = make_dispatcher()
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="基本原理")]))
    dispatcher.dispatch(DoneEvent())

    dispatcher.clear()

    assert answer.clear_calls == 1
    assert dispatcher.section_ids == frozenset()
    assert dispatcher.done is False


def test_report_error_calls_error_reporter() -> None:
    """有 error_reporter 时上报错误。"""
    reported: list[tuple[str, str]] = []
    navigation = FakeNavigation()
    answer = FakeAnswer()
    dispatcher = EventDispatcher(navigation, answer, lambda code, detail: reported.append((code, detail)))

    dispatcher.report_error("E_DEEPSEEK_STREAM", "流中断")

    assert reported == [("E_DEEPSEEK_STREAM", "流中断")]


def test_report_error_without_reporter_does_not_raise() -> None:
    """没有 error_reporter 时只写日志，不报错。"""
    dispatcher, _, _ = make_dispatcher()

    dispatcher.report_error("E_DEEPSEEK_STREAM", "流中断")


def test_new_framework_replaces_declared_sections() -> None:
    """新的框架事件重新记一遍声明过的 section id。"""
    dispatcher, _, answer = make_dispatcher()
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="基本原理")]))
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s2", title="实现机制")]))

    dispatcher.dispatch(ItemEvent(section_id="s1", keyword="旧层", sentence="不该出现"))
    dispatcher.dispatch(ItemEvent(section_id="s2", keyword="新层", sentence="应该出现"))

    assert answer.items == [("s2", "新层", "应该出现")]
    assert answer.fallbacks == ["旧层：不该出现"]


# ----------------------------------------------------------------------
# 兜底整理器只有分发器在用，直接测试放在这个文件里
# ----------------------------------------------------------------------
def test_normalize_fallback_cleans_whitespace_and_blank_lines() -> None:
    """去首尾空白、压缩行尾空白、合并连续多余空行。"""
    assert ResponseNormalizer.normalize_fallback("  第一段  \n\n\n\n第二段   ") == "第一段\n\n第二段"


def test_normalize_fallback_removes_paired_code_fences() -> None:
    """成对的 markdown 代码围栏去掉标记行，围栏里的内容保留。"""
    text = "```python\nprint(1)\n```"

    assert ResponseNormalizer.normalize_fallback(text) == "print(1)"


def test_normalize_fallback_keeps_unpaired_fence() -> None:
    """没有配对的围栏保留不动。"""
    text = "```python\nprint(1)"

    assert ResponseNormalizer.normalize_fallback(text) == text


def test_normalize_fallback_on_empty_text() -> None:
    """空文本整理后还是空串。"""
    assert ResponseNormalizer.normalize_fallback("") == ""
    assert ResponseNormalizer.normalize_fallback("   \n  ") == ""


def test_format_item_joins_keyword_and_sentence() -> None:
    """关键点与表达拼成「关键词：句子」一行。"""
    assert ResponseNormalizer.format_item(" RAII ", " 一句表达。 ") == "RAII：一句表达。"


def test_format_item_with_missing_part() -> None:
    """句子为空时只留关键词，关键词为空时只留句子。"""
    assert ResponseNormalizer.format_item("RAII", "   ") == "RAII"
    assert ResponseNormalizer.format_item("", "只有句子") == "只有句子"


# ----------------------------------------------------------------------
# 代码题模式的事件分发，见《7-代码问题模式》流式事件一节
# ----------------------------------------------------------------------
class FakeCodingNavigation(FakeNavigation):
    """在普通假导航面板上补上代码题模式的方法。"""

    def __init__(self) -> None:
        super().__init__()
        self.steps_calls: list[list[CodeStep]] = []
        self.hints: list[str] = []

    def show_steps(self, steps: list[CodeStep]) -> None:
        """收到解题阶梯。"""
        self.steps_calls.append(steps)

    def append_hint(self, text: str) -> None:
        """收到一条提示。"""
        self.hints.append(text)

    def clear(self) -> None:
        """收到清空请求，阶梯与提示一并清掉。"""
        super().clear()
        self.steps_calls.clear()
        self.hints.clear()


class FakeCodingAnswer(FakeAnswer):
    """在普通假内容面板上补上代码题模式的方法。"""

    def __init__(self) -> None:
        super().__init__()
        self.code_chunks: list[str] = []
        self.code_completes: list[tuple[str, str, str]] = []
        self.steps_calls: list[list[CodeStep]] = []

    def show_steps(self, steps: list[CodeStep]) -> None:
        """收到代码上方的思路块。"""
        self.steps_calls.append(steps)

    def append_code(self, text: str) -> None:
        """收到一段代码。"""
        self.code_chunks.append(text)
        self.code_text += text

    def complete_code(
        self,
        language: str = "cpp",
        standard: str = "cpp17",
        submission_mode: str = "auto",
    ) -> None:
        """收到代码完成通知。"""
        self.code_completes.append((language, standard, submission_mode))

    def clear(self) -> None:
        """收到清空请求，代码一并清掉。"""
        super().clear()
        self.code_chunks.clear()
        self.code_completes.clear()


def make_coding_dispatcher() -> tuple[EventDispatcher, FakeCodingNavigation, FakeCodingAnswer]:
    """装一套分发器与代码题模式的假面板。"""
    navigation = FakeCodingNavigation()
    answer = FakeCodingAnswer()
    return EventDispatcher(navigation, answer), navigation, answer


def test_code_framework_event_goes_to_show_steps() -> None:
    """代码题的框架事件送到导航面板的 show_steps，并带上步骤。"""
    dispatcher, navigation, answer = make_coding_dispatcher()
    steps = [CodeStep(id="p1", title="理解题意", hint="明确状态和约束")]

    dispatcher.dispatch(CodeFrameworkEvent(steps=steps))

    assert len(navigation.steps_calls) == 1
    assert navigation.steps_calls[0][0].id == "p1"
    assert navigation.steps_calls[0][0].title == "理解题意"
    assert navigation.steps_calls[0][0].hint == "明确状态和约束"
    # 右栏代码上方也拿到同一份思路
    assert len(answer.steps_calls) == 1
    assert answer.steps_calls[0][0].title == "理解题意"


def test_code_delta_event_goes_to_append_code() -> None:
    """代码增量事件送到内容面板的 append_code，文本原样传出。"""
    dispatcher, _, answer = make_coding_dispatcher()
    text = "#include <bits/stdc++.h>\nusing namespace std;\n"

    dispatcher.dispatch(CodeDeltaEvent(text=text))

    assert answer.code_chunks == [text]


def test_code_complete_event_goes_to_complete_code() -> None:
    """代码完成事件送到内容面板的 complete_code，三个参数原样传出。"""
    dispatcher, _, answer = make_coding_dispatcher()

    dispatcher.dispatch(
        CodeCompleteEvent(language="cpp", standard="cpp17", submission_mode="acm")
    )

    assert answer.code_completes == [("cpp", "cpp17", "acm")]


def test_coding_fallback_splits_code_and_note() -> None:
    """真实链路：解析器切出的事件进分发器，代码进右栏，说明进左栏提示。

    模型完全不按事件协议输出，解析器把围栏块聚合成一个兜底事件，
    分发器再把它拆成代码正文与说明文字两路。
    """
    dispatcher, navigation, answer = make_coding_dispatcher()
    # 没有代码题事件，靠装配层显式告知分发器这是代码题
    dispatcher.set_coding_mode(True)
    parser = StreamParser(request_id="r1")
    text = "先看题意。\n```cpp\nint main() { return 0; }\n```\n复杂度 O(n)。\n"

    for event in parser.feed(text) + parser.flush():
        dispatcher.dispatch(event)

    assert answer.code_chunks == ["int main() { return 0; }"]
    assert navigation.hints == ["先看题意。", "复杂度 O(n)。"]
    assert answer.fallbacks == []


def test_coding_fallback_without_code_only_adds_hint() -> None:
    """代码题模式下只有说明文字时只追加提示，不追加空代码。"""
    dispatcher, navigation, answer = make_coding_dispatcher()
    dispatcher.dispatch(CodeDeltaEvent(text="int main() {}"))

    dispatcher.dispatch(FallbackEvent(text="接口不明确，按 C++17 完整实现处理。"))

    assert answer.code_chunks == ["int main() {}"]
    assert navigation.hints == ["接口不明确，按 C++17 完整实现处理。"]
    assert answer.fallbacks == []


def test_normal_fallback_still_goes_to_append_fallback() -> None:
    """普通题路径下兜底文本整体落 append_fallback。"""
    dispatcher, navigation, answer = make_coding_dispatcher()

    dispatcher.dispatch(FallbackEvent(text="说明。\n```cpp\nint x = 1;\n```"))

    assert answer.fallbacks == ["说明。\nint x = 1;"]
    assert answer.code_chunks == []
    assert navigation.hints == []


def test_clear_resets_coding_mode() -> None:
    """clear 之后代码题模式被重置，兜底文本回到整体落 append_fallback。"""
    dispatcher, navigation, answer = make_coding_dispatcher()
    dispatcher.dispatch(CodeFrameworkEvent(steps=[CodeStep(id="p1", title="理解题意")]))

    dispatcher.clear()

    dispatcher.dispatch(FallbackEvent(text="```cpp\nint x = 1;\n```"))

    assert answer.clear_calls == 1
    assert answer.code_chunks == []
    assert navigation.hints == []
    assert answer.fallbacks == ["int x = 1;"]


# ----------------------------------------------------------------------
# 代码题模式的显式切换，见《7-代码问题模式》纯代码块兜底一节
# ----------------------------------------------------------------------
def test_clear_clears_navigation_steps_and_hints() -> None:
    """clear 之后左栏的解题阶梯与提示也被清掉。"""
    dispatcher, navigation, answer = make_coding_dispatcher()
    dispatcher.dispatch(CodeFrameworkEvent(steps=[CodeStep(id="p1", title="理解题意")]))
    dispatcher.dispatch(FallbackEvent(text="关键假设写在提示里。"))

    dispatcher.clear()

    assert navigation.clear_calls == 1
    assert navigation.steps_calls == []
    assert navigation.hints == []
    assert answer.clear_calls == 1


def test_set_coding_mode_routes_fallback_to_code_and_hint() -> None:
    """显式切到代码形态后，围栏块进右栏代码区，说明文字进左栏提示。"""
    dispatcher, navigation, answer = make_coding_dispatcher()

    dispatcher.set_coding_mode(True)
    dispatcher.dispatch(FallbackEvent(text="先看题意。\n```cpp\nint x = 1;\n```"))

    assert dispatcher.coding_mode is True
    assert answer.code_mode_calls == 1
    assert answer.code_chunks == ["int x = 1;"]
    assert navigation.hints == ["先看题意。"]
    assert answer.fallbacks == []


def test_set_coding_mode_back_to_normal_keeps_fallback_whole() -> None:
    """显式切回普通形态后，兜底事件整体落右栏兜底区。"""
    dispatcher, navigation, answer = make_coding_dispatcher()
    dispatcher.set_coding_mode(True)

    dispatcher.set_coding_mode(False)
    dispatcher.dispatch(FallbackEvent(text="说明。\n```cpp\nint x = 1;\n```"))

    assert dispatcher.coding_mode is False
    assert answer.fallbacks == ["说明。\nint x = 1;"]
    assert answer.code_chunks == []
    assert navigation.hints == []


def test_clear_resets_coding_mode_flag() -> None:
    """清空之后代码题模式重置为假。"""
    dispatcher, _, _ = make_coding_dispatcher()
    dispatcher.set_coding_mode(True)

    dispatcher.clear()

    assert dispatcher.coding_mode is False


# ----------------------------------------------------------------------
# 截图请求的路由事件，见《4-截图输入与多模态》
# ----------------------------------------------------------------------
def test_route_coding_clears_residue_and_enters_code_mode() -> None:
    """路由到 coding 时清空上一题残留，右栏进代码形态。"""
    dispatcher, navigation, answer = make_coding_dispatcher()
    # 上一题留下的普通题内容
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="旧框架")]))
    dispatcher.dispatch(ItemEvent(section_id="s1", keyword="旧", sentence="旧内容"))

    dispatcher.dispatch(RouteEvent(kind="coding"))

    assert navigation.clear_calls == 1
    assert answer.clear_calls == 1
    assert answer.items == []
    assert answer.code_mode_calls == 1
    assert dispatcher.coding_mode is True


def test_route_normal_clears_residue_and_enters_normal_mode() -> None:
    """路由到 normal 时清空上一题残留，右栏进简答形态。"""
    dispatcher, navigation, answer = make_coding_dispatcher()
    dispatcher.dispatch(CodeFrameworkEvent(steps=[CodeStep(id="p1", title="旧阶梯")]))
    dispatcher.dispatch(CodeDeltaEvent(text="int main() {}"))

    dispatcher.dispatch(RouteEvent(kind="normal"))

    assert navigation.clear_calls == 1
    assert answer.clear_calls == 1
    assert answer.code_chunks == []
    assert answer.normal_mode_calls == 1
    assert dispatcher.coding_mode is False


def test_normal_framework_does_not_flip_route_coding_form() -> None:
    """路由到 coding 后，普通框架事件不把右栏切回简答形态。"""
    dispatcher, _, answer = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="coding"))

    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="说明")]))

    assert dispatcher.coding_mode is True
    assert answer.code_mode_calls == 1


def test_coding_framework_does_not_flip_route_normal_form() -> None:
    """路由到 normal 后，代码题框架事件不把右栏切回代码形态。"""
    dispatcher, navigation, answer = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="normal"))

    dispatcher.dispatch(CodeFrameworkEvent(steps=[CodeStep(id="p1", title="理解题意")]))

    assert dispatcher.coding_mode is False
    assert answer.code_mode_calls == 0
    assert len(navigation.steps_calls) == 1


def test_route_kind_beats_explicit_coding_mode() -> None:
    """装配层显式切过代码形态，路由到 normal 时仍按路由走简答形态。"""
    dispatcher, _, answer = make_coding_dispatcher()
    dispatcher.set_coding_mode(True)

    dispatcher.dispatch(RouteEvent(kind="normal"))

    assert dispatcher.coding_mode is False
    assert answer.normal_mode_calls == 1


def test_route_with_unknown_kind_is_treated_as_normal() -> None:
    """路由事件的 kind 不在允许取值里时按 normal 处理。"""
    dispatcher, _, answer = make_coding_dispatcher()

    dispatcher.dispatch(RouteEvent(kind="debugging"))

    assert dispatcher.coding_mode is False
    assert answer.normal_mode_calls == 1


def test_clear_resets_route_state() -> None:
    """清空之后路由形态与空代码标记都归零。"""
    dispatcher, _, _ = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="coding"))
    dispatcher.dispatch(DoneEvent())

    dispatcher.clear()
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="新框架")]))

    assert dispatcher.coding_mode is False


def test_code_delta_after_route_normal_is_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """路由到 normal 后收到代码增量时不追加代码，右栏留在简答形态。"""
    dispatcher, _, answer = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="normal"))

    caplog.set_level(logging.WARNING, logger="interviewmate")
    dispatcher.dispatch(CodeDeltaEvent(text="int main() { return 0; }"))

    assert answer.code_chunks == []
    assert answer.code_text == ""
    assert answer.code_mode_calls == 0
    assert answer.normal_mode_calls == 1
    assert dispatcher.coding_mode is False
    logs = [
        r
        for r in caplog.records
        if getattr(r, "event", "") == "output.dispatch.code_delta_in_normal_route"
    ]
    assert len(logs) == 1


def test_code_complete_after_route_normal_is_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """路由到 normal 后收到代码完成事件时不通知右栏，也不报空代码。"""
    dispatcher, _, answer = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="normal"))

    caplog.set_level(logging.WARNING, logger="interviewmate")
    dispatcher.dispatch(CodeCompleteEvent(language="cpp", standard="cpp17"))

    assert answer.code_completes == []
    assert answer.notices == []
    assert answer.code_mode_calls == 0
    logs = [
        r
        for r in caplog.records
        if getattr(r, "event", "") == "output.dispatch.code_complete_in_normal_route"
    ]
    assert len(logs) == 1


# ----------------------------------------------------------------------
# 代码题空代码兜底，见《4-截图输入与多模态》
# ----------------------------------------------------------------------
def test_coding_without_code_shows_notice_on_done(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """coding 全程没有 code_delta 时，done 到达后右栏出现提示并记 WARNING。"""
    dispatcher, _, answer = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="coding"))
    dispatcher.dispatch(CodeFrameworkEvent(steps=[CodeStep(id="p1", title="理解题意")]))

    caplog.set_level(logging.WARNING, logger="interviewmate")
    dispatcher.dispatch(DoneEvent())

    assert answer.notices == [EMPTY_CODE_NOTICE]
    logs = [r for r in caplog.records if getattr(r, "event", "") == "coding.empty_code"]
    assert len(logs) == 1
    assert logs[0].levelname == "WARNING"


def test_coding_without_code_shows_notice_on_code_complete(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """code_complete 到达时同样检查，没有代码就给提示。"""
    dispatcher, _, answer = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="coding"))

    caplog.set_level(logging.WARNING, logger="interviewmate")
    dispatcher.dispatch(CodeCompleteEvent(language="cpp", standard="cpp17"))

    assert answer.notices == [EMPTY_CODE_NOTICE]
    assert answer.code_completes == [("cpp", "cpp17", "auto")]


def test_coding_with_code_does_not_show_notice() -> None:
    """这次请求有代码时不出现空代码提示。"""
    dispatcher, _, answer = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="coding"))
    dispatcher.dispatch(CodeDeltaEvent(text="int main() { return 0; }"))

    dispatcher.dispatch(CodeCompleteEvent())
    dispatcher.dispatch(DoneEvent())

    assert answer.notices == []


def test_normal_route_without_code_does_not_show_notice() -> None:
    """普通题本来就没有代码，done 到达时不报空代码。"""
    dispatcher, _, answer = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="normal"))
    dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="原理")]))

    dispatcher.dispatch(DoneEvent())

    assert answer.notices == []


def test_empty_code_notice_is_reported_once() -> None:
    """code_complete 与 done 都到达时提示只出现一次。"""
    dispatcher, _, answer = make_coding_dispatcher()
    dispatcher.dispatch(RouteEvent(kind="coding"))

    dispatcher.dispatch(CodeCompleteEvent())
    dispatcher.dispatch(DoneEvent())

    assert answer.notices == [EMPTY_CODE_NOTICE]
