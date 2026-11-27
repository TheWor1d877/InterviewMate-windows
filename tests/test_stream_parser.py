"""JSONL 流式解析器的测试。

覆盖《3-文本输入与流式回答》解析器容错一节列出的情况：
JSON 被拆到多个分片、一条没收完整就遇到新行、空行、字段缺失、
模型输出普通文字、flush 处理残留行，以及四种事件的解析结果。
代码题的事件见《7-代码问题模式》流式事件一节，
纯代码块兜底见《7-代码问题模式》纯代码块兜底一节。
截图请求的路由事件见《4-截图输入与多模态》，
路由行出现在流的任意位置都不影响后面两种协议的解析。
"""

from __future__ import annotations

import logging

import pytest

from app.llm.stream_parser import StreamParser
from app.output.event_models import (
    CodeCompleteEvent,
    CodeDeltaEvent,
    CodeFrameworkEvent,
    DoneEvent,
    FallbackEvent,
    FrameworkEvent,
    ItemEvent,
    RouteEvent,
    SectionDoneEvent,
)
from app.output.response_normalizer import ResponseNormalizer

# 一条完整的框架事件
FRAMEWORK_LINE = (
    '{"type": "framework", "mode": "normal", '
    '"sections": [{"id": "s1", "title": "基本原理"}]}'
)
# 一条完整的 item 事件
ITEM_LINE = (
    '{"type": "item", "section_id": "s1", "keyword": "RAII", '
    '"sentence": "通过对象生命周期自动管理资源的申请与释放。"}'
)
# 代码题的框架事件
CODE_FRAMEWORK_LINE = (
    '{"type": "framework", "mode": "coding", '
    '"steps": [{"id": "p1", "title": "理解题意", "hint": "明确状态和约束"}]}'
)
# 代码题的代码增量事件
CODE_DELTA_LINE = (
    '{"type": "code_delta", "text": "#include <bits/stdc++.h>\\n'
    'using namespace std;\\n"}'
)
# 代码题的代码完成事件
CODE_COMPLETE_LINE = (
    '{"type": "code_complete", "language": "cpp", "standard": "cpp17", '
    '"submission_mode": "acm"}'
)


def test_json_split_into_several_chunks_is_parsed() -> None:
    """一条 JSON 被拆成多片喂入，仍然能解析成事件。"""
    parser = StreamParser()

    assert parser.feed('{"type": "framework", "mode": "nor') == []
    assert parser.feed('mal", "sections": [{"id": "s1", ') == []
    events = parser.feed('"title": "基本原理"}]}\n')

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, FrameworkEvent)
    assert event.mode == "normal"
    assert event.sections[0].id == "s1"
    assert event.sections[0].title == "基本原理"


def test_line_incomplete_when_next_chunk_arrives() -> None:
    """一条 JSONL 没收完整时，下一片到达后拼起来解析。"""
    parser = StreamParser()

    assert parser.feed(ITEM_LINE[:40]) == []
    events = parser.feed(ITEM_LINE[40:] + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ItemEvent)
    assert event.section_id == "s1"
    assert event.keyword == "RAII"


def test_blank_lines_are_skipped() -> None:
    """空行直接跳过，不产出事件。"""
    parser = StreamParser()

    events = parser.feed("\n\n   \n" + ITEM_LINE + "\n\n")

    assert len(events) == 1
    assert isinstance(events[0], ItemEvent)


def test_one_feed_can_return_several_events() -> None:
    """一次 feed 里可以出多个事件。"""
    parser = StreamParser()

    events = parser.feed(FRAMEWORK_LINE + "\n" + ITEM_LINE + "\n" + '{"type": "done"}\n')

    assert [type(event) for event in events] == [FrameworkEvent, ItemEvent, DoneEvent]


def test_four_event_types_are_parsed() -> None:
    """协议里的四种事件都能解析成对应的事件对象。"""
    parser = StreamParser()
    payload = "\n".join(
        [
            FRAMEWORK_LINE,
            ITEM_LINE,
            '{"type": "section_done", "section_id": "s1"}',
            '{"type": "done"}',
        ]
    )

    events = parser.feed(payload + "\n")

    assert len(events) == 4
    framework, item, section_done, done = events
    assert isinstance(framework, FrameworkEvent)
    assert isinstance(item, ItemEvent)
    assert isinstance(section_done, SectionDoneEvent)
    assert section_done.section_id == "s1"
    assert isinstance(done, DoneEvent)


def test_missing_field_goes_to_fallback() -> None:
    """字段缺失的 JSON 进兜底区，原始文本保留。"""
    parser = StreamParser()
    line = '{"type": "item", "section_id": "s1", "keyword": "RAII"}'

    events = parser.feed(line + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, FallbackEvent)
    assert event.text == line


def test_unknown_event_type_goes_to_fallback() -> None:
    """协议里没有的事件类型进兜底区，原始文本保留。"""
    parser = StreamParser()
    line = '{"type": "summary", "text": "模型多写的一行"}'

    events = parser.feed(line + "\n")

    assert len(events) == 1
    assert isinstance(events[0], FallbackEvent)
    assert events[0].text == line


def test_plain_text_goes_to_fallback() -> None:
    """模型输出普通文字时进兜底区，内容不丢。"""
    parser = StreamParser()
    line = "RAII 是通过对象生命周期自动管理资源。"

    events = parser.feed(line + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, FallbackEvent)
    assert event.text == line


def test_flush_parses_residual_line() -> None:
    """flush 把没有换行符的残留当一整行处理，合法 JSON 解析成事件。"""
    parser = StreamParser()

    assert parser.feed(FRAMEWORK_LINE) == []
    events = parser.flush()

    assert len(events) == 1
    assert isinstance(events[0], FrameworkEvent)


def test_flush_sends_invalid_residual_line_to_fallback() -> None:
    """flush 时残留的非法内容进兜底区。"""
    parser = StreamParser()

    parser.feed("模型输出的一段普通文字")
    events = parser.flush()

    assert len(events) == 1
    assert isinstance(events[0], FallbackEvent)
    assert events[0].text == "模型输出的一段普通文字"


def test_flush_on_empty_buffer_returns_nothing() -> None:
    """缓冲区空的时候 flush 没有事件。"""
    parser = StreamParser()

    assert parser.flush() == []


def test_flush_after_line_ending_returns_nothing() -> None:
    """缓冲区里只有换行符时 flush 没有事件。"""
    parser = StreamParser()

    parser.feed(ITEM_LINE + "\n")
    assert parser.flush() == []


def test_reset_starts_over() -> None:
    """reset 后缓冲区与已声明的 section id 都清空，重新开始。"""
    parser = StreamParser()
    parser.feed(FRAMEWORK_LINE + "\n")
    assert parser.section_ids == frozenset({"s1"})

    parser.feed('{"type": "item", "section_id": "s1", "keyword": "RAII"')
    parser.reset()

    assert parser.section_ids == frozenset()
    assert parser.flush() == []
    events = parser.feed(ITEM_LINE + "\n")
    assert len(events) == 1
    assert isinstance(events[0], ItemEvent)


def test_events_carry_request_and_session_ids() -> None:
    """产出的事件带上 parser 的 request_id 与 session_id。"""
    parser = StreamParser(request_id="req-1", session_id="sess-1")

    events = parser.feed(FRAMEWORK_LINE + "\n" + "普通文字\n")

    assert len(events) == 2
    for event in events:
        assert event.request_id == "req-1"
        assert event.session_id == "sess-1"


def test_declared_section_ids_are_recorded() -> None:
    """解析器记下 framework 声明过的 section id。"""
    parser = StreamParser()
    payload = (
        '{"type": "framework", "mode": "normal", '
        '"sections": [{"id": "s1", "title": "基本原理"}, {"id": "s2", "title": "实现机制"}]}'
    )

    parser.feed(payload + "\n")

    assert parser.section_ids == frozenset({"s1", "s2"})


def test_item_with_unknown_section_id_is_still_parsed() -> None:
    """section_id 不在框架里时解析器照样产出事件，兜底由分发器做。"""
    parser = StreamParser()
    parser.feed(FRAMEWORK_LINE + "\n")

    events = parser.feed(
        '{"type": "item", "section_id": "s9", "keyword": "RAII", "sentence": "一句表达。"}\n'
    )

    assert len(events) == 1
    assert isinstance(events[0], ItemEvent)
    assert events[0].section_id == "s9"


def test_missing_done_event_does_not_break() -> None:
    """结束事件缺失时解析器不报错，已经收到的内容照常产出。"""
    parser = StreamParser()

    events = parser.feed(FRAMEWORK_LINE + "\n" + ITEM_LINE + "\n")

    assert len(events) == 2
    assert parser.flush() == []


def test_invalid_json_between_valid_lines_does_not_break() -> None:
    """一条无效 JSON 不影响前后两条正常事件的解析。"""
    parser = StreamParser()

    events = parser.feed(FRAMEWORK_LINE + "\n" + "{不是 JSON}\n" + ITEM_LINE + "\n")

    assert len(events) == 3
    assert isinstance(events[0], FrameworkEvent)
    assert isinstance(events[1], FallbackEvent)
    assert isinstance(events[2], ItemEvent)


def test_coding_framework_is_parsed_with_steps() -> None:
    """coding 模式的框架事件解析出思路阶梯。"""
    parser = StreamParser()

    events = parser.feed(CODE_FRAMEWORK_LINE + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, CodeFrameworkEvent)
    assert event.type == "framework"
    assert event.mode == "coding"
    assert len(event.steps) == 1
    assert event.steps[0].id == "p1"
    assert event.steps[0].title == "理解题意"
    assert event.steps[0].hint == "明确状态和约束"


def test_normal_framework_still_parsed_with_sections() -> None:
    """普通模式的框架事件照旧解析出分层结构。"""
    parser = StreamParser()

    events = parser.feed(FRAMEWORK_LINE + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, FrameworkEvent)
    assert not isinstance(event, CodeFrameworkEvent)
    assert event.mode == "normal"
    assert len(event.sections) == 1
    assert event.sections[0].id == "s1"
    assert event.sections[0].title == "基本原理"


def test_coding_framework_does_not_record_section_ids() -> None:
    """代码题的框架事件不记 section id，代码题没有这个概念。"""
    parser = StreamParser()

    parser.feed(CODE_FRAMEWORK_LINE + "\n")

    assert parser.section_ids == frozenset()


def test_code_delta_is_parsed() -> None:
    """代码增量事件解析成代码增量事件对象。"""
    parser = StreamParser()

    events = parser.feed(CODE_DELTA_LINE + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, CodeDeltaEvent)
    assert event.text == "#include <bits/stdc++.h>\nusing namespace std;\n"


def test_code_delta_keeps_indentation_and_blank_lines() -> None:
    """代码增量的缩进和空行原样保留，不被裁掉。"""
    parser = StreamParser()
    line = '{"type": "code_delta", "text": "int main() {\\n\\n    return 0;\\n}\\n"}'

    events = parser.feed(line + "\n")

    assert len(events) == 1
    assert events[0].text == "int main() {\n\n    return 0;\n}\n"


def test_code_delta_missing_text_goes_to_fallback() -> None:
    """代码增量事件缺 text 时进兜底区，原始文本保留。"""
    parser = StreamParser()
    line = '{"type": "code_delta"}'

    events = parser.feed(line + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, FallbackEvent)
    assert event.text == line


def test_code_delta_blank_text_goes_to_fallback() -> None:
    """代码增量事件的 text 只有空白时进兜底区。"""
    parser = StreamParser()
    line = '{"type": "code_delta", "text": "   "}'

    events = parser.feed(line + "\n")

    assert len(events) == 1
    assert isinstance(events[0], FallbackEvent)
    assert events[0].text == line


def test_code_complete_is_parsed() -> None:
    """代码完成事件解析出语言、标准和提交格式。"""
    parser = StreamParser()

    events = parser.feed(CODE_COMPLETE_LINE + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, CodeCompleteEvent)
    assert event.language == "cpp"
    assert event.standard == "cpp17"
    assert event.submission_mode == "acm"


def test_code_complete_missing_fields_use_defaults() -> None:
    """代码完成事件缺字段时用默认值，不算错误。"""
    parser = StreamParser()

    events = parser.feed('{"type": "code_complete"}\n')

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, CodeCompleteEvent)
    assert event.language == "cpp"
    assert event.standard == "cpp17"
    assert event.submission_mode == "auto"


def test_coding_flow_is_parsed_in_order() -> None:
    """代码题的一整条流按框架、代码增量、代码完成的顺序解析。"""
    parser = StreamParser()

    events = parser.feed(
        CODE_FRAMEWORK_LINE
        + "\n"
        + CODE_DELTA_LINE
        + "\n"
        + CODE_COMPLETE_LINE
        + "\n"
    )

    assert [type(event) for event in events] == [
        CodeFrameworkEvent,
        CodeDeltaEvent,
        CodeCompleteEvent,
    ]


# ----------------------------------------------------------------------
# 纯代码块兜底，见《7-代码问题模式》纯代码块兜底一节
# ----------------------------------------------------------------------
def test_multi_line_code_block_is_aggregated_into_one_event() -> None:
    """多行围栏块聚合成一个兜底事件，事件里带首尾两行围栏标记。"""
    parser = StreamParser()
    block = "```cpp\nint main() { return 0; }\n```"

    events = parser.feed(block + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, FallbackEvent)
    assert event.text == block


def test_code_block_lines_do_not_produce_separate_events() -> None:
    """围栏块里的行不各自产事件，只出整块一个事件。"""
    parser = StreamParser()

    events = parser.feed("```cpp\n#include <bits/stdc++.h>\nint main() { return 0; }\n```\n")

    assert len(events) == 1
    assert isinstance(events[0], FallbackEvent)
    assert events[0].text.count("```") == 2


def test_code_block_keeps_indentation_and_blank_lines() -> None:
    """围栏块里的缩进与空行原样保留。"""
    parser = StreamParser()

    events = parser.feed("```cpp\nint main() {\n\n    return 0;\n}\n```\n")

    assert len(events) == 1
    assert events[0].text == "```cpp\nint main() {\n\n    return 0;\n}\n```"


def test_text_around_code_block_stays_separate() -> None:
    """围栏块前后的说明文字各自成事件，不被并进代码块。"""
    parser = StreamParser()

    events = parser.feed("先看题意。\n```cpp\nint x = 1;\n```\n复杂度 O(n)。\n")

    assert len(events) == 3
    assert [event.text for event in events] == [
        "先看题意。",
        "```cpp\nint x = 1;\n```",
        "复杂度 O(n)。",
    ]
    assert all(isinstance(event, FallbackEvent) for event in events)


def test_code_block_is_split_into_code_and_note_downstream() -> None:
    """整块事件交给下游后能拆出代码正文与说明文字。"""
    parser = StreamParser()

    events = parser.feed("```cpp\nint main() { return 0; }\n```\n")

    code, note = ResponseNormalizer.extract_code_blocks(events[0].text)

    assert code == "int main() { return 0; }"
    assert note == ""


def test_code_block_with_note_is_split_into_code_and_note_downstream() -> None:
    """代码块前后有说明文字时，下游把说明文字与代码分开。"""
    parser = StreamParser()

    events = parser.feed("先看题意。\n```cpp\nint main() { return 0; }\n```\n复杂度 O(n)。\n")

    notes = [event.text for event in events if event.text != events[1].text]
    code, note = ResponseNormalizer.extract_code_blocks(events[1].text)

    assert code == "int main() { return 0; }"
    assert note == ""
    assert notes == ["先看题意。", "复杂度 O(n)。"]


def test_unclosed_code_block_is_flushed_at_end_of_stream() -> None:
    """围栏没闭合时，流结束时把累积内容当成代码块冲刷出去。"""
    parser = StreamParser()

    assert parser.feed("```cpp\nint main() { return 0; }\n") == []
    events = parser.flush()

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, FallbackEvent)
    assert event.text == "```cpp\nint main() { return 0; }"
    code, note = ResponseNormalizer.extract_code_blocks(event.text)
    assert code == "int main() { return 0; }"
    assert note == ""


def test_single_line_code_block_is_not_dropped() -> None:
    """行内围栏的一行代码不被静默丢弃，下游能拆出代码正文。"""
    parser = StreamParser()

    events = parser.feed("```cpp int main(){ return 0; } ```\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, FallbackEvent)
    code, note = ResponseNormalizer.extract_code_blocks(event.text)
    assert code == "int main(){ return 0; }"
    assert note == ""


def test_single_line_code_block_keeps_language() -> None:
    """行内围栏的语言标记保留下来，高亮语言能认出来。"""
    parser = StreamParser()

    events = parser.feed("```cpp int main(){ return 0; } ```\n")

    assert ResponseNormalizer.detect_code_language(events[0].text) == "cpp"


def test_code_block_survives_split_chunks() -> None:
    """围栏块被拆到多个分片喂入，仍聚合成一个事件。"""
    parser = StreamParser()

    assert parser.feed("```cp") == []
    assert parser.feed("p\nint main() { return 0; }\n") == []
    events = parser.feed("```\n")

    assert len(events) == 1
    assert events[0].text == "```cpp\nint main() { return 0; }\n```"


def test_reset_clears_pending_code_block() -> None:
    """reset 清掉没闭合的围栏，下一次请求不会被旧围栏吞掉。"""
    parser = StreamParser()
    parser.feed("```cpp\nint main() { return 0; }\n")

    parser.reset()

    assert parser.flush() == []
    events = parser.feed("说明文字\n")
    assert len(events) == 1
    assert events[0].text == "说明文字"


def test_normal_question_jsonl_path_is_unchanged() -> None:
    """普通题的 JSONL 事件流逐字未变。"""
    parser = StreamParser()

    events = parser.feed(FRAMEWORK_LINE + "\n" + ITEM_LINE + "\n" + '{"type": "done"}\n')

    assert [type(event) for event in events] == [FrameworkEvent, ItemEvent, DoneEvent]
    assert isinstance(events[0], FrameworkEvent)
    assert events[0].sections[0].id == "s1"
    assert isinstance(events[1], ItemEvent)
    assert events[1].keyword == "RAII"


def test_normal_question_plain_text_fallback_is_unchanged() -> None:
    """普通题下不带围栏的纯文本兜底逐字未变，一行一个事件。"""
    parser = StreamParser()
    lines = ["第一行说明。", "第二行说明。"]

    events = parser.feed("\n".join(lines) + "\n")

    assert len(events) == 2
    assert all(isinstance(event, FallbackEvent) for event in events)
    assert [event.text for event in events] == lines


# ----------------------------------------------------------------------
# 截图请求的路由事件，见《4-截图输入与多模态》
# ----------------------------------------------------------------------
ROUTE_CODING_LINE = '{"event": "route", "kind": "coding"}'
ROUTE_NORMAL_LINE = '{"event": "route", "kind": "normal"}'


def test_route_coding_is_parsed() -> None:
    """kind 为 coding 的路由行解析成路由事件。"""
    parser = StreamParser()

    events = parser.feed(ROUTE_CODING_LINE + "\n")

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, RouteEvent)
    assert event.kind == "coding"
    assert event.type == "route"


def test_route_normal_is_parsed() -> None:
    """kind 为 normal 的路由行解析成路由事件。"""
    parser = StreamParser()

    events = parser.feed(ROUTE_NORMAL_LINE + "\n")

    assert len(events) == 1
    assert isinstance(events[0], RouteEvent)
    assert events[0].kind == "normal"


def test_route_with_invalid_kind_falls_back_to_normal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """kind 非法时按 normal 处理，不抛异常，并记一条 WARNING。"""
    parser = StreamParser(request_id="r1")

    caplog.set_level(logging.WARNING, logger="interviewmate")
    events = parser.feed('{"event": "route", "kind": "debugging"}\n')

    assert len(events) == 1
    assert isinstance(events[0], RouteEvent)
    assert events[0].kind == "normal"
    logs = [r for r in caplog.records if getattr(r, "event", "") == "llm.stream.route_invalid"]
    assert len(logs) == 1
    assert logs[0].levelname == "WARNING"


def test_route_with_missing_kind_falls_back_to_normal() -> None:
    """kind 缺失时按 normal 处理，不抛异常。"""
    parser = StreamParser()

    events = parser.feed('{"event": "route"}\n')

    assert len(events) == 1
    assert isinstance(events[0], RouteEvent)
    assert events[0].kind == "normal"


def test_route_then_normal_protocol_keeps_parsing() -> None:
    """路由行之后，普通题的四类事件照常解析。"""
    parser = StreamParser()

    events = parser.feed(
        ROUTE_NORMAL_LINE
        + "\n"
        + FRAMEWORK_LINE
        + "\n"
        + ITEM_LINE
        + "\n"
        + '{"type": "section_done", "section_id": "s1"}\n'
        + '{"type": "done"}\n'
    )

    assert [type(event) for event in events] == [
        RouteEvent,
        FrameworkEvent,
        ItemEvent,
        SectionDoneEvent,
        DoneEvent,
    ]
    assert parser.section_ids == frozenset({"s1"})


def test_route_then_coding_protocol_keeps_parsing() -> None:
    """路由行之后，代码题的三类事件照常解析。"""
    parser = StreamParser()

    events = parser.feed(
        ROUTE_CODING_LINE
        + "\n"
        + CODE_FRAMEWORK_LINE
        + "\n"
        + CODE_DELTA_LINE
        + "\n"
        + CODE_COMPLETE_LINE
        + "\n"
    )

    assert [type(event) for event in events] == [
        RouteEvent,
        CodeFrameworkEvent,
        CodeDeltaEvent,
        CodeCompleteEvent,
    ]


def test_route_in_the_middle_does_not_break_parsing() -> None:
    """路由行出现在流的中间也不影响前后两行的解析。"""
    parser = StreamParser()

    events = parser.feed(
        FRAMEWORK_LINE + "\n" + ROUTE_CODING_LINE + "\n" + CODE_DELTA_LINE + "\n"
    )

    assert [type(event) for event in events] == [
        FrameworkEvent,
        RouteEvent,
        CodeDeltaEvent,
    ]


def test_route_lines_carry_request_and_session_ids() -> None:
    """路由事件带上解析器的 request_id 与 session_id。"""
    parser = StreamParser(request_id="req-route", session_id="sess-route")

    events = parser.feed(ROUTE_CODING_LINE + "\n")

    assert events[0].request_id == "req-route"
    assert events[0].session_id == "sess-route"
