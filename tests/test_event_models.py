"""模型流式事件模型的测试。

覆盖事件字段与默认值、必填字段校验、模型协议里的事件类型表，
以及模型事件不走事件总线这条约定。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.event_bus import Event
from app.output.event_models import (
    CodeCompleteEvent,
    CodeDeltaEvent,
    CodeFrameworkEvent,
    CodeStep,
    DoneEvent,
    EVENT_TYPES,
    FallbackEvent,
    FrameworkEvent,
    FrameworkSection,
    ItemEvent,
    SectionDoneEvent,
    StreamEvent,
)


def test_event_types_cover_model_protocol() -> None:
    """协议表覆盖普通题四种事件与代码题两类事件，顺序固定。"""
    assert EVENT_TYPES == (
        "framework",
        "item",
        "section_done",
        "done",
        "code_delta",
        "code_complete",
    )


def test_normal_event_types_keep_their_order() -> None:
    """普通题的四种事件类型仍在协议表前面，顺序不变。"""
    assert EVENT_TYPES[:4] == ("framework", "item", "section_done", "done")


def test_fallback_event_is_not_a_model_event_type() -> None:
    """兜底事件是解析器自己产出的，不在协议表里。"""
    assert "fallback" not in EVENT_TYPES


def test_framework_section_keeps_id_and_title() -> None:
    """框架层保留 id 与标题。"""
    section = FrameworkSection(id="s1", title="基本原理")
    assert section.id == "s1"
    assert section.title == "基本原理"


def test_framework_event_defaults() -> None:
    """框架事件默认是普通模式，sections 默认为空列表。"""
    event = FrameworkEvent()
    assert event.type == "framework"
    assert event.mode == "normal"
    assert event.sections == []


def test_framework_event_keeps_sections() -> None:
    """框架事件带着声明出来的各层。"""
    event = FrameworkEvent(
        mode="normal",
        sections=[FrameworkSection(id="s1", title="基本原理")],
        request_id="req-1",
        session_id="sess-1",
    )
    assert event.sections[0].id == "s1"
    assert event.sections[0].title == "基本原理"
    assert event.request_id == "req-1"
    assert event.session_id == "sess-1"


def test_item_event_keeps_all_fields() -> None:
    """内容单元事件保留三个必填字段。"""
    event = ItemEvent(
        section_id="s1",
        keyword="RAII",
        sentence="通过对象生命周期自动管理资源的申请与释放。",
    )
    assert event.type == "item"
    assert event.section_id == "s1"
    assert event.keyword == "RAII"
    assert event.sentence.startswith("通过对象生命周期")


def test_item_event_requires_fields() -> None:
    """内容单元事件缺字段时校验失败。"""
    with pytest.raises(ValidationError):
        ItemEvent(section_id="s1", keyword="RAII")


def test_section_done_event_requires_section_id() -> None:
    """层完成事件缺 section_id 时校验失败。"""
    with pytest.raises(ValidationError):
        SectionDoneEvent()


def test_section_done_event_type() -> None:
    """层完成事件的类型名固定。"""
    event = SectionDoneEvent(section_id="s1")
    assert event.type == "section_done"
    assert event.section_id == "s1"


def test_done_event_type() -> None:
    """完成事件的类型名固定。"""
    assert DoneEvent().type == "done"


def test_fallback_event_keeps_text() -> None:
    """兜底事件保留原始文本。"""
    event = FallbackEvent(text="模型输出的一段普通文字")
    assert event.type == "fallback"
    assert event.text == "模型输出的一段普通文字"


def test_stream_event_requires_type() -> None:
    """基类的 type 必填。"""
    with pytest.raises(ValidationError):
        StreamEvent()


def test_model_events_do_not_inherit_event_bus_event() -> None:
    """模型流式事件不走事件总线，不继承它的 Event 类。"""
    assert not issubclass(StreamEvent, Event)
    assert not issubclass(FrameworkEvent, Event)
    assert not issubclass(ItemEvent, Event)


def test_code_step_keeps_fields() -> None:
    """思路阶梯的一步保留 id、标题和提示。"""
    step = CodeStep(id="p1", title="理解题意", hint="明确状态和约束")
    assert step.id == "p1"
    assert step.title == "理解题意"
    assert step.hint == "明确状态和约束"


def test_code_step_hint_defaults_to_empty() -> None:
    """思路阶梯的一步没给提示时提示为空字符串。"""
    step = CodeStep(id="p1", title="理解题意")
    assert step.hint == ""


def test_code_step_requires_id_and_title() -> None:
    """思路阶梯的一步缺 id 或标题时校验失败。"""
    with pytest.raises(ValidationError):
        CodeStep(id="p1")
    with pytest.raises(ValidationError):
        CodeStep(title="理解题意")


def test_code_framework_event_defaults() -> None:
    """代码题的框架事件默认是 coding 模式，steps 默认为空列表。"""
    event = CodeFrameworkEvent()
    assert event.type == "framework"
    assert event.mode == "coding"
    assert event.steps == []


def test_code_framework_event_keeps_steps() -> None:
    """代码题的框架事件带着声明出来的思路阶梯。"""
    event = CodeFrameworkEvent(
        mode="coding",
        steps=[CodeStep(id="p1", title="理解题意", hint="明确状态和约束")],
        request_id="req-1",
        session_id="sess-1",
    )
    assert event.steps[0].id == "p1"
    assert event.steps[0].title == "理解题意"
    assert event.steps[0].hint == "明确状态和约束"
    assert event.request_id == "req-1"
    assert event.session_id == "sess-1"


def test_code_framework_event_is_not_a_framework_event() -> None:
    """代码题的框架事件与普通题的框架事件并列，不互相继承。"""
    assert issubclass(CodeFrameworkEvent, StreamEvent)
    assert not issubclass(CodeFrameworkEvent, FrameworkEvent)


def test_code_delta_event_keeps_text() -> None:
    """代码增量事件保留代码原文。"""
    event = CodeDeltaEvent(text="#include <bits/stdc++.h>\n")
    assert event.type == "code_delta"
    assert event.text == "#include <bits/stdc++.h>\n"


def test_code_delta_event_requires_text() -> None:
    """代码增量事件缺 text 时校验失败。"""
    with pytest.raises(ValidationError):
        CodeDeltaEvent()


def test_code_complete_event_defaults() -> None:
    """代码完成事件默认是 C++17 自动提交模式。"""
    event = CodeCompleteEvent()
    assert event.type == "code_complete"
    assert event.language == "cpp"
    assert event.standard == "cpp17"
    assert event.submission_mode == "auto"


def test_code_complete_event_keeps_fields() -> None:
    """代码完成事件保留语言、标准和提交格式。"""
    event = CodeCompleteEvent(
        language="cpp",
        standard="cpp17",
        submission_mode="acm",
        request_id="req-1",
        session_id="sess-1",
    )
    assert event.language == "cpp"
    assert event.standard == "cpp17"
    assert event.submission_mode == "acm"
    assert event.request_id == "req-1"
    assert event.session_id == "sess-1"


def test_code_events_do_not_inherit_event_bus_event() -> None:
    """代码题的事件同样不走事件总线。"""
    assert not issubclass(CodeFrameworkEvent, Event)
    assert not issubclass(CodeDeltaEvent, Event)
    assert not issubclass(CodeCompleteEvent, Event)
