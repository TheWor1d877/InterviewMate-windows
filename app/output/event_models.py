"""模型流式事件的数据结构。

事件定义见《3-文本输入与流式回答》流式事件协议一节，模型按 JSONL 一行一个事件输出：

- ``framework``：给出回答的分层结构；
- ``item``：某一层下的一个关键点加一句表达；
- ``section_done``：某一层完成；
- ``done``：整体完成。

代码题的事件见《7-代码问题模式》流式事件一节：

- ``framework``（``mode`` 为 ``coding``）：给出思路阶梯；
- ``code_delta``：追加一段代码；
- ``code_complete``：代码输出完成。

这些事件不走 :mod:`app.core.event_bus`：事件总线只管模块间投递，
模型流式事件由解析器与分发器直接衔接。

:class:`RouteEvent` 是截图请求的路由事件，模型在回复第一行给出，
决定这次按代码题还是普通题回答，见《4-截图输入与多模态》：
它的协议行是 ``{"event": "route", "kind": "..."}``，键名是 ``event`` 而不是 ``type``，
所以不在 :data:`EVENT_TYPES` 里。

:class:`FallbackEvent` 是解析器自己产出的兜底事件，不属于模型协议，
所以不在 :data:`EVENT_TYPES` 里。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "FrameworkSection",
    "StreamEvent",
    "FrameworkEvent",
    "ItemEvent",
    "SectionDoneEvent",
    "DoneEvent",
    "CodeStep",
    "CodeFrameworkEvent",
    "CodeDeltaEvent",
    "CodeCompleteEvent",
    "RouteEvent",
    "FallbackEvent",
    "EVENT_TYPES",
    "ROUTE_KINDS",
]


class FrameworkSection(BaseModel):
    """框架里的一层。"""

    id: str
    title: str


class StreamEvent(BaseModel):
    """模型流式事件基类。"""

    type: str
    request_id: str | None = None
    session_id: str | None = None


class FrameworkEvent(StreamEvent):
    """框架事件：给出回答的分层结构。"""

    type: str = "framework"
    mode: str = "normal"
    sections: list[FrameworkSection] = Field(default_factory=list)


class ItemEvent(StreamEvent):
    """内容单元事件：某一层下的一个关键点加一句表达。"""

    type: str = "item"
    section_id: str
    keyword: str
    sentence: str


class SectionDoneEvent(StreamEvent):
    """某一层完成。"""

    type: str = "section_done"
    section_id: str


class DoneEvent(StreamEvent):
    """整体完成。"""

    type: str = "done"


class CodeStep(BaseModel):
    """代码题思路阶梯的一步。"""

    id: str
    title: str
    hint: str = ""


class CodeFrameworkEvent(StreamEvent):
    """代码题的框架事件：给出思路阶梯。

    见《7-代码问题模式》流式事件一节。它不继承 :class:`FrameworkEvent`：
    代码题走 steps，普通题走 sections，两者是并列的两种模式。
    """

    type: str = "framework"
    mode: str = "coding"
    steps: list[CodeStep] = Field(default_factory=list)


class CodeDeltaEvent(StreamEvent):
    """代码增量事件：追加一段代码。"""

    type: str = "code_delta"
    text: str


class CodeCompleteEvent(StreamEvent):
    """代码完成事件：代码给完了。"""

    type: str = "code_complete"
    language: str = "cpp"
    standard: str = "cpp17"
    submission_mode: str = "auto"


class FallbackEvent(StreamEvent):
    """兜底内容事件：模型输出的普通文字或解析失败的原文。"""

    type: str = "fallback"
    text: str


# 路由事件允许的两种取值：代码题与普通题
ROUTE_CODING: str = "coding"
ROUTE_NORMAL: str = "normal"
ROUTE_KINDS: tuple[str, ...] = (ROUTE_CODING, ROUTE_NORMAL)


class RouteEvent(StreamEvent):
    """路由事件：截图请求先声明这次按代码题还是普通题回答。

    协议行是 ``{"event": "route", "kind": "coding"}``，
    ``kind`` 只允许 ``coding`` 与 ``normal``，取值见 :data:`ROUTE_KINDS`；
    非法或缺失的取值由解析器按 ``normal`` 兜住，见《4-截图输入与多模态》。
    """

    type: str = "route"
    kind: str = ROUTE_NORMAL


# 模型协议里的事件类型，解析器按这张表分派：
# 普通题四种见《3-文本输入与流式回答》流式事件协议一节，
# 代码题两类见《7-代码问题模式》流式事件一节
EVENT_TYPES: tuple[str, ...] = (
    "framework",
    "item",
    "section_done",
    "done",
    "code_delta",
    "code_complete",
)
