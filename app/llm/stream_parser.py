"""模型流式输出的 JSONL 解析器。

模型按 JSONL 一行一个事件输出，普通题见《3-文本输入与流式回答》流式事件协议一节，
代码题见《7-代码问题模式》流式事件一节。
解析器维护一个流式缓冲区，只在读到完整换行时才尝试解析一条事件，
因此一条 JSON 被拆到多个网络分片也能拼回来。

模型不按事件协议输出、直接给一段带 markdown 代码围栏的文本时，
解析器把整个围栏块聚合成一个兜底事件，见《7-代码问题模式》纯代码块兜底一节：
围栏块连首尾两行围栏标记一起交给下游，下游据此把代码正文与说明文字拆开。

容错规则：

- 空行直接跳过；
- 不是合法 JSON 的行当成普通内容，进兜底区；
- 字段缺失的事件进兜底区，原始文本照常交给界面；
- 事件类型不在协议里的行也进兜底区。

解析器只管语法与字段，section_id 是不是在框架里由分发器判断。
解析器自己记一份 framework 声明过的 section id，用于记日志。
代码题没有 section_id 概念，不记。

截图请求的第一行是路由事件 ``{"event": "route", "kind": "coding"}``，
键名是 ``event`` 而不是 ``type``，解析器单独认这一行，产出
:class:`~app.output.event_models.RouteEvent`，见《4-截图输入与多模态》。
``kind`` 非法或缺失时按 ``normal`` 处理并记日志，不抛异常；
路由事件出现在流的任意位置都不改变后续行的解析方式。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.output.event_models import (
    CodeCompleteEvent,
    CodeDeltaEvent,
    CodeFrameworkEvent,
    DoneEvent,
    EVENT_TYPES,
    FallbackEvent,
    FrameworkEvent,
    ItemEvent,
    ROUTE_KINDS,
    ROUTE_NORMAL,
    RouteEvent,
    SectionDoneEvent,
    StreamEvent,
)
from app.output.response_normalizer import ResponseNormalizer
from app.utils.logger import log_event

__all__ = ["StreamParser"]


# 日志里保留的兜底文本长度上限
_LOG_SNIPPET_LIMIT = 120

# 路由事件的名称：协议行是 {"event": "route", "kind": "..."}
_ROUTE_EVENT_NAME = "route"

# 围栏块兜底事件在日志里的来历说明
_CODE_BLOCK_REASON = "模型输出的是代码块"

# 一行开头的 markdown 代码围栏标记：三个以上的反引号或波浪号，允许前导缩进。
# 规则与 ResponseNormalizer 认围栏一致，围栏块才会被下游认出来
_FENCE_LINE_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")

# 一行里的 markdown 代码围栏标记，用来找行内围栏的结束标记
_FENCE_MARKER_PATTERN = re.compile(r"`{3,}|~{3,}")


def _text_field(payload: dict[str, Any], key: str) -> str | None:
    """取一个必填的文本字段。

    取不到、不是字符串、或者只有空白，都算字段缺失，返回 None。
    """
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _raw_text_field(payload: dict[str, Any], key: str) -> str | None:
    """取一个保留原始空白的文本字段。

    取不到、不是字符串、或者只有空白，都算字段缺失，返回 None。
    文本本身原样返回：代码块的缩进和换行不能被裁掉。
    """
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _fence_marker_of(line: str) -> tuple[str, str] | None:
    """取一行里的围栏标记。

    这一行不是围栏行时返回 None，否则返回 (标记, 标记后面的文本)。
    """
    match = _FENCE_LINE_PATTERN.match(line)
    if match is None:
        return None
    return match.group(1), line[match.end() :]


def _find_inline_fence(line: str) -> tuple[str, str, str, str] | None:
    """认行内围栏：同一行里既有开始标记也有结束标记。

    返回 (开始标记, 语言标记, 代码正文, 结束标记)，不是行内围栏时返回 None。
    语言标记取开始标记后的第一个词，只有它是已知语言名时才当成语言标记，
    否则整行余下的内容都是代码正文。
    """
    found = _fence_marker_of(line)
    if found is None:
        return None
    marker, rest = found
    close_at = -1
    close_text = ""
    for match in _FENCE_MARKER_PATTERN.finditer(rest):
        candidate = match.group(0)
        # 结束围栏要用同类标记，长度不短于开始围栏
        if candidate[0] == marker[0] and len(candidate) >= len(marker):
            close_at = match.start()
            close_text = candidate
            break
    if close_at < 0:
        return None
    head = rest[:close_at]
    words = head.split()
    if words and ResponseNormalizer.detect_code_language(marker + words[0]):
        return marker, words[0], head[len(words[0]) :].strip(), close_text
    return marker, "", head.strip(), close_text


class StreamParser:
    """JSONL 流式解析器。

    :param request_id: 本次请求的标识，产出的事件都带上它。
    :param session_id: 本次会话的标识，产出的事件都带上它。
    """

    def __init__(self, request_id: str | None = None, session_id: str | None = None) -> None:
        self._request_id = request_id
        self._session_id = session_id
        # 还没收到换行符的残留文本
        self._buffer = ""
        # framework 声明过的 section id
        self._section_ids: set[str] = set()
        # 当前围栏的开始标记，不在围栏内时是空串
        self._fence_marker = ""
        # 围栏里累积的行，含首尾两行围栏标记
        self._fence_lines: list[str] = []

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空缓冲区、已声明的 section id 集合与围栏累积状态，下次请求前调用。"""
        self._buffer = ""
        self._section_ids.clear()
        self._clear_fence()

    def feed(self, text: str) -> list[StreamEvent]:
        """喂入一段增量文本，返回本次能解析出的事件列表。

        一次 feed 里可能出多个事件。文本没读到换行符时先攒在缓冲区里。
        """
        if not text:
            return []
        self._buffer += text
        events: list[StreamEvent] = []
        while "\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition("\n")
            event = self._consume_line(line)
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> list[StreamEvent]:
        """流结束时调用。

        缓冲区里没有换行符的残留当一整行处理：合法 JSON 解析成事件，
        非法就进兜底区。还在围栏里说明模型输出到一半结束，
        累积的内容当成一个未闭合的代码块冲刷出去。返回剩余事件。
        """
        remaining = self._buffer
        self._buffer = ""
        events: list[StreamEvent] = []
        if remaining:
            event = self._consume_line(remaining)
            if event is not None:
                events.append(event)
        if self._fence_lines:
            # 围栏没闭合，剩余内容算代码块，规则与下游对未闭合围栏的处理一致
            block = "\n".join(self._fence_lines)
            self._clear_fence()
            events.append(self._fallback(block, _CODE_BLOCK_REASON))
        return events

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def section_ids(self) -> frozenset[str]:
        """已经声明过的 section id 集合。"""
        return frozenset(self._section_ids)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _consume_line(self, line: str) -> StreamEvent | None:
        """处理一行文本。

        围栏外的行走 :meth:`_parse_line`，行为与原来一致；
        围栏里的行累积起来，整个围栏块产成一个兜底事件，
        见《7-代码问题模式》纯代码块兜底一节。
        """
        if self._fence_lines:
            # 围栏内：累积这一行，遇到结束围栏就把整块产成一个兜底事件
            self._fence_lines.append(line)
            if self._closes_fence(line):
                block = "\n".join(self._fence_lines)
                self._clear_fence()
                return self._fallback(block, _CODE_BLOCK_REASON)
            return None
        inline = _find_inline_fence(line)
        if inline is not None:
            # 行内围栏：一行里既有开始标记也有结束标记，拆成三行交给下游认
            marker, language, content, close = inline
            block = f"{marker}{language}\n{content}\n{close}"
            return self._fallback(block, _CODE_BLOCK_REASON)
        found = _fence_marker_of(line)
        if found is not None:
            # 围栏开始行：进入围栏状态，开始累积
            self._fence_marker = found[0]
            self._fence_lines = [line]
            return None
        return self._parse_line(line)

    def _closes_fence(self, line: str) -> bool:
        """这一行是不是当前围栏的结束行。

        结束围栏要用同类标记，长度不短于开始围栏。
        """
        found = _fence_marker_of(line)
        if found is None:
            return False
        marker = found[0]
        return marker[0] == self._fence_marker[0] and len(marker) >= len(self._fence_marker)

    def _clear_fence(self) -> None:
        """清空围栏累积状态。"""
        self._fence_marker = ""
        self._fence_lines = []

    def _parse_line(self, line: str) -> StreamEvent | None:
        """解析一行文本。

        空行返回 None；解析失败或者字段缺失时返回兜底事件。
        """
        stripped = line.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except ValueError:
            return self._fallback(stripped, "不是合法 JSON")
        if not isinstance(payload, dict):
            return self._fallback(stripped, "JSON 不是对象")
        if payload.get("event") == _ROUTE_EVENT_NAME or payload.get("type") == _ROUTE_EVENT_NAME:
            # 截图请求的路由事件，键名是 event，不是 type
            return self._build_route_event(payload)
        event_type = payload.get("type")
        if not isinstance(event_type, str) or event_type not in EVENT_TYPES:
            return self._fallback(stripped, "事件类型不在协议里")
        event = self._build_event(event_type, payload)
        if event is None:
            return self._fallback(stripped, "事件字段缺失")
        if isinstance(event, FrameworkEvent):
            # 代码题的框架事件不是 FrameworkEvent，代码题没有 section id 概念
            self._section_ids.update(section.id for section in event.sections)
        elif isinstance(event, ItemEvent) and event.section_id not in self._section_ids:
            # section 是不是在框架里由分发器判断，这里只记一条日志
            log_event(
                "llm.stream.unknown_section",
                "item 的 section_id 不在已声明的框架里",
                level="DEBUG",
                request_id=self._request_id,
                session_id=self._session_id,
                section_id=event.section_id,
            )
        return event

    def _build_route_event(self, payload: dict[str, Any]) -> RouteEvent:
        """构造路由事件。

        ``kind`` 只认 ``coding`` 与 ``normal``，其它取值或缺失时按 ``normal``
        处理并记一条 WARNING，不抛异常，见《4-截图输入与多模态》。
        路由事件不改解析器的任何状态，后面两种协议的行照常解析。

        :param payload: 这一行的 JSON 对象。
        """
        raw = payload.get("kind")
        kind = raw.strip().lower() if isinstance(raw, str) else ""
        if kind not in ROUTE_KINDS:
            log_event(
                "llm.stream.route_invalid",
                "路由事件的 kind 非法或缺失，按 normal 处理",
                level="WARNING",
                request_id=self._request_id,
                session_id=self._session_id,
                kind=raw,
            )
            kind = ROUTE_NORMAL
        log_event(
            "llm.stream.route",
            "收到路由事件",
            level="DEBUG",
            request_id=self._request_id,
            session_id=self._session_id,
            kind=kind,
        )
        return RouteEvent(
            kind=kind,
            request_id=self._request_id,
            session_id=self._session_id,
        )

    def _build_event(self, event_type: str, payload: dict[str, Any]) -> StreamEvent | None:
        """按类型构造事件，字段缺失时返回 None。"""
        if event_type == "framework":
            mode = payload.get("mode")
            mode_text = mode.strip() if isinstance(mode, str) and mode.strip() else "normal"
            if mode_text.lower() == "coding":
                # 代码题的框架事件带 steps，见《7-代码问题模式》流式事件一节
                try:
                    return CodeFrameworkEvent(
                        mode="coding",
                        steps=payload.get("steps", []),
                        request_id=self._request_id,
                        session_id=self._session_id,
                    )
                except ValidationError:
                    return None
            try:
                return FrameworkEvent(
                    mode=mode_text,
                    sections=payload.get("sections", []),
                    request_id=self._request_id,
                    session_id=self._session_id,
                )
            except ValidationError:
                return None
        if event_type == "item":
            section_id = _text_field(payload, "section_id")
            keyword = _text_field(payload, "keyword")
            sentence = _text_field(payload, "sentence")
            if section_id is None or keyword is None or sentence is None:
                return None
            return ItemEvent(
                section_id=section_id,
                keyword=keyword,
                sentence=sentence,
                request_id=self._request_id,
                session_id=self._session_id,
            )
        if event_type == "section_done":
            section_id = _text_field(payload, "section_id")
            if section_id is None:
                return None
            return SectionDoneEvent(
                section_id=section_id,
                request_id=self._request_id,
                session_id=self._session_id,
            )
        if event_type == "code_delta":
            # 代码增量原文照收，空白和缩进不能动，见《7-代码问题模式》流式事件一节
            text = _raw_text_field(payload, "text")
            if text is None:
                return None
            return CodeDeltaEvent(
                text=text,
                request_id=self._request_id,
                session_id=self._session_id,
            )
        if event_type == "code_complete":
            # 三个字段缺失不算错误，用事件里的默认值兜住
            fields: dict[str, str] = {}
            for name in ("language", "standard", "submission_mode"):
                value = _text_field(payload, name)
                if value is not None:
                    fields[name] = value
            return CodeCompleteEvent(
                **fields,
                request_id=self._request_id,
                session_id=self._session_id,
            )
        return DoneEvent(request_id=self._request_id, session_id=self._session_id)

    def _fallback(self, text: str, reason: str) -> FallbackEvent:
        """产出一个兜底事件，原始文本照常交给界面。"""
        log_event(
            "llm.stream.fallback",
            "模型输出进兜底区",
            level="WARNING",
            request_id=self._request_id,
            session_id=self._session_id,
            reason=reason,
            text=text[:_LOG_SNIPPET_LIMIT],
        )
        return FallbackEvent(
            text=text,
            request_id=self._request_id,
            session_id=self._session_id,
        )
