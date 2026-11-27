"""检索接线的测试。

覆盖《6-Dify检索与项目知识》检索流程、故障降级与检索来源展示三节：
策略说要检索就发起检索并把上下文拼进模型请求；策略说不要检索就不发检索请求；
检索失败或结果为空时不崩掉、请求照常发、会话标记没取到项目检索结果；
来源存进会话并送给界面。

检索、模型客户端、解析器与事件分发器都用假实现，验证的是编排逻辑本身。
core 层不 import 检索层与界面层的实现类，这里也不用它们的实现类。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.config import AppConfig, load_config
from app.core.question_manager import QuestionManager

# 题面样例：一个命中项目关键词，一个是不涉及项目的普通知识题
QUESTION_PROJECT = "讲讲我的项目是怎么做的"
QUESTION_PLAIN = "解释一下快速排序"

# 检索回调返回的上下文与来源条目
CONTEXT_TEXT = "[片段 1] 来源文档：设计文档\n项目用 Dify 做项目知识检索。"
SOURCES: list[dict] = [
    {"document_name": "设计文档", "score": 0.83, "content": "项目用 Dify 做项目知识检索。"},
]

# 截图的 data URL
IMAGE_DATA_URL = "data:image/png;base64,AAAA"


class FakeEvent:
    """假事件，只带请求标识与内容。"""

    def __init__(self, request_id: str | None = None, text: str = "") -> None:
        self.request_id = request_id
        self.text = text


class FakeClient:
    """假模型客户端，记录调用与取消。"""

    def __init__(self, chunks: list[str] | None = None) -> None:
        self.chunks: list[str] = list(chunks or ["片段一", "片段二"])
        self.calls: list[tuple[list[dict[str, str]], str]] = []
        self.multimodal_calls: list[tuple[list[dict], str]] = []
        self.cancelled: list[str] = []

    async def chat_stream(self, messages: list[dict[str, str]], request_id: str):
        self.calls.append((messages, request_id))
        for chunk in self.chunks:
            yield chunk

    async def multimodal_stream(self, messages: list[dict], request_id: str):
        self.multimodal_calls.append((messages, request_id))
        for chunk in self.chunks:
            yield chunk

    async def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)


class FakeParser:
    """假解析器，产出带请求标识的假事件。"""

    def __init__(self, request_id: str | None = None) -> None:
        self.request_id = request_id
        self.feeds: list[str] = []

    def reset(self) -> None:
        return None

    def feed(self, text: str) -> list[FakeEvent]:
        self.feeds.append(text)
        return [FakeEvent(request_id=self.request_id, text=text)]

    def flush(self) -> list[FakeEvent]:
        return []


class FakeSink:
    """假事件分发器，记录派发、清屏与错误。"""

    def __init__(self) -> None:
        self.dispatched: list[FakeEvent] = []
        self.clear_count: int = 0
        self.errors: list[tuple[str, str]] = []

    def dispatch(self, event: FakeEvent) -> None:
        self.dispatched.append(event)

    def clear(self) -> None:
        self.clear_count += 1

    def report_error(self, code: str, detail: str) -> None:
        self.errors.append((code, detail))


class FakeRetriever:
    """假检索回调，记录查询，返回固定的上下文与来源。"""

    def __init__(
        self,
        context: str | None = CONTEXT_TEXT,
        sources: list[dict] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.context = context
        self.sources = list(SOURCES if sources is None else sources)
        self.error = error
        self.queries: list[str] = []

    async def __call__(self, query: str) -> tuple[str | None, list[dict]]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.context, list(self.sources)


class BlockingRetriever:
    """假检索回调，挂起直到测试放行。"""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.started: asyncio.Event = asyncio.Event()
        self.cancelled: bool = False
        self._release: asyncio.Event = asyncio.Event()

    async def __call__(self, query: str) -> tuple[str | None, list[dict]]:
        self.queries.append(query)
        self.started.set()
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return CONTEXT_TEXT, list(SOURCES)

    def release(self) -> None:
        """放行挂起的检索。"""
        self._release.set()


@dataclass
class FakeResult:
    """带来源字段的假对象，用来验证 core 层不把检索层的类透出来。"""

    content: str
    document_name: str
    score: float | None = None


class RecordingBuilder:
    """记录调用参数的消息构造回调，末位参数是检索上下文。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(
        self,
        question: str,
        question_type: str,
        context: str | None = None,
    ) -> list[dict[str, str]]:
        self.calls.append((question, question_type, context))
        content = question
        if context:
            content = f"{question}\n\n背景上下文：\n{context}"
        return [{"role": "user", "content": content}]


class TwoArgBuilder:
    """老签名的消息构造回调，只收题面与题型。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, question: str, question_type: str) -> list[dict[str, str]]:
        self.calls.append((question, question_type))
        return [{"role": "user", "content": question}]


class RecordingMultimodalBuilder:
    """记录调用参数的带图消息构造回调，末位参数是检索上下文。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str | None]] = []

    def __call__(
        self,
        question: str,
        image_data_url: str,
        question_type: str,
        context: str | None = None,
    ) -> list[dict]:
        self.calls.append((question, image_data_url, question_type, context))
        return [
            {"role": "system", "content": "系统"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question if context is None else f"{question}{context}"},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]


def make_manager(
    retrieve: object | None = None,
    on_sources: object | None = None,
    builder: object | None = None,
    multimodal_builder: object | None = None,
    client: FakeClient | None = None,
    sink: FakeSink | None = None,
    config: AppConfig | None = None,
) -> tuple[QuestionManager, FakeClient, FakeSink]:
    """搭一套假实现，返回管理器与假对象。

    :param retrieve: 注入的检索回调，不给就是没有注入检索回调。
    :param on_sources: 注入的来源条目回调。
    """
    config = config if config is not None else load_config()
    client = client if client is not None else FakeClient()
    sink = sink if sink is not None else FakeSink()
    builder = builder if builder is not None else RecordingBuilder()
    multimodal_builder = (
        multimodal_builder
        if multimodal_builder is not None
        else RecordingMultimodalBuilder()
    )
    manager = QuestionManager(
        config,
        client,
        lambda request_id, session_id: FakeParser(request_id),
        sink,
        builder,
        build_multimodal_messages=multimodal_builder,
        retrieve=retrieve,
        on_sources=on_sources,
    )
    return manager, client, sink


# ----------------------------------------------------------------------
# 检索的触发条件与流程，见《6-Dify检索与项目知识》检索的触发条件与检索流程两节
# ----------------------------------------------------------------------
async def test_retrieve_is_called_and_context_reaches_the_request() -> None:
    """策略说要检索就发起检索，上下文拼进模型请求。"""
    retriever = FakeRetriever()
    builder = RecordingBuilder()
    manager, client, _sink = make_manager(retrieve=retriever, builder=builder)

    await manager.submit_text(QUESTION_PROJECT)

    assert retriever.queries == [QUESTION_PROJECT]
    assert builder.calls[0][2] == CONTEXT_TEXT
    assert len(client.calls) == 1
    messages, _request_id = client.calls[0]
    assert CONTEXT_TEXT in messages[0]["content"]


async def test_plain_question_does_not_call_retrieve() -> None:
    """普通知识题不发出检索请求。"""
    retriever = FakeRetriever()
    manager, client, _sink = make_manager(retrieve=retriever)

    await manager.submit_text(QUESTION_PLAIN)

    assert retriever.queries == []
    assert len(client.calls) == 1
    session = manager.current_session
    assert session is not None
    assert session.retrieval_sources == []
    assert session.retrieval_missing is False


async def test_multimodal_request_also_carries_context() -> None:
    """带图请求同样带上检索上下文。"""
    retriever = FakeRetriever()
    multimodal = RecordingMultimodalBuilder()
    manager, client, _sink = make_manager(
        retrieve=retriever,
        multimodal_builder=multimodal,
    )

    await manager.submit_text(QUESTION_PROJECT, image_data_url=IMAGE_DATA_URL)

    assert retriever.queries == [QUESTION_PROJECT]
    assert multimodal.calls[0][3] == CONTEXT_TEXT
    assert len(client.multimodal_calls) == 1


async def test_reanalyze_retrieves_again() -> None:
    """重新分析按同一份材料重新检索一次。"""
    retriever = FakeRetriever()
    manager, client, _sink = make_manager(retrieve=retriever)

    await manager.submit_text(QUESTION_PROJECT)
    await manager.reanalyze()

    assert retriever.queries == [QUESTION_PROJECT, QUESTION_PROJECT]
    assert len(client.calls) == 2


# ----------------------------------------------------------------------
# 故障降级，见《6-Dify检索与项目知识》故障降级一节
# ----------------------------------------------------------------------
async def test_retrieve_failure_does_not_raise_and_marks_session() -> None:
    """检索回调抛异常时不崩掉，请求照常发，会话标记没取到。"""
    retriever = FakeRetriever(error=RuntimeError("Dify 连不上"))
    manager, client, _sink = make_manager(retrieve=retriever)

    await manager.submit_text(QUESTION_PROJECT)

    assert len(client.calls) == 1
    messages, _request_id = client.calls[0]
    assert "背景上下文" not in messages[0]["content"]
    session = manager.current_session
    assert session is not None
    assert session.retrieval_missing is True
    assert session.retrieval_sources == []


async def test_empty_retrieval_result_is_treated_as_missing() -> None:
    """检索结果为空时按没取到处理，不把空上下文塞进请求。"""
    retriever = FakeRetriever(context=None, sources=[])
    manager, client, _sink = make_manager(retrieve=retriever)

    await manager.submit_text(QUESTION_PROJECT)

    assert len(client.calls) == 1
    messages, _request_id = client.calls[0]
    assert "背景上下文" not in messages[0]["content"]
    session = manager.current_session
    assert session is not None
    assert session.retrieval_missing is True
    assert session.retrieval_sources == []


async def test_blank_context_is_not_put_into_the_request() -> None:
    """上下文是空串时同样不塞进请求。"""
    retriever = FakeRetriever(context="   ", sources=[])
    manager, client, _sink = make_manager(retrieve=retriever)

    await manager.submit_text(QUESTION_PROJECT)

    messages, _request_id = client.calls[0]
    assert "背景上下文" not in messages[0]["content"]
    session = manager.current_session
    assert session is not None
    assert session.retrieval_missing is True


# ----------------------------------------------------------------------
# 检索来源展示，见《6-Dify检索与项目知识》检索来源展示一节
# ----------------------------------------------------------------------
async def test_sources_are_stored_and_sent_to_on_sources() -> None:
    """来源存进会话，并通过 on_sources 送出一次。"""
    retriever = FakeRetriever()
    received: list[list[dict]] = []
    manager, _client, _sink = make_manager(retrieve=retriever, on_sources=received.append)

    await manager.submit_text(QUESTION_PROJECT)

    session = manager.current_session
    assert session is not None
    assert session.retrieval_sources == SOURCES
    assert len(received) == 1
    assert received[0] == SOURCES


async def test_on_sources_is_not_called_without_results() -> None:
    """没有检索结果时不送来源条目。"""
    received: list[list[dict]] = []
    retriever = FakeRetriever(context=None, sources=[])
    manager, _client, _sink = make_manager(retrieve=retriever, on_sources=received.append)

    await manager.submit_text(QUESTION_PROJECT)

    assert received == []


async def test_sources_are_normalized_to_plain_dicts() -> None:
    """来源条目统一成普通字典，不把检索层的对象透出来。"""
    retriever = FakeRetriever(
        context=CONTEXT_TEXT,
        sources=[
            {
                "document_name": "设计文档",
                "score": 0.83,
                "content": "项目用 Dify 做项目知识检索。",
            }
        ],
    )
    manager, _client, _sink = make_manager(retrieve=retriever)

    await manager.submit_text(QUESTION_PROJECT)

    session = manager.current_session
    assert session is not None
    entry = session.retrieval_sources[0]
    assert type(entry) is dict
    assert entry["document_name"] == "设计文档"
    assert entry["score"] == 0.83
    assert entry["content"] == "项目用 Dify 做项目知识检索。"


async def test_sources_from_objects_are_converted_to_dicts() -> None:
    """回调给出带同名属性的对象时也转成普通字典。"""

    async def retrieve(query: str) -> tuple[str | None, list[dict]]:
        return CONTEXT_TEXT, [FakeResult(content="正文", document_name="设计文档", score=0.5)]

    manager, _client, _sink = make_manager(retrieve=retrieve)

    await manager.submit_text(QUESTION_PROJECT)

    session = manager.current_session
    assert session is not None
    assert session.retrieval_sources == [
        {"document_name": "设计文档", "score": 0.5, "content": "正文"}
    ]


# ----------------------------------------------------------------------
# 请求取消，见《8-集成与验收》请求取消一节
# ----------------------------------------------------------------------
async def test_new_question_cancels_running_retrieval() -> None:
    """新问题到达时，正在进行的检索被取消。"""
    retriever = BlockingRetriever()
    manager, _client, _sink = make_manager(retrieve=retriever)

    task = asyncio.create_task(manager.submit_text(QUESTION_PROJECT))
    await retriever.started.wait()

    await manager.submit_text(QUESTION_PLAIN)

    assert retriever.cancelled is True
    with pytest.raises(asyncio.CancelledError):
        await task
    retriever.release()


# ----------------------------------------------------------------------
# 回归保护：没有注入检索回调时行为与改动前一致
# ----------------------------------------------------------------------
async def test_without_retrieve_callback_behavior_is_unchanged() -> None:
    """没注入检索回调时不发起检索，请求与改动前一样。"""
    builder = TwoArgBuilder()
    manager, client, _sink = make_manager(builder=builder)

    await manager.submit_text(QUESTION_PROJECT)

    assert builder.calls == [(QUESTION_PROJECT, "project")]
    assert len(client.calls) == 1
    messages, _request_id = client.calls[0]
    assert messages == [{"role": "user", "content": QUESTION_PROJECT}]
    session = manager.current_session
    assert session is not None
    assert session.retrieval_sources == []


async def test_retrieval_switch_off_does_not_call_retrieve() -> None:
    """检索开关关闭时不发起检索。"""
    config = load_config()
    config.rag.enabled = False
    retriever = FakeRetriever()
    manager, client, _sink = make_manager(retrieve=retriever, config=config)

    await manager.submit_text(QUESTION_PROJECT)

    assert retriever.queries == []
    assert len(client.calls) == 1
    session = manager.current_session
    assert session is not None
    assert session.retrieval_missing is False
