"""性能指标接线的测试。

覆盖《8-集成与验收》性能统计一节：

- 一次完整流式请求后 ``request_latency``、``ttft``、``first_framework``、
  ``first_item``、``total_latency`` 各有一条样本；
- 代码题的事件也算框架事件与内容单元；
- 同一个请求里同名指标不重复记；
- 请求被取消时不产生 ``total_latency`` 样本；
- 检索跑一次后 ``retrieval_latency`` 有一条样本，检索失败也有。

模型客户端与检索后端都用假实现，不发网络请求，不加载模型。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import load_config
from app.core.exceptions import DeepSeekStreamError, DifyTimeoutError
from app.core.question_manager import QuestionManager
from app.llm.stream_parser import StreamParser
from app.retrieval.dify_retriever import (
    DifyRetriever,
    TransportResponse,
    TransportTimeoutError,
)
from app.retrieval.mock_retriever import MockRetriever
from app.utils.timing import METRIC_NAMES, default_recorder, reset_default_recorder

# 题面样例
QUESTION = "讲讲我的项目是怎么做的"

# 请求类指标，一次完整流式请求后应该各有一条样本
REQUEST_METRICS = (
    "request_latency",
    "ttft",
    "first_framework",
    "first_item",
    "total_latency",
)

# 普通题的事件流，见《3-文本输入与流式回答》流式事件协议一节
_FRAMEWORK_LINE = json.dumps(
    {
        "type": "framework",
        "mode": "normal",
        "sections": [{"id": "s1", "title": "基本原理"}],
    },
    ensure_ascii=False,
)
_ITEM_LINE = json.dumps(
    {
        "type": "item",
        "section_id": "s1",
        "keyword": "资源管理",
        "sentence": "把资源的申请与释放绑定到对象生命周期上。",
    },
    ensure_ascii=False,
)

# 代码题的事件流，见《7-代码问题模式》流式事件一节
_CODING_FRAMEWORK_LINE = json.dumps(
    {
        "type": "framework",
        "mode": "coding",
        "steps": [{"id": "p1", "title": "题意梳理"}],
    },
    ensure_ascii=False,
)
_CODE_DELTA_LINE = json.dumps(
    {"type": "code_delta", "text": "int main() { return 0; }\n"},
    ensure_ascii=False,
)

# 正常题的片段：框架事件在前，内容单元在后
NORMAL_CHUNKS = [f"{_FRAMEWORK_LINE}\n", f"{_ITEM_LINE}\n"]

# 代码题的片段：框架事件在前，代码增量在后
CODING_CHUNKS = [f"{_CODING_FRAMEWORK_LINE}\n", f"{_CODE_DELTA_LINE}\n"]

# Dify 的成功响应体，见《接口契约》Dify 检索响应体一节
_OK_BODY = json.dumps(
    {
        "records": [
            {
                "segment": {
                    "content": "片段正文",
                    "document": {"name": "架构说明.md"},
                },
                "score": 0.8,
            }
        ]
    },
    ensure_ascii=False,
)


class FakeClient:
    """假模型客户端：按预置片段产出流，记录调用与取消。

    :param chunks: 依次产出的流式片段。
    :param name: 客户端名，也是耗时指标的来源标记；
        None 表示不给这个属性，来源退回配置里的取值。
    :param error: 片段产完后抛出的异常。
    """

    def __init__(
        self,
        chunks: list[str] | None = None,
        *,
        name: str | None = "mock",
        error: Exception | None = None,
    ) -> None:
        self.chunks = list(chunks if chunks is not None else NORMAL_CHUNKS)
        self.error = error
        self.calls: list[tuple[list[dict], str]] = []
        self.cancelled: list[str] = []
        # 第一段片段被消费之后置位，测试用它确定请求已经跑起来
        self.first_chunk_consumed = asyncio.Event()
        self._gate: asyncio.Event | None = None
        if name is not None:
            self.name = name

    def block_second_chunk(self) -> None:
        """让第二段片段挂起，直到测试放行。"""
        self._gate = asyncio.Event()

    async def chat_stream(self, messages: list[dict], request_id: str):
        self.calls.append((messages, request_id))
        async for chunk in self._stream(request_id):
            yield chunk

    async def multimodal_stream(self, messages: list[dict], request_id: str):
        self.calls.append((messages, request_id))
        async for chunk in self._stream(request_id):
            yield chunk

    async def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)

    async def _stream(self, request_id: str):
        for index, chunk in enumerate(self.chunks):
            if index == 1 and self._gate is not None:
                await self._gate.wait()
            await asyncio.sleep(0)
            yield chunk
            if index == 0:
                self.first_chunk_consumed.set()
        if self.error is not None:
            raise self.error


class FakeSink:
    """假事件分发器，记录派发与错误。"""

    def __init__(self) -> None:
        self.dispatched: list[object] = []
        self.errors: list[tuple[str, str]] = []

    def dispatch(self, event: object) -> None:
        self.dispatched.append(event)

    def clear(self) -> None:
        return None

    def report_error(self, code: str, detail: str) -> None:
        self.errors.append((code, detail))


class FakeTransport:
    """假的检索传输层，返回预置响应或抛出预置异常。"""

    def __init__(
        self,
        response: TransportResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    async def post(
        self,
        url: str,
        payload: dict,
        headers: dict,
    ) -> TransportResponse:
        self.calls.append({"url": url, "payload": payload, "headers": headers})
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def make_manager(client: FakeClient) -> QuestionManager:
    """搭一套假实现，返回问题管理器。

    解析器用真的流式解析器，产出的是真的事件对象，
    这样事件类型的判断走的是真实路径。

    :param client: 假模型客户端。
    """
    config = load_config()

    def build_messages(question: str, question_type: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": question}]

    def build_multimodal_messages(
        question: str,
        image_data_url: str,
        question_type: str,
        context: str | None = None,
    ) -> list[dict]:
        return [{"role": "user", "content": [{"type": "text", "text": question}]}]

    return QuestionManager(
        config,
        client,
        lambda request_id, session_id: StreamParser(request_id, session_id),
        FakeSink(),
        build_messages,
        build_multimodal_messages=build_multimodal_messages,
    )


@pytest.fixture(autouse=True)
def _clean_recorder():
    """每个用例前后都清空进程级默认记录器。"""
    reset_default_recorder()
    yield
    reset_default_recorder()


# --------------------------------------------------------------------------
# 请求类指标
# --------------------------------------------------------------------------


async def test_full_stream_records_five_request_metrics() -> None:
    """一次完整流式请求后五个请求类指标各有一条样本。"""
    client = FakeClient(NORMAL_CHUNKS)
    manager = make_manager(client)

    await manager.submit_text(QUESTION)

    recorder = default_recorder()
    assert set(recorder.snapshot()) == set(REQUEST_METRICS)
    for name in REQUEST_METRICS:
        samples = recorder.samples(name)
        assert len(samples) == 1, name
        assert samples[0].value_ms >= 0.0
        assert samples[0].source == "mock"
    # 指标名都在《2-骨架与配置》性能指标一节的冻结清单里
    assert set(REQUEST_METRICS) <= set(METRIC_NAMES)


async def test_coding_stream_records_framework_and_item() -> None:
    """代码题的事件也算框架事件与内容单元。

    代码题走 ``framework``（mode 为 coding）与 ``code_delta``，
    见《7-代码问题模式》流式事件一节。
    """
    client = FakeClient(CODING_CHUNKS)
    manager = make_manager(client)

    await manager.submit_text("请实现一个 LRU Cache")

    recorder = default_recorder()
    assert len(recorder.samples("first_framework")) == 1
    assert len(recorder.samples("first_item")) == 1
    assert len(recorder.samples("total_latency")) == 1


async def test_metric_names_are_not_duplicated_in_one_request() -> None:
    """一次请求里同名指标只记一次。"""
    chunks = [
        f"{_FRAMEWORK_LINE}\n",
        f"{_ITEM_LINE}\n",
        f"{_ITEM_LINE}\n",
        f"{_FRAMEWORK_LINE}\n",
    ]
    client = FakeClient(chunks)
    manager = make_manager(client)

    await manager.submit_text(QUESTION)

    recorder = default_recorder()
    for name in REQUEST_METRICS:
        assert len(recorder.samples(name)) == 1, name


async def test_cancelled_request_has_no_total_latency() -> None:
    """请求被取消时不记完整输出耗时，已经记下的指标保留。"""
    client = FakeClient(NORMAL_CHUNKS)
    client.block_second_chunk()
    manager = make_manager(client)

    task = asyncio.create_task(manager.submit_text(QUESTION))
    await client.first_chunk_consumed.wait()
    await manager.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    recorder = default_recorder()
    assert recorder.samples("total_latency") == []
    assert len(recorder.samples("request_latency")) == 1
    assert len(recorder.samples("ttft")) == 1
    assert len(recorder.samples("first_framework")) == 1
    # 第二段片段没到，内容单元还没出现
    assert recorder.samples("first_item") == []


async def test_stream_error_still_records_total_latency() -> None:
    """流异常中断时照样记一条完整输出耗时。"""
    client = FakeClient(NORMAL_CHUNKS, error=DeepSeekStreamError("流断了"))
    manager = make_manager(client)

    await manager.submit_text(QUESTION)

    recorder = default_recorder()
    assert len(recorder.samples("total_latency")) == 1
    assert len(recorder.samples("ttft")) == 1


async def test_source_follows_client_name() -> None:
    """客户端带 name 时用它当来源标记。"""
    client = FakeClient(NORMAL_CHUNKS, name="deepseek")
    manager = make_manager(client)

    await manager.submit_text(QUESTION)

    samples = default_recorder().samples("ttft")
    assert len(samples) == 1
    assert samples[0].source == "deepseek"


async def test_source_falls_back_to_config() -> None:
    """客户端没给 name 时来源取配置里 providers.llm 的取值。"""
    client = FakeClient(NORMAL_CHUNKS, name=None)
    manager = make_manager(client)

    await manager.submit_text(QUESTION)

    samples = default_recorder().samples("ttft")
    assert len(samples) == 1
    assert samples[0].source == load_config().providers.llm


# --------------------------------------------------------------------------
# 检索类指标
# --------------------------------------------------------------------------


async def test_mock_retrieval_records_latency() -> None:
    """假检索后端也记检索耗时。"""
    retriever = MockRetriever()

    results = await retriever.retrieve(QUESTION)

    assert results
    samples = default_recorder().samples("retrieval_latency")
    assert len(samples) == 1
    assert samples[0].source == "mock"
    # 指标名在《2-骨架与配置》性能指标一节的冻结清单里
    assert "retrieval_latency" in METRIC_NAMES


async def test_mock_retrieval_outcome_records_latency_once() -> None:
    """走 retrieve_outcome 也只记一条检索耗时。"""
    retriever = MockRetriever()

    outcome = await retriever.retrieve_outcome(QUESTION)

    assert outcome.ok
    assert len(default_recorder().samples("retrieval_latency")) == 1


async def test_dify_retrieval_records_latency() -> None:
    """真实检索实现记检索耗时，来源是后端名。"""
    transport = FakeTransport(
        response=TransportResponse(status_code=200, body=_OK_BODY)
    )
    retriever = DifyRetriever(
        "https://dify.example.com/v1",
        "ds-1",
        "dataset-key",
        transport=transport,
    )

    results = await retriever.retrieve(QUESTION)

    assert results
    samples = default_recorder().samples("retrieval_latency")
    assert len(samples) == 1
    assert samples[0].source == "dify"


async def test_failed_retrieval_records_latency() -> None:
    """检索失败时照样记一条检索耗时。"""
    transport = FakeTransport(error=TransportTimeoutError("超时了"))
    retriever = DifyRetriever(
        "https://dify.example.com/v1",
        "ds-1",
        "dataset-key",
        transport=transport,
    )

    with pytest.raises(DifyTimeoutError):
        await retriever.retrieve(QUESTION)

    assert len(default_recorder().samples("retrieval_latency")) == 1


async def test_degraded_retrieval_records_latency() -> None:
    """鉴权不过降级时同样记一条检索耗时。"""
    transport = FakeTransport(
        response=TransportResponse(status_code=401, body="{}")
    )
    retriever = DifyRetriever(
        "https://dify.example.com/v1",
        "ds-1",
        "dataset-key",
        transport=transport,
    )

    outcome = await retriever.retrieve_outcome(QUESTION)

    assert outcome.degraded
    assert len(default_recorder().samples("retrieval_latency")) == 1
