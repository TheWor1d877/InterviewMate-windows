"""检索层的测试。

用假的传输层注入进来，不发真实网络请求。
覆盖《接口契约》Dify 检索端点与鉴权、Dify 检索请求体、Dify 检索响应体三节，
以及《6-Dify检索与项目知识》结果整理一节的片段整理规则与故障降级语义。
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

import pytest

from app.core.exceptions import (
    DifyAuthError,
    DifyQueryTooLongError,
    DifyRequestError,
    DifyResponseError,
    DifyTimeoutError,
)
from app.retrieval.base import RetrievalProvider
from app.retrieval.context_builder import CONTEXT_SEPARATOR, ContextBuilder
from app.retrieval.dify_retriever import (
    DifyRetriever,
    TransportError,
    TransportResponse,
    TransportTimeoutError,
)
from app.retrieval.retrieval_models import RetrievalOutcome, RetrievalResult
from app.utils import logger as logger_module
from app.utils.logger import Redactor

# 假服务基址，与《接口契约》Dify 检索端点与鉴权一节一样带 /v1 前缀
_BASE_URL = "https://dify.example.com/v1"
_DATASET_ID = "ds-42"
_API_KEY = "dataset-test-key"

# 《接口契约》Dify 检索请求体一节：query 最长 250 字符
_MAX_QUERY = 250


class _FakeTransport:
    """假的传输层，记录请求并返回预置响应。"""

    def __init__(
        self,
        response: TransportResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> TransportResponse:
        self.calls.append({"url": url, "payload": payload, "headers": headers})
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _ok_body(records: list[dict[str, Any]]) -> str:
    """按《接口契约》Dify 检索响应体一节拼一个响应体。"""
    return json.dumps({"records": records})


def _record(
    content: str = "片段正文",
    document_name: str = "架构说明.md",
    score: float | None = 0.8,
    position: int | None = 1,
) -> dict[str, Any]:
    """一条 records 记录。"""
    segment: dict[str, Any] = {"content": content}
    if document_name is not None:
        segment["document_name"] = document_name
    if position is not None:
        segment["position"] = position
    entry: dict[str, Any] = {"segment": segment}
    if score is not None:
        entry["score"] = score
    return entry


def _make_retriever(transport: _FakeTransport, **kwargs: Any) -> DifyRetriever:
    """造一个用假传输层的检索器。"""
    kwargs.setdefault("top_k", 4)
    return DifyRetriever(
        _BASE_URL,
        _DATASET_ID,
        _API_KEY,
        transport=transport,
        **kwargs,
    )


class _ListHandler(logging.Handler):
    """把日志记录收进列表，方便断言。"""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def log_lines(monkeypatch: pytest.MonkeyPatch) -> _ListHandler:
    """收集 interviewmate 记录器的日志文本。"""
    monkeypatch.setattr(logger_module, "_pipeline_redactor", Redactor(enabled=False))
    handler = _ListHandler()
    target = logging.getLogger("interviewmate")
    old_level = target.level
    target.setLevel(logging.DEBUG)
    target.addHandler(handler)
    yield handler
    target.removeHandler(handler)
    target.setLevel(old_level)


# --------------------------------------------------------------------------
# 请求形态
# --------------------------------------------------------------------------


async def test_request_targets_dataset_retrieve_path() -> None:
    """请求打到 {base_url}/datasets/{id}/retrieve。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport)

    await retriever.retrieve("项目架构")

    path = urlparse(transport.calls[0]["url"]).path
    assert path == "/v1/datasets/ds-42/retrieve"
    assert transport.calls[0]["url"].startswith(_BASE_URL)


async def test_payload_uses_query_and_retrieval_model_fields() -> None:
    """请求体里 query 与 retrieval_model 的字段名正确。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport, top_k=6)

    await retriever.retrieve("项目中如何实现用户认证")

    payload = transport.calls[0]["payload"]
    assert payload["query"] == "项目中如何实现用户认证"
    assert payload["retrieval_model"]["top_k"] == 6
    assert "external_retrieval_model" not in payload


async def test_external_dataset_uses_external_retrieval_model() -> None:
    """外部知识库用 external_retrieval_model，字段名相同。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport, external=True)

    await retriever.retrieve("外部知识库")

    payload = transport.calls[0]["payload"]
    assert payload["external_retrieval_model"]["top_k"] == 4
    assert "retrieval_model" not in payload


async def test_auth_header_carries_api_key() -> None:
    """请求头带 Bearer 令牌与 JSON 内容类型。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport)

    await retriever.retrieve("项目架构")

    headers = transport.calls[0]["headers"]
    assert headers["Authorization"] == f"Bearer {_API_KEY}"
    assert headers["Content-Type"] == "application/json"


async def test_long_query_truncated_within_limit(log_lines: _ListHandler) -> None:
    """查询超长时客户端截断，发出去的请求体在限制之内。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport)
    query = "项" * 500

    await retriever.retrieve(query)

    sent = transport.calls[0]["payload"]["query"]
    assert len(sent) == _MAX_QUERY
    assert len(sent) <= _MAX_QUERY
    assert any("已在客户端截断" in line for line in log_lines.lines)


async def test_query_too_long_without_truncation_raises() -> None:
    """不截断时查询超长映射到 E_DIFY_QUERY_TOO_LONG，不发请求。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport, truncate_query=False)

    with pytest.raises(DifyQueryTooLongError) as excinfo:
        await retriever.retrieve("项" * 500)

    assert excinfo.value.code == "E_DIFY_QUERY_TOO_LONG"
    assert transport.calls == []


# --------------------------------------------------------------------------
# 响应解析
# --------------------------------------------------------------------------


async def test_response_parsed_into_results() -> None:
    """响应体正常时解析出片段内容、文档名与分数。"""
    body = _ok_body([_record(content="检索层实现", document_name="检索.md", score=0.72, position=3)])
    transport = _FakeTransport(TransportResponse(200, body))
    retriever = _make_retriever(transport)

    results = await retriever.retrieve("检索层怎么实现")

    assert len(results) == 1
    first = results[0]
    assert first.content == "检索层实现"
    assert first.document_name == "检索.md"
    assert first.score == pytest.approx(0.72)
    assert first.metadata["position"] == 3


async def test_record_without_score_gives_none() -> None:
    """服务没给分数时分数是 None。"""
    body = _ok_body([_record(score=None)])
    transport = _FakeTransport(TransportResponse(200, body))
    retriever = _make_retriever(transport)

    results = await retriever.retrieve("项目架构")

    assert results[0].score is None


async def test_empty_records_is_ok_and_not_degraded() -> None:
    """Dify 没检索到相关内容时不算降级。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport)

    outcome = await retriever.retrieve_outcome("项目架构")

    assert outcome.ok is True
    assert outcome.degraded is False
    assert outcome.results == []
    assert outcome.error_code is None


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"data": []}),
        json.dumps({"records": "不是数组"}),
        json.dumps({"records": [{"segment": "不是对象"}]}),
        json.dumps({"records": [{"segment": {"content": 123}}]}),
        json.dumps({"records": [{"score": 0.5}]}),
        "不是 JSON",
    ],
)
async def test_malformed_response_degrades_without_raising(body: str) -> None:
    """响应体字段缺失或类型不对时不抛未捕获异常，按降级处理。"""
    transport = _FakeTransport(TransportResponse(200, body))
    retriever = _make_retriever(transport)

    outcome = await retriever.retrieve_outcome("项目架构")

    assert outcome.degraded is True
    assert outcome.ok is False
    assert outcome.error_code == "E_DIFY_RESPONSE"
    assert outcome.results == []
    assert outcome.error_detail


# --------------------------------------------------------------------------
# 错误映射
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failure_maps_to_e_dify_auth(status: int) -> None:
    """鉴权失败映射到 E_DIFY_AUTH。"""
    transport = _FakeTransport(TransportResponse(status, json.dumps({"code": "unauthorized"})))
    retriever = _make_retriever(transport)

    with pytest.raises(DifyAuthError) as excinfo:
        await retriever.retrieve("项目架构")
    assert excinfo.value.code == "E_DIFY_AUTH"

    outcome = await retriever.retrieve_outcome("项目架构")
    assert outcome.degraded is True
    assert outcome.error_code == "E_DIFY_AUTH"


async def test_timeout_maps_to_e_dify_timeout() -> None:
    """超时映射到 E_DIFY_TIMEOUT。"""
    transport = _FakeTransport(error=TransportTimeoutError("读超时"))
    retriever = _make_retriever(transport)

    with pytest.raises(DifyTimeoutError) as excinfo:
        await retriever.retrieve("项目架构")
    assert excinfo.value.code == "E_DIFY_TIMEOUT"

    outcome = await retriever.retrieve_outcome("项目架构")
    assert outcome.degraded is True
    assert outcome.error_code == "E_DIFY_TIMEOUT"


async def test_connection_failure_maps_to_e_dify_response() -> None:
    """连接失败映射到 E_DIFY_RESPONSE。"""
    transport = _FakeTransport(error=TransportError("连接被拒绝"))
    retriever = _make_retriever(transport)

    outcome = await retriever.retrieve_outcome("项目架构")

    assert outcome.degraded is True
    assert outcome.error_code == "E_DIFY_RESPONSE"


async def test_server_error_maps_to_e_dify_response() -> None:
    """服务端返回 5xx 映射到 E_DIFY_RESPONSE。"""
    transport = _FakeTransport(TransportResponse(500, "内部错误"))
    retriever = _make_retriever(transport)

    outcome = await retriever.retrieve_outcome("项目架构")

    assert outcome.degraded is True
    assert outcome.error_code == "E_DIFY_RESPONSE"


async def test_api_key_not_in_logs(log_lines: _ListHandler) -> None:
    """API Key 不进日志。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport)

    await retriever.retrieve_outcome("项目架构")

    assert log_lines.lines
    assert all(_API_KEY not in line for line in log_lines.lines)


# --------------------------------------------------------------------------
# 接口抽象
# --------------------------------------------------------------------------


async def test_dify_retriever_implements_retrieval_provider() -> None:
    """DifyRetriever 实现 RetrievalProvider 接口。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport)

    assert isinstance(retriever, RetrievalProvider)


# --------------------------------------------------------------------------
# 上下文拼装
# --------------------------------------------------------------------------


def _result(content: str, document_name: str, score: float | None) -> RetrievalResult:
    return RetrievalResult(content=content, document_name=document_name, score=score)


def test_context_sorted_by_score_descending() -> None:
    """片段按分数从高到低排列。"""
    builder = ContextBuilder()
    results = [
        _result("低分片段", "低.md", 0.2),
        _result("高分片段", "高.md", 0.9),
        _result("中分片段", "中.md", 0.5),
    ]

    text = builder.build(results)

    assert text.index("高分片段") < text.index("中分片段") < text.index("低分片段")


def test_context_without_score_keeps_order_at_the_end() -> None:
    """分数为 None 的片段排在有分数的后面，同组内保持原有顺序。"""
    builder = ContextBuilder()
    results = [
        _result("无分甲", "甲.md", None),
        _result("有分", "有.md", 0.4),
        _result("无分乙", "乙.md", None),
    ]

    text = builder.build(results)

    assert text.index("有分") < text.index("无分甲") < text.index("无分乙")


def test_context_truncated_to_max_context_chars() -> None:
    """上下文总量截断到 max_context_chars。"""
    builder = ContextBuilder(max_context_chars=120)
    results = [
        _result("甲" * 200, "甲.md", 0.9),
        _result("乙" * 200, "乙.md", 0.8),
    ]

    text = builder.build(results)

    assert len(text) <= 120
    assert "甲" in text


def test_context_keeps_document_name() -> None:
    """每个片段带上文档名，模型能引用来源。"""
    builder = ContextBuilder()
    results = [_result("正文内容", "项目架构说明.md", 0.9)]

    text = builder.build(results)

    assert "项目架构说明.md" in text
    assert "正文内容" in text


def test_context_separates_fragments() -> None:
    """片段之间有明确分隔，避免相邻片段被当成一句。"""
    builder = ContextBuilder()
    results = [
        _result("第一段结尾", "一.md", 0.9),
        _result("第二段开头", "二.md", 0.8),
    ]

    text = builder.build(results)

    assert CONTEXT_SEPARATOR in text
    assert text.index("第一段结尾") < text.index(CONTEXT_SEPARATOR) < text.index("第二段开头")


def test_context_empty_results_is_empty_string() -> None:
    """没有片段时上下文是空串。"""
    assert ContextBuilder().build([]) == ""


def test_outcome_helpers() -> None:
    """RetrievalOutcome 的成功与失败构造。"""
    ok = RetrievalOutcome.success([_result("正文", "甲.md", 0.5)])
    assert ok.ok is True
    assert ok.degraded is False
    assert ok.error_code is None

    failed = RetrievalOutcome.failure("E_DIFY_TIMEOUT", "读超时")
    assert failed.ok is False
    assert failed.degraded is True
    assert failed.error_code == "E_DIFY_TIMEOUT"
    assert failed.error_detail == "读超时"
    assert failed.results == []


# --------------------------------------------------------------------------
# 文档名取法
# --------------------------------------------------------------------------

# 真机 Dify 1.20.0 的响应片段：document 对象嵌在 segment 里，文档名是它的 name
_REAL_RECORD = {
    "segment": {
        "id": "0acbffb8-...",
        "position": 13,
        "document_id": "8fabceba-db8c-4ff0-a6e6-a2532a1ce950",
        "content": "XDG Portal 截图目标\n\n截图目标由配置项 screenshot.target 决定。",
        "word_count": 476,
        "status": "completed",
        "document": {
            "id": "8fabceba-db8c-4ff0-a6e6-a2532a1ce950",
            "data_source_type": "upload_file",
            "name": "接口契约.md",
            "doc_type": None,
            "doc_metadata": None,
        },
    },
    "score": 0.6625407,
    "child_chunks": [],
    "files": [],
    "summary": None,
}


async def test_document_name_from_nested_segment_document() -> None:
    """真实结构：document 嵌在 segment 里，文档名取 document.name。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([_REAL_RECORD])))
    retriever = _make_retriever(transport)

    results = await retriever.retrieve("截图目标怎么配置")

    assert len(results) == 1
    assert results[0].document_name == "接口契约.md"
    assert results[0].content.startswith("XDG Portal 截图目标")
    assert results[0].metadata["document_id"] == "8fabceba-db8c-4ff0-a6e6-a2532a1ce950"


async def test_document_name_from_segment_document_name() -> None:
    """旧结构：文档名直接挂在 segment 上，仍然能取到。"""
    record = {
        "segment": {
            "content": "片段正文",
            "document_name": "旧结构.md",
            "document_id": "doc-1",
        },
        "score": 0.5,
    }
    transport = _FakeTransport(TransportResponse(200, _ok_body([record])))
    retriever = _make_retriever(transport)

    results = await retriever.retrieve("项目架构")

    assert results[0].document_name == "旧结构.md"


async def test_document_name_from_top_level_document() -> None:
    """document 在 record 顶层时，文档名仍然能取到。"""
    record = {
        "segment": {"content": "片段正文", "document_id": "doc-2"},
        "document": {"id": "doc-2", "name": "顶层结构.md"},
        "score": 0.5,
    }
    transport = _FakeTransport(TransportResponse(200, _ok_body([record])))
    retriever = _make_retriever(transport)

    results = await retriever.retrieve("项目架构")

    assert results[0].document_name == "顶层结构.md"


async def test_document_name_missing_keeps_record() -> None:
    """三处都取不到文档名时落空串，但这条片段仍然保留。"""
    record = {
        "segment": {"content": "片段正文", "document_id": "doc-3"},
        "score": 0.5,
    }
    transport = _FakeTransport(TransportResponse(200, _ok_body([record])))
    retriever = _make_retriever(transport)

    results = await retriever.retrieve("项目架构")

    assert len(results) == 1
    assert results[0].document_name == ""
    assert results[0].content == "片段正文"


async def test_document_name_not_a_string_falls_back() -> None:
    """优先位置的值不是字符串时往下试，都不是字符串才落空串。"""
    record = {
        "segment": {
            "content": "片段正文",
            "document": {"name": 123},
            "document_name": "回退.md",
        },
        "score": 0.5,
    }
    transport = _FakeTransport(TransportResponse(200, _ok_body([record])))
    retriever = _make_retriever(transport)

    results = await retriever.retrieve("项目架构")

    assert results[0].document_name == "回退.md"


# --------------------------------------------------------------------------
# 检索设置字段
# --------------------------------------------------------------------------

# Dify 检索端点必填的检索设置字段，缺一个就以 HTTP 400 拒收
_REQUIRED_RETRIEVAL_FIELDS = (
    "search_method",
    "reranking_enable",
    "score_threshold_enabled",
    "score_threshold",
    "top_k",
)


async def test_retrieval_model_carries_all_required_fields() -> None:
    """检索设置五个字段都发出去，服务端才不拒收。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport)

    await retriever.retrieve("项目架构")

    model = transport.calls[0]["payload"]["retrieval_model"]
    for field in _REQUIRED_RETRIEVAL_FIELDS:
        assert field in model
    assert model["top_k"] == 4


async def test_search_method_defaults_to_hybrid_search() -> None:
    """search_method 默认 hybrid_search。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport)

    await retriever.retrieve("项目架构")

    assert transport.calls[0]["payload"]["retrieval_model"]["search_method"] == "hybrid_search"


async def test_search_method_can_be_overridden() -> None:
    """search_method 能被构造参数覆盖。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport, search_method="semantic_search")

    await retriever.retrieve("项目架构")

    model = transport.calls[0]["payload"]["retrieval_model"]
    assert model["search_method"] == "semantic_search"


async def test_retrieval_settings_can_be_overridden() -> None:
    """阈值与重排开关能覆盖，请求体里的取值以配置为准。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(
        transport,
        score_threshold_enabled=True,
        score_threshold=0.5,
        reranking_enable=True,
        top_k=8,
    )

    await retriever.retrieve("项目架构")

    model = transport.calls[0]["payload"]["retrieval_model"]
    assert model["score_threshold_enabled"] is True
    assert model["score_threshold"] == pytest.approx(0.5)
    assert model["reranking_enable"] is True
    assert model["top_k"] == 8


async def test_external_dataset_carries_all_required_fields() -> None:
    """外部知识库换字段名，字段名相同，五项都在。"""
    transport = _FakeTransport(TransportResponse(200, _ok_body([])))
    retriever = _make_retriever(transport, external=True)

    await retriever.retrieve("项目架构")

    model = transport.calls[0]["payload"]["external_retrieval_model"]
    for field in _REQUIRED_RETRIEVAL_FIELDS:
        assert field in model


# --------------------------------------------------------------------------
# 请求被拒的错误码
# --------------------------------------------------------------------------

# 服务端回的请求体校验错，形态见 pydantic 的校验错
_VALIDATION_ERROR_BODY = (
    "3 validation errors for HitTestingPayload\n"
    "retrieval_model.search_method\n Field required\n"
    "retrieval_model.reranking_enable\n Field required\n"
    "retrieval_model.score_threshold_enabled\n Field required"
)


async def test_request_rejected_maps_to_e_dify_request() -> None:
    """请求体被服务端校验拒绝时映射到 E_DIFY_REQUEST，不是 E_DIFY_RESPONSE。"""
    transport = _FakeTransport(TransportResponse(400, _VALIDATION_ERROR_BODY))
    retriever = _make_retriever(transport)

    with pytest.raises(DifyRequestError) as excinfo:
        await retriever.retrieve("项目架构")

    assert excinfo.value.code == "E_DIFY_REQUEST"
    assert excinfo.value.code != "E_DIFY_RESPONSE"
    assert "validation error" in excinfo.value.message
    assert "Field required" in excinfo.value.message


async def test_request_rejected_detail_keeps_server_snippet() -> None:
    """错误详情里带服务端返回的原文片段。"""
    transport = _FakeTransport(TransportResponse(400, _VALIDATION_ERROR_BODY))
    retriever = _make_retriever(transport)

    outcome = await retriever.retrieve_outcome("项目架构")

    assert outcome.degraded is True
    assert outcome.error_code == "E_DIFY_REQUEST"
    assert "validation errors for HitTestingPayload" in outcome.error_detail


async def test_request_rejected_detail_is_truncated() -> None:
    """响应体很长时详情只留片段，不整个塞进去。"""
    body = "validation error for HitTestingPayload " + ("长" * 5000)
    transport = _FakeTransport(TransportResponse(400, body))
    retriever = _make_retriever(transport)

    outcome = await retriever.retrieve_outcome("项目架构")

    assert outcome.error_code == "E_DIFY_REQUEST"
    assert "长" * 5000 not in outcome.error_detail
    assert len(outcome.error_detail) < 500


async def test_other_bad_request_still_maps_to_e_dify_response() -> None:
    """不是请求体校验错的 400 仍按响应异常处理。"""
    transport = _FakeTransport(TransportResponse(400, "内部错误"))
    retriever = _make_retriever(transport)

    outcome = await retriever.retrieve_outcome("项目架构")

    assert outcome.error_code == "E_DIFY_RESPONSE"


async def test_auth_timeout_and_response_mapping_unchanged() -> None:
    """鉴权、超时、响应结构不对这三条映射不变。"""
    auth = _FakeTransport(TransportResponse(401, "unauthorized"))
    assert (await _make_retriever(auth).retrieve_outcome("项目架构")).error_code == "E_DIFY_AUTH"

    timeout = _FakeTransport(error=TransportTimeoutError("读超时"))
    assert (await _make_retriever(timeout).retrieve_outcome("项目架构")).error_code == "E_DIFY_TIMEOUT"

    broken = _FakeTransport(TransportResponse(200, json.dumps({"records": "不是数组"})))
    assert (await _make_retriever(broken).retrieve_outcome("项目架构")).error_code == "E_DIFY_RESPONSE"


# --------------------------------------------------------------------------
# 文档名说明与实现一致
# --------------------------------------------------------------------------


def test_document_name_docstring_matches_implementation() -> None:
    """document_name 的说明与实现一致：真实来源是 document.name。"""
    doc = RetrievalResult.__doc__ or ""

    assert "document.name" in doc
    assert "segment.document_name" in doc
    assert "record.document.name" in doc

