"""Dify 检索实现。

端点、鉴权、请求体与响应体的形态见《接口契约》：

- Dify 检索端点与鉴权一节：向 ``{base_url}/datasets/{dataset_id}/retrieve`` 发 POST，
  请求头带 ``Authorization: Bearer {api_key}`` 与 ``Content-Type: application/json``；
- Dify 检索请求体一节：请求体含 ``query`` 与 ``retrieval_model``，
  外部知识库把 ``retrieval_model`` 换成 ``external_retrieval_model``，
  ``query`` 最长 250 字符，超长时客户端截断；
  ``retrieval_model`` 里的 ``search_method``、``reranking_enable``、
  ``score_threshold_enabled``、``score_threshold``、``top_k`` 五项服务端都要求带上，
  缺一项就以 HTTP 400 拒收，且不继承知识库里已有的检索设置；
- Dify 检索响应体一节：响应含 ``records`` 数组，每项含 ``segment`` 与 ``score``；
  文档名在不同版本的响应里位置不一样，取法见 :func:`_extract_document_name`。

降级见《6-Dify检索与项目知识》故障降级一节：
连接失败、超时、鉴权不过、请求体被服务端拒绝、响应体字段变化
都映射到《2-骨架与配置》错误码表里已有的错误码，
不抛未捕获异常，检索失败不妨碍问题处理继续往下走。

检索耗时单独计时，指标名 ``retrieval_latency``，来源是后端名，
见《8-集成与验收》性能统计一节。失败时同样落库。

本模块不 import ``app.config``：连接参数由调用方从外面传进来。
传输层是接口，默认实现用 httpx 发真实请求，测试时注入假实现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from app.core.exceptions import (
    DifyAuthError,
    DifyQueryTooLongError,
    DifyRequestError,
    DifyResponseError,
    DifyTimeoutError,
    InterviewMateError,
)
from app.retrieval.base import RetrievalProvider, retrieval_timer
from app.retrieval.retrieval_models import RetrievalOutcome, RetrievalResult
from app.utils.logger import log_event

__all__ = [
    "TransportError",
    "TransportTimeoutError",
    "TransportResponse",
    "RetrievalTransport",
    "HttpxRetrievalTransport",
    "DifyRetriever",
    "DEFAULT_TOP_K",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_QUERY_LENGTH",
    "DEFAULT_SEARCH_METHOD",
    "SEARCH_METHODS",
]

# 默认检索参数，取值见《6-Dify检索与项目知识》检索参数一节
DEFAULT_TOP_K: int = 4
DEFAULT_TIMEOUT: float = 8.0
# query 长度上限，见《接口契约》Dify 检索请求体一节
DEFAULT_MAX_QUERY_LENGTH: int = 250

# 检索方式，Dify 的检索端点必填该项，见《接口契约》Dify 检索请求体一节
DEFAULT_SEARCH_METHOD: str = "hybrid_search"
# Dify 支持的检索方式
SEARCH_METHODS: frozenset[str] = frozenset(
    {"keyword_search", "semantic_search", "full_text_search", "hybrid_search"}
)

# 认证失败的状态码，映射到 E_DIFY_AUTH
_AUTH_STATUS = frozenset({401, 403})

# 服务端拒绝请求体时的特征，映射到 E_DIFY_REQUEST
_REQUEST_REJECTED_MARKERS = ("validation error", "field required")
# 响应体片段的最大长度，进错误详情时截断
_SNIPPET_LIMIT = 300


class TransportError(Exception):
    """传输层失败：连接不上、域名解析不了等。"""


class TransportTimeoutError(TransportError):
    """传输层超时。"""


@dataclass(frozen=True)
class TransportResponse:
    """传输层返回的原始响应。

    :param status_code: HTTP 状态码。
    :param body: 响应体原文，JSON 解析由检索实现负责。
    """

    status_code: int
    body: str


@runtime_checkable
class RetrievalTransport(Protocol):
    """检索请求的传输层。"""

    async def post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> TransportResponse:
        """发一次 POST 请求，返回状态码与响应体原文。"""
        raise NotImplementedError


class HttpxRetrievalTransport:
    """默认传输层，用 httpx 发真实请求。

    httpx 的超时异常在这里就转成传输层自己的异常，
    免得检索层直接依赖 httpx 的异常类型。

    :param timeout: 请求超时秒数。
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> TransportResponse:
        """发一次 POST 请求，返回状态码与响应体原文。"""
        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise TransportTimeoutError(f"Dify 检索请求超时：{exc}") from exc
        except httpx.HTTPError as exc:
            raise TransportError(f"Dify 检索请求失败：{exc}") from exc
        return TransportResponse(status_code=response.status_code, body=response.text)

    async def aclose(self) -> None:
        """关闭 HTTP 客户端，进程收尾时调用。"""
        await self._client.aclose()


class DifyRetriever(RetrievalProvider):
    """Dify 知识库检索。

    :param base_url: Dify 基址，含 ``/v1`` 前缀，见《接口契约》Dify 检索端点与鉴权一节。
    :param dataset_id: 知识库 ID，请求打到 ``{base_url}/datasets/{dataset_id}/retrieve``。
    :param api_key: 知识库 API Key，只进 Authorization 头，不进日志。
    :param top_k: 返回片段数量，对应请求体的 ``retrieval_model.top_k``。
    :param timeout: 请求超时秒数，默认传输层用这个值。
    :param max_query_length: 查询长度上限，超长时客户端截断，
        见《6-Dify检索与项目知识》检索参数一节。
    :param truncate_query: 超长时是否截断。False 表示超长直接报
        ``E_DIFY_QUERY_TOO_LONG``，不发请求。
    :param search_method: 检索方式，对应请求体的 ``retrieval_model.search_method``。
        Dify 的检索端点必填该项，缺了会以 HTTP 400 拒收。
    :param score_threshold_enabled: 是否启用分数阈值，对应请求体的
        ``retrieval_model.score_threshold_enabled``。
    :param score_threshold: 分数阈值，对应请求体的 ``retrieval_model.score_threshold``。
    :param reranking_enable: 是否启用重排，对应请求体的
        ``retrieval_model.reranking_enable``。
    :param external: 外部知识库时为 True，请求体改用 ``external_retrieval_model``，
        见《接口契约》Dify 检索请求体一节。
    :param transport: 传输层，测试时注入假实现；运行时传 None 走真实 HTTP。
    """

    # 后端名，对应配置里 ``providers.retrieval`` 的 dify 取值，
    # 也是检索耗时指标的来源标记
    name: str = "dify"

    def __init__(
        self,
        base_url: str,
        dataset_id: str,
        api_key: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        timeout: float = DEFAULT_TIMEOUT,
        max_query_length: int = DEFAULT_MAX_QUERY_LENGTH,
        truncate_query: bool = True,
        search_method: str = DEFAULT_SEARCH_METHOD,
        score_threshold_enabled: bool = False,
        score_threshold: float = 0.0,
        reranking_enable: bool = False,
        external: bool = False,
        transport: RetrievalTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dataset_id = dataset_id
        self._api_key = api_key
        self._top_k = top_k
        self._timeout = timeout
        self._max_query_length = max_query_length
        self._truncate_query = truncate_query
        self._search_method = search_method
        self._score_threshold_enabled = score_threshold_enabled
        self._score_threshold = score_threshold
        self._reranking_enable = reranking_enable
        # 外部知识库换字段名，字段名相同，见《接口契约》Dify 检索请求体一节
        self._model_key = "external_retrieval_model" if external else "retrieval_model"
        self._owns_transport = transport is None
        self._transport = transport if transport is not None else HttpxRetrievalTransport(timeout)

    @property
    def base_url(self) -> str:
        """Dify 基址。"""
        return self._base_url

    @property
    def dataset_id(self) -> str:
        """知识库 ID。"""
        return self._dataset_id

    @property
    def top_k(self) -> int:
        """返回片段数量。"""
        return self._top_k

    @property
    def search_method(self) -> str:
        """检索方式。"""
        return self._search_method

    async def retrieve(self, query: str) -> list[RetrievalResult]:
        """按查询检索，返回命中的片段列表。

        失败时抛 :mod:`app.core.exceptions` 里已有的 Dify 异常，
        异常自带错误码。

        这次检索的耗时记进 ``retrieval_latency``，来源是后端名；
        计时范围从发请求到响应处理结束，失败时同样落库，
        见《8-集成与验收》性能统计一节。

        :param query: 检索查询，超长时按 ``max_query_length`` 截断。
        :raises DifyAuthError: 鉴权不过，对应 ``E_DIFY_AUTH``。
        :raises DifyTimeoutError: 请求超时，对应 ``E_DIFY_TIMEOUT``。
        :raises DifyResponseError: 连接失败或响应体不对，对应 ``E_DIFY_RESPONSE``。
        :raises DifyQueryTooLongError: 查询超长且不截断，对应 ``E_DIFY_QUERY_TOO_LONG``。
        """
        with retrieval_timer(self.name):
            payload = self._build_payload(query)
            response = await self._post(payload)
            results = self._parse_records(response)
        log_event(
            "dify.retrieve.ok",
            "Dify 检索完成",
            count=len(results),
        )
        return results

    async def retrieve_outcome(self, query: str) -> RetrievalOutcome:
        """按查询检索，返回带降级状态的完整结果。

        这一层不抛异常：检索失败时返回 ``degraded=True`` 的结果，
        错误码取自《2-骨架与配置》错误码表，调用方据此更新状态栏，
        见《6-Dify检索与项目知识》故障降级一节。

        :param query: 检索查询，超长时按 ``max_query_length`` 截断。
        :return: 检索结果，失败时 ``results`` 为空列表。
        """
        try:
            results = await self.retrieve(query)
        except InterviewMateError as exc:
            log_event(
                "dify.retrieve.degraded",
                str(exc),
                level="WARNING",
                error_code=exc.code,
            )
            return RetrievalOutcome.failure(exc.code, str(exc))
        except Exception as exc:  # 兜底：检索出问题不让问题处理崩掉
            log_event(
                "dify.retrieve.unexpected",
                f"Dify 检索出现未预期异常：{exc}",
                level="ERROR",
            )
            return RetrievalOutcome.failure("E_DIFY_RESPONSE", str(exc))
        return RetrievalOutcome.success(results)

    async def aclose(self) -> None:
        """关闭底层 HTTP 客户端，进程收尾时调用。

        传输层是外面注入的时不关，避免关掉别人还在用的连接池。
        """
        if not self._owns_transport:
            return
        closer = getattr(self._transport, "aclose", None)
        if closer is None:
            return
        await closer()

    def _build_payload(self, query: str) -> dict[str, Any]:
        """组装请求体，字段见《接口契约》Dify 检索请求体一节。

        ``retrieval_model`` 里这五项 Dify 都要求带上：
        ``search_method``、``reranking_enable``、``score_threshold_enabled``、
        ``score_threshold``、``top_k``。缺任何一项服务端都以 HTTP 400 拒收，
        且不继承知识库里已有的检索设置，所以每次请求都按这里的取值发。
        """
        text = self._prepare_query(query)
        model: dict[str, Any] = {
            "search_method": self._search_method,
            "reranking_enable": self._reranking_enable,
            "score_threshold_enabled": self._score_threshold_enabled,
            "score_threshold": self._score_threshold,
            "top_k": self._top_k,
        }
        return {"query": text, self._model_key: model}

    def _prepare_query(self, query: str) -> str:
        """查询超长时截断，发出去的请求一定在限制之内。

        见《6-Dify检索与项目知识》检索参数一节与《接口契约》Dify 检索请求体一节。
        """
        text = query if isinstance(query, str) else str(query or "")
        limit = self._max_query_length
        if limit <= 0 or len(text) <= limit:
            return text
        if not self._truncate_query:
            raise DifyQueryTooLongError(
                f"检索查询长度 {len(text)} 超过上限 {limit}，未启用截断"
            )
        log_event(
            "dify.query.truncated",
            "检索查询超长，已在客户端截断",
            level="WARNING",
            original_length=len(text),
            limit=limit,
        )
        return text[:limit]

    async def _post(self, payload: dict[str, Any]) -> TransportResponse:
        """发请求并把传输层异常映射成 Dify 异常。"""
        url = self._build_url()
        try:
            response = await self._transport.post(url, payload, self._headers())
        except TransportTimeoutError as exc:
            raise DifyTimeoutError(str(exc)) from exc
        except TransportError as exc:
            raise DifyResponseError(str(exc)) from exc
        self._raise_for_status(response)
        return response

    def _build_url(self) -> str:
        """请求地址，见《接口契约》Dify 检索端点与鉴权一节。"""
        return f"{self._base_url}/datasets/{self._dataset_id}/retrieve"

    def _headers(self) -> dict[str, str]:
        """请求头。API Key 只出现在这里，不进日志字段。"""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _raise_for_status(self, response: TransportResponse) -> None:
        """按状态码映射异常，见《2-骨架与配置》错误码一节。

        请求体被服务端校验拒绝（HTTP 400）映射到 ``E_DIFY_REQUEST``，
        与响应格式异常（``E_DIFY_RESPONSE``）区分开，详情里带服务端返回的原文片段。
        """
        status = response.status_code
        if status in _AUTH_STATUS:
            raise DifyAuthError(f"Dify 检索认证失败（HTTP {status}），检查 API Key")
        if status == 400 and _looks_like_query_too_long(response.body):
            raise DifyQueryTooLongError("Dify 判定检索查询超长")
        if status == 400 and _looks_like_request_rejected(response.body):
            raise DifyRequestError(
                f"Dify 拒绝检索请求体（HTTP 400）：{_body_snippet(response.body)}"
            )
        if status >= 400:
            raise DifyResponseError(f"Dify 检索返回 HTTP {status}")

    def _parse_records(self, response: TransportResponse) -> list[RetrievalResult]:
        """把响应体解析成片段列表，字段见《接口契约》Dify 检索响应体一节。

        响应体字段变了或类型不对时抛 :class:`DifyResponseError`，
        由 :meth:`retrieve_outcome` 收敛成降级结果。
        """
        payload = self._decode_body(response.body)
        records = payload.get("records")
        if not isinstance(records, list):
            raise DifyResponseError("Dify 检索响应里没有 records 数组")
        results: list[RetrievalResult] = []
        for record in records:
            parsed = _parse_record(record)
            if parsed is not None:
                results.append(parsed)
        if records and not results:
            raise DifyResponseError("Dify 检索响应的片段字段结构无法解析")
        return results

    @staticmethod
    def _decode_body(body: str) -> dict[str, Any]:
        """把响应体原文解成 JSON 对象。"""
        try:
            payload = json.loads(body)
        except (TypeError, ValueError) as exc:
            raise DifyResponseError("Dify 检索响应不是合法 JSON") from exc
        if not isinstance(payload, dict):
            raise DifyResponseError("Dify 检索响应不是 JSON 对象")
        return payload


def _parse_record(record: Any) -> RetrievalResult | None:
    """解析一条记录，结构对不上时返回 None 让调用方跳过。"""
    if not isinstance(record, dict):
        return None
    segment = record.get("segment")
    if not isinstance(segment, dict):
        return None
    content = segment.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    metadata: dict[str, Any] = {}
    position = segment.get("position")
    if position is not None:
        metadata["position"] = position
    segment_id = segment.get("id")
    if segment_id is not None:
        metadata["segment_id"] = segment_id
    document_id = segment.get("document_id")
    if document_id is not None:
        metadata["document_id"] = document_id
    return RetrievalResult(
        content=content,
        document_name=_extract_document_name(record, segment),
        score=_as_float(record.get("score")),
        metadata=metadata,
    )


def _extract_document_name(record: dict[str, Any], segment: dict[str, Any]) -> str:
    """取文档名，三处结构按优先级试。

    1. ``segment["document"]["name"]``：document 对象嵌在 segment 里，
       Dify 1.20.0 的真实响应是这个形态，文档名在 document 对象的 name 上；
    2. ``segment["document_name"]``：文档名直接挂在 segment 上的旧结构；
    3. ``record["document"]["name"]``：document 对象在 record 顶层的结构。

    某一处取到的值不是字符串就往下试一处；三处都取不到时落空字符串，
    文档名取不到不影响这条片段本身，只是不带来源名。
    """
    candidates = [
        _nested_name(segment.get("document")),
        segment.get("document_name"),
        _nested_name(record.get("document")),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def _nested_name(document: Any) -> Any:
    """从 document 对象里取 name，取不到时返回 None。"""
    if not isinstance(document, dict):
        return None
    return document.get("name")


def _as_float(value: Any) -> float | None:
    """把分数转成 float，不是数值就返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _looks_like_query_too_long(body: str) -> bool:
    """响应体里有没有查询超长的提示。

    只认 query 与长度提示同时出现的句子，避免把别的 400 误判成超长。
    """
    text = body.lower()
    if "query" not in text:
        return False
    return any(marker in text for marker in ("too long", "exceed", "length"))


def _looks_like_request_rejected(body: str) -> bool:
    """响应体里有没有请求体校验错的提示。

    Dify 校验请求体失败时回的是 pydantic 的校验错，
    形如 ``3 validation errors for HitTestingPayload`` 与 ``Field required``。
    """
    text = body.lower()
    return any(marker in text for marker in _REQUEST_REJECTED_MARKERS)


def _body_snippet(body: str) -> str:
    """取响应体片段进错误详情，压成一行并截断。"""
    text = " ".join(body.split())
    if len(text) > _SNIPPET_LIMIT:
        return text[:_SNIPPET_LIMIT] + "…"
    return text
