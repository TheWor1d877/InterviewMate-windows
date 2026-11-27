"""检索结果结构。

字段与语义见《6-Dify检索与项目知识》结果整理一节：
一条检索片段由正文、文档名、分数与附加信息组成；
:class:`RetrievalOutcome` 表示一次检索的整体结果与降级状态。

错误码见《2-骨架与配置》错误码一节，本模块只存字符串，不自己造错误码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["RetrievalResult", "RetrievalOutcome"]


@dataclass(frozen=True)
class RetrievalResult:
    """一条检索片段。

    :param content: 片段正文，取自响应的 ``records[].segment.content``。
    :param document_name: 文档名，真实响应里取自 ``records[].segment.document.name``。
        代码按三级优先级取：``segment.document.name`` → ``segment.document_name``
        → ``record.document.name``，都取不到时留空串。
    :param score: 相关性分数，取自 ``records[].score``；服务没给或类型不对时为 None。
    :param metadata: 片段位置等附加信息，取自 ``records[].segment.position`` 等字段。
    """

    content: str
    document_name: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalOutcome:
    """一次检索的整体结果。

    :param results: 检索到的片段列表，检索失败时为空列表。
    :param ok: 检索是否正常完成。Dify 没检索到相关内容时也算正常完成。
    :param degraded: 是否处于降级状态。Dify 连不上、超时、鉴权不过或
        响应体字段变了时为 True，见《6-Dify检索与项目知识》故障降级一节。
    :param error_code: 失败时的错误码，取自《2-骨架与配置》错误码表，
        正常完成时为 None。
    :param error_detail: 失败时的详情说明，正常完成时为空串。
    """

    results: list[RetrievalResult]
    ok: bool
    degraded: bool
    error_code: str | None = None
    error_detail: str = ""

    @classmethod
    def success(cls, results: list[RetrievalResult]) -> "RetrievalOutcome":
        """检索正常完成时的结果。"""
        return cls(results=list(results), ok=True, degraded=False)

    @classmethod
    def failure(cls, error_code: str, error_detail: str = "") -> "RetrievalOutcome":
        """检索降级时的结果，``error_code`` 取自已有的错误码表。"""
        return cls(
            results=[],
            ok=False,
            degraded=True,
            error_code=error_code,
            error_detail=error_detail,
        )
