"""检索接口抽象。

接口形状见《6-Dify检索与项目知识》检索接口抽象一节：
问题处理模块与模型客户端只看 :class:`RetrievalProvider`，
不关心底层是 Dify 还是别的框架。当前实现是
:class:`app.retrieval.dify_retriever.DifyRetriever`。

检索耗时单独计时，指标名 ``retrieval_latency``，来源是后端名，
见《8-集成与验收》性能统计一节。真实实现与假实现都走
:func:`retrieval_timer`，这样 mock 模式下这个指标不会是空的。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.retrieval.retrieval_models import RetrievalResult
from app.utils.timing import Timer, default_recorder, timed

__all__ = ["RetrievalProvider", "RETRIEVAL_LATENCY_METRIC", "retrieval_timer"]

# 检索耗时的指标名，见《2-骨架与配置》性能指标一节
RETRIEVAL_LATENCY_METRIC: str = "retrieval_latency"


def retrieval_timer(source: str) -> Timer:
    """返回一个检索耗时计时器，退出时把耗时写进进程级默认记录器。

    计时范围是「发请求到拿到响应结束」，检索失败时同样落库，
    见《8-集成与验收》性能统计一节。

    :param source: 来源标记，后端名，取 ``dify`` 或 ``mock``。
    """
    return timed(
        RETRIEVAL_LATENCY_METRIC,
        source=source,
        recorder=default_recorder(),
    )


@runtime_checkable
class RetrievalProvider(Protocol):
    """检索提供者的接口。"""

    async def retrieve(self, query: str) -> list[RetrievalResult]:
        """按查询检索，返回命中的片段列表。

        失败时抛 :mod:`app.core.exceptions` 里已有的业务异常，
        异常自带错误码。调用方想拿降级结果时用实现类提供的
        ``retrieve_outcome``，由它把异常收敛成 ``RetrievalOutcome``。
        """
        raise NotImplementedError
