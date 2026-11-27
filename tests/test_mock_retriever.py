"""假检索后端的测试。

覆盖《2-骨架与配置》配置结构一节的 ``providers.retrieval=mock`` 行为，
以及《6-Dify检索与项目知识》检索接口抽象一节要求的接口形状。

假后端不发网络请求，用例里把出网入口换成会炸的桩，任何网络调用都让用例失败。
"""

from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest

from app.retrieval.base import RetrievalProvider
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.mock_retriever import (
    DEFAULT_FRAGMENT_POOL,
    DEFAULT_MAX_COUNT,
    DEFAULT_MIN_COUNT,
    MockFragment,
    MockRetriever,
)

# 默认片段池的条数
_POOL_SIZE = len(DEFAULT_FRAGMENT_POOL)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """把出网入口换成会炸的桩，任何网络调用都让用例失败。"""

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("假检索后端不该发起网络请求")

    # 注意：不 patch socket.socket.connect —— Windows 的 Proactor 事件循环
    # 内部用 socketpair 建自管道，会走 connect，patch 掉会让事件循环本身起不来。
    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    monkeypatch.setattr(httpx.AsyncClient, "send", _boom)
    monkeypatch.setattr(httpx.Client, "send", _boom)


# --------------------------------------------------------------------------
# 默认行为
# --------------------------------------------------------------------------


async def test_default_returns_count_in_expected_range() -> None:
    """不传参数时返回的片段条数落在 2 到 4 条之间。"""
    retriever = MockRetriever()

    results = await retriever.retrieve("项目的架构是怎样的")

    assert DEFAULT_MIN_COUNT <= len(results) <= DEFAULT_MAX_COUNT


async def test_default_returns_same_content_every_call() -> None:
    """默认行为每次返回固定内容。"""
    retriever = MockRetriever()

    first = await retriever.retrieve("项目架构")
    second = await retriever.retrieve("项目架构")

    assert [item.content for item in first] == [item.content for item in second]
    assert [item.document_name for item in first] == [
        item.document_name for item in second
    ]


async def test_fragment_fields_are_complete() -> None:
    """每条片段的正文、文档名、分数与附加信息都齐全。"""
    retriever = MockRetriever()

    results = await retriever.retrieve("事件总线的分发键怎么取")

    assert results
    for result in results:
        assert result.content.strip()
        assert result.document_name.strip()
        assert isinstance(result.score, float)
        assert 0.0 < result.score <= 1.0
        assert isinstance(result.metadata, dict)
        assert result.metadata["position"] >= 1
        assert result.metadata["segment_id"]
        assert result.metadata["source"] == "mock"
        assert isinstance(result.metadata["matched_keywords"], list)


async def test_scores_are_descending() -> None:
    """片段按分数从高到低排列，拼上下文时顺序不会被改。"""
    retriever = MockRetriever()

    results = await retriever.retrieve("截图扩展怎么抓图")

    scores = [item.score for item in results if item.score is not None]
    assert scores == sorted(scores, reverse=True)


async def test_implements_retrieval_provider() -> None:
    """假检索后端实现 RetrievalProvider 接口。"""
    assert isinstance(MockRetriever(), RetrievalProvider)


# --------------------------------------------------------------------------
# 片段条数与内容可控
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
async def test_count_controls_number_of_results(count: int) -> None:
    """构造参数能控制返回几条片段。"""
    retriever = MockRetriever(count=count)

    results = await retriever.retrieve("项目架构")

    assert len(results) == min(count, _POOL_SIZE)


async def test_pool_controls_fragment_content() -> None:
    """构造参数能控制片段内容从哪来。"""
    pool = (
        MockFragment(
            content="这是自定义的片段正文",
            document_name="自定义文档.md",
            keywords=("自定义",),
            score=0.7,
        ),
    )
    retriever = MockRetriever(pool=pool)

    results = await retriever.retrieve("自定义的东西")

    assert len(results) == 1
    assert results[0].content == "这是自定义的片段正文"
    assert results[0].document_name == "自定义文档.md"


async def test_default_pool_is_used_when_pool_is_none() -> None:
    """不传片段池时用内置的项目片段池。"""
    retriever = MockRetriever()

    assert retriever.pool == DEFAULT_FRAGMENT_POOL


# --------------------------------------------------------------------------
# 检索结果跟查询相关
# --------------------------------------------------------------------------


async def test_different_keywords_give_different_order() -> None:
    """查询里的关键词不同，命中的片段与顺序不同。"""
    retriever = MockRetriever()

    event_results = await retriever.retrieve("事件总线的分发键怎么取")
    shot_results = await retriever.retrieve("截图扩展怎么抓整屏")

    assert event_results[0].document_name == "app/core/event_bus.py"
    assert shot_results[0].document_name == "app/platform/windows/screenshot.py"
    assert [item.document_name for item in event_results] != [
        item.document_name for item in shot_results
    ]


async def test_matched_keywords_recorded_in_metadata() -> None:
    """命中的关键词记在片段的附加信息里。"""
    retriever = MockRetriever()

    results = await retriever.retrieve("事件总线的分发键怎么取")

    assert "事件总线" in results[0].metadata["matched_keywords"]


async def test_unmatched_query_falls_back_to_base_score_order() -> None:
    """查询没命中任何关键词时按片段池里的基础分从高到低返回。"""
    retriever = MockRetriever()

    results = await retriever.retrieve("跟这个项目完全无关的一句话")

    assert results
    assert all(not item.metadata["matched_keywords"] for item in results)
    expected = [
        fragment.document_name
        for fragment in sorted(DEFAULT_FRAGMENT_POOL, key=lambda item: -item.score)
    ][: len(results)]
    assert [item.document_name for item in results] == expected


async def test_context_builder_keeps_document_names() -> None:
    """片段拼成上下文时带上文档名，来源可展示。"""
    retriever = MockRetriever()

    results = await retriever.retrieve("检索接口是怎么抽象的")
    text = ContextBuilder().build(results)

    assert text
    assert results[0].document_name in text
    assert results[0].content.splitlines()[0] in text


# --------------------------------------------------------------------------
# 降级状态
# --------------------------------------------------------------------------


async def test_retrieve_outcome_is_ok_and_not_degraded() -> None:
    """retrieve_outcome 永远返回 ok=True、degraded=False。"""
    retriever = MockRetriever()

    outcome = await retriever.retrieve_outcome("项目架构")

    assert outcome.ok is True
    assert outcome.degraded is False
    assert outcome.error_code is None
    assert outcome.error_detail == ""
    assert outcome.results


async def test_retrieve_outcome_matches_retrieve() -> None:
    """retrieve_outcome 里的片段与 retrieve 返回的一致。"""
    retriever = MockRetriever()

    outcome = await retriever.retrieve_outcome("事件总线怎么用")
    results = await retriever.retrieve("事件总线怎么用")

    assert outcome.results == results


# --------------------------------------------------------------------------
# 不发网络请求
# --------------------------------------------------------------------------


async def test_retrieve_makes_no_network_call(no_network: None) -> None:
    """检索过程不发起任何网络请求。"""
    retriever = MockRetriever()

    results = await retriever.retrieve("项目架构")

    assert results


async def test_retrieve_outcome_makes_no_network_call(no_network: None) -> None:
    """取降级状态的过程也不发起网络请求。"""
    retriever = MockRetriever()

    outcome = await retriever.retrieve_outcome("项目架构")

    assert outcome.ok is True


def test_module_has_no_network_dependency() -> None:
    """假后端模块自己不引网络库。"""
    import app.retrieval.mock_retriever as module

    assert not hasattr(module, "httpx")
    assert not hasattr(module, "socket")
