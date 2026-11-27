"""模型题型判定客户端的测试。

用假补全客户端验证：合法 JSON 解析、围栏与前后废话容错、
非法回复抛 ``ValueError``、超时抛 ``TimeoutError``。不发网络请求。
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.question_classifier import Classification
from app.llm.question_classifier_client import (
    LEGAL_QUESTION_TYPES,
    DeepSeekQuestionClassifier,
)


class FakeCompletionClient:
    """假 JSON 补全客户端，记录调用并按设定返回或抛错。"""

    def __init__(
        self,
        reply: str = "",
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.reply = reply
        self.error = error
        self.delay = delay
        self.calls: list[dict] = []

    async def complete_json(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.reply


async def test_valid_json_maps_to_classification() -> None:
    """合法 JSON 解出 type、need_rag 与 reason，来源标成 llm。"""
    client = FakeCompletionClient(
        reply='{"type": "coding", "need_rag": false, "reason": "算法题"}'
    )
    classifier = DeepSeekQuestionClassifier(client)

    result = await classifier.classify("给定一个数组，返回两数之和")

    assert isinstance(result, Classification)
    assert result.question_type == "coding"
    assert result.rag_required is False
    assert result.reason == "算法题"
    assert result.source == "llm"
    assert result.matched_rules == ("llm",)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["max_tokens"] == 128
    assert call["temperature"] == 0.0
    assert call["messages"][0]["role"] == "system"
    assert "只输出" in call["messages"][0]["content"]
    assert "JSON" in call["messages"][0]["content"]
    assert call["messages"][1] == {
        "role": "user",
        "content": "给定一个数组，返回两数之和",
    }


async def test_project_classification_sets_rag_required() -> None:
    """project 且 need_rag=true 原样落到判定结果。"""
    client = FakeCompletionClient(
        reply='{"type": "project", "need_rag": true, "reason": "本人项目"}'
    )

    result = await DeepSeekQuestionClassifier(client).classify("我之前的项目怎么做的")

    assert result.question_type == "project"
    assert result.rag_required is True


async def test_code_fence_and_surrounding_text_are_tolerated() -> None:
    """带 ``` 围栏、前后有废话时照样能解出 JSON。"""
    reply = (
        "好的，判定如下：\n"
        "```json\n"
        '{"type": "system_design", "need_rag": false, "reason": "通用设计题"}\n'
        "```\n"
        "以上。"
    )
    client = FakeCompletionClient(reply=reply)

    result = await DeepSeekQuestionClassifier(client).classify("设计一个短链服务")

    assert result.question_type == "system_design"
    assert result.rag_required is False
    assert result.reason == "通用设计题"


@pytest.mark.parametrize("question_type", sorted(LEGAL_QUESTION_TYPES))
async def test_all_legal_types_are_accepted(question_type: str) -> None:
    """五个合法 type 都能解析。"""
    client = FakeCompletionClient(
        reply=f'{{"type": "{question_type}", "need_rag": false, "reason": "依据"}}'
    )

    result = await DeepSeekQuestionClassifier(client).classify("题面")

    assert result.question_type == question_type


@pytest.mark.parametrize(
    "reply",
    [
        # 非法 type：debugging 不在模型判定允许的取值里
        '{"type": "debugging", "need_rag": false, "reason": "报错"}',
        # 缺少 need_rag
        '{"type": "coding", "reason": "算法题"}',
        # 缺少 reason
        '{"type": "coding", "need_rag": false}',
        # need_rag 不是布尔
        '{"type": "coding", "need_rag": "true", "reason": "算法题"}',
        # 不是 JSON
        "这道题是算法题",
        # 空串
        "",
    ],
)
async def test_invalid_reply_raises_value_error(reply: str) -> None:
    """非法 type、缺字段、非 JSON 都抛 ValueError，由上层回退。"""
    client = FakeCompletionClient(reply=reply)

    with pytest.raises(ValueError):
        await DeepSeekQuestionClassifier(client).classify("题面")


async def test_timeout_raises_timeout_error() -> None:
    """超过超时时间抛 TimeoutError，由上层回退规则。"""
    client = FakeCompletionClient(
        reply='{"type": "coding", "need_rag": false, "reason": "算法题"}',
        delay=0.5,
    )
    classifier = DeepSeekQuestionClassifier(client, timeout=0.01)

    with pytest.raises(asyncio.TimeoutError):
        await classifier.classify("题面")


async def test_client_error_propagates() -> None:
    """补全客户端的异常原样抛出，本模块不吞异常。"""
    client = FakeCompletionClient(error=RuntimeError("网络挂了"))
    classifier = DeepSeekQuestionClassifier(client)

    with pytest.raises(RuntimeError):
        await classifier.classify("题面")