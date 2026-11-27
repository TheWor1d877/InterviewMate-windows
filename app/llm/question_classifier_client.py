"""用模型判定题型与检索需求。

规则判定在 :mod:`app.core.question_classifier` 里，模型判定放在这里：
本模块实现 core 层声明的 :class:`~app.core.question_classifier.AsyncQuestionClassifier`
协议，core 层不 import 本模块，见《2-骨架与配置》目录结构约定一节。

一次判定只发一次非流式请求，要求模型只输出一段 JSON；
解析容错写到 :func:`_parse_classification`，出错一律抛 ``ValueError``，
超时交给 ``asyncio.wait_for``，都由上层回退到同步规则。
"""

from __future__ import annotations

import asyncio
import json
from typing import Protocol

from app.core.question_classifier import Classification

__all__ = [
    "JSONCompletionClient",
    "DeepSeekQuestionClassifier",
    "LEGAL_QUESTION_TYPES",
]


# 模型判定允许的题型，取值是《3-文本输入与流式回答》题型判定一节里的子集
LEGAL_QUESTION_TYPES: frozenset[str] = frozenset(
    {"coding", "project", "system_design", "conceptual", "unknown"}
)

# 一次判定的超时（秒）：判定只决定要不要检索，不该拖住提交流程
_DEFAULT_TIMEOUT: float = 3.0
# JSON 回复很短，几十个 token 足够
_DEFAULT_MAX_TOKENS: int = 128
# 判定要稳定，温度取 0
_DEFAULT_TEMPERATURE: float = 0.0

# 判定提示词：只输出 JSON，规则里写明通用知识题不得判成 project
_SYSTEM_PROMPT: str = """你是面试辅助程序的题型判定器。只输出一行 JSON，不要代码块，不要解释，不要任何前后缀。
JSON 字段固定为：
{"type": "coding|project|system_design|conceptual|unknown", "need_rag": true 或 false, "reason": "不超过20字的中文依据"}

判定规则：
1. LeetCode 题面、算法题、要求写代码或给出实现、报错堆栈或异常 → type 为 coding，need_rag 为 false。
2. 询问求职者本人做过的项目、服务、工作经历或其中的技术选型（出现“我的项目”“我之前的项目”“我们服务怎么实现”等指向本人经历的表述）→ type 为 project，need_rag 为 true。
3. 通用八股知识题（如 TCP、数据库索引、shared_ptr 原理）→ type 为 conceptual 或 system_design，need_rag 为 false。
4. 拿不准 → type 为 unknown，need_rag 为 false。

禁止把通用知识题判成 project：题目没有明确指向求职者本人的项目或经历时，不得判 project。
只输出 JSON，不要代码块。"""


class JSONCompletionClient(Protocol):
    """非流式 JSON 补全客户端的能力约定。

    真实实现是 :class:`~app.llm.deepseek_client.DeepSeekClient`，
    测试里注入假实现，本模块不关心底层怎么发请求。
    """

    async def complete_json(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


class DeepSeekQuestionClassifier:
    """用模型判定题型与检索需求。

    :param client: 具备 ``complete_json`` 的非流式客户端。
    :param timeout: 单次判定的超时（秒），超时抛 ``TimeoutError``，
        由上层回退到同步规则。默认 3 秒。
    """

    def __init__(
        self,
        client: JSONCompletionClient,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        self._client = client
        self._timeout = float(timeout)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)

    async def classify(self, text: str) -> Classification:
        """发一次非流式请求，把模型回复解析成判定结果。

        :param text: 用户输入的题面。
        :return: 判定结果，``source`` 恒为 ``"llm"``。
        :raises ValueError: 回复不是合法 JSON，或字段缺失、取值非法。
        :raises TimeoutError: 超过 ``timeout`` 秒没有回复。
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        raw = await asyncio.wait_for(
            self._client.complete_json(
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            ),
            timeout=self._timeout,
        )
        return _parse_classification(raw)


def _parse_classification(raw: str) -> Classification:
    """把模型回复解析成 :class:`Classification`。

    容错只做两件事：去掉 ``` 围栏，取第一个 ``{`` 到最后一个 ``}``。
    其余情况（非 JSON、缺字段、type 非法、need_rag 不是布尔）一律抛
    ``ValueError``，由上层回退规则。

    :param raw: 模型回复的原文。
    """
    if not isinstance(raw, str):
        raise ValueError("题型判定回复不是字符串")
    text = raw.replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("题型判定回复里没有 JSON 对象")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"题型判定回复不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("题型判定回复不是 JSON 对象")

    question_type = payload.get("type")
    if not isinstance(question_type, str) or question_type not in LEGAL_QUESTION_TYPES:
        raise ValueError(f"题型判定回复的 type 非法：{question_type!r}")

    need_rag = payload.get("need_rag")
    if not isinstance(need_rag, bool):
        raise ValueError(f"题型判定回复的 need_rag 非法：{need_rag!r}")

    reason = payload.get("reason")
    if not isinstance(reason, str):
        raise ValueError(f"题型判定回复的 reason 非法：{reason!r}")

    return Classification(
        question_type=question_type,
        rag_required=need_rag,
        matched_rules=("llm",),
        reason=reason.strip(),
        source="llm",
    )