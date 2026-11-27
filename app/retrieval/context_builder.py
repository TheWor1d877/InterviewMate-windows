"""片段整理与上下文拼装。

规则见《6-Dify检索与项目知识》结果整理一节：

- 按分数从高到低排列片段。
- 总量截断到 ``max_context_chars``，默认 8000 字符。
- 每个片段带上文档名，让模型能引用来源。
- 片段之间用明确分隔，避免相邻片段的内容被模型当成一句。

分数为 None 的片段排在有分数的片段后面，同组内保持原有顺序。
只保留相关片段，整个知识库或大量无关文本不进 Prompt。
"""

from __future__ import annotations

from app.retrieval.retrieval_models import RetrievalResult

__all__ = ["ContextBuilder", "DEFAULT_MAX_CONTEXT_CHARS", "CONTEXT_SEPARATOR"]

# 默认上下文长度上限，见《6-Dify检索与项目知识》检索参数一节
DEFAULT_MAX_CONTEXT_CHARS: int = 8000

# 片段之间的分隔，避免相邻片段被当成一句
CONTEXT_SEPARATOR: str = "\n\n---\n\n"


class ContextBuilder:
    """把检索片段拼成给模型看的上下文文本。

    :param max_context_chars: 上下文总长度上限，超出时截断。
    :param separator: 片段之间的分隔串。
    """

    def __init__(
        self,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        separator: str = CONTEXT_SEPARATOR,
    ) -> None:
        self._max_context_chars = int(max_context_chars)
        self._separator = separator

    @property
    def max_context_chars(self) -> int:
        """上下文总长度上限。"""
        return self._max_context_chars

    def build(self, results: list[RetrievalResult]) -> str:
        """把片段拼成上下文文本。

        :param results: 检索片段列表。
        :return: 上下文文本，片段为空时是空串。长度不超过 ``max_context_chars``。
        """
        parts: list[str] = []
        used = 0
        for index, result in enumerate(_sort_by_score(results), start=1):
            block = self._format_block(index, result)
            separator_len = len(self._separator) if parts else 0
            allowance = self._max_context_chars - used - separator_len
            if allowance <= 0:
                break
            if len(block) > allowance:
                block = self._trim(block, allowance)
                if not block:
                    break
            parts.append(block)
            used += len(block) + separator_len
        return self._separator.join(parts)

    def _format_block(self, index: int, result: RetrievalResult) -> str:
        """一个片段的文本，带序号与文档名，让模型能引用来源。"""
        name = result.document_name or "未命名文档"
        return f"[片段 {index}] 来源文档：{name}\n{result.content}"

    def _trim(self, block: str, allowance: int) -> str:
        """把一块内容裁到给定长度内，文档名那一行保留。"""
        if allowance <= 0:
            return ""
        if len(block) <= allowance:
            return block
        head, _, body = block.partition("\n")
        if len(head) + 1 > allowance:
            return block[:allowance]
        keep = allowance - len(head) - 1
        if keep <= 0:
            return head
        return head + "\n" + body[: keep - 1] + "…"


def _sort_by_score(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """按分数从高到低排列，分数为 None 的排在后面。

    排序是稳定的，同组内保持原有顺序。
    """
    return sorted(results, key=_sort_key)


def _sort_key(result: RetrievalResult) -> tuple[int, float]:
    """排序键：有分数的在前且分数高的在前，没分数的排后面。"""
    if result.score is None:
        return (1, 0.0)
    return (0, -result.score)
