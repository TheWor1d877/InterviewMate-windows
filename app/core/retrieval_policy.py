"""是否检索的决策。

检索开关与强制项目模式的取值来自配置，见《2-骨架与配置》配置结构一节；
决策依据是《3-文本输入与流式回答》题型判定一节给出的 ``rag_required``。

本模块只做决策，不发起检索，也不关心检索结果。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.question_classifier import Classification

__all__ = ["RetrievalDecision", "RetrievalPolicy"]


@dataclass(frozen=True)
class RetrievalDecision:
    """是否检索的决策结果。"""

    should_retrieve: bool
    reason: str


class RetrievalPolicy:
    """按开关、强制项目模式与 ``rag_required`` 决定是否检索。

    :param enabled: 检索总开关，对应配置里的 ``rag.enabled``。
    :param force_project_mode: 强制项目模式，对应配置里的 ``rag.force_project_mode``，
        开启后恒检索。
    """

    def __init__(self, enabled: bool = True, force_project_mode: bool = False) -> None:
        self._enabled = bool(enabled)
        self._force_project_mode = bool(force_project_mode)

    @property
    def enabled(self) -> bool:
        """检索总开关是否打开。"""
        return self._enabled

    @property
    def force_project_mode(self) -> bool:
        """是否处于强制项目模式。"""
        return self._force_project_mode

    def decide(self, classification: Classification) -> RetrievalDecision:
        """给出是否检索的决策结果。

        开关关闭时不检索；强制项目模式下恒检索；其余按 ``rag_required`` 决定。

        :param classification: 题型判定结果。
        :return: 决策结果，``reason`` 说明决策依据。
        """
        if not self._enabled:
            return RetrievalDecision(should_retrieve=False, reason="检索开关已关闭")
        if self._force_project_mode:
            return RetrievalDecision(should_retrieve=True, reason="强制项目模式")
        if classification.rag_required:
            return RetrievalDecision(
                should_retrieve=True, reason="题目命中项目类关键词，需要检索项目知识"
            )
        return RetrievalDecision(
            should_retrieve=False, reason="题目不涉及项目知识，无需检索"
        )
