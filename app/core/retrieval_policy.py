"""是否检索的决策。

检索开关与强制项目模式的取值来自配置，见《2-骨架与配置》配置结构一节；
决策依据是题型判定给出的 ``project_scope`` 三档：``mixed`` 与 ``full``
检索，``none`` 不检索。

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
    """按开关、强制项目模式与 ``project_scope`` 决定是否检索。

    :param enabled: 检索总开关，对应配置里的 ``rag.enabled``。
    :param force_project_mode: 强制项目模式，对应配置里的 ``rag.force_project_mode``，
        开启后恒检索，是分类错类时的兜底。
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

        开关关闭时不检索；强制项目模式下恒检索；其余按 ``project_scope``
        映射：``mixed`` 与 ``full`` 检索，``none`` 不检索。

        :param classification: 题型判定结果。
        :return: 决策结果，``reason`` 说明决策依据。
        """
        if not self._enabled:
            return RetrievalDecision(should_retrieve=False, reason="检索开关已关闭")
        if self._force_project_mode:
            return RetrievalDecision(should_retrieve=True, reason="强制项目模式")
        if classification.project_scope == "full":
            return RetrievalDecision(
                should_retrieve=True, reason="题目是项目题，检索项目知识为主"
            )
        if classification.project_scope == "mixed":
            return RetrievalDecision(
                should_retrieve=True, reason="题目是简答题带项目，检索项目知识作补充"
            )
        return RetrievalDecision(
            should_retrieve=False, reason="题目是纯简答题，无需检索项目知识"
        )
