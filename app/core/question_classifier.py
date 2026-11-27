"""题型判定与检索需求判定。

判定顺序与关键词见《3-文本输入与流式回答》题型判定一节：

1. 输入含完整代码块、函数签名、报错堆栈，判为 ``coding`` 或 ``debugging``；
2. 输入命中项目类关键词，判为 ``project``；
3. 输入含算法题特征，判为 ``coding``；
4. 输入含系统设计特征，判为 ``system_design``；
5. 都命中不了，判为 ``unknown``。

判定只做规则匹配，不调用模型。模型判定由 llm 层实现
:class:`AsyncQuestionClassifier` 协议，本模块只声明协议、不 import llm 实现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from app.core.question_session import PROJECT_KEYWORDS, normalize_text

__all__ = [
    "Classification",
    "AsyncQuestionClassifier",
    "QuestionClassifier",
    "DEBUGGING_MARKERS",
    "CODE_MARKERS",
    "ALGORITHM_MARKERS",
    "SYSTEM_DESIGN_MARKERS",
]


# 报错堆栈与错误信息特征，命中判 debugging
DEBUGGING_MARKERS: tuple[str, ...] = (
    "traceback",
    "stacktrace",
    "exception",
    "报错",
    "错误信息",
    "出错",
    "哪里错",
    "异常堆栈",
    "抛出异常",
    "未捕获异常",
)

# 代码块与函数签名特征，命中判 coding
CODE_MARKERS: tuple[str, ...] = (
    "代码块",
    "函数签名",
    "代码围栏",
)

# 算法题特征，命中判 coding
ALGORITHM_MARKERS: tuple[str, ...] = (
    "时间复杂度",
    "空间复杂度",
    "复杂度",
    "输入输出",
    "输出样例",
    "输入样例",
    "样例",
    "动态规划",
    "排序",
    "贪心",
    "二分查找",
    "回溯",
    "背包",
    "前缀和",
    "滑动窗口",
    "双指针",
    "哈希表",
    "二叉树",
    "最长递增子序列",
    "字符串匹配",
    "算法题",
    "leetcode",
    "力扣",
)

# 系统设计特征，命中判 system_design
SYSTEM_DESIGN_MARKERS: tuple[str, ...] = (
    "设计一个",
    "设计一套",
    "怎么架构",
    "如何架构",
    "架构",
    "系统设计",
    "保证一致性",
    "一致性",
    "高并发",
    "分布式",
    "微服务",
    "限流",
    "熔断",
    "负载均衡",
    "分库分表",
    "可用性",
    "扩展性",
    "容量规划",
)

# 代码块围栏：三个反引号或三个波浪号
_CODE_FENCE_PATTERN = re.compile(r"```|~~~")

# 函数签名：def foo(...)、public void foo(...)、class Foo:、foo() -> int 等常见写法
_FUNCTION_SIGNATURE_PATTERN = re.compile(
    r"(?:\bdef\s+\w+\s*\("
    r"|\b(?:async\s+)?(?:function|func|fn)\s+\w+\s*\("
    r"|\bclass\s+\w+\s*[\(:]"
    r"|\bstruct\s+\w+\s*[\{:]"
    r"|\binterface\s+\w+\s*[\{:]"
    r"|\b(?:public|private|protected)\s+(?:static\s+)?(?:async\s+)?[\w<>\[\],\s]+?\s+\w+\s*\("
    r"|\b(?:void|int|long|float|double|bool|char|str|string|object|auto)\s+\w+\s*\("
    r"|\b\w+\s*\([^()]*\)\s*(?:->|:)\s*[\{:A-Za-z]"
    r")"
)

# 报错堆栈：Traceback 头、Error: / Exception: 这类异常类型前缀
_ERROR_STACK_PATTERN = re.compile(
    r"(?:\btraceback\b|\bstack\s*trace\b|\b\w*(?:error|exception)\b\s*:)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Classification:
    """一次判定的结果。

    ``source`` 标明结果来自哪条判定路径：``"llm"`` 是模型判定，
    ``"rules"`` 是本地规则回退；``reason`` 是模型给出的一句话依据，
    规则判定时留空。前三个字段的顺序不变。
    """

    question_type: str
    rag_required: bool
    matched_rules: tuple[str, ...]
    reason: str = ""
    source: str = "rules"


class AsyncQuestionClassifier(Protocol):
    """异步题型判定器的能力约定。

    模型判定实现放 llm 层，core 层只认这个协议，见
    《2-骨架与配置》目录结构约定一节。
    """

    async def classify(self, text: str) -> Classification: ...


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    """判断文本里是否出现任一标记词。"""
    return any(marker in text for marker in markers)


class QuestionClassifier:
    """按《3-文本输入与流式回答》的顺序做规则判定。

    :param force_project_mode: 开启后 ``rag_required`` 恒为 true，
        对应配置里的 ``rag.force_project_mode``。
    """

    def __init__(self, force_project_mode: bool = False) -> None:
        self._force_project_mode = bool(force_project_mode)

    @property
    def force_project_mode(self) -> bool:
        """是否处于强制项目模式。"""
        return self._force_project_mode

    def classify(self, text: str) -> Classification:
        """判定题面属于哪种题型，以及是否需要检索项目知识。

        判定顺序固定为：代码与报错 → 项目关键词 → 算法特征 → 系统设计特征 → 未知。
        关键词匹配用规范化后的文本，代码块与函数签名用原文。

        :param text: 用户输入的题面。
        :return: 判定结果，``matched_rules`` 记录命中的规则名，便于排查。
        """
        raw = text.lower()
        normalized = normalize_text(text)

        # 项目类关键词单独算：命中与否只影响 rag_required，
        # 不影响判定顺序里哪种题型先命中，见《3-文本输入与流式回答》题型判定一节
        project_hit = _contains_any(normalized, PROJECT_KEYWORDS)

        # 规则 1：代码块、函数签名、报错堆栈
        has_code_block = bool(_CODE_FENCE_PATTERN.search(raw)) or _contains_any(
            normalized, CODE_MARKERS
        )
        has_signature = bool(_FUNCTION_SIGNATURE_PATTERN.search(raw))
        has_error = bool(_ERROR_STACK_PATTERN.search(raw)) or _contains_any(
            normalized, DEBUGGING_MARKERS
        )
        if has_error:
            matched = ("error_stack",)
            if has_code_block or has_signature:
                matched = matched + ("coding_block",)
            return Classification(
                question_type="debugging",
                rag_required=self._rag_required(project_hit=project_hit),
                matched_rules=matched,
            )
        if has_code_block or has_signature:
            return Classification(
                question_type="coding",
                rag_required=self._rag_required(project_hit=project_hit),
                matched_rules=("coding_block",),
            )

        # 规则 2：项目类关键词
        if project_hit:
            return Classification(
                question_type="project",
                rag_required=self._rag_required(project_hit=True),
                matched_rules=("project_keyword",),
            )

        # 规则 3：算法题特征
        if _contains_any(normalized, ALGORITHM_MARKERS):
            return Classification(
                question_type="coding",
                rag_required=self._rag_required(project_hit=project_hit),
                matched_rules=("algorithm_feature",),
            )

        # 规则 4：系统设计特征
        if _contains_any(normalized, SYSTEM_DESIGN_MARKERS):
            return Classification(
                question_type="system_design",
                rag_required=self._rag_required(project_hit=project_hit),
                matched_rules=("system_design_feature",),
            )

        # 规则 5：都命中不了
        return Classification(
            question_type="unknown",
            rag_required=self._rag_required(project_hit=project_hit),
            matched_rules=("unknown_fallback",),
        )

    def _rag_required(self, *, project_hit: bool) -> bool:
        """按《3-文本输入与流式回答》题型判定一节算检索需求。

        项目类关键词命中置 true；``coding`` 与 ``debugging`` 默认不检索，
        除非同时命中项目类关键词；强制项目模式下恒为 true。
        """
        if self._force_project_mode:
            return True
        return project_hit
