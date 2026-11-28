"""题型判定与检索决策的测试。

覆盖《3-文本输入与流式回答》题型判定一节的规则顺序、项目类关键词、
检索需求，以及《8-集成与验收》防止重复调用一节里的文本规范化。
"""

from __future__ import annotations

from dataclasses import fields

from app.core.question_classifier import (
    AsyncQuestionClassifier,
    Classification,
    PROJECT_SCOPES,
    QuestionClassifier,
)
from app.core.question_session import (
    PROJECT_KEYWORDS,
    QUESTION_TYPES,
    normalize_text,
)
from app.core.retrieval_policy import RetrievalPolicy

# 各类题型的样例题面
CODE_BLOCK_TEXT = "```python\ndef quick_sort(arr):\n    return arr\n```"
FUNCTION_SIGNATURE_TEXT = "帮我看看这个函数签名怎么写：public void handle(Event e)"
ERROR_TEXT = "运行时报错堆栈：\nTraceback (most recent call last):\n  NameError: name 'x'"
ALGORITHM_TEXT = "给一个排序问题，要求时间复杂度是 O(n log n)，并给出输入输出样例"
SYSTEM_DESIGN_TEXT = "设计一个短链服务，怎么架构才能保证一致性"
UNKNOWN_TEXT = "谈谈你最近学到的一个知识点"


def test_classify_code_block_is_coding() -> None:
    """含完整代码块判为 coding。"""
    result = QuestionClassifier().classify(CODE_BLOCK_TEXT)
    assert result.question_type == "coding"
    assert "coding_block" in result.matched_rules


def test_classify_function_signature_is_coding() -> None:
    """含函数签名判为 coding。"""
    result = QuestionClassifier().classify(FUNCTION_SIGNATURE_TEXT)
    assert result.question_type == "coding"
    assert "coding_block" in result.matched_rules


def test_classify_error_stack_is_debugging() -> None:
    """含报错堆栈判为 debugging。"""
    result = QuestionClassifier().classify(ERROR_TEXT)
    assert result.question_type == "debugging"
    assert "error_stack" in result.matched_rules


def test_classify_project_keyword_is_project() -> None:
    """命中项目类关键词判为 project。"""
    result = QuestionClassifier().classify("讲讲我的项目是怎么做的")
    assert result.question_type == "project"
    assert "project_keyword" in result.matched_rules


def test_every_project_keyword_hits() -> None:
    """八个项目类关键词都能命中，命中即判 project。"""
    classifier = QuestionClassifier()
    for keyword in PROJECT_KEYWORDS:
        result = classifier.classify(f"讲讲{keyword}是怎么做的")
        assert result.question_type == "project", keyword
        assert "project_keyword" in result.matched_rules, keyword


def test_classify_algorithm_feature_is_coding() -> None:
    """含算法题特征判为 coding。"""
    result = QuestionClassifier().classify(ALGORITHM_TEXT)
    assert result.question_type == "coding"
    assert "algorithm_feature" in result.matched_rules


def test_classify_system_design_feature_is_system_design() -> None:
    """含系统设计特征判为 system_design。"""
    result = QuestionClassifier().classify(SYSTEM_DESIGN_TEXT)
    assert result.question_type == "system_design"
    assert "system_design_feature" in result.matched_rules


def test_classify_unknown_when_nothing_matches() -> None:
    """都命中不了判为 unknown。"""
    result = QuestionClassifier().classify(UNKNOWN_TEXT)
    assert result.question_type == "unknown"
    assert result.rag_required is False


def test_question_types_contains_all_six() -> None:
    """题型枚举是约定的六个取值。"""
    assert QUESTION_TYPES == (
        "conceptual",
        "project",
        "coding",
        "debugging",
        "system_design",
        "unknown",
    )


def test_project_keyword_sets_rag_required() -> None:
    """命中项目类关键词时 rag_required 为 true。"""
    result = QuestionClassifier().classify("我们的服务用的是哪个数据库")
    assert result.question_type == "project"
    assert result.rag_required is True


def test_coding_does_not_require_rag_by_default() -> None:
    """coding 默认不检索。"""
    result = QuestionClassifier().classify(CODE_BLOCK_TEXT)
    assert result.question_type == "coding"
    assert result.rag_required is False


def test_debugging_does_not_require_rag_by_default() -> None:
    """debugging 默认不检索。"""
    result = QuestionClassifier().classify(ERROR_TEXT)
    assert result.question_type == "debugging"
    assert result.rag_required is False


def test_code_with_project_keyword_still_requires_rag() -> None:
    """代码题同时命中项目类关键词时，题型按规则 1，检索需求为 true。"""
    result = QuestionClassifier().classify("我写的代码在这里：\n```python\nx = 1\n```")
    assert result.question_type == "coding"
    assert result.rag_required is True


def test_force_project_mode_always_requires_rag() -> None:
    """强制项目模式下 rag_required 恒为 true。"""
    classifier = QuestionClassifier(force_project_mode=True)
    for text in (CODE_BLOCK_TEXT, ERROR_TEXT, UNKNOWN_TEXT, SYSTEM_DESIGN_TEXT):
        assert classifier.classify(text).rag_required is True, text


def test_classification_is_frozen() -> None:
    """判定结果是不可变数据类。"""
    result = Classification(
        question_type="unknown", rag_required=False, matched_rules=("unknown_fallback",)
    )
    assert result.question_type == "unknown"


def test_classification_defaults_to_rules_source() -> None:
    """新加的字段有默认值：reason 为空、source 是 rules、project_scope 是 none。"""
    result = Classification(
        question_type="unknown", rag_required=False, matched_rules=("unknown_fallback",)
    )

    assert result.reason == ""
    assert result.source == "rules"
    assert result.project_scope == "none"


def test_classification_field_order_is_stable() -> None:
    """原三字段顺序不变，reason、source 与 project_scope 排在后面。"""
    names = tuple(item.name for item in fields(Classification))

    assert names == (
        "question_type",
        "rag_required",
        "matched_rules",
        "reason",
        "source",
        "project_scope",
    )


def test_rule_classifier_marks_source_rules() -> None:
    """同步规则判定的来源是 rules，reason 留空。"""
    result = QuestionClassifier().classify(UNKNOWN_TEXT)

    assert result.source == "rules"
    assert result.reason == ""


def test_async_classifier_protocol_exists() -> None:
    """core 层声明了异步判定协议，llm 层按它实现。"""
    assert hasattr(AsyncQuestionClassifier, "classify")


def test_retrieval_policy_disabled_never_retrieves() -> None:
    """检索开关关闭时不检索。"""
    policy = RetrievalPolicy(enabled=False)
    decision = policy.decide(
        Classification(
            question_type="project",
            rag_required=True,
            matched_rules=("project_keyword",),
            project_scope="full",
        )
    )
    assert decision.should_retrieve is False
    assert "关闭" in decision.reason


def test_retrieval_policy_force_project_mode_always_retrieves() -> None:
    """强制项目模式下恒检索，是分类错类时的兜底。"""
    policy = RetrievalPolicy(enabled=True, force_project_mode=True)
    decision = policy.decide(
        Classification(
            question_type="unknown",
            rag_required=False,
            matched_rules=("unknown_fallback",),
            project_scope="none",
        )
    )
    assert decision.should_retrieve is True
    assert decision.reason == "强制项目模式"


def test_retrieval_policy_follows_project_scope() -> None:
    """其余情况按 project_scope 映射：mixed 与 full 检索，none 不检索。"""
    policy = RetrievalPolicy(enabled=True)
    full = policy.decide(
        Classification(
            question_type="project",
            rag_required=True,
            matched_rules=("project_keyword",),
            project_scope="full",
        )
    )
    mixed = policy.decide(
        Classification(
            question_type="conceptual",
            rag_required=True,
            matched_rules=("llm",),
            source="llm",
            project_scope="mixed",
        )
    )
    none_scope = policy.decide(
        Classification(
            question_type="coding",
            rag_required=False,
            matched_rules=("coding_block",),
            project_scope="none",
        )
    )
    assert full.should_retrieve is True
    assert mixed.should_retrieve is True
    assert none_scope.should_retrieve is False
    assert full.reason == "题目是项目题，检索项目知识为主"
    assert mixed.reason == "题目是简答题带项目，检索项目知识作补充"
    assert none_scope.reason == "题目是纯简答题，无需检索项目知识"


def test_normalize_text_removes_punctuation_and_space() -> None:
    """规范化去掉空白与中英文标点，保留字母数字与汉字并转小写。"""
    assert normalize_text("你好，世界！") == "你好世界"
    assert normalize_text("Hello, World!") == "helloworld"
    assert normalize_text("  A B\tC\n") == "abc"
    assert normalize_text("排序 O(n log n)") == "排序onlogn"
    assert normalize_text("，。！？") == ""

# ----------------------------------------------------------------
# 项目范围三分类：本地规则回退只有 none/full 两档
# ----------------------------------------------------------------


def test_project_keyword_sets_project_scope_full() -> None:
    """命中项目类关键词时 project_scope 是 full。"""
    result = QuestionClassifier().classify("讲讲我的项目是怎么做的")

    assert result.project_scope == "full"
    assert result.rag_required is True


def test_no_project_keyword_sets_project_scope_none() -> None:
    """没命中项目类关键词时 project_scope 是 none。"""
    classifier = QuestionClassifier()
    for text in (CODE_BLOCK_TEXT, ERROR_TEXT, ALGORITHM_TEXT, SYSTEM_DESIGN_TEXT):
        result = classifier.classify(text)
        assert result.project_scope == "none", text
        assert result.rag_required is False, text


def test_code_with_project_keyword_sets_project_scope_full() -> None:
    """代码题同时命中项目类关键词时，题型按规则 1，范围是 full。"""
    result = QuestionClassifier().classify("我写的代码在这里：\n```python\nx = 1\n```")

    assert result.question_type == "coding"
    assert result.project_scope == "full"


def test_unknown_text_sets_project_scope_none() -> None:
    """都命中不了的题面 project_scope 是 none。"""
    result = QuestionClassifier().classify(UNKNOWN_TEXT)

    assert result.question_type == "unknown"
    assert result.project_scope == "none"


def test_force_project_mode_does_not_change_project_scope() -> None:
    """强制项目模式只抬 rag_required，不改 project_scope 表达的题目性质。"""
    classifier = QuestionClassifier(force_project_mode=True)
    result = classifier.classify(UNKNOWN_TEXT)

    assert result.rag_required is True
    assert result.project_scope == "none"


def test_project_scopes_has_three_legal_values() -> None:
    """项目范围维度是约定的三档取值。"""
    assert PROJECT_SCOPES == frozenset({"none", "mixed", "full"})

def test_force_project_mode_property_reports_flag() -> None:
    """force_project_mode 属性如实返回构造时的开关。"""
    assert QuestionClassifier().force_project_mode is False
    assert QuestionClassifier(force_project_mode=True).force_project_mode is True


def test_retrieval_policy_properties_report_flags() -> None:
    """enabled 与 force_project_mode 属性如实返回构造时的开关。"""
    default = RetrievalPolicy()
    assert default.enabled is True
    assert default.force_project_mode is False
    strict = RetrievalPolicy(enabled=False, force_project_mode=True)
    assert strict.enabled is False
    assert strict.force_project_mode is True
