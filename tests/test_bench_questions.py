"""延迟测试题目集的测试。

《8-集成与验收》性能统计一节要求「用不少于 20 条代表性问题做延迟测试，
报告 P50 与 P95」，题目集在 :mod:`scripts.bench_questions`。
这个文件盯住那份清单本身：

- 条数够不够《8》要求的 20 条；
- 每条题的字段填没填；
- 期望题型是不是合法的题型标识；
- 题面有没有重复（重复的题面量不了分布）；
- 题目集写的期望与仓库里分类器、检索策略的真实行为对不对得上。

最后一条是防回归用的：期望值是按分类器的真实规则定的，
分类器或检索策略改了规则，这里会立刻红。
"""

from __future__ import annotations

import pytest

from app.core.question_classifier import QuestionClassifier
from app.core.question_session import QUESTION_TYPES
from app.core.retrieval_policy import RetrievalPolicy
from scripts.bench_questions import BENCH_QUESTIONS, BenchQuestion

# 《8-集成与验收》性能统计一节要求的下限
MIN_QUESTION_COUNT = 20

# 题目集当前约定的条数
EXPECTED_QUESTION_COUNT = 24


def _questions() -> list[BenchQuestion]:
    """返回题目清单，逐条做参数化时用。"""
    return list(BENCH_QUESTIONS)


def test_question_count_meets_requirement() -> None:
    """题目条数不少于《8》要求的 20 条。"""
    assert len(BENCH_QUESTIONS) >= MIN_QUESTION_COUNT


def test_question_count_is_stable() -> None:
    """题目条数固定为 24 条，加题要一起改这里的数字。"""
    assert len(BENCH_QUESTIONS) == EXPECTED_QUESTION_COUNT


@pytest.mark.parametrize("question", _questions(), ids=lambda item: item.text)
def test_fields_are_filled(question: BenchQuestion) -> None:
    """每条题的字段都填了，没有空值。"""
    assert question.text.strip(), "题面不能为空"
    assert question.expected_type.strip(), "期望题型不能为空"
    assert isinstance(question.expect_retrieval, bool), "期望检索行为必须是布尔值"


@pytest.mark.parametrize("question", _questions(), ids=lambda item: item.text)
def test_expected_type_is_legal(question: BenchQuestion) -> None:
    """期望题型是合法题型标识，取值见 QUESTION_TYPES。"""
    assert question.expected_type in QUESTION_TYPES


def test_texts_are_unique() -> None:
    """题面没有重复，重复的题面量不出分布。"""
    texts = [question.text for question in BENCH_QUESTIONS]
    duplicates = {text for text in texts if texts.count(text) > 1}
    assert not duplicates, f"题面重复：{sorted(duplicates)}"


@pytest.mark.parametrize("question", _questions(), ids=lambda item: item.text)
def test_expected_type_matches_classifier(question: BenchQuestion) -> None:
    """分类器判出来的题型与题目集写的期望一致。

    这条是防回归的：分类器的判定规则改了，题目集的期望要跟着改，
    或者反过来题目集该换题。
    """
    classification = QuestionClassifier().classify(question.text)
    assert classification.question_type == question.expected_type


@pytest.mark.parametrize("question", _questions(), ids=lambda item: item.text)
def test_expected_retrieval_matches_policy(question: BenchQuestion) -> None:
    """检索策略的决策与题目集写的期望一致。

    判定与决策都走仓库里的实现，不在测试里另算一遍，
    见《6-Dify检索与项目知识》检索流程一节。
    """
    classifier = QuestionClassifier()
    classification = classifier.classify(question.text)
    decision = RetrievalPolicy().decide(classification)
    assert decision.should_retrieve is question.expect_retrieval


def test_project_questions_expect_retrieval() -> None:
    """项目题期望检索，其余题型期望不检索。"""
    for question in BENCH_QUESTIONS:
        if question.expected_type == "project":
            assert question.expect_retrieval is True, question.text
        else:
            assert question.expect_retrieval is False, question.text


def test_every_question_type_is_covered() -> None:
    """题目集覆盖了五种题型，不是只测一种。"""
    covered = {question.expected_type for question in BENCH_QUESTIONS}
    assert covered == {
        "unknown",
        "debugging",
        "system_design",
        "coding",
        "project",
    }
