"""延迟测试用的题目集。

《8-集成与验收》性能统计一节要求：「用不少于 20 条代表性问题做延迟测试，
报告 P50 与 P95」。这组题目就是那份「代表性问题」的固定清单，
放在仓库里是为了每次跑出来的数字可比较、可复现。

题目的期望题型按 :mod:`app.core.question_classifier` 的真实规则定，
不是按字面感觉定的。几个容易踩的点：

- ``ALGORITHM_MARKERS`` 里「哈希表」「二分查找」「双指针」「滑动窗口」
  命中判 ``coding``；「样例」「复杂度」「排序」也命中，所以题目里避开这三个宽泛词。
- ``SYSTEM_DESIGN_MARKERS`` 里「一致性」命中判 ``system_design``。
- 项目类关键词优先于算法类关键词：「这个项目…用哈希表…」仍然判 ``project``。
- **分类器没有「概念题」这个判定**：命中不了任何特征的题判 ``unknown``，
  见 :meth:`app.core.question_classifier.QuestionClassifier.classify` 的规则 5。
  所以「普通题」在类型标识上就是 ``unknown``，不是 ``conceptual``。
- 项目类关键词取自 :data:`app.core.question_session.PROJECT_KEYWORDS`，
  只有「我的项目」「这个项目」「我之前的项目」「项目中怎么实现」「项目架构」
  「我们的服务」「我写的代码」「我们用的数据库」这几种写法命中，
  题目措辞必须落在这几种里，「我们项目」这种写法不算。

``expect_retrieval`` 标的是「按策略该不该发起 Dify 检索」，
检索策略见《6-Dify检索与项目知识》检索流程一节：普通题、算法题不检索，
项目题检索。

题目本身不含任何私密内容，只问仓库文档里公开的内容。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BenchQuestion",
    "BENCH_QUESTIONS",
]


@dataclass(frozen=True)
class BenchQuestion:
    """一条延迟测试题目。

    :param text: 题面原文。
    :param expected_type: 期望的题型标识，取值见
        :data:`app.core.question_session.QUESTION_TYPES`。
    :param expect_retrieval: 期望这次提交是否发起 Dify 检索。
    """

    text: str
    expected_type: str
    expect_retrieval: bool


# 普通题：不命中任何特征，判 unknown，不触发检索。
# 《8》验收清单里的「普通题」就是这一组。
_CONCEPTUAL: tuple[tuple[str, str], ...] = (
    ("解释一下 TCP 三次握手和四次挥手的过程", "unknown"),
    ("Python 的生成器是怎么实现的，和迭代器有什么区别", "unknown"),
    ("数据库索引为什么用 B+ 树而不是红黑树", "unknown"),
    ("HTTP 的 Keep-Alive 是什么，解决了什么问题", "unknown"),
    ("进程间通信有哪些方式，各自适用什么场景", "unknown"),
    ("什么是幂等性，接口设计上怎么保证幂等", "unknown"),
    ("Java 的 HashMap 在并发场景下有什么问题", "unknown"),
    ("协程和线程有什么区别", "unknown"),
)

# 排错题：命中「报错」「错误信息」「抛出异常」等特征，不触发检索
_DEBUGGING: tuple[tuple[str, str], ...] = (
    ("线上接口报错了，错误信息是 connection reset，怎么定位", "debugging"),
    ("这段代码抛出了空指针异常，堆栈信息看不出根因，怎么排查", "debugging"),
)

# 系统设计题：命中「设计一个」「一致性」等特征，不触发检索
_SYSTEM_DESIGN: tuple[tuple[str, str], ...] = (
    ("设计一个短链服务，需要考虑哪些点", "system_design"),
    ("什么是 CAP 定理，一致性和可用性怎么取舍", "system_design"),
)

# 代码题：命中「哈希表」「二分查找」「双指针」等特征，不触发检索
_CODING: tuple[tuple[str, str], ...] = (
    ("实现一个 LRU Cache，用哈希表加双向链表", "coding"),
    ("给定一个整数数组，用哈希表找出和为 target 的两个数并返回下标", "coding"),
    ("合并 K 个有序链表，用双指针返回一条有序链表", "coding"),
    ("判断一个字符串是否回文，用双指针，忽略大小写与标点", "coding"),
    ("二分查找第一个大于等于目标值的位置，数组有序", "coding"),
)

# 项目题：命中项目类关键词，触发 Dify 检索。
# 问的都是仓库文档里有的内容，检索能命中实际片段。
# 措辞必须落进 PROJECT_KEYWORDS 的那几种写法。
_PROJECT: tuple[tuple[str, str], ...] = (
    ("这个项目是怎么做请求去重的", "project"),
    ("这个项目的检索链路是怎么设计的", "project"),
    ("我之前的项目的数据库表结构是怎么设计的", "project"),
    ("项目中怎么实现检索降级", "project"),
    ("这个项目的语音输入链路是怎么实现的", "project"),
    ("我之前的项目的错误码是怎么定义的", "project"),
    ("我们用的数据库表结构是怎么设计的", "project"),
)


def _build() -> tuple[BenchQuestion, ...]:
    """把四组题目拼成一条有序清单。

    顺序固定：概念、排错、系统设计、代码、项目。
    报告里按这个顺序逐条打印，便于两次运行之间逐行对照。
    """
    items: list[BenchQuestion] = []
    for text, expected_type in _CONCEPTUAL:
        items.append(BenchQuestion(text=text, expected_type=expected_type, expect_retrieval=False))
    for text, expected_type in _DEBUGGING:
        items.append(BenchQuestion(text=text, expected_type=expected_type, expect_retrieval=False))
    for text, expected_type in _SYSTEM_DESIGN:
        items.append(BenchQuestion(text=text, expected_type=expected_type, expect_retrieval=False))
    for text, expected_type in _CODING:
        items.append(BenchQuestion(text=text, expected_type=expected_type, expect_retrieval=False))
    for text, expected_type in _PROJECT:
        items.append(BenchQuestion(text=text, expected_type=expected_type, expect_retrieval=True))
    return tuple(items)


BENCH_QUESTIONS: tuple[BenchQuestion, ...] = _build()
