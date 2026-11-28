"""本地假流客户端。

``providers.llm`` 取 ``mock`` 时用这个客户端：不发任何网络请求，
按《3-文本输入与流式回答》流式事件协议一节产出本地假流。

产出格式与真实客户端一致：逐段 yield 增量文本，把这些片段拼起来就是
一行一个 JSON 的事件流（``framework`` / ``item`` / ``section_done`` / ``done``）。
片段长度取 10 到 40 个字符，用来模拟真实的增量流，
片段之间 ``await asyncio.sleep(0)`` 让出事件循环。

代码题请求走代码题的事件流（``framework`` / ``code_delta`` / ``code_complete``），
见《7-代码问题模式》流式事件一节。是不是代码题请求只看 system 文本：
先按代码题 Prompt 里的固定标记 :data:`~app.llm.prompt_coding.CODING_REQUEST_MARKER` 判定，
标记没命中时再看 system 里有没有独立的 mode 与 coding 两个词，
两个词按词边界匹配，model 这类子串不算命中。
判定不按题型字符串判断。

方法签名与 :class:`~app.llm.openai_compat_client.OpenAICompatClient` 一致，
上层按 ``providers.llm`` 切换实现时不用改调用方式。
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, AsyncIterator

from app.config import AppConfig
from app.llm.openai_compat_client import ModelCapability
from app.llm.prompt_coding import CODING_REQUEST_MARKER, detect_submission_mode

__all__ = ["MockClient"]

# 每个增量片段的长度范围
_MIN_CHUNK = 10
_MAX_CHUNK = 40

# 提问前缀，摘掉之后剩下的才是主题
_QUESTION_PREFIXES = (
    "请解释一下",
    "介绍一下",
    "请解释",
    "请介绍",
    "解释一下",
    "讲讲",
    "说说",
    "解释",
    "介绍",
)

# 不算主题词的常见问法
_STOP_WORDS = frozenset(
    {
        "请问",
        "以及",
        "并且",
        "如何",
        "什么",
        "为什么",
        "怎么",
        "哪些",
        "有哪些",
        "一个",
        "这个",
        "那个",
        "可以",
        "需要",
        "分析",
        "阐述",
        "描述",
    }
)

# 表达单元用的固定示例句，见《3-文本输入与流式回答》普通问题的 Prompt 一节
_EXAMPLE_ITEMS: tuple[tuple[str, str], ...] = (
    ("资源管理", "把资源的申请与释放绑定到对象生命周期上，避免遗漏手动释放。"),
    ("统一封装", "用一层薄封装屏蔽底层差异，调用方只关心使用方式。"),
    ("边界约束", "先说清限制条件与取舍，再给出适用范围。"),
    ("落地步骤", "按顺序讲清楚实现步骤，每一步都对应一个具体动作。"),
)

# 代码题事件的 mode 值，见《7-代码问题模式》流式事件一节
_CODING_MODE = "coding"

# 代码题代码的语言与标准，见《7-代码问题模式》右侧完整代码一节
_CODING_LANGUAGE = "cpp"
_CODING_STANDARD = "cpp17"

# 代码题请求的固定标记，放在 app/llm/prompt_coding.py 里，见该模块的 CODING_REQUEST_MARKER
# 标记按字面量匹配，大小写敏感；标记两侧紧贴其他单词字符时不算命中
_CODING_MARKER_PATTERN = re.compile(
    rf"(?<!\w){re.escape(CODING_REQUEST_MARKER)}(?!\w)"
)

# 兜底判定用的两个词，按词边界匹配，
# 避免 model 里的 mode 这类子串误命中
_CODING_FALLBACK_PATTERNS = (
    re.compile(r"\bmode\b", re.IGNORECASE),
    re.compile(r"\bcoding\b", re.IGNORECASE),
)

# 一次 code_delta 事件带几行代码
_CODE_DELTA_LINES = 4

# 题面文本前面的标签，剥掉之后剩下的才是主题
_QUESTION_LABELS = ("题目：", "题目:", "问题：", "问题:")

# 题面里的常见问法前缀，剥掉之后剩下的才是主题
_CODING_TOPIC_PREFIXES = (
    "给定",
    "实现",
    "写一个",
    "找出",
    "计算",
    "设计",
    "求",
    "写",
)

# 解题步骤的后缀与提示，见《7-代码问题模式》左侧解题思路一节
_CODING_STEP_HINTS: dict[str, str] = {
    "题意梳理": "先说清输入、输出与约束",
    "算法选型": "定下用哪种算法解决",
    "数据结构设计": "选好容器再动手写",
    "边界处理": "想清空输入与越界情况",
    "复杂度分析": "给出时间与空间开销",
}

# 题面特征到步骤后缀的映射，见《7-代码问题模式》左侧解题思路一节
_CODING_STEP_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "算法选型",
        ("动态规划", "dp", "贪心", "二分", "回溯", "递归", "排序", "滑动窗口", "前缀和"),
    ),
    (
        "数据结构设计",
        ("哈希", "unordered", "map", "set", "vector", "数组", "链表", "堆", "队列", "栈"),
    ),
    ("边界处理", ("边界", "范围", "溢出", "越界", "负数", "极大", "最小", "最大")),
)

# 三种提交格式下的完整实现，见《7-代码问题模式》右侧完整代码一节
_CODING_CODE_LINES: dict[str, tuple[str, ...]] = {
    "leetcode": (
        "#include <bits/stdc++.h>",
        "using namespace std;",
        "",
        "class Solution {",
        "public:",
        "    int maxSubarraySum(vector<int>& nums) {",
        "        if (nums.empty()) {",
        "            return 0;",
        "        }",
        "        int best = nums[0];",
        "        int cur = nums[0];",
        "        for (int i = 1; i < static_cast<int>(nums.size()); ++i) {",
        "            cur = max(nums[i], cur + nums[i]);",
        "            best = max(best, cur);",
        "        }",
        "        return best;",
        "    }",
        "};",
    ),
    "nowcoder": (
        "#include <bits/stdc++.h>",
        "using namespace std;",
        "",
        "int maxSubarraySum(vector<int>& nums) {",
        "    if (nums.empty()) {",
        "        return 0;",
        "    }",
        "    int best = nums[0];",
        "    int cur = nums[0];",
        "    for (int i = 1; i < static_cast<int>(nums.size()); ++i) {",
        "        cur = max(nums[i], cur + nums[i]);",
        "        best = max(best, cur);",
        "    }",
        "    return best;",
        "}",
    ),
    "acm": (
        "#include <bits/stdc++.h>",
        "using namespace std;",
        "",
        "int maxSubarraySum(vector<int>& nums) {",
        "    if (nums.empty()) {",
        "        return 0;",
        "    }",
        "    int best = nums[0];",
        "    int cur = nums[0];",
        "    for (int i = 1; i < static_cast<int>(nums.size()); ++i) {",
        "        cur = max(nums[i], cur + nums[i]);",
        "        best = max(best, cur);",
        "    }",
        "    return best;",
        "}",
        "",
        "int main() {",
        "    ios::sync_with_stdio(false);",
        "    cin.tie(nullptr);",
        "",
        "    int n = 0;",
        "    if (!(cin >> n)) {",
        "        return 0;",
        "    }",
        "    vector<int> nums(n);",
        "    for (int i = 0; i < n; ++i) {",
        "        cin >> nums[i];",
        "    }",
        "    cout << maxSubarraySum(nums) << '\\n';",
        "    return 0;",
        "}",
    ),
}


class MockClient:
    """本地假流客户端，不发网络请求。"""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        # 被取消的 request_id，取消后不再产出内容
        self._cancelled: set[str] = set()

    async def chat_stream(
        self, messages: list[dict[str, str]], request_id: str
    ) -> AsyncIterator[str]:
        """产出本地假流，逐段 yield 增量文本。

        代码题请求产出代码题的事件流，其余走普通题的事件流，
        见《7-代码问题模式》流式事件一节。
        """
        question = _extract_question(messages)
        if _is_coding_request(messages):
            text = self._render_coding_text(question)
        else:
            text = self._render_text(question)
        for chunk in _split_chunks(text):
            if request_id in self._cancelled:
                return
            await asyncio.sleep(0)
            yield chunk

    async def multimodal_stream(
        self, messages: list[dict], request_id: str
    ) -> AsyncIterator[str]:
        """多模态版本的本地假流，产出内容与文本版一致。

        是不是代码题请求同样只看 system 文本，代码题带截图时也走代码题的事件流。
        """
        question = _extract_question(messages)
        if _is_coding_request(messages):
            text = self._render_coding_text(question)
        else:
            text = self._render_text(question)
        for chunk in _split_chunks(text):
            if request_id in self._cancelled:
                return
            await asyncio.sleep(0)
            yield chunk

    async def cancel(self, request_id: str) -> None:
        """取消指定请求，之后不再产出内容。

        幂等，对已结束或从未开始的 request_id 调用都不报错。
        """
        self._cancelled.add(request_id)

    async def check_health(self) -> ModelCapability:
        """假流没有真实能力，返回一个支持图片的能力对象。"""
        return ModelCapability(
            model_id="mock",
            supports_image=True,
            context_window=None,
            max_output_tokens=None,
        )

    async def aclose(self) -> None:
        """没有连接要关，空实现。"""

    def _render_text(self, question: str) -> str:
        """把事件流渲染成一行一个 JSON 的文本。"""
        return _render_events(_build_events(question))

    def _render_coding_text(self, question: str) -> str:
        """把代码题的事件流渲染成一行一个 JSON 的文本。"""
        return _render_events(_build_coding_events(question))


def _render_events(events: list[dict[str, Any]]) -> str:
    """把事件流渲染成一行一个 JSON 的文本。"""
    lines = [
        json.dumps(event, ensure_ascii=False)
        for event in events
    ]
    return "".join(f"{line}\n" for line in lines)


def _build_events(question: str) -> list[dict[str, Any]]:
    """按流式事件协议拼出完整的事件列表。

    顺序是 framework、逐层的 item 与 section_done、最后的 done，
    见《3-文本输入与流式回答》流式事件协议一节。
    """
    titles = _framework_titles(_extract_topic(question))
    sections = [
        {"id": f"s{index + 1}", "title": title} for index, title in enumerate(titles)
    ]
    events: list[dict[str, Any]] = [
        {"type": "framework", "mode": "normal", "sections": sections}
    ]
    for offset, section in enumerate(sections):
        for keyword, sentence in _items_for(offset):
            events.append(
                {
                    "type": "item",
                    "section_id": section["id"],
                    "keyword": keyword,
                    "sentence": sentence,
                }
            )
        events.append({"type": "section_done", "section_id": section["id"]})
    events.append({"type": "done"})
    return events


def _build_coding_events(question: str) -> list[dict[str, Any]]:
    """按代码题的事件协议拼出完整的事件列表。

    顺序是 framework、若干 code_delta、最后的 code_complete，
    见《7-代码问题模式》流式事件一节。
    """
    submission_mode = detect_submission_mode(question)
    events: list[dict[str, Any]] = [
        {
            "type": "framework",
            "mode": _CODING_MODE,
            "steps": _coding_steps(question, _coding_topic(question)),
        }
    ]
    events.extend(_coding_code_deltas(submission_mode))
    events.append(
        {
            "type": "code_complete",
            "language": _CODING_LANGUAGE,
            "standard": _CODING_STANDARD,
            "submission_mode": submission_mode,
        }
    )
    return events


def _framework_titles(topic: str) -> list[str]:
    """按主题生成三层框架标题。

    标题跟着题目走，不拆成固定的背景、核心内容、项目拓展。
    """
    return [f"{topic} 基本原理", f"{topic} 实现要点", f"{topic} 工程约束"]


def _items_for(offset: int) -> list[tuple[str, str]]:
    """取一层的表达单元，用固定示例句，每层三个。"""
    count = 3
    return [
        _EXAMPLE_ITEMS[(offset + index) % len(_EXAMPLE_ITEMS)]
        for index in range(count)
    ]


def _coding_steps(question: str, topic: str) -> list[dict[str, str]]:
    """按题面生成解题步骤，数量落在 2 到 5 之间。

    标题跟着题目走，见《7-代码问题模式》左侧解题思路一节。
    """
    lowered = question.lower()
    suffixes = ["题意梳理"]
    for suffix, words in _CODING_STEP_SIGNALS:
        if any(word in lowered for word in words):
            suffixes.append(suffix)
    suffixes.append("复杂度分析")
    return [
        {
            "id": f"p{index + 1}",
            "title": f"{topic} {suffix}",
            "hint": _CODING_STEP_HINTS[suffix],
        }
        for index, suffix in enumerate(suffixes[:5])
    ]


def _coding_code_deltas(submission_mode: str) -> list[dict[str, str]]:
    """把完整实现切成多段 code_delta 事件。

    段按行分组，段与段拼起来正好是完整实现，
    见《7-代码问题模式》右侧完整代码一节。
    """
    lines = _CODING_CODE_LINES[submission_mode]
    deltas: list[dict[str, str]] = []
    for start in range(0, len(lines), _CODE_DELTA_LINES):
        block = lines[start : start + _CODE_DELTA_LINES]
        deltas.append({"type": "code_delta", "text": "\n".join(block) + "\n"})
    return deltas


def _extract_topic(question: str) -> str:
    """从问题里取主题词，用于生成随题目变化的框架标题。

    优先取英文词（长度不小于 2），没有就取最长的一段中文。
    """
    text = question.strip()
    for prefix in _QUESTION_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    candidates: list[str] = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        if len(word) >= 2 and word.lower() not in _STOP_WORDS:
            candidates.append(word)
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        run = run.lstrip("的")
        if len(run) >= 2 and run not in _STOP_WORDS:
            candidates.append(run[:6])
    if not candidates:
        return "问题"
    latin = [word for word in candidates if word.isascii()]
    if latin:
        return max(latin, key=len)
    return max(candidates, key=len)


def _coding_topic(question: str) -> str:
    """从题面里取主题词，用于生成随题目变化的解题步骤标题。

    只看题面的第一行，截图说明之类的尾巴不算主题。
    """
    text = _strip_prefix((question or "").strip(), _QUESTION_LABELS)
    text = _strip_prefix(text, _QUESTION_PREFIXES)
    text = _strip_prefix(text, _CODING_TOPIC_PREFIXES)
    first_line = text.splitlines()[0].strip() if text else ""
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", first_line):
        topic = run.lstrip("的")
        if len(topic) >= 2:
            return topic[:8]
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", first_line):
        if len(word) >= 2 and word.lower() not in _STOP_WORDS:
            return word
    return "问题"


def _strip_prefix(text: str, prefixes: tuple[str, ...]) -> str:
    """剥掉开头最长的那个前缀，没有匹配就原样返回。"""
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _is_coding_request(messages: list[dict]) -> bool:
    """判断这次请求是不是代码题请求。

    先按固定标记判：system 文本里出现代码题 Prompt 的标记
    :data:`~app.llm.prompt_coding.CODING_REQUEST_MARKER` 就是代码题请求，
    标记大小写敏感。

    标记没命中时走兜底：system 文本里同时出现独立的 mode 与 coding 两个词
    就算代码题请求。两个词按词边界匹配，model 里的 mode 不算命中。

    判定只看 system 消息的文本内容，不按题型字符串判断。
    """
    for message in messages:
        if message.get("role") != "system":
            continue
        text = _message_text(message)
        if _CODING_MARKER_PATTERN.search(text):
            return True
        if all(pattern.search(text) for pattern in _CODING_FALLBACK_PATTERNS):
            return True
    return False


def _message_text(message: dict) -> str:
    """取一条消息的文本内容，content 是数组时取里面的 text 块。"""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _extract_question(messages: list[dict]) -> str:
    """从消息里取用户问题，多模态时把文字块拼起来。"""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return " ".join(texts)
    return ""


def _split_chunks(text: str) -> list[str]:
    """把文本切成 10 到 40 个字符的片段，模拟增量流。

    切点用固定种子，长度稳定，方便复现。
    """
    rng = random.Random(0)
    chunks: list[str] = []
    index = 0
    while index < len(text):
        remaining = len(text) - index
        size = rng.randint(_MIN_CHUNK, _MAX_CHUNK)
        if remaining - size < _MIN_CHUNK:
            # 剩下的不足 10 个字符，调整块长，避免出现过短的尾块
            size = remaining if remaining <= _MAX_CHUNK else remaining - _MIN_CHUNK
        chunks.append(text[index : index + size])
        index += size
    return chunks
