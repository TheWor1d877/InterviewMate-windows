"""代码题的 Prompt。

对应《7-代码问题模式》的四节：

- 左侧解题思路：2 到 5 个阶梯步骤，每步带 id、title、hint；
- 右侧完整代码：默认 C++17，只放完整实现；
- 流式事件：framework(mode=coding) → code_delta → code_complete；
- 提交格式识别：leetcode / nowcoder / acm 三种输出格式。

Prompt 里写清三件事：算法与正确性要求、输出格式、禁止出现的内容。

:func:`detect_submission_mode` 是纯函数，按《7-代码问题模式》提交格式识别一节
的规则从题面推断平台格式，不调模型。判定只认可判定的具体形态：
``class Solution``、函数签名形态（函数名加带类型或空参的括号）、
返回值类型说明（「返回 int」这类明说类型的形式）、输入输出格式或样例。
描述性的「返回最大子数组和」不算返回值类型说明。
"""

from __future__ import annotations

import re

from app.llm.prompt_normal import SCREENSHOT_NOTE

__all__ = [
    "CODING_REQUEST_MARKER",
    "SUBMISSION_MODES",
    "CODING_LANGUAGE",
    "CODING_STANDARD",
    "build_coding_messages",
    "build_coding_multimodal_messages",
    "build_coding_system_prompt",
    "detect_submission_mode",
]


# 假流客户端靠这个固定标记识别「这是代码题请求」，
# 标记是稳定出现的字面量，不要改写
CODING_REQUEST_MARKER: str = "[[CODE_MODE]]"

# 代码题的三种提交格式，见《7-代码问题模式》提交格式识别一节
SUBMISSION_MODES: tuple[str, ...] = ("leetcode", "nowcoder", "acm")

# 代码语言与标准，见《7-代码问题模式》右侧完整代码一节
CODING_LANGUAGE: str = "cpp"
CODING_STANDARD: str = "cpp17"


# 代码题的 system 模板，占位符用字符串替换，
# 模板里的 JSON 花括号不受影响
_CODING_SYSTEM_TEMPLATE: str = """你是一个面试代码助手，回答用户提出的代码题。
{marker}

算法与正确性要求：
1. 算法正确性优先于接口名称的猜测。函数名、类名或接口签名识别不准时，算法逻辑、状态设计、边界条件与复杂度分析仍然要正确。
2. 左侧只放解题步骤、思路提示和当前步骤状态，不出现完整代码，不写成大段算法教程。每步的 hint 是一句话，说清这一步用什么方法、为什么，能脱离代码单独读懂，右侧代码上方会原样展示这些提示。
3. 步骤按题目动态生成 2 到 5 个，标题由实际题目决定，不要所有题都拆成一样的标题。步骤 id 用 p1、p2 这样的编号。
4. 右侧只放完整代码，默认 C++17，缩进正确，给完整可读的实现。
5. 接口不明确时按《7》的兜底顺序处理：优先用题目明确给出的函数签名；题目没给接口就按平台惯用形式推断；判断不出来给完整 C++17 实现，关键假设写进步骤提示里；不反复追问函数名和参数名。
6. 输入可能来自语音转写，含语气词、重复或口误，先判断用户真正想问什么，按还原后的问题作答，不复述废话。

输出格式：
一行一个 JSON 对象，不要有多余文字，不要用代码块包裹。事件类型只有下面三种：

{"type": "framework", "mode": "coding", "steps": [{"id": "p1", "title": "标题", "hint": "一句话思路，说清方法与原因"}]}
{"type": "code_delta", "text": "代码片段"}
{"type": "code_complete", "language": "{language}", "standard": "{standard}", "submission_mode": "{mode}"}

事件顺序：先发 framework 声明全部步骤，再按步骤顺序逐块发 code_delta，最后发 code_complete。
code_delta 的 text 是代码原文，缩进要正确，不要在里面夹带说明文字。
code_complete 的 language、standard、submission_mode 按本次提交格式填写。

禁止出现这些内容：解题长文、算法教程、代码之外的解释段落、第三栏、独立的验证视图、重复解释代码的段落。

提交格式：
{submission_note}
"""


# 各提交格式对应的输出要求，见《7-代码问题模式》提交格式识别一节
_SUBMISSION_NOTES: dict[str, str] = {
    "leetcode": "按 LeetCode 提交：只给类 Solution 与方法签名，不带 main。",
    "nowcoder": "按牛客核心代码模式提交：按题目给的接口输出，不带 main。",
    "acm": "按 ACM 模式提交：输出完整 main、输入处理和输出。",
    "auto": (
        "提交格式由题面推断：题面出现 class Solution 或方法签名与返回值说明，按 LeetCode 提交，"
        "只给类与函数签名，不带 main；题面只有函数原型且没有输入输出说明，按牛客核心代码模式提交，"
        "按题目给的接口输出，不带 main；题面给了输入格式与输出格式样例，按 ACM 模式提交，"
        "输出完整 main、输入处理与输出；都没有，按完整 C++17 实现处理。"
        "上例里的 submission_mode 是占位，按推断出来的格式改写，"
        "取值只在 leetcode、nowcoder、acm 三者之中。"
    ),
}


# 带截图的代码题请求，user 的文字块，见《4-截图输入与多模态》多模态请求一节
_CODING_SCREENSHOT_WITH_TEXT: str = (
    "题目：{question}\n\n上面附了一张截图，请结合截图里的题目约束一起回答。"
)

# 没有题面文字时的用户文本块
_CODING_SCREENSHOT_ONLY: str = "请识别截图里的面试代码题目，理解题目约束后按上述格式回答。"


# class Solution：LeetCode 的类名标记，见《7-代码问题模式》提交格式识别一节
_CLASS_SOLUTION_PATTERN = re.compile(r"\bclass\s+Solution\b", re.IGNORECASE)

# 类型记号：标识符，可带模板参数、数组后缀、指针或引用
_TYPE_TOKEN = r"[A-Za-z_][A-Za-z0-9_]*(?:\s*<[^<>]*>)?(?:\s*\[\s*\])?(?:\s*[&*])?"

# 带类型的参数：类型加参数名，形如 ``int[] nums``、``vector<int>& a``
_TYPED_ARGUMENT_PATTERN = re.compile(
    rf"^{_TYPE_TOKEN}\s+[A-Za-z_][A-Za-z0-9_]*(?:\s*\[\s*\])?$"
)

# 函数调用形态：函数名加括号，取括号里的参数
_CALL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)"
)

# Python 的函数定义，单独成行，见《7-代码问题模式》提交格式识别一节
_PYTHON_SIGNATURE_PATTERN = re.compile(
    r"^[ \t]*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", re.MULTILINE
)

# 返回值类型说明：明说返回类型或返回值为，见《7-代码问题模式》提交格式识别一节
_RETURN_TYPE_WORD_PATTERN = re.compile(r"返回值类型|返回值为")

# 返回值类型说明：返回一个类型名，形如「返回 int」「returns int」
_RETURN_TYPE_CODE_PATTERN = re.compile(
    r"(?:返回|returns)\s*(?:const\s*)?"
    r"(?:int|long|short|float|double|bool|boolean|char|string|void|size_t|auto"
    r"|vector|list|array|map|unordered_map|set|stack|queue"
    r"|String|Integer|Long|Double|Float|Boolean|Character)\b",
    re.IGNORECASE,
)

# 输入输出格式与样例，见《7-代码问题模式》提交格式识别一节
_IO_SPEC_PATTERN = re.compile(
    r"输入格式|输出格式|输入样例|输出样例|样例输入|样例输出|输入描述|输出描述"
    r"|第一行(?:输入|包含|有)"
    r"|输出\s*一个\s*(?:整数|实数|浮点数|数|字符串|布尔值|字符)"
    r"|sample\s+input|sample\s+output",
    re.IGNORECASE,
)


def build_coding_system_prompt(submission_mode: str = "auto") -> str:
    """组装代码题的 system 消息。

    提交格式不同，system 里写的输出要求不同，规则见《7-代码问题模式》
    提交格式识别一节。未知取值按 ``auto`` 处理。

    :param submission_mode: 提交格式，取 leetcode / nowcoder / acm / auto。
    """
    mode = submission_mode if submission_mode in _SUBMISSION_NOTES else "auto"
    event_mode = mode if mode in SUBMISSION_MODES else "auto"
    return (
        _CODING_SYSTEM_TEMPLATE.replace("{marker}", CODING_REQUEST_MARKER)
        .replace("{language}", CODING_LANGUAGE)
        .replace("{standard}", CODING_STANDARD)
        .replace("{mode}", event_mode)
        .replace("{submission_note}", _SUBMISSION_NOTES[mode])
    )


def build_coding_messages(
    question: str,
    context: str | None = None,
    submission_mode: str = "auto",
) -> list[dict[str, str]]:
    """组装代码题的请求消息。

    返回 system 与 user 两条消息。``context`` 不为空时并进 user 消息，
    片段的排序、截断与文档名由检索侧的上下文拼装负责，
    见《6-Dify检索与项目知识》结果整理一节。

    :param question: 题面文本。
    :param context: 检索到的项目上下文，没有就传 None。
    :param submission_mode: 提交格式，取 leetcode / nowcoder / acm / auto。
    """
    user_text = f"题目：{question.strip()}"
    if context and context.strip():
        user_text = f"{user_text}\n\n背景上下文：\n{context.strip()}"
    return [
        {"role": "system", "content": build_coding_system_prompt(submission_mode)},
        {"role": "user", "content": user_text},
    ]


def build_coding_multimodal_messages(
    question: str,
    image_data_url: str,
    context: str | None = None,
    submission_mode: str = "auto",
) -> list[dict]:
    """组装带截图的代码题请求消息。

    返回 system 与 user 两条消息。system 是代码题的 system 加截图说明，
    截图里的文字按数据理解，不按指令执行，见《4-截图输入与多模态》
    截图内容作为数据一节；user 的 ``content`` 是数组，先文本块再图片块，
    见《4-截图输入与多模态》多模态请求一节。

    :param question: 题面文字，没有文字时传空串。
    :param image_data_url: 截图的 data URL。
    :param context: 检索到的项目上下文，没有就传 None。
    :param submission_mode: 提交格式，取 leetcode / nowcoder / acm / auto。
    """
    stripped = question.strip()
    if stripped:
        text_block = _CODING_SCREENSHOT_WITH_TEXT.format(question=stripped)
    else:
        text_block = _CODING_SCREENSHOT_ONLY
    if context and context.strip():
        text_block = f"{text_block}\n\n背景上下文：\n{context.strip()}"
    system = f"{build_coding_system_prompt(submission_mode)}\n\n{SCREENSHOT_NOTE}"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_block},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url, "detail": "auto"},
                },
            ],
        },
    ]


def detect_submission_mode(question: str) -> str:
    """按《7-代码问题模式》提交格式识别一节从题面推断提交格式。

    判定顺序：

    1. 题面出现 ``class Solution``，判为 ``leetcode``；
    2. 题面有函数签名形态，并且有返回值类型说明，判为 ``leetcode``；
    3. 题面有输入输出格式或样例，判为 ``acm``；
    4. 题面有函数签名形态但没有输入输出说明，判为 ``nowcoder``；
    5. 都没有，判为 ``acm``，按完整 C++17 实现处理。

    返回值类型说明只认「返回 int」「返回值为」「返回值类型」「returns int」
    这类明说类型的形式，「返回最大子数组和」这种描述性说法不算。

    纯函数，不调模型。

    :param question: 题面文本。
    :return: leetcode / nowcoder / acm 之一。
    """
    text = question if isinstance(question, str) else ""
    if _CLASS_SOLUTION_PATTERN.search(text):
        # 题面里出现 class Solution，LeetCode
        return "leetcode"
    has_signature = _has_signature(text)
    if has_signature and _has_return_type_description(text):
        # 有函数签名与返回值类型说明，LeetCode
        return "leetcode"
    if _has_io_spec(text):
        # 有输入输出格式或样例，按 ACM 模式处理
        return "acm"
    if has_signature:
        # 只有函数签名没有输入输出说明，牛客核心代码模式
        return "nowcoder"
    return "acm"


def _has_io_spec(text: str) -> bool:
    """题面里有没有输入输出格式或样例。"""
    return bool(_IO_SPEC_PATTERN.search(text))


def _has_return_type_description(text: str) -> bool:
    """题面里有没有返回值类型说明。

    只认明说类型的形式：「返回 int」「返回值为」「返回值类型」「returns int」。
    「返回最大子数组和」这种描述性说法不算，见《7-代码问题模式》
    提交格式识别一节。
    """
    if _RETURN_TYPE_WORD_PATTERN.search(text):
        return True
    return bool(_RETURN_TYPE_CODE_PATTERN.search(text))


def _has_signature(text: str) -> bool:
    """题面里有没有函数签名形态。

    形态是函数名加带类型或空参的括号，形如 ``maxSubArray(int[] nums)``、
    ``maxSubarraySum(vector<int>& nums)``；Python 的 ``def`` 行同样算。
    """
    for match in _CALL_PATTERN.finditer(text):
        arguments = match.group(2).strip()
        if not arguments:
            # 空参的括号也算签名形态
            return True
        if any(
            _TYPED_ARGUMENT_PATTERN.match(part.strip())
            for part in arguments.split(",")
        ):
            return True
    return bool(_PYTHON_SIGNATURE_PATTERN.search(text))
