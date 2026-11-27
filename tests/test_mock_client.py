"""本地假流客户端的测试。

覆盖普通题的事件流（framework / item / section_done / done），
以及代码题的事件流（framework(mode=coding) / code_delta / code_complete）。
判定与产出规则见《3-文本输入与流式回答》流式事件协议一节
与《7-代码问题模式》流式事件一节、提交格式识别一节。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import app.config
from app.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from app.llm.mock_client import MockClient
from app.llm.prompt_coding import CODING_REQUEST_MARKER

# 普通题的 system：只有 mode，没有 coding
_NORMAL_SYSTEM = (
    "你是一个面试回答助手。\n"
    '{"type": "framework", "mode": "normal", "sections": [{"id": "s1", "title": "标题"}]}\n'
)

# 代码题的 system：mode 与 coding 都出现
_CODING_SYSTEM = (
    "你是一个面试代码助手。\n"
    '{"type": "framework", "mode": "coding", "steps": [{"id": "p1", "title": "标题"}]}\n'
    '{"type": "code_delta", "text": "代码片段"}\n'
)

# 代码题的 system：带固定标记，除了标记没有别的判定信号
_MARKER_SYSTEM = (
    f"你是一个面试代码助手。\n{CODING_REQUEST_MARKER}\n"
    '{"type": "framework", "steps": [{"id": "p1", "title": "标题"}]}\n'
)

# 固定标记被改成小写，标记大小写敏感，不该命中
_LOWERCASE_MARKER_SYSTEM = (
    f"你是一个面试代码助手。\n{CODING_REQUEST_MARKER.lower()}\n"
    '{"type": "framework", "sections": [{"id": "s1", "title": "标题"}]}\n'
)

# 普通题的 system：里面只有 model 这个词，没有标记也没有独立的 coding
_MODEL_SYSTEM = (
    "你是一个面试回答助手，使用 model deepseek-chat 回答用户提出的普通问题。\n"
    '{"type": "framework", "mode": "normal", "sections": [{"id": "s1", "title": "标题"}]}\n'
)

# 兜底判定的 system：没有标记，但有独立的 mode 与 coding 两个词
_FALLBACK_SYSTEM = (
    "你是一个面试代码助手。\n"
    '{"type": "framework", "mode": "coding", "steps": [{"id": "p1", "title": "标题"}]}\n'
)

# LeetCode 题面：出现 class Solution
_LEETCODE_QUESTION = (
    "题目：给定一个整数数组 nums 和一个目标值 target，"
    "请使用哈希表找出和为 target 的两个整数并返回它们的下标。\n"
    "class Solution {\n"
    "public:\n"
    "    vector<int> twoSum(vector<int>& nums, int target);\n"
    "};"
)

# ACM 题面：给了输入格式与输出格式样例
_ACM_QUESTION = (
    "题目：给定一个整数数组，找出最大子数组和。\n"
    "输入格式：第一行一个整数 n，第二行 n 个整数。\n"
    "输出格式：一个整数，表示最大子数组和。\n"
    "样例输入：\n3\n-1 2 3\n"
    "样例输出：\n4"
)

# 牛客题面：只给了函数原型
_NOWCODER_QUESTION = (
    "题目：实现最大子数组和函数。\n"
    "函数原型：\n"
    "int maxSubarraySum(vector<int>& nums);"
)

# ACMICPC 风格题面：有函数签名、有描述性的返回值说法、也有输入输出格式样例
# 描述性的「返回最大子数组和」不算返回值类型说明，判为 acm
_ACMICPC_QUESTION = (
    "题目：实现最大子数组和。\n"
    "函数签名：int maxSubarraySum(vector<int>& nums)\n"
    "返回最大子数组和。\n"
    "输入格式：第一行一个整数 n，第二行 n 个整数。\n"
    "输出格式：一个整数，表示最大子数组和。\n"
    "样例输入：\n3\n-1 2 3\n"
    "样例输出：\n4"
)


@pytest.fixture
def app_config(monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    """加载仓库默认配置，屏蔽可能存在的环境变量覆盖。"""
    for name in ("DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app.config, "DEFAULT_ENV_FILE", Path("/nonexistent/.env"))
    return load_config(DEFAULT_CONFIG_PATH)


def _messages(system: str, question: str) -> list[dict[str, str]]:
    """拼出文本请求的消息。"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _multimodal_messages(system: str, question: str) -> list[dict]:
    """拼出带截图的请求消息。"""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA", "detail": "auto"},
                },
            ],
        },
    ]


def _events(text: str) -> list[dict]:
    """把假流文本解析成事件列表。"""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _code_text(events: list[dict]) -> str:
    """把所有 code_delta 的文本拼起来。"""
    return "".join(
        event["text"] for event in events if event["type"] == "code_delta"
    )


async def _collect(client: MockClient, messages: list[dict]) -> str:
    """收集一次文本假流的全部片段。"""
    chunks = [chunk async for chunk in client.chat_stream(messages, "请求一")]
    return "".join(chunks)


async def _collect_multimodal(client: MockClient, messages: list[dict]) -> str:
    """收集一次多模态假流的全部片段。"""
    chunks = [chunk async for chunk in client.multimodal_stream(messages, "请求二")]
    return "".join(chunks)


# ----------------------------------------------------------------------
# 普通题的事件流，见《3-文本输入与流式回答》流式事件协议一节
# ----------------------------------------------------------------------
async def test_normal_request_keeps_normal_events(app_config: AppConfig) -> None:
    """system 里没有 coding 时仍然产出普通题的事件流。"""
    client = MockClient(app_config)

    text = await _collect(
        client, _messages(_NORMAL_SYSTEM, "解释一下 shared_ptr 的引用计数机制")
    )

    events = _events(text)
    types = [event["type"] for event in events]
    assert events[0]["type"] == "framework"
    assert events[0]["mode"] == "normal"
    assert "sections" in events[0]
    assert "item" in types
    assert "section_done" in types
    assert types[-1] == "done"
    assert "code_delta" not in types
    assert "code_complete" not in types


async def test_normal_request_keeps_chunk_size(app_config: AppConfig) -> None:
    """片段长度仍取 10 到 40 个字符，拼起来是完整的事件流。"""
    client = MockClient(app_config)

    messages = _messages(_NORMAL_SYSTEM, "解释一下索引失效")
    chunks = [chunk async for chunk in client.chat_stream(messages, "请求一")]

    assert len(chunks) > 1
    assert all(10 <= len(chunk) <= 40 for chunk in chunks)
    assert "".join(chunks).endswith("\n")


# ----------------------------------------------------------------------
# 代码题的事件流，见《7-代码问题模式》流式事件一节
# ----------------------------------------------------------------------
async def test_coding_request_produces_coding_events(app_config: AppConfig) -> None:
    """system 里同时有 mode 与 coding 时产出代码题的事件流。"""
    client = MockClient(app_config)

    text = await _collect(client, _messages(_CODING_SYSTEM, _LEETCODE_QUESTION))

    events = _events(text)
    types = [event["type"] for event in events]
    assert events[0]["type"] == "framework"
    assert events[0]["mode"] == "coding"
    assert events[0]["steps"]
    assert "code_delta" in types
    assert types[-1] == "code_complete"
    assert "item" not in types
    assert "section_done" not in types
    assert "done" not in types


async def test_coding_event_order(app_config: AppConfig) -> None:
    """事件顺序是 framework、若干 code_delta、code_complete。"""
    client = MockClient(app_config)

    text = await _collect(client, _messages(_CODING_SYSTEM, _ACM_QUESTION))

    types = [event["type"] for event in _events(text)]
    assert types[0] == "framework"
    assert types[-1] == "code_complete"
    assert set(types[1:-1]) == {"code_delta"}
    assert len(types) > 2


async def test_coding_steps_shape(app_config: AppConfig) -> None:
    """步骤数量在 2 到 5 之间，每步带 id、title、hint。"""
    client = MockClient(app_config)

    text = await _collect(client, _messages(_CODING_SYSTEM, _ACM_QUESTION))

    steps = _events(text)[0]["steps"]
    assert 2 <= len(steps) <= 5
    for index, step in enumerate(steps):
        assert step["id"] == f"p{index + 1}"
        assert step["title"]
        assert step["hint"]
        assert "\n" not in step["hint"]


async def test_coding_step_titles_follow_the_question(app_config: AppConfig) -> None:
    """步骤标题跟着题目走，换一道题标题就变。"""
    client = MockClient(app_config)

    first = await _collect(client, _messages(_CODING_SYSTEM, _ACM_QUESTION))
    second = await _collect(
        client, _messages(_CODING_SYSTEM, "题目：实现二叉树的中序遍历。")
    )

    first_titles = [step["title"] for step in _events(first)[0]["steps"]]
    second_titles = [step["title"] for step in _events(second)[0]["steps"]]
    assert first_titles != second_titles
    # 步骤标题带题目的主题词，不是一套固定标题
    assert first_titles[0].startswith("一个整数数组")
    assert second_titles[0].startswith("二叉树的中序遍历")
    assert first_titles[0].endswith("题意梳理")
    assert second_titles[0].endswith("题意梳理")


async def test_coding_code_deltas_form_complete_cpp(app_config: AppConfig) -> None:
    """code_delta 的文本拼起来是一份完整可读的 C++17 实现。"""
    client = MockClient(app_config)

    text = await _collect(client, _messages(_CODING_SYSTEM, _ACM_QUESTION))

    events = _events(text)
    deltas = [event for event in events if event["type"] == "code_delta"]
    assert len(deltas) > 1
    code = _code_text(events)
    assert "#include <bits/stdc++.h>" in code
    assert "using namespace std;" in code
    assert code.count("{") == code.count("}")
    assert code.count("(") == code.count(")")
    assert "int main(" in code
    assert code.endswith("}\n")


async def test_code_complete_fields(app_config: AppConfig) -> None:
    """code_complete 带 language、standard、submission_mode。"""
    client = MockClient(app_config)

    text = await _collect(client, _messages(_CODING_SYSTEM, _LEETCODE_QUESTION))

    complete = _events(text)[-1]
    assert complete["type"] == "code_complete"
    assert complete["language"] == "cpp"
    assert complete["standard"] == "cpp17"
    assert complete["submission_mode"] == "leetcode"


@pytest.mark.parametrize(
    ("question", "submission_mode"),
    [
        (_LEETCODE_QUESTION, "leetcode"),
        (_ACM_QUESTION, "acm"),
        (_NOWCODER_QUESTION, "nowcoder"),
    ],
)
async def test_submission_mode_follows_the_question(
    app_config: AppConfig, question: str, submission_mode: str
) -> None:
    """提交格式按题面判断，见《7-代码问题模式》提交格式识别一节。"""
    client = MockClient(app_config)

    text = await _collect(client, _messages(_CODING_SYSTEM, question))

    events = _events(text)
    assert events[-1]["submission_mode"] == submission_mode
    code = _code_text(events)
    if submission_mode == "leetcode":
        # LeetCode 提交给类与方法签名，不带 main
        assert "class Solution" in code
        assert "int main(" not in code
    elif submission_mode == "acm":
        # ACM 模式给完整 main、输入处理与输出
        assert "int main(" in code
        assert "class Solution" not in code
    else:
        # 牛客核心代码模式按接口输出，不带 main
        assert "class Solution" not in code
        assert "int main(" not in code


async def test_coding_request_detection_ignores_user_text(
    app_config: AppConfig,
) -> None:
    """判定只看 system 文本，用户消息里出现 coding 不算数。"""
    client = MockClient(app_config)

    text = await _collect(
        client, _messages(_NORMAL_SYSTEM, "解释一下 coding 是怎么做的")
    )

    events = _events(text)
    assert events[0]["mode"] == "normal"
    assert events[0]["type"] == "framework"


async def test_coding_request_detection_needs_both_words(
    app_config: AppConfig,
) -> None:
    """system 里只有 coding 一个词时不算代码题请求。"""
    client = MockClient(app_config)

    text = await _collect(
        client, _messages("请按 coding 的方式回答。", _ACM_QUESTION)
    )

    events = _events(text)
    assert events[0]["type"] == "framework"
    assert events[0]["mode"] == "normal"


# ----------------------------------------------------------------------
# 多模态路径，判定口径与文本路径一致
# ----------------------------------------------------------------------
async def test_multimodal_coding_request_produces_coding_events(
    app_config: AppConfig,
) -> None:
    """带截图的代码题请求同样按 system 内容判定。"""
    client = MockClient(app_config)

    text = await _collect_multimodal(
        client, _multimodal_messages(_CODING_SYSTEM, _ACM_QUESTION)
    )

    events = _events(text)
    types = [event["type"] for event in events]
    assert events[0]["mode"] == "coding"
    assert types[-1] == "code_complete"
    assert "code_delta" in types


async def test_multimodal_normal_request_keeps_normal_events(
    app_config: AppConfig,
) -> None:
    """带截图的普通题请求仍然是普通题的事件流。"""
    client = MockClient(app_config)

    text = await _collect_multimodal(
        client, _multimodal_messages(_NORMAL_SYSTEM, "解释一下索引失效")
    )

    events = _events(text)
    types = [event["type"] for event in events]
    assert events[0]["mode"] == "normal"
    assert types[-1] == "done"
    assert "code_delta" not in types


# ----------------------------------------------------------------------
# 取消，既有行为保持
# ----------------------------------------------------------------------
async def test_cancelled_request_produces_nothing(app_config: AppConfig) -> None:
    """取消之后不再产出内容。"""
    client = MockClient(app_config)
    await client.cancel("请求一")

    text = await _collect(client, _messages(_CODING_SYSTEM, _ACM_QUESTION))

    assert text == ""


async def test_cancel_stops_the_stream(app_config: AppConfig) -> None:
    """流已经开始后取消，剩余片段不再产出。"""
    client = MockClient(app_config)
    messages = _messages(_CODING_SYSTEM, _ACM_QUESTION)

    chunks: list[str] = []
    async for chunk in client.chat_stream(messages, "请求一"):
        chunks.append(chunk)
        await client.cancel("请求一")

    full = await _collect(
        MockClient(app_config), _messages(_CODING_SYSTEM, _ACM_QUESTION)
    )
    assert len(chunks) == 1
    assert "".join(chunks) != full


# ----------------------------------------------------------------------
# 生成的代码能被编译器接受
# ----------------------------------------------------------------------
_CPP_COMPILER = __import__(
    "app.platform.windows.compiler", fromlist=["detect_command"]
).detect_command()


@pytest.mark.skipif(_CPP_COMPILER is None, reason="环境里没有可用的 C++ 编译器")
async def test_generated_cpp_passes_syntax_check(
    app_config: AppConfig, tmp_path: Path
) -> None:
    """ACM 模式的完整实现能过本机 C++ 编译器（cl.exe 或 MinGW g++）的语法检查。"""
    client = MockClient(app_config)

    text = await _collect(client, _messages(_CODING_SYSTEM, _ACM_QUESTION))

    source = tmp_path / "main.cpp"
    source.write_text(_code_text(_events(text)), encoding="utf-8")
    compiler = _CPP_COMPILER or ""
    if os.path.basename(compiler).lower().startswith("cl"):
        command = [compiler, "/nologo", "/std:c++17", "/Zs", str(source)]
    else:
        command = [compiler, "-std=c++17", "-fsyntax-only", str(source)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------------------
# 请求判定：固定标记优先，词边界兜底
# ----------------------------------------------------------------------
async def test_marker_in_system_produces_coding_events(
    app_config: AppConfig,
) -> None:
    """system 里有固定标记时产出代码题的事件流。"""
    client = MockClient(app_config)

    text = await _collect(
        client, _messages(_MARKER_SYSTEM, _ACM_QUESTION)
    )

    events = _events(text)
    types = [event["type"] for event in events]
    assert events[0]["type"] == "framework"
    assert events[0]["mode"] == "coding"
    assert types[-1] == "code_complete"
    assert "code_delta" in types


async def test_system_with_model_word_keeps_normal_events(
    app_config: AppConfig,
) -> None:
    """system 里只有 model 这个词时不判成代码题请求。

    model 里含有 mode 子串，判定按词边界匹配，不该误命中。
    """
    client = MockClient(app_config)

    text = await _collect(
        client, _messages(_MODEL_SYSTEM, "解释一下索引失效")
    )

    events = _events(text)
    types = [event["type"] for event in events]
    assert events[0]["type"] == "framework"
    assert events[0]["mode"] == "normal"
    assert types[-1] == "done"
    assert "code_delta" not in types
    assert "code_complete" not in types


async def test_marker_is_case_sensitive(app_config: AppConfig) -> None:
    """标记大小写敏感，被改成小写就不算命中。"""
    client = MockClient(app_config)

    text = await _collect(
        client, _messages(_LOWERCASE_MARKER_SYSTEM, _ACM_QUESTION)
    )

    events = _events(text)
    assert events[0]["mode"] == "normal"
    assert events[0]["type"] == "framework"


async def test_fallback_words_still_produce_coding_events(
    app_config: AppConfig,
) -> None:
    """没有标记时，独立的 mode 与 coding 两个词仍然判成代码题请求。"""
    client = MockClient(app_config)

    text = await _collect(
        client, _messages(_FALLBACK_SYSTEM, _ACM_QUESTION)
    )

    events = _events(text)
    types = [event["type"] for event in events]
    assert events[0]["mode"] == "coding"
    assert types[-1] == "code_complete"


async def test_multimodal_marker_produces_coding_events(
    app_config: AppConfig,
) -> None:
    """多模态路径按同一个标记判定。"""
    client = MockClient(app_config)

    text = await _collect_multimodal(
        client, _multimodal_messages(_MARKER_SYSTEM, _ACM_QUESTION)
    )

    events = _events(text)
    types = [event["type"] for event in events]
    assert events[0]["mode"] == "coding"
    assert types[-1] == "code_complete"


async def test_multimodal_system_with_model_word_keeps_normal_events(
    app_config: AppConfig,
) -> None:
    """多模态路径里只有 model 这个词时同样是普通题。"""
    client = MockClient(app_config)

    text = await _collect_multimodal(
        client, _multimodal_messages(_MODEL_SYSTEM, "解释一下索引失效")
    )

    events = _events(text)
    assert events[0]["mode"] == "normal"
    assert events[-1]["type"] == "done"


async def test_real_coding_system_prompt_is_detected(
    app_config: AppConfig,
) -> None:
    """代码题的真实 system 文本带标记，假流能认出它是代码题请求。"""
    from app.llm.prompt_coding import build_coding_system_prompt

    client = MockClient(app_config)

    text = await _collect(
        client, _messages(build_coding_system_prompt(), _ACM_QUESTION)
    )

    events = _events(text)
    assert events[0]["mode"] == "coding"
    assert events[-1]["type"] == "code_complete"


# ----------------------------------------------------------------------
# 提交格式判定只有一处，见《7-代码问题模式》提交格式识别一节
# ----------------------------------------------------------------------
async def test_acmicpc_question_is_acm(app_config: AppConfig) -> None:
    """ACMICPC 风格题面判成 acm，不是 leetcode。

    题面有函数签名与描述性的「返回最大子数组和」，还有输入输出格式样例；
    描述性的返回值说法不算返回值类型说明，所以判 acm。
    """
    client = MockClient(app_config)

    text = await _collect(client, _messages(_MARKER_SYSTEM, _ACMICPC_QUESTION))

    events = _events(text)
    complete = events[-1]
    assert complete["type"] == "code_complete"
    assert complete["submission_mode"] == "acm"
    # acm 模式给完整 main、输入处理与输出
    code = _code_text(events)
    assert "int main(" in code
    assert "class Solution" not in code


async def test_mock_client_has_no_submission_mode_copy() -> None:
    """假流客户端里不再有自己的提交格式判定副本。"""
    import app.llm.mock_client as mock_client

    assert not hasattr(mock_client, "_detect_submission_mode")
    assert mock_client.detect_submission_mode.__module__ == "app.llm.prompt_coding"


@pytest.mark.parametrize(
    ("question", "submission_mode"),
    [
        (_LEETCODE_QUESTION, "leetcode"),
        (_ACM_QUESTION, "acm"),
        (_NOWCODER_QUESTION, "nowcoder"),
    ],
)
async def test_submission_mode_matches_prompt_coding(
    app_config: AppConfig, question: str, submission_mode: str
) -> None:
    """假流给出的提交格式与提示词侧一致。"""
    from app.llm.prompt_coding import detect_submission_mode

    client = MockClient(app_config)

    text = await _collect(client, _messages(_MARKER_SYSTEM, question))

    events = _events(text)
    assert events[-1]["submission_mode"] == submission_mode
    assert events[-1]["submission_mode"] == detect_submission_mode(question)
