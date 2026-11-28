"""模型题型判定客户端的测试。

用假补全客户端验证：合法 JSON 解析（type + project_scope + reason）、
围栏与前后废话容错、非法回复抛 ``ValueError``、超时抛 ``TimeoutError``，
以及 ``config/projects.md`` 项目清单的读取与 prompt 注入。不发网络请求。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import app.llm.question_classifier_client as classifier_client_module
from app.core.question_classifier import Classification
from app.llm.question_classifier_client import (
    DEFAULT_PROJECTS_PATH,
    LEGAL_QUESTION_TYPES,
    _MAX_PROJECTS_CHARS,
    _build_system_prompt,
    _load_projects_profile,
    OpenAICompatQuestionClassifier,
)


class FakeCompletionClient:
    """假 JSON 补全客户端，记录调用并按设定返回或抛错。"""

    def __init__(
        self,
        reply: str = "",
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.reply = reply
        self.error = error
        self.delay = delay
        self.calls: list[dict] = []

    async def complete_json(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.reply


# ----------------------------------------------------------------
# JSON 契约解析：type + project_scope + reason
# ----------------------------------------------------------------


async def test_valid_json_maps_to_classification() -> None:
    """合法 JSON 解出 type、project_scope 与 reason，来源标成 llm。"""
    client = FakeCompletionClient(
        reply='{"type": "coding", "project_scope": "none", "reason": "算法题"}'
    )
    classifier = OpenAICompatQuestionClassifier(client, projects_profile="")

    result = await classifier.classify("给定一个数组，返回两数之和")

    assert isinstance(result, Classification)
    assert result.question_type == "coding"
    assert result.project_scope == "none"
    assert result.rag_required is False
    assert result.reason == "算法题"
    assert result.source == "llm"
    assert result.matched_rules == ("llm",)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["max_tokens"] == 128
    assert call["temperature"] == 0.0
    assert call["messages"][0]["role"] == "system"
    assert "只输出" in call["messages"][0]["content"]
    assert "JSON" in call["messages"][0]["content"]
    assert call["messages"][1] == {
        "role": "user",
        "content": "给定一个数组，返回两数之和",
    }


@pytest.mark.parametrize(
    ("scope", "expected_rag"),
    [("none", False), ("mixed", True), ("full", True)],
)
async def test_project_scope_drives_rag_required(scope: str, expected_rag: bool) -> None:
    """rag_required 由 project_scope 推导：mixed 与 full 检索，none 不检索。"""
    client = FakeCompletionClient(
        reply=f'{{"type": "project", "project_scope": "{scope}", "reason": "依据"}}'
    )

    result = await OpenAICompatQuestionClassifier(client, projects_profile="").classify("题面")

    assert result.project_scope == scope
    assert result.rag_required is expected_rag


async def test_code_fence_and_surrounding_text_are_tolerated() -> None:
    """带 ``` 围栏、前后有废话时照样能解出 JSON。"""
    reply = (
        "好的，判定如下：\n"
        "```json\n"
        '{"type": "system_design", "project_scope": "none", "reason": "通用设计题"}\n'
        "```\n"
        "以上。"
    )
    client = FakeCompletionClient(reply=reply)

    result = await OpenAICompatQuestionClassifier(client, projects_profile="").classify(
        "设计一个短链服务"
    )

    assert result.question_type == "system_design"
    assert result.project_scope == "none"
    assert result.rag_required is False
    assert result.reason == "通用设计题"


@pytest.mark.parametrize("question_type", sorted(LEGAL_QUESTION_TYPES))
async def test_all_legal_types_are_accepted(question_type: str) -> None:
    """五个合法 type 都能解析。"""
    client = FakeCompletionClient(
        reply=f'{{"type": "{question_type}", "project_scope": "none", "reason": "依据"}}'
    )

    result = await OpenAICompatQuestionClassifier(client, projects_profile="").classify("题面")

    assert result.question_type == question_type


@pytest.mark.parametrize(
    "reply",
    [
        # 非法 type：debugging 不在模型判定允许的取值里
        '{"type": "debugging", "project_scope": "none", "reason": "报错"}',
        # 缺少 project_scope
        '{"type": "coding", "reason": "算法题"}',
        # project_scope 不是合法取值
        '{"type": "coding", "project_scope": "all", "reason": "算法题"}',
        # 缺少 reason
        '{"type": "coding", "project_scope": "none"}',
        # reason 不是字符串
        '{"type": "coding", "project_scope": "none", "reason": 1}',
        # 不是 JSON
        "这道题是算法题",
        # 空串
        "",
    ],
)
async def test_invalid_reply_raises_value_error(reply: str) -> None:
    """非法 type、缺字段、非法 project_scope、非 JSON 都抛 ValueError，由上层回退。"""
    client = FakeCompletionClient(reply=reply)

    with pytest.raises(ValueError):
        await OpenAICompatQuestionClassifier(client, projects_profile="").classify("题面")


async def test_timeout_raises_timeout_error() -> None:
    """超过超时时间抛 TimeoutError，由上层回退规则。"""
    client = FakeCompletionClient(
        reply='{"type": "coding", "project_scope": "none", "reason": "算法题"}',
        delay=0.5,
    )
    classifier = OpenAICompatQuestionClassifier(client, projects_profile="", timeout=0.01)

    with pytest.raises(asyncio.TimeoutError):
        await classifier.classify("题面")


async def test_client_error_propagates() -> None:
    """补全客户端的异常原样抛出，本模块不吞异常。"""
    client = FakeCompletionClient(error=RuntimeError("网络挂了"))
    classifier = OpenAICompatQuestionClassifier(client, projects_profile="")

    with pytest.raises(RuntimeError):
        await classifier.classify("题面")


# ----------------------------------------------------------------
# 项目清单读取与 prompt 注入
# ----------------------------------------------------------------


def test_load_projects_profile_reads_file(tmp_path: Path) -> None:
    """文件存在时读出首尾去空白后的文本，中间内容原样保留。"""
    file = tmp_path / "projects.md"
    file.write_text("第一行\n\n第二行\n", encoding="utf-8")

    assert _load_projects_profile(file) == "第一行\n\n第二行"


def test_load_projects_profile_missing_file_returns_empty(tmp_path: Path) -> None:
    """文件缺失返回空串，不抛异常。"""
    assert _load_projects_profile(tmp_path / "not-exists.md") == ""


def test_load_projects_profile_directory_returns_empty(tmp_path: Path) -> None:
    """路径是目录时读取失败，返回空串。"""
    assert _load_projects_profile(tmp_path) == ""


def test_load_projects_profile_blank_file_returns_empty(tmp_path: Path) -> None:
    """全空白的文件 strip 后为空串。"""
    file = tmp_path / "projects.md"
    file.write_text("   \n\n  \n", encoding="utf-8")

    assert _load_projects_profile(file) == ""


def test_load_projects_profile_truncates_long_file(tmp_path: Path) -> None:
    """超过长度上限时截断到上限字符。"""
    file = tmp_path / "projects.md"
    file.write_text("测" * (_MAX_PROJECTS_CHARS + 100), encoding="utf-8")

    profile = _load_projects_profile(file)

    assert len(profile) == _MAX_PROJECTS_CHARS


def test_load_projects_profile_bad_encoding_returns_empty(tmp_path: Path) -> None:
    """非 UTF-8 字节解码失败返回空串。"""
    file = tmp_path / "projects.md"
    file.write_bytes("项目清单中文".encode("gbk"))

    assert _load_projects_profile(file) == ""


def test_build_system_prompt_injects_projects_section() -> None:
    """清单非空时 system prompt 注入项目清单段。"""
    prompt = _build_system_prompt("示例项目：一句话定位")

    assert "示例项目：一句话定位" in prompt
    assert "项目清单" in prompt


def test_build_system_prompt_without_profile_has_no_projects_section() -> None:
    """清单为空时不注入项目段，判定规则仍在。"""
    prompt = _build_system_prompt("")

    assert "项目清单" not in prompt
    assert "判定规则" in prompt


async def test_classifier_injects_profile_into_system_message() -> None:
    """构造时传 projects_profile 后，发请求的 system 消息带清单内容。"""
    client = FakeCompletionClient(
        reply='{"type": "project", "project_scope": "full", "reason": "本人项目"}'
    )
    classifier = OpenAICompatQuestionClassifier(
        client, projects_profile="示例项目A：电商订单系统"
    )

    await classifier.classify("我的项目怎么做的")

    system_content = client.calls[0]["messages"][0]["content"]
    assert "示例项目A：电商订单系统" in system_content
    assert "project_scope" in system_content


def test_classifier_reads_default_projects_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """projects_profile 不传时自动读 DEFAULT_PROJECTS_PATH，读不到也不炸。"""
    file = tmp_path / "projects.md"
    file.write_text("默认路径项目", encoding="utf-8")
    monkeypatch.setattr(classifier_client_module, "DEFAULT_PROJECTS_PATH", file)
    client = FakeCompletionClient()

    classifier = OpenAICompatQuestionClassifier(client)

    assert "默认路径项目" in classifier._system_prompt
    assert client.calls == []


def test_default_projects_path_points_to_config_dir() -> None:
    """默认清单路径是仓库根下 config/projects.md。"""
    assert DEFAULT_PROJECTS_PATH.name == "projects.md"
    assert DEFAULT_PROJECTS_PATH.parent.name == "config"

# ----------------------------------------------------------------
# 解析容错的剩余分支
# ----------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, 123, ["{"]])
def test_parse_classification_rejects_non_string_reply(raw: object) -> None:
    """回复不是字符串时抛 ValueError，不做静默转换。"""
    from app.llm.question_classifier_client import _parse_classification

    with pytest.raises(ValueError, match="不是字符串"):
        _parse_classification(raw)


def test_parse_classification_reports_json_decode_error() -> None:
    """花括号包裹但内容不是合法 JSON 时，报 JSONDecodeError 的转写。"""
    from app.llm.question_classifier_client import _parse_classification

    with pytest.raises(ValueError, match="不是合法 JSON"):
        _parse_classification("{不是json}")