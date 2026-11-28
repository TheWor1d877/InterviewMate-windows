"""用模型判定题型、项目范围与检索需求。

规则判定在 :mod:`app.core.question_classifier` 里，模型判定放在这里：
本模块实现 core 层声明的 :class:`~app.core.question_classifier.AsyncQuestionClassifier`
协议，core 层不 import 本模块，见《2-骨架与配置》目录结构约定一节。

一次判定只发一次非流式请求，要求模型只输出一段 JSON；system prompt 里
注入 ``config/projects.md`` 的项目清单，作为 ``project_scope`` 三分类的
判断依据。解析容错写到 :func:`_parse_classification`，出错一律抛
``ValueError``，超时交给 ``asyncio.wait_for``，都由上层回退到同步规则。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Protocol

from app.config import REPO_ROOT
from app.core.question_classifier import PROJECT_SCOPES, Classification

__all__ = [
    "JSONCompletionClient",
    "OpenAICompatQuestionClassifier",
    "LEGAL_QUESTION_TYPES",
    "DEFAULT_PROJECTS_PATH",
]


# 模型判定允许的题型，取值是《3-文本输入与流式回答》题型判定一节里的子集
LEGAL_QUESTION_TYPES: frozenset[str] = frozenset(
    {"coding", "project", "system_design", "conceptual", "unknown"}
)

# 项目清单的默认位置：仓库根下 config/projects.md
DEFAULT_PROJECTS_PATH: Path = REPO_ROOT / "config" / "projects.md"

# 一次判定的超时（秒）：判定只决定要不要检索，不该拖住提交流程
_DEFAULT_TIMEOUT: float = 3.0
# JSON 回复很短，几十个 token 足够
_DEFAULT_MAX_TOKENS: int = 128
# 判定要稳定，温度取 0
_DEFAULT_TEMPERATURE: float = 0.0
# 项目清单注入 prompt 的长度上限（字符）：清单太长时截断，防止撑爆判定请求
_MAX_PROJECTS_CHARS: int = 4000

# 判定提示词模板：只输出 JSON，规则里写明通用知识题不得判成项目；
# __PROJECTS_SECTION__ 占位在构建时替换成项目清单段（不用 str.format，
# 模板里的 JSON 示例自带花括号，format 会把它当占位符）
_SYSTEM_PROMPT_TEMPLATE: str = """你是面试辅助程序的题型判定器。只输出一行 JSON，不要代码块，不要解释，不要任何前后缀。
JSON 字段固定为：
{"type": "coding|project|system_design|conceptual|unknown", "project_scope": "none|mixed|full", "reason": "不超过20字的中文依据"}

判定规则：
1. LeetCode 题面、算法题、要求写代码或给出实现、报错堆栈或异常 → type 为 coding，project_scope 为 none。
2. 通用八股知识题（如 TCP、数据库索引、shared_ptr 原理）→ type 为 conceptual 或 system_design，project_scope 为 none。
3. 主体问通用知识，但要求结合求职者本人的项目来答（出现“结合你的项目”“在你项目里怎么用”等表述）→ type 为 conceptual 或 system_design，project_scope 为 mixed。
4. 问题本身在问求职者本人的项目、服务或工作经历 → type 为 project，project_scope 为 full。
5. 拿不准 → type 为 unknown，project_scope 为 none。
__PROJECTS_SECTION__
禁止把通用知识题判成项目：题目没有明确指向求职者本人的项目或经历时，project_scope 不得是 mixed 或 full。
只输出 JSON，不要代码块。"""

# 项目清单段模板：清单非空时替换进上面的占位
_PROJECTS_SECTION_TEMPLATE: str = """求职者本人的项目清单（判定 project_scope 的依据）：
__PROJECTS__

"""


class JSONCompletionClient(Protocol):
    """非流式 JSON 补全客户端的能力约定。

    真实实现是 :class:`~app.llm.openai_compat_client.OpenAICompatClient`，
    测试里注入假实现，本模块不关心底层怎么发请求。
    """

    async def complete_json(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


class OpenAICompatQuestionClassifier:
    """用模型判定题型、项目范围与检索需求。

    :param client: 具备 ``complete_json`` 的非流式客户端。
    :param timeout: 单次判定的超时（秒），超时抛 ``TimeoutError``，
        由上层回退到同步规则。默认 3 秒。
    :param projects_profile: 注入 system prompt 的项目清单文本；
        None 时自动读 :data:`DEFAULT_PROJECTS_PATH`（读不到就用空清单），
        显式传空串表示不注入项目清单。
    """

    def __init__(
        self,
        client: JSONCompletionClient,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        projects_profile: str | None = None,
    ) -> None:
        self._client = client
        self._timeout = float(timeout)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        if projects_profile is None:
            projects_profile = _load_projects_profile()
        self._projects_profile = str(projects_profile)
        self._system_prompt = _build_system_prompt(self._projects_profile)

    async def classify(self, text: str) -> Classification:
        """发一次非流式请求，把模型回复解析成判定结果。

        :param text: 用户输入的题面。
        :return: 判定结果，``source`` 恒为 ``"llm"``。
        :raises ValueError: 回复不是合法 JSON，或字段缺失、取值非法。
        :raises TimeoutError: 超过 ``timeout`` 秒没有回复。
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": text},
        ]
        raw = await asyncio.wait_for(
            self._client.complete_json(
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            ),
            timeout=self._timeout,
        )
        return _parse_classification(raw)


def _load_projects_profile(path: Path | None = None) -> str:
    """读项目清单文件，作为 ``project_scope`` 判定依据注入 prompt。

    文件缺失、读取失败或解码失败都返回空串：清单是增强项，
    读不到不该拖住启动，判定 prompt 里就不注入项目段。
    内容超过 :data:`_MAX_PROJECTS_CHARS` 时截断，保留头部。

    :param path: 清单路径；None 时读 :data:`DEFAULT_PROJECTS_PATH`。
    """
    target = Path(path) if path is not None else DEFAULT_PROJECTS_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    text = text.strip()
    if len(text) > _MAX_PROJECTS_CHARS:
        text = text[:_MAX_PROJECTS_CHARS].rstrip()
    return text


def _build_system_prompt(projects_profile: str) -> str:
    """组装判定 system prompt，清单非空时注入项目段。

    :param projects_profile: 项目清单文本，空串时不注入项目段。
    """
    profile = projects_profile.strip()
    projects_section = ""
    if profile:
        projects_section = _PROJECTS_SECTION_TEMPLATE.replace("__PROJECTS__", profile)
    return _SYSTEM_PROMPT_TEMPLATE.replace("__PROJECTS_SECTION__", projects_section)


def _parse_classification(raw: str) -> Classification:
    """把模型回复解析成 :class:`Classification`。

    容错只做两件事：去掉 ``` 围栏，取第一个 ``{`` 到最后一个 ``}``。
    其余情况（非 JSON、缺字段、type 非法、project_scope 非法）一律抛
    ``ValueError``，由上层回退规则。``rag_required`` 由 ``project_scope``
    推导：``mixed`` 与 ``full`` 要检索，``none`` 不检索，单一事实来源，
    模型不再单独输出 need_rag。

    :param raw: 模型回复的原文。
    """
    if not isinstance(raw, str):
        raise ValueError("题型判定回复不是字符串")
    text = raw.replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("题型判定回复里没有 JSON 对象")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"题型判定回复不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("题型判定回复不是 JSON 对象")

    question_type = payload.get("type")
    if not isinstance(question_type, str) or question_type not in LEGAL_QUESTION_TYPES:
        raise ValueError(f"题型判定回复的 type 非法：{question_type!r}")

    project_scope = payload.get("project_scope")
    if not isinstance(project_scope, str) or project_scope not in PROJECT_SCOPES:
        raise ValueError(f"题型判定回复的 project_scope 非法：{project_scope!r}")

    reason = payload.get("reason")
    if not isinstance(reason, str):
        raise ValueError(f"题型判定回复的 reason 非法：{reason!r}")

    return Classification(
        question_type=question_type,
        rag_required=project_scope != "none",
        matched_rules=("llm",),
        reason=reason.strip(),
        source="llm",
        project_scope=project_scope,
    )