"""问题会话的状态结构。

题型枚举与项目类关键词见《3-文本输入与流式回答》题型判定一节；
输入结构与《8-集成与验收》三种输入的汇合一节一致。

会话还存本次用到的检索来源与「没取到项目检索结果」的标记，
见《6-Dify检索与项目知识》检索来源展示与故障降级两节。

本模块只放数据结构与文本规范化，不做任何判定，也不发请求。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "QUESTION_TYPES",
    "PROJECT_KEYWORDS",
    "QuestionInput",
    "QuestionSession",
    "normalize_text",
]


# 题型枚举，取值与《3-文本输入与流式回答》题型判定一节一致
QUESTION_TYPES: tuple[str, ...] = (
    "conceptual",
    "project",
    "coding",
    "debugging",
    "system_design",
    "unknown",
)

# 项目类关键词，命中即认为需要检索项目知识，见《3-文本输入与流式回答》题型判定一节
PROJECT_KEYWORDS: tuple[str, ...] = (
    "我的项目",
    "这个项目",
    "我之前的项目",
    "项目中怎么实现",
    "项目架构",
    "我们的服务",
    "我写的代码",
    "我们用的数据库",
)


@dataclass(frozen=True)
class QuestionInput:
    """三种来源统一的输入结构。"""

    session_id: str
    source: str
    text: str | None
    image_bytes: bytes | None
    timestamp: float
    auto_submit: bool
    confidence: float | None


@dataclass
class QuestionSession:
    """当前问题会话的状态。"""

    session_id: str
    created_at: float
    source: str = "text"
    text: str | None = None
    image_bytes: bytes | None = None
    # 截图的 data URL，见《4-截图输入与多模态》多模态请求一节。
    # 没有截图时是 None，重新分析时复用同一张图。
    image_data_url: str | None = None
    question_type: str = "unknown"
    rag_required: bool = False
    request_id: str | None = None
    # 本次会话用到的检索来源，普通字典列表。
    # 每项字段是 document_name、score、content，
    # 见《6-Dify检索与项目知识》检索来源展示一节。
    # 存进会话是为了模型输出结束后按 F3 还能再查。
    retrieval_sources: list[dict[str, Any]] = field(default_factory=list)
    # 本次没取到项目检索结果：检索失败、检索结果为空或没有可用的检索上下文。
    # 回答里不能声称已经检索到项目资料，见《6-Dify检索与项目知识》故障降级一节。
    retrieval_missing: bool = False
    # 项目模式提交时注入的对话日志上下文（``DialogueLog.recent_text()`` 的
    # 全文，含角色标注）。普通模式是空串，组装请求时不注入，
    # 见《5-语音输入与流式ASR》项目模式一节。
    dialogue_context: str = ""


def normalize_text(text: str) -> str:
    """把文本规范化成去重用的比对形式。

    去掉空白与标点差异（中英文标点都算），只保留字母、数字与汉字，
    结果转小写。规范化只用于「同一会话内重复提交」的比对，
    见《8-集成与验收》防止重复调用一节。

    :param text: 原始文本。
    :return: 规范化后的字符串，空串或纯标点的文本返回空串。
    """
    kept: list[str] = []
    for char in text:
        # Unicode 分类里 L 开头是字母（含汉字），N 开头是数字，其余（空白、标点、符号）都丢掉
        if unicodedata.category(char)[0] in ("L", "N"):
            kept.append(char)
    return "".join(kept).lower()
