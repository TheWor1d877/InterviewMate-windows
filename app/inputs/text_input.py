"""底部输入框的文本输入入口。

把输入框里的文字整理成统一输入结构 :class:`~app.core.question_session.QuestionInput`，
交给核心层处理。输入层依赖核心层是允许的方向，见《2-骨架与配置》目录结构约定一节。
"""

from __future__ import annotations

import time

from app.core.question_session import QuestionInput

__all__ = ["build_text_input"]


def build_text_input(text: str, session_id: str) -> QuestionInput:
    """把输入框文本整理成统一输入结构。

    手动回车提交时 ``source`` 写 ``text``、``auto_submit`` 写 true、
    ``confidence`` 写 null，题面去掉首尾空白，见《3-文本输入与流式回答》问题会话一节。

    :param text: 输入框里的原文。
    :param session_id: 所属会话标识。
    :return: 统一输入结构，``image_bytes`` 为 None。
    """
    return QuestionInput(
        session_id=session_id,
        source="text",
        text=text.strip(),
        image_bytes=None,
        timestamp=time.time(),
        auto_submit=True,
        confidence=None,
    )
