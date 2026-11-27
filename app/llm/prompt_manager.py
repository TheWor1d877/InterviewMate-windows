"""按题型挑 Prompt。

普通题走 :func:`app.llm.prompt_normal.build_normal_messages`，
代码题走 :func:`app.llm.prompt_coding.build_coding_messages`，
其余题型仍走普通题，分流规则见《7-代码问题模式》提交格式识别一节。

带截图的请求走普通题或代码题各自的组装函数：普通题用截图专用的 system，
普通题的 system 加截图说明与路由规则，先判这次按代码题还是普通题回答，
见《4-截图输入与多模态》；代码题复用代码题的 system，
同样多一段截图说明，user 的 content 是数组，先文字块再图片块。
"""

from __future__ import annotations

from app.config import AppConfig
from app.llm.prompt_coding import build_coding_messages
from app.llm.prompt_coding import build_coding_multimodal_messages
from app.llm.prompt_coding import detect_submission_mode
from app.llm.prompt_normal import build_normal_messages
from app.llm.prompt_screenshot import build_screenshot_messages
from app.utils.logger import log_event

__all__ = ["PromptManager"]

# 代码题的题型标识，取值见《3-文本输入与流式回答》题型判定一节
_CODING_QUESTION_TYPE = "coding"


class PromptManager:
    """请求消息的组装入口。"""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def build_messages(
        self,
        question: str,
        question_type: str = "unknown",
        context: str | None = None,
    ) -> list[dict[str, str]]:
        """按题型组装请求消息。

        题型是 ``coding`` 时走代码题的 Prompt，其余题型走普通题的 Prompt。
        提交格式由配置里的 ``coding.submission_mode`` 决定：配置显式指定了
        具体平台就用配置值，取 ``auto`` 时按题面推断，
        规则见《7-代码问题模式》提交格式识别一节。

        :param question: 用户的问题文本。
        :param question_type: 题型标识，取值见《3-文本输入与流式回答》题型判定一节。
        :param context: 检索到的背景上下文，没有就传 None。
        """
        log_event(
            "llm.prompt.build",
            "组装请求消息",
            level="DEBUG",
            question_type=question_type,
            has_context=bool(context and context.strip()),
        )
        if question_type == _CODING_QUESTION_TYPE:
            submission_mode = self._resolve_submission_mode(question)
            log_event(
                "llm.prompt.build_coding",
                "按代码题组装请求消息",
                level="DEBUG",
                question_type=question_type,
                submission_mode=submission_mode,
                has_context=bool(context and context.strip()),
            )
            return build_coding_messages(question, context, submission_mode)
        return build_normal_messages(question, context)

    def build_multimodal_messages(
        self,
        question: str,
        image_data_url: str,
        question_type: str = "unknown",
        context: str | None = None,
    ) -> list[dict]:
        """组装带截图的请求消息。

        题型是 ``coding`` 时走代码题的 Prompt，其余题型走截图专用的 Prompt。
        截图专用的 system 由普通题的 system 加截图说明与路由规则拼成，
        模型按截图内容自己判这次按代码题还是普通题回答，
        见《4-截图输入与多模态》。
        两条路的 user 消息 ``content`` 都是数组，先文本块再图片块，
        见《4-截图输入与多模态》多模态请求一节。

        :param question: 题面文字，没有文字时传空串。
        :param image_data_url: 截图的 data URL。
        :param question_type: 题型标识，取值见《3-文本输入与流式回答》题型判定一节。
        :param context: 检索到的背景上下文，没有就传 None。
        """
        log_event(
            "llm.prompt.build_multimodal",
            "组装带截图的请求消息",
            level="DEBUG",
            question_type=question_type,
            has_question=bool(question.strip()),
            has_context=bool(context and context.strip()),
        )
        if question_type == _CODING_QUESTION_TYPE:
            submission_mode = self._resolve_submission_mode(question)
            log_event(
                "llm.prompt.build_coding_multimodal",
                "按代码题组装带截图的请求消息",
                level="DEBUG",
                question_type=question_type,
                submission_mode=submission_mode,
                has_question=bool(question.strip()),
                has_context=bool(context and context.strip()),
            )
            return build_coding_multimodal_messages(
                question, image_data_url, context, submission_mode
            )
        return build_screenshot_messages(question, image_data_url, context)

    def _resolve_submission_mode(self, question: str) -> str:
        """决定代码题用哪种提交格式。

        配置里的 ``coding.submission_mode`` 显式给了具体平台就用配置值，
        取 ``auto`` 时按题面推断，规则见《7-代码问题模式》提交格式识别一节。

        :param question: 用户的问题文本。
        """
        configured = self._config.coding.submission_mode
        if configured != "auto":
            return configured
        return detect_submission_mode(question)
