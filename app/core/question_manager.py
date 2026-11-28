"""问题会话编排：判定题型、决定是否检索、组织请求、接收流式事件。

职责对应《3-文本输入与流式回答》问题会话一节与《8-集成与验收》请求取消、
防止重复调用、超时管理三节：

- 维护当前问题会话，保存题面与题型，供重新分析复用同一份材料；
- 去掉同一会话内的重复提交；
- 判定题型与是否检索；
- 组织模型请求，把流式事件交给事件分发器；
- 新问题到达时取消旧请求并清空界面，旧请求后续到达的事件按 ``request_id`` 丢弃；
- 请求任务与检索任务登记在受管任务注册表里，取消走它，超时也由它兜底，
  见《8-集成与验收》请求取消与超时管理两节。

带截图的请求也在这一层组织：文字与图片合成一条消息一次发出，
语音文字与截图的合并发生在这里，见《4-截图输入与多模态》与语音合并一节。

本模块不实现模型客户端、事件模型与解析器，它们由外部注入：
客户端走 :class:`LLMClient`，解析器按请求新建（``parser_factory``），
事件分发走 :class:`EventSink`，提示词由 ``build_messages`` 与
``build_multimodal_messages`` 两个回调构造。两个回调都是必填的，
core 层不直接 import llm 层的实现类，见《2-骨架与配置》目录结构约定一节。

检索同样由外部注入：``retrieve`` 回调发起检索并返回上下文与来源条目，
``on_sources`` 回调把来源条目送给界面。core 层不 import 检索层与界面层的
实现类，见《2-骨架与配置》目录结构约定一节与
《6-Dify检索与项目知识》检索接口抽象一节。

代码题判定也往外通知一次：``on_coding_mode`` 回调在发出模型请求之前告诉外面
「这次是代码题」，外面据此让右栏先切到代码形态，
见《7-代码问题模式》纯代码块兜底一节。没有注入这个回调时退回到靠事件反推：
分发器收到代码题事件时才切形态。

模型状态也往外报一次：``on_model_state`` 回调带着状态字符串，
取值是 idle、connecting、streaming、error，见《2-骨架与配置》状态字段取值表。
没有注入这个回调时不报，状态由外面自己管。

检索失败不让问题处理崩掉：拿不到项目资料时继续走不带上下文的请求，
并在会话上标记没取到，见《6-Dify检索与项目知识》故障降级一节。

这次请求的耗时指标也在这里记：``request_latency``、``ttft``、
``first_framework``、``first_item``、``total_latency``，
口径见《8-集成与验收》性能统计一节。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from app.config import AppConfig
from app.core.exceptions import InterviewMateError
from app.core.question_classifier import (
    AsyncQuestionClassifier,
    Classification,
    QuestionClassifier,
)
from app.core.question_session import QuestionSession, normalize_text
from app.core.retrieval_policy import RetrievalPolicy
from app.core.task_manager import TaskInfo, TaskManager
from app.utils.logger import log_event
from app.utils.timing import MetricRecorder, default_recorder

if TYPE_CHECKING:
    # 运行期不 import 输出层模块，跨层只通过事件分发器交互
    from app.output.event_models import StreamEvent

__all__ = [
    "LLMClient",
    "StreamParser",
    "EventSink",
    "MessageBuilder",
    "MultimodalMessageBuilder",
    "RetrievalFunc",
    "SourcesCallback",
    "CodingModeCallback",
    "ModelStateCallback",
    "QuestionManager",
    "MODEL_STATE_IDLE",
    "MODEL_STATE_CONNECTING",
    "MODEL_STATE_STREAMING",
    "MODEL_STATE_ERROR",
    "REQUEST_LATENCY_METRIC",
    "TTFT_METRIC",
    "FIRST_FRAMEWORK_METRIC",
    "FIRST_ITEM_METRIC",
    "TOTAL_LATENCY_METRIC",
]


# 文本请求的消息构造回调：题面与题型，末位可以多带一个检索上下文。
# 没有检索上下文时按前两个参数调用，兼容不带 context 参数的旧回调，
# 见《6-Dify检索与项目知识》检索流程一节
MessageBuilder = Callable[..., list[dict[str, str]]]

# 带图请求的消息构造回调：题面、图片 data URL 与题型，末位可以多带一个检索上下文
MultimodalMessageBuilder = Callable[..., list[dict]]

# 检索回调：参数是题面，返回拼好的上下文文本与本次用到的来源条目。
# 上下文是 None 或空串表示没取到项目资料；来源条目是普通字典，
# 字段是 document_name、score、content，见《6-Dify检索与项目知识》结果整理一节
RetrievalFunc = Callable[[str], Awaitable[tuple[str | None, list[dict]]]]

# 来源条目回调：参数是来源条目列表，界面用它刷新检索来源面板，
# 见《6-Dify检索与项目知识》检索来源展示一节
SourcesCallback = Callable[[list[dict]], None]

# 代码题通知回调：参数是布尔，True 表示这次是代码题，False 表示普通题。
# 与输出层分发器的 ``set_coding_mode`` 对得上，
# 见《7-代码问题模式》纯代码块兜底一节
CodingModeCallback = Callable[[bool], None]

# 模型状态回调：参数是状态字符串，取值见下面的四个常量。
# core 层只报状态，不认识界面，由装配层接上去更新状态栏，
# 见《2-骨架与配置》状态字段取值表
ModelStateCallback = Callable[[str], None]

# MODEL 字段的四个取值，见《2-骨架与配置》状态字段取值表：
# 请求发出前是 connecting，收到第一个流式片段是 streaming，
# 正常结束回到 idle，出错是 error
MODEL_STATE_IDLE: str = "idle"
MODEL_STATE_CONNECTING: str = "connecting"
MODEL_STATE_STREAMING: str = "streaming"
MODEL_STATE_ERROR: str = "error"

# 请求任务的总时限余量（秒）：在 ``model.read_timeout`` 之外再加这么多。
# read_timeout 是两次读之间的间隔，整条流可以更久，但不能无限等下去，
# 见《8-集成与验收》超时管理一节
_REQUEST_TIMEOUT_MARGIN: float = 60.0

# 代码题的题型标识，取值见《3-文本输入与流式回答》题型判定一节
_CODING_QUESTION_TYPE = "coding"

# 请求类耗时指标名，取值见《2-骨架与配置》性能指标一节
REQUEST_LATENCY_METRIC: str = "request_latency"
TTFT_METRIC: str = "ttft"
FIRST_FRAMEWORK_METRIC: str = "first_framework"
FIRST_ITEM_METRIC: str = "first_item"
TOTAL_LATENCY_METRIC: str = "total_latency"

# 项目模式提交问题时，对话日志在 user 消息里的段落标题。
# 日志正文（``DialogueLog.recent_text()``，含角色标注）拼在这一段下面，
# 位置在 user 消息靠前、本轮问题之前，见《5-语音输入与流式ASR》项目模式一节
DIALOGUE_CONTEXT_TITLE: str = "对话上下文"

# 项目模式附加给模型的两句指令：拼在检索内容之后（没检索到内容时
# 拼在问题之后），提醒模型围绕问题回答、先理解语音转写的问题原意
DIALOGUE_CONTEXT_NOTES: str = (
    "检索出来的内容不一定对问题有用，请围绕问题进行回答。"
    "面试官的问题来自语音转写，可能不精准，先理解他的意思，再以精准的表述回答。"
)

# 框架事件的事件类型：普通题与代码题共用同一个类型，
# 见《3-文本输入与流式回答》流式事件协议一节与《7-代码问题模式》流式事件一节
_FRAMEWORK_EVENT_TYPE: str = "framework"

# 内容单元的事件类型：普通题的 item 与代码题的 code_delta，
# 见《3-文本输入与流式回答》流式事件协议一节与《7-代码问题模式》流式事件一节
_ITEM_EVENT_TYPES: frozenset[str] = frozenset({"item", "code_delta"})


class LLMClient(Protocol):
    """模型客户端的能力约定。"""

    def chat_stream(
        self, messages: list[dict[str, str]], request_id: str
    ) -> AsyncIterator[str]: ...

    def multimodal_stream(self, messages: list[dict], request_id: str) -> AsyncIterator[str]: ...

    def cancel(self, request_id: str) -> Awaitable[None]: ...


class StreamParser(Protocol):
    """流式解析器的能力约定。

    解析器按请求新建，``request_id`` 与 ``session_id`` 是构造参数，
    ``reset`` 只清缓冲区，不再改这两个标识。
    """

    def feed(self, text: str) -> list[StreamEvent]: ...

    def flush(self) -> list[StreamEvent]: ...

    def reset(self) -> None: ...


class EventSink(Protocol):
    """事件分发器的能力约定。"""

    def dispatch(self, event: StreamEvent) -> None: ...

    def clear(self) -> None: ...

    def report_error(self, code: str, detail: str) -> None: ...


class _RequestMetrics:
    """一次请求的耗时指标接线。

    口径见《8-集成与验收》性能统计一节：

    - ``request_latency`` 从决定发这次请求算到请求交到客户端；
    - ``ttft``、``first_framework``、``first_item``、``total_latency``
      四个一律从请求交到客户端那一刻算起。

    同一个请求里同名指标只记一次，客户端重试导致多次发出时也只记第一次
    那组指标。请求被取消时不记 ``total_latency``（没有完整输出），
    已经记下的 ``ttft`` 等保留；流异常中断时 ``total_latency`` 照样记，
    它表示到中断为止已经发生的耗时。

    :param decided_at: 决定发这次请求的时刻，单调时钟秒数。
    :param source: 来源标记，客户端名，取 ``mock`` 或 ``openai``。
    :param recorder: 样本落库对象。
    """

    def __init__(
        self,
        decided_at: float,
        source: str,
        recorder: MetricRecorder,
    ) -> None:
        self._decided_at = decided_at
        self._source = source
        self._recorder = recorder
        # 请求交到客户端的时刻，还没交出去时是 None
        self._sent_at: float | None = None
        # 已经记过的指标名，同名指标不重复记
        self._recorded: set[str] = set()

    def mark_sent(self) -> None:
        """标记请求交到客户端、开始拉流的时刻，并记请求发起耗时。

        重复调用只认第一次：请求交出去只算一次。
        """
        if self._sent_at is not None:
            return
        self._sent_at = time.perf_counter()
        self._record(
            REQUEST_LATENCY_METRIC,
            (self._sent_at - self._decided_at) * 1000.0,
        )

    def mark_first_chunk(self) -> None:
        """标记首个流式片段到达的时刻，并记首 Token 到达耗时。"""
        if self._sent_at is None or TTFT_METRIC in self._recorded:
            return
        self._record(TTFT_METRIC, (time.perf_counter() - self._sent_at) * 1000.0)

    def observe_event(self, event: Any) -> None:
        """按事件类型记首个框架事件与首个内容单元的到达耗时。

        事件类型看事件上的 ``type`` 字段：``framework`` 是框架事件，
        普通题的 ``item`` 与代码题的 ``code_delta`` 是内容单元，
        见《3-文本输入与流式回答》流式事件协议一节与
        《7-代码问题模式》流式事件一节。core 层不 import 输出层的实现类，
        见《2-骨架与配置》目录结构约定一节。

        :param event: 解析器产出的事件。
        """
        event_type = getattr(event, "type", None)
        if not isinstance(event_type, str):
            return
        if event_type == _FRAMEWORK_EVENT_TYPE:
            self._mark_first(FIRST_FRAMEWORK_METRIC)
        elif event_type in _ITEM_EVENT_TYPES:
            self._mark_first(FIRST_ITEM_METRIC)

    def record_total(self) -> None:
        """记完整输出耗时。请求还没发出去时不记。"""
        if self._sent_at is None:
            return
        self._record(TOTAL_LATENCY_METRIC, (time.perf_counter() - self._sent_at) * 1000.0)

    def _mark_first(self, name: str) -> None:
        """记某个「首个到达」类指标，同名只记一次。

        :param name: 指标名。
        """
        if self._sent_at is None or name in self._recorded:
            return
        self._record(name, (time.perf_counter() - self._sent_at) * 1000.0)

    def _record(self, name: str, value_ms: float) -> None:
        """把一条样本写进记录器，同名指标不重复写。

        :param name: 指标名。
        :param value_ms: 耗时毫秒值。
        """
        if name in self._recorded:
            return
        self._recorded.add(name)
        self._recorder.record(name, value_ms, source=self._source)


class QuestionManager:
    """问题会话的编排者。

    :param config: 应用配置，取 ``rag.enabled`` 与 ``rag.force_project_mode``。
    :param client: 模型客户端。
    :param parser_factory: 解析器工厂，按 ``(request_id, session_id)`` 新建解析器。
        每个请求用各自的解析器，旧请求的残留文本不会混进新请求的缓冲区。
    :param sink: 事件分发器。
    :param build_messages: 构造模型请求消息的回调，参数是题面与题型，
        末位可以多带一个检索上下文。
    :param build_multimodal_messages: 构造带图请求消息的回调，参数是题面、
        图片 data URL 与题型，末位可以多带一个检索上下文。
        必填：core 层不直接 import llm 层的实现类，
        见《2-骨架与配置》目录结构约定一节。
    :param retrieve: 发起检索的回调，参数是题面，返回
        ``(context, sources)``：``context`` 是拼好的上下文文本，
        ``sources`` 是本次用到的来源条目列表。可选：没有注入时只做检索决策，
        不发起真实检索，见《6-Dify检索与项目知识》检索流程一节。
    :param on_sources: 把来源条目送给界面的回调，参数是来源条目列表。
        可选：只在有检索结果时调用一次，见
        《6-Dify检索与项目知识》检索来源展示一节。
    :param on_coding_mode: 通知外面「这次是代码题」的回调，参数是布尔。
        可选：判定出代码题时在发出模型请求之前通知一次，普通题通知假，
        见《7-代码问题模式》纯代码块兜底一节。没有注入时不通知，
        外面退回到靠事件反推。
    :param on_model_state: 通知外面模型状态的回调，参数是状态字符串，
        取值见 :data:`MODEL_STATE_IDLE` 等常量。可选：没有注入时不通知，
        状态由外面自己管，见《2-骨架与配置》状态字段取值表。
    :param llm_classifier: 模型题型判定器，实现
        :class:`~app.core.question_classifier.AsyncQuestionClassifier` 协议。
        可选：没有注入时只走同步规则；注入后判定失败或超时回退到同步规则，
        见《3-文本输入与流式回答》题型判定一节。
    """

    def __init__(
        self,
        config: AppConfig,
        client: LLMClient,
        parser_factory: Callable[[str, str], StreamParser],
        sink: EventSink,
        build_messages: MessageBuilder,
        *,
        build_multimodal_messages: MultimodalMessageBuilder,
        retrieve: RetrievalFunc | None = None,
        on_sources: SourcesCallback | None = None,
        on_coding_mode: CodingModeCallback | None = None,
        on_model_state: ModelStateCallback | None = None,
        llm_classifier: AsyncQuestionClassifier | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._parser_factory = parser_factory
        self._sink = sink
        self._build_messages = build_messages
        self._build_multimodal_messages = build_multimodal_messages
        self._retrieve = retrieve
        self._on_sources = on_sources
        self._on_coding_mode = on_coding_mode
        self._on_model_state = on_model_state
        self._llm_classifier = llm_classifier
        self._classifier = QuestionClassifier(
            force_project_mode=config.rag.force_project_mode
        )
        # 当前会话内的判定结果缓存：键是 normalize_text 后的题面，
        # 同一题重新分析（F10）复用，不再请求模型；新问题清空
        self._classification_cache: dict[str, Classification] = {}
        self._policy = RetrievalPolicy(
            enabled=config.rag.enabled,
            force_project_mode=config.rag.force_project_mode,
        )
        self._session: QuestionSession | None = None
        self._classification: Classification | None = None
        # 当前会话内已提交过的题面规范化形式，用于去重。
        # 集合按会话累积：会话内提交过的题面都留在里面，新问题才清空
        self._seen_normalized: set[str] = set()
        # 当前会话保留的语音文字，截图到达时与图片合成一条请求
        self._voice_text: str = ""
        # 当前正在跑的请求标识，旧请求的事件按它丢弃
        self._active_request_id: str | None = None
        # 当前请求的 asyncio 任务，新问题到达时取消它
        self._request_task: asyncio.Task[None] | None = None
        # 当前正在跑的检索任务，新问题到达或重新分析时取消它
        self._retrieval_task: asyncio.Task[tuple[str | None, list[dict]]] | None = None
        # 检索任务的请求标识，取消时按它找受管任务
        self._retrieval_request_id: str | None = None
        # 受管任务注册表：请求任务与检索任务都登记在这里，
        # 取消走它，超时也由它兜底，见《8-集成与验收》请求取消与超时管理两节
        self._tasks = TaskManager(on_client_cancel=self._cancel_client_request)

    @property
    def current_session(self) -> QuestionSession | None:
        """当前问题会话，没有会话时返回 None。"""
        return self._session

    @property
    def active_request_id(self) -> str | None:
        """当前正在跑的请求标识，没有在跑的请求时返回 None。"""
        return self._active_request_id

    @property
    def voice_text(self) -> str:
        """当前会话保留的语音文字，没有时是空串。"""
        return self._voice_text

    @property
    def managed_tasks(self) -> list[TaskInfo]:
        """受管任务的信息列表，给状态展示与健康检查用。

        请求任务与检索任务都登记在受管任务注册表里，
        见《8-集成与验收》请求取消与超时管理两节。
        """
        return self._tasks.list_tasks()

    def set_voice_text(self, text: str) -> None:
        """记下当前会话的语音文字。

        语音文字由语音输入侧写入，截图到达时与图片合成一条请求一次发出，
        见《4-截图输入与多模态》与语音合并一节。

        :param text: 语音识别出来的文字，空串表示没有语音文字。
        """
        self._voice_text = text.strip()
        log_event(
            "question.voice_text_set",
            "已记下当前会话的语音文字",
            level="DEBUG",
            text_length=len(self._voice_text),
        )

    async def submit_text(
        self,
        text: str,
        image_data_url: str | None = None,
        *,
        dialogue_context: str | None = None,
    ) -> None:
        """提交一条题面，可以带一张截图。

        空文本且没有图直接返回，不建会话也不发请求；同一会话内规范化后重复的
        题面直接忽略，只写日志，不重复请求；其余情况新建会话并发起一次流式请求。

        带图提交不受文本去重限制：纯截图时题面为空，再按一次 F9 要能发新请求，
        见《4-截图输入与多模态》与语音合并一节。带图时文字取当前会话保留的
        语音文字，文字与图片合成一条消息一次发出。

        策略说要检索时先发起检索，把项目资料拼进请求；检索拿不到结果时
        照样发请求，见《6-Dify检索与项目知识》故障降级一节。

        ``dialogue_context`` 是项目模式的对话日志上下文：
        ``DialogueLog.recent_text()`` 的全文，含角色标注。给了它时组装
        请求会把日志拼进 user 消息靠前位置、问题之前，并把两句项目模式
        指令附加在检索内容之后；不给时（普通模式）行为零变化，
        见《5-语音输入与流式ASR》项目模式一节。

        :param text: 用户输入的题面。
        :param image_data_url: 截图的 data URL，没有截图时传 None。
        :param dialogue_context: 对话日志上下文，普通模式不传。
        """
        stripped = text.strip()
        has_image = bool(image_data_url)
        if not stripped and not has_image:
            log_event(
                "question.submit_empty",
                "提交内容为空，不建会话也不发请求",
                level="DEBUG",
            )
            return

        # 带图时文字取会话保留的语音文字，合并发生在这里，
        # 见《4-截图输入与多模态》与语音合并一节
        effective_text = stripped or self._voice_text.strip()

        normalized = normalize_text(effective_text)
        if (
            not has_image
            and self._session is not None
            and normalized in self._seen_normalized
        ):
            log_event(
                "question.submit_duplicated",
                "同一会话内重复提交，已忽略",
                session_id=self._session.session_id,
                request_id=self._session.request_id,
            )
            return

        # 新问题到达：先取消旧请求并清空界面，旧请求后续到达的事件按 request_id 丢弃
        self._abort_active_request("新问题到达")

        session = QuestionSession(
            session_id=str(uuid.uuid4()),
            created_at=time.time(),
            source="screenshot" if has_image else "text",
            text=effective_text or None,
            image_data_url=image_data_url,
            request_id=str(uuid.uuid4()),
            dialogue_context=(dialogue_context or "").strip(),
        )
        self._session = session
        if not has_image:
            # 只有文本提交写进去重集合。集合是会话级累积键（规范化文本），
            # 新问题（``new_question``）才清空：会话里的 A、B、A 不会把 A
            # 提交两次，会话标识由这个集合的生命周期隐式承载，
            # 见《8-集成与验收》问题去重一节。
            # 带图提交既不查也不写这个集合：上面的去重豁免已经保证它能发出去，
            # 动集合会连带放掉后面本该被拦的重复文本
            self._seen_normalized.add(normalized)

        classification = await self._classify(effective_text)
        self._classification = classification
        session.question_type = classification.question_type
        session.rag_required = classification.rag_required
        log_event(
            "question.classified",
            "题型判定完成",
            session_id=session.session_id,
            request_id=session.request_id,
            question_type=classification.question_type,
            rag_required=classification.rag_required,
            matched_rules=",".join(classification.matched_rules),
            source=classification.source,
            text_length=len(effective_text),
            has_image=has_image,
        )

        # 判定出代码题就通知外面，让右栏先切到代码形态。
        # 通知在发出模型请求之前，模型一个协议事件都不给时，
        # 后到的围栏块也能进右栏代码区，见《7-代码问题模式》纯代码块兜底一节
        self._notify_coding_mode(classification.question_type)

        # 策略说要检索就发起检索，把上下文拼进模型请求，
        # 见《6-Dify检索与项目知识》检索流程一节
        decision = self._policy.decide(classification)
        log_event(
            "question.retrieval_decided",
            decision.reason,
            session_id=session.session_id,
            request_id=session.request_id,
            should_retrieve=decision.should_retrieve,
        )

        context, _sources = await self._retrieve_context(
            session, effective_text, decision.should_retrieve
        )

        # 截图提交不等整条流结束：采集与提交链路不能被生成拖住，
        # 见《4-截图输入与多模态》截图流程一节。文本提交仍等请求收尾，
        # 界面靠这个等待收尾状态。
        await self._start_request(
            session,
            classification.question_type,
            context,
            wait=not has_image,
        )

    async def reanalyze(self) -> None:
        """用当前会话同一份材料重新生成一次。

        不受去重限制，这是用户明确要求的重新分析。
        没有当前会话，或会话里既没有题面也没有截图时直接返回。
        带截图的会话重新分析时复用同一张图，
        见《8-集成与验收》重新分析时用同一份材料一节。
        策略说要检索时按同一份题面重新检索一次。
        """
        session = self._session
        if session is None or (not session.text and not session.image_data_url):
            log_event(
                "question.reanalyze_skipped",
                "当前没有可重新分析的问题",
                level="DEBUG",
            )
            return

        self._abort_active_request("重新分析")

        classification = await self._classify(session.text or "")
        self._classification = classification
        session.question_type = classification.question_type
        session.rag_required = classification.rag_required
        session.request_id = str(uuid.uuid4())

        decision = self._policy.decide(classification)
        log_event(
            "question.reanalyze",
            "用同一份材料重新分析",
            session_id=session.session_id,
            request_id=session.request_id,
            question_type=classification.question_type,
            should_retrieve=decision.should_retrieve,
        )
        # 重新分析同样在发出模型请求之前通知代码题判定
        self._notify_coding_mode(classification.question_type)
        # 重新分析同样按策略重新检索，检索期间新问题到达时这次检索会被取消
        context, _sources = await self._retrieve_context(
            session, session.text or "", decision.should_retrieve
        )
        await self._start_request(session, classification.question_type, context)

    async def _classify(self, text: str) -> Classification:
        """判定题型与检索需求：优先模型判定，失败回退同步规则。

        注入的模型判定器抛异常或超时时记一条 WARNING，再用同步规则判一次，
        提交流程不受影响。结果按 ``normalize_text(text)`` 缓存在当前会话内，
        同一题重新分析（F10）直接复用，不再请求模型；``new_question`` 清空缓存。

        :param text: 用户输入的题面。
        :return: 判定结果。
        """
        key = normalize_text(text)
        cached = self._classification_cache.get(key)
        if cached is not None:
            return cached

        classifier = self._llm_classifier
        if classifier is None:
            result = self._classifier.classify(text)
        else:
            try:
                result = await classifier.classify(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    "question.classify_fallback",
                    f"模型题型判定失败，回退同步规则：{exc}",
                    level="WARNING",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    text_length=len(text),
                    session_id=self._session.session_id if self._session else None,
                )
                result = self._classifier.classify(text)

        self._classification_cache[key] = result
        return result

    async def cancel(self) -> None:
        """取消当前正在跑的请求。

        幂等：没有在跑的请求时直接返回，不报错。
        检索也在进行中时一并取消。

        取消走受管任务注册表：任务本身被取消，同时通知客户端关流，
        见《8-集成与验收》请求取消一节。
        """
        request_id = self._active_request_id
        task = self._request_task
        retrieval_id = self._retrieval_request_id
        retrieval_task = self._retrieval_task
        if request_id is None and task is None and retrieval_task is None:
            log_event("question.cancel_noop", "当前没有在跑的请求", level="DEBUG")
            return
        self._active_request_id = None
        self._request_task = None
        self._retrieval_task = None
        self._retrieval_request_id = None
        if request_id is not None:
            # 受管任务注册表会取消任务本身，并通知客户端关流
            await self._tasks.cancel(request_id)
            # 取消了就不在跑模型，状态回到 idle
            self._notify_model_state(MODEL_STATE_IDLE)
        else:
            self._cancel_task(task)
        if retrieval_id is not None:
            await self._tasks.cancel(retrieval_id)
        else:
            self._cancel_task(retrieval_task)
        log_event(
            "question.cancelled",
            "已取消当前请求",
            request_id=request_id,
            session_id=self._session.session_id if self._session else None,
        )

    def new_question(self) -> None:
        """创建新问题：放弃当前会话并开一个新会话。

        旧请求被取消，界面被清空，语音文字也不带到新会话里。
        """
        self._abort_active_request("创建新问题")
        self._session = None
        self._classification = None
        self._classification_cache.clear()
        self._seen_normalized.clear()
        self._voice_text = ""
        log_event("question.new", "已创建新问题")

    async def _start_request(
        self,
        session: QuestionSession,
        question_type: str,
        context: str | None = None,
        *,
        wait: bool = True,
    ) -> None:
        """按请求新建解析器，起一个任务跑这次请求。

        决定发这次请求的时刻在这里取，请求发起耗时从它算起，
        见《8-集成与验收》性能统计一节。检索在更早之前就完成了，
        检索耗时单独记在 ``retrieval_latency`` 里，不算进请求发起耗时。

        请求任务登记在受管任务注册表里，带一个总时限：
        时限是 ``model.read_timeout`` 加一分钟余量。``read_timeout``
        是两次读之间的间隔，整条流本来可以更久，但请求不能放在一个无限等待的
        任务里，所以再宽也留一个上限，见《8-集成与验收》超时管理一节。

        :param session: 当前问题会话。
        :param question_type: 题型标识。
        :param context: 检索到的上下文，没有就传 None。
        :param wait: 是否等这条流跑完再返回。文本提交与重新分析要等，
            界面靠它收尾；截图提交传假，采集与提交链路不能被生成拖住，
            见《4-截图输入与多模态》截图流程一节。
        """
        request_id = session.request_id or str(uuid.uuid4())
        self._active_request_id = request_id
        # 决定发这次请求的时刻
        decided_at = time.perf_counter()
        parser = self._parser_factory(request_id, session.session_id)
        if session.image_data_url:
            # 带图请求：文字与图片合成一条消息，见《4-截图输入与多模态》多模态请求一节
            messages = self._multimodal_messages(
                session.text or "", session.image_data_url, question_type, context
            )
        else:
            messages = self._build_request_messages(
                session.text or "", question_type, context
            )
        # 请求发出前状态是 connecting，见《2-骨架与配置》状态字段取值表
        self._notify_model_state(MODEL_STATE_CONNECTING, request_id)
        handle = self._tasks.register(
            request_id,
            lambda: self._run_stream(session, request_id, messages, parser, decided_at),
            timeout=self._request_timeout(),
        )
        task = handle.task
        self._request_task = task
        if not wait:
            # 提交路径不等待整条流结束：任务已登记在受管任务注册表里，
            # 结束或取消后由回调清掉当前任务引用，提交动作立即返回
            task.add_done_callback(self._release_request_task)
            return
        try:
            await task
        finally:
            if self._request_task is task:
                self._request_task = None
            # 清掉已经结束的登记，注册表不一直长
            self._tasks.sweep()

    def _release_request_task(self, task: asyncio.Task[None]) -> None:
        """截图提交不等流结束时，任务结束后清掉当前任务引用。

        只清自己这一个：之后新提交的任务不能被旧任务的收尾误清掉。
        """
        if self._request_task is task:
            self._request_task = None
        # 清掉已经结束的登记，注册表不一直长
        self._tasks.sweep()

    def _request_timeout(self) -> float:
        """请求任务的总时限（秒）。

        取 ``model.read_timeout`` 再加一分钟余量：``read_timeout`` 限的是
        两次读之间的间隔，整条流可以更久；余量给的是长回答的空间，
        但请求不能放在一个无限等待的任务里，见《8-集成与验收》超时管理一节。
        """
        return float(self._config.model.read_timeout) + _REQUEST_TIMEOUT_MARGIN

    def _build_request_messages(
        self,
        question: str,
        question_type: str,
        context: str | None,
    ) -> list[dict[str, str]]:
        """组装文本请求的消息，有检索上下文时带上它。

        检索上下文是回调的末位参数，没有上下文时按老签名调用，
        兼容不带 context 参数的回调。

        项目模式（会话带对话日志上下文）时在组装前先拼：
        日志进 user 消息靠前位置、问题之前，两句指令拼在检索内容之后
        （没检索到内容时拼在问题之后单独成段，检索上下文传 None）。
        普通模式两步都原样返回，见《5-语音输入与流式ASR》项目模式一节。
        """
        if not self._dialogue_context():
            # 普通模式：行为零变化
            if context is None:
                return self._build_messages(question, question_type)
            return self._build_messages(question, question_type, context)
        if context and context.strip():
            # 检索到内容：日志进问题之前，指令跟在检索内容后面
            question = self._question_with_dialogue_context(question)
            context = f"{context.strip()}\n\n{DIALOGUE_CONTEXT_NOTES}"
            return self._build_messages(question, question_type, context)
        # 没检索到内容（含降级）：指令单独成段跟在问题后面，不带检索段
        question = self._question_with_dialogue_context(
            question, notes=DIALOGUE_CONTEXT_NOTES
        )
        return self._build_messages(question, question_type)

    def _multimodal_messages(
        self,
        question: str,
        image_data_url: str,
        question_type: str,
        context: str | None = None,
    ) -> list[dict]:
        """组装带图请求的消息。

        用装配层注入的回调，core 层不直接 import llm 层，
        见《2-骨架与配置》目录结构约定一节。
        检索上下文是回调的末位参数，没有上下文时按老签名调用，
        兼容不带 context 参数的回调。

        项目模式的对话日志拼装与文本请求同一条路：日志进 user 靠前
        位置、指令拼在检索内容之后（没检索到内容时拼在问题后面），
        见《5-语音输入与流式ASR》项目模式一节。
        """
        if not self._dialogue_context():
            # 普通模式：行为零变化
            if context is None:
                return self._build_multimodal_messages(
                    question, image_data_url, question_type
                )
            return self._build_multimodal_messages(
                question, image_data_url, question_type, context
            )
        if context and context.strip():
            question = self._question_with_dialogue_context(question)
            context = f"{context.strip()}\n\n{DIALOGUE_CONTEXT_NOTES}"
            return self._build_multimodal_messages(
                question, image_data_url, question_type, context
            )
        question = self._question_with_dialogue_context(
            question, notes=DIALOGUE_CONTEXT_NOTES
        )
        return self._build_multimodal_messages(
            question, image_data_url, question_type
        )

    def _dialogue_context(self) -> str:
        """当前会话注入的对话日志上下文，普通模式是空串。"""
        session = self._session
        if session is None:
            return ""
        return (session.dialogue_context or "").strip()

    def _question_with_dialogue_context(self, question: str, notes: str = "") -> str:
        """把对话日志拼进问题文本的靠前位置。

        user 消息的最终顺序是：对话日志 → 本轮问题 → 检索内容（由
        build 回调把 context 追加在问题之后）。检索没取到内容时
        ``notes`` 给出项目模式的附加指令，跟在问题后面单独成段。
        没有注入日志、题面为空时原样返回，普通模式行为零变化。

        :param question: 本轮问题文本。
        :param notes: 可选的附加指令，拼在问题之后单独成段。
        """
        log_text = self._dialogue_context()
        if not log_text or not question.strip():
            return question
        text = f"{DIALOGUE_CONTEXT_TITLE}：\n{log_text}\n\n问题：{question.strip()}"
        if notes:
            text = f"{text}\n\n{notes}"
        return text

    async def _run_stream(
        self,
        session: QuestionSession,
        request_id: str,
        messages: list[dict],
        parser: StreamParser,
        decided_at: float,
    ) -> None:
        """跑一次流式请求，把解析出的事件交给事件分发器。

        带图的会话走 ``multimodal_stream``，其余走 ``chat_stream``。
        客户端异常在这里被吃掉：写日志并调 ``report_error``，不往调用方抛。
        任务被取消时抛出的 ``CancelledError`` 放行，不当成普通异常吞掉。

        模型状态也在这里往外报：收到第一个片段是 streaming，正常结束回到 idle，
        出错是 error，见《2-骨架与配置》状态字段取值表。

        这次请求的耗时指标也在这里记，指标名与口径见《8-集成与验收》
        性能统计一节：``request_latency`` 从决定发请求算起，其余四个
        从请求交到客户端算起。

        :param session: 当前问题会话。
        :param request_id: 本次请求的标识。
        :param messages: 已经组装好的请求消息。
        :param parser: 本次请求的解析器。
        :param decided_at: 决定发这次请求的时刻，单调时钟秒数。
        """
        metrics = _RequestMetrics(
            decided_at,
            source=self._metric_source(),
            recorder=default_recorder(),
        )
        try:
            if session.image_data_url:
                stream = self._client.multimodal_stream(messages, request_id)
            else:
                stream = self._client.chat_stream(messages, request_id)
            iterator = stream.__aiter__()
            while True:
                # 请求交到客户端、开始拉流的那一刻，只记第一次
                metrics.mark_sent()
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    # 流正常结束：记完整输出耗时
                    metrics.record_total()
                    break
                metrics.mark_first_chunk()
                if request_id != self._active_request_id:
                    # 旧请求的残留内容，不再喂给解析器；
                    # 请求已经被取消，没有完整输出，不记完整输出耗时
                    break
                # 收到第一个片段，状态从 connecting 变成 streaming
                self._notify_model_state(MODEL_STATE_STREAMING, request_id)
                for event in parser.feed(chunk):
                    if self._dispatch_event(event, request_id):
                        metrics.observe_event(event)
            if request_id == self._active_request_id:
                for event in parser.flush():
                    if self._dispatch_event(event, request_id):
                        metrics.observe_event(event)
        except asyncio.CancelledError:
            # 取消：没有完整输出，不记 total_latency；
            # 已经记下的 ttft 等保留，见《8-集成与验收》请求取消一节
            self._notify_model_state(MODEL_STATE_IDLE, request_id)
            raise
        except InterviewMateError as exc:
            # 流异常中断：已经发生的耗时是事实，照样记一条 total_latency
            metrics.record_total()
            log_event(
                "question.request_failed",
                f"模型请求失败：{exc.message}",
                level="ERROR",
                request_id=request_id,
                session_id=session.session_id,
                code=exc.code,
            )
            self._notify_model_state(MODEL_STATE_ERROR, request_id)
            self._sink.report_error(exc.code, exc.message)
        except Exception as exc:
            # 流异常中断：已经发生的耗时是事实，照样记一条 total_latency
            metrics.record_total()
            log_event(
                "question.request_failed",
                f"模型请求失败：{exc}",
                level="ERROR",
                request_id=request_id,
                session_id=session.session_id,
                code="E_MODEL_STREAM",
            )
            self._notify_model_state(MODEL_STATE_ERROR, request_id)
            self._sink.report_error("E_MODEL_STREAM", str(exc))
        else:
            # 正常结束：状态回到 idle
            self._notify_model_state(MODEL_STATE_IDLE, request_id)
        finally:
            if self._active_request_id == request_id:
                self._active_request_id = None

    def _notify_model_state(
        self,
        state: str,
        request_id: str | None = None,
    ) -> None:
        """往外报模型状态。

        core 层只报状态，不认识界面：外面注入的回调把状态写到状态栏，
        见《2-骨架与配置》状态字段取值表。没有注入回调时什么都不做。

        给了请求标识时，只有这个请求还是当前请求才报：
        旧请求收尾时不改新请求的状态。

        :param state: 状态字符串，取值见 :data:`MODEL_STATE_IDLE` 等常量。
        :param request_id: 本次请求的标识，None 时不判断新旧。
        """
        if request_id is not None and request_id != self._active_request_id:
            return
        callback = self._on_model_state
        if callback is None:
            return
        try:
            callback(state)
        except Exception as exc:
            # 状态回调挂了不影响请求本身，只记一条日志
            log_event(
                "question.model_state_failed",
                f"模型状态上报失败：{exc}",
                level="WARNING",
                error=str(exc),
                state=state,
            )

    def _metric_source(self) -> str:
        """这次请求的耗时指标来源。

        优先取注入客户端上的 ``name``；客户端没给就取配置里
        ``providers.llm`` 的取值（``mock`` 或 ``openai``）。
        装配层按这个值选客户端，见《2-骨架与配置》配置结构一节。
        """
        name = getattr(self._client, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
        return self._config.providers.llm

    def _dispatch_event(self, event: StreamEvent, request_id: str) -> bool:
        """把事件交给分发器，不属于当前请求的事件直接丢弃。

        :param event: 解析器产出的事件。
        :param request_id: 本次请求的标识。
        :returns: 事件真的派发出去时返回 True，被丢弃时返回 False。
        """
        event_request_id = getattr(event, "request_id", None)
        if request_id != self._active_request_id or event_request_id != request_id:
            # 旧请求后续到达的事件，不更新界面
            log_event(
                "question.event_dropped",
                "事件不属于当前请求，已丢弃",
                level="DEBUG",
                request_id=request_id,
                session_id=self._session.session_id if self._session else None,
                event_request_id=event_request_id,
            )
            return False
        self._sink.dispatch(event)
        return True

    def _cancel_task(self, task: asyncio.Task[Any] | None) -> None:
        """取消请求任务，让旧协程不再往解析器里喂文本。"""
        if task is None or task.done():
            return
        task.cancel()

    async def _cancel_client_request(self, request_id: str) -> None:
        """取消客户端侧的请求流。

        客户端的 ``cancel`` 是协程，这里 await 它；失败只记日志，不影响取消语义。
        """
        try:
            await self._client.cancel(request_id)
        except Exception as exc:
            log_event(
                "question.cancel_failed",
                f"取消请求失败：{exc}",
                level="WARNING",
                request_id=request_id,
                session_id=self._session.session_id if self._session else None,
            )

    def _schedule_client_cancel(self, request_id: str, reason: str) -> None:
        """把客户端侧的取消排进事件循环。

        ``_abort_active_request`` 是同步方法，await 不了客户端的取消协程，
        就排一个任务让它跑。请求任务本身已经先被取消，这里只是补上关流。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环，没有地方排任务，直接跳过
            return
        loop.create_task(self._cancel_client_request(request_id))
        log_event(
            "question.client_cancel_scheduled",
            "已排入客户端侧取消",
            level="DEBUG",
            request_id=request_id,
            reason=reason,
        )

    def _abort_active_request(self, reason: str) -> None:
        """取消在跑的请求并清空界面，新问题、重新分析与新建问题都走这里。

        检索也在进行中时一并取消，旧检索的结果不再进入新会话。
        """
        request_id = self._active_request_id
        task = self._request_task
        retrieval_task = self._retrieval_task
        self._active_request_id = None
        self._request_task = None
        self._retrieval_task = None
        self._retrieval_request_id = None
        self._cancel_task(task)
        self._cancel_task(retrieval_task)
        if request_id is not None:
            self._schedule_client_cancel(request_id, reason)
        self._sink.clear()

    # ------------------------------------------------------------------
    # 检索，见《6-Dify检索与项目知识》检索流程与故障降级两节
    # ------------------------------------------------------------------
    async def _retrieve_context(
        self,
        session: QuestionSession,
        query: str,
        should_retrieve: bool,
    ) -> tuple[str | None, list[dict]]:
        """按决策发起检索，返回上下文文本与来源条目。

        检索失败、检索结果为空或没有注入检索回调时都返回空上下文，
        并在会话上标记没取到项目检索结果，请求照常发出，
        见《6-Dify检索与项目知识》故障降级一节。

        :param session: 当前问题会话。
        :param query: 检索查询，用题面。
        :param should_retrieve: 策略给出的检索决策。
        :return: 上下文文本与来源条目列表，都拿不到时上下文是 None。
        """
        session.retrieval_sources = []
        session.retrieval_missing = False
        if not should_retrieve:
            log_event(
                "question.retrieval_skipped",
                "这次不检索项目资料",
                level="DEBUG",
                session_id=session.session_id,
                request_id=session.request_id,
            )
            return None, []

        retrieve = self._retrieve
        if retrieve is None:
            log_event(
                "question.retrieval_unavailable",
                "没有注入检索回调，不发起检索",
                level="WARNING",
                session_id=session.session_id,
                request_id=session.request_id,
            )
            session.retrieval_missing = True
            return None, []

        if not query.strip():
            log_event(
                "question.retrieval_skipped",
                "题面为空，没有检索查询",
                level="DEBUG",
                session_id=session.session_id,
                request_id=session.request_id,
            )
            session.retrieval_missing = True
            return None, []

        retrieval_id = f"retrieve:{session.request_id}"
        handle = self._tasks.register(
            retrieval_id,
            lambda: self._run_retrieve(retrieve, query),
            timeout=float(self._config.rag.timeout),
        )
        task = handle.task
        self._retrieval_task = task
        self._retrieval_request_id = retrieval_id
        try:
            context, sources = await task
        except asyncio.CancelledError:
            # 检索超时：受管任务注册表到点取消了这次检索。
            # 这按「没取到项目资料」处理，请求照常发出去，
            # 见《6-Dify检索与项目知识》故障降级一节。
            # 用户取消（新问题到达）时放行取消，不当降级处理。
            info = self._tasks.get(retrieval_id)
            if info is None or not info.expired:
                raise
            log_event(
                "question.retrieval_timeout",
                "检索超时，按没取到项目资料处理",
                level="WARNING",
                session_id=session.session_id,
                request_id=session.request_id,
            )
            session.retrieval_missing = True
            return None, []
        except Exception as exc:
            # 检索失败不让问题处理崩掉，继续走不带上下文的请求
            log_event(
                "question.retrieval_failed",
                f"检索失败：{exc}",
                level="WARNING",
                session_id=session.session_id,
                request_id=session.request_id,
                error_code=getattr(exc, "code", None),
            )
            session.retrieval_missing = True
            return None, []
        finally:
            if self._retrieval_task is task:
                self._retrieval_task = None
                self._retrieval_request_id = None
            # 清掉已经结束的登记，注册表不一直长
            self._tasks.sweep()

        normalized_sources = _normalize_sources(sources)
        context_text = context.strip() if isinstance(context, str) else ""
        session.retrieval_sources = normalized_sources
        session.retrieval_missing = not context_text
        if normalized_sources:
            # 有片段就送给界面，模型输出结束后按 F3 还能再查
            self._notify_sources(normalized_sources)
        if not context_text:
            # Dify 没检索到相关内容时用通用知识回答，不伪造项目内部实现
            log_event(
                "question.retrieval_empty",
                "没有取到项目检索结果，按通用知识回答",
                level="INFO",
                session_id=session.session_id,
                request_id=session.request_id,
                source_count=len(normalized_sources),
            )
            return None, normalized_sources

        log_event(
            "question.retrieval_ok",
            "检索完成",
            session_id=session.session_id,
            request_id=session.request_id,
            source_count=len(normalized_sources),
            context_length=len(context_text),
        )
        return context_text, normalized_sources

    async def _run_retrieve(
        self,
        retrieve: RetrievalFunc,
        query: str,
    ) -> tuple[str | None, list[dict]]:
        """调用注入的检索回调，拿回上下文与来源条目。"""
        log_event(
            "question.retrieval_started",
            "开始检索项目资料",
            session_id=self._session.session_id if self._session else None,
            request_id=self._session.request_id if self._session else None,
        )
        context, sources = await retrieve(query)
        return context, sources

    def _notify_sources(self, sources: list[dict]) -> None:
        """把来源条目送给界面，界面拿不到也不算失败。"""
        if self._on_sources is None or not sources:
            return
        try:
            self._on_sources(sources)
        except Exception as exc:
            log_event(
                "question.sources_notify_failed",
                f"检索来源送不出去：{exc}",
                level="WARNING",
                session_id=self._session.session_id if self._session else None,
                request_id=self._session.request_id if self._session else None,
            )

    def _notify_coding_mode(self, question_type: str) -> None:
        """通知外面这次是不是代码题。

        判定出代码题时通知真，其余题型通知假。
        没有注入回调时不通知，外面退回到靠事件反推；
        通知不出去只记日志，不影响这次请求。

        :param question_type: 判定出的题型标识。
        """
        if self._on_coding_mode is None:
            return
        try:
            self._on_coding_mode(question_type == _CODING_QUESTION_TYPE)
        except Exception as exc:
            log_event(
                "question.coding_mode_notify_failed",
                f"代码题通知送不出去：{exc}",
                level="WARNING",
                session_id=self._session.session_id if self._session else None,
                request_id=self._session.request_id if self._session else None,
            )


def _normalize_sources(entries: Any) -> list[dict]:
    """把来源条目统一成普通字典列表。

    字段是 document_name、score、content，界面按这三个字段显示，
    见《6-Dify检索与项目知识》检索来源展示一节。
    core 层不 import 检索层的实现类，条目是带同名属性的对象时按属性取。

    :param entries: 来源条目列表。
    :return: 普通字典列表，取不出字段的条目丢掉。
    """
    if not entries:
        return []
    normalized: list[dict] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            normalized.append(
                {
                    "document_name": str(entry.get("document_name") or ""),
                    "score": entry.get("score"),
                    "content": str(entry.get("content") or ""),
                }
            )
            continue
        document_name = getattr(entry, "document_name", "")
        content = getattr(entry, "content", "")
        if not document_name and not content:
            continue
        normalized.append(
            {
                "document_name": str(document_name or ""),
                "score": getattr(entry, "score", None),
                "content": str(content or ""),
            }
        )
    return normalized
