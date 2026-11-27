"""问题会话编排的测试。

覆盖《3-文本输入与流式回答》问题会话一节与《8-集成与验收》请求取消、
防止重复调用两节：空提交不建会话、同会话去重、新问题取消旧请求并清空、
旧请求的任务与事件被丢弃、客户端异常不外抛、重新分析复用同一份材料。

另外覆盖《4-截图输入与多模态》与语音合并一节：带图提交合法、
文字与图片在同一条消息里、连续两次带图提交不被去重拦、重新分析复用同一张图。

模型客户端、解析器与事件分发器都用假实现，验证的是编排逻辑本身。
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import AppConfig, load_config
from app.core.exceptions import DeepSeekAuthError, DeepSeekTimeoutError
from app.core.question_classifier import Classification
from app.core.question_manager import QuestionManager
from app.inputs.text_input import build_text_input
from app.llm.prompt_manager import PromptManager

# 题面样例
QUESTION_ONE = "讲讲我的项目是怎么做的"
QUESTION_TWO = "解释一下快速排序"

# 截图的 data URL 与语音文字
IMAGE_DATA_URL = "data:image/png;base64,AAAA"
VOICE_TEXT = "请实现一个 LRU Cache"


class FakeEvent:
    """假事件，只带请求标识与内容。"""

    def __init__(self, request_id: str | None = None, text: str = "") -> None:
        self.request_id = request_id
        self.text = text


class FakeClient:
    """假模型客户端，记录调用与取消。"""

    def __init__(self, chunks: list[str] | None = None, error: Exception | None = None) -> None:
        self.chunks: list[str] = list(chunks or [])
        self.error: Exception | None = error
        self.calls: list[tuple[list[dict[str, str]], str]] = []
        # 多模态调用单独记，与文本调用分开
        self.multimodal_calls: list[tuple[list[dict], str]] = []
        self.cancelled: list[str] = []
        # 流式调用开始时置位，测试用它确定请求已经发起
        self.called: asyncio.Event = asyncio.Event()
        self._gate: asyncio.Event | None = None
        self._active_gate: asyncio.Event | None = None

    def block_next_call(self) -> None:
        """让下一次流式调用挂起，直到测试放行。"""
        self._gate = asyncio.Event()

    def release(self) -> None:
        """放行挂起的流式调用。"""
        if self._active_gate is not None:
            self._active_gate.set()

    async def chat_stream(self, messages: list[dict[str, str]], request_id: str):
        self.calls.append((messages, request_id))
        self.called.set()
        gate = self._gate
        if gate is not None:
            self._gate = None
            self._active_gate = gate
            await gate.wait()
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def multimodal_stream(self, messages: list[dict], request_id: str):
        self.multimodal_calls.append((messages, request_id))
        self.called.set()
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)


class FakeParser:
    """假解析器，记录喂进去的文本，产出带请求标识的假事件。"""

    def __init__(
        self,
        request_id: str | None = None,
        session_id: str | None = None,
        fixed_request_id: str | None = None,
    ) -> None:
        self.request_id = request_id
        self.session_id = session_id
        # 固定标记，用来造旧请求的事件
        self.fixed_request_id = fixed_request_id
        self.reset_count: int = 0
        self.flush_count: int = 0
        self.feeds: list[str] = []

    def reset(self) -> None:
        self.reset_count += 1

    def feed(self, text: str) -> list[FakeEvent]:
        self.feeds.append(text)
        tag = self.fixed_request_id if self.fixed_request_id is not None else self.request_id
        return [FakeEvent(request_id=tag, text=text)]

    def flush(self) -> list[FakeEvent]:
        self.flush_count += 1
        return []


class FakeSink:
    """假事件分发器，记录派发、清屏与错误。"""

    def __init__(self) -> None:
        self.dispatched: list[FakeEvent] = []
        self.clear_count: int = 0
        self.errors: list[tuple[str, str]] = []

    def dispatch(self, event: FakeEvent) -> None:
        self.dispatched.append(event)

    def clear(self) -> None:
        self.clear_count += 1

    def report_error(self, code: str, detail: str) -> None:
        self.errors.append((code, detail))


class FakeLLMClassifier:
    """假模型判定器，记录调用次数，按设定返回或抛错。"""

    def __init__(
        self,
        result: Classification | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or Classification(
            question_type="conceptual",
            rag_required=False,
            matched_rules=("llm",),
            reason="通用知识",
            source="llm",
        )
        self.error = error
        self.calls: list[str] = []

    async def classify(self, text: str) -> Classification:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.result


def make_manager(
    client: FakeClient | None = None,
    sink: FakeSink | None = None,
    stale_request_id: str | None = None,
    llm_classifier: FakeLLMClassifier | None = None,
) -> tuple[QuestionManager, FakeClient, list[FakeParser], FakeSink]:
    """搭一套假实现，返回管理器与三个假对象。

    :param stale_request_id: 非空时，工厂造出来的解析器都产出这个请求标识的事件，
        用来模拟旧请求的事件。
    """
    config: AppConfig = load_config()
    client = client if client is not None else FakeClient(chunks=["片段一", "片段二"])
    sink = sink if sink is not None else FakeSink()
    parsers: list[FakeParser] = []

    def parser_factory(request_id: str, session_id: str) -> FakeParser:
        parser = FakeParser(
            request_id=request_id,
            session_id=session_id,
            fixed_request_id=stale_request_id,
        )
        parsers.append(parser)
        return parser

    def build_messages(question: str, question_type: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": question}]

    manager = QuestionManager(
        config,
        client,
        parser_factory,
        sink,
        build_messages,
        build_multimodal_messages=PromptManager(config).build_multimodal_messages,
        llm_classifier=llm_classifier,
    )
    return manager, client, parsers, sink


async def test_submit_empty_text_creates_no_session() -> None:
    """空白文本直接返回，不建会话也不发请求。"""
    manager, client, _parsers, sink = make_manager()
    await manager.submit_text("   \n  ")
    assert manager.current_session is None
    assert client.calls == []
    assert sink.clear_count == 0


async def test_submit_text_creates_session_and_requests() -> None:
    """正常提交建会话、判定题型、发一次请求并派发事件。"""
    manager, client, parsers, sink = make_manager()
    await manager.submit_text(QUESTION_ONE)
    session = manager.current_session
    assert session is not None
    assert session.text == QUESTION_ONE
    assert session.source == "text"
    assert session.request_id is not None
    assert session.question_type == "project"
    assert session.rag_required is True
    assert len(client.calls) == 1
    messages, request_id = client.calls[0]
    assert request_id == session.request_id
    assert messages == [{"role": "user", "content": QUESTION_ONE}]
    assert len(parsers) == 1
    assert parsers[0].request_id == session.request_id
    assert parsers[0].session_id == session.session_id
    assert parsers[0].feeds == ["片段一", "片段二"]
    assert parsers[0].reset_count == 0
    assert len(sink.dispatched) == 2
    assert sink.dispatched[0].request_id == session.request_id


async def test_duplicate_text_in_same_session_is_ignored() -> None:
    """同一会话内规范化后重复的文本被忽略，不重复请求。"""
    manager, client, _parsers, _sink = make_manager()
    await manager.submit_text(QUESTION_ONE)
    await manager.submit_text(QUESTION_ONE)
    assert len(client.calls) == 1


async def test_duplicate_text_with_different_punctuation_is_ignored() -> None:
    """空白与标点差异不算新题面。"""
    manager, client, _parsers, _sink = make_manager()
    await manager.submit_text(QUESTION_ONE)
    await manager.submit_text("  讲讲我的项目,是怎么做的。  ")
    assert len(client.calls) == 1


async def test_repeating_an_earlier_question_is_still_ignored() -> None:
    """A、B、A：会话内去重集合累积，A 不会被提交第二次。"""
    manager, client, _parsers, _sink = make_manager()
    await manager.submit_text(QUESTION_ONE)
    await manager.submit_text(QUESTION_TWO)
    await manager.submit_text(QUESTION_ONE)
    assert len(client.calls) == 2


async def test_new_submit_cancels_old_request_and_clears() -> None:
    """新问题到达时取消旧请求、旧任务，并清空界面。"""
    manager, client, _parsers, sink = make_manager()
    client.block_next_call()
    task = asyncio.create_task(manager.submit_text(QUESTION_ONE))
    await client.called.wait()
    old_request_id = client.calls[0][1]
    assert manager.active_request_id == old_request_id

    await manager.submit_text(QUESTION_TWO)
    assert client.cancelled == [old_request_id]
    assert sink.clear_count >= 1
    assert manager.current_session is not None
    assert manager.current_session.text == QUESTION_TWO

    # 旧协程被取消，CancelledError 放行
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    client.release()


async def test_old_request_task_is_cancelled() -> None:
    """旧请求的 asyncio 任务被取消，不再往解析器里喂文本。"""
    manager, client, parsers, _sink = make_manager()
    client.block_next_call()
    task = asyncio.create_task(manager.submit_text(QUESTION_ONE))
    await client.called.wait()

    await manager.submit_text(QUESTION_TWO)
    with pytest.raises(asyncio.CancelledError):
        await task
    # 旧请求的解析器没有收到第二段文本
    assert parsers[0].feeds == []
    assert parsers[1].feeds == ["片段一", "片段二"]
    client.release()


async def test_events_of_old_request_are_dropped() -> None:
    """事件请求标识与当前请求不一致时被丢弃，不派发给界面。"""
    manager, client, _parsers, sink = make_manager(stale_request_id="旧请求标识")
    await manager.submit_text(QUESTION_ONE)
    assert sink.dispatched == []
    assert len(client.calls) == 1


async def test_each_request_uses_its_own_parser() -> None:
    """每个请求按 (request_id, session_id) 新建解析器。"""
    manager, _client, parsers, _sink = make_manager()
    await manager.submit_text(QUESTION_ONE)
    await manager.submit_text(QUESTION_TWO)
    assert len(parsers) == 2
    assert parsers[0].request_id != parsers[1].request_id
    assert parsers[0].session_id != parsers[1].session_id


async def test_client_error_reports_error_without_raising() -> None:
    """客户端异常走 report_error，不抛给调用方。"""
    client = FakeClient(chunks=["片段"], error=DeepSeekAuthError("密钥无效"))
    manager, _client, _parsers, sink = make_manager(client=client)
    await manager.submit_text(QUESTION_ONE)
    assert sink.errors == [("E_DEEPSEEK_AUTH", "密钥无效")]


async def test_client_timeout_reports_error_without_raising() -> None:
    """超时同样走 report_error。"""
    client = FakeClient(error=DeepSeekTimeoutError("读取超时"))
    manager, _client, _parsers, sink = make_manager(client=client)
    await manager.submit_text(QUESTION_ONE)
    assert sink.errors == [("E_DEEPSEEK_TIMEOUT", "读取超时")]


async def test_reanalyze_reuses_same_material() -> None:
    """重新分析用同一份材料再生成一次，不受去重限制。"""
    manager, client, parsers, _sink = make_manager()
    await manager.submit_text(QUESTION_ONE)
    session = manager.current_session
    assert session is not None
    first_messages, _first_request_id = client.calls[0]

    await manager.reanalyze()
    assert len(client.calls) == 2
    second_messages, second_request_id = client.calls[1]
    assert second_messages == first_messages
    assert second_request_id != client.calls[0][1]
    assert len(parsers) == 2
    assert manager.current_session is not None
    assert manager.current_session.session_id == session.session_id
    assert manager.current_session.text == QUESTION_ONE


async def test_reanalyze_without_session_does_nothing() -> None:
    """没有当前会话时重新分析直接返回。"""
    manager, client, _parsers, _sink = make_manager()
    await manager.reanalyze()
    assert client.calls == []
    assert manager.current_session is None


async def test_cancel_is_idempotent() -> None:
    """取消幂等，没有在跑的请求时调用不报错。"""
    manager, client, _parsers, _sink = make_manager()
    await manager.cancel()
    await manager.cancel()
    assert client.cancelled == []


async def test_cancel_stops_active_request() -> None:
    """取消当前请求后，客户端收到取消通知，任务被取消。"""
    manager, client, _parsers, _sink = make_manager()
    client.block_next_call()
    task = asyncio.create_task(manager.submit_text(QUESTION_ONE))
    await client.called.wait()
    request_id = client.calls[0][1]

    await manager.cancel()
    assert client.cancelled == [request_id]
    with pytest.raises(asyncio.CancelledError):
        await task
    client.release()


async def test_new_question_clears_session_and_view() -> None:
    """创建新问题放弃当前会话并清空界面。"""
    manager, _client, _parsers, sink = make_manager()
    await manager.submit_text(QUESTION_ONE)
    manager.new_question()
    assert manager.current_session is None
    assert sink.clear_count >= 1


async def test_retrieval_disabled_does_not_change_request() -> None:
    """检索开关关闭时照样发请求，只是不检索。"""
    config = load_config()
    config.rag.enabled = False
    client = FakeClient(chunks=["片段"])
    manager = QuestionManager(
        config,
        client,
        lambda request_id, session_id: FakeParser(),
        FakeSink(),
        lambda q, t: [],
        build_multimodal_messages=PromptManager(config).build_multimodal_messages,
    )
    await manager.submit_text(QUESTION_ONE)
    assert len(client.calls) == 1


def test_multimodal_builder_is_required() -> None:
    """没注入 build_multimodal_messages 时构造直接报 TypeError。

    core 层不直接 import llm 层，提示词由装配层注入，
    见《2-骨架与配置》目录结构约定一节。
    """
    config = load_config()
    with pytest.raises(TypeError):
        QuestionManager(
            config,
            FakeClient(),
            lambda request_id, session_id: FakeParser(),
            FakeSink(),
            lambda q, t: [],
        )


def test_build_text_input_fields() -> None:
    """文本输入入口按约定填字段。"""
    item = build_text_input("  讲讲我的项目  ", "会话一")
    assert item.session_id == "会话一"
    assert item.source == "text"
    assert item.text == "讲讲我的项目"
    assert item.image_bytes is None
    assert item.auto_submit is True
    assert item.confidence is None
    assert item.timestamp > 0


def test_build_text_input_keeps_inner_newlines() -> None:
    """多行题面只去掉首尾空白。"""
    item = build_text_input("第一行\n第二行\n", "会话二")
    assert item.text == "第一行\n第二行"


@pytest.mark.parametrize("code", ["E_DEEPSEEK_AUTH", "E_DEEPSEEK_TIMEOUT"])
async def test_report_error_codes(code: str) -> None:
    """错误码按《2-骨架与配置》错误码一节传。"""
    errors = {exc.code for exc in (DeepSeekAuthError("a"), DeepSeekTimeoutError("b"))}
    assert code in errors


# ----------------------------------------------------------------------
# 带截图的提交，见《4-截图输入与多模态》与语音合并一节
# ----------------------------------------------------------------------
async def test_empty_text_with_image_is_a_valid_submit() -> None:
    """空文本但有图时合法：建会话、走多模态接口。"""
    manager, client, _parsers, _sink = make_manager()

    await manager.submit_text("", image_data_url=IMAGE_DATA_URL)

    session = manager.current_session
    assert session is not None
    assert session.image_data_url == IMAGE_DATA_URL
    assert client.calls == []
    assert len(client.multimodal_calls) == 1
    messages, request_id = client.multimodal_calls[0]
    assert request_id == session.request_id


async def test_image_and_text_go_in_one_message() -> None:
    """语音文字与图片在同一条消息里，一次请求发出。"""
    manager, client, _parsers, _sink = make_manager()
    manager.set_voice_text(VOICE_TEXT)

    await manager.submit_text("", image_data_url=IMAGE_DATA_URL)

    assert len(client.multimodal_calls) == 1
    messages, _request_id = client.multimodal_calls[0]
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    blocks = {block["type"]: block for block in user_content}
    assert blocks["text"]["text"].startswith(f"问题：{VOICE_TEXT}")
    assert blocks["image_url"]["image_url"]["url"] == IMAGE_DATA_URL
    assert blocks["image_url"]["image_url"]["detail"] == "auto"
    # 语音文字进了会话，重新分析时复用
    assert manager.current_session is not None
    assert manager.current_session.text == VOICE_TEXT


async def test_two_image_submissions_are_not_deduplicated() -> None:
    """连续两次带图提交不被去重拦，第二次照样发出去。"""
    manager, client, _parsers, _sink = make_manager()

    await manager.submit_text("", image_data_url=IMAGE_DATA_URL)
    await manager.submit_text("", image_data_url=IMAGE_DATA_URL)

    assert len(client.multimodal_calls) == 2
    assert client.calls == []
    assert manager.current_session is not None
    assert manager.current_session.image_data_url == IMAGE_DATA_URL


async def test_image_submit_sends_even_when_voice_text_matches_last_text() -> None:
    """语音文字与上一条文本题面相同时，带图提交照样要发出去。

    带图提交不受文本去重限制，见《4-截图输入与多模态》与语音合并一节。
    """
    manager, client, _parsers, _sink = make_manager()

    await manager.submit_text(VOICE_TEXT)
    assert len(client.calls) == 1

    manager.set_voice_text(VOICE_TEXT)
    await manager.submit_text("", image_data_url=IMAGE_DATA_URL)

    assert len(client.multimodal_calls) == 1
    messages, _request_id = client.multimodal_calls[0]
    assert messages[1]["content"][0]["text"].startswith(f"问题：{VOICE_TEXT}")


async def test_text_dedup_still_works_after_an_image_submit() -> None:
    """带图提交不动文本去重集合，同一段文本再提交仍被拦。

    去重只管文本提交，见《8-集成与验收》问题去重一节。
    """
    manager, client, _parsers, _sink = make_manager()

    await manager.submit_text(VOICE_TEXT)
    assert len(client.calls) == 1

    await manager.submit_text("", image_data_url=IMAGE_DATA_URL)
    assert len(client.multimodal_calls) == 1

    # 同一段文本再提交，被去重拦掉
    await manager.submit_text(VOICE_TEXT)
    assert len(client.calls) == 1


async def test_new_image_replaces_the_old_one() -> None:
    """再按一次 F9，新截图替换旧截图。"""
    manager, client, _parsers, _sink = make_manager()
    new_url = "data:image/png;base64,BBBB"

    await manager.submit_text("", image_data_url=IMAGE_DATA_URL)
    await manager.submit_text("", image_data_url=new_url)

    assert len(client.multimodal_calls) == 2
    assert manager.current_session is not None
    assert manager.current_session.image_data_url == new_url


async def test_reanalyze_reuses_the_same_image() -> None:
    """重新分析复用同一张图。"""
    manager, client, _parsers, _sink = make_manager()
    manager.set_voice_text(VOICE_TEXT)

    await manager.submit_text("", image_data_url=IMAGE_DATA_URL)
    await manager.reanalyze()

    assert len(client.multimodal_calls) == 2
    first_messages, first_request_id = client.multimodal_calls[0]
    second_messages, second_request_id = client.multimodal_calls[1]
    assert second_request_id != first_request_id
    assert second_messages == first_messages
    assert manager.current_session is not None
    assert manager.current_session.image_data_url == IMAGE_DATA_URL


async def test_text_submit_still_uses_chat_stream() -> None:
    """纯文本提交仍走文本接口，不走多模态。"""
    manager, client, _parsers, _sink = make_manager()

    await manager.submit_text(QUESTION_ONE)

    assert len(client.calls) == 1
    assert client.multimodal_calls == []


async def test_empty_text_without_image_is_still_ignored() -> None:
    """空文本又没有图时仍然忽略，不建会话也不发请求。"""
    manager, client, _parsers, _sink = make_manager()

    await manager.submit_text("   ")

    assert manager.current_session is None
    assert client.calls == []
    assert client.multimodal_calls == []


async def test_voice_text_is_stored_and_cleared_with_new_question() -> None:
    """语音文字记在会话上，创建新问题时清掉。"""
    manager, _client, _parsers, _sink = make_manager()

    assert manager.voice_text == ""
    manager.set_voice_text(f"  {VOICE_TEXT}  ")
    assert manager.voice_text == VOICE_TEXT

    manager.new_question()
    assert manager.voice_text == ""


async def test_multimodal_client_error_does_not_raise() -> None:
    """多模态请求的客户端异常同样不外抛。"""
    client = FakeClient(chunks=["片段"], error=DeepSeekAuthError("密钥无效"))
    manager, _client, _parsers, sink = make_manager(client=client)

    await manager.submit_text("", image_data_url=IMAGE_DATA_URL)

    assert sink.errors == [("E_DEEPSEEK_AUTH", "密钥无效")]


# ----------------------------------------------------------------------
# 模型题型判定的接线：命中生效、失败回退、缓存复用
# ----------------------------------------------------------------------
async def test_llm_classification_overrides_rule_result() -> None:
    """注入模型判定器时，type 与 need_rag 以模型结果为准。"""
    classifier = FakeLLMClassifier(
        Classification(
            question_type="conceptual",
            rag_required=False,
            matched_rules=("llm",),
            reason="通用知识",
            source="llm",
        )
    )
    manager, client, _parsers, sink = make_manager(llm_classifier=classifier)

    await manager.submit_text(QUESTION_ONE)

    session = manager.current_session
    assert session is not None
    # 规则判定会得到 project/true，这里被模型结果覆盖
    assert session.question_type == "conceptual"
    assert session.rag_required is False
    assert classifier.calls == [QUESTION_ONE]
    assert len(client.calls) == 1
    assert len(sink.dispatched) == 2


async def test_llm_classification_error_falls_back_to_rules() -> None:
    """模型判定抛异常时回退同步规则，回答照常产出。"""
    classifier = FakeLLMClassifier(error=RuntimeError("判定炸了"))
    manager, client, _parsers, sink = make_manager(llm_classifier=classifier)

    await manager.submit_text(QUESTION_ONE)

    session = manager.current_session
    assert session is not None
    assert session.question_type == "project"
    assert session.rag_required is True
    assert classifier.calls == [QUESTION_ONE]
    assert len(client.calls) == 1
    assert len(sink.dispatched) == 2


async def test_llm_classification_timeout_falls_back_to_rules() -> None:
    """模型判定超时同样回退同步规则，请求不受影响。"""
    classifier = FakeLLMClassifier(error=asyncio.TimeoutError())
    manager, client, _parsers, sink = make_manager(llm_classifier=classifier)

    await manager.submit_text(QUESTION_ONE)

    session = manager.current_session
    assert session is not None
    assert session.question_type == "project"
    assert session.rag_required is True
    assert len(client.calls) == 1
    assert len(sink.dispatched) == 2


async def test_reanalyze_reuses_classification_without_second_call() -> None:
    """同一题重新分析复用已缓存结果，不重复请求模型。"""
    classifier = FakeLLMClassifier(
        Classification(
            question_type="project",
            rag_required=True,
            matched_rules=("llm",),
            reason="本人项目",
            source="llm",
        )
    )
    manager, client, _parsers, _sink = make_manager(llm_classifier=classifier)

    await manager.submit_text(QUESTION_ONE)
    await manager.reanalyze()

    assert classifier.calls == [QUESTION_ONE]
    assert len(client.calls) == 2


async def test_new_question_clears_classification_cache() -> None:
    """创建新问题后缓存清空，同一题面重新判定一次。"""
    classifier = FakeLLMClassifier()
    manager, client, _parsers, _sink = make_manager(llm_classifier=classifier)

    await manager.submit_text(QUESTION_ONE)
    manager.new_question()
    await manager.submit_text(QUESTION_ONE)

    assert classifier.calls == [QUESTION_ONE, QUESTION_ONE]
    assert len(client.calls) == 2
