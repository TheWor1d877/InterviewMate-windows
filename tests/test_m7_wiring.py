"""M7 接线的测试。

覆盖《8-集成与验收》「三种输入的汇合」「全局快捷键」「请求取消」
「防止重复调用」「超时管理」「性能统计」几节在接线之后的行为：

- 三种输入（手工、语音、截图）都走输入路由器，语音停止后自动提交，
  不用再按回车；语音与截图按事件会合时只发一次请求；
- 同一段录音里同一句话不提交两次，会话内出现过截图之后纯语音去重仍生效；
- 语音加截图合成后一次请求就含文字与图片，纯语音提交不带历史截图；
- 五个 socket action 行为不变，未知 action 不崩；
- 请求任务由受管任务注册表托管：取消会取消任务并通知客户端，
  同一请求标识不重复启动；
- DeepSeek 客户端：鉴权错误不重试、超时错误会重试、取消原样透传；
- 状态 payload 的 DEEPSEEK 字段跟着请求生命周期走，ttft 有样本后有值。

模型客户端、ASR 引擎与截图后端都是假的：不发真实网络请求、
不加载 FunASR 模型、不开真实音频设备。重试退避注入假函数，测试不等真时间。
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, AsyncIterator

import httpx
import pytest

import app.main
from app.config import AppConfig
from app.core.exceptions import (
    DeepSeekAuthError,
    DeepSeekStreamError,
)
from app.core.question_manager import QuestionManager
from app.core.task_manager import TaskManager
from app.inputs.global_command import GlobalCommandDispatcher
from app.inputs.input_router import InputRouter
from app.llm.deepseek_client import DeepSeekClient
from app.llm.stream_parser import StreamParser
from app.runtime.socket_protocol import Request
from app.utils.timing import reset_default_recorder
from tests.test_ui_skeleton import make_config

# 宽终端宽度，够两栏排开
WIDE_WIDTH = 120

# 截图的假图片数据，见《4-截图输入与多模态》多模态请求一节
FAKE_IMAGE_DATA_URL = "data:image/png;base64," + base64.b64encode(
    b"FAKE-PNG-BYTES"
).decode("ascii")

# 语音识别出来的题面
VOICE_TEXT = "请实现一个 LRU Cache"


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
class _FakeApp:
    """假界面：只提供装配层与语音链路要用的那几个方法。"""

    def __init__(self) -> None:
        self.questions: list[str] = []
        self.voice_texts: list[str] = []
        self.hints: list[str] = []
        self.exited = 0
        self.screenshot_manager: Any = None

    def show_question(self, text: str) -> None:
        """记下右栏题面。"""
        self.questions.append(text)

    def show_voice_text(self, text: str) -> None:
        """记下临时识别文字。"""
        self.voice_texts.append(text)

    def show_screenshot_question(self) -> None:
        """记下截图题面的显示。"""
        self.questions.append("截图题面")

    async def publish_hint(self, hint: str) -> None:
        """记下状态栏提示位。"""
        self.hints.append(hint)

    def attach_screenshot_manager(self, screenshot_manager: Any) -> None:
        """记下挂上来的截图管理器。"""
        self.screenshot_manager = screenshot_manager

    def exit(self) -> None:
        """记下退出请求。"""
        self.exited += 1


class _FakeManager:
    """假问题管理器：只记语音文字与提交。"""

    def __init__(self) -> None:
        self.voice_texts: list[str] = []
        self.submissions: list[tuple[str, str | None]] = []
        self.submitted = asyncio.Event()

    def set_voice_text(self, text: str) -> None:
        """记下会话保留的语音文字。"""
        self.voice_texts.append(text)

    async def submit_text(self, text: str, image_data_url: str | None = None) -> None:
        """记下一次提交。"""
        self.submissions.append((text, image_data_url))
        self.submitted.set()


class _FakeClient:
    """假模型客户端：记下收到的请求，按脚本产出增量文本。"""

    name = "mock"

    def __init__(self, chunks: tuple[str, ...] = ("回答",)) -> None:
        self.chunks = chunks
        self.calls: list[tuple[list[dict], str]] = []
        self.cancelled: list[str] = []

    async def chat_stream(
        self, messages: list[dict[str, str]], request_id: str
    ) -> AsyncIterator[str]:
        """产出文本请求的增量内容。"""
        self.calls.append((messages, request_id))
        for chunk in self.chunks:
            yield chunk

    async def multimodal_stream(
        self, messages: list[dict], request_id: str
    ) -> AsyncIterator[str]:
        """产出带图请求的增量内容。"""
        self.calls.append((messages, request_id))
        for chunk in self.chunks:
            yield chunk

    async def cancel(self, request_id: str) -> None:
        """记下客户端侧的取消。"""
        self.cancelled.append(request_id)


class _BlockingClient:
    """假模型客户端：请求挂住不结束，等着被取消。"""

    name = "mock"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled: list[str] = []

    async def chat_stream(
        self, messages: list[dict[str, str]], request_id: str
    ) -> AsyncIterator[str]:
        """挂住，等取消。"""
        self.started.set()
        await asyncio.Event().wait()
        yield ""  # pragma: no cover - 不会走到这里

    async def multimodal_stream(
        self, messages: list[dict], request_id: str
    ) -> AsyncIterator[str]:
        """挂住，等取消。"""
        await self.chat_stream(messages, request_id)
        yield ""  # pragma: no cover - 不会走到这里

    async def cancel(self, request_id: str) -> None:
        """记下客户端侧的取消。"""
        self.cancelled.append(request_id)


class _GatedClient:
    """假模型客户端：先等闸门，再产出增量文本。

    用来卡在「请求已发出但还没产出内容」的那一刻，
    看 DEEPSEEK 字段是不是 connecting。第一段产出后再等一道闸门，
    流不立刻结束，streaming 那一下才看得见。
    """

    name = "mock"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finish = asyncio.Event()

    async def chat_stream(
        self, messages: list[dict[str, str]], request_id: str
    ) -> AsyncIterator[str]:
        """等闸门放开再产出内容。"""
        self.started.set()
        await self.release.wait()
        yield "第一段"
        await self.finish.wait()
        yield "第二段"

    async def multimodal_stream(
        self, messages: list[dict], request_id: str
    ) -> AsyncIterator[str]:
        """等闸门放开再产出内容。"""
        await self.chat_stream(messages, request_id)
        yield ""  # pragma: no cover - 不会走到这里

    async def cancel(self, request_id: str) -> None:
        """客户端侧的取消，这里什么都不用做。"""


class _FakeSink:
    """假事件分发器：只记事件，不碰界面。"""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.errors: list[tuple[str, str]] = []
        self.cleared = 0

    def dispatch(self, event: Any) -> None:
        """记下事件。"""
        self.events.append(event)

    def clear(self) -> None:
        """记下清空界面。"""
        self.cleared += 1

    def report_error(self, code: str, detail: str) -> None:
        """记下错误。"""
        self.errors.append((code, detail))


class _FakeEngine:
    """假 ASR 引擎：喂一块返回一句，冲刷不返回内容。"""

    state = "ready"

    def __init__(self, text: str = VOICE_TEXT) -> None:
        self.text = text
        self.fed = asyncio.Event()
        self.flush_calls = 0

    async def ensure_loaded(self) -> None:
        """加载模型，这里什么都不用做。"""

    async def feed(self, block: Any) -> str:
        """喂一块音频，返回增量文本。"""
        self.fed.set()
        return self.text

    async def flush(self) -> str:
        """冲刷，尾巴已经在 feed 里给过了。"""
        self.flush_calls += 1
        return ""


class _FakeCapture:
    """假采集器：块从一个队列里出。

    两路来源时（麦克风与系统回环）都给，各给一句同样的话，
    用来验同一段录音里同一句话不提交两次。
    """

    def __init__(self, separated: bool = False) -> None:
        self.separated = separated
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.monitor_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.started = False

    @property
    def is_running(self) -> bool:
        """采集是不是开着。"""
        return self.started

    async def start(self) -> None:
        """开始采集。"""
        self.started = True

    async def stop(self) -> None:
        """停止采集。"""
        self.started = False

    async def get_block(self) -> Any:
        """等下一块。"""
        return await self.queue.get()

    async def get_microphone_block(self) -> Any:
        """等麦克风这一路的下一块。"""
        return await self.queue.get()

    async def get_system_monitor_block(self) -> Any:
        """等系统回环这一路的下一块。"""
        return await self.monitor_queue.get()

    def push(self, source: str = "microphone") -> None:
        """塞一块音频进去。"""
        block = _FakeBlock(source)
        if source == "microphone":
            self.queue.put_nowait(block)
        else:
            self.monitor_queue.put_nowait(block)


class _FakeBlock:
    """假音频块，只带来源标签。"""

    def __init__(self, source: str) -> None:
        self.source = source


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------
def _build_messages(
    question: str,
    question_type: str,
    context: str | None = None,
) -> list[dict[str, str]]:
    """组装文本请求的消息。"""
    return [{"role": "user", "content": question}]


def _build_multimodal_messages(
    question: str,
    image_data_url: str,
    question_type: str,
    context: str | None = None,
) -> list[dict]:
    """组装带图请求的消息：文字块加图片块。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]


def _make_manager(
    client: Any,
    *,
    config: AppConfig | None = None,
    on_model_state: Any = None,
    retrieve: Any = None,
) -> QuestionManager:
    """造一个用假客户端的问题管理器。"""
    active_config = config if config is not None else make_config()
    return QuestionManager(
        active_config,
        client,
        lambda request_id, session_id: StreamParser(request_id, session_id),
        _FakeSink(),
        _build_messages,
        build_multimodal_messages=_build_multimodal_messages,
        retrieve=retrieve,
        on_model_state=on_model_state,
    )


async def _wait_until(
    predicate: Any,
    timeout: float = 5.0,
) -> bool:
    """等条件成立，最多等 timeout 秒。

    :param predicate: 判断条件的函数。
    :param timeout: 最长等待秒数。
    :returns: 条件成立时返回真。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


async def _no_sleep(delay: float) -> None:
    """假的等待函数：重试退避不等真时间。"""
    return None


async def _collect(stream: AsyncIterator[str]) -> list[str]:
    """把流里的内容收成列表。"""
    return [chunk async for chunk in stream]


def _sse(*chunks: str) -> bytes:
    """造一段 SSE 响应体。"""
    body = "".join(
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": chunk}}]},
            ensure_ascii=False,
        )
        + "\n\n"
        for chunk in chunks
    )
    return (body + "data: [DONE]\n\n").encode("utf-8")


def _sse_chunk(chunk: str) -> bytes:
    """造一段没有结束标记的 SSE 增量。"""
    return (
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": chunk}}]},
            ensure_ascii=False,
        )
        + "\n\n"
    ).encode("utf-8")


@pytest.fixture(autouse=True)
def _clean_recorder():
    """每个用例前后都清空进程级默认记录器，样本不互相干扰。"""
    reset_default_recorder()
    yield
    reset_default_recorder()


# --------------------------------------------------------------------------
# 三种输入的汇合
# --------------------------------------------------------------------------
async def test_voice_final_text_goes_through_input_router() -> None:
    """语音最终文本走输入路由器，自动提交一次，不用再按回车。

    见《5-语音输入与流式ASR》F8 交互一节与
    《8-集成与验收》三种输入的汇合一节。
    """
    config = make_config()
    fake_app = _FakeApp()
    manager = _FakeManager()
    router = app.main.build_input_router(manager)  # type: ignore[arg-type]
    engine = _FakeEngine()
    capture = _FakeCapture()

    voice = app.main.assemble_voice(
        config,
        fake_app,
        manager,  # type: ignore[arg-type]
        state=app.main.MainState(),
        engine=engine,  # type: ignore[arg-type]
        factory=lambda source: capture,  # type: ignore[arg-type]
        router=router,
    )

    await voice.audio_manager.start()
    capture.push("microphone")
    await asyncio.wait_for(engine.fed.wait(), timeout=5)
    await voice.audio_manager.stop()
    await asyncio.wait_for(manager.submitted.wait(), timeout=5)

    # 提交了一次，题面是识别出来的最终文本
    assert manager.submissions == [(VOICE_TEXT, None)]
    # 走的确实是输入路由器，来源是语音
    assert router.last_source == "voice"
    # 语音文字也记进了会话，截图随后到达时能合成一条请求
    assert manager.voice_texts == [VOICE_TEXT]


async def test_same_sentence_in_one_recording_is_submitted_once() -> None:
    """同一段录音里同一句话不提交两次。

    去重由采集管理器负责，接线之后这条还在：
    见《8-集成与验收》防止重复调用一节。
    """
    config = make_config()
    fake_app = _FakeApp()
    manager = _FakeManager()
    router = app.main.build_input_router(manager)  # type: ignore[arg-type]
    engine = _FakeEngine()
    capture = _FakeCapture(separated=True)

    voice = app.main.assemble_voice(
        config,
        fake_app,
        manager,  # type: ignore[arg-type]
        state=app.main.MainState(),
        engine=engine,  # type: ignore[arg-type]
        factory=lambda source: capture,  # type: ignore[arg-type]
        router=router,
    )

    await voice.audio_manager.start()
    # 两路来源识别出同一句话
    capture.push("microphone")
    capture.push("system_monitor")
    await asyncio.wait_for(engine.fed.wait(), timeout=5)
    await asyncio.sleep(0.05)
    await voice.audio_manager.stop()
    await asyncio.wait_for(manager.submitted.wait(), timeout=5)
    await asyncio.sleep(0.05)

    assert manager.submissions == [(VOICE_TEXT, None)]


async def test_screenshot_request_carries_voice_text_and_image() -> None:
    """录音中先截图，语音结束后合成一次请求，请求里同时含语音文字和图片。

    截图在录音进行中到达时暂存，语音最终文本到达时两段材料合成一条消息
    一次发出，总请求数是一次，
    见《4-截图输入与多模态》与语音合并一节与
    《8-集成与验收》三种输入的汇合一节。
    """
    client = _FakeClient()
    manager = _make_manager(client)
    router = app.main.build_input_router(manager)

    # 录音进行中先截图：图片暂存，还不发请求
    router.voice_session_started()
    await router.route_screenshot(FAKE_IMAGE_DATA_URL)
    assert client.calls == []

    # 语音结束后文字到达，与暂存的图片合成一次提交
    await router.route_voice(VOICE_TEXT)
    assert await _wait_until(lambda: bool(client.calls))
    await asyncio.sleep(0.05)

    # 总请求数是一次，不是语音、截图各一次
    assert len(client.calls) == 1
    messages, _request_id = client.calls[0]
    content = messages[0]["content"]
    assert isinstance(content, list)
    text_block = content[0]
    image_block = content[1]
    assert text_block["text"] == VOICE_TEXT
    assert image_block["image_url"]["url"] == FAKE_IMAGE_DATA_URL


async def test_f9_during_voice_session_sends_exactly_one_request() -> None:
    """F9 在语音会话进行中到达：截图先暂存，语音停止后只发一次请求。

    截图采集先于语音最终文本发生，图片在路由器里等语音结束，
    语音文字到达时合成一次提交，
    见《4-截图输入与多模态》与语音合并一节与
    《8-集成与验收》三种输入的汇合一节。
    """
    config = make_config()
    fake_app = _FakeApp()
    client = _FakeClient()
    manager = _make_manager(client, config=config)
    router = app.main.build_input_router(manager)
    engine = _FakeEngine()
    capture = _FakeCapture()

    voice = app.main.assemble_voice(
        config,
        fake_app,
        manager,
        state=app.main.MainState(),
        engine=engine,  # type: ignore[arg-type]
        factory=lambda source: capture,  # type: ignore[arg-type]
        router=router,
    )

    # F8 开始采集
    await voice.audio_manager.start()
    router.voice_session_started()
    # F9 在录音进行中到达：先标记采集，再把图交给路由器，图暂存不发请求
    router.screenshot_started()
    await router.route_screenshot(FAKE_IMAGE_DATA_URL)
    assert client.calls == []

    # F8 停止采集，最终文本与暂存的图合成一次提交
    capture.push("microphone")
    await asyncio.wait_for(engine.fed.wait(), timeout=5)
    await voice.audio_manager.stop()
    for task in voice.submit_tasks:
        await task
    await router.screenshot_finished()

    assert await _wait_until(lambda: bool(client.calls))
    await asyncio.sleep(0.05)
    assert len(client.calls) == 1
    messages, _request_id = client.calls[0]
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["text"] == VOICE_TEXT
    assert content[1]["image_url"]["url"] == FAKE_IMAGE_DATA_URL


async def test_f9_after_voice_stop_waits_for_in_flight_screenshot() -> None:
    """语音文本先到、截图仍在采集：文字暂存，截图到达后只发一次请求。

    停止采集后最终文本先到，截图此时还在采集，
    文字在路由器里等图，图片到达时合成一次提交，
    见《4-截图输入与多模态》与语音合并一节与
    《8-集成与验收》三种输入的汇合一节。
    """
    config = make_config()
    fake_app = _FakeApp()
    client = _FakeClient()
    manager = _make_manager(client, config=config)
    router = app.main.build_input_router(manager)
    engine = _FakeEngine()
    capture = _FakeCapture()

    voice = app.main.assemble_voice(
        config,
        fake_app,
        manager,
        state=app.main.MainState(),
        engine=engine,  # type: ignore[arg-type]
        factory=lambda source: capture,  # type: ignore[arg-type]
        router=router,
    )

    # F8 开始采集，F9 的截图此刻已经开始但在路上
    await voice.audio_manager.start()
    router.voice_session_started()
    router.screenshot_started()

    # F8 停止采集：最终文本先到，因为截图还在路上而暂存，没有发请求
    capture.push("microphone")
    await asyncio.wait_for(engine.fed.wait(), timeout=5)
    await voice.audio_manager.stop()
    for task in voice.submit_tasks:
        await task
    assert client.calls == []

    # 截图到达：与等着的语音文字合成一次提交
    await router.route_screenshot(FAKE_IMAGE_DATA_URL)
    await router.screenshot_finished()

    assert await _wait_until(lambda: bool(client.calls))
    await asyncio.sleep(0.05)
    assert len(client.calls) == 1
    messages, _request_id = client.calls[0]
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["text"] == VOICE_TEXT
    assert content[1]["image_url"]["url"] == FAKE_IMAGE_DATA_URL


async def test_voice_dedup_still_works_after_a_screenshot() -> None:
    """会话内出现过截图之后，纯语音的同一句话仍被去重。

    纯语音提交不再挂历史截图，去重前提恢复：截图 1 次加语音 1 次，
    第二次同样的语音被拦，见《8-集成与验收》防止重复调用一节与
    《4-截图输入与多模态》与语音合并一节。
    """
    client = _FakeClient()
    manager = _make_manager(client)
    router = app.main.build_input_router(manager)

    # 先截图，1 次请求
    await router.route_screenshot(FAKE_IMAGE_DATA_URL)
    assert await _wait_until(lambda: bool(client.calls))
    await asyncio.sleep(0.05)
    assert len(client.calls) == 1

    # 同一句语音提交两次，只发一次
    await router.route_voice(VOICE_TEXT)
    await asyncio.sleep(0.05)
    await router.route_voice(VOICE_TEXT)
    await asyncio.sleep(0.05)

    assert len(client.calls) == 2


async def test_manual_text_goes_through_input_router() -> None:
    """输入框回车走输入路由器，提交一次。

    见《8-集成与验收》三种输入的汇合一节。
    """
    from app.ui.app import InterviewMateApp
    from app.ui.panels.input_panel import QuestionInputPanel

    submissions: list[tuple[str, str | None]] = []

    async def _submit(question: str, image_data_url: str | None) -> None:
        """记下提交。"""
        submissions.append((question, image_data_url))

    router = InputRouter(_submit)
    ui = InterviewMateApp(config=make_config())
    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        ui.attach_input_router(router)
        panel = ui.query_one(QuestionInputPanel)
        panel.focus()
        panel.text = "手输的题面"
        await pilot.pause()
        await pilot.press("enter")
        assert await _wait_until(lambda: bool(submissions))

    assert submissions == [("手输的题面", None)]
    # 走的确实是输入路由器，来源是手工输入
    assert router.last_source == "text"


# --------------------------------------------------------------------------
# 全局快捷键：socket 的五个动作
# --------------------------------------------------------------------------
async def test_socket_actions_keep_their_behavior() -> None:
    """五个 socket action 的行为不变。

    见《2-骨架与配置》进程通信一节与《8-集成与验收》全局快捷键一节。
    """
    config = make_config()
    fake_app = _FakeApp()
    state = app.main.MainState()
    state.attach_app(fake_app)
    state.attach_config(config)
    manager = _FakeManager()
    screenshot_manager = app.main.assemble_screenshot_manager(
        config,
        fake_app,
        manager,  # type: ignore[arg-type]
    )
    state.set_screenshot_manager(screenshot_manager)
    # 预热时要探模型与 Dify，给假客户端与假检索链路
    state.set_model_client(_FakeClient())
    retrieval = app.main.assemble_retrieval(config, fake_app, state=state)
    state.set_retrieval_chain(retrieval)

    handlers = app.main.build_handlers(state)

    # voice-toggle：没装语音链路时忽略这次动作，不崩
    voice_payload = await handlers["voice-toggle"](Request(action="voice-toggle"))
    assert voice_payload["result"] == "ignored"
    assert voice_payload["audio"] == "off"

    # screenshot：真的采集一张并提交
    screenshot_payload = await handlers["screenshot"](Request(action="screenshot"))
    assert screenshot_payload["captured"] is True
    assert screenshot_payload["provider"] == "mock"
    assert len(manager.submissions) == 1

    # status：返回状态字段
    status_payload = await handlers["status"](Request(action="status"))
    assert status_payload["deepseek"] == "idle"
    assert status_payload["asr"] == "idle"
    assert status_payload["dify"] == "offline"
    assert "metrics" in status_payload

    # warmup：五个 provider 都是 mock 时给 mock 标记
    warmup_payload = await handlers["warmup"](Request(action="warmup"))
    assert warmup_payload["deepseek"] == "mock"
    assert warmup_payload["dify"] == "mock"

    # stop：先停采集再退出界面
    stop_payload = await handlers["stop"](Request(action="stop", request_id="req-1"))
    assert stop_payload == {"stopping": True}
    assert fake_app.exited == 1


async def test_unknown_socket_action_does_not_crash() -> None:
    """未知 action 不崩：分发入口兜一层，给出明确的结果标识。

    见《8-集成与验收》全局快捷键一节。
    """
    state = app.main.MainState()
    handlers = app.main.build_handlers(state)

    payload = await handlers["status"](Request(action="没有这个动作"))
    assert payload["result"] == "unknown-action"
    assert payload["action"] == "没有这个动作"

    # 直接调分发入口也是一样的结果
    dispatcher = GlobalCommandDispatcher()
    direct = await dispatcher.dispatch(Request(action="没有这个动作"))
    assert direct["result"] == "unknown-action"


# --------------------------------------------------------------------------
# 请求取消与防止重复调用
# --------------------------------------------------------------------------
async def test_cancel_request_cancels_task_and_notifies_client() -> None:
    """取消请求会真的取消任务，并通知客户端关流。

    见《8-集成与验收》请求取消一节。
    """
    client = _BlockingClient()
    manager = _make_manager(client)
    sink = manager._sink  # noqa: SLF001 - 假事件分发器只是用来确认没报错

    submit_task = asyncio.create_task(manager.submit_text("讲讲缓存怎么写"))
    await asyncio.wait_for(client.started.wait(), timeout=5)
    request_id = manager.active_request_id
    assert request_id is not None

    await manager.cancel()

    # 客户端收到了取消通知
    assert client.cancelled == [request_id]
    # 任务真的被取消了
    with pytest.raises(asyncio.CancelledError):
        await submit_task
    assert submit_task.cancelled()
    assert sink.errors == []


async def test_same_request_id_is_not_started_twice() -> None:
    """同一个请求标识重复登记时不重复启动。

    见《8-集成与验收》防止重复调用一节。
    """
    calls: list[int] = []

    async def _factory() -> None:
        """这次任务要做的事。"""
        calls.append(1)
        await asyncio.sleep(0.05)

    tasks = TaskManager()
    first = tasks.register("req-1", _factory)
    second = tasks.register("req-1", _factory)

    assert first.created is True
    assert second.created is False
    assert first.task is second.task
    await first.task
    assert len(calls) == 1


async def test_request_task_is_managed_by_task_manager() -> None:
    """请求任务登记在受管任务注册表里，取消走它。

    见《8-集成与验收》请求取消与超时管理两节。
    """
    client = _BlockingClient()
    manager = _make_manager(client)

    submit_task = asyncio.create_task(manager.submit_text("讲讲缓存怎么写"))
    await asyncio.wait_for(client.started.wait(), timeout=5)
    request_id = manager.active_request_id
    assert request_id is not None

    # 任务在注册表里，时限是读超时加余量
    info = next(
        (item for item in manager.managed_tasks if item.request_id == request_id),
        None,
    )
    assert info is not None
    assert info.timeout is not None
    assert info.timeout > 0

    await manager.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submit_task


async def test_retrieval_task_uses_rag_timeout() -> None:
    """检索任务也受管，时限取配置里的 rag.timeout。

    见《8-集成与验收》超时管理一节。
    """
    base = make_config()
    config = base.model_copy(
        update={"rag": base.rag.model_copy(update={"force_project_mode": True})}
    )
    release = asyncio.Event()

    async def _retrieve(query: str) -> tuple[str | None, list[dict]]:
        """挂住的检索，等闸门放开。"""
        await release.wait()
        return None, []

    manager = _make_manager(_FakeClient(), config=config, retrieve=_retrieve)

    submit_task = asyncio.create_task(manager.submit_text("讲讲项目怎么做的"))
    assert await _wait_until(
        lambda: any(
            info.request_id.startswith("retrieve:")
            for info in manager.managed_tasks
        )
    )
    info = next(
        item
        for item in manager.managed_tasks
        if item.request_id.startswith("retrieve:")
    )
    assert info.timeout == float(config.rag.timeout)

    release.set()
    await submit_task


# --------------------------------------------------------------------------
# DeepSeek 客户端的重试
# --------------------------------------------------------------------------
async def test_auth_error_is_not_retried() -> None:
    """鉴权错误不重试：Key 不对时重试多少次都没用。

    见《3-文本输入与流式回答》DeepSeek 客户端一节。
    """
    calls: list[int] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        """每次都回 401。"""
        calls.append(1)
        return httpx.Response(401, text="没有权限")

    client = DeepSeekClient(
        make_config(),
        "sk-unit-test-key",
        transport=httpx.MockTransport(_handler),
        retry_sleep=_no_sleep,
    )
    try:
        with pytest.raises(DeepSeekAuthError):
            await _collect(
                client.chat_stream([{"role": "user", "content": "问题"}], "req-auth")
            )
    finally:
        await client.aclose()

    assert len(calls) == 1


async def test_timeout_error_is_retried() -> None:
    """超时错误会重试：还没收到输出时按 max_retries 重试。

    见《3-文本输入与流式回答》DeepSeek 客户端一节与
    《8-集成与验收》超时管理一节。
    """
    calls: list[int] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        """第一次读超时，第二次正常返回。"""
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("读超时")
        return httpx.Response(200, content=_sse("成功"))

    client = DeepSeekClient(
        make_config(),
        "sk-unit-test-key",
        transport=httpx.MockTransport(_handler),
        retry_sleep=_no_sleep,
    )
    try:
        chunks = await _collect(
            client.chat_stream([{"role": "user", "content": "问题"}], "req-timeout")
        )
    finally:
        await client.aclose()

    assert chunks == ["成功"]
    assert len(calls) == 2


async def test_stream_error_after_output_is_not_retried() -> None:
    """已经输出部分内容后不重发整个请求。

    见《3-文本输入与流式回答》DeepSeek 客户端一节。
    """
    calls: list[int] = []

    class _InterruptStream(httpx.AsyncByteStream):
        """先给一段增量再抛读错误，模拟流中断。"""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """给一段内容后中断。"""
            yield _sse_chunk("部分")
            raise httpx.ReadError("流中断")

    def _handler(request: httpx.Request) -> httpx.Response:
        """每次都中断。"""
        calls.append(1)
        return httpx.Response(200, stream=_InterruptStream())

    client = DeepSeekClient(
        make_config(),
        "sk-unit-test-key",
        transport=httpx.MockTransport(_handler),
        retry_sleep=_no_sleep,
    )
    try:
        with pytest.raises(DeepSeekStreamError):
            await _collect(
                client.chat_stream([{"role": "user", "content": "问题"}], "req-broken")
            )
    finally:
        await client.aclose()

    assert len(calls) == 1


async def test_cancelled_error_is_passed_through() -> None:
    """取消原样抛出，不重试。

    见《8-集成与验收》请求取消一节：用户取消的请求不允许自动恢复。
    """
    calls: list[int] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        """直接抛取消。"""
        calls.append(1)
        raise asyncio.CancelledError()

    client = DeepSeekClient(
        make_config(),
        "sk-unit-test-key",
        transport=httpx.MockTransport(_handler),
        retry_sleep=_no_sleep,
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await _collect(
                client.chat_stream([{"role": "user", "content": "问题"}], "req-cancel")
            )
    finally:
        await client.aclose()

    assert len(calls) == 1


# --------------------------------------------------------------------------
# 状态字段与性能统计
# --------------------------------------------------------------------------
async def test_status_payload_reports_model_state_and_ttft() -> None:
    """请求期间 DEEPSEEK 字段是 connecting 与 streaming，结束后回到 idle。

    ttft 有样本后有值，没有样本时是 None。
    见《2-骨架与配置》状态字段取值表与《8-集成与验收》性能统计一节。
    """
    client = _GatedClient()
    state = app.main.MainState()
    manager = _make_manager(client, on_model_state=state.set_deepseek_state)

    submit_task = asyncio.create_task(manager.submit_text("讲讲缓存怎么写"))
    await asyncio.wait_for(client.started.wait(), timeout=5)

    # 请求已发出但还没产出内容：connecting
    assert state.status_payload()["deepseek"] == "connecting"
    # 没有样本时 ttft 是 None
    assert state.status_payload()["ttft"] is None

    client.release.set()
    # 收到第一个片段：streaming
    assert await _wait_until(
        lambda: state.status_payload()["deepseek"] == "streaming"
    )
    # 流还没结束，状态停在 streaming
    assert state.status_payload()["deepseek"] == "streaming"
    client.finish.set()
    await submit_task

    # 正常结束回到 idle
    assert state.status_payload()["deepseek"] == "idle"
    # 有样本后 ttft 有值
    assert state.status_payload()["ttft"] is not None


async def test_model_state_callback_is_optional() -> None:
    """没注入状态回调时不报错，行为与从前一致。

    见《2-骨架与配置》状态字段取值表。
    """
    client = _FakeClient()
    manager = _make_manager(client)
    await manager.submit_text("讲讲缓存怎么写")
    assert len(client.calls) == 1
