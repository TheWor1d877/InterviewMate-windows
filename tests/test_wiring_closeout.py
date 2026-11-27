"""M4/M5/M6 接线收尾的测试。

覆盖四处最后一根线：

- 界面按 F8 触发装配层注入的语音采集回调，回调没接上时安静地什么都不做，
  见《5-语音输入与流式ASR》F8 交互一节；
- 检索来源的「存」与「显示」拆开：检索完成时来源静默存下，浮层不自动弹，
  按 F3 才弹，见《6-Dify检索与项目知识》检索来源展示一节；
- 代码题判定在发出模型请求之前通知分发器切形态，
  见《7-代码问题模式》纯代码块兜底一节；
- 启动时按配置预加载 ASR，加载失败只记日志、不影响启动，
  见《2-骨架与配置》命令行入口一节第 9 步。

模型客户端、采集器与 ASR 引擎都用假实现，不连真实服务也不开真流。
界面用 Textual 的 ``run_test()`` 无头驱动，配置显式构造，
不依赖开发者本地的 ``config/config.yaml``。
"""

from __future__ import annotations

import asyncio
from typing import Any

import app.main
from app.audio.funasr_engine import FunASREngine
from app.core.question_manager import QuestionManager
from app.ui.app import InterviewMateApp
from app.ui.panels.rag_sources import RagSourcesPanel
from app.ui.panels.voice_text import VoiceTextPanel
from app.ui.status_bar import StatusBar
from tests.test_ui_skeleton import make_config

# 宽终端的宽度
WIDE_WIDTH = 120

# 题面样例：一个命中算法特征判成代码题，一个命中项目关键词判成普通题
QUESTION_CODING = "写一个二分查找的函数，说明时间复杂度"
QUESTION_PROJECT = "讲讲我的项目是怎么做的"

# 来源条目样例，字段与界面约定一致
SOURCES: list[dict] = [
    {"document_name": "架构说明.md", "score": 0.87, "content": "项目分为输入层、核心层与输出层。"},
]


# ----------------------------------------------------------------------
# 假实现
# ----------------------------------------------------------------------
class FakePanel:
    """假面板，方法都是空实现，只记必要的调用。"""

    def __init__(self) -> None:
        self.code_modes: list[bool] = []
        self.items: list[str] = []
        self.fallbacks: list[str] = []
        self.hints: list[str] = []

    def show_framework(self, sections: list[dict]) -> None:
        return None

    def show_steps(self, steps: list[dict]) -> None:
        return None

    def append_hint(self, text: str) -> None:
        self.hints.append(text)

    def mark_section_done(self, section_id: str) -> None:
        return None

    def append_item(self, section_id: str, keyword: str, sentence: str) -> None:
        self.items.append(sentence)

    def append_fallback(self, text: str) -> None:
        self.fallbacks.append(text)

    def append_code(self, text: str) -> None:
        return None

    def complete_code(self, language: str, standard: str, submission_mode: str) -> None:
        return None

    def enter_code_mode(self) -> None:
        """记录右栏切到了代码形态。"""
        self.code_modes.append(True)

    def clear(self) -> None:
        return None


class FakeApp:
    """假界面：记录来源的存与显示、语音回调与状态发布。"""

    def __init__(self) -> None:
        self.navigation_panel = FakePanel()
        self.answer_panel = FakePanel()
        self.errors: list[tuple[str, str]] = []
        self.stored_sources: list[list[dict]] = []
        self.shown_sources: list[list[dict]] = []
        self.questions: list[str] = []
        self.hints: list[str] = []
        self.statuses: list[dict] = []
        self.voice_toggle: Any = None
        self.manager: Any = None

    def show_error(self, code: str, detail: str) -> None:
        self.errors.append((code, detail))

    def show_question(self, text: str) -> None:
        """记录显示到右栏顶部的题面。"""
        self.questions.append(text)

    async def publish_hint(self, hint: str) -> None:
        """记录提示位上的文字。"""
        self.hints.append(hint)

    def attach_manager(self, manager: Any) -> None:
        self.manager = manager

    def attach_voice_toggle(self, callback: Any) -> None:
        """记录装配层挂上来的语音采集回调。"""
        self.voice_toggle = callback

    def store_rag_sources(self, entries: list[dict]) -> None:
        """只存来源，不弹浮层。"""
        self.stored_sources.append(list(entries))

    def show_rag_sources(self, entries: list[dict]) -> None:
        """存来源并弹浮层。"""
        self.shown_sources.append(list(entries))

    async def publish_status(self, **fields: Any) -> None:
        self.statuses.append(dict(fields))

    def exit(self) -> None:
        return None


class FakeClient:
    """假模型客户端，记录请求发出的顺序。"""

    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order if order is not None else []
        self.calls: list[list[dict[str, str]]] = []

    async def chat_stream(self, messages: list[dict[str, str]], request_id: str):
        self.order.append("request")
        self.calls.append(messages)
        yield "回答"

    async def multimodal_stream(self, messages: list[dict], request_id: str):
        self.order.append("request")
        yield "回答"

    async def cancel(self, request_id: str) -> None:
        return None


class FakeParser:
    """假解析器，产出带请求标识的假事件。"""

    def __init__(self, request_id: str | None = None) -> None:
        self.request_id = request_id

    def reset(self) -> None:
        return None

    def feed(self, text: str) -> list[Any]:
        return []

    def flush(self) -> list[Any]:
        return []


class FakeSink:
    """假事件分发器。"""

    def __init__(self) -> None:
        self.dispatched: list[Any] = []

    def dispatch(self, event: Any) -> None:
        self.dispatched.append(event)

    def clear(self) -> None:
        return None

    def report_error(self, code: str, detail: str) -> None:
        return None


class FakeManager:
    """假问题管理器，记录语音文字与提交。"""

    def __init__(self) -> None:
        self.voice_texts: list[str] = []
        self.submissions: list[str] = []
        self.submitted = asyncio.Event()

    def set_voice_text(self, text: str) -> None:
        self.voice_texts.append(text)

    async def submit_text(self, text: str, image_data_url: str | None = None) -> None:
        self.submissions.append(text)
        self.submitted.set()


class StreamingCapture:
    """假采集器：先给几块音频，之后一直等着等采集停掉。"""

    def __init__(self, blocks: int = 2) -> None:
        self.blocks = blocks
        self.started = 0
        self.stopped = 0
        self.stopped_event = asyncio.Event()
        self._count = 0
        self._running = False

    @property
    def is_running(self) -> bool:
        """采集是不是开着。"""
        return self._running

    async def start(self) -> None:
        self.started += 1
        self._running = True

    async def stop(self) -> None:
        self.stopped += 1
        self._running = False
        self.stopped_event.set()

    async def get_block(self) -> FakeBlock:
        """给一块音频，之后一直等着。"""
        self._count += 1
        if self._count > self.blocks:
            await asyncio.sleep(3600)
        return FakeBlock()


class StreamingEngine:
    """假 ASR 引擎：每次喂一块给一段增量，冲刷给出剩下的文本。"""

    def __init__(self, partials: list[str], final: str) -> None:
        self.partials = partials
        self.final = final
        # 喂过一次就置位
        self.fed = asyncio.Event()
        # 全部增量都喂完就置位
        self.all_fed = asyncio.Event()
        self._state = "loading"
        self._index = 0

    @property
    def state(self) -> str:
        """引擎状态。"""
        return self._state

    async def ensure_loaded(self) -> None:
        self._state = "ready"

    async def feed(self, block: Any) -> str:
        """给一段增量文本，没有增量时给空串。"""
        self.fed.set()
        if self._index >= len(self.partials):
            return ""
        text = self.partials[self._index]
        self._index += 1
        if self._index >= len(self.partials):
            self.all_fed.set()
        return text

    async def flush(self) -> str:
        return self.final


class FakeBlock:
    """假音频块，只带来源标签。"""

    source = "microphone"


class FakeCapture:
    """假采集器，记录开流与关流。"""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self._running = False
        self._blocks = 0

    @property
    def is_running(self) -> bool:
        """采集是不是开着。"""
        return self._running

    async def start(self) -> None:
        self.started += 1
        self._running = True

    async def stop(self) -> None:
        self.stopped += 1
        self._running = False

    async def get_block(self) -> FakeBlock:
        """给一块音频，之后一直等着，等采集停掉时被取消。"""
        self._blocks += 1
        if self._blocks > 1:
            await asyncio.sleep(3600)
        return FakeBlock()


class FakeEngine:
    """假 ASR 引擎，喂块返回部分结果，冲刷返回剩下的文本。"""

    def __init__(self, partial: str = "你好", final: str = "，请介绍一下项目") -> None:
        self.partial = partial
        self.final = final
        self.loaded = 0
        self._state = "loading"
        self.fed = asyncio.Event()

    @property
    def state(self) -> str:
        """引擎状态。"""
        return self._state

    async def ensure_loaded(self) -> None:
        self.loaded += 1
        self._state = "ready"

    async def feed(self, block: Any) -> str:
        self.fed.set()
        return self.partial

    async def flush(self) -> str:
        return self.final


class FailingEngine:
    """加载失败的假 ASR 引擎。"""

    def __init__(self) -> None:
        self._state = "error"

    @property
    def state(self) -> str:
        """引擎状态。"""
        return self._state

    async def ensure_loaded(self) -> None:
        raise RuntimeError("模型加载失败")

    async def feed(self, block: Any) -> str:
        return ""

    async def flush(self) -> str:
        return ""


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _build_messages(
    question: str, question_type: str, context: str | None = None
) -> list[dict[str, str]]:
    """消息构造回调，只记题型。"""
    return [{"role": "user", "content": question}]


def _build_multimodal_messages(
    question: str, image_data_url: str, question_type: str, context: str | None = None
) -> list[dict]:
    """带图消息构造回调。"""
    return [{"role": "user", "content": question}]


def _make_manager(
    client: FakeClient | None = None,
    sink: FakeSink | None = None,
    on_coding_mode: Any = None,
) -> tuple[QuestionManager, FakeClient]:
    """搭一套假实现，返回管理器与假客户端。

    :param on_coding_mode: 注入的代码题通知回调，不给就是没有注入。
    """
    client = client if client is not None else FakeClient()
    manager = QuestionManager(
        make_config(),
        client,
        lambda request_id, session_id: FakeParser(request_id),
        sink if sink is not None else FakeSink(),
        _build_messages,
        build_multimodal_messages=_build_multimodal_messages,
        on_coding_mode=on_coding_mode,
    )
    return manager, client


# ----------------------------------------------------------------------
# 界面 F8 接真采集，见《5-语音输入与流式ASR》F8 交互一节
# ----------------------------------------------------------------------
async def test_f8_calls_the_injected_voice_callback() -> None:
    """界面按 F8 触发装配层注入的语音采集回调。"""
    calls: list[str] = []
    app = InterviewMateApp(
        config=make_config(),
        voice_toggle=lambda: calls.append("toggle"),
    )

    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        await pilot.press("f8")
        await pilot.pause()

    assert calls == ["toggle"]


async def test_f8_awaits_the_async_voice_callback() -> None:
    """回调是协程时界面把它放到后台跑完。"""
    calls: list[str] = []
    called = asyncio.Event()

    async def _toggle() -> None:
        """假的语音采集回调。"""
        calls.append("toggle")
        called.set()

    app = InterviewMateApp(config=make_config(), voice_toggle=_toggle)

    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        await pilot.press("f8")
        await pilot.pause()
        await asyncio.wait_for(called.wait(), timeout=5)

    assert calls == ["toggle"]


async def test_f8_without_callback_does_nothing() -> None:
    """回调没接上时按 F8 只记日志，不抛异常。"""
    app = InterviewMateApp(config=make_config())

    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        await pilot.press("f8")
        await pilot.pause()
        assert app.is_running


async def test_assembled_voice_wires_f8_to_the_socket_voice_toggle() -> None:
    """装配后界面按 F8 与 socket 的 voice-toggle 走同一条路。"""
    ui = FakeApp()
    manager = FakeManager()
    state = app.main.MainState()
    capture = FakeCapture()
    voice = app.main.assemble_voice(
        make_config(),
        ui,
        manager,
        state=state,
        engine=FakeEngine(),
        factory=lambda source: capture,
    )
    state.set_voice_chain(voice)

    # 界面拿到的就是主进程状态上那个 voice-toggle 实现
    assert ui.voice_toggle == state.toggle_voice_capture

    await ui.voice_toggle()

    assert capture.started == 1
    assert state.audio_on is True

    await state.toggle_voice_capture()

    assert capture.stopped == 1
    assert state.audio_on is False


# ----------------------------------------------------------------------
# 检索来源的存与显示，见《6-Dify检索与项目知识》检索来源展示一节
# ----------------------------------------------------------------------
async def test_on_sources_stores_sources_without_showing_the_panel() -> None:
    """检索完成后来源静默存下，不走显示那条路。"""
    ui = FakeApp()
    chain = app.main.assemble_retrieval(make_config(), ui)

    chain.on_sources(SOURCES)

    assert ui.stored_sources == [SOURCES]
    assert ui.shown_sources == []


async def test_f3_opens_the_panel_with_the_stored_sources() -> None:
    """按 F3 浮层才弹出，内容是刚存的来源。"""
    ui = InterviewMateApp(config=make_config())
    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        chain = app.main.assemble_retrieval(make_config(), ui)

        chain.on_sources(SOURCES)
        await pilot.pause()

        # 检索完成后浮层没有自动弹出
        panel = ui.query_one(RagSourcesPanel)
        assert panel.display is False

        await pilot.press("f3")
        await pilot.pause()

        assert panel.display is True
        text = panel.plain_text()
        assert "架构说明.md" in text
        assert "项目分为输入层、核心层与输出层。" in text


async def test_store_rag_sources_does_not_open_the_panel() -> None:
    """只存来源的方法不打开浮层。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()

        app.store_rag_sources(SOURCES)
        await pilot.pause()

        assert app.query_one(RagSourcesPanel).display is False


async def test_show_rag_sources_still_opens_the_panel() -> None:
    """显示语义保留：show_rag_sources 之后浮层可见。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()

        app.show_rag_sources(SOURCES)
        await pilot.pause()

        panel = app.query_one(RagSourcesPanel)
        assert panel.display is True
        assert "架构说明.md" in panel.plain_text()


# ----------------------------------------------------------------------
# 代码题通知分发器切形态，见《7-代码问题模式》纯代码块兜底一节
# ----------------------------------------------------------------------
async def test_coding_question_notifies_coding_mode() -> None:
    """判定出代码题时通知外面「这次是代码题」。"""
    notified: list[bool] = []
    manager, _client = _make_manager(on_coding_mode=notified.append)

    await manager.submit_text(QUESTION_CODING)

    assert notified == [True]


async def test_plain_question_notifies_coding_mode_false() -> None:
    """普通题通知的是假。"""
    notified: list[bool] = []
    manager, _client = _make_manager(on_coding_mode=notified.append)

    await manager.submit_text(QUESTION_PROJECT)

    assert notified == [False]


async def test_coding_notification_happens_before_the_model_request() -> None:
    """通知在发出模型请求之前，模型不给协议事件时围栏块也能进代码区。"""
    order: list[str] = []
    manager, _client = _make_manager(
        client=FakeClient(order),
        on_coding_mode=lambda enabled: order.append("notify"),
    )

    await manager.submit_text(QUESTION_CODING)

    assert order == ["notify", "request"]


async def test_without_coding_callback_behavior_is_unchanged() -> None:
    """没注入代码题通知回调时行为与改动前一致。"""
    manager, client = _make_manager()

    await manager.submit_text(QUESTION_CODING)

    assert len(client.calls) == 1
    session = manager.current_session
    assert session is not None
    assert session.question_type == "coding"


async def test_assembled_manager_wires_coding_mode_to_the_dispatcher() -> None:
    """装配层把代码题通知接到分发器的 set_coding_mode 上。"""
    ui = FakeApp()
    manager = app.main.assemble_manager(make_config(), FakeClient(), ui)

    await manager.submit_text(QUESTION_CODING)

    # 右栏先切到代码形态等着
    assert ui.answer_panel.code_modes == [True]


async def test_code_block_lands_in_the_code_area_without_protocol_events() -> None:
    """模型不按事件协议输出时，围栏块也进右栏代码区。"""

    class CodeBlockClient(FakeClient):
        """只吐一段带围栏的代码，不给任何协议事件。"""

        async def chat_stream(self, messages: list[dict[str, str]], request_id: str):
            self.order.append("request")
            self.calls.append(messages)
            yield "```cpp\nint main() { return 0; }\n```\n"

    ui = InterviewMateApp(config=make_config())
    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        manager = app.main.assemble_manager(make_config(), CodeBlockClient(), ui)

        await manager.submit_text(QUESTION_CODING)
        await pilot.pause()

        assert "int main" in ui.answer_panel.code_text


# ----------------------------------------------------------------------
# 启动时预加载 ASR，见《2-骨架与配置》命令行入口一节第 9 步
# ----------------------------------------------------------------------
async def test_preload_asr_loads_the_model_and_marks_ready() -> None:
    """预加载走引擎的 ensure_loaded，ASR 状态标成 ready。"""
    engine = FakeEngine()
    state = app.main.MainState()
    state.attach_app(FakeApp())

    result = await app.main.preload_asr(make_config(), engine, state)

    assert result == "ready"
    assert engine.loaded == 1
    assert state.asr_state == "ready"


async def test_preload_asr_failure_marks_error_and_does_not_raise() -> None:
    """加载失败只记日志，ASR 状态标成 error，不影响启动。"""
    engine = FailingEngine()
    state = app.main.MainState()
    state.attach_app(FakeApp())

    result = await app.main.preload_asr(make_config(), engine, state)

    assert result == "error"
    assert state.asr_state == "error"


async def test_preload_is_skipped_when_the_config_does_not_ask_for_it() -> None:
    """配置没有要求预加载时不加载模型。"""
    config = make_config()
    config.asr.preload = False
    engine = FakeEngine()
    state = app.main.MainState()
    state.attach_app(FakeApp())

    result = await app.main.preload_asr(config, engine, state)

    assert result == "idle"
    assert engine.loaded == 0
    assert state.asr_state == "idle"


def test_preload_uses_the_real_asr_engine() -> None:
    """预加载用真实 ASR 引擎，不用假引擎顶替。"""
    config = make_config()
    config.providers.asr = "funasr"

    engine = app.main.select_asr_engine(config)

    assert isinstance(engine, FunASREngine)


# ----------------------------------------------------------------------
# 语音的临时文字与题面，见《5-语音输入与流式ASR》F8 交互一节
# ----------------------------------------------------------------------
async def _wait_for_hint(ui: InterviewMateApp, needle: str, timeout: float = 5.0) -> str:
    """等状态栏提示位出现指定字样，返回提示位当前的文字。"""
    bar = ui.query_one(StatusBar)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if needle in bar.hint:
            return bar.hint
        await asyncio.sleep(0.02)
    return bar.hint


async def test_voice_line_shows_the_accumulated_text() -> None:
    """临时文字行显示的是累计文本，不是最后一次增量。"""
    ui = InterviewMateApp(config=make_config())
    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()

        ui.show_voice_text("欢迎大")
        await pilot.pause()
        ui.show_voice_text("家来")
        await pilot.pause()

        panel = ui.query_one(VoiceTextPanel)
        assert panel.text == "欢迎大家来"
        assert panel.display is True


async def test_voice_line_restarts_from_zero_on_the_next_stream() -> None:
    """一条语音流结束后，下一条语音流从零开始累计。"""
    ui = InterviewMateApp(config=make_config())
    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()

        ui.show_voice_text("上一条说的话")
        await pilot.pause()
        ui.clear_voice_text()
        await pilot.pause()
        ui.show_voice_text("下一条说的话")
        await pilot.pause()

        panel = ui.query_one(VoiceTextPanel)
        assert panel.text == "下一条说的话"
        assert "上一条说的话" not in panel.render().plain


async def test_final_voice_text_is_shown_as_the_question() -> None:
    """语音最终文本进右栏题面，走界面已有的 show_question。"""
    ui = FakeApp()
    manager = FakeManager()
    state = app.main.MainState()
    engine = StreamingEngine(["欢迎大", "家来"], "，请介绍一下项目")
    voice = app.main.assemble_voice(
        make_config(),
        ui,
        manager,
        state=state,
        engine=engine,
        factory=lambda source: StreamingCapture(blocks=2),
    )
    state.set_voice_chain(voice)

    await voice.audio_manager.start()
    await asyncio.wait_for(engine.all_fed.wait(), timeout=5)
    await voice.audio_manager.stop()
    await asyncio.wait_for(manager.submitted.wait(), timeout=5)

    assert ui.questions == ["欢迎大家来，请介绍一下项目"]
    assert manager.submissions == ["欢迎大家来，请介绍一下项目"]


async def test_real_voice_chain_shows_the_question_and_clears_the_line() -> None:
    """真实链路：F8 开始 → 部分结果 → F8 停止。

    右栏顶部出现最终题面，临时识别文字行在停止后是空的。
    """
    ui = InterviewMateApp(config=make_config())
    manager = FakeManager()
    state = app.main.MainState()
    state.attach_app(ui)
    engine = StreamingEngine(["欢迎大", "家来"], "，请介绍一下项目")
    capture = StreamingCapture(blocks=2)
    voice = app.main.assemble_voice(
        make_config(),
        ui,
        manager,
        state=state,
        engine=engine,
        factory=lambda source: capture,
    )
    state.set_voice_chain(voice)

    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = ui.query_one(VoiceTextPanel)

        # F8 开始采集
        await pilot.press("f8")
        await asyncio.wait_for(engine.all_fed.wait(), timeout=5)
        await pilot.pause()

        # 临时文字行显示的是累计文本
        assert panel.text == "欢迎大家来"

        # F8 停止采集
        await pilot.press("f8")
        await asyncio.wait_for(manager.submitted.wait(), timeout=5)
        await pilot.pause()

        # 右栏顶部出现最终题面
        assert ui.question_header.question == "欢迎大家来，请介绍一下项目"
        # 停止后临时文字行清空
        assert panel.text == ""
        assert panel.display is False
        # 最终文本自动提交
        assert manager.submissions == ["欢迎大家来，请介绍一下项目"]


async def test_real_voice_chain_reports_empty_result_in_the_status_bar() -> None:
    """识别为空时状态栏提示位给出失败提示，不发起模型请求。"""
    ui = InterviewMateApp(config=make_config())
    manager = FakeManager()
    state = app.main.MainState()
    state.attach_app(ui)
    engine = StreamingEngine([], "")
    capture = StreamingCapture(blocks=1)
    voice = app.main.assemble_voice(
        make_config(),
        ui,
        manager,
        state=state,
        engine=engine,
        factory=lambda source: capture,
    )
    state.set_voice_chain(voice)

    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()

        # F8 开始采集，再按一次停止，这一段没识别到内容
        await pilot.press("f8")
        await asyncio.wait_for(engine.fed.wait(), timeout=5)
        await pilot.press("f8")
        await asyncio.wait_for(capture.stopped_event.wait(), timeout=5)

        hint = await _wait_for_hint(ui, "识别失败")
        await pilot.pause()

        assert "识别失败" in hint
        # 没识别到内容就不发模型请求
        assert manager.submissions == []
        # 识别不出内容不算加载失败，ASR 状态保持 ready
        assert state.asr_state == "ready"


async def test_real_voice_chain_has_no_failure_hint_when_text_is_recognized() -> None:
    """识别到内容时状态栏不出现失败提示。"""
    ui = InterviewMateApp(config=make_config())
    manager = FakeManager()
    state = app.main.MainState()
    state.attach_app(ui)
    engine = StreamingEngine(["欢迎大"], "家来")
    capture = StreamingCapture(blocks=1)
    voice = app.main.assemble_voice(
        make_config(),
        ui,
        manager,
        state=state,
        engine=engine,
        factory=lambda source: capture,
    )
    state.set_voice_chain(voice)

    async with ui.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()

        await pilot.press("f8")
        await asyncio.wait_for(engine.fed.wait(), timeout=5)
        await pilot.press("f8")
        await asyncio.wait_for(manager.submitted.wait(), timeout=5)
        await pilot.pause()

        assert "识别失败" not in ui.query_one(StatusBar).hint
        assert manager.submissions == ["欢迎大家来"]
