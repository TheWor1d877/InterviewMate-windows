"""按住语义的语音采集测试。

覆盖「按下开始、松开停止」这条路：

- 按下时锁被占，这次的按下忽略；
- 按住开始：第一次真开流，重复按下不再开一次；
- 松开停止：最终文本照常送出去，有新的最终文本时不动路由器里的截图暂存；
- 没听到内容：松开后只给一句提示，不提交；
- 没在采集时松开：空操作，不碰采集器。

采集器、ASR 引擎、界面与输入路由器都用假实现，不加载真实模型，也不开音频设备。
切换路径的行为由 ``tests/test_main_assembly.py`` 里既有的用例看着，这里不重写。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import app.main
from app.audio.audio_manager import (
    RESULT_EMPTY,
    RESULT_STARTED,
    RESULT_STOPPED,
    AudioManager,
)
from app.inputs.hotkey_controller import RESULT_IGNORED, HotkeyController

# 松开后要送出去的最终文本
FINAL_TEXT = "请介绍一下你的项目"


# ----------------------------------------------------------------------
# 假实现
# ----------------------------------------------------------------------
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
        """开始采集。"""
        self.started += 1
        self._running = True

    async def stop(self) -> None:
        """停止采集。"""
        self.stopped += 1
        self._running = False

    async def get_block(self) -> FakeBlock:
        """先给一块音频，之后一直等着，等采集停掉时被取消。"""
        self._blocks += 1
        if self._blocks > 1:
            await asyncio.sleep(3600)
        return FakeBlock()


class FakeEngine:
    """假 ASR 引擎，喂块返回部分结果，冲刷返回剩下的文本。"""

    def __init__(self, partial: str = "", final: str = "") -> None:
        self.partial = partial
        self.final = final
        self._state = "loading"
        # 喂过第一块时置位，测试用它等识别真的开始
        self.fed = asyncio.Event()
        self.flush_calls = 0

    @property
    def state(self) -> str:
        """引擎状态。"""
        return self._state

    async def ensure_loaded(self) -> None:
        """加载模型。假引擎不加载，直接置成 ready。"""
        self._state = "ready"

    async def feed(self, block: Any) -> str:
        """喂一块音频，返回部分结果。"""
        self.fed.set()
        return self.partial

    async def flush(self) -> str:
        """冲刷，返回剩下的文本。"""
        self.flush_calls += 1
        text = self.final
        self.final = ""
        return text


class FakeApp:
    """假界面，只收集状态栏与提示位。"""

    def __init__(self) -> None:
        self.statuses: list[dict[str, Any]] = []
        self.hints: list[str] = []
        self.cleared = 0

    async def publish_status(self, **fields: Any) -> None:
        """收下一次状态快照。"""
        self.statuses.append(dict(fields))

    async def publish_hint(self, hint: str) -> None:
        """收下一次提示。"""
        self.hints.append(hint)

    def clear_voice_text(self) -> None:
        """清临时识别文字。"""
        self.cleared += 1


class FakeRouter:
    """假输入路由器，只记录音会话的两个回调。"""

    def __init__(self) -> None:
        self.session_started = 0
        self.finished = 0

    def voice_session_started(self) -> None:
        """录音开始。"""
        self.session_started += 1

    async def voice_finished(self) -> None:
        """录音会话收尾。"""
        self.finished += 1


@dataclass
class _Harness:
    """一套替身与被测的主进程状态。"""

    state: app.main.MainState
    manager: AudioManager
    hotkey: HotkeyController
    capture: FakeCapture
    engine: FakeEngine
    router: FakeRouter
    app: FakeApp
    start_calls: list[int]
    stop_calls: list[int]


def _make_harness(
    *,
    partial: str = "",
    final: str = "",
) -> _Harness:
    """造一条用假采集器与假引擎的语音链路。

    最终文本回调照 ``assemble_voice`` 的做法记下文本与最终文本计数，
    按住路径的判断就靠这个计数。
    """
    capture = FakeCapture()
    engine = FakeEngine(partial=partial, final=final)
    router = FakeRouter()
    ui = FakeApp()
    state = app.main.MainState()

    async def _on_final_text(text: str, source: str) -> None:
        """最终文本：记到状态上，计数加一。"""
        stripped = text.strip()
        if not stripped:
            return
        state.last_voice_text = stripped
        state.voice_final_seq += 1

    manager = AudioManager(
        lambda source: capture,
        engine,
        _on_final_text,
        source="auto",
    )
    hotkey = HotkeyController(manager)
    state.set_voice_chain(
        app.main.VoiceChain(
            audio_manager=manager,
            hotkey=hotkey,
            engine=engine,
            source="auto",
        )
    )
    state.set_input_router(router)
    state.attach_app(ui)

    # 记下启停到底被调了几次，用来断言重复按下不再开流、没在采集时不碰采集器
    start_calls: list[int] = []
    stop_calls: list[int] = []
    original_start = manager.start
    original_stop = manager.stop

    async def _counting_start() -> None:
        start_calls.append(1)
        await original_start()

    async def _counting_stop() -> str:
        stop_calls.append(1)
        return await original_stop()

    manager.start = _counting_start  # type: ignore[method-assign]
    manager.stop = _counting_stop  # type: ignore[method-assign]

    return _Harness(
        state=state,
        manager=manager,
        hotkey=hotkey,
        capture=capture,
        engine=engine,
        router=router,
        app=ui,
        start_calls=start_calls,
        stop_calls=stop_calls,
    )


# ----------------------------------------------------------------------
# 按下
# ----------------------------------------------------------------------
async def test_handle_voice_start_is_ignored_while_lock_is_held() -> None:
    """上一次动作还没结束时，这次的按下忽略。"""
    harness = _make_harness()

    async with harness.hotkey._lock:
        result = await harness.hotkey.handle_voice_start()

    assert result == RESULT_IGNORED
    assert harness.capture.started == 0


async def test_start_voice_capture_runs_once_and_ignores_repeats() -> None:
    """按住开始：第一次真开流，重复按下不再开一次。"""
    harness = _make_harness()

    first = await harness.state.start_voice_capture()
    second = await harness.state.start_voice_capture()

    assert first["result"] == RESULT_STARTED
    assert first["audio"] == "on"
    assert harness.state.audio_on is True
    assert harness.capture.started == 1
    assert len(harness.start_calls) == 1
    assert second["result"] == RESULT_IGNORED
    assert second["note"] == "已在采集，忽略按住开始"
    assert harness.capture.started == 1
    assert len(harness.start_calls) == 1
    assert harness.router.session_started == 1


# ----------------------------------------------------------------------
# 松开
# ----------------------------------------------------------------------
async def test_stop_voice_capture_submits_final_text_and_skips_router_finish() -> None:
    """松开停止：最终文本送出去，计数变了就不动路由器里的截图暂存。"""
    harness = _make_harness(final=FINAL_TEXT)

    assert (await harness.state.start_voice_capture())["result"] == RESULT_STARTED
    # 等识别真的喂进第一块，冲刷时才有识别流可冲
    await harness.engine.fed.wait()
    result = await harness.state.stop_voice_capture()

    assert result["result"] == RESULT_STOPPED
    assert result["audio"] == "off"
    assert result["text"] == FINAL_TEXT
    assert harness.state.last_voice_text == FINAL_TEXT
    assert harness.state.voice_final_seq == 1
    assert harness.capture.stopped == 1
    # 有新最终文本：本次停止会走 route_voice，不在这里释放截图
    assert harness.router.finished == 0
    assert harness.engine.flush_calls == 1


async def test_empty_stop_publishes_hint_once() -> None:
    """没听到内容：松开后只给一句提示，不提交。"""
    harness = _make_harness(partial="", final="")

    assert (await harness.state.start_voice_capture())["result"] == RESULT_STARTED
    await harness.engine.fed.wait()
    result = await harness.state.stop_voice_capture()

    assert result["result"] == RESULT_EMPTY
    assert harness.app.hints == [app.main.VOICE_EMPTY_HINT]
    assert harness.state.last_voice_text == ""
    assert harness.state.voice_final_seq == 0
    # 计数没变，说明不会走 route_voice，路由器在这里收尾
    assert harness.router.finished == 1


async def test_stop_voice_capture_without_running_is_ignored() -> None:
    """没在采集时松开：空操作，不碰采集器。"""
    harness = _make_harness()

    result = await harness.state.stop_voice_capture()

    assert result["result"] == RESULT_IGNORED
    assert result["note"] == "没在采集，忽略按住停止"
    assert harness.stop_calls == []
    assert harness.capture.stopped == 0
    assert harness.state.audio_on is False