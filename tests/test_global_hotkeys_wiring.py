"""全局热键接到主进程装配上的测试。

覆盖 :func:`app.main.assemble_global_hotkeys` 的接线：

- 提供者可用时 F8 按下开采集、松开停采集，F9 按下截图；
- 按住期间的重复按下不再重复开采集，监听自己的去重照常生效；
- 提供者不可用时不接管，界面标记不动，``running`` 为 False；
- 语音键与截图键配成同一个键时只保留语音绑定；
- 状态方法抛异常时只记日志，监听不崩，后续按键照常处理；
- 关掉监听之后到达的按键一律丢弃，收尾期间不会再被按键打断。

监听与状态都按 :mod:`app.inputs.global_hotkeys` 的接口调用，
提供者用假的，不读真实 ``/dev/input``；状态记录每个协程方法的调用次数。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import app.main
from app.config import AppConfig
from tests.test_ui_skeleton import make_config


# ----------------------------------------------------------------------
# 假实现
# ----------------------------------------------------------------------
class FakeProvider:
    """可控的假提供者：记下 start/stop 次数，把 emit 暴露给测试。"""

    def __init__(self, *, available: bool = True) -> None:
        self._available = available
        self.emit: Callable[[str, bool], None] | None = None
        self.start_calls = 0
        self.stop_calls = 0

    def available(self) -> bool:
        """按构造时的取值回答，不探测任何设备。"""
        return self._available

    def start(self, emit: Callable[[str, bool], None]) -> None:
        """记下启动次数与按键回调。"""
        self.start_calls += 1
        self.emit = emit

    def stop(self) -> None:
        """记下停止次数，之后不再上报。"""
        self.stop_calls += 1
        self.emit = None


class FakeState:
    """假主进程状态，记录每个动作被调了几次。

    ``fail`` 不为 None 时每个动作都先记一次调用再抛它，
    用来验证异常只记日志、不冒给监听。
    """

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.starts = 0
        self.stops = 0
        self.shots = 0
        self._fail = fail

    async def start_voice_capture(self) -> dict[str, Any]:
        """记一次按住开始。"""
        self.starts += 1
        if self._fail is not None:
            raise self._fail
        return {"result": "started"}

    async def stop_voice_capture(self) -> dict[str, Any]:
        """记一次松开停止。"""
        self.stops += 1
        if self._fail is not None:
            raise self._fail
        return {"result": "stopped"}

    async def capture_screenshot(self) -> dict[str, Any]:
        """记一次截图。"""
        self.shots += 1
        if self._fail is not None:
            raise self._fail
        return {"result": "captured"}


class FakeApp:
    """假界面，只记接管标记与挂上来的路由器。"""

    def __init__(self) -> None:
        self.marked = 0
        self.router: Any = None
        self.screenshot_manager: Any = None

    def mark_global_hotkeys_active(self) -> None:
        """记录界面被标记为「按键由全局监听处理」。"""
        self.marked += 1

    def attach_input_router(self, router: Any) -> None:
        """记录挂上来的输入路由器。"""
        self.router = router

    def attach_screenshot_manager(self, manager: Any) -> None:
        """记录挂上来的截图管理器。"""
        self.screenshot_manager = manager


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
@contextmanager
def _collect_log_events() -> Iterator[list[logging.LogRecord]]:
    """收集项目记录器上的日志，用来断言事件名。

    临时把记录器放到 DEBUG：接管成功那条是 INFO，默认级别下收不到。
    """
    logger = logging.getLogger("interviewmate")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _events(records: list[logging.LogRecord]) -> list[str]:
    """把日志记录里的事件名取出来。"""
    return [getattr(record, "event", "") for record in records]


async def _wait_until(
    predicate: Callable[[], bool], timeout: float = 1.0
) -> None:
    """轮询等条件成立，避免对事件循环的调度时序做假设。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("等待条件超时")
        await asyncio.sleep(0.01)


async def _drain(rounds: int = 5) -> None:
    """空转几圈事件循环，给投递与新建的任务留出执行机会。"""
    for _ in range(rounds):
        await asyncio.sleep(0)


def _assemble(
    provider: FakeProvider,
    state: FakeState,
    ui: FakeApp,
    config: AppConfig | None = None,
) -> app.main.HotkeyListener:
    """用假提供者装配监听，事件循环取当前运行的那个。"""
    return app.main.assemble_global_hotkeys(
        config if config is not None else make_config(),
        state,
        ui,
        provider=provider,
        loop=asyncio.get_running_loop(),
    )


# ----------------------------------------------------------------------
# 提供者可用：F8 切换采集、F9 截图
# ----------------------------------------------------------------------
async def test_available_provider_wires_f8_press_release_and_f9_press() -> None:
    """F8 按下开采集、松开停采集，F9 按下截图，重复按下不再重复开始。"""
    provider = FakeProvider()
    state = FakeState()
    ui = FakeApp()
    listener = _assemble(provider, state, ui)
    try:
        assert listener is not None
        assert listener.running is True
        assert provider.start_calls == 1
        assert ui.marked == 1
        assert provider.emit is not None

        # F8 按下：开始采集
        provider.emit("f8", True)
        await _wait_until(lambda: state.starts == 1)

        # 按住期间的重复按下（另一台键盘或键盘重复）不再开一次流
        provider.emit("f8", True)
        await _drain()
        assert state.starts == 1
        assert state.stops == 0

        # F8 松开：停止集采，最终文本由状态自己冲刷提交
        provider.emit("f8", False)
        await _wait_until(lambda: state.stops == 1)

        # F9 按下：截图
        provider.emit("f9", True)
        await _wait_until(lambda: state.shots == 1)

        assert state.starts == 1
        assert state.stops == 1
        assert state.shots == 1
    finally:
        listener.stop()

    assert listener.running is False
    assert provider.stop_calls == 1


async def test_assembled_listener_takes_over_the_ui_bindings() -> None:
    """接管成功后界面被标记，再按 F8 不会走界面的切换语义。"""
    provider = FakeProvider()
    state = FakeState()
    ui = FakeApp()

    listener = _assemble(provider, state, ui)
    try:
        assert ui.marked == 1
        assert listener.running is True
    finally:
        listener.stop()


async def test_unavailable_provider_does_not_take_over() -> None:
    """提供者不可用时 start 失败：不标记界面，running 为 False。"""
    provider = FakeProvider(available=False)
    state = FakeState()
    ui = FakeApp()

    listener = _assemble(provider, state, ui)

    assert listener is not None
    assert listener.running is False
    assert provider.start_calls == 0
    assert ui.marked == 0
    listener.stop()  # 收尾照常调 stop，重复调用安全


async def test_same_key_for_voice_and_screenshot_keeps_only_voice() -> None:
    """两个键配成同一个键时只保留语音绑定，并记一条 WARNING。"""
    config = make_config(hotkeys={"voice_toggle": "F8", "screenshot": "f8"})
    provider = FakeProvider()
    state = FakeState()
    ui = FakeApp()

    with _collect_log_events() as records:
        listener = _assemble(provider, state, ui, config)
    try:
        assert listener.running is True
        assert provider.emit is not None

        provider.emit("f8", True)
        await _wait_until(lambda: state.starts == 1)
        # 这个键没有被当成截图键，截图一次都不触发
        assert state.shots == 0

        provider.emit("f8", False)
        await _wait_until(lambda: state.stops == 1)

        # 另一个键不在绑定表里
        provider.emit("f9", True)
        await _drain()
        assert state.shots == 0
    finally:
        listener.stop()

    assert "main.hotkeys.conflict" in _events(records)


async def test_action_failure_is_logged_and_listener_survives() -> None:
    """状态方法抛异常时只记日志，监听保持运行，后续按键照常处理。"""
    provider = FakeProvider()
    state = FakeState(fail=RuntimeError("状态方法炸了"))
    ui = FakeApp()

    with _collect_log_events() as records:
        listener = _assemble(provider, state, ui)
        try:
            assert provider.emit is not None
            provider.emit("f8", True)
            await _wait_until(lambda: state.starts == 1)
            await _wait_until(
                lambda: "main.hotkeys.action_failed" in _events(records)
            )

            # 监听没崩：再按一次照常分发
            provider.emit("f9", True)
            await _wait_until(lambda: state.shots == 1)
            assert listener.running is True
        finally:
            listener.stop()

    failed = [
        record
        for record in records
        if getattr(record, "event", "") == "main.hotkeys.action_failed"
    ]
    assert failed
    fields = failed[-1].fields  # type: ignore[attr-defined]
    assert fields["key"] in {"f8", "f9"}
    assert fields["error"] == "RuntimeError"


async def test_keys_after_stop_are_dropped() -> None:
    """收尾关掉监听之后到达的按键一律丢弃，不会再动状态。"""
    provider = FakeProvider()
    state = FakeState()
    ui = FakeApp()

    with _collect_log_events() as records:
        listener = _assemble(provider, state, ui)
        assert listener.running is True
        emit = provider.emit
        assert emit is not None
        listener.stop()

        # 后台线程里可能已经拿到的 emit 在停止之后才送到
        emit("f8", True)
        await _drain()
        assert state.starts == 0
        assert listener.running is False

        # 重复 stop 安全
        listener.stop()
        assert listener.running is False

    # 关掉之后没有可用告警：不可用只该在启动失败时记
    assert "main.hotkeys.unavailable" not in _events(records)


async def test_active_listener_is_logged_once() -> None:
    """启动成功记一条接管日志，启动失败记不可用告警。"""
    ui = FakeApp()
    state = FakeState()

    with _collect_log_events() as active_records:
        listener = _assemble(FakeProvider(), state, ui)
    listener.stop()

    with _collect_log_events() as missing_records:
        _assemble(FakeProvider(available=False), state, ui)

    assert _events(active_records).count("main.hotkeys.active") == 1
    assert "main.hotkeys.unavailable" not in _events(active_records)
    assert "main.hotkeys.unavailable" in _events(missing_records)