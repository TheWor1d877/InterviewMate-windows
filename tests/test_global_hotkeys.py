"""全局热键层与 Windows/AHK 触发链路的测试。

覆盖：

- :class:`~app.inputs.global_hotkeys.HotkeyListener` 的去重语义：按下两次只触发
  一次 ``on_press``，松开触发一次 ``on_release``，没按过直接松开不触发，
  ``stop()`` 之后到达的按键一律丢弃；
- 提供者不可用、启动抛异常时 ``start()`` 返回 False 且不向外抛；
- Windows 平台门面用 :class:`~app.inputs.global_hotkeys.NullHotkeyProvider`；
- AutoHotkey v2 脚本：存在、带 v2 声明与 ``#HotIf`` 终端保护、
  F8/F9 分别触发 ``hotkey voice-toggle`` 与 ``hotkey screenshot``，
  且脚本里不写死本机绝对路径；
- AHK 调用的 CLI 链路：``hotkey voice-toggle|screenshot`` 按动作发到 socket，
  socket 侧用假实现，不连真实主进程。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import app.cli
from app.cli import main as cli_main
from app.inputs.global_hotkeys import (
    HotkeyHandlers,
    HotkeyListener,
    NullHotkeyProvider,
)
from app.main import EXIT_OK
from app.platform.windows import WindowsPlatform
from app.runtime.socket_protocol import Response

# ----------------------------------------------------------------------
# 假对象与工具
# ----------------------------------------------------------------------


class _FakeProvider:
    """可控的假提供者：记下 start/stop 次数，并把 emit 暴露给测试。"""

    def __init__(
        self,
        *,
        available: bool = True,
        available_error: Exception | None = None,
        start_error: Exception | None = None,
    ) -> None:
        self._available = available
        self._available_error = available_error
        self._start_error = start_error
        self.emit: Callable[[str, bool], None] | None = None
        self.start_calls = 0
        self.stop_calls = 0

    def available(self) -> bool:
        if self._available_error is not None:
            raise self._available_error
        return self._available

    def start(self, emit: Callable[[str, bool], None]) -> None:
        self.start_calls += 1
        if self._start_error is not None:
            raise self._start_error
        self.emit = emit

    def stop(self) -> None:
        self.stop_calls += 1
        self.emit = None


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    """轮询等待条件成立，避免对事件循环的调度时序做假设。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("等待条件超时")
        await asyncio.sleep(0.01)


# ----------------------------------------------------------------------
# HotkeyListener：去重与生命周期
# ----------------------------------------------------------------------


async def test_press_twice_triggers_on_press_once_and_release_once() -> None:
    """按下两次（含大小写不同）只触发一次 on_press，松开触发一次 on_release。"""
    provider = _FakeProvider()
    presses: list[str] = []
    releases: list[str] = []
    listener = HotkeyListener(
        provider,
        {
            " F8 ": HotkeyHandlers(
                on_press=lambda: presses.append("f8"),
                on_release=lambda: releases.append("f8"),
            )
        },
    )
    assert listener.start() is True
    assert listener.running is True
    assert provider.start_calls == 1

    provider.emit("f8", True)
    provider.emit("F8", True)  # 另一台键盘同时上报同一个键
    await _wait_until(lambda: len(presses) == 1)
    assert presses == ["f8"]
    assert releases == []

    provider.emit("f8", False)
    await _wait_until(lambda: len(releases) == 1)
    assert releases == ["f8"]

    provider.emit("f8", False)  # 重复松开丢掉
    await asyncio.sleep(0.02)
    assert releases == ["f8"]

    listener.stop()
    assert listener.running is False
    assert provider.stop_calls == 1


async def test_release_without_press_is_dropped() -> None:
    """没按过就松开不触发 on_release。"""
    provider = _FakeProvider()
    releases: list[int] = []
    listener = HotkeyListener(
        provider,
        {
            "f9": HotkeyHandlers(
                on_press=lambda: None,
                on_release=lambda: releases.append(1),
            )
        },
    )
    assert listener.start() is True

    provider.emit("f9", False)
    await asyncio.sleep(0.02)
    assert releases == []

    listener.stop()


async def test_emit_after_stop_is_dropped() -> None:
    """stop() 之后到达的按键一律丢弃。"""
    provider = _FakeProvider()
    presses: list[int] = []
    listener = HotkeyListener(
        provider,
        {"f8": HotkeyHandlers(on_press=lambda: presses.append(1))},
    )
    assert listener.start() is True
    # 先留下 provider 启动时拿到的 emit，模拟停止后还有后台线程上报
    emit = provider.emit
    assert emit is not None
    listener.stop()

    emit("f8", True)
    await asyncio.sleep(0.02)
    assert presses == []


async def test_unbound_key_is_ignored() -> None:
    """不在绑定表里的键不触发任何回调。"""
    provider = _FakeProvider()
    presses: list[int] = []
    listener = HotkeyListener(
        provider,
        {"f8": HotkeyHandlers(on_press=lambda: presses.append(1))},
    )
    assert listener.start() is True

    provider.emit("f7", True)
    await asyncio.sleep(0.02)
    assert presses == []

    listener.stop()


async def test_callback_exception_does_not_break_listener() -> None:
    """回调抛异常只记日志，监听保持运行，后续按键照常处理。"""
    provider = _FakeProvider()
    calls: list[str] = []

    def _boom() -> None:
        calls.append("press")
        raise RuntimeError("回调炸了")

    listener = HotkeyListener(
        provider,
        {
            "f8": HotkeyHandlers(
                on_press=_boom,
                on_release=lambda: calls.append("release"),
            )
        },
    )
    assert listener.start() is True

    provider.emit("f8", True)
    await _wait_until(lambda: calls == ["press"])
    assert listener.running is True

    provider.emit("f8", False)
    await _wait_until(lambda: calls == ["press", "release"])

    listener.stop()


async def test_emit_from_worker_thread_reaches_event_loop() -> None:
    """后台线程调 emit 时回调被投递到事件循环。"""
    provider = _FakeProvider()
    presses: list[str] = []
    listener = HotkeyListener(
        provider,
        {"f8": HotkeyHandlers(on_press=lambda: presses.append("f8"))},
    )
    assert listener.start() is True
    emit = provider.emit
    assert emit is not None

    await asyncio.to_thread(emit, "F8", True)
    await _wait_until(lambda: presses == ["f8"])

    listener.stop()


def test_emit_with_closed_loop_is_swallowed() -> None:
    """事件循环已关闭时 emit 只记日志，不向上抛。"""
    provider = _FakeProvider()
    loop = asyncio.new_event_loop()
    listener = HotkeyListener(
        provider,
        {"f8": HotkeyHandlers(on_press=lambda: None)},
        loop=loop,
    )
    assert listener.start() is True
    emit = provider.emit
    assert emit is not None

    loop.close()
    emit("f8", True)  # 不抛

    listener.stop()


async def test_start_returns_false_when_provider_unavailable() -> None:
    """提供者不可用时 start() 返回 False，不启动 provider，也不抛。"""
    provider = _FakeProvider(available=False)
    listener = HotkeyListener(
        provider,
        {"f8": HotkeyHandlers(on_press=lambda: None)},
    )

    assert listener.start() is False
    assert listener.running is False
    assert provider.start_calls == 0

    listener.stop()


async def test_start_returns_false_when_availability_check_raises() -> None:
    """探测可用性时抛异常，按不可用处理，start() 返回 False 且不抛。"""
    provider = _FakeProvider(available_error=OSError("探测失败"))
    listener = HotkeyListener(
        provider,
        {"f8": HotkeyHandlers(on_press=lambda: None)},
    )

    assert listener.start() is False
    assert listener.running is False
    assert provider.start_calls == 0


async def test_start_returns_false_when_provider_start_raises() -> None:
    """提供者启动时抛异常，start() 返回 False 且不抛。"""
    provider = _FakeProvider(start_error=RuntimeError("启动失败"))
    listener = HotkeyListener(
        provider,
        {"f8": HotkeyHandlers(on_press=lambda: None)},
    )

    assert listener.start() is False
    assert listener.running is False


def test_null_provider_is_always_unavailable() -> None:
    """NullHotkeyProvider 恒不可用，start/stop 是空操作。"""
    provider = NullHotkeyProvider()
    assert provider.available() is False
    provider.start(lambda key, pressed: None)
    provider.stop()
    assert provider.available() is False


# ----------------------------------------------------------------------
# Windows 平台接线与 AHK 脚本
# ----------------------------------------------------------------------


def test_windows_platform_exposes_null_hotkeys() -> None:
    """Windows 平台对象带 NullHotkeyProvider，全局热键由 AHK 脚本负责。"""
    platform = WindowsPlatform()
    assert isinstance(platform.hotkeys, NullHotkeyProvider)
    assert platform.hotkeys.available() is False


def _ahk_script() -> Path:
    """仓库内的 AHK 热键脚本路径。"""
    return Path(__file__).resolve().parents[1] / "scripts" / "windows" / "hotkeys.ahk"


def test_ahk_script_exists_and_declares_v2() -> None:
    """热键脚本存在，并要求 AutoHotkey v2。"""
    script = _ahk_script()
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "#Requires AutoHotkey v2.0" in text


def test_ahk_script_guards_windows_terminal() -> None:
    """终端窗口里不接管 F8/F9，避免与终端内绑定重复触发。"""
    text = _ahk_script().read_text(encoding="utf-8")
    assert '#HotIf !WinActive("ahk_exe WindowsTerminal.exe")' in text
    assert "F8::RunHotkey(\"voice-toggle\")" in text
    assert "F9::RunHotkey(\"screenshot\")" in text


def test_ahk_script_calls_cli_hotkey_actions() -> None:
    """脚本经 ``python -m app.cli hotkey`` 触发两个动作。"""
    text = _ahk_script().read_text(encoding="utf-8")
    assert "-m app.cli hotkey " in text
    assert '"voice-toggle"' in text
    assert '"screenshot"' in text


def test_ahk_script_has_no_machine_specific_paths() -> None:
    """脚本用自身目录推导仓库根，不写死本机绝对路径。"""
    text = _ahk_script().read_text(encoding="utf-8")
    assert "A_ScriptDir" in text
    assert "D:\\" not in text
    assert "C:\\Users" not in text


# ----------------------------------------------------------------------
# AHK 调用的 CLI 链路
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("voice-toggle", "voice-toggle"),
        ("screenshot", "screenshot"),
    ],
)
def test_ahk_hotkey_action_reaches_socket(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    action: str,
    expected: str,
) -> None:
    """AHK 调 ``hotkey <动作>`` 时，CLI 把同名动作发到 socket。"""
    sent: list[str] = []

    def fake_send_request(
        request_action: str, *args: Any, **kwargs: Any
    ) -> Response:
        sent.append(request_action)
        return Response(ok=True, payload={"result": "ok"})

    monkeypatch.setattr(app.cli, "is_server_alive", lambda path=None: True)
    monkeypatch.setattr(app.cli, "send_request", fake_send_request)

    assert cli_main(["hotkey", action]) == EXIT_OK
    assert sent == [expected]
    assert '"ok": true' in capsys.readouterr().out