"""全局热键层与 Linux 读 ``/dev/input`` 后端的测试。

覆盖：

- :class:`~app.inputs.global_hotkeys.HotkeyListener` 的去重语义：按下两次只触发
  一次 ``on_press``，松开触发一次 ``on_release``，没按过直接松开不触发，
  ``stop()`` 之后到达的按键一律丢弃；
- 提供者不可用、启动抛异常时 ``start()`` 返回 False 且不向外抛；
- 触发逻辑与 24 字节 ``input_event`` 的解析：按下、松开、长按重复，
  非 F 键与 EV_SYN 被过滤；
- 键盘能力判断：键集含 KEY_A 算键盘，只含 KEY_POWER 不算；
- ioctl 的结果按 CPython 语义从传入的 bytearray 里取：设备名与能力位都
  不能拿返回值当字节用，假 fcntl 会故意返回整数；
- :class:`~app.platform.linux.hotkeys.EvdevHotkeyProvider` 的设备目录为空、
  设备是普通文件、设备没权限时 ``available()`` 为 False 且 ``start()`` 不抛；
- Linux 平台对象挂 ``EvdevHotkeyProvider``，Windows 平台对象挂
  :class:`~app.inputs.global_hotkeys.NullHotkeyProvider`。

不打开真实的 ``/dev/input``：设备目录用 pytest 的 ``tmp_path`` 注入。
"""

from __future__ import annotations

import asyncio
import os
import struct
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from app.inputs.global_hotkeys import (
    GlobalHotkeyProvider,
    HotkeyHandlers,
    HotkeyListener,
    NullHotkeyProvider,
)
from app.platform.linux import LinuxPlatform
from app.platform.linux import hotkeys as linux_hotkeys
from app.platform.linux.hotkeys import (
    KEY_NAME_TO_CODE,
    EvdevHotkeyProvider,
    _parse_key_event,
    is_keyboard_capability,
)
from app.platform.windows import WindowsPlatform

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


def _event(event_type: int, code: int, value: int) -> bytes:
    """按 64 位 Linux 的 input_event 结构体拼一条记录。"""
    return struct.pack("llHHi", 0, 0, event_type, code, value)


def _bits(*set_bits: int, size: int = 96) -> bytes:
    """造一个位集，``set_bits`` 里的位为 1。"""
    buffer = bytearray(size)
    for bit in set_bits:
        buffer[bit // 8] |= 1 << (bit % 8)
    return bytes(buffer)


class _FakeIoctl:
    """按 CPython 语义工作的假 ``fcntl``：写缓冲区，返回整数。

    ``fcntl.ioctl`` 的第三个参数是可变缓冲区时，内核把结果原地写进去，
    返回值是系统调用的整数返回（``EVIOCGNAME`` 是名字长度）。
    这里刻意返回整数，把返回值当字节用的实现会解析出零字节而不是设备名。
    """

    def __init__(self, *, name: str, event_bits: bytes, key_bits: bytes) -> None:
        self.name = name
        self.event_bits = event_bits
        self.key_bits = key_bits

    def ioctl(self, fd: int, request: int, buf: bytearray) -> int:
        """把模拟的设备信息写进 ``buf``，返回一个整数。"""
        if request == linux_hotkeys.EVIOCGNAME:
            payload = self.name.encode("utf-8")
            buf[: len(payload)] = payload
            if len(payload) < len(buf):
                buf[len(payload)] = 0  # 内核用 NUL 结尾
            return len(payload)
        if request == linux_hotkeys._eviocgbit(
            linux_hotkeys.EV_SYN, linux_hotkeys.EVENT_BITS_SIZE
        ):
            data = self.event_bits
        elif request == linux_hotkeys._eviocgbit(
            linux_hotkeys.EV_KEY, linux_hotkeys.KEY_BITS_SIZE
        ):
            data = self.key_bits
        else:
            raise OSError(f"不认识的 ioctl 请求：0x{request:x}")
        buf[: len(data)] = data[: len(buf)]
        return len(data)


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
    provider = _FakeProvider(available_error=OSError("读不了设备"))
    listener = HotkeyListener(
        provider,
        {"f8": HotkeyHandlers(on_press=lambda: None)},
    )

    assert listener.start() is False
    assert listener.running is False
    assert provider.start_calls == 0


async def test_start_returns_false_when_provider_start_raises() -> None:
    """提供者启动时抛异常，start() 返回 False 且不抛。"""
    provider = _FakeProvider(start_error=RuntimeError("设备没了"))
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
# input_event 解析
# ----------------------------------------------------------------------


def test_supported_key_table_covers_f1_to_f12() -> None:
    """支持表只有 f1..f12，键码与内核定义一致。"""
    assert KEY_NAME_TO_CODE == {
        "f1": 59,
        "f2": 60,
        "f3": 61,
        "f4": 62,
        "f5": 63,
        "f6": 64,
        "f7": 65,
        "f8": 66,
        "f9": 67,
        "f10": 68,
        "f11": 87,
        "f12": 88,
    }


@pytest.mark.parametrize("name, code", sorted(KEY_NAME_TO_CODE.items()))
def test_parse_key_event_maps_press_and_release(name: str, code: int) -> None:
    """每条支持键的按下与松开都解析成 ``(键名, 布尔)``。"""
    assert _parse_key_event(_event(1, code, 1)) == (name, True)
    assert _parse_key_event(_event(1, code, 0)) == (name, False)


def test_parse_key_event_filters_repeat_syn_and_other_keys() -> None:
    """长按重复、EV_SYN、非 F 键与长度不对的记录都被过滤。"""
    assert _parse_key_event(_event(1, 66, 2)) is None  # 长按重复
    assert _parse_key_event(_event(0, 0, 0)) is None  # EV_SYN
    assert _parse_key_event(_event(1, 30, 1)) is None  # KEY_A 不在支持表
    assert _parse_key_event(_event(1, 66, 7)) is None  # 未知取值
    assert _parse_key_event(b"\x00" * 4) is None  # 长度不对


# ----------------------------------------------------------------------
# 键盘能力判断
# ----------------------------------------------------------------------


def test_keyboard_capability_requires_ev_key_and_key_a() -> None:
    """事件类型含 EV_KEY 且键集含 KEY_A 才算键盘。"""
    event_bits = _bits(1)  # EV_KEY
    assert is_keyboard_capability(event_bits, _bits(30, 116)) is True
    # 只有 KEY_POWER(116) 没有字母键，不算键盘
    assert is_keyboard_capability(event_bits, _bits(116)) is False
    # 没有 EV_KEY 事件类型，不算键盘
    assert is_keyboard_capability(_bits(0), _bits(30)) is False


# ----------------------------------------------------------------------
# EvdevHotkeyProvider：只碰 tmp_path，不打开真实 /dev/input
# ----------------------------------------------------------------------


def test_evdev_provider_empty_root_is_unavailable(tmp_path: Path) -> None:
    """设备目录为空时 available() 为 False，start() 不抛，重复 stop 安全。"""
    provider = EvdevHotkeyProvider(root=str(tmp_path))
    assert provider.available() is False

    provider.start(lambda key, pressed: None)  # 不抛
    provider.stop()
    provider.stop()  # 重复 stop 安全
    assert provider.available() is False


def test_evdev_provider_regular_file_is_skipped(tmp_path: Path) -> None:
    """event* 是普通文件、读不到能力位时跳过该设备，不抛。"""
    (tmp_path / "event0").write_bytes(b"\x00" * 24)
    provider = EvdevHotkeyProvider(root=str(tmp_path))

    assert provider.available() is False
    provider.start(lambda key, pressed: None)
    provider.stop()
    assert provider.available() is False


def test_ioctl_result_is_read_from_buffer_not_from_return_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """设备名与能力位从 ioctl 的缓冲区取，不能把整数返回值当字节。

    CPython 的 ``fcntl.ioctl`` 在参数是可变缓冲区时原地写入内容、返回
    系统调用的整数返回。拿返回值当字节会让设备名变成一串零字节、能力位
    全零，真实机器上 ``available()`` 因此恒为 False。
    """
    device = tmp_path / "event0"
    device.write_bytes(b"\x00" * 24)
    keyboard_name = "AT Translated Set 2 keyboard"
    fake = _FakeIoctl(
        name=keyboard_name,
        event_bits=_bits(linux_hotkeys.EV_KEY),
        key_bits=_bits(linux_hotkeys.KEY_A, 116),  # KEY_A 与 KEY_POWER
    )
    monkeypatch.setattr(linux_hotkeys, "fcntl", fake)

    fd = os.open(str(device), os.O_RDONLY)
    try:
        assert linux_hotkeys._device_name(fd) == keyboard_name
        event_bits, key_bits = linux_hotkeys._device_capabilities(fd)
    finally:
        os.close(fd)
    assert linux_hotkeys.is_keyboard_capability(event_bits, key_bits) is True

    provider = EvdevHotkeyProvider(root=str(tmp_path))
    assert provider.available() is True

    # 只含 KEY_POWER(116) 的电源键不是键盘
    fake.key_bits = _bits(116)
    assert provider.available() is False

    # 设备名为空不算失败：键盘判定只看能力位
    fake.name = ""
    fake.key_bits = _bits(linux_hotkeys.KEY_A)
    assert provider.available() is True


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root 不受文件权限限制，造不出 PermissionError",
)
def test_evdev_provider_permission_denied_is_skipped(tmp_path: Path) -> None:
    """设备没权限时跳过该设备，available() 为 False，start() 不抛。"""
    device = tmp_path / "event0"
    device.write_bytes(b"\x00" * 24)
    device.chmod(0o000)
    provider = EvdevHotkeyProvider(root=str(tmp_path))

    assert provider.available() is False
    provider.start(lambda key, pressed: None)
    provider.stop()


def test_evdev_provider_start_twice_is_safe(tmp_path: Path) -> None:
    """没有可用键盘时重复 start 也不抛，available() 仍然为 False。"""
    provider = EvdevHotkeyProvider(root=str(tmp_path))
    provider.start(lambda key, pressed: None)
    provider.start(lambda key, pressed: None)
    provider.stop()
    assert provider.available() is False


# ----------------------------------------------------------------------
# 平台接线
# ----------------------------------------------------------------------


def test_linux_platform_exposes_evdev_hotkeys() -> None:
    """Linux 平台对象带 EvdevHotkeyProvider，且满足协议。"""
    platform = LinuxPlatform()
    assert isinstance(platform.hotkeys, EvdevHotkeyProvider)
    assert isinstance(platform.hotkeys, GlobalHotkeyProvider)


def test_windows_platform_exposes_null_hotkeys() -> None:
    """Windows 平台对象带 NullHotkeyProvider。"""
    platform = WindowsPlatform()
    assert isinstance(platform.hotkeys, NullHotkeyProvider)
    assert platform.hotkeys.available() is False


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="只在 Linux 上断言平台门面的实现",
)
def test_get_platform_reuses_one_hotkey_provider() -> None:
    """门面返回的平台对象反复取 hotkeys，拿到的是同一个实例。"""
    from app.platform import get_platform

    assert get_platform().hotkeys is get_platform().hotkeys