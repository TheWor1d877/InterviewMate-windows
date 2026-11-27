"""GNOME 截图后端的测试。

覆盖《4-截图输入与多模态》「截图方式」一节的两条路：

- 静默截图扩展：扩展在线时走扩展抓屏，屏幕上不弹任何提示；
  扩展不在线或调用失败时回退 XDG Portal，Portal 那条会弹一次确认界面；
- XDG Portal：正常截图成功时返回图片字节与像素尺寸，临时文件读完就被删掉；
  用户拒绝、响应里没有图片地址、地址不是本地文件、图片文件读不出来、
  图片尺寸读不出来、等待响应超时、连不上 D-Bus 会话总线，
  一律抛 :class:`~app.core.exceptions.ScreenshotError`。

不真连 D-Bus，也不弹 Portal 确认界面：D-Bus 那几步用假总线与假消息打桩，
图片临时文件放在 pytest 的 ``tmp_path`` 里，读完即删。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.core.exceptions import ScreenshotError
from app.platform.linux import screenshot as screenshot_module
from app.platform.linux.screenshot import (
    EXTENSION_BUS_NAME,
    EXTENSION_OBJECT_PATH,
    GnomeScreenshotBackend,
)
from app.platform.types import Screenshot

try:
    from dbus_next import MessageType
except Exception:  # pragma: no cover - dbus_next 缺失时用等价的假类型
    class MessageType:  # type: ignore[no-redef]
        """dbus_next 不可导入时的占位类型。"""

        METHOD_RETURN = 2
        ERROR = 3


# ----------------------------------------------------------------------
# 假对象
# ----------------------------------------------------------------------


class _FakeMessage:
    """假的消息对象，字段与 dbus_next.Message 对齐。"""

    def __init__(
        self,
        *,
        interface: str,
        member: str,
        path: str,
        body: list[Any],
    ) -> None:
        self.interface = interface
        self.member = member
        self.path = path
        self.body = body


class _FakeReply:
    """假的调用回复，带 body 与消息类型。"""

    def __init__(
        self,
        body: list[Any],
        message_type: Any = MessageType.METHOD_RETURN,
        error_name: str = "",
    ) -> None:
        self.body = body
        self.message_type = message_type
        self.error_name = error_name


class _FakeBus:
    """假的消息总线。

    只实现截图后端用到的那几样：登记与摘掉消息处理器、断开、发送调用。
    ``messages`` 里的消息在登记处理器的当下就送出去，等价于 Portal 立刻回响应；
    ``reply`` 是调用 ``Screenshot`` 方法的回复体。
    """

    def __init__(
        self,
        messages: list[_FakeMessage] | None = None,
        reply: _FakeReply | None = None,
    ) -> None:
        self.messages = list(messages or [])
        self.reply = reply if reply is not None else _FakeReply(["/request/1"])
        self.handlers: list[Any] = []
        self.calls: list[Any] = []
        self.disconnected = False

    def add_message_handler(self, handler: Any) -> None:
        """登记消息处理器，并把预置消息送出去。"""
        self.handlers.append(handler)
        for message in self.messages:
            handler(message)

    def remove_message_handler(self, handler: Any) -> None:
        """摘掉消息处理器。"""
        if handler in self.handlers:
            self.handlers.remove(handler)

    def disconnect(self) -> None:
        """断开总线。"""
        self.disconnected = True

    async def call(self, message: Any) -> _FakeReply:
        """假的 bus.call，记录消息并回固定回复。"""
        self.calls.append(message)
        return self.reply


def _stub_open_bus(bus: Any) -> Any:
    """造一个返回指定总线的 _open_bus 替身。"""

    async def _open_bus() -> Any:
        return bus

    return _open_bus


def _stub_request_screenshot(path: str) -> Any:
    """造一个返回指定 request 路径的 _request_screenshot 替身。"""

    async def _request_screenshot(bus: Any, token: str, target: str) -> str:
        return path

    return _request_screenshot


def _stub_await_response(code: int, results: dict[str, Any]) -> Any:
    """造一个返回指定响应的 _await_response 替身。"""

    async def _await_response(
        bus: Any,
        request_path: str,
        timeout: float,
    ) -> tuple[int, dict[str, Any]]:
        return code, results

    return _await_response


def _stub_extension_online(online: bool) -> Any:
    """造一个返回指定结果的 _extension_online 替身。

    打桩打在类上，方法会带上 self。
    """

    async def _extension_online(self: Any, bus: Any) -> bool:
        return online

    return _extension_online


def _stub_name_has_owner(owned: dict[str, bool]) -> Any:
    """造一个按总线名返回查询结果的 _name_has_owner 替身。"""

    async def _name_has_owner(bus: Any, name: str, timeout: float) -> bool:
        return bool(owned.get(name, False))

    return _name_has_owner


@pytest.fixture(autouse=True)
def _extension_offline_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认把扩展当成没装，截图走 XDG Portal 那条路。

    扩展相关的测试自己把 _extension_online 打桩掉。
    """
    monkeypatch.setattr(
        GnomeScreenshotBackend,
        "_extension_online",
        _stub_extension_online(False),
    )


# ----------------------------------------------------------------------
# capture
# ----------------------------------------------------------------------


async def test_capture_returns_image_and_removes_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """截图成功时返回图片字节与像素尺寸，Portal 的临时文件被删掉。"""
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    expected = image_path.read_bytes()

    bus = _FakeBus()
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(bus))
    monkeypatch.setattr(
        backend,
        "_request_screenshot",
        _stub_request_screenshot("/request/1"),
    )
    monkeypatch.setattr(
        backend,
        "_await_response",
        _stub_await_response(0, {"uri": image_path.as_uri()}),
    )

    shot = await backend.capture()

    assert isinstance(shot, Screenshot)
    assert shot.data == expected
    assert (shot.width, shot.height) == (32, 24)
    # 临时文件读完即删
    assert not image_path.exists()
    assert bus.disconnected is True


async def test_capture_reports_user_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """响应码不是 0 说明用户拒绝，抛 ScreenshotError。"""
    bus = _FakeBus()
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(bus))
    monkeypatch.setattr(
        backend,
        "_request_screenshot",
        _stub_request_screenshot("/request/1"),
    )
    monkeypatch.setattr(
        backend,
        "_await_response",
        _stub_await_response(1, {}),
    )

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert "拒绝" in excinfo.value.message
    assert bus.disconnected is True


async def test_capture_reports_missing_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """响应里没有图片地址时抛 ScreenshotError。"""
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(_FakeBus()))
    monkeypatch.setattr(
        backend,
        "_request_screenshot",
        _stub_request_screenshot("/request/1"),
    )
    monkeypatch.setattr(
        backend,
        "_await_response",
        _stub_await_response(0, {"other": "value"}),
    )

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert "图片地址" in excinfo.value.message


async def test_capture_rejects_non_file_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """图片地址不是本地文件地址时抛 ScreenshotError。"""
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(_FakeBus()))
    monkeypatch.setattr(
        backend,
        "_request_screenshot",
        _stub_request_screenshot("/request/1"),
    )
    monkeypatch.setattr(
        backend,
        "_await_response",
        _stub_await_response(0, {"uri": "https://example.com/shot.png"}),
    )

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert "本地" in excinfo.value.message


async def test_capture_reports_unreadable_image_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """图片文件读不出来时抛 ScreenshotError。"""
    missing = tmp_path / "missing.png"
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(_FakeBus()))
    monkeypatch.setattr(
        backend,
        "_request_screenshot",
        _stub_request_screenshot("/request/1"),
    )
    monkeypatch.setattr(
        backend,
        "_await_response",
        _stub_await_response(0, {"uri": missing.as_uri()}),
    )

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert "读不到" in excinfo.value.message


async def test_capture_reports_unreadable_image_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """图片字节不是能识别的图时，尺寸读不出来，抛 ScreenshotError。"""
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(_FakeBus()))
    monkeypatch.setattr(
        backend,
        "_request_screenshot",
        _stub_request_screenshot("/request/1"),
    )
    monkeypatch.setattr(
        backend,
        "_await_response",
        _stub_await_response(0, {"uri": broken.as_uri()}),
    )

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert "读不出截图尺寸" in excinfo.value.message


async def test_capture_times_out_when_portal_never_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portal 一直没回响应时超时，抛 ScreenshotError。"""
    monkeypatch.setattr(screenshot_module, "CAPTURE_TIMEOUT", 0.01)
    bus = _FakeBus()
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(bus))
    monkeypatch.setattr(
        backend,
        "_request_screenshot",
        _stub_request_screenshot("/request/1"),
    )

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert "超时" in excinfo.value.message
    # 超时后消息处理器摘掉了，不留悬挂回调
    assert bus.handlers == []
    assert bus.disconnected is True


async def test_capture_reports_bus_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """连不上 D-Bus 会话总线时抛 ScreenshotError。"""
    backend = GnomeScreenshotBackend()

    async def _fail() -> Any:
        raise ScreenshotError("连不上 D-Bus 会话总线")

    monkeypatch.setattr(backend, "_open_bus", _fail)

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert "连不上 D-Bus 会话总线" in excinfo.value.message


async def test_capture_sends_alphanumeric_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """handle_token 只含字母数字与下划线，Portal 才认。"""
    seen: list[str] = []

    async def _request(bus: Any, token: str, target: str) -> str:
        seen.append(token)
        return "/request/1"

    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(_FakeBus()))
    monkeypatch.setattr(backend, "_request_screenshot", _request)
    monkeypatch.setattr(
        backend,
        "_await_response",
        _stub_await_response(1, {}),
    )

    with pytest.raises(ScreenshotError):
        await backend.capture()

    assert len(seen) == 1
    assert seen[0].startswith("interviewmate")
    assert seen[0].isalnum()


# ----------------------------------------------------------------------
# _await_response
# ----------------------------------------------------------------------


async def test_await_response_unwraps_variants_and_filters_messages() -> None:
    """只认本次 request 的 Response 信号，结果里的 Variant 取原值。"""
    dbus_next = pytest.importorskip("dbus_next")
    unrelated = _FakeMessage(
        interface="org.freedesktop.portal.Request",
        member="Response",
        path="/request/other",
        body=[0, {}],
    )
    related = _FakeMessage(
        interface="org.freedesktop.portal.Request",
        member="Response",
        path="/request/1",
        body=[0, {"uri": dbus_next.Variant("s", "file:///tmp/a.png")}],
    )
    bus = _FakeBus([unrelated, related])

    code, results = await GnomeScreenshotBackend()._await_response(
        bus, "/request/1", 1.0
    )

    assert code == 0
    assert results == {"uri": "file:///tmp/a.png"}
    # 拿到结果后消息处理器摘掉了
    assert bus.handlers == []


async def test_await_response_times_out_without_signal() -> None:
    """一直没等到 Response 信号时超时，并摘掉消息处理器。"""
    bus = _FakeBus()

    with pytest.raises(ScreenshotError) as excinfo:
        await GnomeScreenshotBackend()._await_response(bus, "/request/1", 0.01)

    assert "超时" in excinfo.value.message
    assert bus.handlers == []


# ----------------------------------------------------------------------
# _request_screenshot 与 _read_image_file
# ----------------------------------------------------------------------


async def test_request_screenshot_sends_expected_message() -> None:
    """调用形态与《接口契约》「XDG Portal 截图调用」一节一致。"""
    bus = _FakeBus(reply=_FakeReply(["/request/abc/interviewmate123"]))

    path = await GnomeScreenshotBackend()._request_screenshot(
        bus, "interviewmate123", "screen"
    )

    message = bus.calls[0]
    assert message.destination == "org.freedesktop.portal.Desktop"
    assert message.path == "/org/freedesktop/portal/desktop"
    assert message.interface == "org.freedesktop.portal.Screenshot"
    assert message.member == "Screenshot"
    assert message.signature == "sa{sv}"
    assert message.body[0] == ""
    options = message.body[1]
    assert options["handle_token"].value == "interviewmate123"
    assert options["modal"].value is True
    assert options["interactive"].value is False
    assert options["target"].value == 1
    assert path == "/request/abc/interviewmate123"


def test_read_image_file_rejects_non_file_uri() -> None:
    """不是本地文件地址的图片地址直接拒绝。"""
    with pytest.raises(ScreenshotError) as excinfo:
        GnomeScreenshotBackend()._read_image_file("https://example.com/shot.png")
    assert "不是本地文件" in excinfo.value.message


# ----------------------------------------------------------------------
# _open_bus
# ----------------------------------------------------------------------


async def test_open_bus_reports_missing_dbus_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dbus-next 不可导入时抛 ScreenshotError，文案与 probe 一致。"""
    monkeypatch.setitem(sys.modules, "dbus_next", None)

    with pytest.raises(ScreenshotError) as excinfo:
        await GnomeScreenshotBackend()._open_bus()

    assert "dbus-next 不可导入" in excinfo.value.message


async def test_open_bus_reports_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """连不上会话总线时抛 ScreenshotError，不真连 D-Bus。"""
    dbus_next_aio = pytest.importorskip("dbus_next.aio")

    class _FailingMessageBus:
        """假的 MessageBus，connect 一律失败。"""

        def __init__(self, *, bus_type: Any = None) -> None:
            self.bus_type = bus_type

        async def connect(self) -> Any:
            raise OSError("没有 D-Bus 会话总线")

    monkeypatch.setattr(dbus_next_aio, "MessageBus", _FailingMessageBus)

    with pytest.raises(ScreenshotError) as excinfo:
        await GnomeScreenshotBackend()._open_bus()

    assert "连不上 D-Bus 会话总线" in excinfo.value.message


async def test_await_response_handles_signal_arriving_later() -> None:
    """响应信号晚到也能等到，结果照样取得到。"""
    bus = _FakeBus()
    backend = GnomeScreenshotBackend()
    message = _FakeMessage(
        interface="org.freedesktop.portal.Request",
        member="Response",
        path="/request/1",
        body=[0, {"uri": "file:///tmp/a.png"}],
    )

    async def _emit_later() -> None:
        await asyncio.sleep(0.01)
        for handler in list(bus.handlers):
            handler(message)

    task = asyncio.create_task(_emit_later())
    code, results = await backend._await_response(bus, "/request/1", 1.0)
    await task

    assert code == 0
    assert results == {"uri": "file:///tmp/a.png"}
    assert bus.handlers == []


# ----------------------------------------------------------------------
# 静默截图扩展
# ----------------------------------------------------------------------


async def test_capture_uses_extension_when_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """扩展在线时走静默抓屏，不碰 Portal。"""
    backend = GnomeScreenshotBackend()
    bus = _FakeBus()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(bus))
    monkeypatch.setattr(
        GnomeScreenshotBackend,
        "_extension_online",
        _stub_extension_online(True),
    )

    async def _portal(bus: Any) -> Screenshot:
        raise AssertionError("扩展在线时不该走 Portal")

    expected = Screenshot(data=b"png-bytes", width=32, height=24)

    async def _extension(bus: Any, target: str, monitor: int) -> Screenshot:
        return expected

    monkeypatch.setattr(backend, "_capture_via_portal", _portal)
    monkeypatch.setattr(backend, "_capture_via_extension", _extension)

    shot = await backend.capture()

    assert shot is expected
    assert bus.disconnected is True


async def test_capture_via_extension_sends_expected_message(
    tmp_path: Path,
) -> None:
    """扩展调用形态与《接口契约》「GNOME 静默截图扩展」一节一致。"""
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    expected = image_path.read_bytes()
    bus = _FakeBus(reply=_FakeReply([expected]))

    shot = await GnomeScreenshotBackend()._capture_via_extension(bus, "screen", 0)

    message = bus.calls[0]
    assert message.destination == EXTENSION_BUS_NAME
    assert message.path == EXTENSION_OBJECT_PATH
    assert message.interface == EXTENSION_BUS_NAME
    assert message.member == "Capture"
    assert message.signature == "si"
    assert message.body == ["screen", 0]
    assert shot.data == expected
    assert (shot.width, shot.height) == (32, 24)


@pytest.mark.parametrize("target", ["screen", "primary", "monitor"])
async def test_request_screenshot_maps_target_to_portal_value(target: str) -> None:
    """Portal 那条路只有整屏。

    Screenshot portal 的目标只有 Screen、Window、Area、Active Window，
    没有 Monitor，所以不管配置成什么，落到的都是整屏目标值。
    """
    bus = _FakeBus(reply=_FakeReply(["/request/abc/interviewmate123"]))

    await GnomeScreenshotBackend()._request_screenshot(
        bus, "interviewmate123", target
    )

    options = bus.calls[0].body[1]
    assert options["target"].value == 1


async def test_capture_via_extension_passes_target_and_monitor(
    tmp_path: Path,
) -> None:
    """扩展调用把截图目标与显示器序号一并带过去。"""
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    bus = _FakeBus(reply=_FakeReply([image_path.read_bytes()]))

    await GnomeScreenshotBackend()._capture_via_extension(bus, "monitor", 2)

    message = bus.calls[0]
    assert message.signature == "si"
    assert message.body == ["monitor", 2]


async def test_capture_via_extension_reports_call_failure() -> None:
    """扩展调不通时抛 ScreenshotError。"""

    class _FailingBus(_FakeBus):
        """假的总线，调用一律失败。"""

        async def call(self, message: Any) -> _FakeReply:
            raise OSError("服务不存在")

    with pytest.raises(ScreenshotError) as excinfo:
        await GnomeScreenshotBackend()._capture_via_extension(_FailingBus(), "screen", 0)

    assert "扩展" in excinfo.value.message


async def test_capture_via_extension_rejects_empty_bytes() -> None:
    """扩展返回空图片时抛 ScreenshotError。"""
    bus = _FakeBus(reply=_FakeReply([b""]))

    with pytest.raises(ScreenshotError) as excinfo:
        await GnomeScreenshotBackend()._capture_via_extension(bus, "screen", 0)

    assert "空图片" in excinfo.value.message


async def test_capture_via_extension_reports_dbus_error_reply() -> None:
    """扩展内部抛错时，D-Bus 回的是错误回复，要把真实错误报出来。

    扩展里出问题时，回复体第一个元素是错误消息字符串，不是图片字节。
    不分开看的话，这个字符串会被当成图片字节去解码，报出来的错会很难懂。
    """
    bus = _FakeBus(
        reply=_FakeReply(
            ["content.get_texture is not a function"],
            message_type=MessageType.ERROR,
            error_name="org.gnome.gjs.JSError.TypeError",
        )
    )

    with pytest.raises(ScreenshotError) as excinfo:
        await GnomeScreenshotBackend()._capture_via_extension(bus, "screen", 0)

    message = excinfo.value.message
    assert "org.gnome.gjs.JSError.TypeError" in message
    assert "content.get_texture is not a function" in message


async def test_capture_via_extension_rejects_non_bytes_payload() -> None:
    """扩展返回的不是字节时报错，不要当成图片继续往下走。"""
    bus = _FakeBus(reply=_FakeReply(["不是图片"]))

    with pytest.raises(ScreenshotError) as excinfo:
        await GnomeScreenshotBackend()._capture_via_extension(bus, "screen", 0)

    assert "不是图片字节" in excinfo.value.message


async def test_capture_falls_back_to_portal_when_extension_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """扩展调用失败时回退 XDG Portal，截图照样能出。"""
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (32, 24), "white").save(image_path)

    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(_FakeBus()))
    monkeypatch.setattr(
        GnomeScreenshotBackend,
        "_extension_online",
        _stub_extension_online(True),
    )

    async def _extension(bus: Any, target: str, monitor: int) -> Screenshot:
        raise ScreenshotError("扩展挂了")

    monkeypatch.setattr(backend, "_capture_via_extension", _extension)
    monkeypatch.setattr(
        backend,
        "_request_screenshot",
        _stub_request_screenshot("/request/1"),
    )
    monkeypatch.setattr(
        backend,
        "_await_response",
        _stub_await_response(0, {"uri": image_path.as_uri()}),
    )

    shot = await backend.capture()

    assert (shot.width, shot.height) == (32, 24)
    assert not image_path.exists()


async def test_extension_online_checks_name_has_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查扩展在不在，走的是 D-Bus 的 NameHasOwner。"""
    # 撤掉默认打桩，这里要测真正的判断逻辑
    monkeypatch.undo()
    bus = _FakeBus(reply=_FakeReply([True]))

    online = await GnomeScreenshotBackend()._extension_online(bus)

    assert online is True
    message = bus.calls[0]
    assert message.member == "NameHasOwner"
    assert message.body == [EXTENSION_BUS_NAME]


async def test_extension_online_is_false_when_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查询本身出错时当成扩展不在线，不影响截图。"""
    # 撤掉默认打桩，这里要测真正的判断逻辑
    monkeypatch.undo()
    backend = GnomeScreenshotBackend()

    async def _fail(bus: Any, name: str, timeout: float) -> bool:
        raise ScreenshotError("查询失败")

    monkeypatch.setattr(backend, "_name_has_owner", _fail)

    assert await backend._extension_online(_FakeBus()) is False


# ----------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------


async def test_probe_reports_extension_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """扩展在线时说明截图不弹提示。"""
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(_FakeBus()))
    monkeypatch.setattr(
        backend,
        "_name_has_owner",
        _stub_name_has_owner({EXTENSION_BUS_NAME: True}),
    )

    detail = await backend.probe(timeout=0.1)

    assert "不弹" in detail


async def test_probe_reports_portal_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """扩展没装但 Portal 在时，说明会弹确认框。"""
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(_FakeBus()))
    monkeypatch.setattr(
        backend,
        "_name_has_owner",
        _stub_name_has_owner({"org.freedesktop.portal.Desktop": True}),
    )

    detail = await backend.probe(timeout=0.1)

    assert "确认框" in detail


async def test_probe_reports_no_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既没扩展也没 Portal 时抛 ScreenshotError。"""
    backend = GnomeScreenshotBackend()
    monkeypatch.setattr(backend, "_open_bus", _stub_open_bus(_FakeBus()))
    monkeypatch.setattr(
        backend,
        "_name_has_owner",
        _stub_name_has_owner({}),
    )

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.probe(timeout=0.1)

    assert "既没有" in excinfo.value.message
