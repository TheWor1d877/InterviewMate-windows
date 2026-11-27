"""Linux 侧的截图后端：GNOME Shell 扩展静默抓屏，XDG Portal 兜底。

见《9-平台差异与Windows适配》「Linux 截图后端」一节：Linux 侧走 Wayland，
截图有两条路。

第一条是本仓库自带的 GNOME Shell 扩展 ``interviewmate-screenshot@local``，
装在 ``extensions/interviewmate-screenshot/`` 下。它在 GNOME 进程内调
``Shell.Screenshot`` 抓整屏，**屏幕上不弹任何提示**，图片字节通过 D-Bus 送回，
全程不落盘。调用形态见《接口契约》「GNOME 静默截图扩展」一节。

第二条是 XDG Portal 的 ``org.freedesktop.portal.Screenshot``，调用形态见
《接口契约》「XDG Portal 截图调用」一节：GNOME 下一定会弹一次截图确认界面，
用户点一次确认才返回图片。扩展没装时回退走这条，程序照样能用。

两条路的图片都只在内存里：扩展把 PNG 字节直接写进内存流，Portal 那边读完就删
它自己建的临时文件。

本模块把 :func:`app.main.probe_screenshot_interface` 的 D-Bus 探测逻辑搬过来，
错误文案照抄原文，下游既有测试依赖这些文案。

真实抓屏那一路限时 ``screenshot.timeout`` 秒，超时报 ``E_SCREENSHOT``；
接口探测那一路（``probe``）有自己的超时参数，语义不变，
见《8-集成与验收》「超时管理」一节。
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from app.config import current_config
from app.core.exceptions import ScreenshotError
from app.platform.types import Screenshot, ScreenshotTarget
from app.utils.logger import log_event

__all__ = [
    "GnomeScreenshotBackend",
    "PORTAL_BUS_NAME",
    "EXTENSION_BUS_NAME",
    "EXTENSION_OBJECT_PATH",
    "CAPTURE_TIMEOUT",
    "DEFAULT_CAPTURE_TIMEOUT",
]


# XDG Portal 的总线名，截图接口挂在它上面
PORTAL_BUS_NAME: str = "org.freedesktop.portal.Desktop"

# 静默截图扩展占的总线名、对象路径与接口名
EXTENSION_BUS_NAME: str = "com.interviewmate.Screenshot"
EXTENSION_OBJECT_PATH: str = "/com/interviewmate/Screenshot"
EXTENSION_INTERFACE: str = "com.interviewmate.Screenshot"

# 等 Portal 回传响应的秒数上限。
# GNOME 下一定会弹截图确认界面，用户点一次确认才返回图片，这段等待要留够。
CAPTURE_TIMEOUT: float = 120.0

# 真实抓屏那一路的默认超时秒数：进程里没加载过配置时用它，
# 见《8-集成与验收》「超时管理」一节
DEFAULT_CAPTURE_TIMEOUT: float = 150.0

# Portal 的对象路径，截图接口挂在它上面
_PORTAL_OBJECT_PATH: str = "/org/freedesktop/portal/desktop"

# 截图接口名与方法名
_SCREENSHOT_INTERFACE: str = "org.freedesktop.portal.Screenshot"
_SCREENSHOT_MEMBER: str = "Screenshot"

# 响应信号所在的接口与信号名，Portal 用它回传截图结果
_REQUEST_INTERFACE: str = "org.freedesktop.portal.Request"
_RESPONSE_MEMBER: str = "Response"

# 整屏目标值，见《接口契约》「XDG Portal 截图目标」一节
_SCREEN_TARGET: int = 1

# 扩展里抓图的方法名
_EXTENSION_CAPTURE_MEMBER: str = "Capture"


def _portal_target_value(target: ScreenshotTarget) -> int:
    """把截图目标翻译成 XDG Portal 的目标值。

    Screenshot portal 只有 Screen、Window、Area、Active Window 四个目标，
    没有 Monitor（Monitor 是 ScreenCast portal 的目标），所以主屏幕与指定屏
    都只能落到整屏上，由调用方决定要不要告警。

    :param target: 截图目标。
    :returns: Portal 的目标值。
    """
    return _SCREEN_TARGET


class GnomeScreenshotBackend:
    """GNOME 截图后端。

    优先走 GNOME Shell 扩展静默抓屏，扩展不在线或调用失败时回退 XDG Portal。
    :meth:`probe` 只查接口在不在，不真截图，所以不会在用户屏幕上弹对话框。

    :param timeout: 真实抓屏那一路的超时秒数，None 时取进程里加载好的配置里的
        ``screenshot.timeout``，见《8-集成与验收》「超时管理」一节。
    """

    # 后端名，对应配置里的 ``providers.screenshot`` 取值
    name: str = "gnome"

    # 该能力需要的第三方包名：Wayland 截屏要 dbus-next
    required_packages: tuple[str, ...] = ("dbus_next",)

    def __init__(self, timeout: float | None = None) -> None:
        self._timeout = timeout

    @property
    def timeout(self) -> float:
        """真实抓屏那一路的超时秒数。

        构造时没给就从进程里加载好的配置里取 ``screenshot.timeout``，
        配置还没加载时用 :data:`DEFAULT_CAPTURE_TIMEOUT`，
        见《8-集成与验收》「超时管理」一节。
        """
        if self._timeout is not None:
            return float(self._timeout)
        config = current_config()
        if config is not None:
            return float(config.screenshot.timeout)
        return DEFAULT_CAPTURE_TIMEOUT

    async def probe(self, timeout: float = 3.0) -> str:
        """检查截图接口能不能用。

        先看静默截图扩展在不在，扩展在就是静默截图；不在就看 Portal 在不在，
        Portal 在也能截图，只是屏幕上会弹一次确认框。

        :param timeout: 超时秒数。
        :returns: 一句中文说明。
        :raises ScreenshotError: dbus-next 不可导入、连不上会话总线、
            查询服务名失败，且 D-Bus 上既没有扩展也没有 portal 服务。
        """
        try:
            bus = await asyncio.wait_for(self._open_bus(), timeout)
        except asyncio.TimeoutError as exc:
            raise ScreenshotError(
                f"连不上 D-Bus 会话总线：超时 {timeout:g} 秒"
            ) from exc
        try:
            extension = await self._name_has_owner(bus, EXTENSION_BUS_NAME, timeout)
            portal = await self._name_has_owner(bus, PORTAL_BUS_NAME, timeout)
        finally:
            bus.disconnect()
        if extension:
            return (
                f"{EXTENSION_BUS_NAME} 在线，截图不弹任何提示"
            )
        if portal:
            return (
                f"静默截图扩展没装，回退 {PORTAL_BUS_NAME}，"
                "截图时 GNOME 会弹一次确认框"
            )
        raise ScreenshotError(
            f"D-Bus 上既没有 {EXTENSION_BUS_NAME} 也没有 {PORTAL_BUS_NAME} 服务"
        )

    async def capture(
        self,
        *,
        target: ScreenshotTarget = "screen",
        monitor: int = 0,
    ) -> Screenshot:
        """做一次截图。

        扩展在线时走静默抓屏，屏幕上不出现任何提示；否则回退 XDG Portal，
        流程见《4-截图输入与多模态》「XDG Portal 调用」一节。

        整个抓屏过程限时 ``screenshot.timeout`` 秒，超时按截图失败处理，
        见《8-集成与验收》「超时管理」一节。接口探测是另一路，
        它的超时参数由 :meth:`probe` 自己定，两边互不影响。

        :param target: 截图目标，``screen`` 是整屏，``primary`` 是主屏幕，
            ``monitor`` 是指定的那块屏。
        :param monitor: 显示器序号，0 起数，只在 ``target`` 是 ``monitor`` 时有意义。
        :returns: 截图结果，字段见 :class:`~app.platform.types.Screenshot`。
        :raises ScreenshotError: 连不上 D-Bus、扩展调用失败且 Portal 也调不通、
            用户拒绝、等待响应超时、返回的图片地址不合法，或图片读不出来；
            整个抓屏超过 ``screenshot.timeout`` 秒也按这个错处理。
        """
        timeout = self.timeout
        try:
            return await asyncio.wait_for(
                self._capture_within_timeout(target, monitor), timeout
            )
        except asyncio.TimeoutError as exc:
            message = f"截图接口超时（{timeout:g} 秒）"
            log_event("screenshot.error", message, level="ERROR", code=ScreenshotError.code)
            raise ScreenshotError(message) from exc

    async def _capture_within_timeout(
        self, target: ScreenshotTarget, monitor: int
    ) -> Screenshot:
        """抓一张图。

        :param target: 截图目标。
        :param monitor: 显示器序号，0 起数。
        :returns: 截图结果，字段见 :class:`~app.platform.types.Screenshot`。
        :raises ScreenshotError: 抓屏过程里的失败。
        """
        bus: Any = None
        try:
            bus = await self._open_bus()
            if await self._extension_online(bus):
                try:
                    return await self._capture_via_extension(bus, target, monitor)
                except ScreenshotError as exc:
                    log_event(
                        "screenshot.warn",
                        f"静默截图扩展失败，改用 XDG Portal：{exc.message}",
                        level="WARNING",
                    )
            return await self._capture_via_portal(bus, target)
        except ScreenshotError as exc:
            log_event("screenshot.error", exc.message, level="ERROR", code=exc.code)
            raise
        finally:
            if bus is not None:
                try:
                    bus.disconnect()
                except Exception:
                    log_event(
                        "screenshot.warn",
                        "断开 D-Bus 会话总线失败",
                        level="WARNING",
                    )

    # ------------------------------------------------------------------
    # 静默截图扩展
    # ------------------------------------------------------------------
    async def _capture_via_extension(
        self, bus: Any, target: ScreenshotTarget, monitor: int
    ) -> Screenshot:
        """通过 GNOME Shell 扩展静默抓一张图。

        :param bus: 已连接的 D-Bus 会话总线。
        :param target: 截图目标，扩展按它决定抓整屏还是某一块屏。
        :param monitor: 显示器序号，0 起数，只在 ``target`` 是 ``monitor`` 时有意义。
        :returns: 截图结果，字段见 :class:`~app.platform.types.Screenshot`。
        :raises ScreenshotError: 调用扩展失败、扩展没返回图片，
            或图片读不出尺寸。
        """
        try:
            from dbus_next import Message, MessageType
        except Exception as exc:
            raise ScreenshotError(f"dbus-next 不可导入：{exc}") from exc
        try:
            reply = await bus.call(
                Message(
                    destination=EXTENSION_BUS_NAME,
                    path=EXTENSION_OBJECT_PATH,
                    interface=EXTENSION_INTERFACE,
                    member=_EXTENSION_CAPTURE_MEMBER,
                    signature="si",
                    body=[target, monitor],
                )
            )
        except Exception as exc:
            raise ScreenshotError(f"调用静默截图扩展失败：{exc}") from exc
        # 扩展内部抛错时，D-Bus 回的是错误回复，不是图片字节，这里要分开看
        if reply.message_type == MessageType.ERROR:
            detail = str(reply.body[0]) if reply.body else ""
            raise ScreenshotError(
                f"静默截图扩展报错：{reply.error_name}"
                + (f"：{detail}" if detail else "")
            )
        if not reply.body:
            raise ScreenshotError("静默截图扩展没有返回图片")
        payload = reply.body[0]
        if not isinstance(payload, (bytes, bytearray)):
            raise ScreenshotError(
                "静默截图扩展返回的不是图片字节："
                f"{type(payload).__name__}"
            )
        data = bytes(payload)
        if not data:
            raise ScreenshotError("静默截图扩展返回的是空图片")
        width, height = self._image_size(data)
        log_event(
            "screenshot.captured",
            "GNOME 扩展静默截图完成",
            width=width,
            height=height,
            size=len(data),
        )
        return Screenshot(data=data, width=width, height=height)

    async def _extension_online(self, bus: Any) -> bool:
        """静默截图扩展是不是已经装好并启用了。

        :param bus: 已连接的 D-Bus 会话总线。
        :returns: 扩展的总线名有人持有返回 True。
        """
        try:
            return await self._name_has_owner(bus, EXTENSION_BUS_NAME, 3.0)
        except ScreenshotError:
            return False

    # ------------------------------------------------------------------
    # XDG Portal
    # ------------------------------------------------------------------
    async def _capture_via_portal(self, bus: Any, target: ScreenshotTarget) -> Screenshot:
        """通过 XDG Portal 抓一张图，GNOME 下会弹一次确认界面。

        :param bus: 已连接的 D-Bus 会话总线。
        :param target: 截图目标。Screenshot portal 没有 Monitor 目标，
            主屏幕与指定屏在这条路上只能抓整屏，会记一条告警。
        :returns: 截图结果，字段见 :class:`~app.platform.types.Screenshot`。
        :raises ScreenshotError: Portal 调用失败、用户拒绝、等待响应超时、
            返回的图片地址不合法，或图片读不出来。
        """
        if target != "screen":
            log_event(
                "screenshot.warn",
                "XDG Portal 没有 Monitor 目标，这一条路只能抓整屏，"
                f"配置里的 target={target} 不生效",
                level="WARNING",
            )
        # handle_token 只允许字母数字与下划线，用 uuid 的十六进制串保证唯一
        token = "interviewmate" + uuid4().hex
        request_path = await self._request_screenshot(bus, token, target)
        code, results = await self._await_response(
            bus, request_path, CAPTURE_TIMEOUT
        )
        if code != 0:
            raise ScreenshotError(
                f"用户拒绝了截图请求或 Portal 截图失败，响应码 {code}"
            )
        raw_uri = results.get("uri")
        if raw_uri is None or raw_uri == "":
            raise ScreenshotError("Portal 响应里没有图片地址")
        uri = str(getattr(raw_uri, "value", raw_uri))
        if not uri.startswith("file://"):
            raise ScreenshotError(f"Portal 返回的不是本地图片地址：{uri}")
        data = self._read_image_file(uri)
        width, height = self._image_size(data)
        log_event(
            "screenshot.captured",
            "XDG Portal 截图完成",
            width=width,
            height=height,
            size=len(data),
        )
        return Screenshot(data=data, width=width, height=height)

    async def _open_bus(self) -> Any:
        """连上 D-Bus 会话总线。

        :returns: 已连接的 dbus-next 会话总线对象。
        :raises ScreenshotError: dbus-next 不可导入，或连不上会话总线。
        """
        try:
            from dbus_next import BusType
            from dbus_next.aio import MessageBus
        except Exception as exc:
            raise ScreenshotError(f"dbus-next 不可导入：{exc}") from exc
        try:
            return await MessageBus(bus_type=BusType.SESSION).connect()
        except Exception as exc:
            raise ScreenshotError(f"连不上 D-Bus 会话总线：{exc}") from exc

    async def _name_has_owner(
        self, bus: Any, name: str, timeout: float
    ) -> bool:
        """查一个总线名有没有人持有。

        :param bus: 已连接的 D-Bus 会话总线。
        :param name: 总线名。
        :param timeout: 超时秒数。
        :returns: 有人持有返回 True。
        :raises ScreenshotError: 查询失败。
        """
        try:
            from dbus_next import Message
        except Exception as exc:
            raise ScreenshotError(f"dbus-next 不可导入：{exc}") from exc
        try:
            reply = await asyncio.wait_for(
                bus.call(
                    Message(
                        destination="org.freedesktop.DBus",
                        path="/org/freedesktop/DBus",
                        interface="org.freedesktop.DBus",
                        member="NameHasOwner",
                        signature="s",
                        body=[name],
                    )
                ),
                timeout,
            )
        except Exception as exc:
            raise ScreenshotError(f"查询 {name} 失败：{exc}") from exc
        return bool(reply.body[0]) if reply.body else False

    async def _request_screenshot(
        self, bus: Any, token: str, target: ScreenshotTarget
    ) -> str:
        """发起 Screenshot 调用，返回 org.freedesktop.portal.Request 的对象路径。

        :param bus: 已连接的 D-Bus 会话总线。
        :param token: 本次请求的令牌，只含字母数字与下划线。
        :param target: 截图目标，决定传给 Portal 的目标值。
        :returns: request 的对象路径。
        :raises ScreenshotError: dbus-next 不可导入、调用失败，
            或返回体里没有对象路径。
        """
        try:
            from dbus_next import Message, Variant
        except Exception as exc:
            raise ScreenshotError(f"dbus-next 不可导入：{exc}") from exc
        options = {
            "handle_token": Variant("s", token),
            "modal": Variant("b", True),
            "interactive": Variant("b", False),
            "target": Variant("u", _portal_target_value(target)),
        }
        try:
            reply = await bus.call(
                Message(
                    destination=PORTAL_BUS_NAME,
                    path=_PORTAL_OBJECT_PATH,
                    interface=_SCREENSHOT_INTERFACE,
                    member=_SCREENSHOT_MEMBER,
                    signature="sa{sv}",
                    body=["", options],
                )
            )
        except Exception as exc:
            raise ScreenshotError(f"调用 XDG Portal 截图接口失败：{exc}") from exc
        if not reply.body or not reply.body[0]:
            raise ScreenshotError("XDG Portal 截图接口没有返回 request 路径")
        return str(reply.body[0])

    async def _await_response(
        self, bus: Any, request_path: str, timeout: float
    ) -> tuple[int, dict[str, Any]]:
        """等 Response 信号，返回 (response_code, results)。

        :param bus: 已连接的 D-Bus 会话总线。
        :param request_path: request 的对象路径。
        :param timeout: 等待秒数。
        :returns: 二元组，响应码与结果字典；结果里的值已从 Variant 里取出来。
        :raises ScreenshotError: 等待超时，或信号内容里没有响应码。
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[int, dict[str, Any]]] = loop.create_future()
        failure: str | None = None

        def handler(msg: Any) -> None:
            """筛出本次 request 的 Response 信号，把结果塞进 future。"""
            nonlocal failure
            if future.done():
                return
            if (
                getattr(msg, "interface", None) != _REQUEST_INTERFACE
                or getattr(msg, "member", None) != _RESPONSE_MEMBER
                or getattr(msg, "path", None) != request_path
            ):
                return
            body = list(getattr(msg, "body", None) or [])
            try:
                code = int(body[0])
                raw = body[1] if len(body) > 1 else {}
                results = {
                    str(key): getattr(value, "value", value)
                    for key, value in dict(raw).items()
                }
            except Exception as exc:
                failure = f"Portal 的响应信号内容不对：{exc}"
                future.set_result((0, {}))
                return
            future.set_result((code, results))

        bus.add_message_handler(handler)
        try:
            code, results = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError as exc:
            raise ScreenshotError(
                f"等待 Portal 截图响应超时（{timeout:g} 秒）"
            ) from exc
        finally:
            # 摘掉消息处理器，别留下悬挂回调
            bus.remove_message_handler(handler)
        if failure is not None:
            raise ScreenshotError(failure)
        return code, results

    def _read_image_file(self, uri: str) -> bytes:
        """按 file:// URI 读图片字节，读完立即删掉临时文件。

        :param uri: Portal 返回的 file:// 地址。
        :returns: 图片字节。
        :raises ScreenshotError: 地址不是本地文件，或文件读不出来。
        """
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ScreenshotError(f"图片地址不是本地文件：{uri}")
        path_text = unquote(parsed.path)
        if not path_text:
            raise ScreenshotError(f"图片地址里没有路径：{uri}")
        path = Path(path_text)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ScreenshotError(f"读不到 Portal 返回的图片文件：{exc}") from exc
        finally:
            # 临时文件读完立刻删，删不掉不影响截图结果
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log_event(
                    "screenshot.warn",
                    "删除 Portal 临时图片失败",
                    level="WARNING",
                )

    def _image_size(self, data: bytes) -> tuple[int, int]:
        """读图片字节的像素尺寸。

        只读图片头，不解码像素。

        :param data: 图片字节。
        :returns: (宽, 高)。
        :raises ScreenshotError: 图片读不出来。
        """
        try:
            from PIL import Image

            with Image.open(BytesIO(data)) as image:
                width, height = image.size
        except Exception as exc:
            raise ScreenshotError(f"读不出截图尺寸：{exc}") from exc
        return width, height
