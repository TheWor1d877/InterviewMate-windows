"""Windows 侧的截图后端：``mss``。

Windows 侧用 ``mss`` 抓屏：纯 Python，没有系统级依赖，不弹确认框，也不落盘。
抓到的位图在内存里转成 PNG 字节交给上层，全程不写文件。

三种截图目标的取法：

- ``screen``：取 ``mss`` 的 ``monitors[0]``，也就是整个虚拟桌面。多显示器时
  按各屏在桌面坐标系里的布局拼成一张，非主屏落在负坐标也能覆盖到；
- ``primary``：按主屏的宽高在显示器表里找那一块；取不到主屏尺寸时退回
  ``monitors[1]``；
- ``monitor``：取 ``monitors[monitor + 1]``，序号 0 起数，越界报错。

所有失败都转成 :class:`~app.core.exceptions.ScreenshotError`（``E_SCREENSHOT``）。
抓屏是阻塞调用，放到线程里跑，上层用 ``asyncio.wait_for`` 限时仍然有效。
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any

from app.core.exceptions import ScreenshotError
from app.platform.types import Screenshot, ScreenshotTarget
from app.utils.logger import log_event

__all__ = ["MssScreenshotBackend"]


# 显示器表的第 0 项是整个虚拟桌面，真正的显示器从第 1 项起
_VIRTUAL_DESKTOP_INDEX: int = 0
_FIRST_MONITOR_INDEX: int = 1


class MssScreenshotBackend:
    """基于 ``mss`` 的截图后端。"""

    # 后端名，对应配置里的 ``providers.screenshot`` 取值
    name: str = "mss"

    # 该能力需要的第三方包名
    required_packages: tuple[str, ...] = ("mss",)

    async def probe(self, timeout: float = 3.0) -> str:
        """检查截图接口能不能用。

        只导入 ``mss`` 并取一次显示器列表，不抓屏，所以不会在用户屏幕上
        留下任何痕迹。

        :param timeout: 超时秒数。
        :returns: 一句中文说明。
        :raises ScreenshotError: mss 不可导入、取不到显示器列表，或超时。
        """
        try:
            monitors = await asyncio.wait_for(
                asyncio.to_thread(self._list_monitors), timeout
            )
        except asyncio.TimeoutError as exc:
            raise ScreenshotError(f"截图接口探测超时（{timeout:g} 秒）") from exc
        except ScreenshotError as exc:
            log_event("screenshot.error", exc.message, level="ERROR", code=exc.code)
            raise
        desktop = monitors[_VIRTUAL_DESKTOP_INDEX]
        width = int(desktop.get("width", 0))
        height = int(desktop.get("height", 0))
        if width <= 0 or height <= 0:
            raise ScreenshotError(f"mss 返回的虚拟桌面尺寸无效：{width}x{height}")
        count = len(monitors) - 1
        return (
            f"mss 可用，检测到 {count} 块显示器，"
            f"虚拟桌面 {width}x{height}"
        )

    async def capture(
        self,
        *,
        target: ScreenshotTarget = "screen",
        monitor: int = 0,
    ) -> Screenshot:
        """做一次截图。

        抓屏、转 PNG 都在内存里完成，不落盘。

        :param target: 截图目标，``screen`` 是整屏（多屏拼成一张），
            ``primary`` 是主屏幕，``monitor`` 是指定的那块屏。
        :param monitor: 显示器序号，0 起数，只在 ``target`` 是 ``monitor`` 时有意义。
        :returns: 截图结果，字段见 :class:`~app.platform.types.Screenshot`。
        :raises ScreenshotError: mss 不可导入、显示器序号越界、抓屏失败，
            或位图编码失败。
        """
        try:
            return await asyncio.to_thread(self._capture_sync, target, monitor)
        except ScreenshotError as exc:
            log_event("screenshot.error", exc.message, level="ERROR", code=exc.code)
            raise

    # ------------------------------------------------------------------
    # 同步实现，跑在线程里
    # ------------------------------------------------------------------
    def _capture_sync(self, target: ScreenshotTarget, monitor: int) -> Screenshot:
        """抓一张图并转成 PNG 字节。"""
        mss = self._import_mss()
        try:
            sct = mss.mss()
        except Exception as exc:
            raise ScreenshotError(f"mss 初始化失败：{exc}") from exc
        try:
            monitors = list(sct.monitors)
            box = self._select_monitor(monitors, target, monitor)
            try:
                shot = sct.grab(box)
            except Exception as exc:
                raise ScreenshotError(f"mss 抓屏失败：{exc}") from exc
            data = self._encode_png(shot)
        finally:
            close = getattr(sct, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    log_event(
                        "screenshot.warn",
                        "关闭 mss 抓屏句柄失败",
                        level="WARNING",
                    )
        width, height = self._size_of(shot)
        log_event(
            "screenshot.captured",
            "mss 截图完成",
            target=target,
            monitor=monitor,
            width=width,
            height=height,
            size=len(data),
        )
        return Screenshot(data=data, width=width, height=height)

    def _list_monitors(self) -> list[dict[str, Any]]:
        """取显示器列表，失败转成截图错误。"""
        mss = self._import_mss()
        try:
            sct = mss.mss()
        except Exception as exc:
            raise ScreenshotError(f"mss 初始化失败：{exc}") from exc
        try:
            monitors = list(sct.monitors)
        except Exception as exc:
            raise ScreenshotError(f"取不到显示器列表：{exc}") from exc
        finally:
            close = getattr(sct, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    log_event(
                        "screenshot.warn",
                        "关闭 mss 抓屏句柄失败",
                        level="WARNING",
                    )
        if len(monitors) <= _FIRST_MONITOR_INDEX:
            raise ScreenshotError("mss 没有检测到任何显示器")
        return monitors

    def _import_mss(self) -> Any:
        """导入 ``mss``，不可用时转成截图错误。"""
        try:
            import mss
        except Exception as exc:
            raise ScreenshotError(f"mss 不可导入：{exc}") from exc
        return mss

    def _select_monitor(
        self,
        monitors: list[dict[str, Any]],
        target: ScreenshotTarget,
        monitor: int,
    ) -> dict[str, Any]:
        """按截图目标挑出要抓的矩形。"""
        if len(monitors) <= _FIRST_MONITOR_INDEX:
            raise ScreenshotError("mss 没有检测到任何显示器")
        if target == "screen":
            return monitors[_VIRTUAL_DESKTOP_INDEX]
        if target == "primary":
            return self._primary_monitor(monitors)
        if target == "monitor":
            if monitor < 0:
                raise ScreenshotError(f"显示器序号不能为负数：{monitor}")
            index = monitor + _FIRST_MONITOR_INDEX
            if index >= len(monitors):
                raise ScreenshotError(
                    f"显示器序号越界：{monitor}，本机只有 {len(monitors) - 1} 块显示器"
                )
            return monitors[index]
        raise ScreenshotError(f"不支持的截图目标：{target}")

    def _primary_monitor(self, monitors: list[dict[str, Any]]) -> dict[str, Any]:
        """找出主屏那块矩形。

        新版 mss 直接在显示器表里给出 ``is_primary``，先认这个标记；
        没有这个字段时用 ``GetSystemMetrics`` 取主屏宽高后按尺寸匹配；
        DPI 缩放导致尺寸对不上或匹配不唯一时，退回 ``monitors[1]``，
        保证总能抓到一块屏。
        """
        for item in monitors[_FIRST_MONITOR_INDEX:]:
            if item.get("is_primary"):
                return item
        size = self._primary_size()
        if size is not None:
            width, height = size
            matched = [
                item
                for item in monitors[_FIRST_MONITOR_INDEX:]
                if item.get("width") == width and item.get("height") == height
            ]
            if len(matched) == 1:
                return matched[0]
        return monitors[_FIRST_MONITOR_INDEX]

    def _primary_size(self) -> tuple[int, int] | None:
        """读主屏宽高，读不到返回 None。"""
        try:
            import ctypes

            user32 = ctypes.windll.user32
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        return width, height

    def _encode_png(self, shot: Any) -> bytes:
        """把 mss 的内存位图编码成 PNG 字节。"""
        width, height = self._size_of(shot)
        raw = getattr(shot, "bgra", None)
        if raw is None:
            raw = getattr(shot, "rgb", None)
            raw_mode = "RGB"
        else:
            raw_mode = "BGRX"
        if not raw:
            raise ScreenshotError("mss 返回的位图没有像素数据")
        try:
            from PIL import Image

            image = Image.frombytes(
                "RGB", (width, height), bytes(raw), "raw", raw_mode
            )
            buffer = BytesIO()
            image.save(buffer, format="PNG")
        except ScreenshotError:
            raise
        except Exception as exc:
            raise ScreenshotError(f"位图编码成 PNG 失败：{exc}") from exc
        return buffer.getvalue()

    def _size_of(self, shot: Any) -> tuple[int, int]:
        """取位图的像素尺寸。"""
        size = getattr(shot, "size", None)
        try:
            width, height = int(size[0]), int(size[1])
        except Exception as exc:
            raise ScreenshotError(f"mss 返回的位图尺寸无效：{size!r}") from exc
        if width <= 0 or height <= 0:
            raise ScreenshotError(f"mss 返回的位图尺寸无效：{size!r}")
        return width, height