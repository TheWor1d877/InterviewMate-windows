"""Windows 侧 mss 截图后端的测试。

覆盖《4-截图输入与多模态》「mss 采集」一节：

- ``probe`` 能导入 mss 并取到显示器列表即视为可用，不真截图；
- ``capture`` 把内存位图转成 PNG 字节，``screen`` 抓虚拟桌面（多屏拼成一张），
  ``primary`` 抓主屏，``monitor`` 抓指定屏；显示器序号越界、位图不合法、
  抓屏失败与 mss 不可导入一律转成 ``E_SCREENSHOT``。

测试不抓真实屏幕：``mss`` 用假模块替身，假模块返回假位图；图片内容只在
内存里，不落盘。
"""

from __future__ import annotations

import sys
import time
from io import BytesIO
from typing import Any

import pytest
from PIL import Image

from app.core.exceptions import ScreenshotError
from app.platform.windows.screenshot import MssScreenshotBackend
from app.platform.types import Screenshot

# 一块 1920x1080 的主屏与一块 1280x1024 的副屏，整屏由两块拼成
_DESKTOP = {"left": -1280, "top": 0, "width": 3200, "height": 1080}
_PRIMARY = {"left": 0, "top": 0, "width": 1920, "height": 1080}
_SECONDARY = {"left": -1280, "top": 0, "width": 1280, "height": 1024}


# ----------------------------------------------------------------------
# 假对象
# ----------------------------------------------------------------------


class _FakeShot:
    """假的 mss 位图，字段与 ``mss.screenshot.ScreenShot`` 对齐。"""

    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        # 每个像素 4 字节 BGRX，蓝色
        self.bgra = bytes([255, 0, 0, 0]) * (width * height)


class _BrokenShot:
    """尺寸不合法的位图。"""

    def __init__(self, size: Any) -> None:
        self.size = size
        self.bgra = b""


class _FakeSct:
    """假的 mss 上下文对象，只实现截图用到的那几样。"""

    def __init__(
        self,
        monitors: list[dict[str, Any]],
        *,
        error: Exception | None = None,
        shot: Any | None = None,
    ) -> None:
        self.monitors = monitors
        self.error = error
        self.shot = shot
        self.grabbed: list[dict[str, Any]] = []
        self.closed = False

    def grab(self, box: dict[str, Any]) -> Any:
        self.grabbed.append(dict(box))
        if self.error is not None:
            raise self.error
        if self.shot is not None:
            return self.shot
        return _FakeShot(int(box["width"]), int(box["height"]))

    def close(self) -> None:
        self.closed = True


class _FakeMss:
    """假的 mss 模块。"""

    def __init__(self, sct: _FakeSct) -> None:
        self.sct = sct

    def mss(self) -> _FakeSct:
        return self.sct

    def MSS(self) -> _FakeSct:
        return self.sct


def _use_fake_mss(backend: MssScreenshotBackend, sct: _FakeSct) -> None:
    """把后端用的 mss 模块换成假模块。"""
    fake = _FakeMss(sct)
    backend._import_mss = lambda: fake


def _png_size(data: bytes) -> tuple[int, int]:
    """用 Pillow 解出 PNG 的像素尺寸。"""
    with Image.open(BytesIO(data)) as image:
        return image.size


# ----------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------


async def test_probe_reports_monitor_count_and_desktop_size() -> None:
    """mss 能导入且能取到显示器列表时，说明显示器数量与整屏尺寸。"""
    backend = MssScreenshotBackend()
    sct = _FakeSct([_DESKTOP, _PRIMARY, _SECONDARY])
    _use_fake_mss(backend, sct)

    detail = await backend.probe(timeout=0.5)

    assert "mss 可用" in detail
    assert "2 块显示器" in detail
    assert "3200x1080" in detail
    assert sct.closed is True


async def test_probe_reports_missing_mss(monkeypatch: pytest.MonkeyPatch) -> None:
    """mss 不可导入时 probe 抛 ScreenshotError。"""
    monkeypatch.setitem(sys.modules, "mss", None)

    with pytest.raises(ScreenshotError) as excinfo:
        await MssScreenshotBackend().probe()

    assert "mss 不可导入" in excinfo.value.message


async def test_probe_reports_no_monitor() -> None:
    """显示器列表里只有虚拟桌面、没有真实屏时抛 ScreenshotError。"""
    backend = MssScreenshotBackend()
    _use_fake_mss(backend, _FakeSct([_DESKTOP]))

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.probe()

    assert "没有检测到任何显示器" in excinfo.value.message


async def test_probe_rejects_invalid_desktop_size() -> None:
    """虚拟桌面尺寸无效时抛 ScreenshotError。"""
    backend = MssScreenshotBackend()
    empty = {"left": 0, "top": 0, "width": 0, "height": 0}
    _use_fake_mss(backend, _FakeSct([empty, _PRIMARY]))

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.probe()

    assert "虚拟桌面尺寸无效" in excinfo.value.message


async def test_probe_times_out() -> None:
    """取显示器列表超过超时时间时抛 ScreenshotError。"""
    backend = MssScreenshotBackend()

    def _slow() -> list[dict[str, Any]]:
        time.sleep(0.05)
        return [_DESKTOP, _PRIMARY]

    backend._list_monitors = _slow

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.probe(timeout=0.001)

    assert "超时" in excinfo.value.message


# ----------------------------------------------------------------------
# capture：三种目标
# ----------------------------------------------------------------------


async def test_capture_screen_returns_png_of_virtual_desktop() -> None:
    """screen 抓虚拟桌面，抓屏矩形就是 monitors[0]，结果是不落盘的 PNG。"""
    backend = MssScreenshotBackend()
    sct = _FakeSct([_DESKTOP, _PRIMARY, _SECONDARY])
    _use_fake_mss(backend, sct)

    shot = await backend.capture(target="screen")

    assert isinstance(shot, Screenshot)
    assert shot.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert (shot.width, shot.height) == (3200, 1080)
    assert _png_size(shot.data) == (3200, 1080)
    assert sct.grabbed == [_DESKTOP]
    assert sct.closed is True


async def test_capture_screen_stitches_multiple_monitors() -> None:
    """screen 把多块屏按虚拟桌面尺寸拼成一张，尺寸是两块屏的包围盒。"""
    desktop = {"left": -800, "top": -600, "width": 2880, "height": 1680}
    left = {"left": -800, "top": -600, "width": 800, "height": 600}
    right = {"left": 0, "top": 0, "width": 2080, "height": 1080}
    backend = MssScreenshotBackend()
    sct = _FakeSct([desktop, left, right])
    _use_fake_mss(backend, sct)

    shot = await backend.capture(target="screen")

    assert (shot.width, shot.height) == (2880, 1680)
    assert sct.grabbed == [desktop]


async def test_capture_primary_uses_main_monitor_rect() -> None:
    """primary 按主屏宽高匹配显示器表，抓的是主屏那块矩形。"""
    backend = MssScreenshotBackend()
    sct = _FakeSct([_DESKTOP, _PRIMARY, _SECONDARY])
    _use_fake_mss(backend, sct)
    backend._primary_size = lambda: (1920, 1080)

    shot = await backend.capture(target="primary")

    assert sct.grabbed == [_PRIMARY]
    assert (shot.width, shot.height) == (1920, 1080)


async def test_capture_primary_prefers_is_primary_flag() -> None:
    """显示器表带 is_primary 标记时直接认它，不被尺寸匹配干扰。"""
    backend = MssScreenshotBackend()
    marked = {"left": 0, "top": 0, "width": 1280, "height": 720, "is_primary": True}
    other = {"left": 1280, "top": 0, "width": 1920, "height": 1080}
    sct = _FakeSct([_DESKTOP, other, marked])
    _use_fake_mss(backend, sct)
    backend._primary_size = lambda: (1920, 1080)

    await backend.capture(target="primary")

    assert sct.grabbed == [marked]


async def test_capture_primary_falls_back_to_first_monitor() -> None:
    """主屏尺寸对不上（例如 DPI 缩放）时退回 monitors[1]。"""
    backend = MssScreenshotBackend()
    sct = _FakeSct([_DESKTOP, _PRIMARY, _SECONDARY])
    _use_fake_mss(backend, sct)
    backend._primary_size = lambda: None

    await backend.capture(target="primary")

    assert sct.grabbed == [_PRIMARY]


async def test_capture_monitor_uses_requested_index() -> None:
    """monitor 的序号 0 起数，对应显示器表里 monitors[序号 + 1]。"""
    backend = MssScreenshotBackend()
    sct = _FakeSct([_DESKTOP, _PRIMARY, _SECONDARY])
    _use_fake_mss(backend, sct)

    shot = await backend.capture(target="monitor", monitor=1)

    assert sct.grabbed == [_SECONDARY]
    assert (shot.width, shot.height) == (1280, 1024)


@pytest.mark.parametrize("monitor", [2, 5])
async def test_capture_monitor_out_of_range_raises(monitor: int) -> None:
    """显示器序号越界时抛 ScreenshotError。"""
    backend = MssScreenshotBackend()
    _use_fake_mss(backend, _FakeSct([_DESKTOP, _PRIMARY, _SECONDARY]))

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture(target="monitor", monitor=monitor)

    assert "越界" in excinfo.value.message


async def test_capture_monitor_negative_raises() -> None:
    """显示器序号为负时抛 ScreenshotError。"""
    backend = MssScreenshotBackend()
    _use_fake_mss(backend, _FakeSct([_DESKTOP, _PRIMARY]))

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture(target="monitor", monitor=-1)

    assert "不能为负数" in excinfo.value.message


# ----------------------------------------------------------------------
# capture：失败转 E_SCREENSHOT
# ----------------------------------------------------------------------


async def test_capture_wraps_grab_failure_as_screenshot_error() -> None:
    """抓屏失败转成 ScreenshotError，错误码是 E_SCREENSHOT。"""
    backend = MssScreenshotBackend()
    sct = _FakeSct([_DESKTOP, _PRIMARY], error=OSError("桌面不可访问"))
    _use_fake_mss(backend, sct)

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert excinfo.value.code == "E_SCREENSHOT"
    assert "mss 抓屏失败" in excinfo.value.message
    assert "桌面不可访问" in excinfo.value.message
    assert sct.closed is True


async def test_capture_wraps_missing_mss(monkeypatch: pytest.MonkeyPatch) -> None:
    """mss 不可导入时 capture 抛 ScreenshotError。"""
    monkeypatch.setitem(sys.modules, "mss", None)

    with pytest.raises(ScreenshotError) as excinfo:
        await MssScreenshotBackend().capture()

    assert "mss 不可导入" in excinfo.value.message


async def test_capture_rejects_invalid_bitmap_size() -> None:
    """位图尺寸不合法时抛 ScreenshotError。"""
    backend = MssScreenshotBackend()
    sct = _FakeSct([_DESKTOP, _PRIMARY], shot=_BrokenShot((0, 0)))
    _use_fake_mss(backend, sct)

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert "位图尺寸无效" in excinfo.value.message


async def test_capture_rejects_bitmap_without_pixels() -> None:
    """位图没有像素数据时抛 ScreenshotError。"""
    backend = MssScreenshotBackend()
    sct = _FakeSct([_DESKTOP, _PRIMARY], shot=_BrokenShot((8, 8)))
    _use_fake_mss(backend, sct)

    with pytest.raises(ScreenshotError) as excinfo:
        await backend.capture()

    assert "没有像素数据" in excinfo.value.message


async def test_capture_closes_screen_even_on_failure() -> None:
    """抓屏失败时也要关掉 mss 句柄。"""
    backend = MssScreenshotBackend()
    sct = _FakeSct([_DESKTOP, _PRIMARY], error=RuntimeError("抓屏挂了"))
    _use_fake_mss(backend, sct)

    with pytest.raises(ScreenshotError):
        await backend.capture()

    assert sct.closed is True