"""假截图后端。

开发期与 CI 里不抓真实屏幕，也要能验证「采集 → 处理 → 提交带图请求」
这条链路，所以 ``providers.screenshot=mock`` 时用这个后端：
每次采集都返回同一张固定的小 PNG。

这个后端不做任何平台判断，见《4-截图输入与多模态》假截图后端一节。
"""

from __future__ import annotations

import base64

from app.platform.types import Screenshot, ScreenshotTarget

__all__ = [
    "MockScreenshotBackend",
    "MOCK_SCREENSHOT_PNG",
    "MOCK_SCREENSHOT_WIDTH",
    "MOCK_SCREENSHOT_HEIGHT",
]


# 固定 PNG：320x180，用 Pillow 画完 "Mock Screenshot" 后 base64 内嵌成常量。
# 不落盘、不每次现画，保证每次拿到的字节完全一样。
_MOCK_PNG_BASE64: str = (
    "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"
    "zp9DxX+H8gAAAABJRU5ErkJggg=="
)

# 假图的图片字节
MOCK_SCREENSHOT_PNG: bytes = base64.b64decode(_MOCK_PNG_BASE64)

# 假图的像素尺寸，与上面那张 PNG 一致
MOCK_SCREENSHOT_WIDTH: int = 320
MOCK_SCREENSHOT_HEIGHT: int = 180


class MockScreenshotBackend:
    """假截图后端，实现 :class:`app.platform.types.ScreenshotBackend` 协议。

    采集不抓真实屏幕，只返回固定的小 PNG。
    """

    # 后端名，对应配置里 ``providers.screenshot`` 的 mock 取值
    name: str = "mock"

    # 这个后端不依赖任何第三方包
    required_packages: tuple[str, ...] = ()

    async def probe(self, timeout: float = 3.0) -> str:
        """检查截图接口能不能用。假后端一直可用。

        :param timeout: 超时秒数，本实现用不上。
        :returns: 一句中文说明。
        """
        return "mock 后端可用，返回固定 PNG"

    async def capture(
        self,
        *,
        target: ScreenshotTarget = "screen",
        monitor: int = 0,
    ) -> Screenshot:
        """做一次截图，返回固定的小 PNG。

        :param target: 截图目标，假后端只有一块屏，这个值用不上。
        :param monitor: 显示器序号，假后端用不上。
        :returns: 截图结果，字段见 :class:`~app.platform.types.Screenshot`。
        """
        return Screenshot(
            data=MOCK_SCREENSHOT_PNG,
            width=MOCK_SCREENSHOT_WIDTH,
            height=MOCK_SCREENSHOT_HEIGHT,
        )
