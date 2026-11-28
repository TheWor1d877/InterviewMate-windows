"""截图图片的压缩、缩放、编码与体积校验。

本模块平台中立：截图原始字节来自平台截图后端（Windows 侧由 ``mss`` 抓屏，
见《4-截图输入与多模态》截图调用一节）。本模块只接收图片字节，不做任何平台判断。

处理规则见《4-截图输入与多模态》图片处理一节：

- 保持原始宽高比，长边压到配置的 ``max_edge``，只压不放大；
- 截图里的题目约束、代码与报错信息要能读清，不为省流量把含文字的图缩得太小；
- 格式按配置取 PNG 或 WebP，默认 PNG，PNG 对文字与代码的呈现更稳；
- 处理完的数据只在内存里，不写文件。

体积上限见《接口契约》模型图片输入限制一节：单边不超过 8192 像素，
单图不超过 32 MiB，整个请求体不超过 48 MiB。体积按编码后的字节数算，
因为 base64 会让体积增加约三分之一。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from app.config import ScreenshotSettings
from app.core.exceptions import ConfigError, ScreenshotError
from app.utils.logger import log_event

__all__ = [
    "ProcessedImage",
    "ImageProcessor",
    "MAX_EDGE_LIMIT",
    "MAX_IMAGE_BYTES",
    "MAX_REQUEST_BYTES",
    "SUPPORTED_FORMATS",
]


# 单边像素上限，见《接口契约》模型图片输入限制一节
MAX_EDGE_LIMIT: int = 8192
# 单图上限 32 MiB，同上
MAX_IMAGE_BYTES: int = 32 * 1024 * 1024
# 整个请求体上限 48 MiB，同上
MAX_REQUEST_BYTES: int = 48 * 1024 * 1024

# 支持的图片格式，取值见《4-截图输入与多模态》图片处理一节
SUPPORTED_FORMATS: tuple[str, ...] = ("png", "webp")

# 估计解码后像素体积时按每像素 4 字节（RGBA）算
_BYTES_PER_PIXEL: int = 4

# WebP 编码质量，见《4-截图输入与多模态》图片处理一节的格式选择
_WEBP_QUALITY: int = 85


def _normalize_format(value: str) -> str:
    """把配置里的格式名归一化成小写关键字。

    :param value: 配置里的格式名，允许带首尾空白与大小写混排。
    :returns: 归一化后的格式名，取值在 :data:`SUPPORTED_FORMATS` 里。
    :raises ConfigError: 归一化后的格式名不是 png 或 webp。
    """
    name = value.strip().lower()
    if name not in SUPPORTED_FORMATS:
        legal = "、".join(SUPPORTED_FORMATS)
        raise ConfigError(
            f"截图格式 {value!r} 不支持；合法取值是 {legal}，"
            "见《4-截图输入与多模态》图片处理一节"
        )
    return name


@dataclass(frozen=True)
class ProcessedImage:
    """处理完的图片。

    :param data: 编码后的图片字节。
    :param mime: MIME 类型，取 ``image/png`` 或 ``image/webp``。
    :param width: 处理后的像素宽。
    :param height: 处理后的像素高。
    """

    data: bytes
    mime: str
    width: int
    height: int


class ImageProcessor:
    """把截图原始字节处理成可提交给多模态接口的图片。

    处理器是无状态的：每次 :meth:`process` 都重新解码、缩放与编码，
    进程内不缓存图片，也没有全局可变状态。

    :param settings: 截图配置，``format`` 取 png 或 webp，
        ``max_edge`` 是长边像素上限。
    :raises ConfigError: ``settings.format`` 归一化后不是 png 或 webp。
    """

    def __init__(self, settings: ScreenshotSettings) -> None:
        """按截图配置建处理器。

        :param settings: 截图配置；``max_edge`` 已是正整数，直接采用。
        :raises ConfigError: ``settings.format`` 归一化后不是 png 或 webp。
        """
        self._format: str = _normalize_format(settings.format)
        self._max_edge: int = settings.max_edge

    def process(self, raw: bytes) -> ProcessedImage:
        """把截图原始字节处理成可提交的图片。

        流程：解开字节 → 按比例压长边 → 编码 → 校验单边、单图体积与请求体体积，
        详见《4-截图输入与多模态》图片处理一节与《接口契约》模型图片输入限制一节。
        全程只在内存里处理，不写任何文件。

        :param raw: 截图原始字节，通常是 PNG。
        :returns: 处理好的图片，含编码字节、MIME 类型与像素尺寸。
        :raises ScreenshotError: 字节解不开，或结果超出体积上限。
        """
        try:
            image = self._decode(raw)
            image = self._normalize_mode(image)
            image = self._resize_to(image, self._max_edge)
            data, mime = self._encode(image)

            if max(image.size) > MAX_EDGE_LIMIT:
                # 配置的 max_edge 大于硬上限，按硬上限等比重采样后重新编码，
                # 见《接口契约》模型图片输入限制一节
                image = self._resize_to(image, MAX_EDGE_LIMIT)
                data, mime = self._encode(image)
                if max(image.size) > MAX_EDGE_LIMIT:
                    raise ScreenshotError(
                        f"图片单边仍超过 {MAX_EDGE_LIMIT} 像素，放弃当前截图"
                    )

            width, height = image.size
            pixel_bytes = width * height * _BYTES_PER_PIXEL
            if pixel_bytes > MAX_IMAGE_BYTES:
                raise ScreenshotError(
                    f"图片解码后体积 {pixel_bytes} 字节超过上限 "
                    f"{MAX_IMAGE_BYTES} 字节，放弃当前截图"
                )
            if len(data) > MAX_IMAGE_BYTES:
                raise ScreenshotError(
                    f"图片编码后体积 {len(data)} 字节超过上限 "
                    f"{MAX_IMAGE_BYTES} 字节，放弃当前截图"
                )
        except ScreenshotError as exc:
            log_event(
                "vision.image.error",
                "截图图片处理失败",
                level="WARNING",
                error=str(exc),
            )
            raise

        log_event(
            "vision.image.process",
            "截图图片处理完成",
            level="DEBUG",
            width=width,
            height=height,
            format=self._format,
            bytes=len(data),
        )
        return ProcessedImage(data=data, mime=mime, width=width, height=height)

    def to_data_url(self, image: ProcessedImage) -> str:
        """把处理好的图片转成 data URL。

        格式是 ``data:<mime>;base64,<编码内容>``，编码内容不带换行，
        见《4-截图输入与多模态》多模态请求一节。

        :param image: 处理好的图片，本方法不修改它。
        :returns: data URL 字符串。
        :raises ScreenshotError: data URL 的字节数超过请求体上限。
        """
        encoded = base64.b64encode(image.data).decode("ascii")
        url = f"data:{image.mime};base64,{encoded}"
        if len(url.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ScreenshotError(
                f"图片 data URL 体积超过请求体上限 {MAX_REQUEST_BYTES} 字节，放弃当前截图"
            )
        return url

    def _decode(self, raw: bytes) -> Image.Image:
        """解开截图原始字节，返回可用的图片对象。

        :param raw: 截图原始字节。
        :returns: 解码后的图片，内容已完整载入内存。
        :raises ScreenshotError: 字节为空、不是图片或数据被截断。
        """
        if not raw:
            raise ScreenshotError("截图数据为空，无法解码")
        try:
            with Image.open(BytesIO(raw)) as opened:
                # load 才会真正解码，截断的数据在这里报错
                opened.load()
                return opened.copy()
        except Exception as exc:
            # Pillow 的解码失败有多种异常类型，统一归到截图失败
            raise ScreenshotError(f"截图数据解码失败：{exc}") from exc

    def _normalize_mode(self, image: Image.Image) -> Image.Image:
        """把图片转成目标格式能存、又便于重采样的模式。

        带 alpha 的图转 RGBA 保留透明通道，其余转 RGB；PNG 下保留灰度，
        避免灰度截图转 RGB 后体积变三倍。

        :param image: 原图。
        :returns: 模式已归一化的新图。
        """
        has_alpha = "A" in image.getbands() or (
            image.mode == "P" and "transparency" in image.info
        )
        if has_alpha:
            return image.convert("RGBA")
        if self._format == "webp":
            # WebP 只存 RGB 与 RGBA
            return image.convert("RGB")
        if image.mode in ("1", "L"):
            # 二值图要转灰度，否则重采样会被迫用最近邻
            return image.convert("L")
        return image.convert("RGB")

    def _resize_to(self, image: Image.Image, limit: int) -> Image.Image:
        """按比例把长边压到 ``limit``，只压不放大。

        :param image: 原图。
        :param limit: 长边像素上限。
        :returns: 缩放后的新图；原图长边不超过 ``limit`` 时返回原图。
        """
        width, height = image.size
        if max(width, height) <= limit:
            return image
        if width >= height:
            new_width = limit
            new_height = max(1, round(height * limit / width))
        else:
            new_height = limit
            new_width = max(1, round(width * limit / height))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _encode(self, image: Image.Image) -> tuple[bytes, str]:
        """按配置格式编码图片。

        :param image: 待编码的图片。
        :returns: 编码后的字节与对应的 MIME 类型。
        """
        buffer = BytesIO()
        if self._format == "webp":
            image.save(buffer, format="WEBP", quality=_WEBP_QUALITY)
            return buffer.getvalue(), "image/webp"
        # PNG 无损，对文字与代码的呈现更稳，见《4-截图输入与多模态》图片处理一节
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"
