"""截图图片处理器的测试。

覆盖《4-截图输入与多模态》图片处理一节的缩放、格式与内存内处理规则，
以及《接口契约》DeepSeek 图片输入限制一节的单边像素、单图体积与请求体体积上限。

测试里的图片都用 Pillow 现造，不依赖任何真实截图。
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.config import ScreenshotSettings
from app.core.exceptions import ConfigError, ScreenshotError
from app.vision.image_processor import (
    MAX_EDGE_LIMIT,
    MAX_IMAGE_BYTES,
    MAX_REQUEST_BYTES,
    ImageProcessor,
    ProcessedImage,
)


def _settings(max_edge: int = 1600, fmt: str = "png") -> ScreenshotSettings:
    """造一份截图配置。

    :param max_edge: 长边像素上限。
    :param fmt: 图片格式。
    """
    return ScreenshotSettings(
        enabled=True,
        target="screen",
        max_edge=max_edge,
        format=fmt,
        persist_files=False,
    )


def _make_png(width: int, height: int, mode: str = "RGB") -> bytes:
    """在内存里造一张图片，返回 PNG 字节。

    :param width: 像素宽。
    :param height: 像素高。
    :param mode: 颜色模式。
    """
    if mode == "RGBA":
        color: object = (10, 20, 30, 128)
    elif mode == "L":
        color = 96
    else:
        color = (10, 20, 30)
    buffer = BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_long_edge_is_scaled_to_max_edge_keeping_ratio() -> None:
    """长边超过 max_edge 时等比缩放，缩放后长边正好等于 max_edge。"""
    processor = ImageProcessor(_settings(max_edge=1000))

    result = processor.process(_make_png(4000, 3000))

    assert result.width == 1000
    assert result.height == 750
    assert max(result.width, result.height) == 1000
    assert result.width / result.height == pytest.approx(4000 / 3000, rel=1e-3)


def test_portrait_image_keeps_ratio_after_scaling() -> None:
    """竖图同样等比缩放，长边压到 max_edge。"""
    processor = ImageProcessor(_settings(max_edge=800))

    result = processor.process(_make_png(1200, 3000))

    assert result.height == 800
    assert result.width == 320
    assert result.width / result.height == pytest.approx(1200 / 3000, rel=1e-3)


def test_small_image_is_not_upscaled() -> None:
    """原图比 max_edge 小就保持原尺寸，不放大。"""
    processor = ImageProcessor(_settings(max_edge=1600))

    result = processor.process(_make_png(320, 240))

    assert (result.width, result.height) == (320, 240)


def test_png_format_is_used() -> None:
    """格式配成 png 时，mime 是 image/png，数据也是真的 PNG。"""
    processor = ImageProcessor(_settings(fmt="png"))

    result = processor.process(_make_png(64, 48))

    assert result.mime == "image/png"
    with Image.open(BytesIO(result.data)) as image:
        assert image.format == "PNG"
        assert image.size == (64, 48)


def test_webp_format_is_used() -> None:
    """格式配成 webp 时，mime 是 image/webp，数据也是真的 WebP。"""
    processor = ImageProcessor(_settings(fmt="webp"))

    result = processor.process(_make_png(64, 48))

    assert result.mime == "image/webp"
    with Image.open(BytesIO(result.data)) as image:
        assert image.format == "WEBP"
        assert image.size == (64, 48)


def test_format_name_is_normalized() -> None:
    """格式名带空白或大小写混排时按归一化后的名字处理。"""
    processor = ImageProcessor(_settings(fmt="  WebP "))

    result = processor.process(_make_png(32, 32))

    assert result.mime == "image/webp"


def test_illegal_format_raises_config_error() -> None:
    """格式不是 png 或 webp 时抛 ConfigError。"""
    with pytest.raises(ConfigError):
        ImageProcessor(_settings(fmt="jpg"))


def test_broken_bytes_raise_screenshot_error() -> None:
    """不是图片的字节抛 ScreenshotError。"""
    processor = ImageProcessor(_settings())

    with pytest.raises(ScreenshotError):
        processor.process(b"this-is-not-an-image")


def test_empty_bytes_raise_screenshot_error() -> None:
    """空字节抛 ScreenshotError。"""
    processor = ImageProcessor(_settings())

    with pytest.raises(ScreenshotError):
        processor.process(b"")


def test_truncated_bytes_raise_screenshot_error() -> None:
    """截断的图片数据抛 ScreenshotError。"""
    processor = ImageProcessor(_settings())
    raw = _make_png(64, 48)

    with pytest.raises(ScreenshotError):
        processor.process(raw[: len(raw) // 2])


def test_edge_above_hard_limit_is_scaled_back() -> None:
    """单边超过 8192 时按硬上限等比压回 8192。"""
    processor = ImageProcessor(_settings(max_edge=10000))

    result = processor.process(_make_png(9000, 100))

    assert result.width == MAX_EDGE_LIMIT
    assert result.height == 91
    assert max(result.width, result.height) == MAX_EDGE_LIMIT


def test_alpha_channel_is_kept() -> None:
    """原图带 alpha 通道时保留透明通道。"""
    processor = ImageProcessor(_settings())

    result = processor.process(_make_png(64, 48, mode="RGBA"))

    with Image.open(BytesIO(result.data)) as image:
        assert image.mode == "RGBA"


def test_to_data_url_format_has_no_newline() -> None:
    """data URL 是 data:<mime>;base64,<编码内容>，编码内容不带换行。"""
    processor = ImageProcessor(_settings())
    result = processor.process(_make_png(32, 32))

    url = processor.to_data_url(result)

    assert url.startswith("data:image/png;base64,")
    assert "\n" not in url
    assert "\r" not in url
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == result.data


def test_to_data_url_over_request_limit_raises_screenshot_error() -> None:
    """data URL 超过 48 MiB 时抛 ScreenshotError。"""
    processor = ImageProcessor(_settings())
    # base64 让体积增加约三分之一，37 MiB 的原始字节编码后超过 48 MiB
    oversized = ProcessedImage(
        data=b"\x00" * (37 * 1024 * 1024),
        mime="image/png",
        width=1,
        height=1,
    )

    with pytest.raises(ScreenshotError):
        processor.to_data_url(oversized)


def test_to_data_url_does_not_change_image() -> None:
    """转 data URL 不改动传入的图片对象。"""
    processor = ImageProcessor(_settings())
    result = processor.process(_make_png(32, 32))
    before = (result.data, result.mime, result.width, result.height)

    processor.to_data_url(result)

    assert (result.data, result.mime, result.width, result.height) == before


def test_processing_writes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """处理与转 data URL 都不落盘，当前目录里没有新文件。"""
    monkeypatch.chdir(tmp_path)
    processor = ImageProcessor(_settings())

    result = processor.process(_make_png(128, 96))
    processor.to_data_url(result)

    assert os.listdir(tmp_path) == []


def test_limits_match_contract_values() -> None:
    """三个上限常量与《接口契约》DeepSeek 图片输入限制一节一致。"""
    assert MAX_EDGE_LIMIT == 8192
    assert MAX_IMAGE_BYTES == 32 * 1024 * 1024
    assert MAX_REQUEST_BYTES == 48 * 1024 * 1024
