"""截图流程编排的测试。

覆盖《4-截图输入与多模态》截图流程一节的采集、处理、编码、提交四步，
以及两条失败路径：截图失败、图片处理失败都转成 ``E_SCREENSHOT`` 提示，
异常不往调用方抛，程序继续跑。

另外验采集耗时：成功与失败都往进程级记录器里落一条 ``capture_latency`` 样本。

再验界面这侧的两件事：F9 的截图流程走后台任务，不卡界面；
提示位单独改时其余五个状态字段不被重置。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from io import BytesIO
from typing import Any

import pytest
from PIL import Image

from app.config import AppConfig, load_config
from app.core.exceptions import ScreenshotError
from app.platform.types import Screenshot
from app.ui.app import InterviewMateApp
from app.ui.panels.question_header import QuestionHeader
from app.ui.status_bar import StatusBar
from app.utils.timing import default_recorder, reset_default_recorder
from app.vision.image_processor import ImageProcessor, ProcessedImage
from app.vision.mock_screenshot import (
    MOCK_SCREENSHOT_PNG,
    MOCK_SCREENSHOT_HEIGHT,
    MOCK_SCREENSHOT_WIDTH,
    MockScreenshotBackend,
)
from app.vision.screenshot_manager import (
    SCREENSHOT_HINT,
    ScreenshotManager,
)

# 采集到的原始字节与处理后的 data URL
RAW_BYTES = b"raw-screenshot-bytes"
DATA_URL = "data:image/png;base64,AAAA"


@pytest.fixture(autouse=True)
def _clean_default_recorder() -> Iterator[None]:
    """每个用例前清空进程级记录器，避免样本互相串味。"""
    reset_default_recorder()
    yield
    reset_default_recorder()


class FakeBackend:
    """假截图后端，记录采集次数，按需要抛异常。"""

    name = "fake"
    required_packages: tuple[str, ...] = ()

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.capture_count = 0
        self.last_target: str = ""
        self.last_monitor: int = -1

    async def capture(
        self,
        *,
        target: str = "screen",
        monitor: int = 0,
    ) -> Screenshot:
        """返回一张假截图，或抛出配置的异常。"""
        self.capture_count += 1
        self.last_target = target
        self.last_monitor = monitor
        if self.error is not None:
            raise self.error
        return Screenshot(data=RAW_BYTES, width=800, height=600)


class FakeProcessor:
    """假图片处理器，记录处理过的字节，按需要抛异常。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.processed: list[bytes] = []

    def process(self, raw: bytes) -> ProcessedImage:
        """返回一张假图片，或抛出配置的异常。"""
        self.processed.append(raw)
        if self.error is not None:
            raise self.error
        return ProcessedImage(data=b"processed", mime="image/png", width=800, height=600)

    def to_data_url(self, image: ProcessedImage) -> str:
        """返回固定的 data URL。"""
        return DATA_URL


class FakeQuestionManager:
    """假问题管理器，记录提交的文字与图片。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.submissions: list[tuple[str, str | None]] = []

    async def submit_text(self, text: str, image_data_url: str | None = None) -> None:
        """记录一次提交。"""
        self.submissions.append((text, image_data_url))
        if self.error is not None:
            raise self.error


class HintRecorder:
    """记录提示位内容的回调。"""

    def __init__(self) -> None:
        self.hints: list[str] = []

    async def __call__(self, hint: str) -> None:
        """记下这次的提示位内容。"""
        self.hints.append(hint)


class FakeRouter:
    """假输入路由器，只记截图生命周期标记。"""

    def __init__(self) -> None:
        self.started = 0
        self.finished = 0

    def screenshot_started(self) -> None:
        """记下截图开始。"""
        self.started += 1

    async def screenshot_finished(self) -> None:
        """记下截图结束。"""
        self.finished += 1


def make_manager(
    backend: FakeBackend | None = None,
    processor: FakeProcessor | None = None,
    question_manager: FakeQuestionManager | None = None,
    *,
    hint_publisher: Any | None = None,
    config: AppConfig | None = None,
    router: Any | None = None,
) -> tuple[ScreenshotManager, FakeBackend, FakeProcessor, FakeQuestionManager]:
    """搭一套假实现，返回截图管理器与三个假对象。"""
    config = config if config is not None else load_config()
    backend = backend if backend is not None else FakeBackend()
    processor = processor if processor is not None else FakeProcessor()
    question_manager = (
        question_manager if question_manager is not None else FakeQuestionManager()
    )
    manager = ScreenshotManager(
        config,
        backend,
        processor,
        question_manager,  # type: ignore[arg-type]
        hint_publisher=hint_publisher,
        router=router,
    )
    return manager, backend, processor, question_manager


# ----------------------------------------------------------------------
# 采集成功
# ----------------------------------------------------------------------
async def test_capture_success_submits_image_request() -> None:
    """采集成功后提交一次带图请求，图片是处理后的 data URL。"""
    hints = HintRecorder()
    manager, backend, processor, question_manager = make_manager(
        hint_publisher=hints
    )

    assert await manager.capture_and_submit() is True

    assert backend.capture_count == 1
    assert processor.processed == [RAW_BYTES]
    assert question_manager.submissions == [("", DATA_URL)]


async def test_capture_forwards_configured_target_to_backend() -> None:
    """配置里的截图目标与显示器序号会原样传给后端。"""
    base = load_config()
    settings = base.screenshot.model_copy(update={"target": "monitor", "monitor": 2})
    manager, backend, _processor, _question_manager = make_manager(
        config=base.model_copy(update={"screenshot": settings})
    )

    await manager.capture_and_submit()

    assert backend.last_target == "monitor"
    assert backend.last_monitor == 2


async def test_success_clears_the_hint() -> None:
    """采集期间提示「截图中」，成功后清掉提示位。"""
    hints = HintRecorder()
    manager, _backend, _processor, _question_manager = make_manager(
        hint_publisher=hints
    )

    await manager.capture_and_submit()

    assert hints.hints == [SCREENSHOT_HINT, ""]


async def test_capture_without_hint_publisher_still_submits() -> None:
    """没有提示回调时只写日志，流程照走。"""
    manager, _backend, _processor, question_manager = make_manager()

    await manager.capture_and_submit()

    assert question_manager.submissions == [("", DATA_URL)]


async def test_router_marks_screenshot_started_and_finished_on_every_path() -> None:
    """挂了输入路由器时，成功与失败都各标记一次开始与结束。"""
    success_router = FakeRouter()
    success_manager, *_rest = make_manager(router=success_router)

    assert await success_manager.capture_and_submit() is True
    assert success_router.started == 1
    assert success_router.finished == 1

    failure_router = FakeRouter()
    failure_manager, *_rest = make_manager(
        backend=FakeBackend(error=ScreenshotError("截图接口不可用")),
        router=failure_router,
    )

    assert await failure_manager.capture_and_submit() is False
    assert failure_router.started == 1
    assert failure_router.finished == 1


# ----------------------------------------------------------------------
# 采集失败与图片处理失败
# ----------------------------------------------------------------------
async def test_capture_failure_reports_error_hint() -> None:
    """采集失败时提示位出现 E_SCREENSHOT，异常不外抛，也不提交。"""
    hints = HintRecorder()
    manager, _backend, _processor, question_manager = make_manager(
        backend=FakeBackend(error=ScreenshotError("截图接口不可用")),
        hint_publisher=hints,
    )

    assert await manager.capture_and_submit() is False

    assert question_manager.submissions == []
    assert hints.hints[-1].startswith("E_SCREENSHOT")
    assert "截图接口不可用" in hints.hints[-1]


async def test_process_failure_reports_error_hint() -> None:
    """图片处理失败时同样转成 E_SCREENSHOT 提示，不提交。"""
    hints = HintRecorder()
    manager, _backend, _processor, question_manager = make_manager(
        processor=FakeProcessor(error=ScreenshotError("截图数据为空，无法解码")),
        hint_publisher=hints,
    )

    await manager.capture_and_submit()

    assert question_manager.submissions == []
    assert hints.hints[-1].startswith("E_SCREENSHOT")
    assert "截图数据为空，无法解码" in hints.hints[-1]


async def test_unexpected_error_is_also_caught() -> None:
    """不是 ScreenshotError 的异常同样被吃掉，转成 E_SCREENSHOT 提示。"""
    hints = HintRecorder()
    manager, _backend, _processor, question_manager = make_manager(
        backend=FakeBackend(error=RuntimeError("总线断了")),
        hint_publisher=hints,
    )

    await manager.capture_and_submit()

    assert question_manager.submissions == []
    assert hints.hints[-1].startswith("E_SCREENSHOT")


async def test_submit_failure_reports_error_hint() -> None:
    """提交失败也转成 E_SCREENSHOT 提示，异常不外抛。"""
    hints = HintRecorder()
    manager, _backend, _processor, _question_manager = make_manager(
        question_manager=FakeQuestionManager(error=RuntimeError("提交炸了")),
        hint_publisher=hints,
    )

    await manager.capture_and_submit()

    assert hints.hints[-1].startswith("E_SCREENSHOT")


# ----------------------------------------------------------------------
# 采集耗时指标
# ----------------------------------------------------------------------
async def test_capture_records_latency_sample() -> None:
    """采集成功后落一条 capture_latency 样本，值非负，来源是后端名。"""
    manager, _backend, _processor, _question_manager = make_manager()

    await manager.capture_and_submit()

    samples = default_recorder().samples("capture_latency")
    assert len(samples) == 1
    assert samples[0].value_ms >= 0.0
    assert samples[0].source == "fake"


async def test_capture_failure_also_records_latency_sample() -> None:
    """采集失败时这次耗时同样落库。"""
    manager, _backend, _processor, _question_manager = make_manager(
        backend=FakeBackend(error=ScreenshotError("截图接口不可用")),
    )

    assert await manager.capture_and_submit() is False

    samples = default_recorder().samples("capture_latency")
    assert len(samples) == 1
    assert samples[0].value_ms >= 0.0
    assert samples[0].source == "fake"


async def test_process_failure_records_no_extra_sample() -> None:
    """图片处理失败不再采一次图，采集耗时只有一条样本。"""
    manager, _backend, _processor, _question_manager = make_manager(
        processor=FakeProcessor(error=ScreenshotError("截图数据为空，无法解码")),
    )

    await manager.capture_and_submit()

    assert len(default_recorder().samples("capture_latency")) == 1


async def test_mock_backend_latency_source_is_mock() -> None:
    """用假后端时来源记的是 mock。"""
    config: AppConfig = load_config()
    manager = ScreenshotManager(
        config,
        MockScreenshotBackend(),
        ImageProcessor(config.screenshot),
        FakeQuestionManager(),  # type: ignore[arg-type]
    )

    await manager.capture_and_submit()

    samples = default_recorder().samples("capture_latency")
    assert len(samples) == 1
    assert samples[0].source == "mock"


# ----------------------------------------------------------------------
# 假截图后端
# ----------------------------------------------------------------------
async def test_mock_backend_returns_fixed_png() -> None:
    """假后端返回固定的小 PNG，每次拿到的字节都一样。"""
    backend = MockScreenshotBackend()

    first = await backend.capture()
    second = await backend.capture()

    assert backend.name == "mock"
    assert first.data == MOCK_SCREENSHOT_PNG
    assert first.data == second.data
    assert (first.width, first.height) == (MOCK_SCREENSHOT_WIDTH, MOCK_SCREENSHOT_HEIGHT)


def test_mock_backend_png_is_decodable() -> None:
    """假图的字节能被 Pillow 解码成一张 PNG。"""
    image = Image.open(BytesIO(MOCK_SCREENSHOT_PNG))

    assert image.format == "PNG"
    assert image.size == (MOCK_SCREENSHOT_WIDTH, MOCK_SCREENSHOT_HEIGHT)


async def test_mock_backend_png_goes_through_image_processor() -> None:
    """假后端采到的图能过图片处理器，并走完一次带图提交。"""
    config: AppConfig = load_config()
    backend = MockScreenshotBackend()
    question_manager = FakeQuestionManager()
    manager = ScreenshotManager(
        config,
        backend,
        ImageProcessor(config.screenshot),
        question_manager,  # type: ignore[arg-type]
    )

    assert await manager.capture_and_submit() is True

    assert len(question_manager.submissions) == 1
    text, data_url = question_manager.submissions[0]
    assert text == ""
    assert data_url is not None
    assert data_url.startswith("data:image/png;base64,")


# ----------------------------------------------------------------------
# 界面这侧
# ----------------------------------------------------------------------
class FakeScreenshotManager:
    """假截图管理器，记录界面有没有调它。"""

    def __init__(self) -> None:
        self.called = asyncio.Event()
        self.calls = 0

    async def capture_and_submit(self) -> None:
        """记录一次截图提交。"""
        self.calls += 1
        self.called.set()


async def test_action_screenshot_runs_capture_in_background() -> None:
    """F9 触发截图，流程在后台跑，题面写到右栏顶部。"""
    app = InterviewMateApp(config=load_config())
    fake = FakeScreenshotManager()
    app.attach_screenshot_manager(fake)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        app.action_screenshot()
        await pilot.pause()
        await fake.called.wait()
        await pilot.pause()

        assert fake.calls == 1
        header = app.query_one(QuestionHeader)
        assert header.question == "截图提问"


async def test_action_screenshot_without_manager_only_logs() -> None:
    """截图管理器没挂上时按 F9 不报错。"""
    app = InterviewMateApp(config=load_config())

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        app.action_screenshot()
        await pilot.pause()

        assert app.is_running


async def test_publish_hint_keeps_other_status_fields() -> None:
    """只改提示位时其余五个状态字段按快照原样重发，不被重置。"""
    app = InterviewMateApp(config=load_config())

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await app.publish_status(
            running="running",
            audio="on",
            asr="ready",
            dify="ready",
            deepseek="streaming",
            ttft_ms=120.0,
            hint="生成中",
        )
        await pilot.pause()

        await app.publish_hint("E_SCREENSHOT 截图失败")
        await pilot.pause()

        bar = app.query_one(StatusBar)
        assert bar.hint == "E_SCREENSHOT 截图失败"
        snapshot = bar.status_snapshot()
        assert snapshot["running"] == "running"
        assert snapshot["audio"] == "on"
        assert snapshot["asr"] == "ready"
        assert snapshot["dify"] == "ready"
        assert snapshot["deepseek"] == "streaming"
