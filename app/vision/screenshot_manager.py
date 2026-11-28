"""截图流程的编排。

按《4-截图输入与多模态》截图流程一节编排：采集 → 处理 → 编码 → 提交。
采集用平台截图后端，处理与编码用 :class:`~app.vision.image_processor.ImageProcessor`，
提交交给 :class:`~app.core.question_manager.QuestionManager`。

分工边界：

- 本模块只管拿到图并提交，语音文字的合并不在这里做，
  合并发生在 QuestionManager 里，见《4-截图输入与多模态》与语音合并一节；
- 截图失败只写状态栏提示位，错误码 ``E_SCREENSHOT``，
  程序继续运行，不退出，也不改状态栏的 MODEL 字段；
- 截图照常采集、照常提交，不按模型支不支持图片分叉：
  本项目固定用配置里的模型，它声明支持 image 模态，
  见《4-截图输入与多模态》模型视觉能力一节。

采集这一步单独计时，指标名 ``capture_latency``，来源是后端名，
见《2-骨架与配置》性能指标一节。

采集限时 ``screenshot.timeout`` 秒，超时按采集失败处理，错误码 ``E_SCREENSHOT``，
见《8-集成与验收》「超时管理」一节。

提示位由外部注入：``hint_publisher`` 是异步回调，传空串表示清掉提示。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from app.config import AppConfig
from app.core.exceptions import ScreenshotError
from app.platform.types import ScreenshotBackend
from app.utils.logger import log_event
from app.utils.timing import default_recorder, timed
from app.vision.image_processor import ImageProcessor

if TYPE_CHECKING:
    # 运行期不 import 核心层的实现类，跨层只通过接口交互
    from app.core.question_manager import QuestionManager
    from app.inputs.input_router import InputRouter

__all__ = [
    "ScreenshotManager",
    "SCREENSHOT_HINT",
    "SCREENSHOT_ERROR_CODE",
    "CAPTURE_LATENCY_METRIC",
]


# 采集过程中的提示位文案
SCREENSHOT_HINT: str = "截图中"

# 截图采集耗时的指标名，见《2-骨架与配置》性能指标一节
CAPTURE_LATENCY_METRIC: str = "capture_latency"

# 截图失败的错误码，复用核心层异常上的取值，见《2-骨架与配置》错误码一节
SCREENSHOT_ERROR_CODE: str = ScreenshotError.code

# 写状态栏提示位的最长等待（秒）。界面忙时提示位可能迟迟不回，
# 不能让它把截图采集一起拖住，超时只记日志并继续采集。
HINT_TIMEOUT_SECONDS: float = 1.0


class ScreenshotManager:
    """截图流程的编排者。

    :param config: 应用配置。
    :param backend: 平台截图后端，Windows 侧是 mss。
    :param processor: 图片处理器，负责缩放、编码与体积校验。
    :param manager: 问题管理器，图片转成 data URL 后交给它提交。
    :param hint_publisher: 状态栏提示位的异步回调，传空串表示清掉提示；
        没给时只写日志，不提示。
    :param router: 输入路由器，用来标记一次截图「在路上」。
        给了它，:meth:`capture_and_submit` 进出时分别调
        ``screenshot_started`` 与 ``screenshot_finished``，语音文字会等这张图；
        没给时行为与从前一致，只采集与提交。
    """

    def __init__(
        self,
        config: AppConfig,
        backend: ScreenshotBackend,
        processor: ImageProcessor,
        manager: QuestionManager,
        *,
        hint_publisher: Callable[[str], Awaitable[None]] | None = None,
        router: InputRouter | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._processor = processor
        self._manager = manager
        self._hint_publisher = hint_publisher
        self._router = router

    @property
    def backend_name(self) -> str:
        """当前用的截图后端名。

        对应配置里 ``providers.screenshot`` 的取值：``mock`` 是假后端，
        真实实现是平台后端名（Windows 侧是 ``mss``）。
        """
        return self._backend.name

    async def capture_and_submit(self) -> bool:
        """采集一张截图并提交一次带图的请求。

        流程：标记截图在路上 → 提示「截图中」→ 采集 → 处理 → 编码 → 提交 →
        清掉提示位 → 标记截图结束。
        采集、处理、编码与提交任何一步失败都转成 ``E_SCREENSHOT`` 提示，
        异常不往调用方抛，程序继续运行。
        挂了输入路由器时，失败路径同样走到 ``screenshot_finished``：
        等着的语音文字由路由器单独提交，不会丢掉。

        采集这一步单独计时，指标名 ``capture_latency``，来源是后端名；
        采集失败时这次耗时同样落库，日志级别是 WARNING。

        采集限时 ``screenshot.timeout`` 秒，超时按采集失败处理，
        见《8-集成与验收》「超时管理」一节。

        :returns: 提交成功返回 True；任何一步失败返回 False。
        """
        router = self._router

        if router is not None:
            router.screenshot_started()
        try:
            return await self._capture_and_submit()
        finally:
            if router is not None:
                try:
                    await router.screenshot_finished()
                except Exception as exc:
                    # 收尾失败不覆盖这次截图的结果，只记日志
                    log_event(
                        "vision.screenshot.router_finished_failed",
                        f"截图收尾失败：{exc}",
                        level="WARNING",
                        error=_first_line(exc),
                    )

    async def _capture_and_submit(self) -> bool:
        """采集并提交的实际流程，由 :meth:`capture_and_submit` 包一层收尾。"""
        await self._publish_hint(SCREENSHOT_HINT)
        timer = timed(
            CAPTURE_LATENCY_METRIC,
            source=self.backend_name,
            recorder=default_recorder(),
        )
        timeout = float(self._config.screenshot.timeout)
        try:
            # 只包住后端采集这一步，图片处理与编码不计入采集耗时
            with timer:

                screenshot = await asyncio.wait_for(
                    self._backend.capture(
                        target=self._config.screenshot.target,
                        monitor=self._config.screenshot.monitor,
                    ),
                    timeout,
                )
        except asyncio.TimeoutError as exc:
            # 采集超时：转成截图失败，错误码还是 E_SCREENSHOT
            await self._fail(
                "截图失败", ScreenshotError(f"截图接口超时（{timeout:g} 秒）")
            )
            self._log_capture_latency(timer.elapsed_ms, failed=True, error="截图接口超时")
            return False
        except Exception as exc:
            # 采集失败也算一次真实采集耗时，同样落库
            await self._fail("截图失败", exc)
            self._log_capture_latency(timer.elapsed_ms, failed=True, error=_first_line(exc))
            return False
        self._log_capture_latency(timer.elapsed_ms, failed=False)


        try:
            processed = self._processor.process(screenshot.data)
            data_url = self._processor.to_data_url(processed)
        except Exception as exc:
            # 图片处理失败归到 E_SCREENSHOT
            await self._fail("截图失败", exc)
            return False

        try:
            # 文字由 QuestionManager 取会话保留的语音文字，这里只交图
            await self._manager.submit_text("", image_data_url=data_url)
        except Exception as exc:
            await self._fail("截图提交失败", exc)
            return False

        await self._publish_hint("")
        return True

    def _log_capture_latency(
        self, elapsed_ms: float, *, failed: bool, error: str = ""
    ) -> None:
        """记一条采集耗时日志。

        事件名 ``metric.capture_latency``，毫秒值与来源写进日志字段；
        采集失败时级别是 WARNING，并带上失败原因。
        """
        fields: dict[str, Any] = {
            "metric": CAPTURE_LATENCY_METRIC,
            "value_ms": round(elapsed_ms, 3),
            "source": self.backend_name,
        }
        if failed:
            fields["failed"] = True
            fields["error"] = error
        log_event(
            "metric.capture_latency",
            f"截图采集耗时 {elapsed_ms:.1f} 毫秒",
            level="WARNING" if failed else "INFO",
            **fields,
        )

    async def _fail(self, summary: str, exc: Exception) -> None:
        """把失败写成状态栏提示位，不抛异常。"""
        detail = _first_line(exc)
        log_event(
            "vision.screenshot.failed",
            f"{summary}：{detail}",
            level="WARNING",
            error=detail,
        )
        await self._publish_hint(f"{SCREENSHOT_ERROR_CODE} {summary}：{detail}")

    async def _publish_hint(self, hint: str) -> None:
        """写状态栏提示位，没给回调时只写日志。"""
        publisher = self._hint_publisher
        if publisher is None:
            log_event(
                "vision.screenshot.hint",
                f"提示位：{hint}" if hint else "清掉提示位",
                level="DEBUG",
            )
            return
        try:
            # 界面忙时提示位可能长时间不回，限时等待后继续采集
            await asyncio.wait_for(publisher(hint), HINT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            log_event(
                "vision.screenshot.hint_timeout",
                f"写提示位超时（{HINT_TIMEOUT_SECONDS:g} 秒），截图流程继续",
                level="WARNING",
                timeout=HINT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            # 提示写不出去不影响截图流程
            log_event(
                "vision.screenshot.hint_failed",
                f"写提示位失败：{exc}",
                level="WARNING",
            )


def _first_line(exc: Exception) -> str:
    """取异常的第一行文字，作为一句话说明。"""
    text = str(exc).strip()
    if not text:
        return "未知原因"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "未知原因"
