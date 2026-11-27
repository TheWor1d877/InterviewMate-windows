"""三种输入的统一分发。

语音、截图、手工输入三种来源在这里收成一条提交路径，见《8-集成与验收》三种输入的汇合一节。

这一层只做三件事：

- 决定 ``source``：语音是 ``voice``、截图是 ``screenshot``、手工输入是 ``text``；
- 表达 ``auto_submit`` 语义：够格自动提交的材料才触发提交回调，
  ``auto_submit`` 为 false 的材料只留在路由器里；
- 安排提交时机：语音与截图按事件会合（rendezvous）。语音最终文本到达时，
  截图若还在采集就暂存文字等它；截图到达时，录音若还在进行就暂存图片等语音结束；
  两边都不在路上时各自立即提交，纯语音提交没有任何等待，
  见《4-截图输入与多模态》与语音合并一节与《8-集成与验收》三种输入的汇合一节。

「截图在采集」由 :meth:`InputRouter.screenshot_started` 与
:meth:`InputRouter.screenshot_finished` 这对同步/异步入口标记，
「录音进行中」由 :meth:`InputRouter.voice_session_started` 标记。
本模块不自己开定时器，也不等时间。

提交入口由外面注入，形状是 ``(question, image_data_url)``，指向核心层的提交入口。
本模块不 import 核心层的问题管理器实现类；输入层依赖核心层是允许的方向，
见《2-骨架与配置》目录结构约定一节。

纯语音提交不带历史截图：会话保留的截图归核心层的问题管理器，
只在合成带图请求与重新分析时用。去重与「语音文字加截图合成一条请求」的规则
同样由提交入口那一侧负责，本模块不重复实现，
见《8-集成与验收》防止重复调用一节。

本模块不认识模型客户端、不发网络请求、不碰界面。
"""

from __future__ import annotations

import base64
import binascii
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from app.core.question_session import QuestionInput
from app.inputs.text_input import build_text_input
from app.utils.logger import log_event

__all__ = [
    "SOURCE_TEXT",
    "SOURCE_VOICE",
    "SOURCE_SCREENSHOT",
    "SOURCES",
    "SubmitFunc",
    "PendingMaterial",
    "InputRouter",
]


# source 的三个取值，与《8-集成与验收》三种输入的汇合一节一致
SOURCE_TEXT: str = "text"
SOURCE_VOICE: str = "voice"
SOURCE_SCREENSHOT: str = "screenshot"
SOURCES: tuple[str, ...] = (SOURCE_VOICE, SOURCE_SCREENSHOT, SOURCE_TEXT)


class SubmitFunc(Protocol):
    """提交回调的形状。

    参数是题面与图片的 data URL，异步返回。核心层的提交入口形状就是这样，
    这里只用协议声明，不 import 那个实现类，见《2-骨架与配置》目录结构约定一节。
    """

    async def __call__(self, question: str, image_data_url: str | None) -> None: ...


@dataclass(frozen=True)
class PendingMaterial:
    """留在路由器里、还没提交的材料。

    统一输入结构投影出来的形状，记来源、文字、图片与自动提交语义。
    """

    source: str
    text: str
    image_data_url: str | None
    auto_submit: bool


class InputRouter:
    """三种输入的统一分发。

    语音与截图按事件会合：两段材料都在路上时暂存先到的一段，后到的一段到达时
    合成一次提交；只有一段在路上时立即提交，不等待。

    :param submit: 提交回调，形状是 ``(question, image_data_url)``，
        指向核心层的提交入口。
    :param session_id: 会话标识，不给就现生成一个 UUID。
    """

    def __init__(
        self,
        submit: SubmitFunc,
        *,
        session_id: str | None = None,
    ) -> None:
        self._submit = submit
        self._session_id = session_id or str(uuid.uuid4())
        # 最后一条送进来的输入结构，来源与自动提交语义从这里看
        self._last_input: QuestionInput | None = None
        # 会合中暂存的语音文字与截图，两者都在场时合成一次提交
        self._pending_text: str | None = None
        self._pending_image: str | None = None
        # 暂存材料的自动提交语义：false 的材料只留在路由器里
        self._pending_auto_submit: bool = True
        # 正在采集的截图数量，大于 0 表示截图还在路上
        self._screenshots_in_flight: int = 0
        # 录音进行中：截图先到时暂存图片等语音结束
        self._voice_busy: bool = False

    @property
    def session_id(self) -> str:
        """当前会话标识。"""
        return self._session_id

    @property
    def last_input(self) -> QuestionInput | None:
        """最后一条送进来的输入结构，没有时返回 None。"""
        return self._last_input

    @property
    def last_source(self) -> str | None:
        """最后一条输入的来源，没有输入时返回 None。"""
        return self._last_input.source if self._last_input is not None else None

    def voice_session_started(self) -> None:
        """标记录音开始：截图先到时暂存图片，等语音结束再决定怎么提交。"""
        self._voice_busy = True

    async def voice_finished(self) -> None:
        """标记录音结束并冲刷残留材料。

        语音结束但本次没有产出最终文本（空语音或与上一句重复）时，
        先到的那张截图没有语音可以合并，这里把它单独提交，避免留到下一次提问。
        截图还在路上时不动残留材料，等 :meth:`screenshot_finished` 收尾。
        """
        self._voice_busy = False
        if self._screenshots_in_flight == 0:
            await self._flush_pending(reason="voice_finished")

    def screenshot_started(self) -> None:
        """标记一次截图开始采集：语音文字到达时暂存，等这张图。"""
        self._screenshots_in_flight += 1

    async def screenshot_finished(self) -> None:
        """标记一次截图采集结束。

        截图没产出图（采集或处理失败）时，等着的语音文字没有图可合并，
        这里把它单独提交，不让语音文本丢掉。
        """
        self._screenshots_in_flight = max(0, self._screenshots_in_flight - 1)
        if self._screenshots_in_flight > 0:
            return
        text = self._pending_text
        if text is None:
            return
        self._pending_text = None
        self._pending_auto_submit = True
        log_event(
            "input.router.submit",
            "截图没产出图，等着的语音文字单独提交",
            source=SOURCE_VOICE,
            text_length=len(text),
        )
        await self._submit(text, None)

    async def route_text(self, text: str) -> None:
        """手工输入回车提交。

        ``source`` 是 ``text``、``auto_submit`` 是 true，内容非空时立刻提交一次，
        与先到的语音或截图材料互不合并且直接作废，手工输入独立成一题，
        见《3-文本输入与流式回答》问题会话一节。

        :param text: 输入框里的原文。
        """
        await self.route(build_text_input(text, self._session_id))

    async def route_voice(self, text: str) -> None:
        """语音最终结果。

        ``source`` 是 ``voice``、``auto_submit`` 是 true。
        截图已经拿到时与它合成一次提交；截图还在路上时暂存文字等图；
        两者都没有时立即提交，纯语音路径不等待，
        见《4-截图输入与多模态》与语音合并一节与《8-集成与验收》三种输入的汇合一节。

        :param text: 语音识别出来的最终文字。
        """
        await self.route(self._build_input(SOURCE_VOICE, text, None))

    async def route_screenshot(self, image_data_url: str) -> None:
        """截图。

        ``source`` 是 ``screenshot``、``auto_submit`` 是 true。
        语音文字已经到达时合成一次提交；录音还在进行时暂存图片等语音结束；
        两者都没有时立即提交图片，
        见《4-截图输入与多模态》与语音合并一节。

        :param image_data_url: 截图的 data URL。
        """
        await self.route(self._build_input(SOURCE_SCREENSHOT, None, image_data_url))

    async def route(self, question_input: QuestionInput) -> None:
        """统一入口：按输入结构里的 ``source`` 分发。

        ``auto_submit`` 为 false 的语音材料只把内容留在路由器里，不触发提交回调，
        见《8-集成与验收》三种输入的汇合一节。空内容（没有文字也没有图）不提交。

        :param question_input: 统一输入结构。
        """
        self._last_input = question_input
        source = question_input.source
        if source not in SOURCES:
            log_event(
                "input.router.unknown_source",
                "未知来源，按手工输入处理",
                level="WARNING",
                source=source,
            )
            source = SOURCE_TEXT
        text = (question_input.text or "").strip()
        image_data_url = self._image_data_url(question_input)
        if not text and not image_data_url:
            log_event(
                "input.router.empty",
                "输入内容为空，不提交",
                level="DEBUG",
                source=source,
            )
            return
        if source == SOURCE_VOICE:
            if not question_input.auto_submit:
                # 不允许自动提交的语音材料只留在路由器里，等显式冲刷
                self._pending_text = text
                self._pending_auto_submit = False
                log_event(
                    "input.router.hold",
                    "语音材料不允许自动提交，留在路由器里",
                    level="DEBUG",
                    source=source,
                    text_length=len(text),
                )
                return
            await self._dispatch_voice(text)
            return
        if source == SOURCE_SCREENSHOT:
            await self._dispatch_screenshot(image_data_url)
            return
        await self._dispatch_text(text)

    async def submit_pending(self) -> None:
        """显式冲刷当前暂存的材料。

        文本与图都在时合成一次提交，只有其中一项时单独提交；
        不受截图是否在路上限制（测试与外部收尾会用到）。
        ``auto_submit`` 为 false 的材料不提交，继续留在路由器里，
        见《8-集成与验收》三种输入的汇合一节。
        """
        await self._flush_pending(reason="submit_pending")

    def reset_session(self) -> None:
        """会话级重置：Ctrl+N 新会话时调用。

        丢掉会合中暂存的材料与最后一条输入结构，新会话不继承旧会话的输入；
        正在采集的截图与录音状态不动，它们属于进行中的动作。
        会话保留的截图归问题管理器，它的 ``new_question`` 会一起清掉。
        """
        self._pending_text = None
        self._pending_image = None
        self._pending_auto_submit = True
        self._last_input = None
        log_event(
            "input.router.reset",
            "已重置输入路由器的会话状态",
            level="DEBUG",
        )

    async def _flush_pending(self, *, reason: str) -> None:
        """把暂存的材料提交掉：文本与图合成一项，或缺项时单独提交。"""
        text = self._pending_text
        image = self._pending_image
        if text is None and image is None:
            return
        if not self._pending_auto_submit:
            log_event(
                "input.router.pending_not_auto",
                "留下的材料不允许自动提交，继续留在路由器里",
                level="DEBUG",
                reason=reason,
            )
            return
        self._pending_text = None
        self._pending_image = None
        self._pending_auto_submit = True
        log_event(
            "input.router.submit",
            "冲刷暂存材料",
            source=SOURCE_VOICE,
            has_voice_text=bool(text),
            has_image=bool(image),
            reason=reason,
        )
        await self._submit(text or "", image)

    async def _dispatch_voice(self, text: str) -> None:
        """语音最终结果到达：先清录音标记，再按截图位置决定合并或立即提交。"""
        self._voice_busy = False
        image = self._pending_image
        if image:
            self._pending_image = None
            self._pending_text = None
            self._pending_auto_submit = True
            log_event(
                "input.router.submit",
                "语音与已到的截图合成一次提交",
                source=SOURCE_VOICE,
                has_voice_text=True,
                has_image=True,
            )
            await self._submit(text, image)
            return
        if self._screenshots_in_flight > 0:
            # 截图还在路上：先存文字，等截图到达时合成一次
            self._pending_text = text
            self._pending_auto_submit = True
            log_event(
                "input.router.hold",
                "截图还在路上，语音文字暂存等图",
                level="DEBUG",
                source=SOURCE_VOICE,
                text_length=len(text),
            )
            return
        log_event(
            "input.router.submit",
            "语音文字立即提交",
            source=SOURCE_VOICE,
            text_length=len(text),
        )
        await self._submit(text, None)

    async def _dispatch_screenshot(self, image_data_url: str | None) -> None:
        """截图到达：先清作废材料，再按语音位置决定合并、暂存或立即提交。"""
        if self._pending_text is not None and not self._pending_auto_submit:
            # 不允许自动提交的残留文字作废，截图自己单独提交
            log_event(
                "input.router.pending_dropped",
                "残留的语音文字不允许自动提交，作废",
                level="DEBUG",
                source=SOURCE_VOICE,
            )
            self._pending_text = None
            self._pending_auto_submit = True
        text = self._pending_text
        if text is not None:
            self._pending_text = None
            self._pending_auto_submit = True
            log_event(
                "input.router.submit",
                "截图与已到的语音文字合成一次提交",
                source=SOURCE_SCREENSHOT,
                has_voice_text=True,
                has_image=bool(image_data_url),
            )
            await self._submit(text, image_data_url)
            return
        if self._voice_busy:
            # 录音还在进行：暂存图片，等语音最终文本到达时合成一次
            self._pending_image = image_data_url
            self._pending_auto_submit = True
            log_event(
                "input.router.hold",
                "录音进行中，截图暂存等语音结束",
                level="DEBUG",
                source=SOURCE_SCREENSHOT,
            )
            return
        log_event(
            "input.router.submit",
            "截图立即提交",
            source=SOURCE_SCREENSHOT,
            has_image=bool(image_data_url),
        )
        await self._submit("", image_data_url)

    async def _dispatch_text(self, text: str) -> None:
        """手工输入：独立成一题，会合中留的语音文字与截图作废。"""
        self._pending_text = None
        self._pending_image = None
        self._pending_auto_submit = True
        log_event(
            "input.router.submit",
            "提交文本请求",
            source=SOURCE_TEXT,
            text_length=len(text),
        )
        await self._submit(text, None)

    def _build_input(
        self,
        source: str,
        text: str | None,
        image_data_url: str | None,
    ) -> QuestionInput:
        """按来源造统一输入结构。

        ``auto_submit`` 写 true：语音最终结果、截图、手工回车都是够格自动提交的材料，
        见《8-集成与验收》三种输入的汇合一节。
        """
        stripped = (text or "").strip()
        return QuestionInput(
            session_id=self._session_id,
            source=source,
            text=stripped or None,
            image_bytes=_data_url_to_bytes(image_data_url),
            timestamp=time.time(),
            auto_submit=True,
            confidence=None,
        )

    @staticmethod
    def _image_data_url(question_input: QuestionInput) -> str | None:
        """从输入结构里取图片材料。

        《8-集成与验收》三种输入的汇合一节的输入结构用 ``image_bytes`` 记图，
        本仓库实际链路用的是 data URL，见《4-截图输入与多模态》多模态请求一节。
        所以这里把字节按 PNG 编成 data URL，提交回调收的就是它。
        没有字节时没有图片材料。
        """
        data = question_input.image_bytes
        if not data:
            return None
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _data_url_to_bytes(data_url: str | None) -> bytes | None:
    """把 data URL 里的 base64 载荷解成图片字节，解不出来时返回 None。"""
    if not data_url:
        return None
    _, separator, payload = data_url.partition(",")
    if not separator or not payload:
        return None
    try:
        return base64.b64decode(payload)
    except (binascii.Error, ValueError):
        return None