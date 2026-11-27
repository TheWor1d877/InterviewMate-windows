"""三种输入统一分发的测试。

覆盖《8-集成与验收》三种输入的汇合一节与《4-截图输入与多模态》与语音合并一节：

- 三种来源各自走一次提交，``source`` 分别是 ``text``、``voice``、``screenshot``；
- 空内容不提交；
- 纯语音到达立即提交，不需要任何窗口或 ``submit_pending``；
- 录音中先到截图、语音结束后合成一次请求；
- 语音先到但截图仍在采集时，语音暂存，截图到达后合成一次请求；
- 抓图失败时 ``screenshot_finished`` 把等着的语音文字单独提交；
- 空语音结束时 ``voice_finished`` 把残留截图单独提交；
- 手工输入丢弃会合中的材料，独立成一题；
- ``auto_submit`` 为 false 的材料只留在路由器里；
- ``reset_session`` 清掉会合中的材料。

提交入口用假实现，被测的是分发逻辑本身。
"""

from __future__ import annotations

import base64

from app.core.question_session import QuestionInput
from app.inputs.input_router import (
    SOURCE_SCREENSHOT,
    SOURCE_TEXT,
    SOURCE_VOICE,
    InputRouter,
)

# 题面与图片样例
TEXT_QUESTION = "解释一下快速排序"
VOICE_TEXT = "请实现一个 LRU Cache"
IMAGE_DATA_URL = "data:image/png;base64,QUJD"
IMAGE_BYTES = b"ABC"


class FakeSubmit:
    """假提交回调，记录每次提交的题面与图片。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(self, question: str, image_data_url: str | None) -> None:
        self.calls.append((question, image_data_url))

    @property
    def count(self) -> int:
        """提交了几次。"""
        return len(self.calls)


def _make_input(
    source: str,
    text: str | None = None,
    image_bytes: bytes | None = None,
    *,
    auto_submit: bool = True,
) -> QuestionInput:
    """造一条统一输入结构。"""
    return QuestionInput(
        session_id="sess-1",
        source=source,
        text=text,
        image_bytes=image_bytes,
        timestamp=0.0,
        auto_submit=auto_submit,
        confidence=None,
    )


async def test_text_source_submits_once() -> None:
    """手工输入回车走一次提交，source 是 text。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    await router.route_text(TEXT_QUESTION)

    assert submit.calls == [(TEXT_QUESTION, None)]
    assert router.last_source == SOURCE_TEXT
    # 路由器自己造的输入结构带自己的会话标识
    assert router.last_input is not None
    assert router.last_input.session_id == router.session_id


async def test_voice_source_submits_immediately() -> None:
    """纯语音到达立即提交，不需要窗口超时，也不需要 ``submit_pending``。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    await router.route_voice(VOICE_TEXT)

    assert submit.calls == [(VOICE_TEXT, None)]
    assert router.last_source == SOURCE_VOICE


async def test_screenshot_source_submits_once() -> None:
    """截图走一次提交，source 是 screenshot。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    await router.route_screenshot(IMAGE_DATA_URL)

    assert submit.calls == [("", IMAGE_DATA_URL)]
    assert router.last_source == SOURCE_SCREENSHOT


async def test_empty_content_is_not_submitted() -> None:
    """空文本且没有图时不提交。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    await router.route_text("")
    await router.route_text("   ")
    await router.route_voice("")
    await router.route_screenshot("")

    assert submit.count == 0


async def test_auto_submit_false_keeps_content_without_submitting() -> None:
    """auto_submit 为 false 时只把内容留在路由器里，不触发提交回调。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    await router.route(_make_input(SOURCE_VOICE, VOICE_TEXT, auto_submit=False))

    assert submit.count == 0

    # 显式冲刷与录音结束都不提交：这份材料本来就不许自动提交
    await router.submit_pending()
    await router.voice_finished()
    assert submit.count == 0


async def test_screenshot_during_recording_merges_with_voice_once() -> None:
    """录音中先到截图时暂存，语音结束后合成一次请求。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    router.voice_session_started()
    await router.route_screenshot(IMAGE_DATA_URL)
    # 录音还在进行，截图先存着，没有提交
    assert submit.count == 0

    await router.route_voice(VOICE_TEXT)

    assert submit.calls == [(VOICE_TEXT, IMAGE_DATA_URL)]


async def test_voice_waits_for_in_flight_screenshot_then_merges_once() -> None:
    """语音先到但截图仍在采集时暂存，截图到达后合成一次请求。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    router.voice_session_started()
    router.screenshot_started()
    await router.route_voice(VOICE_TEXT)
    # 截图还在路上，语音文字先存着，没有提交
    assert submit.count == 0

    await router.route_screenshot(IMAGE_DATA_URL)
    await router.screenshot_finished()

    assert submit.calls == [(VOICE_TEXT, IMAGE_DATA_URL)]


async def test_screenshot_failure_submits_waited_voice_text() -> None:
    """抓图失败时，等着的语音文字由 ``screenshot_finished`` 单独提交。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    router.voice_session_started()
    router.screenshot_started()
    await router.route_voice(VOICE_TEXT)
    assert submit.count == 0

    # 这次抓图没产出图
    await router.screenshot_finished()

    assert submit.calls == [(VOICE_TEXT, None)]


async def test_empty_voice_finishes_and_submits_lingering_screenshot() -> None:
    """空语音结束时，录音中先到的截图单独提交，不留给下一次提问。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    router.voice_session_started()
    await router.route_screenshot(IMAGE_DATA_URL)
    assert submit.count == 0

    await router.voice_finished()

    assert submit.calls == [("", IMAGE_DATA_URL)]


async def test_text_input_drops_pending_material() -> None:
    """手工输入独立成一题，会合中留下的截图作废。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    router.voice_session_started()
    await router.route_screenshot(IMAGE_DATA_URL)
    await router.route_text(TEXT_QUESTION)

    assert submit.calls == [(TEXT_QUESTION, None)]

    # 截图已经作废，录音结束也不会再提交它
    await router.voice_finished()
    assert submit.count == 1


async def test_submit_pending_flushes_text_while_screenshot_in_flight() -> None:
    """显式冲刷不受截图在路上的限制，等着的文字单独提交。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    router.voice_session_started()
    router.screenshot_started()
    await router.route_voice(VOICE_TEXT)
    assert submit.count == 0

    await router.submit_pending()

    assert submit.calls == [(VOICE_TEXT, None)]


async def test_submit_pending_flushes_image_without_text() -> None:
    """只有图时显式冲刷提交空文本加图。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    router.voice_session_started()
    await router.route_screenshot(IMAGE_DATA_URL)
    await router.submit_pending()

    assert submit.calls == [("", IMAGE_DATA_URL)]


async def test_route_uses_source_from_question_input() -> None:
    """统一入口按输入结构里的 source 分发。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    await router.route(_make_input(SOURCE_VOICE, VOICE_TEXT))

    assert submit.calls == [(VOICE_TEXT, None)]
    assert router.last_source == SOURCE_VOICE
    assert router.last_input is not None
    # 统一入口尊重输入结构自带的会话标识
    assert router.last_input.session_id == "sess-1"


async def test_route_with_image_bytes_encodes_data_url() -> None:
    """输入结构里的图片字节按 PNG 编成 data URL 交给提交入口。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    await router.route(_make_input(SOURCE_SCREENSHOT, image_bytes=IMAGE_BYTES))

    assert submit.count == 1
    question, image_data_url = submit.calls[0]
    assert question == ""
    assert image_data_url is not None
    assert image_data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_data_url.split(",", 1)[1]) == IMAGE_BYTES


async def test_unknown_source_falls_back_to_text() -> None:
    """未知来源按手工输入处理，内容非空时照常提交。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    await router.route(_make_input("telepathy", TEXT_QUESTION))

    assert submit.calls == [(TEXT_QUESTION, None)]
    assert router.last_source == "telepathy"


async def test_submit_pending_without_pending_material_does_nothing() -> None:
    """没有留下材料时提交不报错，也不产生提交。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    await router.submit_pending()

    assert submit.count == 0


async def test_reset_session_clears_pending_material() -> None:
    """会话级重置清掉会合中的材料，录音结束也不提交。"""
    submit = FakeSubmit()
    router = InputRouter(submit)

    router.voice_session_started()
    await router.route_screenshot(IMAGE_DATA_URL)
    router.reset_session()

    assert router.last_input is None

    await router.voice_finished()
    assert submit.count == 0