"""截图请求路由与空代码兜底的界面链路测试。

覆盖《4-截图输入与多模态》里的一条链路：模型先给出路由事件，
分发器按 kind 清空上一题残留并预置右栏形态，之后的 framework 不再改形态；
代码题一个代码片段都没返回时，右栏出现一行重试提示。

测试用 Textual 的 ``run_test()`` 无头驱动真实面板，
配置在文件内显式构造，不依赖开发者本地的 ``config/config.yaml``。
"""

from __future__ import annotations

from app.llm.stream_parser import StreamParser
from app.output.event_dispatcher import EMPTY_CODE_NOTICE, EventDispatcher
from app.output.event_models import (
    CodeCompleteEvent,
    CodeDeltaEvent,
    CodeFrameworkEvent,
    CodeStep,
    DoneEvent,
    FrameworkEvent,
    FrameworkSection,
    RouteEvent,
)
from app.ui.app import InterviewMateApp
from tests.test_ui_skeleton import make_config

WIDE_WIDTH = 120
WIDE_HEIGHT = 30

# 一条代码题截图请求的完整事件流
_ROUTE_CODING = '{"event": "route", "kind": "coding"}'
_CODE_FRAMEWORK = (
    '{"type": "framework", "mode": "coding", '
    '"steps": [{"id": "p1", "title": "读题与约束", "hint": "看清输入范围"}]}'
)
_CODE_DELTA = '{"type": "code_delta", "text": "#include <bits/stdc++.h>\\n"}'
_CODE_COMPLETE = (
    '{"type": "code_complete", "language": "cpp", "standard": "cpp17", '
    '"submission_mode": "leetcode"}'
)


def _make_dispatcher(app: InterviewMateApp) -> EventDispatcher:
    """用界面上的真实面板装一个分发器。"""
    return EventDispatcher(app.navigation_panel, app.answer_panel, app.show_error)


async def test_route_coding_clears_residue_and_switches_to_code_form() -> None:
    """路由到 coding 时右栏清空上一题内容并进代码形态。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, WIDE_HEIGHT)) as pilot:
        await pilot.pause()
        dispatcher = _make_dispatcher(app)
        app.answer_panel.append_item("s1", "旧关键点", "上一题残留")

        dispatcher.dispatch(RouteEvent(kind="coding"))
        await pilot.pause()

        assert app.answer_panel.mode == "code"
        assert "上一题残留" not in app.answer_panel.plain_text()


async def test_route_normal_clears_code_residue_and_switches_to_normal_form() -> None:
    """路由到 normal 时右栏清空上一题代码并回到简答形态。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, WIDE_HEIGHT)) as pilot:
        await pilot.pause()
        dispatcher = _make_dispatcher(app)
        dispatcher.dispatch(RouteEvent(kind="coding"))
        dispatcher.dispatch(CodeDeltaEvent(text="int main() { return 0; }"))

        dispatcher.dispatch(RouteEvent(kind="normal"))
        await pilot.pause()

        assert app.answer_panel.mode == "normal"
        assert "int main" not in app.answer_panel.plain_text()


async def test_framework_does_not_flip_route_form() -> None:
    """路由定下的形态不被后来的 framework 改回去。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, WIDE_HEIGHT)) as pilot:
        await pilot.pause()
        dispatcher = _make_dispatcher(app)

        dispatcher.dispatch(RouteEvent(kind="coding"))
        dispatcher.dispatch(FrameworkEvent(sections=[FrameworkSection(id="s1", title="说明")]))
        await pilot.pause()
        assert app.answer_panel.mode == "code"

        dispatcher.dispatch(RouteEvent(kind="normal"))
        dispatcher.dispatch(CodeFrameworkEvent(steps=[CodeStep(id="p1", title="理解题意")]))
        await pilot.pause()
        assert app.answer_panel.mode == "normal"


async def test_screenshot_stream_coding_route_fills_code_area() -> None:
    """解析器到分发器的真实链路：coding 路由之后左栏出阶梯、右栏出代码。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, WIDE_HEIGHT)) as pilot:
        await pilot.pause()
        dispatcher = _make_dispatcher(app)
        parser = StreamParser("req-shot", "sess-shot")
        stream = "\n".join(
            [_ROUTE_CODING, _CODE_FRAMEWORK, _CODE_DELTA, _CODE_COMPLETE]
        )

        for event in parser.feed(stream + "\n") + parser.flush():
            dispatcher.dispatch(event)
        await pilot.pause()

        assert app.answer_panel.mode == "code"
        assert "#include <bits/stdc++.h>" in app.answer_panel.code_text
        assert "读题与约束" in app.navigation_panel.plain_text()


async def test_empty_code_shows_visible_notice_in_code_form() -> None:
    """coding 全程没有代码时，右栏显示一行可见提示。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, WIDE_HEIGHT)) as pilot:
        await pilot.pause()
        dispatcher = _make_dispatcher(app)
        dispatcher.dispatch(RouteEvent(kind="coding"))
        dispatcher.dispatch(CodeFrameworkEvent(steps=[CodeStep(id="p1", title="理解题意")]))

        dispatcher.dispatch(CodeCompleteEvent())
        dispatcher.dispatch(DoneEvent())
        await pilot.pause()

        assert app.answer_panel.mode == "code"
        assert EMPTY_CODE_NOTICE in app.answer_panel.plain_text()
        assert app.answer_panel.code_notice == EMPTY_CODE_NOTICE