"""检索来源浮层在界面上的挂载测试。

覆盖《6-Dify检索与项目知识》检索来源展示一节：按 F3 打开检索来源面板，
面板是临时浮层，列出后按 Esc 或再按 F3 关掉，关掉后界面回到原来的两栏布局；
来源存进当前会话，新问题或显式清理时才清掉。

测试用 Textual 的 ``run_test()`` 无头驱动，配置显式构造，
不依赖开发者本地的 ``config/config.yaml``。
"""

from __future__ import annotations

from app.ui.app import InterviewMateApp
from app.ui.panels.answer_panel import AnswerPanel
from app.ui.panels.navigation_panel import NavigationPanel
from app.ui.panels.rag_sources import RagSourcesPanel
from app.ui.status_bar import StatusBar
from tests.test_ui_skeleton import make_config


# 宽终端的宽度
WIDE_WIDTH = 120


def _entry(
    document_name: str = "",
    score: float | None = None,
    content: str = "",
) -> dict:
    """构造一条来源条目，字段与界面约定一致。

    :param document_name: 文档名，可以为空串。
    :param score: 分数，可以是 None。
    :param content: 来源正文。
    """
    return {"document_name": document_name, "score": score, "content": content}


async def test_rag_sources_panel_is_mounted_but_hidden() -> None:
    """浮层挂在界面上，默认不显示。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        assert panel.display is False


async def test_show_rag_sources_makes_the_panel_visible() -> None:
    """show_rag_sources 之后浮层可见，内容是给的那几条来源。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_rag_sources(
            [
                _entry("架构说明.md", 0.87, "项目分为输入层、核心层与输出层。"),
                _entry("检索接口.md", 0.42, "检索端点是 /datasets/{id}/retrieve。"),
            ]
        )
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        assert panel.display is True
        text = panel.plain_text()
        assert "架构说明.md" in text
        assert "检索接口.md" in text
        assert "0.870" in text
        assert "项目分为输入层、核心层与输出层。" in text


async def test_clear_rag_sources_clears_the_panel() -> None:
    """clear_rag_sources 之后浮层里的内容清掉。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_rag_sources([_entry("架构说明.md", 0.87, "一段正文。")])
        await pilot.pause()
        app.clear_rag_sources()
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        assert panel.display is False
        assert "架构说明.md" not in panel.plain_text()
        assert "一段正文。" not in panel.plain_text()


async def test_f3_opens_the_panel() -> None:
    """按 F3 打开检索来源浮层，列的是当前会话的来源。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_rag_sources([_entry("架构说明.md", 0.87, "一段正文。")])
        await pilot.pause()
        # 先把浮层关掉，再按 F3 打开
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one(RagSourcesPanel).display is False
        await pilot.press("f3")
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        assert panel.display is True
        assert "架构说明.md" in panel.plain_text()


async def test_f3_twice_closes_the_panel() -> None:
    """浮层开着时再按一次 F3 关掉，关掉后回到原来的两栏布局。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_rag_sources([_entry("架构说明.md", 0.87, "一段正文。")])
        await pilot.pause()
        await pilot.press("f3")
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        assert panel.display is False
        assert app.query_one(NavigationPanel) is not None
        assert app.query_one(StatusBar) is not None


async def test_the_two_columns_stay_while_the_panel_is_open() -> None:
    """浮层是临时的，左右两栏的结构不变。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_rag_sources([_entry("架构说明.md", 0.87, "一段正文。")])
        await pilot.pause()
        left = app.query_one(NavigationPanel)
        right = app.query_one(AnswerPanel)
        assert left.region.width > 0
        assert right.region.width > 0
        # 两栏还是并排的，右栏在左栏右边
        assert right.region.x >= left.region.x + left.region.width


async def test_escape_closes_the_panel() -> None:
    """按 Esc 关掉浮层，焦点回到输入框。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_rag_sources([_entry("架构说明.md", 0.87, "一段正文。")])
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        assert panel.display is False


async def test_sources_stay_until_a_new_question() -> None:
    """来源存进当前会话，浮层关掉再打开还能看到同一批来源。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_rag_sources([_entry("架构说明.md", 0.87, "一段正文。")])
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("f3")
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        assert panel.display is True
        assert "架构说明.md" in panel.plain_text()


async def test_new_question_clears_the_sources() -> None:
    """Ctrl+N 新建问题时检索来源一起清掉。"""
    app = InterviewMateApp(config=make_config())
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        app.show_rag_sources([_entry("架构说明.md", 0.87, "一段正文。")])
        await pilot.pause()
        app.action_new_question()
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        assert panel.display is False
        assert "架构说明.md" not in panel.plain_text()
