"""检索来源浮层面板的测试。

覆盖《6-Dify检索与项目知识》检索来源展示一节对面板的要求：
列出文档名、分数与正文前若干行，没有片段时显示空态提示。

测试用 Textual 的 ``run_test()`` 无头驱动，测试应用只挂一个面板，
不依赖开发者本地的配置，也不依赖主界面的两栏结构。
"""

from __future__ import annotations

import inspect
from pathlib import Path

from rich.cells import cell_len
from textual.app import App, ComposeResult

from app.ui.panels.rag_sources import (
    ELLIPSIS,
    EMPTY_TEXT,
    PREVIEW_LINES,
    RagSourcesPanel,
)


# 宽终端与窄终端的宽度
WIDE_WIDTH = 120
NARROW_WIDTH = 40


class _PanelApp(App):
    """只挂一个检索来源面板的测试应用。"""

    def compose(self) -> ComposeResult:
        yield RagSourcesPanel(id="rag-sources")


def _entry(
    document_name: str = "",
    score: float | None = None,
    content: str = "",
) -> dict:
    """构造一条片段，字段与面板约定一致。

    :param document_name: 文档名，可以为空串。
    :param score: 分数，可以是 None。
    :param content: 片段正文。
    """
    return {"document_name": document_name, "score": score, "content": content}


# ----------------------------------------------------------------------
# 片段渲染
# ----------------------------------------------------------------------
async def test_show_sources_renders_names_scores_and_previews() -> None:
    """两条片段的文档名、分数与正文预览都渲染出来。"""
    app = _PanelApp()
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        panel.show_sources(
            [
                _entry("架构说明.md", 0.87, "项目分为输入层、核心层与输出层。"),
                _entry("检索接口.md", 0.42, "检索端点是 /datasets/{id}/retrieve。"),
            ]
        )
        await pilot.pause()
        text = panel.plain_text()
        assert "架构说明.md" in text
        assert "检索接口.md" in text
        assert "0.870" in text
        assert "0.420" in text
        assert "项目分为输入层、核心层与输出层。" in text
        assert "/datasets/{id}/retrieve" in text


async def test_show_sources_keeps_the_given_order() -> None:
    """片段按传进来的顺序显示。"""
    app = _PanelApp()
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        panel.show_sources(
            [
                _entry("第一份.md", 0.9, "第一段正文。"),
                _entry("第二份.md", 0.8, "第二段正文。"),
            ]
        )
        await pilot.pause()
        text = panel.plain_text()
        assert text.index("第一份.md") < text.index("第二份.md")


async def test_score_none_is_not_rendered() -> None:
    """分数是 None 时不显示分数那一行，也不报错。"""
    app = _PanelApp()
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        panel.show_sources([_entry("来源文档.md", None, "只有正文。")])
        await pilot.pause()
        text = panel.plain_text()
        assert "来源文档.md" in text
        assert "只有正文。" in text
        assert "分数" not in text


async def test_empty_content_does_not_break_the_panel() -> None:
    """正文为空字符串时面板照常渲染，不崩。"""
    app = _PanelApp()
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        panel.show_sources([_entry("空正文.md", 0.5, "")])
        await pilot.pause()
        text = panel.plain_text()
        assert "空正文.md" in text
        assert "0.500" in text


async def test_empty_document_name_is_not_rendered() -> None:
    """文档名为空时不显示文档名那一行，分数与正文照常显示。"""
    app = _PanelApp()
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        panel.show_sources([_entry("", 0.6, "没有文档名的片段。")])
        await pilot.pause()
        text = panel.plain_text()
        assert "0.600" in text
        assert "没有文档名的片段。" in text


async def test_preview_is_capped_and_marked() -> None:
    """正文预览最多 PREVIEW_LINES 行，超出的部分加省略标记。"""
    app = _PanelApp()
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        content = "\n".join(f"正文第{index}行" for index in range(1, 11))
        panel.show_sources([_entry("长文档.md", 0.5, content)])
        await pilot.pause()
        text = panel.plain_text()
        shown = [
            line
            for line in text.splitlines()
            if line.strip().startswith("正文第")
        ]
        assert len(shown) == PREVIEW_LINES
        assert "正文第4行" in text
        assert "正文第5行" not in text
        assert ELLIPSIS in text


async def test_empty_entries_show_the_empty_hint() -> None:
    """没有片段时显示空态提示，不显示空白。"""
    app = _PanelApp()
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        panel.show_sources([])
        await pilot.pause()
        assert EMPTY_TEXT in panel.plain_text()


async def test_clear_removes_the_entries() -> None:
    """clear 把片段清掉，回到没有片段的状态。"""
    app = _PanelApp()
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        panel.show_sources([_entry("文档.md", 0.75, "一段正文。")])
        await pilot.pause()
        assert "文档.md" in panel.plain_text()
        panel.clear()
        await pilot.pause()
        text = panel.plain_text()
        assert "文档.md" not in text
        assert "一段正文。" not in text
        assert "0.750" not in text
        assert EMPTY_TEXT in text


# ----------------------------------------------------------------------
# 面板形态
# ----------------------------------------------------------------------
async def test_panel_has_a_title() -> None:
    """面板第一行是标题，与现有面板的写法一致。"""
    app = _PanelApp()
    async with app.run_test(size=(WIDE_WIDTH, 30)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        assert panel.render().plain.splitlines()[0] == "检索来源"


async def test_narrow_panel_does_not_overflow() -> None:
    """面板放不下时文本截断，不横向溢出。"""
    app = _PanelApp()
    async with app.run_test(size=(NARROW_WIDTH, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one(RagSourcesPanel)
        panel.show_sources(
            [
                _entry(
                    "这是一个非常非常长的文档名.md",
                    0.87,
                    "这是一段非常非常长的片段正文，宽度不够时要截断。",
                )
            ]
        )
        await pilot.pause()
        for line in panel.plain_text().splitlines():
            assert cell_len(line) <= panel.size.width


def test_panel_does_not_import_the_retrieval_layer() -> None:
    """面板不 import 检索层的实现类，片段由界面转换成普通字典传进来。"""
    source_path = inspect.getsourcefile(RagSourcesPanel)
    assert source_path is not None
    source = Path(source_path).read_text(encoding="utf-8")
    assert "app.retrieval" not in source
    assert "RetrievalResult" not in source
