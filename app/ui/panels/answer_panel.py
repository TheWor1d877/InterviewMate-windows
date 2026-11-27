"""右侧「关键回答内容」面板。

面板按模型实际输出顺序流式追加内容：一个表达单元渲染成「关键点：表达」同一行，
兜底文本直接追加在后面。不生成额外栏目，也不按层分组。

代码题走另一种形态：代码逐段追加进代码缓冲，整块用 ``rich.syntax.Syntax`` 高亮，
代码上方先给一个思路块，列出每步标题与一句话思路，完成后在代码块上方标一行
语言、标准与提交模式。两种形态不混排，见《7-代码问题模式》右侧完整代码一节。

形态是锁定的：进了代码形态就只收代码，普通形态的内容被忽略，
右栏已经显示的代码不会被后来的普通事件藏掉。
代码形态也可以先切过去等着，代码随后再追加进来。
截图请求按路由事件预置形态，普通题可以直接切回简答形态，
见《4-截图输入与多模态》；代码题没返回代码时在代码形态下显示一行重试提示。
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static

from app.utils.logger import log_event

__all__ = ["AnswerPanel"]


# 追加内容的两种类型：表达单元与兜底文本
KIND_ITEM = "item"
KIND_FALLBACK = "fallback"

# 右栏的两种形态：普通内容与代码，同一时刻只呈现一种
MODE_NORMAL = "normal"
MODE_CODE = "code"

# 代码块的高亮语言：代码题默认 C++，见《7-代码问题模式》右侧完整代码一节
CODE_LEXER = "cpp"

# 语言标记到高亮语言名的对应，界面层自己判断，不依赖输出层的整理模块
_LEXER_ALIASES = {
    "c": "c",
    "cc": "cpp",
    "cpp": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "go": "go",
    "golang": "go",
    "java": "java",
    "js": "javascript",
    "javascript": "javascript",
    "py": "python",
    "python": "python",
    "rs": "rust",
    "rust": "rust",
    "ts": "typescript",
    "typescript": "typescript",
}

# 代码完成后的状态标记
COMPLETE_MARK = "✓"

# 代码题思路块的小标题
APPROACH_TITLE = "思路"

# 流式追加时合并 layout 刷新的时间窗口（秒），落在 50~100ms 区间内
REFRESH_THROTTLE_INTERVAL = 0.075


def _step_field(step: Any, name: str) -> str:
    """取步骤对象上的一个字段，取不到时是空串。

    步骤既可以是带属性的对象，也可以是字典，两种都认。

    :param step: 一步解题步骤。
    :param name: 字段名，取 title、hint。
    """
    if isinstance(step, dict):
        value = step.get(name, "")
    else:
        value = getattr(step, name, "")
    if value is None:
        return ""
    return str(value)


def _resolve_lexer(language: str) -> str:
    """把语言标记转成高亮用的语言名，认不出时用默认的 C++。

    :param language: 模型给的语言标记，比如 cpp、c++、python。
    """
    return _LEXER_ALIASES.get(language.strip().lower(), CODE_LEXER)



class AnswerPanel(Static):
    """右侧「关键回答内容」面板。"""

    TITLE = "关键回答内容"
    # 代码形态下的栏标题
    CODE_TITLE = "完整代码"
    # 面板可聚焦，焦点从输入框移过来之后 Q 键才退出程序
    can_focus = True

    def __init__(self, **kwargs: Any) -> None:
        # 属性要在 super().__init__ 之前赋值，Textual 允许子类这样做
        # 每一项是 (类型, 所属层标识, 关键点, 文本)
        self._entries: list[tuple[str, str, str, str]] = []
        # 当前形态：普通内容还是代码
        self._mode: str = MODE_NORMAL
        # 代码缓冲，按追加顺序存每一段代码
        self._code_buffer: list[str] = []
        # 代码是否已经完成
        self._code_complete: bool = False
        # 完成时的语言、标准与提交模式
        self._code_meta: tuple[str, str, str] = ("cpp", "cpp17", "auto")
        # 代码块的高亮语言，默认 C++
        self._code_language: str = CODE_LEXER
        # 代码形态下的一行提示，比如代码题没返回代码时的重试提示
        self._notice: str = ""
        # 代码题的解题步骤，渲染成代码上方的思路块
        self._code_steps: list[Any] = []
        # 流式追加的刷新节流：窗口内挂起的刷新与到期补刷的定时器
        self._refresh_timer: Timer | None = None
        self._refresh_pending: bool = False
        # 上一次真正发出 layout 刷新的时刻，用来划分合并窗口
        self._last_refresh_at: float = 0.0
        super().__init__("", markup=False, **kwargs)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def append_item(self, section_id: str, keyword: str, sentence: str) -> None:
        """追加一个表达单元：关键词一行，句子一行。

        代码形态下这条内容被忽略，右栏已经显示的代码不会被藏掉。

        :param section_id: 所属层的标识，只记下来，不单独成栏。
        :param keyword: 关键点。
        :param sentence: 一句表达。
        """
        if self._mode == MODE_CODE:
            log_event(
                "ui.answer.ignore_item",
                "代码形态下忽略普通表达单元",
                level="WARNING",
                section_id=section_id,
            )
            return
        self._entries.append((KIND_ITEM, section_id, keyword, sentence))
        self._request_refresh()

    def append_fallback(self, text: str) -> None:
        """追加兜底文本。

        代码形态下这条内容被忽略，右栏已经显示的代码不会被藏掉。

        :param text: 兜底文本。
        """
        if self._mode == MODE_CODE:
            log_event(
                "ui.answer.ignore_fallback",
                "代码形态下忽略兜底文本",
                level="WARNING",
            )
            return
        self._entries.append((KIND_FALLBACK, "", "", text))
        self._request_refresh()

    def append_code(self, text: str) -> None:
        """追加一段代码。

        代码只往后追加，不重写已经进来的内容，右栏切成代码形态，
        见《7-代码问题模式》右侧完整代码一节。

        :param text: 一段代码文本。
        """
        if not text:
            return
        self._code_buffer.append(text)
        self._mode = MODE_CODE
        self._request_refresh()

    def enter_code_mode(self) -> None:
        """切到代码形态，但不渲染任何内容。

        代码题模式下右栏可以先切形态等着，代码随后再追加进来。
        模型完全不按事件协议输出时，围栏块里的代码是后面才到的，
        见《7-代码问题模式》纯代码块兜底一节。

        """
        self._mode = MODE_CODE
        self.refresh(layout=True)

    def enter_normal_mode(self) -> None:
        """切回简答形态，但不删除已经有的内容。

        截图请求按路由事件预置形态：普通题走简答栏，代码题走代码栏，
        见《4-截图输入与多模态》。内容由分发器在路由事件到达时先清空。
        """
        self._mode = MODE_NORMAL
        self.refresh(layout=True)

    def show_steps(self, steps: list[Any]) -> None:
        """记下代码题的解题步骤，渲染成代码上方的思路块。

        右栏只展示每步的标题与一句话思路，完整代码仍占主体，
        见《7-代码问题模式》右侧完整代码一节。

        :param steps: 解题步骤，每步带 title 与 hint 两个字段。
        """
        self._code_steps = list(steps)
        self.refresh(layout=True)

    def show_code_notice(self, text: str) -> None:
        """在代码栏显示一行提示。

        代码题一个代码片段都没返回时用它提示用户重试，不静默留白，
        见《4-截图输入与多模态》。提示只在代码形态下渲染。

        :param text: 提示内容，空串表示清掉提示。
        """
        self._notice = text.strip()
        self._mode = MODE_CODE
        self.refresh(layout=True)

    def complete_code(
        self,
        language: str = "cpp",
        standard: str = "cpp17",
        submission_mode: str = "auto",
    ) -> None:
        """标记代码已完成。

        语言、标准与提交模式显示在代码块上方的一行小字里。语言同时决定高亮语言，
        认不出时按默认的 C++ 处理。已经渲染过的代码块按新语言重渲染。

        :param language: 代码语言。
        :param standard: 语言标准，比如 cpp17。
        :param submission_mode: 提交模式，取 auto、leetcode、nowcoder、acm。
        """
        self._code_complete = True
        self._code_meta = (language, standard, submission_mode)
        self._code_language = _resolve_lexer(language)
        self._mode = MODE_CODE
        self.refresh(layout=True)

    def clear(self) -> None:
        """清空内容，代码缓冲一并清掉。"""
        self._entries.clear()
        self._code_buffer.clear()
        self._code_complete = False
        self._code_meta = ("cpp", "cpp17", "auto")
        self._code_language = CODE_LEXER
        self._notice = ""
        self._code_steps = []
        self._mode = MODE_NORMAL
        self.refresh(layout=True)

    # ------------------------------------------------------------------
    # 刷新节流
    # ------------------------------------------------------------------
    def _request_refresh(self) -> None:
        """请求一次 layout 刷新，把 50~100ms 内的多次追加合并成少量刷新。

        流式追加是高频路径，逐条 ``refresh(layout=True)`` 会让每个事件都触发
        全屏 reflow。这里用时间窗口合并：窗口首次追加立即刷新，窗口内其余追加
        挂起，窗口到期统一补刷一次，保证最后一次追加一定会被刷新。
        """
        self._refresh_pending = True
        if not self.is_mounted:
            # 还没挂到界面上时没有消息循环，直接刷新即可
            self._refresh_pending = False
            self.refresh(layout=True)
            return
        now = time.monotonic()
        if now - self._last_refresh_at < REFRESH_THROTTLE_INTERVAL:
            # 窗口内：等定时器到期补刷
            self._ensure_refresh_timer()
            return
        # 窗口外：立即刷新一次，后续请求在窗口内合并
        self._last_refresh_at = now
        self._refresh_pending = False
        self.refresh(layout=True)
        self._ensure_refresh_timer()

    def _ensure_refresh_timer(self) -> None:
        """确保窗口到期时有一次补刷，未挂载时不创建定时器。"""
        if self._refresh_timer is None and self.is_mounted:
            self._refresh_timer = self.set_timer(
                REFRESH_THROTTLE_INTERVAL, self._on_refresh_timer
            )

    def _on_refresh_timer(self) -> None:
        """节流窗口到期：把窗口内挂起的刷新补上。"""
        self._refresh_timer = None
        self._flush_refresh()
        self._last_refresh_at = time.monotonic()

    def _flush_refresh(self) -> None:
        """执行挂起的 layout 刷新，没有挂起时什么也不做。"""
        if not self._refresh_pending:
            return
        self._refresh_pending = False
        self.refresh(layout=True)

    def flush_pending(self) -> None:
        """立即执行挂起的刷新，流结束时保证结束态布局完整。"""
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None
        self._flush_refresh()
        self._last_refresh_at = time.monotonic()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        """当前形态，取值是 ``normal`` 或 ``code``。"""
        return self._mode

    @property
    def code_text(self) -> str:
        """当前代码缓冲的完整内容，没收到代码时是空串。"""
        return "".join(self._code_buffer)

    @property
    def code_notice(self) -> str:
        """代码形态下的一行提示，没有提示时是空串。"""
        return self._notice

    @property
    def code_complete(self) -> bool:
        """代码是否已经标记完成。"""
        return self._code_complete

    @property
    def code_language(self) -> str:
        """当前代码块的高亮语言，默认 C++。"""
        return self._code_language

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self) -> RenderableType:
        """返回面板内容：代码形态返代码块，普通形态返正文。"""
        if self._mode == MODE_CODE:
            return self._build_code_view()
        return self._build_normal_view()

    def plain_text(self) -> str:
        """返回正文的纯文本，不含标题行，供测试与调试用。"""
        if self._mode == MODE_CODE:
            return self._build_code_text().plain
        return self._build_text().plain

    def _build_normal_view(self) -> Text:
        """普通形态：第一行是标题，后面是正文。"""
        text = Text()
        title = Text(self.TITLE, style="bold")
        self._fit(title)
        text.append_text(title)
        text.append("\n")
        text.append_text(self._build_text())
        return text

    def _build_code_view(self) -> Group:
        """代码形态：标题、完成状态行、提示行、代码块。"""
        title = Text(self.CODE_TITLE, style="bold")
        self._fit(title)
        parts: list[RenderableType] = [title]
        if self._code_steps:
            parts.append(self._approach_block())
            parts.append(Text(""))
        has_code = bool(self.code_text.strip())
        if self._code_complete and has_code:
            parts.append(self._complete_line())
        if self._notice:
            parts.append(self._notice_line())
        parts.append(self._build_syntax())
        return Group(*parts)

    def _approach_block(self) -> Text:
        """把解题步骤拼成代码上方的思路块。

        每步一行「序号. 标题：一句话思路」，与左侧阶梯一致。
        """
        block = Text()
        header = Text(APPROACH_TITLE, style="bold")
        self._fit(header)
        block.append_text(header)
        for index, step in enumerate(self._code_steps, start=1):
            line = Text()
            line.append(f"{index}. {_step_field(step, 'title')}", style="bold")
            hint = _step_field(step, "hint")
            if hint:
                line.append(f"：{hint}", style="dim")
            self._fit(line)
            block.append("\n")
            block.append_text(line)
        return block

    def _notice_line(self) -> Text:
        """代码栏的提示行，比如代码题没返回代码时的重试提示。"""
        line = Text(self._notice, style="bold yellow")
        self._fit(line)
        return line

    def _build_syntax(self) -> Syntax:
        """按当前高亮语言渲染代码，认不出的语言按 C++ 处理。"""
        return Syntax(
            self.code_text,
            self._code_language,
            background_color="default",
            word_wrap=False,
        )

    def _complete_line(self) -> Text:
        """拼出代码完成后的状态行。"""
        language, standard, submission_mode = self._code_meta
        line = Text()
        line.append(f"{COMPLETE_MARK} 代码完成", style="bold")
        line.append(f" · 语言 {language}", style="dim")
        line.append(f" · 标准 {standard}", style="dim")
        line.append(f" · 模式 {submission_mode}", style="dim")
        self._fit(line)
        return line

    def _build_code_text(self) -> Text:
        """代码形态的纯文本：完成状态行、提示行加代码正文。"""
        text = Text()
        if self._code_steps:
            text.append_text(self._approach_block())
            text.append("\n\n")
        if self._code_complete and self.code_text.strip():
            text.append_text(self._complete_line())
            text.append("\n")
        if self._notice:
            text.append_text(self._notice_line())
            text.append("\n")
        text.append(self.code_text)
        return text

    def _build_text(self) -> Text:
        """按追加顺序拼出面板内容。"""
        text = Text()
        for index, (kind, _section_id, keyword, content) in enumerate(self._entries):
            if index:
                # 两条内容之间空一行，读起来清楚些
                text.append("\n")
            if kind == KIND_ITEM:
                # 小标题独占一行，正文另起一行，读起来是分点而不是一坨段落
                if keyword:
                    text.append(keyword, style="bold")
                    text.append("\n")
                text.append(content)
            else:
                text.append(content)
        return text

    def _fit(self, line: Text) -> None:
        """按面板内容宽度截断一行，放不下时末尾留省略号。"""
        width = self.size.width or 0
        if width > 0:
            line.truncate(width, overflow="ellipsis")
