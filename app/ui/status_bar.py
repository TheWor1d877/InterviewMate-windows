"""顶部状态栏。

状态栏独立于内容区，显示八项，前六项取值见《2-骨架与配置》状态字段取值表，
摆放位置见《3-文本输入与流式回答》终端界面一节的布局图。
八项固定顺序为 RUNNING、AUDIO、ASR、DIFY、MODEL、TTFT、MODE、SESSION，
后两项是项目模式加的：MODE 显示当前是项目模式还是普通模式，
SESSION 显示当前会话 ID，没有会话时显示占位符。

状态栏是两行：第一行是八项状态，第二行是提示位，放错误码与一句话，
例如 ``E_MODEL_TIMEOUT 请求超时``，没有提示时第二行为空。

状态行按段着色：应用名加粗，标签与分隔符用弱化色，取值用正文色；
提示位是错误码时用错误色。底色是中性深色，不用高亮色。

终端窄的时候整行放不下，右端会被直接裁掉，TTFT 那段看不见。
状态栏按控件宽度自适应：放不下就从尾部去掉整项（先 SESSION，再 MODE、TTFT……），
只剩应用名还放不下才截断应用名。
"""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.strip import Strip
from textual.widgets import Static

from app.core.event_bus import Event

__all__ = [
    "STATUS_ORDER",
    "STATUS_LABELS",
    "RUNNING_VALUES",
    "AUDIO_VALUES",
    "ASR_VALUES",
    "DIFY_VALUES",
    "MODEL_VALUES",
    "DEFAULT_APP_NAME",
    "TTFT_EMPTY",
    "ERROR_CODE_PREFIX",
    "StatusUpdate",
    "StatusBar",
]


# 八项的固定顺序，与《3-文本输入与流式回答》终端界面一节的布局图一致，
# mode 与 session 是项目模式加的两项
STATUS_ORDER: tuple[str, ...] = (
    "running",
    "audio",
    "asr",
    "dify",
    "model",
    "ttft",
    "mode",
    "session",
)

# 短标签，状态栏用短标签不用文字描述
STATUS_LABELS: dict[str, str] = {
    "running": "RUNNING",
    "audio": "AUDIO",
    "asr": "ASR",
    "dify": "DIFY",
    "model": "MODEL",
    "ttft": "TTFT",
    "mode": "MODE",
    "session": "SESSION",
}

# 各字段的合法取值，见《2-骨架与配置》状态字段取值表
RUNNING_VALUES: tuple[str, ...] = ("running", "stopped")
AUDIO_VALUES: tuple[str, ...] = ("off", "on")
ASR_VALUES: tuple[str, ...] = ("idle", "loading", "ready", "error")
DIFY_VALUES: tuple[str, ...] = ("offline", "ready", "degraded")
MODEL_VALUES: tuple[str, ...] = ("idle", "connecting", "streaming", "error")

# 字段名到合法取值的映射，供参数校验使用
FIELD_VALUES: dict[str, tuple[str, ...]] = {
    "running": RUNNING_VALUES,
    "audio": AUDIO_VALUES,
    "asr": ASR_VALUES,
    "dify": DIFY_VALUES,
    "model": MODEL_VALUES,
}

# 各字段的初始取值，与 StatusUpdate 的默认值保持一致
DEFAULT_FIELD_VALUES: dict[str, str] = {
    "running": "stopped",
    "audio": "off",
    "asr": "idle",
    "dify": "offline",
    "model": "idle",
}

# 应用名默认值，真实取值来自配置里的 app.name
DEFAULT_APP_NAME = "InterviewMate"

# TTFT 没有数据时的显示文本，见《2-骨架与配置》状态字段取值表
TTFT_EMPTY = "-"

# 错误码的前缀，提示位以它开头就按错误色显示
ERROR_CODE_PREFIX = "E_"

# 模式字段的合法取值与显示文本：项目模式显示「项目」，普通模式显示「普通」
MODE_LABELS: dict[str, str] = {"normal": "普通", "project": "项目"}

# 会话 ID 没有时显示的占位
SESSION_EMPTY = "-"


class StatusUpdate(Event):
    """状态栏更新事件。"""

    type: str = "status.update"
    running: str = "stopped"
    audio: str = "off"
    asr: str = "idle"
    dify: str = "offline"
    model: str = "idle"
    ttft_ms: float | None = None
    hint: str = ""


class StatusBar(Static):
    """顶部状态栏：两行文本。

    第一行是八项状态，渲染格式固定为::

        {应用名} | RUNNING={值} | AUDIO={值} | ASR={值} | DIFY={值} | MODEL={值} | TTFT={值} | MODE={值} | SESSION={值}

    第二行是提示位，放错误码与一句话，没有提示时为空行。

    TTFT 有值时显示整数毫秒数，没有值时显示 ``-``；
    MODE 取 ``MODE_LABELS`` 里的显示文本（普通 / 项目）；
    SESSION 是当前会话 ID，没有会话时显示 ``-``。

    :meth:`render` 返回八项齐全的固定格式；真正画到屏幕上的是
    :meth:`render_line`，它按控件宽度裁剪：放不下就从尾部去掉整项，
    只剩应用名还放不下才截断应用名。窄终端下不会出现被硬裁掉一半的字段。
    """

    def __init__(self, app_name: str = DEFAULT_APP_NAME, **kwargs: Any) -> None:
        # 属性要在 super().__init__ 之前赋值，Textual 允许子类这样做
        self.app_name = app_name
        self._values: dict[str, str] = dict(DEFAULT_FIELD_VALUES)
        self._ttft_ms: float | None = None
        # 第二行的提示文本
        self._hint: str = ""
        # 项目模式字段：normal / project
        self._mode: str = "normal"
        # 当前会话 ID，没有会话时是 None
        self._session_id: str | None = None
        # on_resize 记下的控件宽度，渲染阶段拿不到宽度时用它兜底
        self._width: int = 0
        # 主题里的颜色变量，第一次用到时取一次就缓存起来
        self._theme_colors: dict[str, str | None] | None = None
        # markup=False：状态行是纯文本，不做标记解析
        super().__init__("", markup=False, **kwargs)

    def on_resize(self, event: events.Resize) -> None:
        """记住控件的内容宽度，渲染阶段拿不到宽度时用它兜底。"""
        self._width = self.size.width or event.size.width

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self) -> str:
        """返回状态行文本，六项齐全的固定格式。"""
        return self._plain_line(self._entries())

    def render_line(self, y: int) -> Strip:
        """渲染状态栏的这一行，按控件宽度裁剪后再交给样式层。

        第 0 行是六项状态，第 1 行是提示位。状态行按段着色：应用名加粗，
        标签与分隔符用弱化色，取值用正文色；提示位是错误码时用错误色。
        样式层会把这一行补齐到控件内容宽度，裁剪过的短行不会被截断。
        """
        if y == 0:
            return self._status_strip()
        if y == 1:
            text = self._cut_to_width(self._hint)
            return Strip([Segment(text, self._hint_style())])
        return super().render_line(y)

    def _status_strip(self) -> Strip:
        """把状态行按段着色后拼成一条。"""
        base = self.visual_style.rich_style
        name_style = base + Style(bold=True, color=self._theme_color("text-primary"))
        weak_style = base + Style(color=self._theme_color("text-secondary"))
        entries, name = self._fit_entries(self._entries())
        segments = [Segment(name, name_style)]
        for label, value in entries:
            segments.append(Segment(" | ", weak_style))
            segments.append(Segment(label, weak_style))
            segments.append(Segment("=", weak_style))
            segments.append(Segment(value, base))
        return Strip(segments)

    def _hint_style(self) -> Style:
        """提示位的样式：错误码用错误色，其它提示用弱化色。"""
        base = self.visual_style.rich_style
        if self._hint.startswith(ERROR_CODE_PREFIX):
            return base + Style(color=self._theme_color("text-error"))
        return base + Style(color=self._theme_color("text-secondary"))

    def _entries(self) -> list[tuple[str, str]]:
        """八项状态，返回 (标签, 取值) 列表，顺序按 :data:`STATUS_ORDER`。"""
        entries: list[tuple[str, str]] = []
        for key in STATUS_ORDER:
            if key == "ttft":
                entries.append((STATUS_LABELS[key], self._format_ttft()))
            elif key == "mode":
                entries.append((STATUS_LABELS[key], MODE_LABELS[self._mode]))
            elif key == "session":
                entries.append((STATUS_LABELS[key], self._session_id or SESSION_EMPTY))
            else:
                entries.append((STATUS_LABELS[key], self._values[key]))
        return entries

    def _plain_line(
        self, entries: list[tuple[str, str]], name: str | None = None
    ) -> str:
        """按固定格式拼出状态行文本。

        :param entries: 六项状态。
        :param name: 应用名，不给时用 :attr:`app_name`。
        """
        app_name = self.app_name if name is None else name
        parts = [app_name]
        parts.extend(f"{label}={value}" for label, value in entries)
        return " | ".join(parts)

    def _available_width(self) -> int:
        """可用宽度：优先用控件自己的内容宽度，拿不到就用 on_resize 记下的值。"""
        return self.size.width or self._width

    def _fit_entries(self, entries: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
        """按可用宽度裁剪状态行。

        放不下就从尾部去掉整项（先 TTFT，再 MODEL……），
        只剩应用名还放不下时再截断应用名。

        :param entries: 六项状态。
        :return: 裁剪后的六项与裁剪后的应用名。
        """
        width = self._available_width()
        if width <= 0:
            # 宽度未知，先整行给出，等控件拿到宽度后重画
            return entries, self.app_name
        kept = list(entries)
        while kept and len(self._plain_line(kept)) > width:
            kept.pop()
        name = self.app_name
        if len(self._plain_line(kept, name)) > width:
            # 只剩应用名还放不下，截断应用名
            name = name[:width]
        return kept, name

    def _cut_to_width(self, text: str) -> str:
        """按可用宽度截断一段文本，放不下时末尾留省略号。"""
        width = self._available_width()
        if width <= 0 or cell_len(text) <= width:
            return text
        kept: list[str] = []
        used = 0
        for char in text:
            char_width = cell_len(char)
            if used + char_width > width - 1:
                break
            kept.append(char)
            used += char_width
        return "".join(kept) + "…"

    def status_snapshot(self) -> dict[str, str]:
        """返回八项状态的字符串快照，键顺序与 :data:`STATUS_ORDER` 一致。"""
        snapshot: dict[str, str] = {}
        for key in STATUS_ORDER:
            if key == "ttft":
                snapshot[key] = self._format_ttft()
            elif key == "mode":
                snapshot[key] = MODE_LABELS[self._mode]
            elif key == "session":
                snapshot[key] = self._session_id or SESSION_EMPTY
            else:
                snapshot[key] = self._values[key]
        return snapshot

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------
    def update_status(
        self,
        *,
        running: str = "stopped",
        audio: str = "off",
        asr: str = "idle",
        dify: str = "offline",
        model: str = "idle",
        ttft_ms: float | None = None,
        hint: str = "",
    ) -> None:
        """更新六项状态与提示位。

        取值必须落在《2-骨架与配置》状态字段取值表的取值域里，非法取值抛
        :class:`ValueError`，消息里列出该字段的合法取值。

        :param hint: 提示位文本，一般是错误码加一句话，空串表示没有提示。
        :raises ValueError: 某个字段的取值不合法。
        """
        candidates = {
            "running": running,
            "audio": audio,
            "asr": asr,
            "dify": dify,
            "model": model,
        }
        # 先整体校验再写入，避免非法取值造成状态写了一半
        for field, value in candidates.items():
            allowed = FIELD_VALUES[field]
            if value not in allowed:
                raise ValueError(
                    f"状态字段 {field} 的取值非法：{value!r}，"
                    f"合法取值为 {'、'.join(allowed)}"
                )
        checked_ttft = self._check_ttft(ttft_ms)

        self._values = candidates
        self._ttft_ms = checked_ttft
        self._hint = hint
        self.refresh()

    def set_hint(self, hint: str) -> None:
        """只更新提示位，不动状态项。

        :param hint: 提示位文本，空串表示没有提示。
        """
        self._hint = hint
        self.refresh()

    @property
    def hint(self) -> str:
        """当前提示位文本。"""
        return self._hint

    def set_mode(self, mode: str) -> None:
        """更新模式字段：项目模式是 ``project``，普通模式是 ``normal``。

        :param mode: 模式取值。
        :raises ValueError: 取值不在 :data:`MODE_LABELS` 里。
        """
        if mode not in MODE_LABELS:
            raise ValueError(
                f"状态字段 mode 的取值非法：{mode!r}，"
                f"合法取值为 {'、'.join(MODE_LABELS)}"
            )
        self._mode = mode
        self.refresh()

    @property
    def mode(self) -> str:
        """当前模式取值：``normal`` 或 ``project``。"""
        return self._mode

    def set_session(self, session_id: str | None) -> None:
        """更新当前会话 ID。

        :param session_id: 会话 ID，None 或空串按没有会话处理，显示占位符。
        """
        self._session_id = (session_id or "").strip() or None
        self.refresh()

    @property
    def session_id(self) -> str | None:
        """当前会话 ID，没有会话时是 None。"""
        return self._session_id

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _theme_color(self, name: str) -> str | None:
        """取主题里的颜色变量。

        取不到具体颜色（例如值是 ``auto 87%`` 这种对比色）时返回 None，
        调用方退回样式本身的颜色。

        :param name: 颜色变量名，例如 ``text-primary``。
        """
        if self._theme_colors is None:
            try:
                variables = self.app.get_css_variables()
            except Exception:
                variables = {}
            self._theme_colors = dict(variables)
        value = self._theme_colors.get(name)
        if isinstance(value, str) and value.startswith("#"):
            return value
        return None

    @staticmethod
    def _check_ttft(ttft_ms: float | None) -> float | None:
        """校验 TTFT 取值：None 或非负毫秒数。"""
        if ttft_ms is None:
            return None
        if isinstance(ttft_ms, bool) or not isinstance(ttft_ms, (int, float)):
            raise ValueError(
                f"状态字段 ttft_ms 的取值非法：{ttft_ms!r}，"
                "合法取值为 None 或非负毫秒数"
            )
        if ttft_ms < 0:
            raise ValueError(
                f"状态字段 ttft_ms 的取值非法：{ttft_ms!r}，"
                "合法取值为 None 或非负毫秒数"
            )
        return float(ttft_ms)

    def _format_ttft(self) -> str:
        """TTFT 的显示文本：整数毫秒数，没有数据时是 ``-``。"""
        if self._ttft_ms is None:
            return TTFT_EMPTY
        return str(int(self._ttft_ms))
