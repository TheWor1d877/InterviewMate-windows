"""项目模式的对话日志：按句落库的会话记录。

双流转写出的文字按句写进 ``logs/dialogues/{session_id}.md``，一行一条，
格式是 ``{HH:MM:SS} [{角色}] {句子}``，角色取「我」与「面试官」；问题句
由 F8 截取窗口触发追加，行格式是 ``{HH:MM:SS} **[问题]** {句子}``。

会话 ID 取本地时间 ``YYYYMMDD-HHMMSS``；同一秒内重开时在文件名上加
``-2``、``-3`` 序号区分，保证每个会话一个文件、互不覆盖。

``clear`` 只封存不删除：当前会话文件原样保留，另开一个新会话文件，
返回新会话 ID。

内存里留一份按写入顺序排的行缓存：

- :attr:`full_text` 返回缓存全文，供界面的上下文视图用；
- :meth:`recent_text` 按时间序从尾部取 ``max_chars`` 个字符，在行边界
  截断，不重排，喂给前缀缓存类模型的上下文时顺序稳定。

写文件用追加模式：落库由句末标点或时间窗口触发，秒级频率，
每句开关一次文件足够，不需要常驻句柄。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.utils.logger import REPO_ROOT, log_event

__all__ = [
    "ROLE_SELF",
    "ROLE_INTERVIEWER",
    "ROLES",
    "QUESTION_MARK_PREFIX",
    "DEFAULT_DIALOGUE_LOG_DIR",
    "DialogueLog",
]


# 麦克风一路的固定角色：考生本人
ROLE_SELF: str = "我"

# 回环一路的固定角色：面试官
ROLE_INTERVIEWER: str = "面试官"

# 角色的合法取值
ROLES: tuple[str, ...] = (ROLE_SELF, ROLE_INTERVIEWER)

# 问题句的记号前缀，行格式是 ``{HH:MM:SS} **[问题]** {句子}``
QUESTION_MARK_PREFIX: str = "**[问题]**"

# 默认日志目录：logs/dialogues 已被 gitignore，见《2-骨架与配置》日志一节
DEFAULT_DIALOGUE_LOG_DIR: Path = REPO_ROOT / "logs" / "dialogues"


def _clean_text(text: str) -> str:
    """压平一句话里的换行并去首尾空白。

    对话日志一行一条，句子中间的换行会破坏行格式，压成空格。

    :param text: 原始句子。
    :returns: 清理后的句子，只剩空白时是空串。
    """
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return cleaned.strip()


class DialogueLog:
    """对话日志：每个会话一个 md 文件，按句追加。

    :param log_dir: 日志目录，默认 ``logs/dialogues``。
    """

    def __init__(self, log_dir: str | Path = DEFAULT_DIALOGUE_LOG_DIR) -> None:
        self._log_dir = Path(log_dir)
        self._session_id: str | None = None
        self._session_path: Path | None = None
        # 按写入顺序排的行缓存，recent_text 与 full_text 都从这里出
        self._lines: list[str] = []

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def session_id(self) -> str | None:
        """当前会话 ID，未开会话时是 None。"""
        return self._session_id

    @property
    def session_path(self) -> Path | None:
        """当前会话文件路径，未开会话时是 None。

        契约注解写 ``Path``；这里放宽成 ``Path | None``，与
        :attr:`session_id` 的可空语义对称：未开会话时没有路径可返回，
        集成层配合 :meth:`is_active` 使用。
        """
        return self._session_path

    @property
    def full_text(self) -> str:
        """当前会话的全部对话行，按时间序，界面上下文视图用。"""
        return "\n".join(self._lines)

    def is_active(self) -> bool:
        """当前有没有开着会话。"""
        return self._session_path is not None

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------
    def start_session(self) -> str:
        """新建会话文件，之后的句子都追加到它，返回会话 ID。

        :returns: 会话 ID，形如 ``20261128-153000``。
        """
        self._log_dir.mkdir(parents=True, exist_ok=True)
        base = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = base
        serial = 1
        # 同一秒内重开时加序号，保证每个会话一个文件、互不覆盖
        while (self._log_dir / f"{name}.md").exists():
            serial += 1
            name = f"{base}-{serial}"
        path = self._log_dir / f"{name}.md"
        path.write_text(f"# 对话记录 {name}\n", encoding="utf-8")
        self._session_id = name
        self._session_path = path
        self._lines = []
        log_event(
            "dialogue.session_started",
            "开始新的对话会话",
            session_id=name,
            path=str(path),
        )
        return name

    def clear(self) -> str:
        """封存当前会话并重开一轮，返回新会话 ID。

        当前会话文件原样保留，只另开新文件；行缓存随之清空。

        :returns: 新会话 ID。
        """
        session_id = self.start_session()
        log_event(
            "dialogue.session_cleared",
            "对话日志已封存，重开一轮",
            session_id=session_id,
        )
        return session_id

    # ------------------------------------------------------------------
    # 落库
    # ------------------------------------------------------------------
    def append(self, role: str, text: str) -> None:
        """落库一句话，一行一条：``{HH:MM:SS} [{role}] {句子}``。

        会话未开启时自动开一个，句子不因忘开会话而丢失。

        :param role: 说话角色，取 :data:`ROLES` 里的一个。
        :param text: 句子文本，压平换行后只剩空白时忽略。
        :raises ValueError: 角色不合法。
        """
        if role not in ROLES:
            legal = "、".join(ROLES)
            raise ValueError(f"未知的对话角色：{role!r}，合法取值：{legal}")
        cleaned = _clean_text(text)
        if not cleaned:
            return
        self._ensure_session()
        self._write(f"{self._now()} [{role}] {cleaned}")
        log_event(
            "dialogue.sentence_logged",
            "对话句子已落库",
            role=role,
            session_id=self._session_id or "",
        )

    def mark_question(self, text: str) -> None:
        """把一句话标记成问题，行格式：``{HH:MM:SS} **[问题]** {句子}``。

        :param text: 问题文本，压平换行后只剩空白时忽略。
        """
        cleaned = _clean_text(text)
        if not cleaned:
            return
        self._ensure_session()
        self._write(f"{self._now()} {QUESTION_MARK_PREFIX} {cleaned}")
        log_event(
            "dialogue.question_marked",
            "对话句子已标记为问题",
            session_id=self._session_id or "",
        )

    def recent_text(self, max_chars: int = 8000) -> str:
        """按时间序返回最近 ``max_chars`` 个字符的上下文全文。

        从行缓存尾部往前收集完整的行，在行边界截断，不重排，
        喂给前缀缓存类模型时顺序稳定。最近一行自己超限时取它的尾部，
        保证至少有内容。

        :param max_chars: 字符数上限，非正数时返回空串。
        :returns: 最近上下文全文，按时间序（早 → 晚）。
        """
        if max_chars <= 0 or not self._lines:
            return ""
        selected: list[str] = []
        total = 0
        for line in reversed(self._lines):
            need = len(line) + (1 if selected else 0)
            if total + need > max_chars:
                if not selected and need > max_chars:
                    # 最近的行自己超限：截它的尾部，保证至少有内容
                    selected.append(line[-max_chars:])
                break
            selected.append(line)
            total += need
        selected.reverse()
        return "\n".join(selected)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _ensure_session(self) -> None:
        """没有会话时自动开一个。"""
        if self.is_active():
            return
        log_event(
            "dialogue.session_autostarted",
            "会话未开启，落库前自动开启新会话",
            level="DEBUG",
        )
        self.start_session()

    def _write(self, line: str) -> None:
        """追加一行到文件与行缓存。"""
        self._lines.append(line)
        path = self._session_path
        if path is None:
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    @staticmethod
    def _now() -> str:
        """行首的时间戳，取本地时间。"""
        return datetime.now().strftime("%H:%M:%S")