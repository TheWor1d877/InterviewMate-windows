"""项目模式对话日志的测试。

覆盖：会话文件创建与命名、追加格式、角色校验、空句忽略、换行压平、
问题记号、recent_text 的顺序与行边界截断、full_text、clear 封存重开、
未开会话时 append 自动开会话、同秒重开的文件名去重。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.dialogue.dialogue_log import (
    QUESTION_MARK_PREFIX,
    ROLE_INTERVIEWER,
    ROLE_SELF,
    DialogueLog,
)

# 会话 ID 的取值形状：YYYYMMDD-HHMMSS，同秒重开时再加 -N 序号
SESSION_ID_PATTERN = re.compile(r"^\d{8}-\d{6}(?:-\d+)?$")

# 行格式：{HH:MM:SS} [{角色}] {句子}
SENTENCE_LINE_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2} \[(.+?)\] (.+)$")


def _make_log(tmp_path: Path) -> DialogueLog:
    """建一个写到临时目录的对话日志。"""
    return DialogueLog(log_dir=tmp_path / "dialogues")


def _session_files(log: DialogueLog) -> list[Path]:
    """列出日志目录下的会话文件。"""
    assert log.session_path is not None
    return sorted(log._log_dir.glob("*.md"))  # noqa: SLF001 - 测试读内部目录


def test_start_session_creates_file_and_returns_id(tmp_path: Path) -> None:
    """start_session 建会话文件并返回会话 ID。"""
    log = _make_log(tmp_path)
    session_id = log.start_session()

    assert SESSION_ID_PATTERN.match(session_id)
    assert log.is_active()
    assert log.session_id == session_id
    assert log.session_path is not None
    assert log.session_path.name == f"{session_id}.md"
    assert log.session_path.exists()
    # 文件头部一行标题
    head = log.session_path.read_text(encoding="utf-8").splitlines()[0]
    assert head == f"# 对话记录 {session_id}"


def test_is_active_false_before_start(tmp_path: Path) -> None:
    """没开会话时 is_active 是假，session_id 与路径都是 None。"""
    log = _make_log(tmp_path)

    assert not log.is_active()
    assert log.session_id is None
    assert log.session_path is None


def test_append_writes_role_lines(tmp_path: Path) -> None:
    """append 按角色一行一条写入，格式是 ``时间 [角色] 句子``。"""
    log = _make_log(tmp_path)
    log.start_session()
    log.append(ROLE_SELF, "面试官你好")
    log.append(ROLE_INTERVIEWER, "先做个自我介绍")

    path = log.session_path
    assert path is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # 头部一行 + 两条对话行
    match = SENTENCE_LINE_PATTERN.match(lines[1])
    assert match is not None and match.group(1) == ROLE_SELF
    assert match is not None and match.group(2) == "面试官你好"
    match = SENTENCE_LINE_PATTERN.match(lines[2])
    assert match is not None and match.group(1) == ROLE_INTERVIEWER
    # full_text 按写入顺序排，不含文件头部
    assert log.full_text.splitlines() == lines[1:]


def test_append_rejects_unknown_role(tmp_path: Path) -> None:
    """角色不在合法取值里时抛 ValueError，不写行。"""
    log = _make_log(tmp_path)
    log.start_session()

    with pytest.raises(ValueError):
        log.append("考官", "拼写错了")
    assert log.full_text == ""


def test_append_ignores_blank_text(tmp_path: Path) -> None:
    """压平换行后只剩空白的句子不落库。"""
    log = _make_log(tmp_path)
    log.start_session()
    log.append(ROLE_SELF, "   ")
    log.append(ROLE_SELF, "")

    assert log.full_text == ""


def test_append_flattens_newlines(tmp_path: Path) -> None:
    """句子里的换行压成空格，保证一行一条。"""
    log = _make_log(tmp_path)
    log.start_session()
    log.append(ROLE_SELF, "第一行\n第二行\r\n第三行")

    lines = log.full_text.splitlines()
    assert len(lines) == 1
    assert "第一行 第二行 第三行" in lines[0]


def test_append_auto_starts_session(tmp_path: Path) -> None:
    """会话未开启时 append 自动开会话，句子不丢。"""
    log = _make_log(tmp_path)
    log.append(ROLE_SELF, "还没开会话就说话")

    assert log.is_active()
    assert "还没开会话就说话" in log.full_text


def test_mark_question_writes_marker_line(tmp_path: Path) -> None:
    """mark_question 追加 ``**[问题]**`` 记号行。"""
    log = _make_log(tmp_path)
    log.start_session()
    log.mark_question("介绍一下你的项目经验")

    lines = log.full_text.splitlines()
    assert len(lines) == 1
    assert lines[0].endswith(f"{QUESTION_MARK_PREFIX} 介绍一下你的项目经验")
    assert QUESTION_MARK_PREFIX in log.session_path.read_text(encoding="utf-8")


def test_mark_question_ignores_blank(tmp_path: Path) -> None:
    """空白问题不落库。"""
    log = _make_log(tmp_path)
    log.start_session()
    log.mark_question("  ")

    assert log.full_text == ""


def test_recent_text_returns_all_within_limit(tmp_path: Path) -> None:
    """上限足够大时 recent_text 返回全部内容，顺序不变。"""
    log = _make_log(tmp_path)
    log.start_session()
    log.append(ROLE_SELF, "第一句")
    log.append(ROLE_INTERVIEWER, "第二句")
    log.append(ROLE_SELF, "第三句")

    assert log.recent_text(8000) == log.full_text


def test_recent_text_truncates_on_line_boundary(tmp_path: Path) -> None:
    """上限不够时从尾部往前收完整的行，行边界对齐、顺序不变。"""
    log = _make_log(tmp_path)
    log.start_session()
    log.append(ROLE_SELF, "一" * 10)
    log.append(ROLE_INTERVIEWER, "二" * 8)
    log.append(ROLE_SELF, "三" * 6)

    stored = log.full_text.splitlines()
    full = log.full_text
    # 上限比全文少一个字符：最早的那行装不下，其余的都要装下
    recent = log.recent_text(len(full) - 1)

    assert recent == "\n".join(stored[1:])
    assert len(recent) <= len(full) - 1
    # 上限回到全文长度时三行都收进来
    assert log.recent_text(len(full)) == full


def test_recent_text_single_oversized_line_tail(tmp_path: Path) -> None:
    """最近一行自己超限时截它的尾部，保证至少有内容。"""
    log = _make_log(tmp_path)
    log.start_session()
    long_text = "字" * 300
    log.append(ROLE_SELF, long_text)

    recent = log.recent_text(50)
    assert len(recent) == 50
    assert recent == long_text[-50:]


def test_recent_text_empty_or_zero_limit(tmp_path: Path) -> None:
    """空会话与非正上限时返回空串。"""
    log = _make_log(tmp_path)
    assert log.recent_text(8000) == ""
    log.start_session()
    assert log.recent_text(8000) == ""
    log.append(ROLE_SELF, "有内容")
    assert log.recent_text(0) == ""
    assert log.recent_text(-1) == ""


def test_clear_keeps_old_file_and_starts_new(tmp_path: Path) -> None:
    """clear 封存旧文件并重开一轮，返回新会话 ID。"""
    log = _make_log(tmp_path)
    first_id = log.start_session()
    log.append(ROLE_SELF, "旧会话的句子")
    old_path = log.session_path

    new_id = log.clear()

    assert new_id != first_id
    assert SESSION_ID_PATTERN.match(new_id)
    assert log.session_id == new_id
    assert log.session_path != old_path
    # 旧文件原样保留
    assert old_path is not None and old_path.exists()
    assert "旧会话的句子" in old_path.read_text(encoding="utf-8")
    # 新会话从空白开始
    assert log.full_text == ""
    assert len(_session_files(log)) == 2


def test_start_session_dedups_same_second_names(tmp_path: Path) -> None:
    """同一秒内连续重开时文件名加序号，互不覆盖。"""
    log = _make_log(tmp_path)
    ids = [log.start_session() for _ in range(10)]

    assert len(set(ids)) == 10
    assert all(SESSION_ID_PATTERN.match(session_id) for session_id in ids)
    assert len(_session_files(log)) == 10