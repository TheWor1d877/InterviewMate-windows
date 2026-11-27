"""系统剪贴板读取的单元测试。

``app.utils.clipboard`` 读剪贴板靠子进程，测试里把
``app.utils.clipboard.subprocess.run`` 换成假实现，按命令名分发返回值或异常，
不真的调用 ``wl-paste`` / ``xclip``。

覆盖：候选命令回落、全部不可用返回空串、PRIMARY 选区走另一组命令、
非 UTF-8 字节不抛异常、换行归一与末尾换行只去掉一个。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.utils.clipboard import read_clipboard


class _Completed:
    """假的 ``CompletedProcess``：只需要 ``returncode`` 与 ``stdout`` 两个字段。"""

    def __init__(self, returncode: int, stdout: bytes) -> None:
        self.returncode = returncode
        self.stdout = stdout


def _install_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    handlers: dict[str, Any],
) -> list[tuple[str, ...]]:
    """按命令名装假实现，返回被调用过的命令元组列表。

    ``handlers`` 的值可以是 ``bytes``（返回码 0 的输出）、
    :class:`_Completed`（自定义返回码）或异常实例（调用时抛出）。
    """
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> _Completed:
        calls.append(tuple(command))
        handler = handlers[command[0]]
        if isinstance(handler, BaseException):
            raise handler
        if isinstance(handler, bytes):
            return _Completed(0, handler)
        return handler

    monkeypatch.setattr("app.utils.clipboard.subprocess.run", fake_run)
    return calls


def test_falls_back_to_xclip_when_wl_paste_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wl-paste 不存在时回落 xclip，并返回 xclip 的文本。"""
    calls = _install_fake_run(
        monkeypatch,
        {
            "wl-paste": FileNotFoundError("wl-paste 不存在"),
            "xclip": "xclip 内容".encode("utf-8"),
        },
    )

    assert read_clipboard() == "xclip 内容"
    assert calls == [
        ("wl-paste", "--no-newline"),
        ("xclip", "-selection", "clipboard", "-o"),
    ]


def test_falls_back_to_xclip_when_wl_paste_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wl-paste 返回非零当作不可用，回落 xclip。"""
    calls = _install_fake_run(
        monkeypatch,
        {
            "wl-paste": _Completed(1, b"ignored"),
            "xclip": "xclip 内容".encode("utf-8"),
        },
    )

    assert read_clipboard() == "xclip 内容"
    assert [call[0] for call in calls] == ["wl-paste", "xclip"]


def test_returns_empty_when_all_commands_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两条候选命令都不可用时返回空串。"""
    calls = _install_fake_run(
        monkeypatch,
        {
            "wl-paste": FileNotFoundError("wl-paste 不存在"),
            "xclip": _Completed(1, b""),
        },
    )

    assert read_clipboard() == ""
    assert len(calls) == 2


def test_command_failure_logs_debug_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每条失败的候选命令补一条 DEBUG 日志，字段带命令名。"""
    events: list[tuple[str, dict[str, Any]]] = []

    def fake_log_event(event: str, message: str = "", **kwargs: Any) -> None:
        events.append((event, {"message": message, **kwargs}))

    monkeypatch.setattr("app.utils.clipboard.log_event", fake_log_event)
    _install_fake_run(
        monkeypatch,
        {
            "wl-paste": FileNotFoundError("wl-paste 不存在"),
            "xclip": _Completed(1, b""),
        },
    )

    read_clipboard()

    assert events == [
        (
            "clipboard.command_failed",
            {"message": "剪贴板命令不可用，换下一条", "level": "DEBUG", "command": "wl-paste"},
        ),
        (
            "clipboard.command_failed",
            {"message": "剪贴板命令不可用，换下一条", "level": "DEBUG", "command": "xclip"},
        ),
    ]


def test_primary_uses_primary_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """primary=True 时用带 --primary / primary 选区的那组命令。"""
    calls = _install_fake_run(
        monkeypatch,
        {
            "wl-paste": FileNotFoundError("wl-paste 不存在"),
            "xclip": "选区内容".encode("utf-8"),
        },
    )

    assert read_clipboard(primary=True) == "选区内容"
    assert calls == [
        ("wl-paste", "--primary", "--no-newline"),
        ("xclip", "-selection", "primary", "-o"),
    ]


def test_non_utf8_bytes_are_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 UTF-8 字节不抛异常，替换字符之外的可读部分保留。"""
    raw = b"\xff\xfeabc"
    _install_fake_run(monkeypatch, {"wl-paste": raw})

    text = read_clipboard()

    assert text == raw.decode("utf-8", errors="replace")
    assert "abc" in text


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (b"a\r\nb\n", "a\nb"),
        (b"abc\n", "abc"),
        (b"a\n\n", "a\n"),
        (b"\n", ""),
        (b"", ""),
    ],
)
def test_newline_normalization(
    monkeypatch: pytest.MonkeyPatch, stdout: bytes, expected: str
) -> None:
    """CRLF 归一成 LF，末尾只去掉一个换行。"""
    _install_fake_run(monkeypatch, {"wl-paste": stdout})

    assert read_clipboard() == expected