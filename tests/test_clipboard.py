"""系统剪贴板读写的单元测试（Windows PowerShell 方案）。

``app.utils.clipboard`` 靠 PowerShell 子进程读写剪贴板，测试里把
``app.utils.clipboard.subprocess.run`` 换成假实现，按命令脚本分发返回值或异常，
不真的调用 ``Get-Clipboard`` / ``Set-Clipboard``。

覆盖：读取命令形态、PRIMARY 参数保留但按普通剪贴板读取、命令不可用时返回空串、
非 UTF-8 字节不抛异常、换行归一、写入命令与标准输入、写入失败的降级语义。
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from app.utils.clipboard import (
    CLIPBOARD_TIMEOUT,
    read_clipboard,
    write_clipboard,
)


class _Completed:
    """假的 ``CompletedProcess``：只需要 ``returncode`` 与 ``stdout`` 两个字段。"""

    def __init__(self, returncode: int, stdout: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout


# 调用记录：命令行元组与传给 subprocess.run 的关键字参数
Call = tuple[tuple[str, ...], dict[str, Any]]


def _install_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> list[Call]:
    """装假 ``subprocess.run``，返回调用记录。

    ``handler`` 可以是 :class:`_Completed`、``bytes``（返回码 0 的输出）
    或异常实例；也可以是接收 ``(command, kwargs)`` 的函数，
    便于按读取/写入脚本分别返回不同结果。
    """
    calls: list[Call] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> _Completed:
        calls.append((tuple(command), dict(kwargs)))
        if callable(handler) and not isinstance(handler, BaseException):
            result = handler(command, kwargs)
        else:
            result = handler
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, bytes):
            return _Completed(0, result)
        return result

    monkeypatch.setattr("app.utils.clipboard.subprocess.run", fake_run)
    return calls


def _script(command: tuple[str, ...]) -> str:
    """取 PowerShell 的 ``-Command`` 脚本。"""
    return command[-1]


def test_read_uses_powershell_get_clipboard_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """读取用 PowerShell ``Get-Clipboard -Raw``，并在后台超时内完成。"""
    calls = _install_fake_run(monkeypatch, "剪贴板内容".encode("utf-8"))

    assert read_clipboard() == "剪贴板内容"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:4] == (
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
    )
    assert "Get-Clipboard -Raw" in _script(command)
    assert "OutputEncoding" in _script(command)
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == CLIPBOARD_TIMEOUT
    assert "input" not in kwargs


def test_primary_reads_normal_clipboard_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows 没有 PRIMARY 选区，primary=True 仍走同一组读取命令。"""
    calls = _install_fake_run(monkeypatch, b"same")

    assert read_clipboard(primary=True) == "same"
    assert len(calls) == 1
    assert "Get-Clipboard -Raw" in _script(calls[0][0])


def test_returns_empty_when_command_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """命令返回非零时当作不可用，返回空串。"""
    _install_fake_run(monkeypatch, _Completed(1, b"ignored"))

    assert read_clipboard() == ""


def test_returns_empty_when_powershell_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PowerShell 不在 PATH 时返回空串，不抛异常。"""
    _install_fake_run(monkeypatch, FileNotFoundError("powershell.exe 不存在"))

    assert read_clipboard() == ""


def test_returns_empty_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """PowerShell 超时时返回空串，不把异常交给调用方。"""
    _install_fake_run(
        monkeypatch,
        subprocess.TimeoutExpired(cmd="powershell.exe", timeout=CLIPBOARD_TIMEOUT),
    )

    assert read_clipboard() == ""


def test_command_failure_logs_debug_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """读取失败补一条 DEBUG 日志，字段带命令名。"""
    events: list[tuple[str, dict[str, Any]]] = []

    def fake_log_event(event: str, message: str = "", **kwargs: Any) -> None:
        events.append((event, {"message": message, **kwargs}))

    monkeypatch.setattr("app.utils.clipboard.log_event", fake_log_event)
    _install_fake_run(monkeypatch, _Completed(1, b""))

    read_clipboard()

    assert events == [
        (
            "clipboard.command_failed",
            {
                "message": "剪贴板命令不可用",
                "level": "DEBUG",
                "command": "powershell.exe",
            },
        )
    ]


def test_non_utf8_bytes_are_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 UTF-8 字节不抛异常，替换字符之外的可读部分保留。"""
    raw = b"\xff\xfeabc"
    _install_fake_run(monkeypatch, raw)

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
    _install_fake_run(monkeypatch, stdout)

    assert read_clipboard() == expected


# ----------------------------------------------------------------------
# 写入
# ----------------------------------------------------------------------


def test_write_uses_powershell_set_clipboard_and_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写入用 ``Set-Clipboard``，文本经标准输入以 UTF-8 传入。"""
    calls = _install_fake_run(monkeypatch, _Completed(0))

    assert write_clipboard("多行\n内容") is True

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:4] == (
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
    )
    assert "Set-Clipboard" in _script(command)
    assert "InputEncoding" in _script(command)
    assert kwargs["input"] == "多行\n内容".encode("utf-8")
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == CLIPBOARD_TIMEOUT


def test_write_returns_false_on_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写入命令返回非零时返回 False，不抛异常。"""
    _install_fake_run(monkeypatch, _Completed(1))

    assert write_clipboard("内容") is False


def test_write_returns_false_when_command_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PowerShell 不可用时写入返回 False。"""
    _install_fake_run(monkeypatch, FileNotFoundError("powershell.exe 不存在"))

    assert write_clipboard("内容") is False


def test_write_failure_logs_debug_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写入失败补一条 DEBUG 日志，字段带命令名。"""
    events: list[tuple[str, dict[str, Any]]] = []

    def fake_log_event(event: str, message: str = "", **kwargs: Any) -> None:
        events.append((event, {"message": message, **kwargs}))

    monkeypatch.setattr("app.utils.clipboard.log_event", fake_log_event)
    _install_fake_run(monkeypatch, _Completed(1))

    assert write_clipboard("内容") is False
    assert events == [
        (
            "clipboard.write_failed",
            {
                "message": "剪贴板写入失败",
                "level": "DEBUG",
                "command": "powershell.exe",
            },
        )
    ]