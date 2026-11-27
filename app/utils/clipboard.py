"""读取系统剪贴板。

Wayland 优先用 ``wl-paste``，X11 与 XWayland 用 ``xclip``。
命令不存在、超时或返回非零时当作不可用，换下一条命令；
全部不可用时返回空串，由调用方退回终端原生粘贴提示。
"""

from __future__ import annotations

import subprocess

from app.utils.logger import log_event

__all__ = ["read_clipboard", "CLIPBOARD_TIMEOUT"]


# 单条命令的等待上限（秒）：剪贴板读取只做一次尝试，不能拖住按键响应
CLIPBOARD_TIMEOUT: float = 1.0

# 普通剪贴板与 PRIMARY 选区的中键粘贴，各给一组候选命令
_CLIPBOARD_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("wl-paste", "--no-newline"),
    ("xclip", "-selection", "clipboard", "-o"),
)
_PRIMARY_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("wl-paste", "--primary", "--no-newline"),
    ("xclip", "-selection", "primary", "-o"),
)


def read_clipboard(*, primary: bool = False) -> str:
    """读一次剪贴板文本，读不到时返回空串。

    :param primary: True 读 PRIMARY 选区（鼠标中键粘贴用），False 读普通剪贴板。
    """
    commands = _PRIMARY_COMMANDS if primary else _CLIPBOARD_COMMANDS
    for command in commands:
        text = _run(command)
        if text is not None:
            return text
        log_event(
            "clipboard.command_failed",
            "剪贴板命令不可用，换下一条",
            level="DEBUG",
            command=command[0],
        )
    return ""


def _run(command: tuple[str, ...]) -> str | None:
    """跑一条命令读剪贴板，命令不可用时返回 None，可用但为空时返回空串。

    ``stdout`` 按字节取回再解码：选中图片等二进制内容时输出不是合法 UTF-8，
    交给 ``text=True`` 解码会抛 ``UnicodeDecodeError``，异常会穿出调用方的
    线程封装让粘贴直接失效，所以这里用 ``errors="replace"`` 兜住。
    """
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=CLIPBOARD_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").removesuffix("\n")