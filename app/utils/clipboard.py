"""读写系统剪贴板（Windows）。

读取用 PowerShell 的 ``Get-Clipboard -Raw``，写入用 ``Set-Clipboard``。
命令不存在、超时或返回非零时按不可用处理：读取返回空串，
由调用方退回终端原生粘贴提示；写入返回 False，不抛未处理异常。

Windows 剪贴板没有 PRIMARY 选区，``read_clipboard(primary=True)``
为了保留既有调用契约仍然接受这个参数，实际按普通剪贴板读取。
"""

from __future__ import annotations

import subprocess

from app.utils.logger import log_event

__all__ = ["read_clipboard", "write_clipboard", "CLIPBOARD_TIMEOUT"]


# 单条命令的等待上限（秒）：PowerShell 冷启动偏慢，
# 给到 3 秒，调用方仍在后台线程里等，不阻塞按键响应。
CLIPBOARD_TIMEOUT: float = 3.0

# PowerShell 可执行文件
_POWERSHELL: str = "powershell.exe"

# 读取脚本：显式把输出编码设成 UTF-8，避免中文按系统代码页解码成乱码
_READ_SCRIPT: str = (
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
    "Get-Clipboard -Raw"
)

# 写入脚本：从标准输入读 UTF-8 文本，再一次性设置剪贴板，
# 不把文本拼进命令行，避免引号与特殊字符被 PowerShell 解释
_WRITE_SCRIPT: str = (
    "[Console]::InputEncoding=[System.Text.Encoding]::UTF8; "
    "Set-Clipboard -Value ([Console]::In.ReadToEnd())"
)

_READ_COMMAND: tuple[str, ...] = (
    _POWERSHELL,
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    _READ_SCRIPT,
)

_WRITE_COMMAND: tuple[str, ...] = (
    _POWERSHELL,
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    _WRITE_SCRIPT,
)


def read_clipboard(*, primary: bool = False) -> str:
    """读一次剪贴板文本，读不到时返回空串。

    :param primary: 保留既有调用契约。Windows 没有 PRIMARY 选区，
        这个参数为 True 时也按普通剪贴板读取。
    """
    if primary:
        log_event(
            "clipboard.primary_unsupported",
            "Windows 没有 PRIMARY 选区，按普通剪贴板读取",
            level="DEBUG",
        )
    text = _run(_READ_COMMAND)
    if text is None:
        log_event(
            "clipboard.command_failed",
            "剪贴板命令不可用",
            level="DEBUG",
            command=_POWERSHELL,
        )
        return ""
    return text


def write_clipboard(text: str) -> bool:
    """把文本写进系统剪贴板。

    :param text: 要写入的文本。
    :returns: 写入成功返回 True；命令不可用或返回非零时返回 False。
    """
    try:
        completed = subprocess.run(
            _WRITE_COMMAND,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=CLIPBOARD_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is None or completed.returncode != 0:
        log_event(
            "clipboard.write_failed",
            "剪贴板写入失败",
            level="DEBUG",
            command=_POWERSHELL,
        )
        return False
    return True


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