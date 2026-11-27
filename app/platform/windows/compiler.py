"""Windows 侧的代码编译器探测：``cl.exe`` 或 MinGW-w64 的 ``g++``。

Windows 侧优先用 Visual Studio 的 ``cl.exe``，没有时退回 MinGW-w64 的
``g++``。
两者的调用方式不同，上层拿 :meth:`MsvcCompiler.command` 返回的完整命令即可。

探测顺序：

1. ``PATH`` 里的 ``cl.exe``；
2. 用 Visual Studio 的 ``vswhere.exe`` 找安装目录，再取最新版本
   ``VC\\Tools\\MSVC\\*\\bin\\Host*\\*\\cl.exe``；
3. ``PATH`` 里的 ``g++``（MinGW-w64）。

都没有时 :meth:`MsvcCompiler.available` 返回 False，
:meth:`MsvcCompiler.unavailable_reason` 给出一句中文说明；
:meth:`MsvcCompiler.command` 仍返回默认的 ``cl.exe``，保持既有接口不变。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

__all__ = ["MsvcCompiler"]


# 默认编译命令，与既有接口的返回值一致
DEFAULT_COMMAND: str = "cl.exe"

# 探测不到编译器时的说明，不抛异常，由调用方按 available() 决定后续动作
UNAVAILABLE_REASON: str = (
    "没有找到 cl.exe（PATH 或 Visual Studio 安装）或 MinGW-w64 的 g++"
)


def _find_cl() -> str | None:
    """找 ``cl.exe``：先 PATH，再 Visual Studio 安装目录。"""
    found = shutil.which("cl.exe")
    if found:
        return found

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None

    try:
        completed = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    install_path = completed.stdout.strip()
    if not install_path:
        return None

    msvc_root = Path(install_path) / "VC" / "Tools" / "MSVC"
    if not msvc_root.is_dir():
        return None
    # 版本号形如 14.40.33807，按字符串倒序取最新的一个
    for candidate in sorted(msvc_root.glob("*/bin/Host*/*/cl.exe"), reverse=True):
        if candidate.is_file():
            return str(candidate)
    return None


def _find_gpp() -> str | None:
    """找 MinGW-w64 的 ``g++``。"""
    for name in ("g++.exe", "g++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def detect_command() -> str | None:
    """探测可用的编译命令，返回完整路径；都没有时返回 None。"""
    return _find_cl() or _find_gpp()


class MsvcCompiler:
    """Windows 侧编译器后端：``cl.exe`` 优先，MinGW-w64 ``g++`` 兜底。"""

    # 后端名
    name: str = "msvc"

    # 编译器是系统自带的，不需要额外第三方包
    required_packages: tuple[str, ...] = ()

    def __init__(self) -> None:
        # 构造时探测一次，后续 command() 与 available() 用同一个结果
        self._command: str | None = detect_command()

    def command(self) -> str:
        """返回编译命令。

        :returns: 探测到的完整命令；都探测不到时返回默认的 ``cl.exe``，
            此时 :meth:`available` 为 False。
        """
        return self._command or DEFAULT_COMMAND

    def available(self) -> bool:
        """编译器是否可用。

        :returns: 找到 ``cl.exe`` 或 ``g++`` 返回 True。
        """
        return self._command is not None

    def unavailable_reason(self) -> str:
        """编译器不可用时的中文说明，可用时返回空串。"""
        if self._command is not None:
            return ""
        return UNAVAILABLE_REASON