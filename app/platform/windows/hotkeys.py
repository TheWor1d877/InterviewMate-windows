"""AutoHotkey v2 全局热键提供者（Windows）。

终端失焦后的 F8/F9 由 ``scripts/windows/hotkeys.ahk`` 接管，脚本自己
注册按键，命中后通过 ``python -m app.cli hotkey voice-toggle|screenshot``
通知主进程。本提供者不注册也不注入任何按键，只回答一个问题：
「AutoHotkey 是否已经在运行仓库里的 hotkeys.ahk」。

探测采用进程名与脚本路径双重校验：进程名必须像 AutoHotkey，且命令行中
必须出现本仓库 hotkeys.ahk 的绝对路径。进程存在但读不到命令行时按不可用
处理——宁可退回界面内按键，也不能因为误判让界面错误地短路 F8/F9。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from app.inputs.global_hotkeys import NullHotkeyProvider

__all__ = ["AhkHotkeyProvider"]

# 脚本相对仓库根的位置
SCRIPT_RELATIVE_PATH = Path("scripts") / "windows" / "hotkeys.ahk"


def _repo_script_path() -> Path:
    """按模块位置推导仓库内的 hotkeys.ahk 路径。

    ``app/platform/windows/hotkeys.py`` 向上四级是仓库根。
    """
    return Path(__file__).resolve().parents[3] / SCRIPT_RELATIVE_PATH


def _normalize_path_text(value: str) -> str:
    """把命令行里的路径片段归一成可比较的形式。"""
    return os.path.normcase(os.path.normpath(value.strip().strip('"')))


class AhkHotkeyProvider(NullHotkeyProvider):
    """探测 AutoHotkey v2 是否已加载本仓库的 hotkeys.ahk。

    继承 :class:`~app.inputs.global_hotkeys.NullHotkeyProvider`：未探测到 AHK
    时行为与占位提供者一致（恒不可用、start/stop 不注册按键），探测到之后
    才报告可用。
    """

    def __init__(self, script_path: Path | None = None) -> None:
        self._script_path = (
            Path(script_path) if script_path is not None else _repo_script_path()
        )
        # 可用标记，仅在 start 时更新；按键由 AHK 负责，这里不做注册
        self._active = False

    @property
    def script_path(self) -> Path:
        """本提供者校验的 AHK 脚本绝对路径。"""
        return self._script_path

    @property
    def active(self) -> bool:
        """最近一次 start 后的可用标记。"""
        return self._active

    def available(self) -> bool:
        """AutoHotkey 是否正在运行本仓库的 hotkeys.ahk。

        只做探测，不启动任何进程或线程，也不抛异常；读不到进程信息时返回
        False。
        """
        return self._ahk_running_with_script()

    def start(self, emit: Callable[[str, bool], None]) -> None:
        """记录可用标记。

        按键由 AHK 脚本注册并通过 CLI 通知主进程，``emit`` 在这里没有用途，
        保留参数是为了满足 :class:`~app.inputs.global_hotkeys.GlobalHotkeyProvider`
        的协议。
        """
        del emit
        self._active = self._ahk_running_with_script()

    def stop(self) -> None:
        """清掉可用标记。不关闭 AutoHotkey，重复调用安全。"""
        self._active = False

    def _ahk_running_with_script(self) -> bool:
        """进程名与脚本路径都匹配才算可用。"""
        try:
            import psutil
        except Exception:
            # psutil 不可用时不声称全局热键可用
            return False

        expected = _normalize_path_text(str(self._script_path))
        try:
            processes = psutil.process_iter(["name", "cmdline"])
        except Exception:
            return False

        for process in processes:
            try:
                name = str(process.info.get("name") or "").lower()
                if "autohotkey" not in name:
                    continue
                command_line = process.info.get("cmdline")
                if not command_line:
                    # 进程在但读不到命令行：按不可用处理，宁可不短路
                    continue
                if any(
                    _normalize_path_text(str(part)) == expected
                    for part in command_line
                ):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception:
                # 单个进程查询失败不影响其余进程，也不能据此判定可用
                continue
        return False