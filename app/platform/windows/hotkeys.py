"""AutoHotkey v2 全局热键提供者（Windows）。

终端失焦后的 F8/F9 由 ``scripts/windows/hotkeys.ahk`` 接管，脚本自己
注册按键，命中后通过 ``python -m app.cli hotkey voice-toggle|screenshot``
通知主进程。本提供者不注册也不注入任何按键，只回答一个问题：
「AutoHotkey 是否已经在运行本仓库的 hotkeys.ahk」。

探测采用进程名与脚本路径双重校验：进程名必须像 AutoHotkey，且命令行中
必须出现本仓库 hotkeys.ahk 的绝对路径。进程存在但读不到命令行时按不可用
处理——宁可退回界面内按键，也不能因为误判让界面错误地短路 F8/F9。

两种「不可用」要区分开：

- 完全没检测到 AutoHotkey 进程：可能只是还没启动，短暂重试一下，覆盖
  ``interviewmate.cmd start`` 先拉起 AHK、主进程随后启动的竞态；
- 检测到 AutoHotkey 但加载的是别的仓库路径下的 hotkeys.ahk：这类实例会把
  CLI 发到别的仓库的 socket，本进程既收不到按键，也不能短路面内按键，
  只记一条 WARNING 让用户重启 AHK，不做无意义的等待。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from app.inputs.global_hotkeys import NullHotkeyProvider
from app.utils.logger import log_event

__all__ = ["AhkHotkeyProvider"]

# 脚本相对仓库根的位置
SCRIPT_RELATIVE_PATH = Path("scripts") / "windows" / "hotkeys.ahk"

# 没检测到 AHK 进程时的短重试：覆盖 AHK 与主进程的启动竞态，
# 总等待不超过这个秒数，避免无 AHK 时拖慢启动太久。
_AVAILABILITY_RETRY_SECONDS: float = 0.6
_AVAILABILITY_RETRY_INTERVAL: float = 0.2


def _repo_script_path() -> Path:
    """按模块位置推导仓库内的 hotkeys.ahk 路径。

    ``app/platform/windows/hotkeys.py`` 向上四级是仓库根。
    """
    return Path(__file__).resolve().parents[3] / SCRIPT_RELATIVE_PATH


def _normalize_path_text(value: str) -> str:
    """把命令行里的路径片段归一成可比较的形式。

    去掉两侧引号与 Windows 长路径前缀 ``\\\\?\\``，再做 ``normcase`` /
    ``normpath``：同一路径在命令行里可能带引号、用正斜杠或带长路径前缀。
    """
    text = value.strip().strip('"')
    if text.startswith("\\\\?\\"):
        text = text[4:]
    return os.path.normcase(os.path.normpath(text))


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
        # 已经提示过的不可用原因，避免每次探测都重复写日志
        self._warned_missing = False
        self._warned_foreign: set[str] = set()

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
        False。完全没检测到 AHK 进程时最多重试 ``_AVAILABILITY_RETRY_SECONDS``
        秒，覆盖 AHK 刚被拉起、尚未出现在进程表里的竞态。
        """
        deadline = time.monotonic() + _AVAILABILITY_RETRY_SECONDS
        while True:
            matched, ahk_count, foreign = self._scan_processes()
            if matched:
                self._warned_missing = False
                self._warned_foreign.clear()
                return True
            if ahk_count > 0:
                if foreign:
                    self._report_foreign(foreign)
                else:
                    self._report_unreadable()
                return False
            if time.monotonic() >= deadline:
                self._report_missing()
                return False
            time.sleep(_AVAILABILITY_RETRY_INTERVAL)

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

    def _report_foreign(self, foreign: list[str]) -> None:
        """提示 AHK 加载的是其它仓库路径的同名脚本，本条提示只写一次。"""
        for script in foreign:
            if script in self._warned_foreign:
                continue
            self._warned_foreign.add(script)
            log_event(
                "input.hotkey.foreign_script",
                "AutoHotkey 正在运行其它路径的 hotkeys.ahk，本仓库热键未接管；"
                "请退出该 AHK 实例后重跑 interviewmate.cmd start（会自动拉起热键助手）",
                level="WARNING",
                script=script,
                expected=str(self._script_path),
            )

    def _report_missing(self) -> None:
        """提示没有检测到本仓库的 AHK 脚本，本条提示只写一次。"""
        if self._warned_missing:
            return
        self._warned_missing = True
        log_event(
            "input.hotkey.ahk_missing",
            "未检测到运行本仓库 hotkeys.ahk 的 AutoHotkey 进程，"
            "F8/F9 仍由界面处理，终端失焦时不可用",
            level="WARNING",
            expected=str(self._script_path),
        )

    def _report_unreadable(self) -> None:
        """提示 AHK 进程在但读不到命令行，本条提示只写一次。"""
        if "unreadable" in self._warned_foreign:
            return
        self._warned_foreign.add("unreadable")
        log_event(
            "input.hotkey.cmdline_unreadable",
            "检测到 AutoHotkey 进程但读不到命令行，按不可用处理",
            level="WARNING",
            expected=str(self._script_path),
        )

    def _scan_processes(self) -> tuple[bool, int, list[str]]:
        """扫描 AHK 进程。

        :returns: ``(是否加载本仓库脚本, AHK 进程数, 其它仓库 hotkeys.ahk 路径列表)``。
            完全读不到进程信息时返回 ``(False, 0, [])``。
        """
        try:
            import psutil
        except Exception:
            # psutil 不可用时不声称全局热键可用
            return False, 0, []

        expected = _normalize_path_text(str(self._script_path))
        expected_name = self._script_path.name.lower()
        matched = False
        ahk_count = 0
        foreign: list[str] = []

        try:
            processes = psutil.process_iter(["name", "cmdline"])
        except Exception:
            return False, 0, []

        for process in processes:
            try:
                name = str(process.info.get("name") or "").lower()
                if "autohotkey" not in name:
                    continue
                ahk_count += 1
                command_line = process.info.get("cmdline")
                if not command_line:
                    # 进程在但读不到命令行：按不可用处理，宁可不短路
                    continue
                script_args = [
                    str(part)
                    for part in command_line
                    if _normalize_path_text(str(part)).endswith(".ahk")
                ]
                if expected in {
                    _normalize_path_text(part) for part in script_args
                }:
                    matched = True
                    continue
                for part in script_args:
                    normalized = _normalize_path_text(part)
                    if (
                        os.path.basename(normalized) == expected_name
                        and normalized not in foreign
                    ):
                        foreign.append(normalized)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception:
                # 单个进程查询失败不影响其余进程，也不能据此判定可用
                continue

        return matched, ahk_count, foreign

    def _ahk_running_with_script(self) -> bool:
        """进程名与脚本路径都匹配才算可用。"""
        matched, _, _ = self._scan_processes()
        return matched