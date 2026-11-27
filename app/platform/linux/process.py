"""Linux 侧的进程存活探测：``os.kill(pid, 0)``。

见《9-平台差异与Windows适配》「Windows 进程存活探测与退出」一节：
Linux 侧用 ``os.kill(pid, 0)`` 探测进程是否还活着，Windows 侧必须换实现
（CPython 在 Windows 上对除 ``CTRL_C_EVENT`` 与 ``CTRL_BREAK_EVENT`` 之外的
sig 值一律调 ``TerminateProcess``，``os.kill(pid, 0)`` 会把目标进程杀掉）。

本模块把 :func:`app.runtime.process_manager.is_process_alive` 的逻辑搬过来，
行为不变。两边的差异按文档保留，不在这里统一。
"""

from __future__ import annotations

import os

__all__ = ["LinuxProcessBackend"]


class LinuxProcessBackend:
    """基于 ``os.kill`` 的进程存活探测后端。"""

    # 后端名
    name: str = "linux"

    # 该能力不需要额外第三方包
    required_packages: tuple[str, ...] = ()

    def is_alive(self, pid: int) -> bool:
        """判断 PID 对应的进程是不是还活着。

        ``pid`` 小于等于 0 一律算不活着。进程存在返回 True；
        进程不存在返回 False；权限不足说明进程存在，同样返回 True。

        :param pid: 进程 PID。
        """
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # 进程不存在
            return False
        except PermissionError:
            # 进程存在，只是当前用户没权限发信号
            return True
        except OSError:
            # 其它 OS 错误（含 errno.EPERM 的其它形态）按不活着处理
            return False
        return True
