"""Windows 侧的进程存活探测：``psutil``。

统一走 ``psutil`` 做进程存活探测，不依赖平台信号：
用 ``psutil.Process(pid).is_running()`` 判断进程是否存活。

``psutil`` 按 WIN-MILESTONES.md W7.1 拆进 Windows 专有依赖，模块顶层不导入，
方法体里按需导入，保证 :mod:`app.platform.windows` 在没装 ``psutil`` 的环境也能导入。

注意：``is_running()`` 只看 PID 数字，Windows 的 PID 会被系统复用，PID 文件里
的残留 PID 指向别的活进程时这里只能答"活着"。进程身份的校验在上层
（:func:`app.runtime.process_manager.is_app_process` 与调用方的判定链），
本模块不做身份判断。
"""

from __future__ import annotations

__all__ = ["PsutilProcessBackend"]


class PsutilProcessBackend:
    """基于 ``psutil`` 的进程存活探测后端。"""

    # 后端名
    name: str = "psutil"

    # 该能力需要的第三方包名
    required_packages: tuple[str, ...] = ("psutil",)

    def is_alive(self, pid: int) -> bool:
        """判断 PID 对应的进程是不是还活着。

        ``pid`` 小于等于 0 一律算不活着。进程存在返回 True；
        进程不存在返回 False；权限不足说明进程存在，同样返回 True。

        :param pid: 进程 PID。
        """
        if pid <= 0:
            return False
        import psutil

        try:
            return psutil.Process(pid).is_running()
        except psutil.NoSuchProcess:
            # 进程不存在
            return False
        except psutil.AccessDenied:
            # 进程存在，只是当前用户没权限查它
            return True
        except psutil.Error:
            # 其它 psutil 错误（进程刚好退出、查询失败等）按不活着处理
            return False
