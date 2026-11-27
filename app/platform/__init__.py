"""平台门面。

《2-骨架与配置》目录结构约定一节规定：平台相关的代码收进
:mod:`app.platform`，平台在进程启动时选定一次，上层任何模块都不判断
``sys.platform``。

InterviewMate 现在只支持 Windows。本模块是全仓唯一判断 ``sys.platform``
的地方，:func:`get_platform` 按当前平台返回实现：

- Windows：``sys.platform`` 等于 ``win32``，见 :mod:`app.platform.windows`；
- 其它平台不支持，抛 :class:`NotImplementedError` 并给出清晰说明。

接口定义在 :mod:`app.platform.types`。
"""

from __future__ import annotations

import sys

from app.platform.types import Platform

__all__ = ["get_platform"]


def get_platform() -> Platform:
    """返回当前平台的实现。

    同一进程反复调用拿到的是同一个对象。

    :returns: 当前平台的实现，字段见 :class:`~app.platform.types.Platform`。
    :raises NotImplementedError: 当前平台不是 Windows。
    """
    if sys.platform == "win32":
        from app.platform.windows import get_platform as _windows_platform

        return _windows_platform()
    raise NotImplementedError(
        f"当前平台不支持：{sys.platform}。InterviewMate 仅支持 Windows（win32）。"
    )
