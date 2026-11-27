"""平台门面。

《9-平台差异与Windows适配》「目录结构」一节规定：平台相关的代码收进
:mod:`app.platform`，平台在进程启动时选定一次，上层任何模块都不判断
``sys.platform``。

本模块是全仓唯一判断 ``sys.platform`` 的地方，:func:`get_platform` 按当前
平台返回实现：

- Linux：``sys.platform`` 以 ``linux`` 开头，见 :mod:`app.platform.linux`；
- Windows：``sys.platform`` 等于 ``win32``，见 :mod:`app.platform.windows`；
- 其它平台不支持，抛 :class:`NotImplementedError`。

接口定义在 :mod:`app.platform.types`，两边实现同构。
"""

from __future__ import annotations

import sys

from app.platform.types import Platform

__all__ = ["get_platform"]


def get_platform() -> Platform:
    """返回当前平台的实现。

    同一进程反复调用拿到的是同一个对象。

    :returns: 当前平台的实现，字段见 :class:`~app.platform.types.Platform`。
    :raises NotImplementedError: 当前平台既不是 Linux 也不是 Windows。
    """
    if sys.platform.startswith("linux"):
        from app.platform.linux import get_platform as _linux_platform

        return _linux_platform()
    if sys.platform == "win32":
        from app.platform.windows import get_platform as _windows_platform

        return _windows_platform()
    raise NotImplementedError(f"当前平台不支持：{sys.platform}")
