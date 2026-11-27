"""Windows 侧的截图后端：``mss`` 占位。

见《9-平台差异与Windows适配》「Windows 截图后端」一节：Windows 侧用 ``mss``
抓整屏，纯 Python，没有系统级依赖，不弹确认框，也不落盘。

《9-平台差异与Windows适配》「实施顺序」一节把 Windows 侧的实现排在 Linux 侧
M2 到 M8 之后，其中截图排在传输层与进程管理之后。本模块现在只有接口占位：
方法体抛 :class:`NotImplementedError`。

Windows 侧不装 ``dbus-next``。
"""

from __future__ import annotations

from app.platform.types import Screenshot, ScreenshotTarget

__all__ = ["MssScreenshotBackend"]


class MssScreenshotBackend:
    """基于 ``mss`` 的截图后端（本轮只有接口占位）。"""

    # 后端名，对应配置里的 ``providers.screenshot`` 取值
    name: str = "mss"

    # 该能力需要的第三方包名
    required_packages: tuple[str, ...] = ("mss",)

    async def probe(self, timeout: float = 3.0) -> str:
        """检查截图接口能不能用。

        :param timeout: 超时秒数。
        :raises NotImplementedError: 始终抛异常。
        """
        raise NotImplementedError(
            "Windows 侧的截图接口探测尚未实现，"
            "见《9-平台差异与Windows适配》的实施顺序"
        )

    async def capture(
        self,
        *,
        target: ScreenshotTarget = "screen",
        monitor: int = 0,
    ) -> Screenshot:
        """做一次截图。

        :param target: 截图目标，Windows 侧的实现还没做，用不上。
        :param monitor: 显示器序号，用不上。
        :raises NotImplementedError: 始终抛异常。
        """
        raise NotImplementedError(
            "Windows 侧的截图尚未实现，见《9-平台差异与Windows适配》的实施顺序"
        )
