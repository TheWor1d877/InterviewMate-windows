"""Windows 侧的代码编译器：``cl.exe`` 占位。

见《9-平台差异与Windows适配》「平台能力矩阵」一节：Windows 侧用 ``cl.exe``
或 MinGW-w64 的 ``g++``。

本模块现在只有接口占位：编译命令已经定下来，调用编译器本身排在后续里程碑。
"""

from __future__ import annotations

__all__ = ["MsvcCompiler"]


class MsvcCompiler:
    """``cl.exe`` 编译器后端（本轮只有接口占位）。"""

    # 后端名
    name: str = "msvc"

    # 编译器是系统自带的，不需要额外第三方包
    required_packages: tuple[str, ...] = ()

    def command(self) -> str:
        """返回编译命令。"""
        return "cl.exe"
