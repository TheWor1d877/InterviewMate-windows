"""Linux 侧的代码编译器：``g++``。

见《9-平台差异与Windows适配》「平台能力矩阵」一节：Linux 侧用 ``g++``，
Windows 侧用 ``cl.exe`` 或 MinGW-w64 的 ``g++``。

M1 阶段没有代码编译功能，这里只放接口最小实现。
"""

from __future__ import annotations

__all__ = ["GppCompiler"]


class GppCompiler:
    """``g++`` 编译器后端。"""

    # 后端名
    name: str = "gpp"

    # 编译器是系统自带的，不需要额外第三方包
    required_packages: tuple[str, ...] = ()

    def command(self) -> str:
        """返回编译命令。"""
        return "g++"
