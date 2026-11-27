"""全局热键的提供者接口与按键分发。

Textual 的按键绑定只在本终端有焦点时生效，也拿不到松键，做不了
「按住说话」。这一层把「全局按键」抽象出来：平台后端直接从系统输入拿
按下与松开，通过 :meth:`GlobalHotkeyProvider.start` 给的回调送出来，
:class:`HotkeyListener` 负责归一键名、按「当前按住集合」去重，再把动作
投递到事件循环上的回调。

约定：

- 提供者只在 :meth:`GlobalHotkeyProvider.available` 为真时启动；
- ``emit`` 从后台线程调用，第二个参数 True 是按下、False 是松开；
- 多台键盘同时上报同一个键时，靠 :class:`HotkeyListener` 的按住集合去重；
- 不支持全局热键的平台用 :class:`NullHotkeyProvider`，``available()``
  恒为 False，上层照常接线，不用判断平台。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.utils.logger import log_event

__all__ = [
    "HotkeyHandlers",
    "GlobalHotkeyProvider",
    "NullHotkeyProvider",
    "HotkeyListener",
]


@dataclass(frozen=True)
class HotkeyHandlers:
    """一个热键的按下与松开回调。

    ``on_release`` 可以不给：只关心按下的热键只写 ``on_press`` 即可。
    """

    on_press: Callable[[], None]
    on_release: Callable[[], None] | None = None


@runtime_checkable
class GlobalHotkeyProvider(Protocol):
    """全局按键提供者。

    实现方自己决定按键从哪里来（Linux 侧读 ``/dev/input``），
    对外只暴露「能不能用」「开始」「停止」三件事。
    """

    def available(self) -> bool:
        """全局热键现在能不能用。

        只做探测，不启动任何线程，也不抛异常。
        """
        ...

    def start(self, emit: Callable[[str, bool], None]) -> None:
        """开始上报按键。

        :param emit: 按键回调，参数是键名与是否按下（True 按下、False 松开）。
            实现方在后台线程里调用它。
        """
        ...

    def stop(self) -> None:
        """停止上报并释放资源，重复调用不报错。"""
        ...


class NullHotkeyProvider:
    """不支持全局热键的平台用的占位提供者。

    ``available()`` 恒为 False，``start`` 与 ``stop`` 都是空操作。
    """

    def available(self) -> bool:
        """恒为 False：本平台没有全局热键。"""
        return False

    def start(self, emit: Callable[[str, bool], None]) -> None:
        """空操作。"""

    def stop(self) -> None:
        """空操作。"""


class HotkeyListener:
    """把提供者送来的按键分发给绑定的回调。

    键名在构造时做 ``strip().lower()`` 归一，提供者回传的键名同样处理。
    按键按提供者的 ``emit(key, pressed)`` 维护「当前按住集合」去重：
    第一次按下才调 ``on_press``，松开且确实按过才调 ``on_release``，
    重复按下与没按过的松开直接丢弃——多台键盘同时上报同一个键时靠这个去重。

    ``emit`` 来自后台线程，回调统一经 ``loop.call_soon_threadsafe``
    投递到事件循环。循环已关闭或回调抛异常时只记日志，不向上抛。
    :meth:`stop` 之后到达的 ``emit`` 一律丢弃。

    :param provider: 全局按键提供者，不可用时 :meth:`start` 返回 False。
    :param bindings: 键名到 :class:`HotkeyHandlers` 的映射。
    :param loop: 回调要投递到的事件循环，None 表示 :meth:`start` 时
        取 :func:`asyncio.get_running_loop`。
    """

    def __init__(
        self,
        provider: GlobalHotkeyProvider,
        bindings: Mapping[str, HotkeyHandlers],
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._provider = provider
        self._bindings = {
            key.strip().lower(): handlers for key, handlers in bindings.items()
        }
        self._loop = loop
        self._running = False
        # 当前按住的键，只在事件循环线程里读写
        self._pressed: set[str] = set()

    @property
    def running(self) -> bool:
        """监听是不是开着。"""
        return self._running

    def start(self) -> bool:
        """启动监听。

        :returns: 启动成功返回 True；提供者不可用、启动过程出错或当前
            没有事件循环时返回 False 并记 WARNING 日志，不抛异常。
        """
        if self._running:
            return True
        try:
            if not self._provider.available():
                log_event(
                    "input.hotkey.unavailable",
                    "全局热键提供者不可用，退回终端内的按键绑定",
                    level="WARNING",
                )
                return False
            if self._loop is None:
                self._loop = asyncio.get_running_loop()
            self._running = True
            self._pressed.clear()
            self._provider.start(self._on_emit)
        except Exception as exc:
            self._running = False
            self._pressed.clear()
            log_event(
                "input.hotkey.start_failed",
                "启动全局热键失败，退回终端内的按键绑定",
                level="WARNING",
                error=type(exc).__name__,
            )
            return False
        return True

    def stop(self) -> None:
        """停止监听，之后到达的按键一律丢弃。重复调用安全。"""
        was_running = self._running
        self._running = False
        self._pressed.clear()
        try:
            self._provider.stop()
        except Exception as exc:
            log_event(
                "input.hotkey.stop_failed",
                "停止全局热键时出错",
                level="WARNING",
                error=type(exc).__name__,
            )
        if was_running:
            log_event("input.hotkey.stopped", "全局热键已停止")

    def _on_emit(self, key: str, pressed: bool) -> None:
        """提供者的回调，从后台线程来：只做筛选，实际处理投递到事件循环。"""
        if not self._running:
            return
        name = str(key).strip().lower()
        handlers = self._bindings.get(name)
        if handlers is None:
            return
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._dispatch, name, bool(pressed), handlers)
        except RuntimeError:
            # 事件循环已关闭，丢键即可
            log_event(
                "input.hotkey.loop_closed",
                "事件循环已关闭，丢弃这次按键",
                level="WARNING",
                key=name,
            )

    def _dispatch(
        self,
        name: str,
        pressed: bool,
        handlers: HotkeyHandlers,
    ) -> None:
        """在事件循环里做去重并调回调。"""
        if not self._running:
            return
        if pressed:
            if name in self._pressed:
                # 多台键盘同时上报同一个键，重复按下丢掉
                return
            self._pressed.add(name)
            self._invoke(handlers.on_press, name, "press")
            return
        if name not in self._pressed:
            # 没按过就松开，丢掉
            return
        self._pressed.discard(name)
        if handlers.on_release is not None:
            self._invoke(handlers.on_release, name, "release")

    def _invoke(
        self,
        callback: Callable[[], None],
        name: str,
        action: str,
    ) -> None:
        """调一个回调，抛异常只记日志。"""
        try:
            callback()
        except Exception as exc:
            log_event(
                "input.hotkey.callback_failed",
                "全局热键回调抛异常，已忽略",
                level="ERROR",
                key=name,
                action=action,
                error=type(exc).__name__,
            )