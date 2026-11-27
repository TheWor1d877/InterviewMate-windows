"""F8 的触发逻辑。

见《5-语音输入与流式ASR》「F8 交互」一节。

按键绑定在界面层，这个模块只提供「按 F8 发生了」这个入口：
界面层把 F8 绑好之后调 :meth:`HotkeyController.handle_voice_toggle`，
配置里的键名是 ``hotkeys.voice_toggle``，见《2-骨架与配置》配置结构一节。

语义：

- 没在采集时按 F8：开始采集与识别；
- 已在采集时按 F8：停止采集、冲刷最终文本并自动提交，
  不需要再按回车；
- 上一次动作还没结束时又按了一次：这次按键忽略，
  不让两个停止流程交叉。
"""

from __future__ import annotations

import asyncio

from app.audio.audio_manager import (
    RESULT_ERROR,
    RESULT_STARTED,
    AudioManager,
)
from app.utils.logger import log_event

__all__ = [
    "RESULT_IGNORED",
    "HotkeyController",
]


# 上一次动作还没结束时又按了一次，这次按键被忽略
RESULT_IGNORED: str = "ignored"


class HotkeyController:
    """F8 的触发逻辑。

    :param audio_manager: 语音采集管理器，F8 的动作最终落到它身上。
    """

    def __init__(self, audio_manager: AudioManager) -> None:
        self._manager = audio_manager
        # 一次只跑一个动作，停止流程不交叉
        self._lock = asyncio.Lock()

    @property
    def audio_manager(self) -> AudioManager:
        """接的语音采集管理器。"""
        return self._manager

    @property
    def is_running(self) -> bool:
        """采集是不是开着。"""
        return self._manager.is_running

    async def handle_voice_toggle(self) -> str:
        """按 F8 发生了一次。

        :returns: 这次动作的结果标识，含义见 :meth:`AudioManager.toggle`；
            上一次动作还没结束时返回 ``ignored``。
        """
        if self._lock.locked():
            # 上一次动作还没结束，这次按键忽略，不让两个停止流程交叉
            log_event(
                "audio.hotkey.ignored",
                "上一次语音动作还没结束，忽略这次按键",
            )
            return RESULT_IGNORED
        async with self._lock:
            result = await self._manager.toggle()
            log_event("audio.hotkey.toggled", "语音采集状态已切换", result=result)
            return result

    async def handle_voice_start(self) -> str:
        """按住语音键的入口：开始采集。

        与 :meth:`handle_voice_toggle` 共用同一把锁，
        上一次动作还没结束时这次的按下忽略。

        :returns: ``started`` 开始采集成功，``error`` 没开起来；
            上一次动作还没结束时返回 ``ignored``。
        """
        if self._lock.locked():
            # 上一次动作还没结束，这次按下忽略
            log_event(
                "audio.hotkey.ignored",
                "上一次语音动作还没结束，忽略这次按键",
            )
            return RESULT_IGNORED
        async with self._lock:
            await self._manager.start()
            result = (
                RESULT_STARTED if self._manager.is_running else RESULT_ERROR
            )
            log_event("audio.hotkey.started", "按住语音键，开始采集", result=result)
            return result

    async def handle_voice_stop(self) -> str:
        """松开语音键的入口：停止采集并冲刷最终文本。

        与 :meth:`handle_voice_toggle` 共用同一把锁，
        上一次动作还没结束时这次的松开忽略。

        :returns: 这次动作的结果标识，含义见
            :meth:`~app.audio.audio_manager.AudioManager.stop`；
            上一次动作还没结束时返回 ``ignored``。
        """
        if self._lock.locked():
            # 上一次动作还没结束，这次松开忽略
            log_event(
                "audio.hotkey.ignored",
                "上一次语音动作还没结束，忽略这次按键",
            )
            return RESULT_IGNORED
        async with self._lock:
            result = await self._manager.stop()
            log_event("audio.hotkey.stopped", "松开语音键，停止采集", result=result)
            return result
