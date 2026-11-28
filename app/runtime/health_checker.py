"""运行状态探测与状态文本格式化。

见《2-骨架与配置》命令行入口一节的状态字段取值表：

| 状态项 | 取值 |
|---|---|
| RUNNING | ``running``、``stopped`` |
| AUDIO | ``off``、``on`` |
| ASR | ``idle``、``loading``、``ready``、``error`` |
| DIFY | ``offline``、``ready``、``degraded`` |
| MODEL | ``idle``、``connecting``、``streaming``、``error`` |
| TTFT | 毫秒数，没有数据时显示 ``-`` |

``providers`` 是「状态字段名 → 取当前值的可调用对象」的映射，可调用对象
可以是同步函数也可以是协程函数。取不到的字段落兜底值，不让整次探测失败。

本模块不 import 任何推理或音频依赖，也不发网络请求。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping

__all__ = [
    "STATUS_FIELDS",
    "PROVIDER_FIELDS",
    "HealthStatus",
    "HealthChecker",
]


# 状态字段名，顺序与《2-骨架与配置》命令行入口一节的状态表一致
STATUS_FIELDS: tuple[str, ...] = (
    "running",
    "audio",
    "asr",
    "dify",
    "model",
    "ttft",
)

# 可以从 providers 取值的字段；running 由调用方自己判断，不从 providers 取
PROVIDER_FIELDS: tuple[str, ...] = ("audio", "asr", "dify", "model", "ttft")

# 各字段的合法取值，取值表见《2-骨架与配置》命令行入口一节
_ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "audio": frozenset({"off", "on"}),
    "asr": frozenset({"idle", "loading", "ready", "error"}),
    "dify": frozenset({"offline", "ready", "degraded"}),
    "model": frozenset({"idle", "connecting", "streaming", "error"}),
}

# 取值失败时的兜底值
_FALLBACK_VALUES: dict[str, Any] = {
    "audio": "off",
    "asr": "error",
    "dify": "degraded",
    "model": "error",
    "ttft": None,
}

# 没有 provider 时用的缺省值
_DEFAULT_VALUES: dict[str, Any] = {
    "audio": "off",
    "asr": "idle",
    "dify": "offline",
    "model": "idle",
    "ttft": None,
}


@dataclass(frozen=True)
class HealthStatus:
    """一份运行状态快照。

    字段名与《2-骨架与配置》命令行入口一节的状态项对应：
    ``running``、``audio``、``asr``、``dify``、``model``，
    另有 ``ttft_ms`` 存首字延迟毫秒数，没有数据时为 None。
    """

    running: str = "stopped"
    audio: str = "off"
    asr: str = "idle"
    dify: str = "offline"
    model: str = "idle"
    ttft_ms: float | None = None


class HealthChecker:
    """按 providers 探测运行状态。

    :param providers: 状态字段名到取当前值的可调用对象的映射，
        key 只认 ``audio``、``asr``、``dify``、``model``、``ttft``。
    """

    def __init__(self, providers: Mapping[str, Callable[[], Any]] | None = None) -> None:
        self._providers: dict[str, Callable[[], Any]] = dict(providers or {})

    async def collect(self) -> HealthStatus:
        """探测一次状态。

        逐个字段取 providers 的当前值：同步的直接调，异步的 await。
        某个 provider 抛异常或返回非法值时，该字段落兜底值，其它字段照常。

        ``running`` 不从 providers 取，留在缺省的 ``stopped``，由调用方按
        主进程是否在运行自己填。

        :returns: 状态快照。
        """
        values: dict[str, Any] = {}
        for field_name in PROVIDER_FIELDS:
            values[field_name] = await self._collect_field(field_name)
        return HealthStatus(
            running=HealthStatus.running,
            audio=values["audio"],
            asr=values["asr"],
            dify=values["dify"],
            model=values["model"],
            ttft_ms=values["ttft"],
        )

    def format_status(
        self,
        status: HealthStatus,
        *,
        app_name: str = "InterviewMate",
        pid: int | None = None,
    ) -> str:
        """把状态渲染成命令行侧要打印的文本。

        格式见《2-骨架与配置》命令行入口一节的 ``status`` 输出：
        首行是应用名与运行状态，第二行是 PID，空行之后依次是
        Audio、ASR、Dify、Model、TTFT。PID 没有数据写 ``-``，
        TTFT 没有数据写 ``-``，有数据写整数毫秒。末尾不额外加空行。

        :param status: 状态快照。
        :param app_name: 应用名，默认 ``InterviewMate``。
        :param pid: 主进程 PID，None 时写 ``-``。
        """
        pid_text = "-" if pid is None else str(pid)
        if status.ttft_ms is None:
            ttft_text = "-"
        else:
            ttft_text = str(int(round(status.ttft_ms)))
        lines = [
            f"{app_name}: {status.running}",
            f"PID: {pid_text}",
            "",
            f"Audio: {status.audio}",
            f"ASR: {status.asr}",
            f"Dify: {status.dify}",
            f"Model: {status.model}",
            f"TTFT: {ttft_text}",
        ]
        return "\n".join(lines)

    async def _collect_field(self, field_name: str) -> Any:
        """取一个字段的当前值，失败时落兜底值。"""
        provider = self._providers.get(field_name)
        if provider is None:
            return _DEFAULT_VALUES[field_name]
        try:
            value = provider()
            if inspect.isawaitable(value):
                value = await value
        except Exception:
            # provider 抛异常不影响其它字段
            return _FALLBACK_VALUES[field_name]
        return self._validate(field_name, value)

    @staticmethod
    def _validate(field_name: str, value: Any) -> Any:
        """校验取值合法性，非法时落兜底值。"""
        if field_name == "ttft":
            # TTFT 是毫秒数，没有数据时是 None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return _FALLBACK_VALUES["ttft"]
            return float(value)
        allowed = _ALLOWED_VALUES[field_name]
        if not isinstance(value, str) or value not in allowed:
            return _FALLBACK_VALUES[field_name]
        return value
