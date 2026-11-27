"""耗时统计。

指标名与含义见《2-骨架与配置》性能指标一节，共 9 个，每个指标记毫秒值和来源；
P50、P90、P95 的统计口径见《8-集成与验收》性能统计一节。

百分位用最近秩法：值升序排序后取 ``index = ceil(p/100 * n) - 1``，
再用 ``min(index, n-1)`` 夹紧。

进程级默认记录器见 :func:`default_recorder`：各模块把耗时样本记到那里，
主进程的状态 payload 从它取统计，见《8-集成与验收》性能统计一节。

本模块只依赖标准库，不引入重依赖，也不发起网络请求。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "METRIC_NAMES",
    "MetricSample",
    "MetricStats",
    "MetricRecorder",
    "Timer",
    "timed",
    "default_recorder",
    "reset_default_recorder",
]


# 指标名与《2-骨架与配置》性能指标一节的表逐字一致
METRIC_NAMES: tuple[str, ...] = (
    "capture_latency",
    "asr_latency",
    "asr_final_latency",
    "retrieval_latency",
    "request_latency",
    "ttft",
    "first_framework",
    "first_item",
    "total_latency",
)


@dataclass(frozen=True)
class MetricSample:
    """一条指标样本。

    :param name: 指标名，取值在 :data:`METRIC_NAMES` 里。
    :param value_ms: 耗时毫秒值，非负有限数。
    :param source: 来源，例如 ``mock``、``dify``、``deepseek``。
    :param ts: 记录时刻的 Unix 时间戳（秒）。
    """

    name: str
    value_ms: float
    source: str
    ts: float


@dataclass(frozen=True)
class MetricStats:
    """一个指标的统计结果。

    百分位按最近秩法算，见《8-集成与验收》性能统计一节。

    :param name: 指标名。
    :param count: 样本数。
    :param min_ms: 最小值。
    :param max_ms: 最大值。
    :param mean_ms: 平均值。
    :param p50_ms: P50。
    :param p90_ms: P90。
    :param p95_ms: P95。
    """

    name: str
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float


def _percentile(values: list[float], p: int) -> float:
    """最近秩法百分位，``values`` 必须已升序排序。

    ``index = ceil(p/100 * n) - 1``，再用 ``min(index, n-1)`` 夹紧。
    向上取整用整数运算算，避免浮点误差。
    """
    n = len(values)
    if n == 0:
        raise ValueError("百分位计算需要至少一个样本")
    rank = -((-p * n) // 100)  # ceil(p * n / 100)
    index = min(rank - 1, n - 1)
    return values[max(index, 0)]


class MetricRecorder:
    """记录指标样本并算统计量。

    样本按记录顺序追加，同一实例多次 record 的顺序稳定。
    这里不要求线程安全；调用方自己保证并发写入时的互斥。
    """

    def __init__(self) -> None:
        # 按指标名分组的样本
        self._samples: dict[str, list[MetricSample]] = {}
        # 全局记录顺序，samples(None) 按这个顺序返回
        self._order: list[MetricSample] = []

    def record(self, name: str, value_ms: float, *, source: str = "") -> MetricSample:
        """记录一条样本并返回它。

        :param name: 指标名，取值必须在 :data:`METRIC_NAMES` 里。
        :param value_ms: 耗时毫秒值，必须是有限非负数。
        :param source: 来源标记。
        :raises ValueError: 指标名非法，或耗时不是有限非负数。
        """
        if name not in METRIC_NAMES:
            legal = "、".join(METRIC_NAMES)
            raise ValueError(f"未知指标名 {name!r}；合法指标名：{legal}")
        try:
            value = float(value_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"指标 {name!r} 的耗时必须是数字，收到 {value_ms!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"指标 {name!r} 的耗时必须是有限数，收到 {value_ms!r}")
        if value < 0:
            raise ValueError(f"指标 {name!r} 的耗时不能是负数，收到 {value_ms!r}")

        sample = MetricSample(name=name, value_ms=value, source=source, ts=time.time())
        self._samples.setdefault(name, []).append(sample)
        self._order.append(sample)
        return sample

    def samples(self, name: str | None = None) -> list[MetricSample]:
        """返回样本副本。

        ``name`` 为 None 时按记录顺序返回全部样本，否则只返回该指标的样本。
        指标名非法时返回空列表，不抛异常。
        """
        if name is None:
            return list(self._order)
        return list(self._samples.get(name, []))

    def stats(self, name: str) -> MetricStats | None:
        """算一个指标的统计量，没有样本时返回 None。

        指标名非法时同样返回 None，因为非法指标名不会有样本。
        """
        values = sorted(sample.value_ms for sample in self._samples.get(name, []))
        count = len(values)
        if count == 0:
            return None
        return MetricStats(
            name=name,
            count=count,
            min_ms=values[0],
            max_ms=values[-1],
            mean_ms=sum(values) / count,
            p50_ms=_percentile(values, 50),
            p90_ms=_percentile(values, 90),
            p95_ms=_percentile(values, 95),
        )

    def snapshot(self) -> dict[str, list[MetricSample]]:
        """返回按指标名分组的样本副本，只包含有样本的指标。"""
        return {name: list(items) for name, items in self._samples.items()}

    def reset(self, name: str | None = None) -> None:
        """清空样本，``name`` 为 None 时清空全部。"""
        if name is None:
            self._samples.clear()
            self._order.clear()
            return
        self._samples.pop(name, None)
        self._order = [sample for sample in self._order if sample.name != name]


# 进程级默认记录器：各模块把耗时样本记到这里，
# 主进程的状态 payload 从这里取统计，见《8-集成与验收》性能统计一节
_default_recorder: MetricRecorder = MetricRecorder()


def default_recorder() -> MetricRecorder:
    """返回进程级默认记录器。

    同一个进程里各处记的耗时样本都落在这个记录器上，
    ``interviewmate status`` 通过主进程的状态 payload 读到它的统计。
    """
    return _default_recorder


def reset_default_recorder() -> None:
    """清空进程级默认记录器里的全部样本，测试里用。"""
    _default_recorder.reset()


class Timer:
    """上下文管理器，退出时把耗时毫秒写进 recorder。

    用单调时钟（默认 :func:`time.perf_counter`）算耗时，测试里可注入自定义
    clock。``elapsed_ms`` 随时可读：已经退出时是最终耗时，正在跑时是当前耗时，
    还没进入时是 0。

    ``sample`` 在退出前是 None，退出后是写进 recorder 的那条样本。
    **没给 recorder 时只算时间不落库，退出后 ``sample`` 仍然是 None。**

    :param name: 指标名，取值在 :data:`METRIC_NAMES` 里；
        非法指标名在退出并写 recorder 时抛 ValueError。
    :param source: 来源标记。
    :param recorder: 样本落库对象，为 None 时只计时不落库。
    :param clock: 时钟函数，返回秒。
    """

    def __init__(
        self,
        name: str,
        *,
        source: str = "",
        recorder: MetricRecorder | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.name = name
        self.source = source
        self.recorder = recorder
        self.clock = clock
        self._start: float | None = None
        self._elapsed_ms: float | None = None
        self._sample: MetricSample | None = None

    def __enter__(self) -> "Timer":
        self._start = self.clock()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._start is None:
            return None
        elapsed_ms = (self.clock() - self._start) * 1000.0
        self._elapsed_ms = elapsed_ms
        if self.recorder is not None:
            self._sample = self.recorder.record(
                self.name, elapsed_ms, source=self.source
            )
        return None

    @property
    def elapsed_ms(self) -> float:
        """已消耗的毫秒数。"""
        if self._elapsed_ms is not None:
            return self._elapsed_ms
        if self._start is not None:
            return (self.clock() - self._start) * 1000.0
        return 0.0

    @property
    def sample(self) -> MetricSample | None:
        """退出前是 None；退出后是写进 recorder 的样本，没给 recorder 时仍是 None。"""
        return self._sample


def timed(
    name: str,
    *,
    source: str = "",
    recorder: MetricRecorder | None = None,
) -> Timer:
    """返回一个 :class:`Timer`，按上下文管理器用法使用。

    ``with timed("ttft", recorder=recorder) as timer:`` 退出后
    ``timer.sample`` 就是落库的样本。
    """
    return Timer(name, source=source, recorder=recorder)
