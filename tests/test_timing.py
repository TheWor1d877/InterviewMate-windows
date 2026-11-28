"""耗时统计的测试。

覆盖《2-骨架与配置》性能指标一节的 9 个指标名，
以及《8-集成与验收》性能统计一节要求的 P50、P90、P95 和最大延迟。
"""

from __future__ import annotations

import math
import typing

import pytest

from app.utils.timing import (
    METRIC_NAMES,
    MetricRecorder,
    MetricSample,
    MetricStats,
    Timer,
    default_recorder,
    reset_default_recorder,
    timed,
)


def test_metric_names_match_design() -> None:
    """指标名与《2-骨架与配置》性能指标一节的表逐字一致。"""
    assert METRIC_NAMES == (
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


# --------------------------------------------------------------------------
# record 的校验
# --------------------------------------------------------------------------


def test_record_rejects_unknown_metric_name() -> None:
    """非法指标名抛 ValueError，消息里列出合法指标名。"""
    recorder = MetricRecorder()

    with pytest.raises(ValueError) as excinfo:
        recorder.record("unknown_latency", 12.0)

    message = str(excinfo.value)
    assert "unknown_latency" in message
    for name in METRIC_NAMES:
        assert name in message


def test_record_rejects_negative_value() -> None:
    """负的耗时抛 ValueError。"""
    recorder = MetricRecorder()

    with pytest.raises(ValueError) as excinfo:
        recorder.record("ttft", -1.0)

    assert "ttft" in str(excinfo.value)


def test_record_rejects_non_finite_value() -> None:
    """无穷大和非数不是合法耗时。"""
    recorder = MetricRecorder()

    with pytest.raises(ValueError):
        recorder.record("ttft", math.inf)
    with pytest.raises(ValueError):
        recorder.record("ttft", math.nan)


def test_record_rejects_non_numeric_value() -> None:
    """非数值耗时抛 ValueError。"""
    recorder = MetricRecorder()

    with pytest.raises(ValueError):
        recorder.record("ttft", "很快")  # type: ignore[arg-type]


def test_record_returns_sample_with_fields() -> None:
    """record 返回的样本带指标名、耗时、来源和时间戳。"""
    recorder = MetricRecorder()

    sample = recorder.record("ttft", 123.5, source="mock")

    assert isinstance(sample, MetricSample)
    assert sample.name == "ttft"
    assert sample.value_ms == 123.5
    assert sample.source == "mock"
    assert sample.ts > 0


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


def test_stats_returns_none_for_empty_metric() -> None:
    """没有样本时 stats 返回 None。"""
    recorder = MetricRecorder()

    assert recorder.stats("ttft") is None


def test_stats_returns_none_for_unknown_metric() -> None:
    """指标名非法时同样没有样本，返回 None。"""
    recorder = MetricRecorder()

    assert recorder.stats("unknown_latency") is None


def test_stats_count_min_max_mean() -> None:
    """count、min、max、mean 算得对。"""
    recorder = MetricRecorder()
    for value in (10.0, 20.0, 30.0):
        recorder.record("request_latency", value)

    stats = recorder.stats("request_latency")

    assert stats is not None
    assert stats.name == "request_latency"
    assert stats.count == 3
    assert stats.min_ms == 10.0
    assert stats.max_ms == 30.0
    assert stats.mean_ms == 20.0


def test_stats_percentiles_with_one_sample() -> None:
    """只有一个样本时三个百分位都是这个值。"""
    recorder = MetricRecorder()
    recorder.record("capture_latency", 42.0)

    stats = recorder.stats("capture_latency")

    assert stats is not None
    assert stats.p50_ms == 42.0
    assert stats.p90_ms == 42.0
    assert stats.p95_ms == 42.0


def test_stats_percentiles_with_two_samples() -> None:
    """两个样本时按最近秩法取：P50 取第 1 个，P90 与 P95 取第 2 个。"""
    recorder = MetricRecorder()
    recorder.record("total_latency", 10.0)
    recorder.record("total_latency", 20.0)

    stats = recorder.stats("total_latency")

    assert stats is not None
    assert stats.p50_ms == 10.0
    assert stats.p90_ms == 20.0
    assert stats.p95_ms == 20.0


def test_stats_percentiles_with_ten_samples() -> None:
    """十个样本时按最近秩法取：P50 第 5 个、P90 第 9 个、P95 第 10 个。"""
    recorder = MetricRecorder()
    for value in range(1, 11):
        recorder.record("first_item", float(value))

    stats = recorder.stats("first_item")

    assert stats is not None
    assert stats.count == 10
    assert stats.min_ms == 1.0
    assert stats.max_ms == 10.0
    assert stats.mean_ms == 5.5
    assert stats.p50_ms == 5.0
    assert stats.p90_ms == 9.0
    assert stats.p95_ms == 10.0


def test_stats_percentiles_with_twenty_samples() -> None:
    """二十个样本时 P90 取第 18 个、P95 取第 19 个。"""
    recorder = MetricRecorder()
    for value in range(1, 21):
        recorder.record("asr_latency", float(value))

    stats = recorder.stats("asr_latency")

    assert stats is not None
    assert stats.p50_ms == 10.0
    assert stats.p90_ms == 18.0
    assert stats.p95_ms == 19.0


def test_stats_ignores_insertion_order() -> None:
    """百分位按值排序算，与写入顺序无关。"""
    recorder = MetricRecorder()
    for value in (30.0, 10.0, 20.0):
        recorder.record("first_framework", value)

    stats = recorder.stats("first_framework")

    assert stats is not None
    assert stats.p50_ms == 20.0
    assert stats.min_ms == 10.0
    assert stats.max_ms == 30.0


# --------------------------------------------------------------------------
# samples / snapshot / reset
# --------------------------------------------------------------------------


def test_samples_by_metric_name() -> None:
    """按指标名取样本。"""
    recorder = MetricRecorder()
    recorder.record("ttft", 1.0)
    recorder.record("total_latency", 2.0)
    recorder.record("ttft", 3.0)

    samples = recorder.samples("ttft")

    assert [sample.value_ms for sample in samples] == [1.0, 3.0]


def test_samples_all_keeps_record_order() -> None:
    """取全部样本时按记录顺序。"""
    recorder = MetricRecorder()
    recorder.record("ttft", 1.0)
    recorder.record("total_latency", 2.0)
    recorder.record("ttft", 3.0)

    samples = recorder.samples()

    assert [(s.name, s.value_ms) for s in samples] == [
        ("ttft", 1.0),
        ("total_latency", 2.0),
        ("ttft", 3.0),
    ]


def test_samples_returns_copy() -> None:
    """改返回的列表不影响记录器内部。"""
    recorder = MetricRecorder()
    recorder.record("ttft", 1.0)

    recorder.samples().clear()
    recorder.samples("ttft").clear()

    assert len(recorder.samples()) == 1
    assert len(recorder.samples("ttft")) == 1


def test_snapshot_groups_by_metric() -> None:
    """snapshot 按指标名分组。"""
    recorder = MetricRecorder()
    recorder.record("ttft", 1.0)
    recorder.record("total_latency", 2.0)

    snapshot = recorder.snapshot()

    assert set(snapshot) == {"ttft", "total_latency"}
    assert [s.value_ms for s in snapshot["ttft"]] == [1.0]
    assert [s.value_ms for s in snapshot["total_latency"]] == [2.0]


def test_reset_one_metric() -> None:
    """reset 只清指定指标。"""
    recorder = MetricRecorder()
    recorder.record("ttft", 1.0)
    recorder.record("total_latency", 2.0)

    recorder.reset("ttft")

    assert recorder.samples("ttft") == []
    assert len(recorder.samples("total_latency")) == 1
    assert recorder.stats("ttft") is None


def test_reset_all_metrics() -> None:
    """reset 不带名字时清空全部。"""
    recorder = MetricRecorder()
    recorder.record("ttft", 1.0)
    recorder.record("total_latency", 2.0)

    recorder.reset()

    assert recorder.samples() == []
    assert recorder.snapshot() == {}
    assert recorder.stats("total_latency") is None


# --------------------------------------------------------------------------
# Timer
# --------------------------------------------------------------------------


def test_timer_records_sample_on_exit() -> None:
    """退出后耗时写进 recorder，sample 是那条样本。"""
    recorder = MetricRecorder()

    with Timer("ttft", source="mock", recorder=recorder) as timer:
        pass

    assert timer.elapsed_ms >= 0.0
    assert timer.sample is not None
    assert timer.sample.name == "ttft"
    assert timer.sample.source == "mock"
    assert timer.sample.value_ms == timer.elapsed_ms
    assert len(recorder.samples("ttft")) == 1


def test_timer_sample_is_none_before_exit() -> None:
    """退出前 sample 是 None。"""
    recorder = MetricRecorder()
    timer = Timer("ttft", recorder=recorder)

    with timer:
        assert timer.sample is None

    assert timer.sample is not None


def test_timer_without_recorder_keeps_sample_none() -> None:
    """没给 recorder 时只算时间不落库，退出后 sample 仍是 None。"""
    timer = Timer("ttft")

    with timer:
        pass

    assert timer.elapsed_ms >= 0.0
    assert timer.sample is None


def test_timer_elapsed_ms_readable_any_time() -> None:
    """elapsed_ms 随时可读，进入之前是 0。"""
    timer = Timer("request_latency")

    assert timer.elapsed_ms == 0.0

    with timer:
        assert timer.elapsed_ms >= 0.0

    assert timer.elapsed_ms >= 0.0


def test_timer_with_injected_clock() -> None:
    """注入自定义时钟后耗时是确定值。"""
    ticks = iter([10.0, 10.25])

    def clock() -> float:
        return next(ticks)

    recorder = MetricRecorder()
    with Timer("first_framework", recorder=recorder, clock=clock) as timer:
        pass

    assert timer.elapsed_ms == 250.0
    assert timer.sample is not None
    assert timer.sample.value_ms == 250.0


def test_timer_does_not_swallow_exception() -> None:
    """块里抛的异常照常往外传，计时仍然落库。"""
    recorder = MetricRecorder()

    with pytest.raises(RuntimeError, match="炸了"):
        with Timer("ttft", recorder=recorder) as timer:
            raise RuntimeError("炸了")

    assert timer.elapsed_ms >= 0.0
    assert len(recorder.samples("ttft")) == 1


def test_timer_rejects_unknown_metric_name_on_exit() -> None:
    """指标名非法时退出落库抛 ValueError。"""
    recorder = MetricRecorder()

    with pytest.raises(ValueError):
        with Timer("unknown_latency", recorder=recorder):
            pass


# --------------------------------------------------------------------------
# timed
# --------------------------------------------------------------------------


def test_timer_exit_annotations_resolve() -> None:
    """Timer.__exit__ 的注解在运行期能解析出来。

    注解里的 Any 必须在模块里导入，否则 get_type_hints 会抛 NameError。
    """
    hints = typing.get_type_hints(Timer.__exit__)

    assert isinstance(hints, dict)
    assert {"exc_type", "exc", "tb"} <= set(hints)
    assert hints["return"] is type(None)


def test_timed_returns_usable_timer() -> None:
    """timed 返回一个可用的 Timer。"""
    recorder = MetricRecorder()

    timer = timed("total_latency", source="openai", recorder=recorder)

    assert isinstance(timer, Timer)
    with timer:
        pass
    assert timer.sample is not None
    assert timer.sample.source == "openai"
    assert len(recorder.samples("total_latency")) == 1


def test_timed_without_recorder_only_measures() -> None:
    """timed 不给 recorder 时只计时不落库。"""
    timer = timed("capture_latency")

    with timer:
        pass

    assert timer.elapsed_ms >= 0.0
    assert timer.sample is None


def test_metric_stats_is_frozen_dataclass() -> None:
    """统计结果是不可变的数据类。"""
    stats = MetricStats(
        name="ttft",
        count=1,
        min_ms=1.0,
        max_ms=1.0,
        mean_ms=1.0,
        p50_ms=1.0,
        p90_ms=1.0,
        p95_ms=1.0,
    )

    with pytest.raises(Exception):
        stats.count = 2  # type: ignore[misc]


# --------------------------------------------------------------------------
# 进程级默认记录器
# --------------------------------------------------------------------------


def test_default_recorder_is_process_level() -> None:
    """默认记录器是同一个实例，各处记的样本都落在这里。"""
    reset_default_recorder()

    recorder = default_recorder()
    assert default_recorder() is recorder

    recorder.record("capture_latency", 12.0, source="mock")

    assert len(default_recorder().samples("capture_latency")) == 1


def test_reset_default_recorder_clears_samples() -> None:
    """重置默认记录器后样本清空，统计也没有了。"""
    reset_default_recorder()
    default_recorder().record("capture_latency", 12.0, source="mock")

    reset_default_recorder()

    assert default_recorder().samples() == []
    assert default_recorder().stats("capture_latency") is None


def test_timed_with_default_recorder_records_sample() -> None:
    """计时器接到默认记录器上时，退出后样本落进默认记录器。"""
    reset_default_recorder()

    with timed("capture_latency", source="mss", recorder=default_recorder()) as timer:
        pass

    samples = default_recorder().samples("capture_latency")
    assert len(samples) == 1
    assert samples[0].source == "mss"
    assert samples[0].value_ms == timer.elapsed_ms
