"""延迟测试报告里指标汇总表的测试。

这个文件盯住汇总表的几条冻结约定：

- 按 :data:`app.utils.timing.METRIC_NAMES` 顺序列出全部九项，一项不漏；
- 没有样本的指标 count 写 0，P50、P90、P95 与 max 写占位符「—」，
  不用 0.0 冒充实测值；
- 有样本时百分位按最近秩法算，数值正确；
- ``--report-only`` 与实时跑完的汇总走同一个格式化函数，同一份统计量
  输出的汇总表逐字相同。

样本直接用 :class:`app.utils.timing.MetricRecorder` 构造，
不跑 bench：真跑会连真实 LLM 与 ASR。
"""

from __future__ import annotations

import pytest

from app.utils.timing import METRIC_NAMES, MetricRecorder
from scripts.bench_latency import (
    _stats_from_samples,
    main,
    render_metric_summary,
    render_report,
)

# 无样本时的统计列占位符，见 bench_latency 模块里的 EMPTY_CELL
PLACEHOLDER = "—"

# 文本链路能产出的指标：语音与截图链路才有的 capture/asr 三项没有样本
TEXT_LINK_METRICS = (
    "retrieval_latency",
    "request_latency",
    "ttft",
    "first_framework",
    "first_item",
    "total_latency",
)


def _summary_rows(text: str) -> list[list[str]]:
    """把汇总表的数据行拆成单元格列表，表头与分隔行不算。"""
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] == "指标" or set(cells[0]) <= {"-"}:
            continue
        rows.append(cells)
    return rows


def _payload() -> dict:
    """一份最小但结构完整的落盘数据，供 render_report 用。"""
    return {
        "started_at": "2026-01-01T00:00:00+08:00",
        "finished_at": "2026-01-01T00:01:00+08:00",
        "config_path": "config/config.yaml",
        "providers": {"llm": "mock", "retrieval": "mock"},
        "question_count": 0,
        "total_questions": 24,
        "interval_s": 0.0,
        "rows": [],
        "samples": [],
    }


def _record_text_metrics() -> MetricRecorder:
    """记一组文本链路的样本，ttft 取 1..10 毫秒便于手算百分位。"""
    recorder = MetricRecorder()
    for name in TEXT_LINK_METRICS:
        recorder.record(name, 100.0, source="mock")
    recorder.reset("ttft")
    for value in range(1, 11):
        recorder.record("ttft", float(value), source="mock")
    return recorder


def _live_stats(recorder: MetricRecorder) -> dict:
    """实时路径取统计量的方式，与跑完后出报告一致。"""
    return {name: recorder.stats(name) for name in METRIC_NAMES}


def _report_only_stats(recorder: MetricRecorder) -> dict:
    """只读路径取统计量的方式：从落盘样本重算。"""
    samples = [
        {
            "name": sample.name,
            "value_ms": sample.value_ms,
            "source": sample.source,
            "ts": sample.ts,
        }
        for sample in recorder.samples()
    ]
    return _stats_from_samples(samples)


def test_summary_lists_all_metrics_in_metric_names_order() -> None:
    """汇总表按 METRIC_NAMES 顺序列出九项，一项不漏、一行不多。"""
    rows = _summary_rows(render_metric_summary({}))
    assert [row[0] for row in rows] == list(METRIC_NAMES)
    assert len(rows) == len(METRIC_NAMES) == 9


def test_zero_sample_rows_show_placeholder_not_zero() -> None:
    """没有样本的指标 count 写 0，四个统计列全写占位符。"""
    rows = _summary_rows(render_metric_summary({}))
    for row in rows:
        assert row[1] == "0", row
        assert row[2:] == [PLACEHOLDER] * 4, row
        assert "0.0" not in " ".join(row), row


def test_text_link_report_keeps_asr_rows_empty() -> None:
    """文本链路只产出六项样本，capture/asr 三项照样在表里，写占位符。"""
    recorder = _record_text_metrics()
    rows = {row[0]: row for row in _summary_rows(render_metric_summary(_live_stats(recorder)))}

    for name in ("capture_latency", "asr_latency", "asr_final_latency"):
        assert rows[name][1] == "0", rows[name]
        assert rows[name][2:] == [PLACEHOLDER] * 4, rows[name]

    for name in TEXT_LINK_METRICS:
        expected_count = "10" if name == "ttft" else "1"
        assert rows[name][1] == expected_count, rows[name]
        assert rows[name][2] != PLACEHOLDER, rows[name]


def test_percentiles_with_samples_are_correct() -> None:
    """十个样本时按最近秩法：P50=5、P90=9、P95=10、max=10。"""
    recorder = _record_text_metrics()
    rows = {row[0]: row for row in _summary_rows(render_metric_summary(_live_stats(recorder)))}

    ttft = rows["ttft"]
    assert ttft[1] == "10"
    assert ttft[2] == "5.000"
    assert ttft[3] == "9.000"
    assert ttft[4] == "10.000"
    assert ttft[5] == "10.000"


def test_report_only_and_live_share_one_formatter() -> None:
    """两条路径走同一个格式化函数，同一份样本输出的汇总表逐字相同。"""
    recorder = _record_text_metrics()
    live = _live_stats(recorder)
    report_only = _report_only_stats(recorder)

    live_summary = render_metric_summary(live)
    assert live_summary == render_metric_summary(report_only)

    live_report = render_report(_payload(), live)
    only_report = render_report(_payload(), report_only)
    assert live_report == only_report
    assert live_report.count("## 指标汇总") == 1
    assert live_summary in live_report


def test_help_exits_zero() -> None:
    """命令行 --help 正常退出，退出码 0。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0