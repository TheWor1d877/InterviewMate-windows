#!/usr/bin/env python3
"""延迟测试 runner。

《8-集成与验收》性能统计一节要求：「用不少于 20 条代表性问题做延迟测试，
报告 P50 与 P95」。这个脚本就是跑那份延迟测试的命令行入口，
题目集是 :mod:`scripts.bench_questions` 里的固定清单。

用法::

    ./.venv/Scripts/python.exe scripts/bench_latency.py                  # 全量 24 条
    ./.venv/Scripts/python.exe scripts/bench_latency.py --limit 3        # 只跑前 3 条
    ./.venv/Scripts/python.exe scripts/bench_latency.py --out report.md  # 报告写文件
    ./.venv/Scripts/python.exe scripts/bench_latency.py --report-only    # 只读上次样本

跑真链路时换一份配置即可::

    ./.venv/Scripts/python.exe scripts/bench_latency.py --config config/real.yaml

配置里把 ``providers.llm`` 改成 ``deepseek``、``providers.retrieval`` 改成
``dify``，就会真的发请求；默认配置四个 provider 全是 ``mock``，不发网络请求。

组装一律走仓库自己的装配函数：模型客户端用 :func:`app.main.select_client`，
检索链路用 :func:`app.main.assemble_retrieval`，问题管理器用
:func:`app.main.assemble_manager`。请求体由
:class:`app.llm.prompt_manager.PromptManager` 构造，脚本自己不拼消息体，
口径与界面里跑的那条链路一致，见《8-集成与验收》性能统计一节。

界面不启动。装配函数要的是导航面板、内容面板、错误上报与来源存储四样东西，
脚本用仓库自己的两个面板类接住，事件照常派发，指标照常记录。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# 仓库根放进导入路径：直接跑脚本时 sys.path[0] 是 scripts/，
# 下面这些 app.* 与 scripts.* 的导入都要靠这一行。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import AppConfig, load_config
from app.core.exceptions import ConfigError
from app.core.question_manager import QuestionManager
from app.main import (
    EXIT_ARG_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_ERROR,
    EXIT_OK,
    assemble_manager,
    assemble_retrieval,
    close_client,
    close_retrieval,
    resolve_config_path,
    select_client,
)
from app.utils.timing import (
    METRIC_NAMES,
    MetricRecorder,
    MetricStats,
    default_recorder,
    reset_default_recorder,
)
from scripts.bench_questions import BENCH_QUESTIONS, BenchQuestion

__all__ = [
    "BENCH_DIR",
    "BenchRow",
    "build_parser",
    "main",
    "render_metric_summary",
    "render_report",
    "run_bench",
]

# 样本落盘目录。仓库根的 /runtime/ 已在 .gitignore 里，
# 这个目录下的 JSON 属于运行期产物，不进 Git。
BENCH_DIR: Path = REPO_ROOT / "runtime" / "bench"

# 两条题目之间的默认间隔秒数：别把服务打爆。
DEFAULT_INTERVAL_S: float = 0.5

# 逐题表里用到的四个耗时指标，名字取自《2-骨架与配置》性能指标一节。
_TTFT: str = "ttft"
_FIRST_FRAMEWORK: str = "first_framework"
_FIRST_ITEM: str = "first_item"
_TOTAL_LATENCY: str = "total_latency"
_REQUEST_LATENCY: str = "request_latency"


@dataclass
class BenchRow:
    """一条题目的实测结果。

    字段与《8-集成与验收》性能统计一节的报告表格一一对应。

    :param index: 序号，从 1 起。
    :param text: 题面原文。
    :param expected_type: 题目集里写的期望题型。
    :param actual_type: 实际判定出来的题型。
    :param expect_retrieval: 题目集里写的期望检索行为。
    :param actual_retrieval: 实际有没有发起检索。
    :param request_sent: 有没有真的把请求交给模型客户端。
    :param ttft_ms: 首 Token 到达耗时，毫秒。
    :param first_framework_ms: 首个框架事件到达耗时，毫秒。
    :param first_item_ms: 首个内容单元到达耗时，毫秒。
    :param total_latency_ms: 完整输出耗时，毫秒。
    :param wall_ms: 这一条题目的墙钟耗时，含检索，毫秒。
    """

    index: int
    text: str
    expected_type: str
    actual_type: str
    expect_retrieval: bool
    actual_retrieval: bool
    request_sent: bool
    ttft_ms: float | None
    first_framework_ms: float | None
    first_item_ms: float | None
    total_latency_ms: float | None
    wall_ms: float


class _HeadlessApp:
    """不启动界面的宿主，给仓库的装配函数用。

    装配函数要的是四样东西：左侧导航面板、右侧内容面板、错误上报、
    检索来源存储。这里用仓库自己的两个面板类接住前两样，
    后两样记在内存里，面板不显示但事件照常派发，
    指标照常按《8-集成与验收》性能统计一节记。
    """

    def __init__(self) -> None:
        # 面板类在函数里导入：只读样本出报告时不用把界面层拉进来
        from app.ui.panels.answer_panel import AnswerPanel
        from app.ui.panels.navigation_panel import NavigationPanel

        self._navigation = NavigationPanel()
        self._answer = AnswerPanel()
        self.errors: list[tuple[str, str]] = []
        self.rag_sources: list[dict] = []
        self.manager: QuestionManager | None = None

    @property
    def navigation_panel(self) -> Any:
        """左侧「回答流程」面板，供装配层建分发器用。"""
        return self._navigation

    @property
    def answer_panel(self) -> Any:
        """右侧「关键回答内容」面板，供装配层建分发器用。"""
        return self._answer

    def show_error(self, code: str, detail: str) -> None:
        """把错误记下来。

        :param code: 错误码，例如 ``E_DEEPSEEK_TIMEOUT``。
        :param detail: 一句话说明。
        """
        self.errors.append((code, detail))

    def attach_manager(self, manager: QuestionManager) -> None:
        """接住装配层造好的问题管理器。

        :param manager: 问题管理器。
        """
        self.manager = manager

    def store_rag_sources(self, entries: list[dict]) -> None:
        """把检索来源存下来，不开浮层。

        :param entries: 本次用到的来源条目。
        """
        self.rag_sources = list(entries)


# ----------------------------------------------------------------------
# 命令行
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    :returns: 解析器。
    """
    parser = argparse.ArgumentParser(
        prog="bench_latency.py",
        description=(
            "《8-集成与验收》性能统计一节的延迟测试 runner："
            "按题目集逐条提交，报 P50 与 P95"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="配置文件路径，默认 config/config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 条题目，默认全跑",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="把 Markdown 报告写到文件，默认只打印到终端",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="只读 runtime/bench/ 里已有的样本打印统计，不发任何请求",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help=f"两条题目之间的间隔秒数，默认 {DEFAULT_INTERVAL_S}",
    )
    return parser


# ----------------------------------------------------------------------
# 跑测试
# ----------------------------------------------------------------------
def _sample_counts(recorder: MetricRecorder) -> dict[str, int]:
    """取各指标当前的样本数，用来切分每条题目的样本。

    :param recorder: 记录器。
    """
    return {name: len(recorder.samples(name)) for name in METRIC_NAMES}


def _first_new_sample(
    recorder: MetricRecorder, name: str, before: int
) -> float | None:
    """取某个指标在本次提交中新落的第一条样本值。

    :param recorder: 记录器。
    :param name: 指标名。
    :param before: 提交前的样本数。
    :return: 毫秒值，没有新样本时返回 None。
    """
    samples = recorder.samples(name)
    if len(samples) <= before:
        return None
    return samples[before].value_ms


def _build_row(
    index: int,
    question: BenchQuestion,
    manager: QuestionManager,
    retrieve_calls: list[str],
    recorder: MetricRecorder,
    before: dict[str, int],
    wall_ms: float,
) -> BenchRow:
    """把一条题目的实测结果拼成一行。

    实际题型取当前会话上判定出来的题型，实际检索看检索回调有没有被调用，
    两者都是这次运行里真实发生的行为，不是另算一遍。

    :param index: 序号。
    :param question: 题目。
    :param manager: 问题管理器。
    :param retrieve_calls: 本次提交期间被调用的检索查询。
    :param recorder: 记录器。
    :param before: 提交前的各指标样本数。
    :param wall_ms: 本次提交的墙钟耗时，毫秒。
    """
    session = manager.current_session
    actual_type = session.question_type if session is not None else ""
    return BenchRow(
        index=index,
        text=question.text,
        expected_type=question.expected_type,
        actual_type=actual_type,
        expect_retrieval=question.expect_retrieval,
        actual_retrieval=bool(retrieve_calls),
        request_sent=(
            _first_new_sample(recorder, _REQUEST_LATENCY, before[_REQUEST_LATENCY])
            is not None
        ),
        ttft_ms=_first_new_sample(recorder, _TTFT, before[_TTFT]),
        first_framework_ms=_first_new_sample(
            recorder, _FIRST_FRAMEWORK, before[_FIRST_FRAMEWORK]
        ),
        first_item_ms=_first_new_sample(recorder, _FIRST_ITEM, before[_FIRST_ITEM]),
        total_latency_ms=_first_new_sample(
            recorder, _TOTAL_LATENCY, before[_TOTAL_LATENCY]
        ),
        wall_ms=wall_ms,
    )


async def run_bench(
    config_path: Path,
    questions: list[BenchQuestion],
    interval_s: float,
) -> tuple[list[BenchRow], MetricRecorder, AppConfig]:
    """按题目集顺序逐条提交，跑完返回逐题结果与记录器。

    :param config_path: 配置文件路径。
    :param questions: 要跑的题目。
    :param interval_s: 两条题目之间的间隔秒数。
    :returns: 逐题结果、记录器与这次用的配置。
    :raises ConfigError: 配置读不到或校验不过。
    """
    config = load_config(str(config_path))

    # 本次跑之前把进程级记录器清空，统计里只有这一轮的数据
    reset_default_recorder()
    recorder = default_recorder()

    app = _HeadlessApp()
    client = select_client(config)
    chain = assemble_retrieval(config, app)

    # 检索回调包一层，只为记下这次是不是真的发起了检索
    retrieve_calls: list[str] = []

    async def _counting_retrieve(query: str) -> tuple[str | None, list[dict]]:
        """记录一次检索调用，再转给仓库自己的检索实现。

        :param query: 检索查询，用题面。
        """
        retrieve_calls.append(query)
        return await chain.retrieve(query)

    manager = assemble_manager(
        config,
        client,
        app,
        retrieve=_counting_retrieve,
        on_sources=chain.on_sources,
    )

    rows: list[BenchRow] = []
    try:
        for index, question in enumerate(questions, 1):
            before = _sample_counts(recorder)
            retrieve_calls.clear()
            started = time.perf_counter()
            await manager.submit_text(question.text)
            wall_ms = (time.perf_counter() - started) * 1000.0
            row = _build_row(
                index, question, manager, retrieve_calls, recorder, before, wall_ms
            )
            rows.append(row)
            print(
                f"[{index}/{len(questions)}] {row.actual_type} "
                f"检索={'是' if row.actual_retrieval else '否'} "
                f"总耗时={row.wall_ms:.1f}ms",
                file=sys.stderr,
                flush=True,
            )
            if index < len(questions) and interval_s > 0:
                await asyncio.sleep(interval_s)
    finally:
        await close_retrieval(chain)
        await close_client(client)

    return rows, recorder, config


# ----------------------------------------------------------------------
# 样本落盘与读回
# ----------------------------------------------------------------------
def _display_path(path: Path) -> str:
    """把路径写成仓库相对路径，仓库外的写绝对路径。

    :param path: 路径。
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _build_payload(
    config: Any,
    config_path: Path,
    rows: list[BenchRow],
    recorder: MetricRecorder,
    started_at: str,
    finished_at: str,
    interval_s: float,
) -> dict[str, Any]:
    """把这次跑的原始数据拼成落盘的 JSON 结构。

    :param config: 应用配置。
    :param config_path: 配置文件路径。
    :param rows: 逐题结果。
    :param recorder: 记录器。
    :param started_at: 开跑时刻。
    :param finished_at: 跑完时刻。
    :param interval_s: 题目之间的间隔秒数。
    """
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": _display_path(config_path),
        "providers": {
            "llm": config.providers.llm,
            "retrieval": config.providers.retrieval,
        },
        "question_count": len(rows),
        "total_questions": len(BENCH_QUESTIONS),
        "interval_s": interval_s,
        "rows": [asdict(row) for row in rows],
        # 原始样本：指标名、毫秒值、来源、时间戳
        "samples": [
            {
                "name": sample.name,
                "value_ms": sample.value_ms,
                "source": sample.source,
                "ts": sample.ts,
            }
            for sample in recorder.samples()
        ],
    }


def _sample_file_path(moment: datetime) -> Path:
    """给这次跑算一个样本文件名，同一秒里跑两次也不覆盖。

    :param moment: 跑完的时刻。
    """
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"bench-{moment.strftime('%Y%m%d-%H%M%S')}"
    path = BENCH_DIR / f"{stem}.json"
    index = 1
    while path.exists():
        path = BENCH_DIR / f"{stem}-{index}.json"
        index += 1
    return path


def save_payload(payload: dict[str, Any], moment: datetime) -> Path:
    """把这次的原始样本写到 runtime/bench/ 下的 JSON 文件。

    :param payload: 落盘结构。
    :param moment: 跑完的时刻，用来算文件名。
    :returns: 写出的文件路径。
    """
    path = _sample_file_path(moment)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_latest_payload() -> tuple[dict[str, Any], Path]:
    """读回 runtime/bench/ 里最新一份样本。

    :returns: 落盘结构与读到的文件路径。
    :raises FileNotFoundError: 目录里没有样本文件。
    """
    if not BENCH_DIR.exists():
        raise FileNotFoundError(f"{_display_path(BENCH_DIR)} 目录不存在，没有样本可读")
    files = sorted(path for path in BENCH_DIR.glob("bench-*.json") if path.is_file())
    if not files:
        raise FileNotFoundError(f"{_display_path(BENCH_DIR)} 里没有样本文件")
    path = files[-1]
    return json.loads(path.read_text(encoding="utf-8")), path


def _stats_from_samples(samples: list[dict[str, Any]]) -> dict[str, MetricStats | None]:
    """按落盘的样本重算统计量。

    百分位用仓库自己的 :class:`~app.utils.timing.MetricRecorder` 算，
    口径与实时跑的那次一致，见《8-集成与验收》性能统计一节。

    :param samples: 原始样本列表。
    """
    recorder = MetricRecorder()
    for sample in samples:
        recorder.record(
            str(sample["name"]),
            float(sample["value_ms"]),
            source=str(sample.get("source", "")),
        )
    return {name: recorder.stats(name) for name in METRIC_NAMES}


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def _fmt_ms(value: float | None) -> str:
    """毫秒值写成三位小数，没有值时写短横线。

    :param value: 毫秒值。
    """
    if value is None:
        return "-"
    return f"{value:.3f}"


def _fmt_bool(value: bool) -> str:
    """布尔值写成「是」或「否」。

    :param value: 布尔值。
    """
    return "是" if value else "否"


def _provider_note(providers: dict[str, str]) -> str:
    """写一句 provider 是 mock 还是真实。

    :param providers: provider 取值，键是 llm 与 retrieval。
    """
    llm = providers.get("llm", "-")
    retrieval = providers.get("retrieval", "-")
    llm_note = "本地假流，不发网络请求" if llm == "mock" else "真实模型服务"
    retrieval_note = "本地片段池，不发网络请求" if retrieval == "mock" else "真实检索服务"
    return f"LLM={llm}（{llm_note}）；检索={retrieval}（{retrieval_note}）"


def _mismatches(rows: list[dict[str, Any]]) -> list[str]:
    """挑出实际行为与题目集期望不一致的题目。

    不一致要如实报出来：这条核对是为了暴露分类器与检索策略的真实行为，
    不是为了让它们通过，见《8-集成与验收》性能统计一节。

    :param rows: 逐题结果。
    """
    items: list[str] = []
    for row in rows:
        problems: list[str] = []
        if row["actual_type"] != row["expected_type"]:
            problems.append(
                f"题型期望 {row['expected_type']}，实际 {row['actual_type'] or '（没判定出）'}"
            )
        if bool(row["actual_retrieval"]) != bool(row["expect_retrieval"]):
            problems.append(
                f"检索期望 {_fmt_bool(bool(row['expect_retrieval']))}，"
                f"实际 {_fmt_bool(bool(row['actual_retrieval']))}"
            )
        if not problems:
            continue
        items.append(f"- 第 {row['index']} 条「{row['text']}」：" + "；".join(problems))
    return items


# 汇总表里没有样本的统计格一律写这个占位符，不用 0.0 冒充实测值
EMPTY_CELL: str = "—"


def _is_empty(item: MetricStats | None) -> bool:
    """判断一个指标是不是没有样本。

    :param item: 统计量，没有样本时可能为 None。
    """
    return item is None or item.count == 0


def _metric_row(name: str, item: MetricStats | None) -> str:
    """渲染汇总表里的一行。

    :param name: 指标名。
    :param item: 该指标的统计量，没有样本时为 None。
    """
    if _is_empty(item):
        return f"| {name} | 0 | " + " | ".join([EMPTY_CELL] * 4) + " |"
    assert item is not None  # 上面已挡掉无样本的情况，这里只是给类型检查看
    return (
        f"| {name} | {item.count} | {item.p50_ms:.3f} | {item.p90_ms:.3f} | "
        f"{item.p95_ms:.3f} | {item.max_ms:.3f} |"
    )


def render_metric_summary(stats: dict[str, MetricStats | None]) -> str:
    """按 :data:`~app.utils.timing.METRIC_NAMES` 顺序渲染指标汇总表。

    实时跑完与 ``--report-only`` 都走这个函数，同一份统计量输出逐字相同。
    九个指标一项不落；没有样本的指标写 count=0，P50、P90、P95 与 max
    写占位符 :data:`EMPTY_CELL`，不用 0.0 冒充实测值，
    见《8-集成与验收》性能统计一节。

    :param stats: 指标名到统计量的映射，没有样本的指标取值为 None。
    """
    lines: list[str] = [
        "## 指标汇总",
        "",
        "| 指标 | count | p50(ms) | p90(ms) | p95(ms) | max(ms) |",
        "|---|---|---|---|---|---|",
    ]
    for name in METRIC_NAMES:
        lines.append(_metric_row(name, stats.get(name)))
    return "\n".join(lines)


def render_report(
    payload: dict[str, Any],
    stats: dict[str, MetricStats | None],
    sample_file: str = "",
) -> str:
    """把一次运行的数据渲染成 Markdown 报告。

    段落顺序固定：说明、逐题表、指标汇总表、不一致清单、无样本指标清单，
    便于两次运行之间逐行对照。

    :param payload: 落盘结构。
    :param stats: 各指标的统计量。
    :param sample_file: 样本文件路径，写在说明段里。
    """
    rows: list[dict[str, Any]] = list(payload.get("rows", []))
    providers: dict[str, str] = dict(payload.get("providers", {}))
    question_count = int(payload.get("question_count", len(rows)))
    total_questions = int(payload.get("total_questions", len(BENCH_QUESTIONS)))

    lines: list[str] = []
    lines.append("# 延迟测试报告")
    lines.append("")
    lines.append(f"- 跑的时间：{payload.get('finished_at', '-')}")
    lines.append(f"- 配置来源：{payload.get('config_path', '-')}")
    scope = (
        f"{question_count} 条（题目集共 {total_questions} 条）"
        if question_count != total_questions
        else f"{question_count} 条"
    )
    # 真正把请求交给模型客户端的次数，少于题目条数说明有题目没发出去
    sent_requests = sum(1 for row in rows if row.get("request_sent"))
    lines.append(f"- 题目条数：{scope}")
    lines.append(f"- 模型请求：实际发出 {sent_requests} 次")
    lines.append(f"- provider：{_provider_note(providers)}")
    lines.append(f"- 题目之间的间隔：{payload.get('interval_s', '-')} 秒")
    if sample_file:
        lines.append(f"- 样本文件：{sample_file}")
    lines.append("")

    lines.append("## 逐题结果")
    lines.append("")
    lines.append(
        "| # | 题型（期望/实际） | 是否检索（期望/实际） | TTFT(ms) | "
        "首个框架(ms) | 首个内容单元(ms) | 完整输出(ms) | 总耗时(ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in rows:
        type_cell = f"{row['expected_type']} / {row['actual_type'] or '-'}"
        if row["actual_type"] != row["expected_type"]:
            type_cell += "（不一致）"
        retrieval_cell = (
            f"{_fmt_bool(bool(row['expect_retrieval']))} / "
            f"{_fmt_bool(bool(row['actual_retrieval']))}"
        )
        if bool(row["actual_retrieval"]) != bool(row["expect_retrieval"]):
            retrieval_cell += "（不一致）"
        lines.append(
            f"| {row['index']} | {type_cell} | {retrieval_cell} | "
            f"{_fmt_ms(row.get('ttft_ms'))} | {_fmt_ms(row.get('first_framework_ms'))} | "
            f"{_fmt_ms(row.get('first_item_ms'))} | {_fmt_ms(row.get('total_latency_ms'))} | "
            f"{_fmt_ms(row.get('wall_ms'))} |"
        )
    if not rows:
        lines.append("| - | - | - | - | - | - | - | - |")
    lines.append("")

    lines.append(render_metric_summary(stats))
    lines.append("")

    lines.append("## 与题目集期望不一致的题目")
    lines.append("")
    mismatch_items = _mismatches(rows)
    if mismatch_items:
        lines.append(f"共 {len(mismatch_items)} 条不一致：")
        lines.append("")
        lines.extend(mismatch_items)
    else:
        lines.append("无")
    lines.append("")

    lines.append("## 无样本指标")
    lines.append("")
    empty = [name for name in METRIC_NAMES if _is_empty(stats.get(name))]
    if empty:
        lines.append("以下指标这次没有样本：")
        lines.append("")
        for name in empty:
            lines.append(f"- {name}")
    else:
        lines.append("无")
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def _write_report(text: str, out_path: str | None) -> None:
    """把报告写到文件或打印到终端。

    :param text: 报告正文。
    :param out_path: 输出文件路径，None 时打印到终端。
    """
    if out_path is None:
        print(text)
        return
    path = Path(out_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"报告已写到 {_display_path(path)}")


def _run_from_args(args: argparse.Namespace) -> int:
    """跑一轮延迟测试并出报告。

    :param args: 命令行参数。
    """
    if args.limit is not None and args.limit < 1:
        print("--limit 至少是 1", file=sys.stderr)
        return EXIT_ARG_ERROR
    if args.interval < 0:
        print("--interval 不能是负数", file=sys.stderr)
        return EXIT_ARG_ERROR

    config_path = resolve_config_path(args.config)
    questions = list(BENCH_QUESTIONS)
    if args.limit is not None:
        questions = questions[: args.limit]

    started = datetime.now().astimezone()
    try:
        rows, recorder, config = asyncio.run(
            run_bench(config_path, questions, args.interval)
        )
    except ConfigError as exc:
        print(f"配置读取失败：{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except FileNotFoundError as exc:
        print(f"配置文件读不到：{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as exc:  # 兜底：跑挂了要说清楚，不静默退出
        print(f"延迟测试跑失败：{exc}", file=sys.stderr)
        return EXIT_ERROR

    finished = datetime.now().astimezone()
    payload = _build_payload(
        config,
        config_path,
        rows,
        recorder,
        started.isoformat(timespec="seconds"),
        finished.isoformat(timespec="seconds"),
        args.interval,
    )
    sample_path = save_payload(payload, finished)
    stats = {name: recorder.stats(name) for name in METRIC_NAMES}
    report = render_report(payload, stats, _display_path(sample_path))
    _write_report(report, args.out)
    return EXIT_OK


def _report_only(args: argparse.Namespace) -> int:
    """只读已有样本打印报告，不发任何请求。

    :param args: 命令行参数。
    """
    try:
        payload, sample_path = load_latest_payload()
    except FileNotFoundError as exc:
        print(f"没有可读的样本：{exc}", file=sys.stderr)
        return EXIT_ERROR
    except json.JSONDecodeError as exc:
        print(f"样本文件读不出来：{exc}", file=sys.stderr)
        return EXIT_ERROR
    stats = _stats_from_samples(list(payload.get("samples", [])))
    report = render_report(payload, stats, _display_path(sample_path))
    _write_report(report, args.out)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    :param argv: 参数列表，None 时取进程参数。
    :returns: 退出码。
    """
    args = build_parser().parse_args(argv)
    if args.report_only:
        return _report_only(args)
    return _run_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
