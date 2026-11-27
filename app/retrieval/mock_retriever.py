"""假检索后端。

``providers.retrieval`` 取 ``mock`` 时用这个后端，见《2-骨架与配置》配置结构一节：
不连 Dify、不发网络请求，按《6-Dify检索与项目知识》检索接口抽象一节的
:class:`~app.retrieval.base.RetrievalProvider` 形状返回本地片段。

这个后端只服务于本地开发与 CI：它让「判定要不要检索 → 发起检索 → 整理片段 →
拼上下文 → 展示来源」整条链路在没有外部服务时也能跑通并被验证，
写法照《4-截图输入与多模态》假截图后端一节。

片段内容取自 InterviewMate 自身的模块、配置与设计约定，不是通用占位文字：
模型拿到这些片段时答的是本项目的事。片段的顺序与分数跟着查询里的关键词走，
满足《6-Dify检索与项目知识》结果整理一节按分数从高到低排列片段的规则。

这个后端不做任何平台判断，不发网络请求，也不写文件。

检索耗时同样单独计时，指标名 ``retrieval_latency``，来源是后端名，
见《8-集成与验收》性能统计一节。假后端也计时，
这样 mock 模式下这个指标不会是空的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.retrieval.base import RetrievalProvider, retrieval_timer
from app.retrieval.retrieval_models import RetrievalOutcome, RetrievalResult

__all__ = [
    "MockRetriever",
    "MockFragment",
    "DEFAULT_FRAGMENT_POOL",
    "DEFAULT_MIN_COUNT",
    "DEFAULT_MAX_COUNT",
]

# 默认返回的片段条数区间，上界与《6-Dify检索与项目知识》检索参数一节的 top_k 一致
DEFAULT_MIN_COUNT: int = 2
DEFAULT_MAX_COUNT: int = 4

# 命中一个关键词给多少分。基础分最高 0.47，命中一个关键词就到 0.49 以上，
# 命中与没命中的片段在分数上分得开。
_MATCH_SCORE_STEP: float = 0.08

# 分数上限
_MAX_SCORE: float = 0.97

# 查询里切出的英文词，最短两个字符
_ASCII_WORD_PATTERN = re.compile(r"[a-z0-9_]{2,}")


@dataclass(frozen=True)
class MockFragment:
    """片段池里的一条内容。

    :param content: 片段正文。
    :param document_name: 文档名，模块路径或文档小节名。
    :param keywords: 命中这条片段的关键词。中文按子串匹配，英文按整词匹配。
    :param score: 基础分，没命中关键词时用这个分。
    """

    content: str
    document_name: str
    keywords: tuple[str, ...] = ()
    score: float = 0.6


# 默认片段池，内容取自 InterviewMate 自身的模块、配置与设计约定。
# 顺序即默认顺序：查询没命中任何关键词时按这个顺序返回。
DEFAULT_FRAGMENT_POOL: tuple[MockFragment, ...] = (
    MockFragment(
        content=(
            "事件总线负责模块间的事件流转。分发键取 getattr(event, \"type\", None)，"
            "取不到时回退成 type(event).__name__；发布事件时按订阅顺序把事件"
            "投递给该类型的所有 handler；订阅与取消订阅都按事件类型登记。"
        ),
        document_name="app/core/event_bus.py",
        keywords=("事件总线", "事件", "订阅", "分发", "event_bus", "handler"),
        score=0.47,
    ),
    MockFragment(
        content=(
            "问题处理模块负责一次提问的完整编排：先判定题型，再决定要不要检索项目知识，"
            "然后组织请求发给模型客户端，接收流式事件并转成界面事件。"
            "题型与检索需求是两个独立字段，代码题也可能同时是项目题。"
        ),
        document_name="app/core/question_manager.py",
        keywords=("问题处理", "题型", "检索决策", "question_manager", "编排"),
        score=0.46,
    ),
    MockFragment(
        content=(
            "配置分 app、ui、hotkeys、audio、asr、screenshot、deepseek、rag、coding、"
            "privacy、providers 几段。providers 下四个键各取 mock 或真实实现名："
            "llm 取 mock 或 deepseek，retrieval 取 mock 或 dify，asr 取 mock 或 funasr，"
            "screenshot 取 mock 或 mss。开发期四个键全设 mock。"
        ),
        document_name="《2-骨架与配置》配置结构",
        keywords=("配置", "providers", "mock", "dify", "rag", "config", "top_k"),
        score=0.45,
    ),
    MockFragment(
        content=(
            "进程通信走本机 TCP loopback（回环地址上的随机端口），一行一个 JSON。"
            "请求体是 action 与 request_id，响应体是 ok、error 与 payload；"
            "action 与 request_id 必须是字符串，"
            "ok 必须是布尔值，payload 必须是对象，类型不对就抛 SocketError，"
            "对应错误码 E_SOCKET。"
        ),
        document_name="app/runtime/socket_protocol.py",
        keywords=("进程通信", "socket", "ipc", "协议", "socket_protocol"),
        score=0.43,
    ),
    MockFragment(
        content=(
            "命令行入口提供 start、warmup、stop、status、test、devices、audio-test "
            "七个命令，hotkey 命令下还有截图等子动作。start 按顺序读配置并校验必填项、"
            "检查依赖包、检查音频设备与截图接口、预加载 ASR，最后启动 Textual 界面。"
        ),
        document_name="app/cli.py",
        keywords=("命令行", "cli", "入口", "子命令", "start", "status", "warmup"),
        score=0.46,
    ),
    MockFragment(
        content=(
            "主布局是竖排四段：顶部状态栏、中间左右两栏内容区、底部文字输入框、"
            "输入框下面的快捷键提示栏。左右两栏按 ui.left_ratio 与 ui.right_ratio 分，"
            "默认 22 比 78；终端变窄时两栏结构不变，右栏占满剩余空间。"
        ),
        document_name="app/ui/layout.py",
        keywords=("两栏", "布局", "界面", "layout", "左栏", "右栏", "textual", "状态栏"),
        score=0.47,
    ),
    MockFragment(
        content=(
            "截图走 Windows 侧的 mss 后端：直接在内存里抓屏，整屏取虚拟桌面、"
            "主屏按系统主屏尺寸匹配、指定屏按序号取，抓到的位图转成 PNG 字节交给"
            "主进程，全程不落盘，屏幕上不弹任何确认框。"
        ),
        document_name="app/platform/windows/screenshot.py",
        keywords=("截图扩展", "截图", "mss", "抓屏", "windows"),
        score=0.42,
    ),
    MockFragment(
        content=(
            "providers.screenshot 取 mock 时用内置假后端：不抓真实屏幕，"
            "每次采集都返回同一张固定的小 PNG（320x180）。假后端不做任何平台判断，"
            "这条链路的验证不需要真去抓屏。"
        ),
        document_name="《4-截图输入与多模态》假截图后端",
        keywords=("假后端", "假截图", "截图", "screenshot", "png", "mock"),
        score=0.41,
    ),
    MockFragment(
        content=(
            "模型输出是一行一个 JSON 的流式事件流，事件类型有 framework、item、"
            "section_done、done；解析器容忍字段缺失与类型不对，不让坏事件打断整段输出。"
            "代码题另外产出 code_delta 与 code_complete。"
        ),
        document_name="app/llm/stream_parser.py",
        keywords=("流式", "事件流", "模型", "deepseek", "stream", "解析", "prompt"),
        score=0.45,
    ),
    MockFragment(
        content=(
            "音频采集分麦克风与系统回环两路，分别采集再混合；采样率 16000、单声道、"
            "按 200 毫秒分块，块队列有界。语音解码成文本后作为问题投递到事件总线，"
            "与文字输入走同一条处理路径。"
        ),
        document_name="app/audio/audio_source.py",
        keywords=("语音", "音频", "麦克风", "asr", "采样率", "audio", "回环"),
        score=0.43,
    ),
    MockFragment(
        content=(
            "检索框架可以替换，问题处理模块与模型客户端只看 RetrievalProvider 接口，"
            "不关心底层是 Dify 还是别的框架。当前实现是 DifyRetriever，"
            "向 {base_url}/datasets/{dataset_id}/retrieve 发 POST；"
            "检索参数 top_k 默认 4、超时 8 秒，检索上下文总量截断到 8000 字符。"
        ),
        document_name="《6-Dify检索与项目知识》检索接口抽象",
        keywords=("检索", "dify", "知识库", "rag", "检索接口", "provider", "top_k"),
        score=0.46,
    ),
)


class MockRetriever(RetrievalProvider):
    """假检索后端，实现 :class:`app.retrieval.base.RetrievalProvider` 协议。

    检索不发网络请求，片段来自本地片段池，顺序与分数跟着查询里的关键词走。

    :param count: 固定返回的片段条数。None 时按查询命中的条数决定，
        命中数落在 2 到 4 条之间，见《6-Dify检索与项目知识》检索参数一节的 top_k。
    :param pool: 片段池，None 时用 :data:`DEFAULT_FRAGMENT_POOL`。
    """

    # 后端名，对应配置里 ``providers.retrieval`` 的 mock 取值
    name: str = "mock"

    def __init__(
        self,
        *,
        count: int | None = None,
        pool: Sequence[MockFragment] | None = None,
    ) -> None:
        self._count = None if count is None else max(0, int(count))
        self._pool = tuple(pool) if pool is not None else DEFAULT_FRAGMENT_POOL

    @property
    def count(self) -> int | None:
        """固定返回的片段条数，None 表示按查询命中的条数决定。"""
        return self._count

    @property
    def pool(self) -> tuple[MockFragment, ...]:
        """片段池。"""
        return self._pool

    async def retrieve(self, query: str) -> list[RetrievalResult]:
        """按查询检索，返回本地片段列表。

        这次检索的耗时记进 ``retrieval_latency``，来源是后端名，
        见《8-集成与验收》性能统计一节。

        :param query: 检索查询，查询里的关键词决定片段的顺序与分数。
        :return: 命中的片段列表，按分数从高到低排列，
            字段见 :class:`~app.retrieval.retrieval_models.RetrievalResult`。
        """
        with retrieval_timer(self.name):
            return self._search(query)

    async def retrieve_outcome(self, query: str) -> RetrievalOutcome:
        """按查询检索，返回带降级状态的完整结果。

        方法形状与 :meth:`app.retrieval.dify_retriever.DifyRetriever.retrieve_outcome`
        一致。假后端不连外部服务，不会失败，所以永远返回 ``ok=True``、
        ``degraded=False`` 的结果，见《6-Dify检索与项目知识》故障降级一节。

        走 :meth:`retrieve` 取片段，检索耗时只记一条。

        :param query: 检索查询。
        :return: 检索结果，``error_code`` 为 None，``error_detail`` 为空串。
        """
        results = await self.retrieve(query)
        return RetrievalOutcome.success(results)

    async def aclose(self) -> None:
        """没有连接要关，空实现。"""
        return None

    def _search(self, query: str) -> list[RetrievalResult]:
        """按关键词给片段排序并取前几条。"""
        text = query if isinstance(query, str) else str(query or "")
        lowered = text.lower()
        words = set(_ASCII_WORD_PATTERN.findall(lowered))
        matched: list[tuple[MockFragment, tuple[str, ...]]] = []
        others: list[tuple[MockFragment, tuple[str, ...]]] = []
        for fragment in self._pool:
            hits = tuple(key for key in fragment.keywords if _hit(key, lowered, words))
            if hits:
                matched.append((fragment, hits))
            else:
                others.append((fragment, hits))
        # 命中的排在前面，命中多的在前；同组内按基础分排，分数相同时保持原顺序
        matched.sort(key=lambda item: (-len(item[1]), -item[0].score))
        others.sort(key=lambda item: -item[0].score)
        limit = self._limit(len(matched))
        return [
            self._to_result(index, fragment, hits)
            for index, (fragment, hits) in enumerate(
                (matched + others)[:limit], start=1
            )
        ]

    def _limit(self, matched_count: int) -> int:
        """这次返回几条片段。"""
        if self._count is not None:
            return self._count
        # 没指定条数时按命中数取，至少两条、最多四条
        return min(max(DEFAULT_MIN_COUNT, matched_count), DEFAULT_MAX_COUNT)

    def _to_result(
        self,
        index: int,
        fragment: MockFragment,
        hits: tuple[str, ...],
    ) -> RetrievalResult:
        """把片段池里的一条内容变成检索结果。

        分数跟着命中的关键词数走，保证返回顺序与按分数从高到低排列一致，
        见《6-Dify检索与项目知识》结果整理一节。
        """
        score = min(_MAX_SCORE, fragment.score + _MATCH_SCORE_STEP * len(hits))
        return RetrievalResult(
            content=fragment.content,
            document_name=fragment.document_name,
            score=round(score, 3),
            metadata={
                "position": index,
                "segment_id": f"mock-segment-{index}",
                "source": "mock",
                "matched_keywords": list(hits),
            },
        )


def _hit(keyword: str, lowered: str, words: set[str]) -> bool:
    """关键词是否出现在查询里。

    英文关键词按整词匹配，避免 ``cli`` 命中 ``client``；
    中文不分词，按子串匹配。

    :param keyword: 片段声明的关键词。
    :param lowered: 查询的小写形式。
    :param words: 查询里切出的英文词。
    """
    key = keyword.strip().lower()
    if not key:
        return False
    if key.isascii():
        return key in words
    return key in lowered
