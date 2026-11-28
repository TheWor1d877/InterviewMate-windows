"""拼音级热词纠正。

见《5-语音输入与流式ASR》「重识别与热词纠正」一节：最终文本由
SenseVoiceSmall 重识别产出后，再由本模块把识别文本里的同音/近音错字
纠正成热词写法。

算法移植自 ``scripts/experiments/asr_compare_funasr.py`` 的拼音级热词
后处理（实验已验证：纯汉字 CER 0.0023，零误伤），阈值与保护区规则
原样保留：

- pypinyin 把识别文本与热词各自转成拼音序列，在识别文本的连续汉字段
  （两个以上汉字）里滑窗匹配热词拼音，编辑距离不超过阈值就替换成
  热词写法；
- 阈值 ``thr = int(热词拼音长度 × 0.35)``：2 字词仅拼音完全相同
  （距离 0）才替换，3 字词允许距离 1，越短越严，避免近音短词误伤；
- 保护区：片段的拼音与文字都与热词完全相同的不参与匹配，防止正确
  热词的子串被「补字」误替换；同音错字（文字不同）不在保护之列；
- 候选按「距离小、位置靠前、窗口长」排序后贪心取不重叠片段，
  多个片段从后往前替换，保证前段的下标不失效。

本模块纯逻辑、无模型依赖。热词表默认从 ``config/hotwords.txt`` 读，
每行一词，``#`` 开头是注释行，空行忽略；文件不存在或为空时热词表为空，
纠正直接原样跳过。路径可用环境变量 ``INTERVIEWMATE_HOTWORDS`` 覆盖。
表里的纯英文热词（HPACK、QUIC 等）不参与拼音匹配：英文写法靠模型
自己识别，拼音匹配只管汉字错法。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.utils.logger import log_event

try:  # pypinyin 缺失时热词纠正整体跳过，不影响识别链路
    from pypinyin import lazy_pinyin
except Exception as exc:  # pragma: no cover - 环境里装了 pypinyin，走不到
    lazy_pinyin = None  # type: ignore[assignment]
    _PYPINYIN_IMPORT_ERROR = str(exc)
else:
    _PYPINYIN_IMPORT_ERROR = ""

__all__ = [
    "HOTWORDS_PATH_ENV",
    "DEFAULT_HOTWORDS_DIR",
    "DEFAULT_HOTWORDS_PATH",
    "THR_RATIO",
    "Correction",
    "HotwordCorrector",
    "load_hotwords",
    "default_hotwords_path",
    "load_default_corrector",
]


# 热词路径的环境变量名，覆盖默认的 config/hotwords.txt
HOTWORDS_PATH_ENV: str = "INTERVIEWMATE_HOTWORDS"

# 默认热词表所在目录：仓库根 config/
DEFAULT_HOTWORDS_DIR: Path = Path(__file__).resolve().parents[2] / "config"

# 默认热词表路径
DEFAULT_HOTWORDS_PATH: Path = DEFAULT_HOTWORDS_DIR / "hotwords.txt"

# 阈值系数：thr = int(热词拼音长度 × THR_RATIO)，
# 见 scripts/experiments/asr_compare_funasr.py 实验取值
THR_RATIO: float = 0.35

# 汉字判定与连续汉字段提取，见实验脚本
HAN_RE: re.Pattern[str] = re.compile(r"[\u4e00-\u9fff]")
HAN_RUN_RE: re.Pattern[str] = re.compile(r"[\u4e00-\u9fff]{2,}")


@dataclass(frozen=True)
class Correction:
    """一处热词替换的明细。

    :param original: 被替换的原文片段。
    :param replacement: 替换成热词写法。
    :param distance: 拼音序列的编辑距离。
    """

    original: str
    replacement: str
    distance: int


def han_to_pinyin_list(han: str) -> list[str]:
    """逐单字注音，返回拼音序列。

    逐字注音绕开 pypinyin 词组库的多音字差异：同一套注音体系下，
    同音字的拼音序列一致，匹配才有可比性。

    :param han: 纯汉字串。
    :returns: 每个汉字一个拼音的列表。
    """
    out: list[str] = []
    for ch in han:
        out.extend(lazy_pinyin(ch))
    return out


def edit_distance(a: "str | list[str]", b: "str | list[str]") -> int:
    """编辑距离，元素可以是字符也可以是拼音（按整体比较）。

    滑窗都很短（热词拼音长度以内），纯 DP 足够，不引第三方加速。

    :param a: 序列一。
    :param b: 序列二。
    :returns: 最少编辑操作数。
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i]
        for j in range(1, lb + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1])))
        prev = cur
    return prev[-1]


def load_hotwords(path: str | Path) -> list[str]:
    """读热词表文件，返回热词列表。

    每行一词，``#`` 开头是注释行，空行忽略；文件不存在或为空时返回
    空列表，调用方拿空表时纠正整体跳过，不报错。

    :param path: 热词表路径。
    :returns: 热词列表，按文件里的顺序。
    """
    file_path = Path(path)
    if not file_path.is_file():
        return []
    words: list[str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line)
    return words


def default_hotwords_path() -> Path:
    """热词表的默认路径，环境变量 :data:`HOTWORDS_PATH_ENV` 优先。

    :returns: 热词表路径，环境变量给了空串时回落默认值。
    """
    env_value = os.environ.get(HOTWORDS_PATH_ENV, "").strip()
    if env_value:
        return Path(env_value)
    return DEFAULT_HOTWORDS_PATH


def load_default_corrector() -> "HotwordCorrector":
    """从默认路径加载热词纠正器。

    :returns: 纠正器，热词表不存在或为空时是空表纠正器，
        纠正调用直接原样跳过。
    """
    return HotwordCorrector(load_hotwords(default_hotwords_path()))


class HotwordCorrector:
    """拼音级热词纠正器。

    :param hotwords: 热词列表，空表时 :meth:`correct` 原样返回。
    """

    def __init__(self, hotwords: Iterable[str] = ()) -> None:
        self._hotwords: list[str] = [
            word.strip() for word in hotwords if word.strip()
        ]
        # 拼音索引：热词 → (热词汉字部分, 拼音序列)。只收两个以上汉字的
        # 热词：单字纠正要靠词组上下文，本算法不做单字替换；
        # 纯英文热词不进拼音索引，英文写法靠模型自己识别。
        self._pinyin: dict[str, tuple[str, list[str]]] = {}
        if lazy_pinyin is not None:
            for word in self._hotwords:
                han = "".join(HAN_RE.findall(word))
                if len(han) >= 2:
                    self._pinyin[word] = (han, han_to_pinyin_list(han))

    @property
    def hotwords(self) -> tuple[str, ...]:
        """热词表，按加载顺序。"""
        return tuple(self._hotwords)

    def correct(self, text: str, *, thr_ratio: float = THR_RATIO) -> tuple[str, list[Correction]]:
        """对识别文本做热词纠正。

        :param text: 识别文本。
        :param thr_ratio: 阈值系数，实验默认 0.35。
        :returns: ``(纠正后文本, 替换明细)``。没有替换时明细是空列表，
            文本原样返回；pypinyin 不可用或热词表为空时同样原样返回。
        """
        if not text or not self._pinyin:
            return text, []
        if lazy_pinyin is None:
            # 只在真要用时提示：环境里缺 pypinyin，热词纠正整体跳过
            log_event(
                "hotword.corrector_skipped",
                "pypinyin 不可用，热词纠正跳过",
                level="WARNING",
                error=_PYPINYIN_IMPORT_ERROR,
            )
            return text, []

        result = list(text)
        corrections: list[Correction] = []
        runs = list(HAN_RUN_RE.finditer(text))
        for match in reversed(runs):
            # 从后往前替换：前段的下标不因后段长度变化而失效
            start, end = match.start(), match.end()
            run = match.group()
            py = han_to_pinyin_list(run)

            # 第一遍：拼音与文字都和热词完全相同的片段记为保护区，
            # 正确热词的子串不再参与匹配，防止被「补字」误替换；
            # 同音错字文字不同，不受保护
            protected: list[tuple[int, int]] = []
            for hotword, (hotword_han, hotword_py) in self._pinyin.items():
                word_len = len(hotword_py)
                for s in range(0, len(py) - word_len + 1):
                    if py[s : s + word_len] == hotword_py and run[s : s + word_len] == hotword_han:
                        protected.append((s, s + word_len))

            # 第二遍：收集候选（片段文字与热词不同才可能进候选，
            # 文字相同且拼音相同的片段已在保护区里）
            candidates: list[tuple[int, int, int, str]] = []
            for hotword, (hotword_han, hotword_py) in self._pinyin.items():
                threshold = int(len(hotword_py) * thr_ratio)
                for window in range(2, min(len(hotword_py), len(py)) + 1):
                    # 窗口比热词拼音短得超过阈值时距离必然超限，跳过
                    if len(hotword_py) - window > threshold:
                        continue
                    for s in range(0, len(py) - window + 1):
                        e = s + window
                        if any(ps <= s < pe or ps < e <= pe for ps, pe in protected):
                            continue
                        distance = edit_distance(py[s:e], hotword_py)
                        if distance <= threshold:
                            candidates.append((distance, s, e, hotword_han))
            if not candidates:
                continue
            # 距离小者优先；同距离取靠前、窗口更长的片段
            candidates.sort(key=lambda item: (item[0], item[1], -item[2]))
            chosen: list[tuple[int, int, int, str]] = []
            last_end = -1
            for candidate in candidates:
                s, e = candidate[1], candidate[2]
                if s >= last_end:
                    chosen.append(candidate)
                    last_end = e
            for _distance, s, e, hotword_han in reversed(chosen):
                corrections.append(
                    Correction(
                        original=run[s:e],
                        replacement=hotword_han,
                        distance=_distance,
                    )
                )
                result[start + s : start + e] = list(hotword_han)
        return "".join(result), corrections

    def correct_text(self, text: str) -> str:
        """只要纠正后文本，不要明细。

        :param text: 识别文本。
        :returns: 纠正后的文本。
        """
        return self.correct(text)[0]