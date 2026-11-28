"""拼音级热词纠正器的测试。

覆盖《5-语音输入与流式ASR》「重识别与热词纠正」一节的行为：

- 2 字热词仅拼音完全相同（距离 0）才替换；
- 3 字以上热词允许拼音编辑距离 1 的近音替换；
- 距离超过阈值不替换，避免近音短词误伤；
- 保护区：拼音与文字都与热词相同的片段不再参与匹配；
- 空热词表与英文热词：纠正原样跳过；
- 热词表加载：注释行与空行忽略、文件缺失返回空表、
  环境变量覆盖默认路径；
- 中英文混排文本只动汉字段。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.audio.hotword_corrector import (
    DEFAULT_HOTWORDS_PATH,
    HOTWORDS_PATH_ENV,
    Correction,
    HotwordCorrector,
    default_hotwords_path,
    load_hotwords,
)

pytest.importorskip("pypinyin")


def test_two_char_hotword_replaced_on_same_pinyin() -> None:
    """2 字热词仅拼音完全相同（距离 0）才替换。"""
    corrector = HotwordCorrector(["队头", "分帧"])

    text, corrections = corrector.correct("什么是对头")
    assert text == "什么是队头"
    text, corrections = corrector.correct("这个分针技术")
    assert text == "这个分帧技术"


def test_two_char_hotword_near_miss_not_replaced() -> None:
    """2 字热词近音（距离 1）不替换，避免误伤。"""
    corrector = HotwordCorrector(["分帧"])
    # 分帧 fen/zhen 与 纷争 fen/zheng 距离 1，超过阈值 0
    text, corrections = corrector.correct("一场纷争")
    assert text == "一场纷争"
    assert corrections == []


def test_three_char_hotword_allows_distance_one() -> None:
    """3 字热词允许拼音编辑距离 1 的近音替换。"""
    corrector = HotwordCorrector(["霍夫曼"])
    text, _ = corrector.correct("请讲讲或夫曼")
    assert text == "请讲讲霍夫曼"


def test_correct_hotword_is_protected() -> None:
    """已经正确的热词不被处理：整词与子串都保持原样。"""
    corrector = HotwordCorrector(["队头", "队头阻塞", "多路复用"])
    text, corrections = corrector.correct("HTTP2 的队头阻塞问题")
    assert text == "HTTP2 的队头阻塞问题"
    assert corrections == []
    text, corrections = corrector.correct("支持多路复用")
    assert text == "支持多路复用"
    assert corrections == []


def test_same_pinyin_wrong_chars_not_protected() -> None:
    """同音错字（文字不同）不受保护区保护，照常替换。"""
    corrector = HotwordCorrector(["队头阻塞"])
    text, corrections = corrector.correct("什么是对头阻塞")
    assert text == "什么是队头阻塞"
    assert len(corrections) == 1
    assert corrections[0].original == "对头阻塞"


def test_empty_table_returns_text_unchanged() -> None:
    """空热词表原样返回，不做任何替换。"""
    corrector = HotwordCorrector([])
    text, corrections = corrector.correct("对头")
    assert text == "对头"
    assert corrections == []


def test_english_hotword_ignored() -> None:
    """纯英文热词不参与拼音匹配，文本不被改动。"""
    corrector = HotwordCorrector(["HPACK", "QUIC"])
    text, corrections = corrector.correct("HPACK 头部压缩与 QUIC")
    assert text == "HPACK 头部压缩与 QUIC"
    assert corrections == []


def test_mixed_text_only_han_runs_changed() -> None:
    """中英文混排文本只替换汉字段里的错字，英文与数字不动。"""
    corrector = HotwordCorrector(["队头阻塞", "HTTP2"])
    text, corrections = corrector.correct("HTTP2 的对头阻塞问题")
    assert text == "HTTP2 的队头阻塞问题"
    assert len(corrections) == 1


def test_punctuation_splits_han_runs() -> None:
    """标点把汉字段切开，各段独立匹配，多处错字各自替换。"""
    corrector = HotwordCorrector(["队头", "分帧"])
    text, corrections = corrector.correct("对头，分针。")
    assert text == "队头，分帧。"
    assert len(corrections) == 2


def test_correction_details() -> None:
    """替换明细带原文片段、热词写法与拼音编辑距离。"""
    corrector = HotwordCorrector(["队头阻塞"])
    text, corrections = corrector.correct("什么是对头阻塞")
    assert text == "什么是队头阻塞"
    assert corrections == [
        Correction(original="对头阻塞", replacement="队头阻塞", distance=0)
    ]


def test_load_hotwords_skips_comments_and_blanks(tmp_path: Path) -> None:
    """热词表每行一词，注释行与空行忽略。"""
    path = tmp_path / "hotwords.txt"
    path.write_text(
        "# 注释行\n\n队头阻塞\n  分帧  \n# 又一条注释\nQUIC\n",
        encoding="utf-8",
    )
    assert load_hotwords(path) == ["队头阻塞", "分帧", "QUIC"]


def test_load_hotwords_missing_file_returns_empty(tmp_path: Path) -> None:
    """热词表文件不存在时返回空表，不报错。"""
    assert load_hotwords(tmp_path / "missing.txt") == []


def test_default_hotwords_path_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """环境变量覆盖默认热词路径，空值回落默认路径。"""
    env_path = tmp_path / "words.txt"
    monkeypatch.setenv(HOTWORDS_PATH_ENV, str(env_path))
    assert default_hotwords_path() == env_path
    monkeypatch.setenv(HOTWORDS_PATH_ENV, "   ")
    assert default_hotwords_path() == DEFAULT_HOTWORDS_PATH
    monkeypatch.delenv(HOTWORDS_PATH_ENV)
    assert default_hotwords_path() == DEFAULT_HOTWORDS_PATH


def test_load_default_corrector_reads_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """默认纠正器从环境变量指定的文件加载热词。"""
    path = tmp_path / "words.txt"
    path.write_text("自定义热词\n", encoding="utf-8")
    monkeypatch.setenv(HOTWORDS_PATH_ENV, str(path))
    from app.audio.hotword_corrector import load_default_corrector

    corrector = load_default_corrector()
    assert corrector.hotwords == ("自定义热词",)