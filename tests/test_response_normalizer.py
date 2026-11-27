"""兜底整理器的测试。

覆盖《7-代码问题模式》纯代码块兜底一节对整理模块的要求：
模型不按事件协议输出、直接给一段带 markdown 代码围栏的文本时，
代码正文与说明文字要拆开，代码正文一个字符都不改。

既有的整理规则见《3-文本输入与流式回答》，对应用例放在
``tests/test_event_dispatcher.py``，这里只补代码块拆分与语言标记的用例。
"""

from __future__ import annotations

from app.output.response_normalizer import ResponseNormalizer


# ----------------------------------------------------------------------
# 纯代码块兜底
# ----------------------------------------------------------------------
def test_extract_code_blocks_splits_code_and_note() -> None:
    """围栏里的内容是代码，围栏之外的文字是说明文字。"""
    text = "先说思路\n\n```cpp\nint main() {\n    return 0;\n}\n```\n\n复杂度 O(n)"

    code, note = ResponseNormalizer.extract_code_blocks(text)

    assert code == "int main() {\n    return 0;\n}"
    assert note == "先说思路\n\n复杂度 O(n)"


def test_extract_code_blocks_without_fence_treats_all_as_note() -> None:
    """没有围栏时整段都当说明文字，代码正文是空串。"""
    code, note = ResponseNormalizer.extract_code_blocks("  普通文字  \n\n\n第二段   ")

    assert code == ""
    assert note == "普通文字\n\n第二段"


def test_extract_code_blocks_keeps_code_verbatim() -> None:
    """代码正文保留原始缩进与换行，一个字符都不改。"""
    code = "if (a) {\n\t\treturn 0;\n}"

    extracted, _note = ResponseNormalizer.extract_code_blocks(f"```cpp\n{code}\n```")

    assert extracted == code


def test_extract_code_blocks_with_only_code_has_empty_note() -> None:
    """只有代码时说明文字是空串。"""
    code, note = ResponseNormalizer.extract_code_blocks("```cpp\nint x = 1;\n```")

    assert code == "int x = 1;"
    assert note == ""


def test_extract_code_blocks_joins_notes_in_order() -> None:
    """围栏前后的说明文字按出现顺序拼接。"""
    text = "前面一句\n\n```cpp\nint x = 1;\n```\n\n后面一句"

    code, note = ResponseNormalizer.extract_code_blocks(text)

    assert code == "int x = 1;"
    assert note == "前面一句\n\n后面一句"


def test_extract_code_blocks_joins_multiple_blocks() -> None:
    """多个围栏块的代码按顺序拼在一起。"""
    text = "```cpp\nint a = 1;\n```\n\n```cpp\nint b = 2;\n```"

    code, note = ResponseNormalizer.extract_code_blocks(text)

    assert code == "int a = 1;\nint b = 2;"
    assert note == ""


def test_extract_code_blocks_with_unclosed_fence_keeps_rest_as_code() -> None:
    """围栏没闭合时，剩下的内容都算代码。"""
    code, note = ResponseNormalizer.extract_code_blocks("```cpp\nint x = 1;")

    assert code == "int x = 1;"
    assert note == ""


def test_extract_code_blocks_on_empty_text() -> None:
    """空文本拆出两个空串。"""
    assert ResponseNormalizer.extract_code_blocks("") == ("", "")


def test_extract_code_blocks_cleans_the_note() -> None:
    """说明文字按兜底整理的规则收拾：去首尾空白、合并多余空行。"""
    text = "  说明一句   \n\n\n\n另一句  \n\n```cpp\nint x = 1;\n```"

    code, note = ResponseNormalizer.extract_code_blocks(text)

    assert code == "int x = 1;"
    assert note == "说明一句\n\n另一句"


# ----------------------------------------------------------------------
# 围栏上的语言标记
# ----------------------------------------------------------------------
def test_detect_code_language_recognizes_cpp_aliases() -> None:
    """cpp 与 c++ 都认成 C++。"""
    assert ResponseNormalizer.detect_code_language("```cpp\nint x = 1;\n```") == "cpp"
    assert ResponseNormalizer.detect_code_language("```c++\nint x = 1;\n```") == "cpp"


def test_detect_code_language_recognizes_python() -> None:
    """python 标记认成 python。"""
    assert (
        ResponseNormalizer.detect_code_language("```python\nprint(1)\n```") == "python"
    )


def test_detect_code_language_without_known_tag() -> None:
    """没有围栏或者语言不认识时返回空串。"""
    assert ResponseNormalizer.detect_code_language("普通文字") == ""
    assert ResponseNormalizer.detect_code_language("```text\nx\n```") == ""


# ----------------------------------------------------------------------
# 既有方法没有变
# ----------------------------------------------------------------------
def test_existing_methods_keep_their_behavior() -> None:
    """兜底整理与内容单元拼接触景原样。"""
    assert ResponseNormalizer.normalize_fallback("  第一段  \n\n\n\n第二段   ") == (
        "第一段\n\n第二段"
    )
    assert ResponseNormalizer.normalize_fallback("```python\nprint(1)\n```") == "print(1)"
    assert ResponseNormalizer.format_item(" RAII ", " 一句表达。 ") == "RAII：一句表达。"
