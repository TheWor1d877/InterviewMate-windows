"""兜底内容与内容单元的文本整理。

模型输出普通文字、或者一条事件解析失败时，原始文本进兜底区。
这里只做显示前的整理：去首尾空白、合并多余空行、去掉成对的 markdown 代码围栏、
压缩行尾空白。内容一个字符都不删。

模型不按事件协议输出、直接给一段带代码围栏的文本时，
:meth:`ResponseNormalizer.extract_code_blocks` 把代码正文与说明文字拆开，
见《7-代码问题模式》纯代码块兜底一节。
"""

from __future__ import annotations

import re

__all__ = ["ResponseNormalizer"]


# 一行里的 markdown 代码围栏标记：三个以上的反引号或波浪号，允许前导缩进
_FENCE_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[^\n]*$")

# 连续三个以上的换行压成两个，也就是最多留一个空行
_MULTIPLE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")

# 行尾空白
_TRAILING_SPACE_PATTERN = re.compile(r"[ \t]+$", re.MULTILINE)

# 语言标记到高亮语言名的对应，键是小写标记
_LANGUAGE_ALIASES = {
    "c": "c",
    "cc": "cpp",
    "cpp": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "go": "go",
    "golang": "go",
    "java": "java",
    "js": "javascript",
    "javascript": "javascript",
    "py": "python",
    "python": "python",
    "rs": "rust",
    "rust": "rust",
    "ts": "typescript",
    "typescript": "typescript",
}


def _fence_language_tag(line: str) -> str:
    """取围栏行上的语言标记，没有标记时是空串。

    :param line: 围栏行。
    """
    match = _FENCE_PATTERN.match(line)
    if match is None:
        return ""
    info = line[match.end(1) :].strip()
    if not info:
        return ""
    # 语言标记只取第一个词，后面的可能是高亮选项
    return info.split()[0]


def _find_closing_fence(lines: list[str], start: int, marker: str) -> int:
    """找出结束围栏的行号，没有配对的围栏返回 -1。

    结束围栏要用同类标记，长度不短于起始围栏。

    :param lines: 全部行。
    :param start: 从第几行开始找。
    :param marker: 起始围栏的标记。
    """
    cursor = start
    total = len(lines)
    while cursor < total:
        match = _FENCE_PATTERN.match(lines[cursor])
        if match is not None:
            other = match.group(1)
            if other[0] == marker[0] and len(other) >= len(marker):
                return cursor
        cursor += 1
    return -1


class ResponseNormalizer:
    """整理兜底文本与内容单元文本。

    方法都是静态的，不持有状态。
    """

    @staticmethod
    def normalize_fallback(text: str) -> str:
        """整理一段兜底文本。

        依次做四件事：统一换行符、去掉成对的 markdown 代码围栏标记行、
        压缩行尾空白、合并连续多余空行，最后去掉首尾空白。

        围栏只去掉标记行，围栏里面的代码内容保留。
        """
        if not text:
            return ""
        unified = text.replace("\r\n", "\n").replace("\r", "\n")
        kept = ResponseNormalizer._strip_code_fences(unified.split("\n"))
        result = "\n".join(kept)
        result = _TRAILING_SPACE_PATTERN.sub("", result)
        result = _MULTIPLE_BLANK_LINES_PATTERN.sub("\n\n", result)
        return result.strip()

    @staticmethod
    def format_item(keyword: str, sentence: str) -> str:
        """把关键点与表达拼成一行文本，格式是「关键词：句子」。

        句子为空时只留关键词，关键词为空时只留句子。
        """
        key = keyword.strip()
        body = sentence.strip()
        if not key:
            return body
        if not body:
            return key
        return f"{key}：{body}"

    @staticmethod
    def extract_code_blocks(text: str) -> tuple[str, str]:
        """把一段文本拆成代码正文与说明文字。

        围栏里的内容是代码，围栏之外的文字是说明文字，按出现顺序拼接。
        代码正文一个字符都不改，原始缩进与换行原样保留。
        说明文字按 :meth:`normalize_fallback` 的规则整理。
        没有围栏时整段都当说明文字。

        :param text: 模型给的原始文本。
        :return: (代码正文, 说明文字)。
        """
        if not text:
            return "", ""
        unified = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = unified.split("\n")
        code_parts: list[str] = []
        note_lines: list[str] = []
        index = 0
        total = len(lines)
        while index < total:
            line = lines[index]
            match = _FENCE_PATTERN.match(line)
            if match is None:
                note_lines.append(line)
                index += 1
                continue
            closing = _find_closing_fence(lines, index + 1, match.group(1))
            if closing == -1:
                # 围栏没有闭合，剩下的内容都算代码
                code_parts.append("\n".join(lines[index + 1 :]))
                break
            code_parts.append("\n".join(lines[index + 1 : closing]))
            index = closing + 1
        if not code_parts:
            # 没有围栏，整段都当说明文字
            return "", ResponseNormalizer.normalize_fallback(text)
        code = "\n".join(code_parts)
        note = ResponseNormalizer.normalize_fallback("\n".join(note_lines))
        return code, note

    @staticmethod
    def detect_code_language(text: str) -> str:
        """认出文本里代码围栏的语言标记。

        给代码块选高亮语言用，认不出时返回空串。代码题默认 C++，
        见《7-代码问题模式》右侧完整代码一节。

        :param text: 可能带代码围栏的文本。
        """
        if not text:
            return ""
        unified = text.replace("\r\n", "\n").replace("\r", "\n")
        for line in unified.split("\n"):
            match = _FENCE_PATTERN.match(line)
            if match is None:
                continue
            language = _LANGUAGE_ALIASES.get(_fence_language_tag(line).lower())
            if language:
                return language
        return ""

    @staticmethod
    def _strip_code_fences(lines: list[str]) -> list[str]:
        """去掉成对出现的 markdown 代码围栏标记行。

        围栏里的内容原样保留。没有配对的单独一行围栏保留不动。
        """
        result: list[str] = []
        total = len(lines)
        index = 0
        while index < total:
            line = lines[index]
            match = _FENCE_PATTERN.match(line)
            if match is None:
                result.append(line)
                index += 1
                continue
            marker = match.group(1)
            closing = -1
            cursor = index + 1
            while cursor < total:
                candidate = _FENCE_PATTERN.match(lines[cursor])
                if candidate is not None:
                    other = candidate.group(1)
                    # 结束围栏要用同类标记，长度不短于起始围栏
                    if other[0] == marker[0] and len(other) >= len(marker):
                        closing = cursor
                        break
                cursor += 1
            if closing == -1:
                # 没有配对的围栏，这行保留
                result.append(line)
                index += 1
                continue
            result.extend(lines[index + 1 : closing])
            index = closing + 1
        return result
