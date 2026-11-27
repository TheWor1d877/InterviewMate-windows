"""日志封装与脱敏。

日志格式、级别与落盘位置见《2-骨架与配置》日志一节：

- 级别分 ``DEBUG``、``INFO``、``WARNING``、``ERROR``，默认 ``INFO``；
- 每条日志带 ``ts``、``level``、``event``、``request_id``、``session_id``，
  没有会话上下文的启动期日志后两个字段写 ``-``；
- 脱敏在日志封装层做：命中 ``Authorization`` 头的值、``sk-`` 开头的字符串、
  长度超过 1024 字符的 base64 片段时一律换成 ``***``；
  ``Authorization`` 头保留前缀与 ``Bearer`` / ``Basic`` 方案名，只换令牌部分；
- 字段值（``request_id``、``session_id`` 与额外字段）里的空白字符渲染时换成 ``-``，
  防止值里的空白把日志行伪造成额外的字段；``message`` 是自由文本，不做规整；
- 日志写到 ``logs/interviewmate-{日期}.log``，同时输出到终端。

脱敏在写入前完成，不影响调用方传入的内容：结构化数据先深拷贝再改，
文本脱敏返回新串。

终端输出走 stderr，不写 stdout，避免污染 stdout 上的结构化输出。
Textual 界面运行时终端归界面所有，调用方传 ``console=False`` 关掉终端输出，
日志只追加落盘，见 :func:`setup_logging`。同一个开关还会摘掉根记录器上写终端的
handler 并补一个 ``NullHandler``，第三方库导入时调 ``logging.basicConfig()``
挂不回终端（funasr 导入时就把 httpx 的 INFO 请求日志按自己的格式打到 stderr）。

界面运行期间第三方库用 ``print()`` 打的提示（torchaudio 的 ffmpeg 通知就是一例）
由 :func:`redirect_console` 接进日志文件，见 :func:`app.main.run_main`。
"""

from __future__ import annotations

import contextlib
import copy
import logging
import re
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "LOG_LEVELS",
    "DEFAULT_LOG_LEVEL",
    "REDACTED",
    "MAX_BASE64_LENGTH",
    "LOG_FILE_TEMPLATE",
    "LOG_DIR_NAME",
    "REPO_ROOT",
    "Redactor",
    "redact_text",
    "redact_data",
    "get_logger",
    "setup_logging",
    "log_event",
    "current_log_path",
    "redirect_console",
]


# 仓库根目录：app/utils/logger.py 往上三级，与当前工作目录无关。
# 相对 log_dir 按这个根目录解析，见《2-骨架与配置》日志一节的落盘位置要求。
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# 级别分 DEBUG、INFO、WARNING、ERROR，默认 INFO，见《2-骨架与配置》日志一节
LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
DEFAULT_LOG_LEVEL: str = "INFO"

# 脱敏后替换敏感内容用的占位串
REDACTED: str = "***"

# 超过这个长度的 base64 片段按敏感内容处理
MAX_BASE64_LENGTH: int = 1024

# 日志文件名模板，{date} 取本地日期
LOG_FILE_TEMPLATE: str = "interviewmate-{date}.log"

# 默认日志目录，相对路径按仓库根解析
LOG_DIR_NAME: str = "logs"

# 日志记录器名字，setup_logging 把 handler 装到这个记录器上
_LOGGER_NAME: str = "interviewmate"

# 级别名到 logging 数值级别的对应
_LEVEL_NUMBERS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# 规则 a：Authorization 头的值。保留前缀与 bearer/basic 方案名，
# 只把令牌部分换成 REDACTED，例如 ``Authorization: Bearer abc`` 变成
# ``Authorization: Bearer ***``。
_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(authorization\s*[:=]\s*)((?:bearer|basic)\s+)?([^\s,;}\]\"']+)"
)

# 规则 b：sk- 开头的密钥。
# 左边加负向后顾断言，sk 前一个字符是字母、数字或下划线时不算命中，
# 这样 task-、ask-me 这类普通词不会被误脱敏。
# 这个负向前瞻是**有意**的防误伤设计：sk- 前面不能是字母、数字或下划线。
#
# 同理，给字段值做字符规整（把空白字符换成占位符）时，替换字符一律用 '-' 不要用 '_'：
# '_' 就在上面这个负向前瞻的字符类里，规整后紧跟 '_' 的 sk- 密钥会绕开脱敏；
# '-' 不在那个字符类里，用 '-' 才不会让密钥漏网。
_SK_PATTERN = re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_\-]+")

# 规则 c：长度超过 MAX_BASE64_LENGTH 的 base64 片段
_BASE64_PATTERN = re.compile(rf"[A-Za-z0-9+/]{{{MAX_BASE64_LENGTH + 1},}}={{0,2}}")

# 字段值里的空白字符：日志行是空格分隔的 key=value，值里带空白就能伪造出额外字段。
# 这些空白一律换成 '-'（理由见上面 sk- 那条规则的注释）。
_FIELD_WHITESPACE_PATTERN = re.compile(r"\s")


def _redact_authorization(match: re.Match[str]) -> str:
    """替换 Authorization 头的令牌部分，保留前缀与 bearer/basic 方案名。

    ``Authorization: Bearer abc`` 变成 ``Authorization: Bearer ***``。
    """
    scheme = match.group(2) or ""
    return match.group(1) + scheme + REDACTED


class Redactor:
    """按《2-骨架与配置》日志一节的规则做脱敏。

    三条规则按顺序生效：``Authorization`` 头的值 → ``sk-`` 开头的密钥 →
    长度超过 :data:`MAX_BASE64_LENGTH` 的 base64 片段，命中后都换成 :data:`REDACTED`。

    ``enabled=False`` 时原样返回，对应配置里的 ``privacy.redact_secrets_in_logs`` 开关。
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        """脱敏是否开启。"""
        return self._enabled

    def redact_text(self, text: str) -> str:
        """对一段文本脱敏，返回脱敏后的新串，不修改入参。"""
        if not self._enabled:
            return text
        result = _AUTHORIZATION_PATTERN.sub(_redact_authorization, text)
        result = _SK_PATTERN.sub(REDACTED, result)
        result = _BASE64_PATTERN.sub(REDACTED, result)
        return result

    def redact_data(self, value: Any) -> Any:
        """对结构化数据脱敏，返回新对象，不修改入参。

        ``dict`` 与 ``list`` 先深拷贝再改，返回值和入参没有共享的容器；
        字典的键保持原样，只处理键对应的值。
        """
        if not self._enabled:
            return value
        if isinstance(value, dict):
            return self._redact_mapping(copy.deepcopy(value))
        if isinstance(value, list):
            return self._redact_sequence(copy.deepcopy(value))
        if isinstance(value, tuple):
            return tuple(self._redact_sequence(list(copy.deepcopy(value))))
        if isinstance(value, str):
            return self._redact_string(value)
        return value

    def _redact_mapping(self, mapping: dict[Any, Any]) -> dict[Any, Any]:
        """脱敏一个字典（已经是副本），就地改并返回。"""
        for key in list(mapping.keys()):
            value = mapping[key]
            if isinstance(key, str) and key.strip().lower() == "authorization":
                # 键名等于 authorization 时，值整体替换
                mapping[key] = REDACTED
            else:
                mapping[key] = self._redact_value(value)
        return mapping

    def _redact_sequence(self, items: list[Any]) -> list[Any]:
        """脱敏一个序列（已经是副本），返回新列表。"""
        return [self._redact_value(item) for item in items]

    def _redact_value(self, value: Any) -> Any:
        """脱敏一个值，容器递归处理，其余原样返回。"""
        if isinstance(value, dict):
            return self._redact_mapping(value)
        if isinstance(value, list):
            return self._redact_sequence(value)
        if isinstance(value, tuple):
            return tuple(self._redact_sequence(list(value)))
        if isinstance(value, str):
            return self._redact_string(value)
        return value

    def _redact_string(self, text: str) -> str:
        """脱敏一个字符串值：以 sk- 开头的整体替换，其余按文本规则处理。"""
        if text.startswith("sk-"):
            return REDACTED
        return self.redact_text(text)


# 模块级纯函数用的脱敏器，始终开启。
# 日志流水线是否脱敏由 setup_logging 的 redact 参数控制，两套状态互不影响。
_default_redactor = Redactor(enabled=True)

# 日志流水线用的脱敏器，setup_logging 按 redact 参数替换它
_pipeline_redactor = Redactor(enabled=True)


def redact_text(text: str) -> str:
    """按日志规则脱敏一段文本，返回脱敏后的新串。

    这是纯函数，始终按规则脱敏，不受 ``setup_logging(redact=False)`` 影响。
    """
    return _default_redactor.redact_text(text)


def redact_data(value: Any) -> Any:
    """按日志规则脱敏结构化数据，返回新对象，入参不被修改。

    这是纯函数，始终按规则脱敏，不受 ``setup_logging(redact=False)`` 影响。
    """
    return _default_redactor.redact_data(value)


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """返回标准库的 Logger，不做任何配置。

    :func:`setup_logging` 装 handler 的就是 ``name`` 默认指向的记录器。
    """
    return logging.getLogger(name)


def _normalize_level(level: str) -> str:
    """把级别名转成大写并校验，非法时抛 ValueError。"""
    legal = "、".join(sorted(LOG_LEVELS))
    if not isinstance(level, str):
        raise ValueError(f"日志级别必须是字符串，收到 {type(level).__name__}；合法级别：{legal}")
    name = level.upper()
    if name not in LOG_LEVELS:
        raise ValueError(f"未知日志级别 {level!r}；合法级别：{legal}")
    return name


def _text_or_dash(value: Any) -> str:
    """None 写成 ``-``，其余用 str() 转成字符串。"""
    if value is None:
        return "-"
    return str(value)


def _escape_newlines(text: str) -> str:
    """把换行符转义成字面 ``\\n``，保证一条日志记录只占一个物理行。

    只处理渲染出来的字符串，不改调用方传入的对象。
    """
    return text.replace("\r\n", "\\r\\n").replace("\r", "\\r").replace("\n", "\\n")


def _safe_field_text(value: Any) -> str:
    """渲染一个字段值：换行先转义，剩下的空白字符换成 ``-``。

    日志行是空格分隔的 ``key=value``，字段值里带空白就能渲染出假的额外字段，
    所以值里的空白一律换成 ``-``（不用 ``_`` 的理由见 ``sk-`` 那条规则的注释）。

    换行先转义是既有约定（一条记录只占一个物理行），转义后的文本里没有换行，
    剩下的空格、制表符、回车等空白再由这里换成 ``-``。
    """
    text = _escape_newlines(_text_or_dash(value))
    return _FIELD_WHITESPACE_PATTERN.sub("-", text)


def _render_line(
    *,
    ts: str,
    level: str,
    event: Any,
    message: str,
    request_id: Any,
    session_id: Any,
    fields: dict[str, Any],
) -> str:
    """按《2-骨架与配置》日志一节的格式拼一行日志。

    形如 ``{ts} {level} event={event} request_id={值或-} session_id={值或-}
    {message} key=value``，额外字段按传入顺序追加在末尾。

    ``request_id``、``session_id`` 与额外字段的值做字符规整（空白换成 ``-``），
    防止值里的空白伪造出额外字段；``message`` 是给人看的自由文本，不做规整。
    """
    parts = [
        f"{ts} {level}",
        f"event={_text_or_dash(event)}",
        f"request_id={_safe_field_text(request_id)}",
        f"session_id={_safe_field_text(session_id)}",
    ]
    if message:
        parts.append(str(message))
    for key, value in fields.items():
        parts.append(f"{key}={_safe_field_text(value)}")
    return " ".join(parts)


class _EventFormatter(logging.Formatter):
    """把日志记录渲染成《2-骨架与配置》日志一节规定的单行格式。

    渲染完成后再过一遍脱敏，保证写进文件和终端的内容都是脱敏后的。
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).astimezone().isoformat(
            timespec="microseconds"
        )
        line = _render_line(
            ts=ts,
            level=record.levelname.upper(),
            event=getattr(record, "event", "-"),
            message=record.getMessage(),
            request_id=getattr(record, "request_id", None),
            session_id=getattr(record, "session_id", None),
            fields=getattr(record, "fields", None) or {},
        )
        # 先转义换行，再脱敏，保证一条记录只占一个物理行
        line = _escape_newlines(line)
        line = _pipeline_redactor.redact_text(line)
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


# setup_logging 装过的 handler，重复调用时靠它保证不重复挂载
_state: dict[str, Any] = {
    "file_handler": None,
    "stream_handler": None,
    "mirror_handler": None,
    "path": None,
    "mirror_path": None,
}


def _remove_handler(logger: logging.Logger, handler: logging.Handler | None) -> None:
    """摘掉并关闭一个 handler。"""
    if handler is None:
        return
    logger.removeHandler(handler)
    handler.close()


def _silence_root_logger() -> None:
    """摘掉根记录器上写终端的 handler，补一个 NullHandler 占位。

    第三方库导入时会调 ``logging.basicConfig()``（funasr 就是一例），
    此时根记录器没有 handler 就按自己的格式把 INFO 级的 httpx 请求日志
    打到 stderr。根记录器上已有 handler 时 ``basicConfig`` 什么都不做，
    所以这里先摘掉写终端的、再补一个 :class:`logging.NullHandler`，
    之后第三方再调 ``basicConfig`` 也挂不上终端 handler。

    只摘写 ``stdout`` / ``stderr`` 的 handler，写文件的 handler 与
    测试框架挂的捕获 handler 不动。
    """
    root = logging.getLogger()
    terminal_streams = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    for handler in list(root.handlers):
        stream = getattr(handler, "stream", None)
        if isinstance(handler, logging.StreamHandler) and stream in terminal_streams:
            root.removeHandler(handler)
            handler.close()
    if not root.handlers:
        root.addHandler(logging.NullHandler())


def current_log_path() -> Path | None:
    """返回 :func:`setup_logging` 当前写的日志文件路径，没配置过返回 None。"""
    path = _state["path"]
    return Path(path) if path is not None else None


@contextlib.contextmanager
def redirect_console(path: str | Path) -> Iterator[TextIO]:
    """把 ``sys.stdout`` 与 ``sys.stderr`` 临时接到日志文件，退出时恢复。

    第三方库在导入或加载模型时用 ``print()`` 打的提示（torchaudio 的
    ffmpeg 通知就是一例）会追加进日志文件，不再打断界面。Textual 的 Linux
    驱动直接写 ``sys.__stderr__``、读 ``sys.__stdin__``，换掉这两个对象
    不影响界面本身。退出时恢复原来的对象并关闭文件。
    """
    previous_out = sys.stdout
    previous_err = sys.stderr
    stream = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream
    try:
        yield stream
    finally:
        sys.stdout = previous_out
        sys.stderr = previous_err
        stream.close()


def _resolve_log_dir(log_dir: str | Path) -> Path:
    """相对路径按仓库根解析，绝对路径原样使用。"""
    path = Path(log_dir)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _sync_handlers(
    logger: logging.Logger,
    log_path: Path,
    stream: TextIO | None,
    console: bool = True,
    mirror_path: Path | None = None,
) -> None:
    """让记录器上的 handler 与目标文件、目标终端保持一致。

    重复调用不会重复挂 handler：文件目标没变就沿用旧 handler，
    变了就把旧的摘掉换一个；终端流变了同样替换，handler 总数不变。
    ``console=False`` 时摘掉终端 handler 且不再挂新的，日志只落盘。
    ``mirror_path`` 给出时再挂一个文件 handler 写同一份内容。
    """
    file_handler = _state["file_handler"]
    stream_handler = _state["stream_handler"]
    mirror_handler = _state["mirror_handler"]

    if file_handler is not None and _state["path"] != log_path:
        _remove_handler(logger, file_handler)
        file_handler = None
    if file_handler is None:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(_EventFormatter())
        logger.addHandler(file_handler)
        _state["file_handler"] = file_handler
        _state["path"] = log_path

    if mirror_handler is not None and _state["mirror_path"] != mirror_path:
        _remove_handler(logger, mirror_handler)
        mirror_handler = None
    if mirror_path is not None and mirror_handler is None:
        mirror_handler = logging.FileHandler(mirror_path, encoding="utf-8")
        mirror_handler.setFormatter(_EventFormatter())
        logger.addHandler(mirror_handler)
        _state["mirror_handler"] = mirror_handler
        _state["mirror_path"] = mirror_path

    if not console:
        # 界面运行时终端归 Textual 所有，stderr 上的日志会把界面画花
        _remove_handler(logger, stream_handler)
        _state["stream_handler"] = None
        return

    target_stream = stream if stream is not None else sys.stderr
    if stream_handler is not None and getattr(stream_handler, "stream", None) is not target_stream:
        _remove_handler(logger, stream_handler)
        stream_handler = None
    if stream_handler is None:
        stream_handler = logging.StreamHandler(target_stream)
        stream_handler.setFormatter(_EventFormatter())
        logger.addHandler(stream_handler)
        _state["stream_handler"] = stream_handler


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    *,
    log_dir: str | Path = LOG_DIR_NAME,
    filename_template: str = LOG_FILE_TEMPLATE,
    mirror_filename_template: str | None = None,
    stream: TextIO | None = None,
    console: bool = True,
    redact: bool = True,
    force: bool = False,
) -> Path:
    """配置日志输出并返回日志文件路径。

    日志同时写到 ``{log_dir}/interviewmate-{YYYY-MM-DD}.log`` 和终端（stderr）。
    ``log_dir`` 是相对路径时按仓库根解析，与当前工作目录无关。
    ``redact=False`` 时关闭脱敏，对应配置里的 ``privacy.redact_secrets_in_logs``。
    ``console=False`` 时不再写终端，只把日志追加落盘：Textual 界面运行时
    终端归界面所有，stderr 上的日志会把整屏画乱。同一个开关还会摘掉根记录器上
    写终端的 handler，第三方库后来再调 ``logging.basicConfig()`` 也挂不回来。
    ``mirror_filename_template`` 给出时同一份日志再写一个文件，
    TUI 用它把稳定文件与按日文件都留一份。

    重复调用不会重复挂 handler；``force=True`` 时先摘掉之前挂的 handler 再挂新的。

    :raises ValueError: ``level`` 不在 :data:`LOG_LEVELS` 里。
    """
    level_name = _normalize_level(level)

    global _pipeline_redactor
    _pipeline_redactor = Redactor(enabled=redact)

    target_dir = _resolve_log_dir(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    date_text = datetime.now().strftime("%Y-%m-%d")
    log_path = target_dir / filename_template.format(date=date_text)
    mirror_path = (
        target_dir / mirror_filename_template.format(date=date_text)
        if mirror_filename_template
        else None
    )

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_LEVEL_NUMBERS[level_name])
    # 记录器自己带 handler，不再往上冒泡，避免重复输出
    logger.propagate = False

    if force:
        _remove_handler(logger, _state["file_handler"])
        _remove_handler(logger, _state["stream_handler"])
        _remove_handler(logger, _state["mirror_handler"])
        _state["file_handler"] = None
        _state["stream_handler"] = None
        _state["mirror_handler"] = None
        _state["path"] = None
        _state["mirror_path"] = None

    if not console:
        _silence_root_logger()
    _sync_handlers(logger, log_path, stream, console, mirror_path)
    return log_path


def _redact_field_value(value: Any) -> Any:
    """脱敏一个额外字段的值。"""
    if value is None:
        return None
    if isinstance(value, str):
        return _pipeline_redactor.redact_text(value)
    return _pipeline_redactor.redact_data(value)


def _redact_slot(value: str | None) -> str | None:
    """脱敏 request_id、session_id 这类专用槽位。

    它们直接进 LogRecord 的 extra，内存里会留着原文，
    所以塞进去之前先过一遍脱敏（与消息、额外字段走同一个流水线脱敏器，
    ``setup_logging(redact=False)`` 时同样不脱敏）。
    """
    if value is None:
        return None
    return _pipeline_redactor.redact_text(str(value))


def log_event(
    event: str,
    message: str = "",
    *,
    level: str = "INFO",
    logger: logging.Logger | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    **fields: Any,
) -> None:
    """写一条带固定字段的日志。

    输出格式见《2-骨架与配置》日志一节：
    ``{ts} {level} event={event} request_id={值或-} session_id={值或-}
    {message} key=value``。``request_id`` 与 ``session_id`` 是 None 时写 ``-``，
    额外字段的 value 用 ``str()``，None 写 ``-``。

    消息与字段值在交给 logging 之前先脱敏，日志写盘时再脱敏一次，
    保证调用方传入的内容不被修改。``request_id`` 与 ``session_id`` 同样先脱敏，
    内存里的 LogRecord 不留原文。

    :raises ValueError: ``level`` 不在 :data:`LOG_LEVELS` 里。
    """
    level_name = _normalize_level(level)
    target = logger if logger is not None else get_logger()
    text = _pipeline_redactor.redact_text(str(message))
    safe_fields = {key: _redact_field_value(value) for key, value in fields.items()}
    target.log(
        _LEVEL_NUMBERS[level_name],
        text,
        extra={
            "event": event if event else "-",
            "request_id": _redact_slot(request_id),
            "session_id": _redact_slot(session_id),
            "fields": safe_fields,
        },
    )
