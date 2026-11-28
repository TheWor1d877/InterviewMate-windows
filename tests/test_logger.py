"""日志封装与脱敏的测试。

覆盖《2-骨架与配置》日志一节的级别、固定字段、脱敏规则与落盘路径，
以及《8-集成与验收》验收约束里「日志文件里搜不到 MODEL_API_KEY 的值和
Authorization 头的值」这一条。
"""

from __future__ import annotations

import copy
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from app.utils import logger as logger_module
from app.utils.logger import (
    DEFAULT_LOG_LEVEL,
    LOG_DIR_NAME,
    LOG_FILE_TEMPLATE,
    LOG_LEVELS,
    MAX_BASE64_LENGTH,
    REDACTED,
    Redactor,
    current_log_path,
    get_logger,
    log_event,
    redact_data,
    redact_text,
    redirect_console,
    setup_logging,
)

# 一行日志的固定结构：ts、level、event、request_id、session_id，后面是消息和额外字段
_LINE_PATTERN = re.compile(
    r"^(?P<ts>\S+) (?P<level>[A-Z]+) "
    r"event=(?P<event>\S*) "
    r"request_id=(?P<request_id>\S*) "
    r"session_id=(?P<session_id>\S*)"
    r"(?: (?P<rest>.*))?$"
)

# ts 形如 2026-11-25T13:30:00.123456+08:00
_TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}[+-]\d{2}:\d{2}$")


@pytest.fixture(autouse=True)
def _restore_pipeline_redactor() -> Any:
    """用例结束后把日志流水线的脱敏开关恢复成默认开启。

    setup_logging(redact=False) 会改到模块级的流水线脱敏器，
    不恢复的话后面的用例会跟着不脱敏。
    """
    yield
    logger_module._pipeline_redactor = Redactor(enabled=True)


def _read_line(path: Path) -> str:
    """读日志文件的最后一行并去掉换行。"""
    content = path.read_text(encoding="utf-8")
    assert content.endswith("\n"), "日志行应以换行结尾"
    return content.rstrip("\n").splitlines()[-1]


def _parse_line(line: str) -> dict[str, str]:
    """把一行日志拆成各字段。"""
    match = _LINE_PATTERN.match(line)
    assert match is not None, f"日志行不符合约定格式：{line!r}"
    return match.groupdict()


def _own_handlers() -> list[logging.Handler]:
    """取当前记录器上由 setup_logging 装上的 handler。

    pytest 自己也会往记录器上挂 LogCaptureHandler，所以这里按 formatter 过滤，
    只数本模块装的 handler。
    """
    logger = logging.getLogger("interviewmate")
    return [
        h
        for h in logger.handlers
        if isinstance(h.formatter, logger_module._EventFormatter)
    ]


class _ListHandler(logging.Handler):
    """把日志记录攒起来的 handler，用来看清 log_event 传了什么。"""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# --------------------------------------------------------------------------
# 常量与日志行格式
# --------------------------------------------------------------------------


def test_constants_match_design() -> None:
    """常量取值与《2-骨架与配置》日志一节一致。"""
    assert LOG_LEVELS == frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
    assert DEFAULT_LOG_LEVEL == "INFO"
    assert REDACTED == "***"
    assert MAX_BASE64_LENGTH == 1024
    assert LOG_FILE_TEMPLATE == "interviewmate-{date}.log"
    assert LOG_DIR_NAME == "logs"


def test_log_line_contains_required_fields(tmp_path: Path) -> None:
    """一行日志带 ts、level、event、request_id、session_id。"""
    path = setup_logging("DEBUG", log_dir=tmp_path)
    log_event("startup", "进程启动", level="DEBUG")

    parsed = _parse_line(_read_line(path))
    assert _TS_PATTERN.match(parsed["ts"]) is not None, parsed["ts"]
    assert parsed["level"] == "DEBUG"
    assert parsed["event"] == "startup"
    assert parsed["request_id"] == "-"
    assert parsed["session_id"] == "-"
    assert parsed["rest"] == "进程启动"


def test_startup_log_without_session_context_writes_dash(tmp_path: Path) -> None:
    """没有会话上下文的启动期日志，后两个字段写 -。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("config_loaded")

    parsed = _parse_line(_read_line(path))
    assert parsed["request_id"] == "-"
    assert parsed["session_id"] == "-"
    assert parsed["rest"] is None


def test_log_event_carries_request_and_session_ids(tmp_path: Path) -> None:
    """带 request_id、session_id 的日志行里出现这两个值。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("llm_request", "发起请求", request_id="req-42", session_id="sess-7")

    parsed = _parse_line(_read_line(path))
    assert parsed["request_id"] == "req-42"
    assert parsed["session_id"] == "sess-7"
    assert parsed["event"] == "llm_request"


def test_log_record_with_newline_stays_one_line(tmp_path: Path) -> None:
    """message 与额外字段里的换行被转义，一条记录只占一个物理行。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("multiline", "line1\nline2", detail="a\nb")

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    assert len(lines) == 1
    assert "line1\\nline2" in lines[0]
    assert "detail=a\\nb" in lines[0]
    assert "\n" not in lines[0]


def test_extra_fields_are_appended_as_key_value(tmp_path: Path) -> None:
    """额外字段以 key=value 追加在消息后面。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("asr_result", "识别完成", request_id="req-1", model="mock", chars=128)

    parsed = _parse_line(_read_line(path))
    assert parsed["rest"] == "识别完成 model=mock chars=128"


def test_extra_field_none_value_written_as_dash(tmp_path: Path) -> None:
    """额外字段的值是 None 时写 -。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("asr_result", "识别完成", source=None)

    parsed = _parse_line(_read_line(path))
    assert parsed["rest"] == "识别完成 source=-"


def test_field_value_whitespace_becomes_dash(tmp_path: Path) -> None:
    """字段值里的空白换成 -，伪造不出额外字段；message 的空白保留。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("ev", "消息里 有 空格", note="值里 有 空格")

    line = _read_line(path)
    # 字段值里的空格被换成 -，值里没有空白
    assert "note=值里-有-空格" in line
    assert "note=值里 有 空格" not in line
    assert "note=值里-有-空格" in line.split()
    # message 是给人看的自由文本，空格保留
    assert "消息里 有 空格" in line


def test_slot_values_whitespace_becomes_dash(tmp_path: Path) -> None:
    """request_id、session_id 里的空白也换成 -。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("ev", "处理请求", request_id="req 1", session_id="sess\t1")

    parsed = _parse_line(_read_line(path))
    assert parsed["request_id"] == "req-1"
    assert parsed["session_id"] == "sess-1"


def test_field_value_normalization_uses_dash_not_underscore(tmp_path: Path) -> None:
    """字段值规整的替换字符是 - 不是 _，紧跟分隔符的 sk- 密钥仍被脱敏。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("ev", "处理请求", note="x sk-TOPSECRETKEY")

    line = _read_line(path)
    assert "TOPSECRETKEY" not in line
    assert "note=x-***" in line
    assert "note=x_" not in line


def test_log_event_level_is_uppercase(tmp_path: Path) -> None:
    """级别输出为大写。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("slow_request", "请求偏慢", level="warning")

    parsed = _parse_line(_read_line(path))
    assert parsed["level"] == "WARNING"


def test_log_event_uses_given_logger() -> None:
    """传了 logger 就写进那个记录器，字段挂在日志记录上。"""
    custom = logging.getLogger("test.custom.event")
    custom.setLevel(logging.DEBUG)
    handler = _ListHandler()
    custom.addHandler(handler)
    try:
        log_event(
            "custom_event",
            "自定义记录器",
            logger=custom,
            request_id="req-9",
            session_id="sess-3",
            topic="算法",
        )
    finally:
        custom.removeHandler(handler)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.getMessage() == "自定义记录器"
    assert record.event == "custom_event"  # type: ignore[attr-defined]
    assert record.request_id == "req-9"  # type: ignore[attr-defined]
    assert record.session_id == "sess-3"  # type: ignore[attr-defined]
    assert record.fields == {"topic": "算法"}  # type: ignore[attr-defined]


def test_log_event_redacts_slots_in_memory(tmp_path: Path) -> None:
    """request_id、session_id 进 LogRecord 之前先脱敏，内存里不留原文。"""
    setup_logging("INFO", log_dir=tmp_path)
    custom = logging.getLogger("test.redact.slots")
    custom.setLevel(logging.DEBUG)
    handler = _ListHandler()
    custom.addHandler(handler)
    try:
        log_event(
            "llm_request",
            "发起请求",
            logger=custom,
            request_id="sk-FAKESECRET123",
            session_id="sk-FAKESESSION456",
        )
    finally:
        custom.removeHandler(handler)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert "FAKESECRET123" not in str(record.request_id)
    assert "FAKESESSION456" not in str(record.session_id)
    assert record.request_id == REDACTED  # type: ignore[attr-defined]
    assert record.session_id == REDACTED  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# 脱敏规则
# --------------------------------------------------------------------------


def test_redact_authorization_header_value() -> None:
    """Authorization 头的值被换成 ***。"""
    text = "Authorization: Bearer super-secret-token"
    result = redact_text(text)
    assert "super-secret-token" not in result
    assert REDACTED in result
    assert result.startswith("Authorization:")


def test_redact_authorization_value_without_scheme() -> None:
    """不带方案的 Authorization 值同样被换掉。"""
    result = redact_text("authorization=raw-token-value")
    assert "raw-token-value" not in result
    assert result == "authorization=" + REDACTED


def test_redact_sk_prefixed_key() -> None:
    """sk- 开头的密钥被换成 ***。"""
    result = redact_text("api_key=sk-abcdef0123456789")
    assert "sk-abcdef0123456789" not in result
    assert REDACTED in result


def test_redact_long_base64_fragment() -> None:
    """超过 1024 字符的 base64 片段被换成 ***。"""
    payload = "QUJD" * 300  # 1200 字符，超过阈值
    result = redact_text(f"body={payload}")
    assert payload not in result
    assert result == f"body={REDACTED}"


def test_three_rules_together_leave_no_secret() -> None:
    """三条规则都命中后，消息里不含任何原始密钥片段。"""
    auth_secret = "bearer-token-abc"
    sk_secret = "sk-zzz999888777"
    b64_secret = "QUJD" * 400
    message = f"Authorization: Bearer {auth_secret} key={sk_secret} blob={b64_secret}"

    result = redact_text(message)

    assert auth_secret not in result
    assert sk_secret not in result
    assert b64_secret not in result
    assert result.count(REDACTED) == 3


def test_redact_does_not_touch_normal_words() -> None:
    """task-、ask-me 这类普通词不被误脱敏。"""
    text = "task-list ask-me task- ask-me"
    assert redact_text(text) == text


def test_redact_authorization_keeps_scheme_name() -> None:
    """Authorization 头保留前缀与方案名，只把令牌部分换成 ***。"""
    result = redact_text("Authorization: Bearer abc123")
    assert "Bearer ***" in result
    assert "abc123" not in result
    assert result == "Authorization: Bearer " + REDACTED
    # 方案名是 Basic 时同样保留
    basic = redact_text("Authorization: Basic dXNlcjpwYXNz")
    assert "Basic ***" in basic
    assert "dXNlcjpwYXNz" not in basic
    # 另外两条规则没被破坏：sk- 前缀与超长 base64
    assert redact_text("sk-abc") == REDACTED
    assert redact_text("x" + "A" * 1100) == REDACTED


def test_base64_at_threshold_is_not_redacted() -> None:
    """长度不超过 1024 的 base64 片段不被改。"""
    payload = "QUJD" * 256  # 正好 1024 字符
    assert len(payload) == MAX_BASE64_LENGTH
    assert redact_text(f"body={payload}") == f"body={payload}"


def test_redact_data_does_not_modify_caller_dict() -> None:
    """脱敏结构化数据不修改调用方传入的对象。"""
    payload = {
        "authorization": "Bearer abc123",
        "api_key": "sk-plain-key",
        "nested": {"note": "ok"},
    }
    before = copy.deepcopy(payload)

    result = redact_data(payload)

    assert result == {
        "authorization": REDACTED,
        "api_key": REDACTED,
        "nested": {"note": "ok"},
    }
    assert payload == before
    assert payload["authorization"] == "Bearer abc123"


def test_redact_data_returns_new_containers() -> None:
    """脱敏结构化数据返回新容器，不与入参共享。"""
    payload = {"headers": {"authorization": "Bearer abc"}, "items": ["sk-secret", "plain"]}

    result = redact_data(payload)

    assert result is not payload
    assert result["headers"] is not payload["headers"]
    assert result["items"] is not payload["items"]
    assert result["headers"]["authorization"] == REDACTED
    assert result["items"] == [REDACTED, "plain"]


def test_redact_data_redacts_nested_sk_value() -> None:
    """嵌套结构里以 sk- 开头的字符串值被换掉。"""
    result = redact_data({"a": {"b": ["sk-secret", "keep"]}})
    assert result == {"a": {"b": [REDACTED, "keep"]}}


def test_redactor_disabled_keeps_text_unchanged() -> None:
    """Redactor(enabled=False) 时文本不脱敏。"""
    redactor = Redactor(enabled=False)
    text = "Authorization: Bearer abc sk-secret"
    assert redactor.enabled is False
    assert redactor.redact_text(text) == text


def test_redactor_disabled_keeps_data_unchanged() -> None:
    """Redactor(enabled=False) 时结构化数据不脱敏。"""
    redactor = Redactor(enabled=False)
    payload = {"authorization": "Bearer abc"}
    assert redactor.redact_data(payload) == payload


# --------------------------------------------------------------------------
# setup_logging
# --------------------------------------------------------------------------


def test_setup_logging_writes_log_file(tmp_path: Path) -> None:
    """setup_logging 把日志写进文件。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    log_event("boot", "启动完成")

    assert path.exists()
    assert "启动完成" in path.read_text(encoding="utf-8")


def test_setup_logging_returns_expected_file_name(tmp_path: Path) -> None:
    """日志文件名是 interviewmate-{日期}.log。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    today = datetime.now().strftime("%Y-%m-%d")

    assert path.name == f"interviewmate-{today}.log"
    assert path.parent == tmp_path


def test_setup_logging_writes_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """日志同时输出到终端 stderr，不污染 stdout。"""
    setup_logging("INFO", log_dir=tmp_path)
    log_event("boot", "终端输出检查")

    captured = capsys.readouterr()
    assert "终端输出检查" in captured.err
    assert captured.out == ""


def test_setup_logging_console_false_writes_only_to_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """TUI 模式只追加落盘，stderr 上没有输出。"""
    path = setup_logging(
        "INFO",
        log_dir=tmp_path,
        filename_template="interviewmate.log",
        console=False,
        force=True,
    )
    capsys.readouterr()  # 清掉重新装管道之前的输出

    log_event("tui.boot", "界面运行只落盘")
    log_event("tui.hotkey", "第二次写入")

    captured = capsys.readouterr()
    text = path.read_text(encoding="utf-8")
    assert path == tmp_path / "interviewmate.log"
    assert "界面运行只落盘" in text
    assert "第二次写入" in text
    assert captured.err == ""
    assert captured.out == ""


def test_setup_logging_console_false_removes_stream_handler(tmp_path: Path) -> None:
    """console=False 摘掉终端 handler，只留文件 handler。"""
    setup_logging("INFO", log_dir=tmp_path, force=True)

    setup_logging("INFO", log_dir=tmp_path, console=False)

    handlers = _own_handlers()
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.FileHandler)


def test_setup_logging_mirror_keeps_stable_and_dated_files(
    tmp_path: Path,
) -> None:
    """给了镜像文件名时稳定文件与按日文件都写，内容一致。"""
    path = setup_logging(
        "INFO",
        log_dir=tmp_path,
        filename_template="interviewmate.log",
        mirror_filename_template=LOG_FILE_TEMPLATE,
        console=False,
        force=True,
    )
    log_event("tui.boot", "稳定文件与按日文件都写")

    dated = tmp_path / LOG_FILE_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d")
    )
    assert path == tmp_path / "interviewmate.log"
    assert "稳定文件与按日文件都写" in path.read_text(encoding="utf-8")
    assert "稳定文件与按日文件都写" in dated.read_text(encoding="utf-8")
    handlers = _own_handlers()
    assert len(handlers) == 2
    assert all(isinstance(handler, logging.FileHandler) for handler in handlers)


def test_setup_logging_console_true_restores_stderr_after_tui(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """一次性命令回到默认行为：关过终端输出之后重新打开。"""
    setup_logging(
        "INFO",
        log_dir=tmp_path,
        filename_template="interviewmate.log",
        mirror_filename_template=LOG_FILE_TEMPLATE,
        console=False,
        force=True,
    )
    setup_logging("INFO", log_dir=tmp_path)

    log_event("cli.status", "终端输出恢复")

    captured = capsys.readouterr()
    assert "终端输出恢复" in captured.err
    # 镜像 handler 也跟着摘掉，回到文件加终端两个 handler
    assert len(_own_handlers()) == 2


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    """重复调用不重复挂 handler。"""
    setup_logging("INFO", log_dir=tmp_path)
    before = _own_handlers()

    setup_logging("INFO", log_dir=tmp_path)
    after = _own_handlers()

    assert len(before) == 2
    assert len(after) == 2
    assert after == before


def test_setup_logging_force_replaces_handlers(tmp_path: Path) -> None:
    """force=True 时先清掉之前挂的 handler 再挂新的。"""
    setup_logging("INFO", log_dir=tmp_path / "a")
    old_handler = next(
        h for h in _own_handlers() if isinstance(h, logging.FileHandler)
    )

    path = setup_logging("INFO", log_dir=tmp_path / "b", force=True)
    log_event("boot", "换目录之后")

    handlers = _own_handlers()
    assert old_handler not in handlers
    assert old_handler.stream is None  # 旧 handler 已关闭
    assert len(handlers) == 2
    assert path.parent == tmp_path / "b"
    assert "换目录之后" in path.read_text(encoding="utf-8")


def test_setup_logging_invalid_level_raises(tmp_path: Path) -> None:
    """非法级别抛 ValueError，消息里列出合法值。"""
    with pytest.raises(ValueError) as excinfo:
        setup_logging("TRACE", log_dir=tmp_path)

    message = str(excinfo.value)
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        assert level in message


def test_setup_logging_relative_log_dir_resolves_to_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """相对 log_dir 按仓库根解析，与当前工作目录无关。"""
    monkeypatch.setattr(logger_module, "REPO_ROOT", tmp_path)

    path = setup_logging("INFO", log_dir="logs")

    assert path.parent == tmp_path / "logs"


def test_setup_logging_redact_false_keeps_secret(tmp_path: Path) -> None:
    """redact=False 时日志里保留原始内容，对应隐私开关关掉的情形。"""
    path = setup_logging("INFO", log_dir=tmp_path, redact=False)
    log_event("llm_request", "api_key=sk-keep-this-value")

    assert "sk-keep-this-value" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# log_event
# --------------------------------------------------------------------------


def test_log_event_invalid_level_raises() -> None:
    """非法级别抛 ValueError。"""
    with pytest.raises(ValueError) as excinfo:
        log_event("boot", "启动", level="VERBOSE")

    message = str(excinfo.value)
    assert "VERBOSE" in message
    assert "INFO" in message


def test_log_file_contains_no_secret(tmp_path: Path) -> None:
    """日志文件里搜不到 API Key 的值和 Authorization 头的值。"""
    api_key = "sk-model-key-000111"
    auth_value = "bearer-token-value-222"

    path = setup_logging("INFO", log_dir=tmp_path)
    log_event(
        "llm_request",
        f"Authorization: Bearer {auth_value}",
        request_id="req-1",
        api_key=api_key,
    )
    content = path.read_text(encoding="utf-8")

    assert api_key not in content
    assert auth_value not in content
    assert REDACTED in content


def test_get_logger_returns_logger_with_name() -> None:
    """get_logger 返回同名记录器。"""
    logger = get_logger("interviewmate.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "interviewmate.test"
    assert get_logger().name == "interviewmate"


# --------------------------------------------------------------------------
# 终端静默与输出重定向
# --------------------------------------------------------------------------


@pytest.fixture
def _restore_root_logger() -> Any:
    """用例结束后把根记录器的 handler 与级别恢复原样。"""
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    for handler in before:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


def test_console_false_silences_root_terminal_handler(
    tmp_path: Path, _restore_root_logger: Any
) -> None:
    """console=False 摘掉根记录器写终端的 handler，第三方 basicConfig 挂不回来。"""
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)

    setup_logging("INFO", log_dir=tmp_path, console=False)

    assert handler not in root.handlers
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)
    # 第三方库再调 basicConfig 也不会新增 handler
    snapshot = list(root.handlers)
    logging.basicConfig(level=logging.INFO)
    assert root.handlers == snapshot


def test_redirect_console_captures_prints_into_log_file(tmp_path: Path) -> None:
    """redirect_console 把 stdout/stderr 的写接进日志文件，退出后恢复。"""
    target = tmp_path / "console.log"
    original_out = sys.stdout
    original_err = sys.stderr

    with redirect_console(target):
        print("标准输出")
        print("标准错误", file=sys.stderr)
        assert sys.stdout is not original_out
        assert sys.stderr is not original_err

    assert sys.stdout is original_out
    assert sys.stderr is original_err
    content = target.read_text(encoding="utf-8")
    assert "标准输出" in content
    assert "标准错误" in content


def test_redirect_console_restores_streams_on_error(tmp_path: Path) -> None:
    """块里抛异常时同样恢复原来的流。"""
    original_out = sys.stdout

    with pytest.raises(RuntimeError):
        with redirect_console(tmp_path / "console.log"):
            raise RuntimeError("boom")

    assert sys.stdout is original_out


def test_current_log_path_follows_setup_logging(tmp_path: Path) -> None:
    """current_log_path 返回最近一次 setup_logging 的落盘路径。"""
    path = setup_logging("INFO", log_dir=tmp_path)
    assert current_log_path() == path
