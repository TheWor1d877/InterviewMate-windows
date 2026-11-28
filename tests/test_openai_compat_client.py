"""模型客户端与 mock 客户端的测试。

用 httpx.MockTransport 注入传输层，不发真实网络请求。
覆盖《3-文本输入与流式回答》模型客户端一节的流式读取、重试与取消语义，
以及《接口契约》模型文本请求体、多模态请求体、模型能力探测三节的字段。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

import app.config
from app.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from app.core.exceptions import (
    ModelAuthError,
    ModelStreamError,
    ModelTimeoutError,
)
from app.llm.openai_compat_client import OpenAICompatClient, ModelCapability
from app.llm.mock_client import MockClient
from app.utils import logger as logger_module
from app.utils.logger import Redactor

# 事件类型白名单，见《3-文本输入与流式回答》流式事件协议一节
_EVENT_TYPES = frozenset({"framework", "item", "section_done", "done"})


@pytest.fixture
def app_config(monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    """加载仓库默认配置，屏蔽可能存在的环境变量覆盖。"""
    for name in ("MODEL_BASE_URL", "MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app.config, "DEFAULT_ENV_FILE", Path("/nonexistent/.env"))
    return load_config(DEFAULT_CONFIG_PATH)


class _ListHandler(logging.Handler):
    """把日志记录收进列表，方便断言。"""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def log_lines(monkeypatch: pytest.MonkeyPatch) -> _ListHandler:
    """收集 interviewmate 记录器的日志文本。

    这里特意关掉脱敏，断言的是客户端自己没有把密钥写进日志，
    而不是靠脱敏规则兜住。
    """
    monkeypatch.setattr(logger_module, "_pipeline_redactor", Redactor(enabled=False))
    handler = _ListHandler()
    target = logging.getLogger("interviewmate")
    old_level = target.level
    target.setLevel(logging.DEBUG)
    target.addHandler(handler)
    yield handler
    target.removeHandler(handler)
    target.setLevel(old_level)


class _InterruptStream(httpx.AsyncByteStream):
    """先给一段增量再抛读错误，模拟流中断。"""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._payload
        raise httpx.ReadError("流中断")


class _LiveStream(httpx.AsyncByteStream):
    """给一段增量后一直挂着，模拟还在流式输出。

    关流之后循环自己退出，读方不会一直等。
    """

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._payload
        while not self._closed:
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        self._closed = True


def _sse(*chunks: str) -> bytes:
    """把增量文本拼成 SSE 响应体。"""
    parts = [
        "data: "
        + json.dumps({"choices": [{"delta": {"content": chunk}}]}, ensure_ascii=False)
        for chunk in chunks
    ]
    return ("\n\n".join(parts) + "\n\n").encode("utf-8")


async def _collect(stream: AsyncIterator[str]) -> list[str]:
    """把流里的片段收成列表。"""
    return [chunk async for chunk in stream]


def _make_client(
    config: AppConfig,
    handler: Any,
    api_key: str = "sk-unit-test-key",
) -> OpenAICompatClient:
    """用 MockTransport 建客户端。"""
    return OpenAICompatClient(config, api_key, transport=httpx.MockTransport(handler))


async def test_stream_yields_incremental_text(app_config: AppConfig) -> None:
    """SSE 正常流逐块产出增量文本，请求体字段按接口契约填。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=_sse("你好", "，", "世界"))

    client = _make_client(app_config, handler)
    try:
        chunks = await _collect(
            client.chat_stream([{"role": "user", "content": "问题"}], "req-1")
        )
    finally:
        await client.aclose()

    assert chunks == ["你好", "，", "世界"]
    assert len(captured) == 1
    payload = json.loads(captured[0].content)
    settings = app_config.model
    assert payload["model"] == settings.name
    assert payload["stream"] is True
    assert payload["thinking"] == {"type": settings.thinking}
    assert payload["reasoning_effort"] == settings.reasoning_effort
    assert payload["temperature"] == settings.temperature
    assert payload["max_output_tokens"] == settings.max_output_tokens
    assert payload["messages"] == [{"role": "user", "content": "问题"}]
    assert captured[0].headers["authorization"] == "Bearer sk-unit-test-key"


async def test_done_marker_ends_stream(app_config: AppConfig) -> None:
    """出现 [DONE] 就结束，之后的 data 行不再产出。"""
    body = (
        b'data: {"choices": [{"delta": {"content": "\xe7\xac\xac\xe4\xb8\x80\xe6\xae\xb5"}}]}\n\n'
        b"data: [DONE]\n\n"
        b'data: {"choices": [{"delta": {"content": "\xe4\xb8\x8d\xe5\xba\x94\xe8\xaf\xa5\xe5\x87\xba\xe7\x8e\xb0"}}]}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _make_client(app_config, handler)
    try:
        chunks = await _collect(
            client.chat_stream([{"role": "user", "content": "问题"}], "req-2")
        )
    finally:
        await client.aclose()

    assert chunks == ["第一段"]


async def test_empty_data_lines_are_skipped(app_config: AppConfig) -> None:
    """空的 data 行跳过，不影响后面的内容。"""
    body = (
        b"data: \n\n"
        b"data:\n\n"
        b'data: {"choices": [{"delta": {"content": "\xe6\x9c\x89\xe5\x86\x85\xe5\xae\xb9"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _make_client(app_config, handler)
    try:
        chunks = await _collect(
            client.chat_stream([{"role": "user", "content": "问题"}], "req-3")
        )
    finally:
        await client.aclose()

    assert chunks == ["有内容"]


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error(app_config: AppConfig, status: int) -> None:
    """401 与 403 映射到 ModelAuthError，且不重试。"""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(status)
        return httpx.Response(status, json={"error": "invalid api key"})

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelAuthError):
            await _collect(
                client.chat_stream([{"role": "user", "content": "问题"}], "req-4")
            )
    finally:
        await client.aclose()

    assert len(calls) == 1


async def test_timeout_error(app_config: AppConfig) -> None:
    """读超时映射到 ModelTimeoutError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("读取超时")

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelTimeoutError):
            await _collect(
                client.chat_stream([{"role": "user", "content": "问题"}], "req-5")
            )
    finally:
        await client.aclose()


@pytest.mark.parametrize("status", [400, 500, 503])
async def test_other_status_error(app_config: AppConfig, status: int) -> None:
    """其它 4xx 与 5xx 映射到 ModelStreamError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="服务异常")

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelStreamError):
            await _collect(
                client.chat_stream([{"role": "user", "content": "问题"}], "req-6")
            )
    finally:
        await client.aclose()


async def test_connect_error(app_config: AppConfig) -> None:
    """连接失败映射到 ModelStreamError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("连不上")

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelStreamError):
            await _collect(
                client.chat_stream([{"role": "user", "content": "问题"}], "req-7")
            )
    finally:
        await client.aclose()


async def test_cancel_is_idempotent_and_stops_output(app_config: AppConfig) -> None:
    """取消是幂等的，取消之后不再产出内容。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse("第一段", "第二段"))

    client = _make_client(app_config, handler)
    try:
        stream = client.chat_stream([{"role": "user", "content": "问题"}], "req-8")
        assert await stream.__anext__() == "第一段"
        await client.cancel("req-8")
        # 幂等：重复取消不报错
        await client.cancel("req-8")
        # 从没开始过的请求取消也不报错
        await client.cancel("never-started")
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
    finally:
        await client.aclose()


async def test_cancel_live_stream_ends_cleanly(app_config: AppConfig) -> None:
    """还在流式输出时取消，关流后干净地结束，不抛异常。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_LiveStream(
                b'data: {"choices": [{"delta": {"content": "\xe7\xac\xac\xe4\xb8\x80\xe6\xae\xb5"}}]}\n\n'
            ),
        )

    client = _make_client(app_config, handler)
    try:
        stream = client.chat_stream([{"role": "user", "content": "问题"}], "req-18")
        assert await stream.__anext__() == "第一段"
        await client.cancel("req-18")
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
    finally:
        await client.aclose()


async def test_retries_before_first_output(app_config: AppConfig) -> None:
    """还没收到模型输出时按配置重试一次。"""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500, text="内部错误")
        return httpx.Response(200, content=_sse("成功"))

    client = _make_client(app_config, handler)
    try:
        chunks = await _collect(
            client.chat_stream([{"role": "user", "content": "问题"}], "req-9")
        )
    finally:
        await client.aclose()

    assert chunks == ["成功"]
    assert len(calls) == 2


async def test_no_retry_after_output(app_config: AppConfig) -> None:
    """已经输出部分内容后不重发整个请求。"""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            200,
            stream=_InterruptStream(
                b'data: {"choices": [{"delta": {"content": "\xe9\x83\xa8\xe5\x88\x86"}}]}\n\n'
            ),
        )

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelStreamError):
            await _collect(
                client.chat_stream([{"role": "user", "content": "问题"}], "req-10")
            )
    finally:
        await client.aclose()

    assert len(calls) == 1


async def test_no_retry_when_disabled(app_config: AppConfig) -> None:
    """max_retries 为 0 时不重试。"""
    settings = app_config.model.model_copy(update={"max_retries": 0})
    config = app_config.model_copy(update={"model": settings})
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, text="内部错误")

    client = _make_client(config, handler)
    try:
        with pytest.raises(ModelStreamError):
            await _collect(
                client.chat_stream([{"role": "user", "content": "问题"}], "req-11")
            )
    finally:
        await client.aclose()

    assert len(calls) == 1


async def test_multimodal_body_shape(app_config: AppConfig) -> None:
    """多模态请求体的 content 是数组，图片块带 detail。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=_sse("看图说话"))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这段代码有什么问题"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
        }
    ]
    client = _make_client(app_config, handler)
    try:
        chunks = await _collect(client.multimodal_stream(messages, "req-12"))
    finally:
        await client.aclose()

    assert chunks == ["看图说话"]
    payload = json.loads(captured[0].content)
    blocks = payload["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "这段代码有什么问题"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert blocks[1]["image_url"]["detail"] == "auto"


async def test_check_health_parses_capability(app_config: AppConfig) -> None:
    """能力探测解析 id、输入模态、上下文长度与最大输出。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path.endswith("/models")
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": app_config.model.name,
                        "input_modalities": ["text", "image"],
                        "context_window": 1048576,
                        "max_output_tokens": 393216,
                    }
                ]
            },
        )

    client = _make_client(app_config, handler)
    try:
        capability = await client.check_health()
    finally:
        await client.aclose()

    assert isinstance(capability, ModelCapability)
    assert capability.model_id == app_config.model.name
    assert capability.supports_image is True
    assert capability.context_window == 1048576
    assert capability.max_output_tokens == 393216


async def test_check_health_without_image_modality(app_config: AppConfig) -> None:
    """输入模态里没有 image 时 supports_image 为 False。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": app_config.model.name,
                        "input_modalities": ["text"],
                        "context_window": 8192,
                    }
                ]
            },
        )

    client = _make_client(app_config, handler)
    try:
        capability = await client.check_health()
    finally:
        await client.aclose()

    assert capability.supports_image is False
    assert capability.context_window == 8192
    assert capability.max_output_tokens is None


async def test_check_health_auth_error(app_config: AppConfig) -> None:
    """能力探测遇到 401 映射到 ModelAuthError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelAuthError):
            await client.check_health()
    finally:
        await client.aclose()


async def test_key_and_image_not_logged(
    app_config: AppConfig, log_lines: _ListHandler
) -> None:
    """API Key 与图片内容不进日志。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse("回答"))

    client = _make_client(app_config, handler)
    try:
        await _collect(
            client.multimodal_stream(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "看图"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,QUJDREVGR0hJSktMTU5PUA=="
                                },
                            },
                        ],
                    }
                ],
                "req-13",
            )
        )
    finally:
        await client.aclose()

    text = "\n".join(log_lines.lines)
    assert text
    assert "sk-unit-test-key" not in text
    assert "QUJDREVGR0hJSktMTU5PUA==" not in text


async def test_mock_client_emits_valid_events(app_config: AppConfig) -> None:
    """mock 产出每一行都是合法 JSON，事件类型合法。"""
    client = MockClient(app_config)
    try:
        chunks = await _collect(
            client.chat_stream([{"role": "user", "content": "解释 shared_ptr"}], "req-14")
        )
    finally:
        await client.aclose()

    assert chunks
    # 每块是 10 到 40 个字符的增量片段
    assert all(10 <= len(chunk) <= 40 for chunk in chunks)
    lines = [line for line in "".join(chunks).splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    assert events
    assert all(event["type"] in _EVENT_TYPES for event in events)
    assert events[0]["type"] == "framework"
    assert events[-1]["type"] == "done"
    assert [event["type"] for event in events].count("section_done") == len(
        events[0]["sections"]
    )


async def test_mock_client_titles_follow_question(app_config: AppConfig) -> None:
    """mock 的框架标题跟着题目走，不同问题得到不同标题。"""
    client = MockClient(app_config)

    async def _titles(question: str) -> list[str]:
        chunks = await _collect(
            client.chat_stream([{"role": "user", "content": question}], "req-15")
        )
        events = [
            json.loads(line)
            for line in "".join(chunks).splitlines()
            if line.strip()
        ]
        return [section["title"] for section in events[0]["sections"]]

    first = await _titles("解释 C++ 中 shared_ptr 的引用计数机制")
    second = await _titles("MySQL 索引失效的常见原因")
    assert first != second
    assert "shared_ptr" in " ".join(first)
    assert "MySQL" in " ".join(second)


async def test_mock_client_cancel_stops_output(app_config: AppConfig) -> None:
    """mock 取消之后不再产出内容，取消是幂等的。"""
    client = MockClient(app_config)
    stream = client.chat_stream([{"role": "user", "content": "解释索引"}], "req-16")
    assert await stream.__anext__()
    await client.cancel("req-16")
    await client.cancel("req-16")
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


async def test_mock_client_check_health(app_config: AppConfig) -> None:
    """mock 的能力探测返回 model_id 为 mock、支持图片的能力对象。"""
    client = MockClient(app_config)
    capability = await client.check_health()

    assert isinstance(capability, ModelCapability)
    assert capability.model_id == "mock"
    assert capability.supports_image is True


async def test_mock_client_multimodal_matches_text(app_config: AppConfig) -> None:
    """mock 的多模态产出与文本版一致。"""
    client = MockClient(app_config)
    chunks = await _collect(
        client.multimodal_stream(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "解释索引"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,QUJD"},
                        },
                    ],
                }
            ],
            "req-17",
        )
    )

    lines = [line for line in "".join(chunks).splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    assert events[0]["type"] == "framework"
    assert events[-1]["type"] == "done"
    assert all(event["type"] in _EVENT_TYPES for event in events)


# ----------------------------------------------------------------------
# 非流式 JSON 补全
# ----------------------------------------------------------------------
async def test_complete_json_returns_content_and_sends_non_stream(
    app_config: AppConfig,
) -> None:
    """非流式请求体 stream=false，max_tokens 与 temperature 原样带上。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"type": "coding"}'}}]},
        )

    messages = [{"role": "user", "content": "题面"}]
    client = _make_client(app_config, handler)
    try:
        content = await client.complete_json(
            messages, max_tokens=64, temperature=0.2
        )
    finally:
        await client.aclose()

    assert content == '{"type": "coding"}'
    assert len(captured) == 1
    payload = json.loads(captured[0].content)
    assert payload["model"] == app_config.model.name
    assert payload["messages"] == messages
    assert payload["stream"] is False
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.2
    assert captured[0].headers["authorization"] == "Bearer sk-unit-test-key"


@pytest.mark.parametrize("status", [401, 403])
async def test_complete_json_auth_error(app_config: AppConfig, status: int) -> None:
    """401 与 403 映射到 ModelAuthError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "invalid api key"})

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelAuthError):
            await client.complete_json(
                [{"role": "user", "content": "题面"}],
                max_tokens=64,
                temperature=0.0,
            )
    finally:
        await client.aclose()


@pytest.mark.parametrize("status", [400, 500, 503])
async def test_complete_json_other_status_error(
    app_config: AppConfig, status: int
) -> None:
    """其它 4xx 与 5xx 映射到 ModelStreamError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="服务异常")

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelStreamError):
            await client.complete_json(
                [{"role": "user", "content": "题面"}],
                max_tokens=64,
                temperature=0.0,
            )
    finally:
        await client.aclose()


async def test_complete_json_timeout_error(app_config: AppConfig) -> None:
    """读超时映射到 ModelTimeoutError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("读取超时")

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelTimeoutError):
            await client.complete_json(
                [{"role": "user", "content": "题面"}],
                max_tokens=64,
                temperature=0.0,
            )
    finally:
        await client.aclose()


async def test_complete_json_rejects_malformed_response(
    app_config: AppConfig,
) -> None:
    """响应缺 choices 或 message.content 时抛 ModelStreamError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _make_client(app_config, handler)
    try:
        with pytest.raises(ModelStreamError):
            await client.complete_json(
                [{"role": "user", "content": "题面"}],
                max_tokens=64,
                temperature=0.0,
            )
    finally:
        await client.aclose()
