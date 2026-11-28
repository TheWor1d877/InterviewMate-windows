"""模型客户端。

实现《3-文本输入与流式回答》模型客户端一节与《接口契约》里 模型客户端相关的四节：

- 文本请求体、多模态请求体的字段见《接口契约》模型文本请求体与多模态请求体两节；
- 思考模式与推理强度见《接口契约》模型思考模式与推理强度一节；
- 模型能力探测见《接口契约》模型能力探测一节。

本模块只做传输层的事：把请求发出去，把增量文本交出去。
不解析事件、不攒完整回答，事件解析由《3》流式事件协议一节的解析器负责。

一个进程内复用一个 :class:`httpx.AsyncClient`，连接池跟着进程生命周期走。
异常一律映射到《2-骨架与配置》错误码表里已有的类型，不新增异常类。
API Key、Authorization 头与请求体里的图片内容不进日志。

重试由 :mod:`app.utils.retry` 统一管：还没收到模型输出时，
超时类与流读取错误按 ``model.max_retries`` 重试，鉴权错误不重试，
用户取消的请求原样抛出，见《3-文本输入与流式回答》模型客户端一节。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from app.config import AppConfig
from app.core.exceptions import (
    ModelAuthError,
    ModelStreamError,
    ModelTimeoutError,
)
from app.utils.logger import log_event
from app.utils.retry import RetryPolicy, retry_async

__all__ = ["ModelCapability", "OpenAICompatClient"]

# SSE 行的前缀，见《接口契约》模型文本请求体一节
_DATA_PREFIX = "data:"
# SSE 结束标志
_DONE_MARKER = "[DONE]"
# 认证失败的状态码，映射到 E_MODEL_AUTH
_AUTH_STATUS = frozenset({401, 403})


@dataclass(frozen=True)
class ModelCapability:
    """模型能力探测结果。

    字段与《接口契约》模型能力探测一节对得上：
    ``model_id`` 是模型标识，``supports_image`` 表示输入模态里有没有 ``image``，
    ``context_window`` 与 ``max_output_tokens`` 是服务没给时留空。
    """

    model_id: str
    supports_image: bool
    context_window: int | None = None
    max_output_tokens: int | None = None


class OpenAICompatClient:
    """OpenAI 兼容协议的文本与多模态流式客户端。

    :param config: 应用配置，模型客户端相关的取值都从 ``config.model`` 读。
    :param api_key: 模型 API Key，只进 Authorization 头，不进日志。
    :param transport: httpx 传输层，测试时注入 ``httpx.MockTransport``；
        运行时传 None 走真实网络。
    :param retry_sleep: 重试退避的等待函数，签名是 ``async def sleep(delay) -> None``。
        默认用 :func:`asyncio.sleep`；测试里注入假函数代替真等。
    """

    def __init__(
        self,
        config: AppConfig,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        retry_sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._retry_sleep = retry_sleep
        settings = config.model
        self._client = httpx.AsyncClient(
            # 四个超时都要显式给，httpx 不接受只给一部分的 Timeout
            timeout=httpx.Timeout(
                connect=settings.connect_timeout,
                read=settings.read_timeout,
                write=settings.read_timeout,
                pool=settings.connect_timeout,
            ),
            transport=transport,
        )
        # request_id -> 正在读取的响应流，取消时关掉它
        self._streams: dict[str, httpx.Response] = {}
        # 被取消的 request_id：取消后不再产出内容，也不重试
        self._cancelled: set[str] = set()

    async def chat_stream(
        self, messages: list[dict[str, str]], request_id: str
    ) -> AsyncIterator[str]:
        """发起文本流式请求，逐段产出增量文本。

        每一段增量文本立刻 yield 出去，不在客户端里攒成完整字符串。
        """
        async for chunk in self._stream(self._build_payload(messages), request_id):
            yield chunk

    async def multimodal_stream(
        self, messages: list[dict], request_id: str
    ) -> AsyncIterator[str]:
        """发起多模态流式请求，逐段产出增量文本。

        ``messages`` 里用户消息的 ``content`` 是数组，元素是文本块或图片块，
        见《接口契约》模型多模态请求体一节。
        """
        payload = self._build_payload(_normalize_multimodal_messages(messages))
        async for chunk in self._stream(payload, request_id):
            yield chunk

    async def cancel(self, request_id: str) -> None:
        """取消指定请求：关掉它的响应流，之后不再产出内容。

        幂等，对已结束或从未开始的 request_id 调用都不报错。
        """
        self._cancelled.add(request_id)
        response = self._streams.pop(request_id, None)
        if response is None:
            return
        try:
            await response.aclose()
        except Exception:
            # 关流失败不影响取消语义，记一条日志就返回
            log_event(
                "model.stream.cancel",
                "关闭响应流失败",
                level="WARNING",
                request_id=request_id,
            )

    async def check_health(self) -> ModelCapability:
        """探测模型能力，见《接口契约》模型能力探测一节。

        向 ``{base_url}/models`` 发 GET，解析 ``data`` 数组里的 id、input_modalities、
        context_window 与 max_output_tokens。输入模态里没有 image 时
        ``supports_image`` 为 False。

        模型支不支持图片由调用方决定，客户端只报结果：输入模态里没有 ``image``
        也不影响截图采集与提交，见《4-截图输入与多模态》模型视觉能力一节。
        """
        url = self._build_url("/models")
        try:
            response = await self._client.get(url, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"模型能力探测超时：{exc}") from exc
        except httpx.HTTPError as exc:
            raise ModelStreamError(f"模型能力探测失败：{exc}") from exc
        self._raise_for_status(response, "模型能力探测")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelStreamError("模型列表响应不是合法 JSON") from exc
        entry = _pick_model_entry(payload, self._config.model.name)
        capability = ModelCapability(
            model_id=str(entry.get("id") or self._config.model.name),
            supports_image=_has_image_modality(entry.get("input_modalities")),
            context_window=_as_int(entry.get("context_window")),
            max_output_tokens=_as_int(entry.get("max_output_tokens")),
        )
        log_event(
            "model.health.ok",
            "模型能力探测完成",
            request_id=None,
            model_id=capability.model_id,
            supports_image=capability.supports_image,
        )
        return capability

    async def aclose(self) -> None:
        """关闭 HTTP 客户端，进程收尾时调用。"""
        await self._client.aclose()

    async def complete_json(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """发一次非流式请求，返回 ``choices[0].message.content``。

        题型判定这类一次性小请求走这里，与流式方法共用基址、请求头、
        超时与错误码映射，见《接口契约》模型文本请求体一节。

        :param messages: 请求消息列表。
        :param max_tokens: 回复的最大 token 数。
        :param temperature: 采样温度。
        :return: 模型回复的完整文本。
        :raises ModelAuthError: 401 或 403。
        :raises ModelTimeoutError: 请求超时。
        :raises ModelStreamError: 其它 HTTP 错误、响应不是合法 JSON 或结构对不上。
        """
        url = self._build_url("/chat/completions")
        payload = {
            "model": self._config.model.name,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = await self._client.post(
                url, json=payload, headers=self._headers()
            )
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"模型非流式请求超时：{exc}") from exc
        except httpx.HTTPError as exc:
            raise ModelStreamError(f"模型非流式请求失败：{exc}") from exc
        self._raise_for_status(response, "非流式请求")
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelStreamError("模型非流式响应不是合法 JSON") from exc
        content = _extract_message_content(data)
        log_event(
            "model.complete_json.done",
            "非流式请求结束",
            request_id=None,
            content_length=len(content),
        )
        return content

    async def _stream(
        self, payload: dict[str, Any], request_id: str
    ) -> AsyncIterator[str]:
        """发一次流式请求，必要时在没有任何输出时重试。

        重试规则见《3-文本输入与流式回答》模型客户端一节：
        还没收到任何模型输出时最多按 ``config.model.max_retries`` 重试；
        已经输出部分内容后不重发整个请求；用户取消的请求不重试。

        重试由 :func:`~app.utils.retry.retry_async` 统一管：
        超时类与流读取错误可重试，鉴权错误不重试，
        ``asyncio.CancelledError`` 原样抛出。
        """
        try:
            first, stream = await retry_async(
                lambda: self._open_stream(payload, request_id),
                policy=self._retry_policy(),
                is_retryable=_is_retryable_stream_error,
                sleep=self._retry_sleep,
                request_id=request_id,
            )
            if first is not None:
                yield first
            async for chunk in stream:
                yield chunk
            log_event(
                "model.stream.done",
                "流式请求结束",
                request_id=request_id,
            )
        finally:
            self._cancelled.discard(request_id)

    def _retry_policy(self) -> RetryPolicy:
        """重试策略：重试次数上限就是配置里的 ``model.max_retries``。

        退避是指数型的，单次不超过 5 秒，见《8-集成与验收》超时管理一节。
        """
        return RetryPolicy(max_retries=self._config.model.max_retries)

    async def _open_stream(
        self, payload: dict[str, Any], request_id: str
    ) -> tuple[str | None, AsyncIterator[str]]:
        """开一次流并取出第一个片段。

        第一个片段之前的失败（连接超时、读流错误、HTTP 错误）都在这里抛出来，
        由重试那层判定要不要重试。第一个片段之后的失败不再重发整个请求：
        已经产出的内容会重复，见《3-文本输入与流式回答》模型客户端一节。

        :param payload: 请求体。
        :param request_id: 本次请求的标识。
        :returns: 第一个片段（没有内容时是 None）与这条流。
        """
        stream = self._stream_once(payload, request_id)
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            # 空流：没有内容可产出
            return None, stream
        except Exception:
            # 这次尝试失败：把流收干净，别把响应连接挂着
            await stream.aclose()
            raise
        return first, stream

    async def _stream_once(
        self, payload: dict[str, Any], request_id: str
    ) -> AsyncIterator[str]:
        """跑一次流式请求，逐行读 SSE 并产出增量文本。"""
        url = self._build_url("/chat/completions")
        try:
            async with self._client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as response:
                self._streams[request_id] = response
                try:
                    self._raise_for_status(response, "流式请求")
                    async for line in response.aiter_lines():
                        if request_id in self._cancelled:
                            # 用户取消，不再产出内容
                            return
                        stripped = line.strip()
                        if not stripped.startswith(_DATA_PREFIX):
                            continue
                        data = stripped[len(_DATA_PREFIX) :].strip()
                        if not data:
                            # 空的 data 行跳过
                            continue
                        if data == _DONE_MARKER:
                            break
                        content = _extract_delta_content(data)
                        if content:
                            yield content
                finally:
                    self._streams.pop(request_id, None)
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"模型请求超时：{exc}") from exc
        except (httpx.HTTPError, httpx.StreamError) as exc:
            # httpx.StreamError 不是 HTTPError 的子类，关闭流后读会抛它
            if request_id in self._cancelled:
                # 取消导致的读失败不算流中断
                return
            raise ModelStreamError(f"模型流式请求失败：{exc}") from exc

    def _build_payload(self, messages: list[dict]) -> dict[str, Any]:
        """组装请求体，字段见《接口契约》模型文本请求体一节。"""
        settings = self._config.model
        return {
            "model": settings.name,
            "messages": messages,
            "stream": True,
            "thinking": {"type": settings.thinking},
            "reasoning_effort": settings.reasoning_effort,
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
        }

    def _build_url(self, path: str) -> str:
        """拼请求地址，基址末尾的斜杠去掉。"""
        base = self._config.model.base_url.rstrip("/")
        return f"{base}{path}"

    def _headers(self) -> dict[str, str]:
        """请求头。

        API Key 只出现在这里，不进日志字段。
        """
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response, action: str) -> None:
        """按状态码映射异常，见《2-骨架与配置》错误码一节。"""
        status = response.status_code
        if status in _AUTH_STATUS:
            raise ModelAuthError(f"模型 {action}认证失败（HTTP {status}），检查 API Key")
        if status >= 400:
            raise ModelStreamError(f"模型 {action}返回 HTTP {status}")


def _is_retryable_stream_error(exc: BaseException) -> bool:
    """流式请求里哪些异常值得重试。

    超时类与流读取错误重试；鉴权错误不重试：Key 不对时重试多少次都是一样的，
    见《3-文本输入与流式回答》模型客户端一节。
    用户取消的 ``asyncio.CancelledError`` 由重试那层原样抛出，不在这里判定。

    :param exc: 捕获到的异常。
    """
    if isinstance(exc, ModelAuthError):
        return False
    return isinstance(exc, (ModelTimeoutError, ModelStreamError))


def _extract_delta_content(data: str) -> str:
    """从一条 SSE 数据里取增量文本。

    取 ``choices[0].delta.content``；不是合法 JSON 或结构对不上时返回空串，
    这类容错由《3》解析器容错一节兜底，客户端不因此中断整条流。
    """
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _extract_message_content(payload: Any) -> str:
    """从非流式响应里取 ``choices[0].message.content``。

    结构对不上时抛 :class:`ModelStreamError`：非流式请求只取一次内容，
    没有增量解析那层容错可以兜底。

    :param payload: 响应 JSON 解析出来的对象。
    """
    if not isinstance(payload, dict):
        raise ModelStreamError("模型非流式响应不是对象")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelStreamError("模型非流式响应缺少 choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ModelStreamError("模型非流式响应的 choices 结构不对")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ModelStreamError("模型非流式响应缺少 message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ModelStreamError("模型非流式响应缺少 content")
    return content


def _normalize_multimodal_messages(messages: list[dict]) -> list[dict]:
    """补齐多模态消息里的图片块缺省字段。

    图片块的 ``image_url`` 缺 ``detail`` 时补成 ``auto``，其余原样保留。
    不改调用方传进来的对象。
    """
    normalized: list[dict] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue
        blocks: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                image = block.get("image_url")
                if isinstance(image, dict) and "detail" not in image:
                    block = {**block, "image_url": {**image, "detail": "auto"}}
            blocks.append(block)
        normalized.append({**message, "content": blocks})
    return normalized


def _pick_model_entry(payload: Any, model: str) -> dict[str, Any]:
    """从模型列表里挑出当前模型的条目。

    优先取 id 等于 ``model`` 的那条；没有就取第一条。
    """
    if not isinstance(payload, dict):
        raise ModelStreamError("模型列表响应不是对象")
    items = payload.get("data")
    if not isinstance(items, list) or not items:
        raise ModelStreamError("模型列表为空")
    entries = [item for item in items if isinstance(item, dict)]
    if not entries:
        raise ModelStreamError("模型列表没有可用条目")
    for entry in entries:
        if entry.get("id") == model:
            return entry
    return entries[0]


def _has_image_modality(modalities: Any) -> bool:
    """输入模态里有没有 ``image``。"""
    if not isinstance(modalities, list):
        return False
    return any(str(item) == "image" for item in modalities)


def _as_int(value: Any) -> int | None:
    """把服务给的数值字段转成 int，不是整数就返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
