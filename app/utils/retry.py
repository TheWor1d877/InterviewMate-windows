"""重试辅助。

支撑《8-集成与验收》超时管理一节与《8-集成与验收》性能统计一节里「减少耗时的做法」
提到的失败重试，同时满足《8-集成与验收》请求取消一节与《3-文本输入与流式回答》
异常处理一节对取消的处理要求。

规则：

- 超时类异常（``E_DIFY_TIMEOUT``、``E_DEEPSEEK_TIMEOUT``）可重试；
  鉴权类异常（``E_DIFY_AUTH``、``E_DEEPSEEK_AUTH``）不重试，Key 不对重试也没用，直接抛出；
  其它异常默认也不重试，保守处理，宁可少重试也不掩盖真实缺陷。
- ``asyncio.CancelledError`` 原样抛出，绝不重试：用户取消的请求不允许自动恢复。
- 重试用指数退避，初始延迟、倍数、抖动与单次上限都可配；
  sleep 函数可注入，测试里用假函数代替真等。

本模块不引入重依赖，不发网络请求，不读配置文件，不认识模型客户端与界面。
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from app.core.exceptions import (
    DeepSeekAuthError,
    DeepSeekTimeoutError,
    DifyAuthError,
    DifyTimeoutError,
)
from app.utils.logger import log_event

__all__ = [
    "RETRYABLE_EXCEPTIONS",
    "NON_RETRYABLE_EXCEPTIONS",
    "RetryPolicy",
    "default_is_retryable",
    "retry_async",
]


# 默认可重试的异常类型：超时类，错误码见《2-骨架与配置》错误码一节
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    DifyTimeoutError,
    DeepSeekTimeoutError,
)

# 默认不可重试的异常类型：鉴权类，重试不会让 Key 变对
NON_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    DifyAuthError,
    DeepSeekAuthError,
)

T = TypeVar("T")


def default_is_retryable(exc: BaseException) -> bool:
    """默认的可重试判定。

    超时类异常可重试；鉴权类异常不可重试；其它异常一律不重试，
    保守处理，宁可少重试也不掩盖真实缺陷。

    调用方要改判定口径时，把自己的谓词传给 :func:`retry_async` 的
    ``is_retryable`` 参数，不必改这里的默认值。
    """
    if isinstance(exc, NON_RETRYABLE_EXCEPTIONS):
        return False
    return isinstance(exc, RETRYABLE_EXCEPTIONS)


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略。

    退避是指数型：第 ``n`` 次失败后等 ``initial_delay * multiplier ** (n - 1)`` 秒，
    单次不超过 ``max_delay``。

    :param max_retries: 最多重试几次，0 表示不重试；与《2-骨架与配置》配置结构里的
        ``deepseek.max_retries`` 同一个口径。
    :param initial_delay: 第一次重试前的等待秒数。
    :param multiplier: 指数退避的倍数，不能小于 1。
    :param max_delay: 单次等待的上限秒数。
    :param jitter: 抖动比例，取值 0 到 1；0 表示不抖动，
        否则实际等待落在 ``delay * (1 - jitter)`` 到 ``delay * (1 + jitter)`` 之间。
    :param random_fn: 取随机数的函数，测试里注入固定值让抖动可复现。
    """

    max_retries: int = 1
    initial_delay: float = 0.2
    multiplier: float = 2.0
    max_delay: float = 5.0
    jitter: float = 0.0
    random_fn: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries 不能是负数，收到 {self.max_retries!r}")
        if self.initial_delay < 0:
            raise ValueError(f"initial_delay 不能是负数，收到 {self.initial_delay!r}")
        if self.multiplier < 1:
            raise ValueError(f"multiplier 不能小于 1，收到 {self.multiplier!r}")
        if self.max_delay < 0:
            raise ValueError(f"max_delay 不能是负数，收到 {self.max_delay!r}")
        if not 0 <= self.jitter <= 1:
            raise ValueError(f"jitter 必须在 0 到 1 之间，收到 {self.jitter!r}")

    @property
    def max_attempts(self) -> int:
        """最多尝试几次，含第一次。"""
        return self.max_retries + 1

    def delay_for(self, attempt: int) -> float:
        """第 ``attempt`` 次失败后、下一次尝试前等多少秒。

        ``attempt`` 从 1 起数：第一次失败等 ``initial_delay``，之后按 ``multiplier``
        指数增长，单次不超过 ``max_delay``。

        :raises ValueError: ``attempt`` 小于 1。
        """
        if attempt < 1:
            raise ValueError(f"attempt 从 1 起数，收到 {attempt!r}")
        base = self.initial_delay * (self.multiplier ** (attempt - 1))
        delay = min(base, self.max_delay)
        if self.jitter:
            # 抖动按比例放大或缩小，random_fn 返回 0 到 1 之间的数
            delay = delay * (1.0 - self.jitter + 2.0 * self.jitter * self.random_fn())
        return max(0.0, delay)


async def retry_async(
    factory: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    is_retryable: Callable[[BaseException], bool] | None = None,
    is_failure: Callable[[T], bool] | None = None,
    failure_error: Callable[[T], BaseException] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    request_id: str | None = None,
) -> T:
    """执行协程工厂，失败时按退避策略重试。

    ``factory`` 每次调用返回一个 awaitable，每轮重试都重新调用一次，
    也就是说重试会重新发起请求而不是复用上一次的结果。

    可重试判定有两层：

    - 异常级：异常先过 ``is_retryable`` 谓词，谓词为 None 时用
      :func:`default_is_retryable`（只重试超时类异常，鉴权类异常直接抛出）。
      判定为不可重试时立刻原样抛出，不消耗重试次数。
    - 结果级（可选）：有些失败不在异常里，例如 HTTP 返回 5xx。
      给了 ``is_failure`` 就多一层结果判定，谓词返回 True 算失败，
      此时用 ``failure_error`` 把结果转成异常再按同一套退避重试；
      结果级失败一律算可重试，不受 ``is_retryable`` 过滤。
      ``is_failure`` 与 ``failure_error`` 要么一起给、要么都不给；
      不需要结果级判定时不传这两个参数，由调用方自己包一层把失败转成异常。

    ``asyncio.CancelledError`` 原样抛出，绝不重试，
    见《8-集成与验收》请求取消一节：用户取消的请求不允许自动恢复。

    重试次数用尽时抛出最后一次的异常，并把尝试次数写进日志。

    :param factory: 协程工厂，每次调用返回一个新的 awaitable。
    :param policy: 重试策略，None 时用默认策略（最多重试 1 次）。
    :param is_retryable: 异常级可重试谓词，None 时用 :func:`default_is_retryable`。
    :param is_failure: 结果级失败判定，None 表示不做结果级判定。
    :param failure_error: 把失败的结果转成异常的函数，
        给了 ``is_failure`` 就必须给这个。
    :param sleep: 等待函数，签名是 ``async def sleep(delay) -> None``，
        None 时用 :func:`asyncio.sleep`；测试里注入假函数代替真等。
    :param request_id: 请求标识，透传给日志。
    :raises ValueError: 给了 ``is_failure`` 却没有给 ``failure_error``。
    """
    if is_failure is not None and failure_error is None:
        raise ValueError("给了 is_failure 就必须同时给 failure_error")

    active_policy = RetryPolicy() if policy is None else policy
    predicate = default_is_retryable if is_retryable is None else is_retryable
    sleeper = asyncio.sleep if sleep is None else sleep

    attempt = 0
    while True:
        attempt += 1
        try:
            result = await factory()
        except asyncio.CancelledError:
            # 用户取消的请求不重试，原样抛出
            log_event(
                "retry.cancelled",
                "请求被取消，不重试",
                level="WARNING",
                request_id=request_id,
                attempt=attempt,
            )
            raise
        except Exception as exc:
            if not predicate(exc):
                log_event(
                    "retry.not_retryable",
                    f"异常不可重试，直接抛出：{exc}",
                    level="ERROR",
                    request_id=request_id,
                    attempt=attempt,
                    error=str(exc),
                )
                raise
            if attempt >= active_policy.max_attempts:
                log_event(
                    "retry.exhausted",
                    f"重试次数用尽，抛出最后一次的异常：{exc}",
                    level="ERROR",
                    request_id=request_id,
                    attempt=attempt,
                    attempts=attempt,
                    max_retries=active_policy.max_retries,
                    error=str(exc),
                )
                raise
            delay = active_policy.delay_for(attempt)
            log_event(
                "retry.retry",
                "准备重试",
                level="WARNING",
                request_id=request_id,
                attempt=attempt,
                attempts=attempt,
                max_retries=active_policy.max_retries,
                delay_s=delay,
                error=str(exc),
            )
            await sleeper(delay)
            continue

        # 结果级判定：失败不在异常里的情况，例如 HTTP 5xx
        if is_failure is not None and failure_error is not None and is_failure(result):
            error = failure_error(result)
            if attempt >= active_policy.max_attempts:
                log_event(
                    "retry.exhausted",
                    f"重试次数用尽，抛出最后一次的异常：{error}",
                    level="ERROR",
                    request_id=request_id,
                    attempt=attempt,
                    attempts=attempt,
                    max_retries=active_policy.max_retries,
                    error=str(error),
                )
                raise error
            delay = active_policy.delay_for(attempt)
            log_event(
                "retry.retry",
                "准备重试",
                level="WARNING",
                request_id=request_id,
                attempt=attempt,
                attempts=attempt,
                max_retries=active_policy.max_retries,
                delay_s=delay,
                error=str(error),
            )
            await sleeper(delay)
            continue

        return result
