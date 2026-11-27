"""重试辅助的测试。

覆盖《8-集成与验收》超时管理一节与《8-集成与验收》请求取消一节对重试的要求：
超时类异常会重试、鉴权类异常不重试、取消原样抛出、退避延迟可注入且测试里不真等。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterator

import pytest

from app.core.exceptions import (
    DeepSeekAuthError,
    DeepSeekTimeoutError,
    DifyAuthError,
    DifyResponseError,
    DifyTimeoutError,
)
from app.utils.retry import (
    NON_RETRYABLE_EXCEPTIONS,
    RETRYABLE_EXCEPTIONS,
    RetryPolicy,
    default_is_retryable,
    retry_async,
)


class _FakeSleep:
    """假 sleep：记录每次等待的秒数，不真等。"""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class _Sequence:
    """按顺序产出结果或抛出异常，记录被调用了几次。

    异常按实例原样抛出，所以测试能断言最终抛出的就是最后一个异常。
    """

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self) -> Any:
        index = min(self.calls, len(self.outcomes) - 1)
        self.calls += 1
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def retry_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """收集 interviewmate 记录器上的重试日志。

    临时打开 propagate，让记录走到 pytest 的 handler 上。
    """
    project_logger = logging.getLogger("interviewmate")
    old_propagate = project_logger.propagate
    project_logger.propagate = True
    caplog.set_level(logging.INFO, logger="interviewmate")
    try:
        yield caplog
    finally:
        project_logger.propagate = old_propagate


def _events(caplog: pytest.LogCaptureFixture, event: str) -> list[logging.LogRecord]:
    """挑出指定事件名的日志记录。"""
    return [r for r in caplog.records if getattr(r, "event", "") == event]


async def test_timeout_failure_then_success_retries() -> None:
    """第一次失败第二次成功：会重试，并拿到第二次的结果。"""
    sleeper = _FakeSleep()
    call = _Sequence(DeepSeekTimeoutError("超时"), "成功结果")

    result = await retry_async(call, policy=RetryPolicy(max_retries=2), sleep=sleeper)

    assert result == "成功结果"
    assert call.calls == 2
    assert len(sleeper.delays) == 1


async def test_dify_timeout_error_also_retries() -> None:
    """Dify 超时（E_DIFY_TIMEOUT）同样可重试。"""
    sleeper = _FakeSleep()
    call = _Sequence(DifyTimeoutError("检索超时"), "片段")

    result = await retry_async(call, policy=RetryPolicy(max_retries=1), sleep=sleeper)

    assert result == "片段"
    assert call.calls == 2


async def test_exhausted_raises_last_exception() -> None:
    """超过最大次数：抛出最后一次的异常，不是第一次的。"""
    sleeper = _FakeSleep()
    last = DeepSeekTimeoutError("最后一次")
    call = _Sequence(DeepSeekTimeoutError("第一次"), last)

    with pytest.raises(DeepSeekTimeoutError) as info:
        await retry_async(call, policy=RetryPolicy(max_retries=1), sleep=sleeper)

    assert info.value is last
    assert call.calls == 2


async def test_auth_error_is_not_retried() -> None:
    """鉴权异常不重试，直接抛出，也不等待。"""
    sleeper = _FakeSleep()
    call = _Sequence(DifyAuthError("Key 不对"))

    with pytest.raises(DifyAuthError):
        await retry_async(call, policy=RetryPolicy(max_retries=3), sleep=sleeper)

    assert call.calls == 1
    assert sleeper.delays == []


async def test_deepseek_auth_error_is_not_retried() -> None:
    """DeepSeek 鉴权异常同样不重试。"""
    sleeper = _FakeSleep()
    call = _Sequence(DeepSeekAuthError("Key 不对"))

    with pytest.raises(DeepSeekAuthError):
        await retry_async(call, policy=RetryPolicy(max_retries=3), sleep=sleeper)

    assert call.calls == 1


async def test_cancelled_error_propagates_without_retry() -> None:
    """取消原样抛出且不重试：用户取消的请求不允许自动恢复。"""
    sleeper = _FakeSleep()
    call = _Sequence(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await retry_async(call, policy=RetryPolicy(max_retries=3), sleep=sleeper)

    assert call.calls == 1
    assert sleeper.delays == []


async def test_cancelled_error_not_retried_even_with_always_retry_predicate() -> None:
    """即使谓词对所有异常都说可重试，取消也不重试。"""
    sleeper = _FakeSleep()
    call = _Sequence(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await retry_async(
            call,
            policy=RetryPolicy(max_retries=3),
            is_retryable=lambda exc: True,
            sleep=sleeper,
        )

    assert call.calls == 1
    assert sleeper.delays == []


async def test_backoff_delays_are_injected_and_not_actually_waited() -> None:
    """退避延迟走注入的 sleep，测试里没有真实等待。"""
    sleeper = _FakeSleep()
    call = _Sequence(DeepSeekTimeoutError("超时"))
    policy = RetryPolicy(max_retries=3, initial_delay=0.5, multiplier=2.0)

    start = time.perf_counter()
    with pytest.raises(DeepSeekTimeoutError):
        await retry_async(call, policy=policy, sleep=sleeper)
    elapsed = time.perf_counter() - start

    assert sleeper.delays == [0.5, 1.0, 2.0]
    assert elapsed < 0.5


async def test_success_does_not_retry() -> None:
    """一次成功就不再重试。"""
    sleeper = _FakeSleep()
    call = _Sequence("成功")

    result = await retry_async(call, policy=RetryPolicy(max_retries=3), sleep=sleeper)

    assert result == "成功"
    assert call.calls == 1
    assert sleeper.delays == []


async def test_zero_max_retries_does_not_retry() -> None:
    """max_retries 为 0 时只试一次。"""
    sleeper = _FakeSleep()
    call = _Sequence(DeepSeekTimeoutError("超时"))

    with pytest.raises(DeepSeekTimeoutError):
        await retry_async(call, policy=RetryPolicy(max_retries=0), sleep=sleeper)

    assert call.calls == 1
    assert sleeper.delays == []


async def test_unlisted_exception_is_not_retried_by_default() -> None:
    """未列入的异常默认不重试，保守处理。"""
    sleeper = _FakeSleep()
    call = _Sequence(DifyResponseError("响应格式异常"))

    with pytest.raises(DifyResponseError):
        await retry_async(call, policy=RetryPolicy(max_retries=3), sleep=sleeper)

    assert call.calls == 1
    assert sleeper.delays == []


async def test_injected_predicate_can_widen_retry() -> None:
    """注入谓词后由调用方决定要不要重试。"""
    sleeper = _FakeSleep()
    call = _Sequence(DifyResponseError("响应格式异常"), "片段")

    result = await retry_async(
        call,
        policy=RetryPolicy(max_retries=2),
        is_retryable=lambda exc: isinstance(exc, DifyResponseError),
        sleep=sleeper,
    )

    assert result == "片段"
    assert call.calls == 2


async def test_result_level_failure_is_retried() -> None:
    """结果级失败（HTTP 5xx 这类不在异常里的失败）也按退避重试。"""
    sleeper = _FakeSleep()
    call = _Sequence(500, 200)

    result = await retry_async(
        call,
        policy=RetryPolicy(max_retries=2),
        is_failure=lambda code: code >= 500,
        failure_error=lambda code: DifyResponseError(f"HTTP {code}"),
        sleep=sleeper,
    )

    assert result == 200
    assert call.calls == 2
    assert len(sleeper.delays) == 1


async def test_result_level_failure_exhausted_raises_failure_error() -> None:
    """结果级失败也受最大次数限制，最后抛出 failure_error 造的异常。"""
    sleeper = _FakeSleep()
    call = _Sequence(503)

    with pytest.raises(DifyResponseError):
        await retry_async(
            call,
            policy=RetryPolicy(max_retries=1),
            is_failure=lambda code: code >= 500,
            failure_error=lambda code: DifyResponseError(f"HTTP {code}"),
            sleep=sleeper,
        )

    assert call.calls == 2


async def test_result_level_failure_needs_failure_error() -> None:
    """只给 is_failure 不给 failure_error 是用法错误。"""
    with pytest.raises(ValueError):
        await retry_async(_Sequence(500), is_failure=lambda code: code >= 500)


async def test_exhausted_log_records_attempts(retry_logs: pytest.LogCaptureFixture) -> None:
    """次数用尽时把试了几次写进日志。"""
    sleeper = _FakeSleep()
    call = _Sequence(DeepSeekTimeoutError("超时"))

    with pytest.raises(DeepSeekTimeoutError):
        await retry_async(
            call,
            policy=RetryPolicy(max_retries=2),
            sleep=sleeper,
            request_id="req-1",
        )

    exhausted = _events(retry_logs, "retry.exhausted")
    assert len(exhausted) == 1
    fields = getattr(exhausted[0], "fields", {})
    assert fields["attempts"] == 3
    assert fields["max_retries"] == 2
    assert getattr(exhausted[0], "request_id", None) == "req-1"

    retries = _events(retry_logs, "retry.retry")
    assert [getattr(r, "fields", {}).get("attempt") for r in retries] == [1, 2]


async def test_not_retryable_is_logged(retry_logs: pytest.LogCaptureFixture) -> None:
    """不可重试的异常也会写一条日志，便于排查。"""
    sleeper = _FakeSleep()
    call = _Sequence(DifyAuthError("Key 不对"))

    with pytest.raises(DifyAuthError):
        await retry_async(call, policy=RetryPolicy(max_retries=2), sleep=sleeper)

    assert len(_events(retry_logs, "retry.not_retryable")) == 1
    assert _events(retry_logs, "retry.retry") == []


def test_default_is_retryable_covers_timeout_and_auth() -> None:
    """默认谓词：超时可重试，鉴权不可重试。"""
    assert default_is_retryable(DifyTimeoutError("超时")) is True
    assert default_is_retryable(DeepSeekTimeoutError("超时")) is True
    assert default_is_retryable(DifyAuthError("Key 不对")) is False
    assert default_is_retryable(DeepSeekAuthError("Key 不对")) is False
    assert default_is_retryable(DifyResponseError("响应异常")) is False
    assert default_is_retryable(ValueError("其它")) is False


def test_exception_type_groups_match_docs() -> None:
    """异常分组与《2-骨架与配置》错误码一节对得上。"""
    assert RETRYABLE_EXCEPTIONS == (DifyTimeoutError, DeepSeekTimeoutError)
    assert NON_RETRYABLE_EXCEPTIONS == (DifyAuthError, DeepSeekAuthError)


def test_policy_caps_delay_at_max_delay() -> None:
    """单次等待不超过 max_delay。"""
    policy = RetryPolicy(max_retries=3, initial_delay=1.0, multiplier=10.0, max_delay=2.0)

    assert [policy.delay_for(n) for n in (1, 2, 3)] == [1.0, 2.0, 2.0]


def test_policy_jitter_stays_in_bounds() -> None:
    """抖动按比例放大或缩小等待时间。"""
    low = RetryPolicy(initial_delay=1.0, jitter=0.5, random_fn=lambda: 0.0)
    high = RetryPolicy(initial_delay=1.0, jitter=0.5, random_fn=lambda: 1.0)

    assert low.delay_for(1) == pytest.approx(0.5)
    assert high.delay_for(1) == pytest.approx(1.5)


def test_policy_rejects_invalid_values() -> None:
    """非法参数当场报错，不等到用的时候才炸。"""
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
    with pytest.raises(ValueError):
        RetryPolicy(initial_delay=-1.0)
    with pytest.raises(ValueError):
        RetryPolicy(multiplier=0.5)
    with pytest.raises(ValueError):
        RetryPolicy(max_delay=-1.0)
    with pytest.raises(ValueError):
        RetryPolicy(jitter=1.5)


def test_policy_max_attempts_counts_first_call() -> None:
    """max_attempts 含第一次尝试。"""
    assert RetryPolicy(max_retries=2).max_attempts == 3
    assert RetryPolicy(max_retries=0).max_attempts == 1


def test_delay_for_rejects_bad_attempt() -> None:
    """attempt 从 1 起数。"""
    with pytest.raises(ValueError):
        RetryPolicy().delay_for(0)
