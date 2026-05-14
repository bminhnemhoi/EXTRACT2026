"""Tenacity policies tuned for the LLM client.

We want short, bounded retries: a stuck request shouldn't burn the API
timeout budget. Network errors and 5xx responses retry up to 3 times with
exponential backoff; everything else fails fast.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity import (
    retry_if_exception as _retry_if_exception,
)

T = TypeVar("T")


def _retryable(exc: BaseException) -> bool:
    """Network noise and transient 5xx responses are worth a retry."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return False


def _log_retry(state: RetryCallState) -> None:
    """Quiet hook — emits nothing today; structured logging hooks here later."""
    return None


def llm_retry() -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory: wrap an LLM call with our standard retry policy."""
    return retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError),
        )
        | _retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        stop=stop_after_attempt(3),
        reraise=True,
        before_sleep=_log_retry,
    )
