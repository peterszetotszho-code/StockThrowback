"""Failure classification and retry with exponential backoff + jitter."""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class TransientError(Exception):
    """Retryable failure: network, timeout, rate-limit, upstream 5xx/429."""


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    retryable: tuple[type[Exception], ...] = (TransientError,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> T:
    """Run ``fn``, retrying on retryable exceptions with exponential backoff.

    Backoff formula: ``delay = min(max_delay, base_delay * 2 ** attempt)``
    multiplied by ``(1 + jitter * random.random())``.

    Args:
        fn: Zero-argument callable.
        max_attempts: Total attempts before giving up.
        base_delay: Initial delay in seconds.
        max_delay: Upper bound on the delay in seconds.
        jitter: Fraction of the delay added as random jitter.
        retryable: Exception types that trigger a retry.
        on_retry: Optional callback ``(exception, attempt_number)``.

    Returns:
        The return value of ``fn``.

    Raises:
        The last retryable exception if all attempts are exhausted.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable as exc:
            if attempt == max_attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay *= 1 + jitter * random.random()
            if on_retry is not None:
                on_retry(exc, attempt + 1)
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
