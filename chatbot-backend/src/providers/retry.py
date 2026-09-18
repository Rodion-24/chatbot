"""Exponential backoff with jitter for provider calls.

The Anthropic and OpenAI SDKs already retry connection errors, 408/409/429
and 5xx internally (`max_retries`). This wrapper covers what they don't:
the outer call as a whole, and raw httpx transport errors from any provider
that talks HTTP directly.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

import httpx

from src.config import Settings

logger = logging.getLogger(__name__)

#: Status codes worth retrying. 4xx other than these signal a bad request.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class RetryableError(Exception):
    """Raised by provider adapters to force a retry."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RetryableError | httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    # SDK exceptions expose `.status_code` without a shared base class.
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in RETRYABLE_STATUS


def _backoff(attempt: int, settings: Settings) -> float:
    """Full-jitter exponential backoff for a 0-based attempt number."""
    ceiling = min(
        settings.retry_initial_backoff_s * (2**attempt),
        settings.retry_max_backoff_s,
    )
    return random.uniform(0, ceiling)


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    settings: Settings,
    op: str = "provider call",
) -> T:
    """Await `fn()`, retrying transient failures with exponential backoff."""
    last: BaseException | None = None

    for attempt in range(settings.retry_max_attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            if not _is_retryable(exc):
                raise
            last = exc
            if attempt == settings.retry_max_attempts - 1:
                break
            delay = _backoff(attempt, settings)
            logger.warning(
                "%s failed (%s), retry %d/%d in %.2fs",
                op,
                type(exc).__name__,
                attempt + 1,
                settings.retry_max_attempts - 1,
                delay,
            )
            await asyncio.sleep(delay)

    assert last is not None
    raise last
