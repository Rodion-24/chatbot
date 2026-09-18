"""Retry/backoff behaviour for provider calls."""

from __future__ import annotations

import httpx
import pytest

from src.config import Settings
from src.providers.retry import RetryableError, with_retry


@pytest.fixture
def settings() -> Settings:
    # Near-zero backoff keeps the test fast.
    return Settings(
        retry_max_attempts=4,
        retry_initial_backoff_s=0.001,
        retry_max_backoff_s=0.002,
    )


async def test_returns_immediately_on_success(settings: Settings) -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert await with_retry(fn, settings=settings) == "ok"
    assert calls == 1


async def test_retries_then_succeeds(settings: Settings) -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RetryableError("transient")
        return "ok"

    assert await with_retry(fn, settings=settings) == "ok"
    assert calls == 3


async def test_gives_up_after_max_attempts(settings: Settings) -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise RetryableError("always down")

    with pytest.raises(RetryableError):
        await with_retry(fn, settings=settings)
    assert calls == settings.retry_max_attempts


async def test_does_not_retry_non_retryable(settings: Settings) -> None:
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        await with_retry(fn, settings=settings)
    assert calls == 1


async def test_retries_retryable_status_code(settings: Settings) -> None:
    calls = 0
    request = httpx.Request("POST", "https://example.test")

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.HTTPStatusError(
                "rate limited",
                request=request,
                response=httpx.Response(429, request=request),
            )
        return "ok"

    assert await with_retry(fn, settings=settings) == "ok"
    assert calls == 2


async def test_does_not_retry_client_error_status(settings: Settings) -> None:
    calls = 0
    request = httpx.Request("POST", "https://example.test")

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise httpx.HTTPStatusError(
            "bad request",
            request=request,
            response=httpx.Response(400, request=request),
        )

    with pytest.raises(httpx.HTTPStatusError):
        await with_retry(fn, settings=settings)
    assert calls == 1
