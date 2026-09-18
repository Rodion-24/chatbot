"""Langfuse wiring (langfuse v4.x, OpenTelemetry-based).

Tracing is optional: without keys nothing is initialised and every helper
here turns into a no-op, so the app runs with no observability backend.
An observability failure must never break a request.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from langfuse import Langfuse, observe, propagate_attributes

from src.config import get_settings

logger = logging.getLogger(__name__)

__all__ = ["observe", "trace_context", "update_generation", "update_span", "flush"]


@lru_cache(maxsize=1)
def _client() -> Langfuse | None:
    """Build the Langfuse client once, or None when keys are missing.

    The SDK reads credentials from real environment variables; ours live in
    .env and are parsed by pydantic-settings, so they have to be passed in
    explicitly. Without this the SDK silently disables itself.
    """
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:  # noqa: BLE001 - tracing must not break startup
        logger.warning("langfuse client init failed; tracing disabled", exc_info=True)
        return None


@contextmanager
def trace_context(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    trace_name: str | None = None,
) -> Iterator[None]:
    """Attach trace-level attributes to everything observed inside the block."""
    if _client() is None:
        yield
        return
    try:
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            trace_name=trace_name,
        ):
            yield
    except Exception:  # noqa: BLE001
        logger.debug("langfuse propagate_attributes failed", exc_info=True)
        yield


def update_generation(
    *,
    input: Any = None,  # noqa: A002 - matches the langfuse keyword
    output: Any = None,
    model: str | None = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
    metadata: Any = None,
) -> None:
    """Set model/usage/cost on the active @observe(as_type="generation") span."""
    client = _client()
    if client is None:
        return
    try:
        client.update_current_generation(
            input=input,
            output=output,
            model=model,
            usage_details=usage_details,
            cost_details=cost_details,
            metadata=metadata,
        )
    except Exception:  # noqa: BLE001
        logger.debug("langfuse update_current_generation failed", exc_info=True)


def update_span(**kwargs: Any) -> None:
    """Set attributes on the active @observe span.

    Only valid inside a decorated function — outside one there is no span to
    update and the SDK logs a context error.
    """
    client = _client()
    if client is None:
        return
    try:
        client.update_current_span(**kwargs)
    except Exception:  # noqa: BLE001
        logger.debug("langfuse update_current_span failed", exc_info=True)


async def flush() -> None:
    """Flush buffered spans on shutdown."""
    client = _client()
    if client is None:
        return
    try:
        import asyncio

        # flush() is blocking (HTTP export) — keep it off the event loop.
        await asyncio.to_thread(client.flush)
    except Exception:  # noqa: BLE001
        logger.debug("langfuse flush failed", exc_info=True)
