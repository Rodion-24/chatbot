"""Liveness and readiness."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status

from src.api.schemas import HealthResponse
from src.config import get_settings
from src.db import pool as db_pool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    settings = get_settings()
    database = "down"

    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        database = "up"
    except Exception:  # noqa: BLE001 - health must report, not raise
        logger.warning("database healthcheck failed", exc_info=True)

    ok = database == "up"
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if ok else "degraded",
        database=database,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
    )


@router.get("/health/live")
async def live() -> dict[str, str]:
    """Process is up — no dependency checks."""
    return {"status": "ok"}
