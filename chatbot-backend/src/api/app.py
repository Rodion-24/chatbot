"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import chat, health
from src.config import get_settings
from src.db import pool as db_pool
from src.observability import flush as flush_observability
from src.providers.factory import build_embedding_provider, build_llm_provider

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    await db_pool.init_pool(settings)
    app.state.llm = build_llm_provider(settings)
    app.state.embedder = build_embedding_provider(settings)
    logger.info(
        "started with llm_provider=%s embedding_provider=%s",
        settings.llm_provider,
        settings.embedding_provider,
    )

    try:
        yield
    finally:
        await app.state.llm.aclose()
        await app.state.embedder.aclose()
        await db_pool.close_pool()
        await flush_observability()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    # The SSE stream is read by a browser on a different origin (the Vite
    # dev server), so the API has to opt in to cross-origin requests.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
