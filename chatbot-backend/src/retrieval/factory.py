"""Retriever selection, driven by config."""

from __future__ import annotations

import asyncpg

from src.config import Settings
from src.providers.base import EmbeddingProvider
from src.retrieval.base import Retriever


def build_retriever(
    settings: Settings, embedder: EmbeddingProvider, conn: asyncpg.Connection
) -> Retriever:
    """Build the configured retriever, wrapped in a re-ranker when enabled."""
    match settings.retriever:
        case "hybrid":
            from src.retrieval.hybrid import HybridRetriever

            retriever: Retriever = HybridRetriever(embedder, conn)
        case "vector":
            from src.retrieval.vector import VectorRetriever

            retriever = VectorRetriever(embedder, conn)
        case unknown:  # pragma: no cover - pydantic constrains the literal
            raise ValueError(f"unsupported retriever: {unknown}")

    if settings.rerank_enabled:
        # Wraps rather than replaces, so re-ranking composes with either search.
        from src.retrieval.rerank import RerankingRetriever

        retriever = RerankingRetriever(
            retriever, settings, candidates=settings.rerank_candidates
        )

    return retriever
