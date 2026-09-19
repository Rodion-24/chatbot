"""Retriever selection, driven by config."""

from __future__ import annotations

import asyncpg

from src.config import Settings
from src.providers.base import EmbeddingProvider
from src.retrieval.base import Retriever


def build_retriever(
    settings: Settings, embedder: EmbeddingProvider, conn: asyncpg.Connection
) -> Retriever:
    match settings.retriever:
        case "hybrid":
            from src.retrieval.hybrid import HybridRetriever

            return HybridRetriever(embedder, conn)
        case "vector":
            from src.retrieval.vector import VectorRetriever

            return VectorRetriever(embedder, conn)
        case unknown:  # pragma: no cover - pydantic constrains the literal
            raise ValueError(f"unsupported retriever: {unknown}")
