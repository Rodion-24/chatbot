"""Naive vector search: exact scan, no index.

Stage 2 baseline. Every query compares against every row, which is the
point: an exhaustive scan gives exact results to measure an approximate
index (HNSW) against later.
"""

from __future__ import annotations

import logging

import asyncpg

from src.observability import observe, update_span
from src.providers.base import EmbeddingProvider
from src.retrieval.base import RetrievedChunk

logger = logging.getLogger(__name__)

#: Columns returned by a search. `embedding` is deliberately absent: it is
#: used inside the ORDER BY expression and never shipped back to Python.
SEARCH_COLS = "c.id, c.doc_id, c.content, c.section_title, c.meta"


class VectorRetriever:
    """Finds chunks whose embedding is closest to the query embedding."""

    def __init__(self, embedder: EmbeddingProvider, conn: asyncpg.Connection) -> None:
        self._embedder = embedder
        self._conn = conn

    @observe(name="vector_search")
    async def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """Embed `query` and return the `top_k` nearest chunks."""
        vectors, usage = await self._embedder.embed([query])
        query_vector = vectors[0]

        rows = await self._conn.fetch(
            f"""
            SELECT {SEARCH_COLS}, c.embedding <=> $1 AS distance
            FROM chunks c
            ORDER BY c.embedding <=> $1
            LIMIT $2
            """,
            query_vector,
            top_k,
        )

        results = [_to_retrieved(row) for row in rows]

        update_span(
            input=query,
            output={"hits": len(results)},
            metadata={
                "top_k": top_k,
                "embed_tokens": usage.input_tokens,
                "best_distance": results[0].score if results else None,
                "embed_model": self._embedder.model,
            },
        )
        logger.debug("search %r returned %d chunks", query, len(results))
        return results


def _to_retrieved(row: asyncpg.Record) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["id"],
        doc_id=row["doc_id"],
        content=row["content"],
        # Cosine distance: 0.0 is identical, 2.0 is opposite. Lower is better.
        score=float(row["distance"]),
        section_title=row["section_title"],
        meta=row["meta"] or {},
    )
