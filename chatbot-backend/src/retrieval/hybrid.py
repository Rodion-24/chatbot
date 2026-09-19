"""Hybrid retrieval: dense vectors + lexical full-text, fused with RRF.

Each half covers the other's blind spot. Vector search matches meaning but
misses exact identifiers — asked for `lists`, it returns chunks that are
merely *about* indexing. Full-text matches the literal term but knows no
synonyms. Reciprocal Rank Fusion merges the two rankings without needing a
shared scale between a cosine distance and a ts_rank.
"""

from __future__ import annotations

import logging

import asyncpg

from src.observability import observe, update_span
from src.providers.base import EmbeddingProvider
from src.retrieval.base import RetrievedChunk

logger = logging.getLogger(__name__)

#: Candidates pulled from each arm before fusion.
CANDIDATES = 20

#: RRF smoothing constant. 60 is the value from the original paper; it damps
#: the gap between ranks 1 and 2 so one arm cannot dominate on its own.
RRF_K = 60

#: Text search config. Must match the one the `tsv` column is generated with
#: (migration 0002) or the stemmed lexemes will not line up.
TS_CONFIG = "english"

SEARCH_SQL = f"""
WITH q AS (
    -- plainto_tsquery ANDs every term, so a natural-language question only
    -- matches a chunk containing *all* of its words — which is almost none.
    -- Rewriting to OR lets ts_rank_cd do the ranking instead of the filter.
    SELECT NULLIF(
        replace(plainto_tsquery('{TS_CONFIG}', $2)::text, ' & ', ' | '),
        ''
    )::tsquery AS tq
),
vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> $1) AS rank
    FROM chunks
    ORDER BY embedding <=> $1
    LIMIT {CANDIDATES}
),
lex AS (
    SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, q.tq) DESC) AS rank
    FROM chunks c, q
    WHERE q.tq IS NOT NULL AND c.tsv @@ q.tq
    ORDER BY ts_rank_cd(c.tsv, q.tq) DESC
    LIMIT {CANDIDATES}
)
SELECT
    c.id, c.doc_id, c.content, c.section_title, c.meta,
    COALESCE(1.0 / ({RRF_K} + vec.rank), 0)
      + COALESCE(1.0 / ({RRF_K} + lex.rank), 0) AS score,
    vec.rank AS vec_rank,
    lex.rank AS lex_rank
FROM vec
FULL OUTER JOIN lex USING (id)
JOIN chunks c USING (id)
ORDER BY score DESC
LIMIT $3
"""


class HybridRetriever:
    """Vector + full-text search fused with Reciprocal Rank Fusion."""

    def __init__(self, embedder: EmbeddingProvider, conn: asyncpg.Connection) -> None:
        self._embedder = embedder
        self._conn = conn

    @observe(name="hybrid_search")
    async def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        vectors, usage = await self._embedder.embed([query])

        rows = await self._conn.fetch(SEARCH_SQL, vectors[0], query, top_k)
        results = [_to_retrieved(row) for row in rows]

        update_span(
            input=query,
            output={"hits": len(results)},
            metadata={
                "top_k": top_k,
                "candidates": CANDIDATES,
                "embed_tokens": usage.input_tokens,
                # How many of the returned chunks each arm contributed, which
                # is the quickest read on whether fusion is doing anything.
                "from_vector": sum(1 for r in rows if r["vec_rank"] is not None),
                "from_lexical": sum(1 for r in rows if r["lex_rank"] is not None),
            },
        )
        return results


def _to_retrieved(row: asyncpg.Record) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["id"],
        doc_id=row["doc_id"],
        content=row["content"],
        # RRF score: higher is better, unlike the raw cosine distance the
        # vector-only retriever reports.
        score=float(row["score"]),
        section_title=row["section_title"],
        meta=row["meta"] or {},
    )
