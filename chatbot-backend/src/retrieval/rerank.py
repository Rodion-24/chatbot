"""Re-ranking wrapper around any retriever.

Embedding search compresses a chunk into 1536 numbers, which is fast enough
to scan a whole corpus but blunt: detail is lost in the compression. A
re-ranker reads the question and the candidate text together, so it can tell
apart chunks whose vectors look alike. That is far too slow for the corpus,
which is why it only ever sees the shortlist a cheaper search produced.

Wraps a retriever rather than replacing one, so it composes with either the
vector or the hybrid search.
"""

from __future__ import annotations

import json
import logging

from anthropic import AsyncAnthropic

from src.config import Settings
from src.generation.prompts import load_prompt
from src.observability import observe, update_span
from src.providers.base import Usage
from src.providers.pricing import estimate_cost_usd
from src.providers.retry import with_retry
from src.retrieval.base import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

RERANK_PROMPT = "rerank_v1"

#: Chunks are truncated before being sent: the re-ranker needs enough to
#: judge relevance, not the whole passage, and the prompt is billed per token.
EXCERPT_CHARS = 700

#: Forces a list of indices back, so nothing has to be parsed out of prose.
RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "indices": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Excerpt numbers, most relevant first.",
        }
    },
    "required": ["indices"],
    "additionalProperties": False,
}


class RerankingRetriever(Retriever):
    """Fetches `candidates` from an inner retriever, returns the best `top_k`."""

    def __init__(
        self,
        inner: Retriever,
        settings: Settings,
        *,
        candidates: int = 20,
        model: str | None = None,
    ) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the reranker")
        self._inner = inner
        self._settings = settings
        self._candidates = candidates
        self._model = model or settings.rerank_model
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=2)

    @observe(name="rerank")
    async def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        shortlist = await self._inner.search(query, top_k=self._candidates)
        if len(shortlist) <= top_k:
            # Nothing to choose between — skip the call and the cost.
            return shortlist

        order, cost = await self._rank(query, shortlist, top_k)
        if order is None:
            logger.warning("rerank failed; falling back to the original order")
            return shortlist[:top_k]

        ranked = [shortlist[i] for i in order]
        update_span(
            input=query,
            output={"kept": len(ranked)},
            metadata={
                "candidates": len(shortlist),
                "top_k": top_k,
                "model": self._model,
                "cost_usd": cost,
                # How far the re-ranker moved things: 0 means it agreed with
                # the search, higher means it disagreed.
                "rank_shift": sum(abs(i - pos) for pos, i in enumerate(order)),
            },
        )
        return ranked

    async def _rank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> tuple[list[int] | None, float]:
        numbered = "\n\n".join(
            f"[{i}] {c.section_title or ''}\n{c.content[:EXCERPT_CHARS]}"
            for i, c in enumerate(chunks)
        )
        user = (
            f"Question:\n{query}\n\n"
            f"Excerpts:\n{numbered}\n\n"
            f"Return the {top_k} most relevant excerpt numbers, best first."
        )

        async def _call():
            return await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=load_prompt(RERANK_PROMPT),
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": RANKING_SCHEMA}},
            )

        try:
            msg = await with_retry(_call, settings=self._settings, op="rerank")
        except Exception:  # noqa: BLE001 - degrades to the search order
            logger.exception("rerank call failed")
            return None, 0.0

        text = "".join(b.text for b in msg.content if b.type == "text")
        try:
            indices = json.loads(text)["indices"]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("rerank returned unusable JSON: %.200s", text)
            return None, 0.0

        usage = Usage(
            input_tokens=msg.usage.input_tokens or 0,
            output_tokens=msg.usage.output_tokens or 0,
        )
        cost = estimate_cost_usd(msg.model, usage)

        # The model can hallucinate an index or repeat one; keep only valid,
        # unseen positions so a bad reply degrades instead of raising.
        seen: set[int] = set()
        clean = [
            i
            for i in indices
            if isinstance(i, int) and 0 <= i < len(chunks) and not (i in seen or seen.add(i))
        ]
        # An empty reply is a legitimate answer, not a failure: asked about
        # something the corpus does not cover, the model correctly finds
        # nothing relevant. Fall through to the backfill below, which keeps
        # the search order — the generator still has to see *something* to
        # refuse against.

        # Backfill from the original order if the model returned too few.
        for i in range(len(chunks)):
            if len(clean) >= top_k:
                break
            if i not in seen:
                clean.append(i)
        return clean[:top_k], float(cost) if cost else 0.0

    async def aclose(self) -> None:
        await self._client.close()
