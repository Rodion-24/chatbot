"""Search interface. Implementations land here later (vector / BM25 / hybrid)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: int
    doc_id: int
    content: str
    score: float
    section_title: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    async def search(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        """Return the top_k chunks most relevant to `query`."""
        ...


class NullRetriever(Retriever):
    """Placeholder used while /api/chat answers without retrieval."""

    async def search(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        return []
