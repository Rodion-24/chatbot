"""Ingestion interface. Chunking and embedding are deliberately not implemented."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SourceDocument:
    title: str
    content: str
    source_uri: str | None = None
    version: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: int
    chunks_written: int


@runtime_checkable
class Ingestor(Protocol):
    async def ingest(self, document: SourceDocument) -> IngestResult:
        """Persist a document and its chunks."""
        ...


class NotImplementedIngestor(Ingestor):
    async def ingest(self, document: SourceDocument) -> IngestResult:
        raise NotImplementedError("ingestion pipeline is not implemented yet")
