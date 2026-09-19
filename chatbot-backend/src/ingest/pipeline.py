"""The ingestion pipeline: text in, rows in `documents` and `chunks` out."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import asyncpg

from src.ingest.base import IngestResult, SourceDocument
from src.ingest.chunker import TextChunk, chunk_text
from src.ingest.structural import chunk_by_structure
from src.observability import observe, update_span
from src.providers.base import EmbeddingProvider, Usage

logger = logging.getLogger(__name__)

#: Texts sent to the embedding API per request. Large enough to keep the
#: number of round trips low, small enough that one failure is cheap.
EMBED_BATCH_SIZE = 50


class MarkdownIngestor:
    """Chunks a document, embeds the chunks and stores both."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        conn: asyncpg.Connection,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "structural",
    ) -> None:
        self._embedder = embedder
        self._conn = conn
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        # "structural" cuts on markdown headings and keeps code blocks whole;
        # "naive" is the stage-2 fixed window, kept so the two can be compared.
        self._strategy = strategy

    @observe(name="ingest_document")
    async def ingest(self, document: SourceDocument) -> IngestResult:
        """Store `document` and its chunks. Runs as a single transaction."""
        if self._strategy == "structural":
            pairs = chunk_by_structure(document.content)
            chunks = [c for c, _ in pairs]
            headings = [h for _, h in pairs]
        else:
            chunks = chunk_text(
                document.content, size=self._chunk_size, overlap=self._chunk_overlap
            )
            headings = [""] * len(chunks)
        if not chunks:
            raise ValueError(f"document {document.title!r} produced no chunks")

        vectors, usage = await self._embed_all(chunks)

        # Nothing is written until every embedding succeeded: a half-indexed
        # document would silently degrade search results.
        async with self._conn.transaction():
            doc_id = await self._insert_document(document)
            await self._insert_chunks(doc_id, chunks, vectors, headings, document.meta)

        update_span(
            output={"document_id": doc_id, "chunks_written": len(chunks)},
            metadata={
                "title": document.title,
                "chunks": len(chunks),
                "embed_tokens": usage.input_tokens,
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
                "strategy": self._strategy,
            },
        )
        logger.info(
            "ingested %r: %d chunks, %d embedding tokens",
            document.title,
            len(chunks),
            usage.input_tokens,
        )
        return IngestResult(document_id=doc_id, chunks_written=len(chunks))

    async def _embed_all(self, chunks: Sequence[TextChunk]) -> tuple[list[list[float]], Usage]:
        """Embed every chunk, one batch of EMBED_BATCH_SIZE at a time."""
        vectors: list[list[float]] = []
        total = Usage()

        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            batch_vectors, batch_usage = await self._embedder.embed([c.content for c in batch])
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"embedder returned {len(batch_vectors)} vectors for {len(batch)} texts"
                )
            vectors.extend(batch_vectors)
            total = Usage(input_tokens=total.input_tokens + batch_usage.input_tokens)
            logger.debug("embedded %d/%d chunks", len(vectors), len(chunks))

        return vectors, total

    async def _insert_document(self, document: SourceDocument) -> int:
        return await self._conn.fetchval(
            """
            INSERT INTO documents (title, source_uri, version)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            document.title,
            document.source_uri,
            document.version,
        )

    async def _insert_chunks(
        self,
        doc_id: int,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
        headings: Sequence[str],
        meta: dict,
    ) -> None:
        """Bulk-insert chunks. `embedding` is written here and never selected."""
        await self._conn.executemany(
            """
            INSERT INTO chunks
                (doc_id, chunk_index, content, section_title, meta, embedding, embed_model)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            [
                (
                    doc_id,
                    chunk.index,
                    chunk.content,
                    heading or None,
                    meta,
                    vector,
                    self._embedder.model,
                )
                for chunk, vector, heading in zip(chunks, vectors, headings, strict=True)
            ],
        )
