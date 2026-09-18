"""Data access.

Two rules are enforced structurally rather than by review:

- No ``SELECT *`` — every query names its columns.
- ``embedding`` never appears in a SELECT or RETURNING list. It is written on
  INSERT and used inside expressions (``embedding <=> $1``), never shipped
  back to Python. Keeping it out of the column constants below makes that
  the default rather than something to remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

#: Column lists live in one place so SELECT and RETURNING cannot drift apart.
DOCUMENT_COLS = "id, title, source_uri, version, created_at"

#: Note the absence of `embedding` — see the module docstring.
CHUNK_COLS = "id, doc_id, chunk_index, content, section_title, meta, embed_model"


@dataclass(frozen=True, slots=True)
class Document:
    id: int
    title: str
    source_uri: str | None
    version: str | None
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class Chunk:
    id: int
    doc_id: int
    chunk_index: int
    content: str
    section_title: str | None
    meta: dict[str, Any]
    embed_model: str


class DocumentRepository:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def create(self, *, title: str, source_uri: str | None, version: str | None) -> Document:
        row = await self._conn.fetchrow(
            f"""
            INSERT INTO documents (title, source_uri, version)
            VALUES ($1, $2, $3)
            RETURNING {DOCUMENT_COLS}
            """,
            title,
            source_uri,
            version,
        )
        return _to_document(row)

    async def get(self, doc_id: int) -> Document | None:
        row = await self._conn.fetchrow(
            f"SELECT {DOCUMENT_COLS} FROM documents WHERE id = $1",
            doc_id,
        )
        return _to_document(row) if row else None

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[Document]:
        rows = await self._conn.fetch(
            f"""
            SELECT {DOCUMENT_COLS}
            FROM documents
            ORDER BY id DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        return [_to_document(r) for r in rows]

    async def delete(self, doc_id: int) -> bool:
        result = await self._conn.execute("DELETE FROM documents WHERE id = $1", doc_id)
        return result.endswith(" 1")


class ChunkRepository:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def list_by_document(self, doc_id: int) -> list[Chunk]:
        rows = await self._conn.fetch(
            f"""
            SELECT {CHUNK_COLS}
            FROM chunks
            WHERE doc_id = $1
            ORDER BY chunk_index
            """,
            doc_id,
        )
        return [_to_chunk(r) for r in rows]

    async def count(self) -> int:
        return await self._conn.fetchval("SELECT count(*) FROM chunks")


def _to_document(row: asyncpg.Record) -> Document:
    return Document(
        id=row["id"],
        title=row["title"],
        source_uri=row["source_uri"],
        version=row["version"],
        created_at=row["created_at"],
    )


def _to_chunk(row: asyncpg.Record) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        section_title=row["section_title"],
        # The pool registers a jsonb codec, so this is already a dict.
        meta=row["meta"] or {},
        embed_model=row["embed_model"],
    )
