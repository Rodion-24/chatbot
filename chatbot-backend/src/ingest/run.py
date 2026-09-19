"""Entry point for loading the corpus into Postgres.

uv run python -m src.ingest.run
uv run python -m src.ingest.run --skip-existing
uv run python -m src.ingest.run --chunk-size 1000 --chunk-overlap 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import asyncpg

from src.config import Settings, get_settings
from src.db import pool as db_pool
from src.ingest.base import SourceDocument
from src.ingest.loader import load_directory
from src.ingest.pipeline import MarkdownIngestor
from src.observability import flush as flush_observability
from src.providers.factory import build_embedding_provider

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


async def _existing_document_id(conn: asyncpg.Connection, source_uri: str) -> int | None:
    return await conn.fetchval("SELECT id FROM documents WHERE source_uri = $1", source_uri)


async def ingest_corpus(
    settings: Settings,
    *,
    directory: Path = CORPUS_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "structural",
    skip_existing: bool = False,
) -> None:
    documents = load_directory(directory)
    if not documents:
        logger.warning("no markdown files found in %s", directory)
        return

    embedder = build_embedding_provider(settings)
    await db_pool.init_pool(settings)

    try:
        async with db_pool.acquire() as conn:
            for document in documents:
                await _ingest_one(
                    conn,
                    embedder,
                    document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    strategy=strategy,
                    skip_existing=skip_existing,
                )
    finally:
        await embedder.aclose()
        await db_pool.close_pool()
        await flush_observability()


async def _ingest_one(
    conn: asyncpg.Connection,
    embedder,
    document: SourceDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
    skip_existing: bool,
) -> None:
    existing = await _existing_document_id(conn, document.source_uri)

    if existing is not None:
        if skip_existing:
            logger.info("skipping %r: already ingested (id=%d)", document.title, existing)
            return
        # Chunks go with it via ON DELETE CASCADE, so re-ingesting replaces
        # the document instead of leaving a second copy behind.
        await conn.execute("DELETE FROM documents WHERE id = $1", existing)
        logger.info("replacing %r (old id=%d)", document.title, existing)

    ingestor = MarkdownIngestor(embedder, conn, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result = await ingestor.ingest(document)
    logger.info(
        "stored %r as document %d with %d chunks",
        document.title,
        result.document_id,
        result.chunks_written,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the documentation corpus")
    parser.add_argument("--directory", type=Path, default=CORPUS_DIR)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument(
        "--strategy",
        choices=["structural", "naive"],
        default="structural",
        help="structural: split on headings; naive: fixed character window",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave already-ingested documents alone instead of replacing them",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )
    await ingest_corpus(
        settings,
        directory=args.directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        strategy=args.strategy,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    asyncio.run(_main())
