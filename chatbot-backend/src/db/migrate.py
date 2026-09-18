"""Minimal forward-only SQL migration runner.

Applies every migrations/*.sql not yet recorded in schema_migrations, in
filename order, each inside a transaction.

    uv run python -m src.db.migrate
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import asyncpg

from src.config import get_settings

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version     text PRIMARY KEY,
  applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


async def _applied_versions(conn: asyncpg.Connection) -> set[str]:
    rows = await conn.fetch("SELECT version FROM schema_migrations")
    return {r["version"] for r in rows}


def _discover() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


async def run_migrations(dsn: str) -> list[str]:
    """Apply pending migrations. Returns the versions applied in this run."""
    conn = await asyncpg.connect(dsn)
    applied: list[str] = []
    try:
        await conn.execute(_CREATE_TABLE)
        done = await _applied_versions(conn)

        for path in _discover():
            version = path.stem
            if version in done:
                continue
            sql = path.read_text(encoding="utf-8")
            logger.info("applying migration %s", version)
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
            applied.append(version)
    finally:
        await conn.close()

    if not applied:
        logger.info("no pending migrations")
    return applied


async def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    applied = await run_migrations(settings.asyncpg_dsn)
    print(f"applied: {applied or 'nothing'}")


if __name__ == "__main__":
    asyncio.run(_main())
