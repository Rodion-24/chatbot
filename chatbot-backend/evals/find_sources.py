"""Propose `expected_source_ids` by locating each answer's facts in the corpus.

Filling 60 entries by hand is slow and error-prone. This searches the chunk
table for the literal facts an answer must contain (`must_include`, plus
distinctive terms from `expected_answer`) and proposes the chunks that hold
them. Output is a review list, not a commit: every proposal names the chunk
and the term that matched so it can be checked against the README.

    uv run python -m evals.find_sources          # print proposals
    uv run python -m evals.find_sources --write  # write them into golden.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

from src.config import get_settings
from src.db import pool as db_pool

logger = logging.getLogger(__name__)

SOURCE = Path(__file__).resolve().parent / "dataset" / "golden.json"

#: Words too common in this corpus to identify a chunk on their own.
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "use",
    "used",
    "using",
    "can",
    "you",
    "are",
    "not",
    "but",
    "its",
    "it's",
    "from",
    "your",
    "index",
    "indexes",
    "vector",
    "vectors",
    "pgvector",
    "postgres",
    "documentation",
    "docs",
    "does",
    "default",
    "type",
    "types",
    "data",
}


def _terms(question: dict) -> list[str]:
    """Distinctive literals worth searching for, longest first."""
    terms: list[str] = list(question.get("must_include", []))
    # Identifiers, quoted names and numbers out of the expected answer.
    expected = question.get("expected_answer", "")
    terms += re.findall(r"`([^`]+)`", expected)
    terms += re.findall(r"\b([a-z_]+_[a-z_]+)\b", expected)
    terms += re.findall(r"\b(\d[\d,]{2,})\b", expected)

    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        t = t.strip()
        if len(t) < 3 or t.lower() in STOPWORDS or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return sorted(out, key=len, reverse=True)


async def propose(conn, *, doc_title: str = "pgvector") -> dict[int, list[dict]]:
    """Propose source chunks, searching only the main documentation.

    The changelog lives in the same table and repeats the same identifiers,
    so unrestricted matching drifts into release notes.
    """
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    proposals: dict[int, list[dict]] = {}
    doc_id = await conn.fetchval("SELECT id FROM documents WHERE title = $1", doc_title)
    if doc_id is None:
        raise ValueError(f"document {doc_title!r} not found — run the ingest first")

    for q in data["questions"]:
        # An impossible question has no source by definition.
        if q["category"] == "impossible":
            proposals[q["id"]] = []
            continue

        hits: dict[int, dict] = {}
        for term in _terms(q):
            rows = await conn.fetch(
                """
                SELECT id, section_title
                FROM chunks
                WHERE doc_id = $2 AND content LIKE '%' || $1 || '%'
                ORDER BY id
                LIMIT 5
                """,
                term,
                doc_id,
            )
            for row in rows:
                hit = hits.setdefault(
                    row["id"],
                    {"id": row["id"], "section": row["section_title"], "terms": []},
                )
                hit["terms"].append(term)

        # Rank by how many distinct terms matched, then by how much the
        # chunk's heading overlaps the question — a tie on terms is usually
        # broken correctly by the section the question is really about.
        qwords = {w.lower() for w in re.findall(r"[a-zA-Z_]{4,}", q["question"])}

        def _key(h: dict, words: set[str] = qwords) -> tuple[int, int]:
            heading = (h["section"] or "").lower()
            overlap = sum(1 for w in words if w in heading)
            return (len(h["terms"]), overlap)

        ranked = sorted(hits.values(), key=_key, reverse=True)
        proposals[q["id"]] = ranked[:2]

    return proposals


def write_back(proposals: dict[int, list[dict]]) -> int:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    raw = SOURCE.read_text(encoding="utf-8")
    # Strip anything written by a previous run so this is idempotent.
    raw = re.sub(r'\n\s*"expected_source_ids": \[[^\]]*\],', "", raw)
    written = 0

    for q in data["questions"]:
        ids = [h["id"] for h in proposals.get(q["id"], [])]
        if not ids:
            continue
        # The field is absent from golden.json (the converter supplies it),
        # so insert it after must_not_include rather than rewrite a value.
        pattern = re.compile(
            rf'(\{{ "id": {q["id"]},.*?"must_not_include": \[[^\]]*\],)', re.S
        )
        raw, n = pattern.subn(
            rf'\g<1>\n      "expected_source_ids": {json.dumps(ids)},', raw, count=1
        )
        written += n

    SOURCE.write_text(raw, encoding="utf-8")
    return written


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    await db_pool.init_pool(settings)
    try:
        async with db_pool.acquire() as conn:
            proposals = await propose(conn)
    finally:
        await db_pool.close_pool()

    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in data["questions"]}
    empty = []
    for qid, hits in sorted(proposals.items()):
        q = by_id[qid]
        if q["category"] == "impossible":
            continue
        if not hits:
            empty.append(qid)
            continue
        ids = ", ".join(str(h["id"]) for h in hits)
        print(f"#{qid:<3} [{q['category']:9}] → {ids}")
        for h in hits:
            print(f"      {h['id']:<4} {str(h['section'])[:46]:48} {h['terms'][:3]}")

    if empty:
        print(f"\nбез предложений (нужны вручную): {empty}")

    if args.write:
        n = write_back(proposals)
        print(f"\nзаписано в golden.json: {n} вопросов")


if __name__ == "__main__":
    logging.basicConfig(level="WARNING")
    asyncio.run(_main())
