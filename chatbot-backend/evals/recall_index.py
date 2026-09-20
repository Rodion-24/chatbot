"""Measure what an approximate index costs in recall.

pgvector does exact search by default: every row is scanned, so the result
is the true nearest neighbours. An HNSW index walks a graph instead — much
faster, but it can miss. This captures the exact answers first, builds the
index, re-runs the same queries and reports the overlap.

    uv run python -m evals.recall_index            # measure and drop the index
    uv run python -m evals.recall_index --keep     # leave the index in place
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path

import asyncpg

from src.config import get_settings
from src.db import pool as db_pool
from src.providers.factory import build_embedding_provider

logger = logging.getLogger(__name__)

DATASET = Path(__file__).resolve().parent / "dataset" / "golden.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

INDEX_NAME = "chunks_embedding_hnsw_idx"
#: Defaults from the pgvector docs: m connections per layer, and the size of
#: the candidate list used while building the graph.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

TOP_K = 10

NEAREST_SQL = """
SELECT id
FROM chunks
ORDER BY embedding <=> $1
LIMIT $2
"""


async def _nearest(conn: asyncpg.Connection, vector, k: int) -> tuple[list[int], float]:
    """Top-k chunk ids for one query vector, with the wall-clock time."""
    started = time.perf_counter()
    rows = await conn.fetch(NEAREST_SQL, vector, k)
    return [r["id"] for r in rows], time.perf_counter() - started


async def _run_all(conn, vectors: list, k: int) -> tuple[list[list[int]], list[float]]:
    results, timings = [], []
    for vec in vectors:
        ids, elapsed = await _nearest(conn, vec, k)
        results.append(ids)
        timings.append(elapsed)
    return results, timings


def _recall(exact: list[list[int]], approx: list[list[int]]) -> list[float]:
    """Per-query recall: the share of exact neighbours the index also found."""
    out = []
    for e, a in zip(exact, approx, strict=True):
        if not e:
            continue
        out.append(len(set(e) & set(a)) / len(e))
    return out


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * q))] * 1000, 2)


async def main(keep_index: bool, k: int) -> None:
    settings = get_settings()
    questions = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embedder = build_embedding_provider(settings)
    await db_pool.init_pool(settings)

    try:
        # Embed once and reuse, so the comparison isn't measuring OpenAI.
        vectors, usage = await embedder.embed([q["question"] for q in questions])
        print(f"embedded {len(vectors)} questions ({usage.input_tokens} tokens)")

        async with db_pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT count(*) FROM pg_indexes WHERE indexname = $1", INDEX_NAME
            )
            if existing:
                await conn.execute(f"DROP INDEX {INDEX_NAME}")
                print(f"dropped pre-existing {INDEX_NAME}")

            print("\nexact scan (no index) — this is the ground truth")
            exact, exact_times = await _run_all(conn, vectors, k)
            print(f"  p50 {_p(exact_times, 0.5)}ms   p95 {_p(exact_times, 0.95)}ms")

            print(f"\nbuilding HNSW (m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})")
            build_started = time.perf_counter()
            await conn.execute(
                f"CREATE INDEX {INDEX_NAME} ON chunks "
                f"USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
            )
            build_s = time.perf_counter() - build_started
            size = await conn.fetchval(
                "SELECT pg_size_pretty(pg_relation_size($1))", INDEX_NAME
            )
            print(f"  built in {build_s:.2f}s, size {size}")

            # Without this the planner picks a seq scan on a table this
            # small, and the "approximate" run is really the exact one again.
            await conn.execute("SET enable_seqscan = off")
            plan = await conn.fetchval(
                "EXPLAIN (FORMAT JSON) SELECT id FROM chunks "
                "ORDER BY embedding <=> $1 LIMIT $2",
                vectors[0],
                k,
            )
            uses_index = "Index Scan" in json.dumps(plan)
            print(f"  planner uses the index: {uses_index}")
            if not uses_index:
                print("  WARNING: still a seq scan — recall below is meaningless")

            rows = []
            for ef_search in (1, 2, 4, 10, 40, 100):
                await conn.execute(f"SET hnsw.ef_search = {ef_search}")
                approx, approx_times = await _run_all(conn, vectors, k)
                recalls = _recall(exact, approx)
                rows.append(
                    {
                        "ef_search": ef_search,
                        "recall": round(statistics.mean(recalls), 4),
                        "perfect": sum(1 for r in recalls if r == 1.0),
                        "p50_ms": _p(approx_times, 0.5),
                        "p95_ms": _p(approx_times, 0.95),
                    }
                )

            header = (
                f"\n{'ef_search':>10} {'recall@' + str(k):>10} "
                f"{'exact hits':>12} {'p50 ms':>9} {'p95 ms':>9}"
            )
            print(header)
            for r in rows:
                print(
                    f"{r['ef_search']:>10} {r['recall']:>10.1%} "
                    f"{r['perfect']:>7}/{len(questions):<4} {r['p50_ms']:>9} {r['p95_ms']:>9}"
                )

            summary = {
                "corpus_chunks": await conn.fetchval("SELECT count(*) FROM chunks"),
                "queries": len(questions),
                "top_k": k,
                "index": {
                    "m": HNSW_M,
                    "ef_construction": HNSW_EF_CONSTRUCTION,
                    "build_s": round(build_s, 2),
                    "size": size,
                },
                "exact": {"p50_ms": _p(exact_times, 0.5), "p95_ms": _p(exact_times, 0.95)},
                "planner_used_index": uses_index,
                "hnsw": rows,
            }

            await conn.execute("SET enable_seqscan = on")

            if not keep_index:
                await conn.execute(f"DROP INDEX {INDEX_NAME}")
                print(f"\ndropped {INDEX_NAME} — exact search restored")
            else:
                print(f"\nkept {INDEX_NAME}")
            summary["index_kept"] = keep_index

        path = RESULTS_DIR / "recall-hnsw.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved → {path.relative_to(Path.cwd())}")
    finally:
        await embedder.aclose()
        await db_pool.close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure HNSW recall vs exact search")
    parser.add_argument("--keep", action="store_true", help="leave the index in place")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    logging.basicConfig(level="WARNING")
    asyncio.run(main(args.keep, args.top_k))
