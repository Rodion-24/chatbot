"""Run the golden set against the assistant and record the metrics.

Two modes, because they answer different questions:

    --mode retrieval   did search surface the right chunks? (cents, seconds)
    --mode full        was the answer itself right? (dollars, minutes)

    uv run python -m evals.runner
    uv run python -m evals.runner --mode retrieval
    uv run python -m evals.runner --label baseline --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from evals.judge import JUDGE_MODEL, LLMJudge
from evals.metrics import JUDGED_CATEGORIES, Score, context_recall, score_answer
from src.config import Settings, get_settings
from src.db import pool as db_pool
from src.generation.context import build_user_message
from src.generation.prompts import SYSTEM_PROMPT, load_prompt
from src.providers.base import Message, StreamResult
from src.providers.factory import build_embedding_provider, build_llm_provider
from src.providers.pricing import estimate_cost_usd
from src.retrieval.base import Retriever
from src.retrieval.factory import build_retriever

logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "dataset" / "golden.jsonl"
SOURCE_PATH = EVALS_DIR / "dataset" / "golden.json"
RESULTS_DIR = EVALS_DIR / "results"


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """Read golden.jsonl, warning if it is older than its source."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — generate it with: uv run python -m evals.build_dataset"
        )
    if SOURCE_PATH.exists() and SOURCE_PATH.stat().st_mtime > path.stat().st_mtime:
        logger.warning(
            "%s is newer than %s — run `uv run python -m evals.build_dataset` "
            "or you are scoring against a stale dataset",
            SOURCE_PATH.name,
            path.name,
        )
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


async def _answer(llm, settings: Settings, question: str, context) -> tuple[str, float]:
    """Generate one answer, returning the text and its cost."""
    messages = [Message(role="user", content=build_user_message(question, context))]
    result = StreamResult()
    parts: list[str] = []
    async for delta in llm.stream_chat(
        messages=messages,
        system=load_prompt(),
        max_tokens=settings.max_tokens,
        result=result,
    ):
        parts.append(delta)
    cost = estimate_cost_usd(result.model, result.usage)
    return "".join(parts), float(cost) if cost is not None else 0.0


async def run_one(
    entry: dict,
    *,
    retriever: Retriever,
    llm,
    settings: Settings,
    mode: str,
    judge: LLMJudge | None = None,
) -> Score:
    started = time.perf_counter()
    try:
        chunks = await retriever.search(entry["question"], top_k=settings.retrieval_top_k)
        retrieved_ids = [c.chunk_id for c in chunks]

        if mode == "retrieval":
            # No generation: score retrieval only, leaving answer-based
            # fields at their defaults.
            recall = context_recall(retrieved_ids, entry.get("expected_source_ids", []))
            return Score(
                id=entry["id"],
                category=entry["category"],
                # With expected chunks recorded, "did it find them" is the
                # verdict; without, fall back to "did it find anything".
                passed=recall == 1.0 if recall is not None else bool(retrieved_ids),
                retrieved_ids=retrieved_ids,
                context_recall=recall,
                latency_s=time.perf_counter() - started,
            )

        answer, cost = await _answer(llm, settings, entry["question"], chunks)
        score = score_answer(
            question=entry,
            answer=answer,
            retrieved_ids=retrieved_ids,
            latency_s=time.perf_counter() - started,
            cost_usd=cost,
        )

        if judge and entry["category"] in JUDGED_CATEGORIES:
            verdict = await judge.judge(
                question=entry["question"],
                answer=answer,
                expected=entry["expected_answer"],
            )
            score.judge_verdict = verdict.verdict
            score.judge_reason = verdict.reason
            score.cost_usd += verdict.cost_usd
            # The judge overrules the substring check for these categories:
            # a refusal has to name the concept it is denying, which the
            # substring rule cannot tell apart from inventing it.
            if verdict.verdict != "error":
                if entry["category"] == "impossible":
                    # Both verdicts mean success here: the question has no
                    # documented answer, so declining *is* the correct
                    # answer. Only an invented fact is a failure.
                    score.passed = verdict.verdict in ("refused", "correct")
                else:
                    score.passed = verdict.verdict == "correct"

        return score
    except Exception as exc:  # noqa: BLE001 - recorded, so one bad question
        logger.exception("question %s failed", entry["id"])  # doesn't kill the run
        return Score(
            id=entry["id"],
            category=entry["category"],
            passed=False,
            latency_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def summarise(scores: list[Score], mode: str) -> dict:
    """Aggregate per-category and overall numbers."""
    by_cat: dict[str, list[Score]] = defaultdict(list)
    for s in scores:
        by_cat[s.category].append(s)

    latencies = sorted(s.latency_s for s in scores)
    recalls = [s.context_recall for s in scores if s.context_recall is not None]

    def pct(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, int(len(vals) * q))
        return round(vals[idx], 3)

    summary = {
        "mode": mode,
        "total": len(scores),
        "passed": sum(1 for s in scores if s.passed),
        "pass_rate": round(sum(1 for s in scores if s.passed) / len(scores), 3) if scores else 0.0,
        "errors": sum(1 for s in scores if s.error),
        "cost_usd": round(sum(s.cost_usd for s in scores), 4),
        "latency_p50_s": pct(latencies, 0.5),
        "latency_p95_s": pct(latencies, 0.95),
        "context_recall": round(statistics.mean(recalls), 3) if recalls else None,
        "by_category": {},
    }

    for cat, items in sorted(by_cat.items()):
        entry = {
            "total": len(items),
            "passed": sum(1 for s in items if s.passed),
            "pass_rate": round(sum(1 for s in items if s.passed) / len(items), 3),
        }
        if mode == "full":
            entry["must_include_hit"] = round(statistics.mean(s.must_include_hit for s in items), 3)
            if cat == "impossible":
                # The headline metric: how often it correctly declined.
                entry["refusal_accuracy"] = round(
                    sum(1 for s in items if s.refused and not s.forbidden_hit) / len(items), 3
                )
        summary["by_category"][cat] = entry

    return summary


def _print(summary: dict, scores: list[Score]) -> None:
    print(f"\n{'=' * 62}")
    print(
        f"  mode={summary['mode']}  passed {summary['passed']}/{summary['total']}"
        f"  ({summary['pass_rate']:.0%})"
    )
    print(f"{'=' * 62}")
    for cat, s in summary["by_category"].items():
        extra = ""
        if "refusal_accuracy" in s:
            extra = f"  refusal_acc={s['refusal_accuracy']:.0%}"
        elif "must_include_hit" in s:
            extra = f"  must_include={s['must_include_hit']:.0%}"
        print(f"  {cat:12} {s['passed']:2}/{s['total']:<3} {s['pass_rate']:6.0%}{extra}")
    print(f"\n  latency p50={summary['latency_p50_s']}s  p95={summary['latency_p95_s']}s")
    print(f"  cost ${summary['cost_usd']}")
    if summary["context_recall"] is not None:
        print(f"  context recall {summary['context_recall']:.0%}")
    if summary["errors"]:
        print(f"  ERRORS: {summary['errors']}")

    failed = [s for s in scores if not s.passed]
    if failed:
        print(f"\n  failed ({len(failed)}):")
        for s in failed[:12]:
            why = s.error or (
                f"leaked {s.forbidden_hit}"
                if s.forbidden_hit
                else f"must_include {s.must_include_hit:.0%}"
                if s.must_include_hit < 1
                else "no refusal"
            )
            print(f"    #{s.id:<3} {s.category:11} {why}")
        if len(failed) > 12:
            print(f"    … and {len(failed) - 12} more")
    print()


def _write_results(out_dir: Path, label: str, summary: dict, scores: list[Score]) -> Path:
    """Write one numbered result file and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(out_dir.glob("*.json"))) + 1
    path = out_dir / f"{n:03d}-{label}.json"
    path.write_text(
        json.dumps(
            {"summary": summary, "scores": [asdict(s) for s in scores]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


async def run(
    *,
    mode: str,
    label: str,
    limit: int | None,
    use_judge: bool = True,
    out_dir: Path = RESULTS_DIR,
) -> dict:
    settings = get_settings()
    dataset = load_dataset()
    if limit:
        dataset = dataset[:limit]

    embedder = build_embedding_provider(settings)
    llm = build_llm_provider(settings) if mode == "full" else None
    judge = LLMJudge(settings) if mode == "full" and use_judge else None
    await db_pool.init_pool(settings)

    scores: list[Score] = []
    try:
        async with db_pool.acquire() as conn:
            retriever = build_retriever(settings, embedder, conn)
            for i, entry in enumerate(dataset, 1):
                score = await run_one(
                    entry,
                    retriever=retriever,
                    llm=llm,
                    settings=settings,
                    mode=mode,
                    judge=judge,
                )
                scores.append(score)
                mark = "ok " if score.passed else "FAIL"
                print(f"  [{i:2}/{len(dataset)}] {mark} #{entry['id']:<3} {entry['category']}")
    finally:
        await embedder.aclose()
        if llm:
            await llm.aclose()
        if judge:
            await judge.aclose()
        await db_pool.close_pool()

    summary = summarise(scores, mode)
    summary["label"] = label
    summary["model"] = settings.anthropic_model if mode == "full" else None
    summary["embedding_model"] = settings.openai_embedding_model
    summary["prompt_version"] = SYSTEM_PROMPT if mode == "full" else None
    summary["judge"] = JUDGE_MODEL if (mode == "full" and use_judge) else None
    summary["retriever"] = settings.retriever
    summary["top_k"] = settings.retrieval_top_k
    summary["timestamp"] = datetime.now(UTC).isoformat()

    # Disk I/O is blocking, so it runs off the event loop.
    path = await asyncio.to_thread(_write_results, out_dir, label, summary, scores)

    _print(summary, scores)
    print(f"  saved → {path.relative_to(Path.cwd())}\n")
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the golden eval set")
    p.add_argument("--mode", choices=["full", "retrieval"], default="full")
    p.add_argument("--label", default="baseline", help="name recorded with the results")
    p.add_argument("--limit", type=int, help="only run the first N questions")
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the LLM judge and score impossible/ambiguous by substring only",
    )
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()
    logging.basicConfig(level="WARNING", format="%(levelname)s %(name)s: %(message)s")
    await run(
        mode=args.mode,
        label=args.label,
        limit=args.limit,
        use_judge=not args.no_judge,
    )


if __name__ == "__main__":
    asyncio.run(_main())
