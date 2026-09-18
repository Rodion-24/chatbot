"""Eval runner — not implemented.

Intended shape: read evals/dataset/golden.jsonl, run each question through
the pipeline, score it, and report aggregate metrics.

    uv run python -m evals.runner
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "golden.jsonl"


@dataclass(frozen=True, slots=True)
class GoldenExample:
    question: str
    expected_answer: str | None = None
    expected_doc_ids: list[int] | None = None


def load_dataset(path: Path = DATASET_PATH) -> list[GoldenExample]:
    if not path.exists():
        return []
    examples: list[GoldenExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        examples.append(
            GoldenExample(
                question=raw["question"],
                expected_answer=raw.get("expected_answer"),
                expected_doc_ids=raw.get("expected_doc_ids"),
            )
        )
    return examples


async def run() -> None:
    raise NotImplementedError("eval runner is not implemented yet")


if __name__ == "__main__":
    raise SystemExit("eval runner is not implemented yet")
