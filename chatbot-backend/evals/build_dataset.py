"""Convert the readable golden.json into the golden.jsonl the runner reads.

golden.json is the source of truth: pretty-printed, grouped, and carrying a
`_source` note per question so an expected answer can be checked against the
README quickly. The .jsonl is generated — never edit it by hand.

    uv run python -m evals.build_dataset
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
SOURCE = DATASET_DIR / "golden.json"
OUTPUT = DATASET_DIR / "golden.jsonl"

REQUIRED = ("id", "category", "question", "expected_answer")
CATEGORIES = ("factual", "multihop", "impossible", "filtered", "ambiguous")


def build() -> int:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    questions = data["questions"]

    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        raise ValueError(f"duplicate ids: {dupes}")

    lines = []
    for q in questions:
        missing = [k for k in REQUIRED if k not in q]
        if missing:
            raise ValueError(f"question {q.get('id')}: missing {missing}")
        if q["category"] not in CATEGORIES:
            raise ValueError(f"question {q['id']}: unknown category {q['category']!r}")

        lines.append(
            json.dumps(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "expected_answer": q["expected_answer"],
                    # Filled in after the first run, once chunk ids are stable.
                    "expected_source_ids": q.get("expected_source_ids", []),
                    "category": q["category"],
                    "must_include": q.get("must_include", []),
                    "must_not_include": q.get("must_not_include", []),
                },
                ensure_ascii=False,
            )
        )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


if __name__ == "__main__":
    n = build()
    counts = Counter(
        json.loads(line)["category"] for line in OUTPUT.read_text(encoding="utf-8").splitlines()
    )
    print(f"wrote {OUTPUT.relative_to(Path.cwd())}: {n} questions")
    for cat in CATEGORIES:
        print(f"  {cat:12} {counts[cat]}")
