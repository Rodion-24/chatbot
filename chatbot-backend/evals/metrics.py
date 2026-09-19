"""Scoring for one eval example.

Deliberately substring-based, per the project rules: cheap, deterministic
and reproducible. An LLM judge is reserved for faithfulness on multihop
questions, where a string check cannot work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Phrases that mark an honest "not in the documentation" answer. Matched
#: case-insensitively against the answer text.
#: Categories the LLM judge grades. Elsewhere substring checks are exact,
#: deterministic and free, so a non-deterministic judge would add only noise.
JUDGED_CATEGORIES = ("impossible", "ambiguous")

REFUSAL_MARKERS = (
    "not covered",
    "does not cover",
    "do not cover",
    "don't cover",
    "doesn't cover",
    "not provide",
    "does not provide",
    "no built-in",
    "not available in",
    "not documented",
    "not mentioned",
    "no mention",
    "doesn't mention",
    "does not mention",
    "not in the documentation",
    "not described",
    "no information",
    "cannot find",
    "could not find",
    "not specified",
    "не описано",
    "не упоминается",
    "нет информации",
    "в документации нет",
)


@dataclass(slots=True)
class Score:
    """Per-question outcome. `passed` is the headline verdict."""

    id: int
    category: str
    passed: bool
    must_include_hit: float = 0.0
    forbidden_hit: list[str] = field(default_factory=list)
    refused: bool = False
    retrieved_ids: list[int] = field(default_factory=list)
    context_recall: float | None = None
    latency_s: float = 0.0
    cost_usd: float = 0.0
    answer: str = ""
    error: str | None = None
    #: Set only for categories the LLM judge grades; None elsewhere.
    judge_verdict: str | None = None
    judge_reason: str = ""


def looks_like_refusal(answer: str) -> bool:
    """True when the answer admits the documentation does not cover it."""
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def must_include_ratio(answer: str, required: list[str]) -> float:
    """Fraction of required substrings present. 1.0 when nothing is required."""
    if not required:
        return 1.0
    low = answer.lower()
    hits = sum(1 for term in required if term.lower() in low)
    return hits / len(required)


def forbidden_present(answer: str, forbidden: list[str]) -> list[str]:
    """Which banned substrings leaked into the answer."""
    low = answer.lower()
    return [term for term in forbidden if term.lower() in low]


def context_recall(retrieved: list[int], expected: list[int]) -> float | None:
    """Fraction of expected chunk ids that retrieval surfaced.

    None when the question has no expected ids recorded yet — those are
    filled in after the first run, once chunk ids are stable.
    """
    if not expected:
        return None
    found = sum(1 for cid in expected if cid in retrieved)
    return found / len(expected)


def score_answer(
    *,
    question: dict,
    answer: str,
    retrieved_ids: list[int],
    latency_s: float,
    cost_usd: float,
) -> Score:
    """Judge one answer against its golden entry."""
    category = question["category"]
    required = question.get("must_include", [])
    forbidden = question.get("must_not_include", [])

    ratio = must_include_ratio(answer, required)
    leaked = forbidden_present(answer, forbidden)
    refused = looks_like_refusal(answer)

    if category == "impossible":
        # Success here means refusing: no banned phrasing, and an explicit
        # admission that the docs don't cover it.
        passed = not leaked and refused
    elif category == "ambiguous":
        # No single right answer here. The bar is: nothing invented, and
        # either the required terms appear or it engaged with the question.
        # Length is a poor proxy — a correct one-line answer is still correct.
        passed = not leaked and ratio == 1.0 and bool(answer.strip())
    else:
        # factual / multihop / filtered: every required term must appear.
        passed = ratio == 1.0 and not leaked

    return Score(
        id=question["id"],
        category=category,
        passed=passed,
        must_include_hit=ratio,
        forbidden_hit=leaked,
        refused=refused,
        retrieved_ids=retrieved_ids,
        context_recall=context_recall(retrieved_ids, question.get("expected_source_ids", [])),
        latency_s=latency_s,
        cost_usd=cost_usd,
        answer=answer,
    )
