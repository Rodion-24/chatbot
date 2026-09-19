"""LLM judge for verdicts a substring check cannot reach.

A refusal names the concept in order to deny it ("there is no built-in
reranker"), so `must_not_include` fires on correct answers. Meaning has to
be read, not matched — which is what this does.

Deliberately narrow: the judge grades only `impossible` and `ambiguous`.
Everywhere else substring checks are exact, deterministic and free, and a
non-deterministic judge would only add noise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

from anthropic import AsyncAnthropic

from src.config import Settings
from src.generation.prompts import load_prompt
from src.providers.retry import with_retry

logger = logging.getLogger(__name__)

JUDGE_PROMPT = "judge_v1"

#: A cheaper model than the assistant's, and a different one on purpose: a
#: judge sharing the model under test tends to agree with itself.
JUDGE_MODEL = "claude-haiku-4-5"

Verdict = Literal["correct", "incorrect", "refused", "error"]

#: Forces the reply into this exact shape, so nothing has to be parsed out
#: of free text.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["correct", "incorrect", "refused"],
        },
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Judgement:
    verdict: Verdict
    reason: str
    cost_usd: float = 0.0


class LLMJudge:
    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the judge")
        self._settings = settings
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=2)

    async def judge(self, *, question: str, answer: str, expected: str) -> Judgement:
        """Grade one answer. A judge failure returns 'error', never raises."""
        user = (
            f"Question:\n{question}\n\n"
            f"Assistant's answer:\n{answer}\n\n"
            f"What the documentation says:\n{expected}"
        )

        async def _call():
            return await self._client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=512,
                system=load_prompt(JUDGE_PROMPT),
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            )

        try:
            msg = await with_retry(_call, settings=self._settings, op="judge")
        except Exception as exc:  # noqa: BLE001 - a judge failure is data, not a crash
            logger.warning("judge call failed: %s", exc)
            return Judgement(verdict="error", reason=str(exc))

        text = "".join(b.text for b in msg.content if b.type == "text")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("judge returned non-JSON: %.200s", text)
            return Judgement(verdict="error", reason="unparseable judge reply")

        from src.providers.base import Usage
        from src.providers.pricing import estimate_cost_usd

        usage = Usage(
            input_tokens=msg.usage.input_tokens or 0,
            output_tokens=msg.usage.output_tokens or 0,
        )
        cost = estimate_cost_usd(msg.model, usage)

        return Judgement(
            verdict=parsed.get("verdict", "error"),
            reason=parsed.get("reason", ""),
            cost_usd=float(cost) if cost is not None else 0.0,
        )

    async def aclose(self) -> None:
        await self._client.close()
