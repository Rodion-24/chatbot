"""Per-model pricing and cost computation.

Rates are USD per 1M tokens. Keep this table in sync with the providers'
public pricing pages — it is the only place cost is defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.providers.base import Usage

_PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal = Decimal(0)
    cache_write_per_mtok: Decimal = Decimal(0)


def _d(v: str) -> Decimal:
    return Decimal(v)


#: model id -> pricing. Anthropic rates per the Claude API pricing table;
#: OpenAI rates per platform.openai.com/pricing.
PRICING: dict[str, ModelPricing] = {
    # --- Anthropic ---
    "claude-opus-5": ModelPricing(_d("5.00"), _d("25.00"), _d("0.50"), _d("6.25")),
    "claude-sonnet-5": ModelPricing(_d("2.00"), _d("10.00"), _d("0.20"), _d("2.50")),
    "claude-haiku-4-5": ModelPricing(_d("1.00"), _d("5.00"), _d("0.10"), _d("1.25")),
    # --- OpenAI ---
    "gpt-4o": ModelPricing(_d("2.50"), _d("10.00"), _d("1.25")),
    "gpt-4o-mini": ModelPricing(_d("0.15"), _d("0.60"), _d("0.075")),
    # --- OpenAI embeddings (output rate is 0) ---
    "text-embedding-3-small": ModelPricing(_d("0.02"), _d("0")),
    "text-embedding-3-large": ModelPricing(_d("0.13"), _d("0")),
}


def _lookup(model: str) -> ModelPricing | None:
    """Find pricing for a model id, tolerating a dated suffix.

    Responses carry dated ids (`claude-haiku-4-5-20251001`) while the table
    is keyed by the base id. Without this, every dated id misses the table
    and its cost is silently reported as unknown.
    """
    pricing = PRICING.get(model)
    if pricing is not None:
        return pricing
    # Longest prefix wins, so `claude-opus-4-5` can't shadow `claude-opus-4`.
    matches = [key for key in PRICING if model.startswith(key)]
    if not matches:
        return None
    return PRICING[max(matches, key=len)]


def estimate_cost_usd(model: str, usage: Usage) -> Decimal | None:
    """Cost of one call, or None when the model is not in the table.

    Unknown models return None rather than 0 so a missing entry is visible
    in traces instead of silently reporting free requests.
    """
    pricing = _lookup(model)
    if pricing is None:
        return None

    return (
        pricing.input_per_mtok * usage.input_tokens
        + pricing.output_per_mtok * usage.output_tokens
        + pricing.cache_read_per_mtok * usage.cache_read_input_tokens
        + pricing.cache_write_per_mtok * usage.cache_creation_input_tokens
    ) / _PER_MILLION
