"""Resolve model pricing and compute request cost.

Pricing lives in the `model_pricing` table (time-versioned). Cost is computed
with Decimal for money-grade precision.
"""
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ModelPricing

_MILLION = Decimal(1_000_000)
_CENT_PRECISION = Decimal("0.000001")  # 6 dp, matches Numeric(12, 6)


def resolve_pricing(
    db: Session,
    provider: str,
    model: str,
    fallback_model: str | None = None,
    when: datetime | None = None,
) -> ModelPricing | None:
    """Return the price row in effect at `when` for `model`.

    Tries the resolved model id first, then the requested alias (OpenAI often
    resolves an alias like `gpt-4o-mini` to a dated snapshot, which won't be in
    the rate book — the alias fallback covers that).
    """
    when = when or datetime.now(timezone.utc)
    candidates = [model]
    if fallback_model and fallback_model != model:
        candidates.append(fallback_model)

    for candidate in candidates:
        stmt = (
            select(ModelPricing)
            .where(
                ModelPricing.provider == provider,
                ModelPricing.model == candidate,
                ModelPricing.effective_from <= when,
                (ModelPricing.effective_to.is_(None)) | (ModelPricing.effective_to > when),
            )
            .order_by(ModelPricing.effective_from.desc())
            .limit(1)
        )
        row = db.execute(stmt).scalar_one_or_none()
        if row is not None:
            return row
    return None


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    pricing: ModelPricing | None,
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    """Return (input_cost, output_cost, total_cost) or (None, None, None)."""
    if pricing is None:
        return None, None, None

    input_rate = Decimal(pricing.input_per_1m)
    output_rate = Decimal(pricing.output_per_1m)
    cached_rate = (
        Decimal(pricing.cached_input_per_1m)
        if pricing.cached_input_per_1m is not None
        else input_rate
    )

    non_cached = max(input_tokens - cached_tokens, 0)
    input_cost = (
        (Decimal(non_cached) / _MILLION) * input_rate
        + (Decimal(cached_tokens) / _MILLION) * cached_rate
    )
    output_cost = (Decimal(output_tokens) / _MILLION) * output_rate
    total_cost = input_cost + output_cost

    return (
        input_cost.quantize(_CENT_PRECISION),
        output_cost.quantize(_CENT_PRECISION),
        total_cost.quantize(_CENT_PRECISION),
    )
