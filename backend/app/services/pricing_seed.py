"""Seed initial model pricing on startup (idempotent).

This only inserts a current row (effective_to IS NULL) for a model if one does
not already exist — so it never clobbers a price you've edited live in the DB.
To change a price at runtime, close the current row (set `effective_to = now`)
and insert a new one; don't edit this file for that.

Rates are USD per 1M tokens.
"""
import logging
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ModelPricing

logger = logging.getLogger(__name__)

# (provider, model, input_per_1m, output_per_1m, cached_input_per_1m)
DEFAULT_PRICING: list[tuple[str, str, str, str, str | None]] = [
    ("openai", "gpt-4o-mini", "0.15", "0.60", "0.075"),
]


def seed_pricing(db: Session) -> None:
    inserted = 0
    for provider, model, inp, out, cached in DEFAULT_PRICING:
        exists = db.execute(
            select(ModelPricing.id)
            .where(
                ModelPricing.provider == provider,
                ModelPricing.model == model,
                ModelPricing.effective_to.is_(None),
            )
            .limit(1)
        ).first()
        if exists:
            continue
        db.add(
            ModelPricing(
                provider=provider,
                model=model,
                input_per_1m=Decimal(inp),
                output_per_1m=Decimal(out),
                cached_input_per_1m=Decimal(cached) if cached is not None else None,
                currency="USD",
            )
        )
        inserted += 1

    if inserted:
        db.commit()
        logger.info("Seeded %d model pricing row(s).", inserted)
