"""Persist collected usage records, computing cost per row.

Best-effort: a failure here is logged and swallowed so it can never break the
user-facing request. Returns a `TurnUsage` summary (the same numbers it stored)
so the route can surface usage to the client — single source of truth.
"""
import logging
from decimal import Decimal

from sqlalchemy.orm import Session

from app.models import LlmUsage
from app.schemas import TurnUsage, TurnUsageCall
from app.services import pricing as pricing_service
from app.services.llm import UsageContext

logger = logging.getLogger(__name__)


def persist(db: Session, ctx: UsageContext) -> TurnUsage | None:
    if not ctx.records:
        return None

    rows: list[LlmUsage] = []
    calls: list[TurnUsageCall] = []
    currency = "USD"
    total_cost = Decimal("0")
    cost_complete = True
    in_tokens = out_tokens = tot_tokens = 0

    for r in ctx.records:
        price = pricing_service.resolve_pricing(
            db, r.provider, r.model, fallback_model=r.model_requested
        )
        if price is None:
            logger.warning(
                "No pricing for provider=%s model=%s (op=%s); cost left null.",
                r.provider, r.model, r.operation,
            )
        else:
            currency = price.currency

        input_cost, output_cost, call_cost = pricing_service.compute_cost(
            r.input_tokens, r.output_tokens, r.cached_tokens, price
        )

        rows.append(
            LlmUsage(
                request_id=ctx.request_id,
                provider=r.provider,
                model_requested=r.model_requested,
                model=r.model,
                operation=r.operation,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                total_tokens=r.total_tokens,
                cached_tokens=r.cached_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=call_cost,
                currency=price.currency if price else "USD",
                pricing_id=price.id if price else None,
                latency_ms=r.latency_ms,
                raw_usage=r.raw_usage,
            )
        )
        calls.append(
            TurnUsageCall(
                operation=r.operation,
                model=r.model,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                total_tokens=r.total_tokens,
                cost=float(call_cost) if call_cost is not None else None,
            )
        )

        in_tokens += r.input_tokens
        out_tokens += r.output_tokens
        tot_tokens += r.total_tokens
        if call_cost is None:
            cost_complete = False
        else:
            total_cost += call_cost

    try:
        db.add_all(rows)
        db.commit()
    except Exception:
        logger.exception("Failed to persist %d usage row(s).", len(rows))
        db.rollback()
        return None

    # Primary model = the synthesis call (falls back to the last call).
    primary_model = next(
        (c.model for c in calls if c.operation == "synthesize"), calls[-1].model
    )

    return TurnUsage(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        total_tokens=tot_tokens,
        cost=float(total_cost) if cost_complete else None,
        currency=currency,
        model=primary_model,
        calls=len(calls),
        request_id=ctx.request_id,
        breakdown=calls,
    )
