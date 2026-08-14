"""Usage & cost reporting endpoints."""
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import LlmUsage
from app.schemas import UsageBreakdown, UsageRow, UsageSummary, UsageTotals

router = APIRouter(prefix="/api/usage", tags=["usage"])


def _apply_filters(stmt, *, date_from, date_to, model, operation):
    if date_from is not None:
        stmt = stmt.where(LlmUsage.created_at >= date_from)
    if date_to is not None:
        stmt = stmt.where(LlmUsage.created_at <= date_to)
    if model:
        stmt = stmt.where(LlmUsage.model == model)
    if operation:
        stmt = stmt.where(LlmUsage.operation == operation)
    return stmt


@router.get("/summary", response_model=UsageSummary)
def usage_summary(
    db: Session = Depends(get_db),
    date_from: datetime | None = Query(None, alias="from"),
    date_to: datetime | None = Query(None, alias="to"),
    model: str | None = None,
):
    """Aggregate totals plus per-model and per-operation breakdowns."""
    cost = func.coalesce(func.sum(LlmUsage.total_cost), 0)
    inp = func.coalesce(func.sum(LlmUsage.input_tokens), 0)
    out = func.coalesce(func.sum(LlmUsage.output_tokens), 0)
    tot = func.coalesce(func.sum(LlmUsage.total_tokens), 0)
    cnt = func.count(LlmUsage.id)

    totals_stmt = _apply_filters(
        select(cnt, inp, out, tot, cost),
        date_from=date_from, date_to=date_to, model=model, operation=None,
    )
    c, i, o, t, k = db.execute(totals_stmt).one()
    totals = UsageTotals(
        count=c, input_tokens=i, output_tokens=o, total_tokens=t,
        total_cost=float(k), currency="USD",
    )

    def breakdown(group_col) -> list[UsageBreakdown]:
        stmt = _apply_filters(
            select(group_col, cnt, inp, out, tot, cost).group_by(group_col).order_by(cost.desc()),
            date_from=date_from, date_to=date_to, model=model, operation=None,
        )
        return [
            UsageBreakdown(
                key=key, count=c, input_tokens=i, output_tokens=o,
                total_tokens=t, total_cost=float(k),
            )
            for key, c, i, o, t, k in db.execute(stmt).all()
        ]

    return UsageSummary(
        totals=totals,
        by_model=breakdown(LlmUsage.model),
        by_operation=breakdown(LlmUsage.operation),
    )


@router.get("", response_model=list[UsageRow])
def usage_rows(
    db: Session = Depends(get_db),
    date_from: datetime | None = Query(None, alias="from"),
    date_to: datetime | None = Query(None, alias="to"),
    model: str | None = None,
    operation: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Most recent usage rows (paginated)."""
    stmt = _apply_filters(
        select(LlmUsage),
        date_from=date_from, date_to=date_to, model=model, operation=operation,
    )
    stmt = stmt.order_by(LlmUsage.created_at.desc()).limit(limit).offset(offset)
    return db.execute(stmt).scalars().all()
