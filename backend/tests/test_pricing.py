"""Cost arithmetic.

Every phase of this project reports cost/query, and Phase 8 (prompt caching)
argues its entire case from the cached-token discount path — which nothing
exercises today. If this math is wrong, every experiment conclusion is wrong.
"""
from decimal import Decimal

from app.models import ModelPricing
from app.services.pricing import compute_cost

# gpt-4o-mini rates (USD per 1M tokens), matching pricing_seed.DEFAULT_PRICING.
RATES = {"input_per_1m": Decimal("0.15"), "output_per_1m": Decimal("0.60")}


def pricing(cached: str | None = "0.075") -> ModelPricing:
    return ModelPricing(
        provider="openai",
        model="gpt-4o-mini",
        cached_input_per_1m=Decimal(cached) if cached is not None else None,
        currency="USD",
        **RATES,
    )


def test_missing_pricing_yields_no_cost():
    """A model absent from the rate book must report null cost, not zero.

    Zero would silently understate spend in the usage summary; null makes the
    gap visible.
    """
    assert compute_cost(1000, 500, 0, None) == (None, None, None)


def test_input_and_output_priced_at_their_own_rates():
    inp, out, total = compute_cost(1_000_000, 1_000_000, 0, pricing())
    assert inp == Decimal("0.150000")
    assert out == Decimal("0.600000")
    assert total == Decimal("0.750000")


def test_cached_tokens_billed_at_the_discounted_rate():
    # 600 uncached @ 0.15/1M + 400 cached @ 0.075/1M
    inp, _, _ = compute_cost(1000, 0, 400, pricing())
    assert inp == Decimal("0.000120")


def test_cached_tokens_are_cheaper_than_uncached():
    """The premise Phase 8 rests on: caching must actually reduce input cost."""
    uncached, _, _ = compute_cost(1000, 0, 0, pricing())
    cached, _, _ = compute_cost(1000, 0, 1000, pricing())
    assert cached < uncached


def test_missing_cached_rate_falls_back_to_input_rate():
    """No discount configured means no discount applied — never a free ride."""
    inp, _, _ = compute_cost(1000, 0, 1000, pricing(cached=None))
    assert inp == Decimal("0.000150")


def test_cached_exceeding_input_never_produces_negative_cost():
    """Providers can report cached_tokens oddly; cost must not go negative."""
    inp, _, total = compute_cost(100, 0, 999_999, pricing())
    assert inp >= 0
    assert total >= 0


def test_zero_tokens_costs_nothing():
    assert compute_cost(0, 0, 0, pricing()) == (
        Decimal("0.000000"),
        Decimal("0.000000"),
        Decimal("0.000000"),
    )


def test_total_is_the_sum_of_its_parts():
    inp, out, total = compute_cost(12_345, 6_789, 2_000, pricing())
    assert total == inp + out


def test_cost_is_quantized_to_six_decimal_places():
    """Matches Numeric(12, 6) so the stored value equals the returned value."""
    _, _, total = compute_cost(1, 1, 0, pricing())
    assert total.as_tuple().exponent == -6
