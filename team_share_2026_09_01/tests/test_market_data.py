"""VEP quote schema: monthly baseload semantics, validation, missing months."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pde_option_model.market_data import (MarketDataError, MarketQuoteSet,
                                          MonthlyBaseloadQuote,
                                          MonthlyQuoteMisuseError,
                                          assert_not_point_forward, load_quotes)

from .conftest import QUOTES_CSV, QUOTES_JSON


def test_monthly_quote_has_no_single_maturity(quotes):
    """A monthly VEP contract must NOT be treated as a point-maturity forward."""
    q = quotes.quotes[0]
    with pytest.raises(MonthlyQuoteMisuseError):
        _ = q.maturity_utc
    with pytest.raises(MonthlyQuoteMisuseError):
        q.as_point_forward()
    with pytest.raises(MonthlyQuoteMisuseError):
        assert_not_point_forward(q)
    with pytest.raises(MonthlyQuoteMisuseError):
        assert_not_point_forward(quotes.quotes)


def test_point_forward_reading_would_give_a_different_answer(model, quotes):
    """Collapsing EBM0226 to its month-midpoint is numerically wrong, not just banned."""
    q = quotes.get(2026, 2)
    hrs = q.delivery.hours_utc()
    h = np.asarray((hrs - model.valuation_utc).total_seconds(), float) / 3600.0
    true_average = float(np.mean(model.expected_spot(h)))
    midpoint = float(model.expected_spot(np.array([h[len(h) // 2]]))[0])
    start = float(model.expected_spot(np.array([h[0]]))[0])
    end = float(model.expected_spot(np.array([h[-1]]))[0])
    assert abs(true_average - q.price_TRY_MWh) < 1e-8
    assert abs(midpoint - true_average) > 1e-6
    assert abs(start - true_average) > 1e-6
    assert abs(end - true_average) > 1e-6


def test_delivery_hour_counts_per_contract(quotes):
    expected = {"EBM0226": 672, "EBM0326": 744, "EBM0426": 720,
                "EBM0526": 744, "EBM0626": 720, "EBM0726": 744}
    assert {q.contract_name: q.n_delivery_hours for q in quotes.quotes} == expected


def test_csv_and_json_agree(quotes):
    other = load_quotes(QUOTES_JSON)
    assert quotes.valuation_utc == other.valuation_utc
    assert [q.price_TRY_MWh for q in quotes.quotes] == \
           [q.price_TRY_MWh for q in other.quotes]


def test_valuation_is_turkish_day_end(quotes):
    assert quotes.valuation_utc == pd.Timestamp("2025-12-31T20:00:00Z")


def test_missing_january_is_detected(quotes):
    """January 2026 is inside the horizon but carries no observed quote."""
    assert quotes.missing_months() == ["2026-01"]
    assert quotes.partial_months() == ["2025-12"]
    assert quotes.gap_months() == []


def test_validate_warns_about_the_unquoted_month(quotes):
    warns = quotes.validate()
    assert any("2026-01" in w for w in warns)
    assert any("near-term anchored" in w for w in warns)


def test_interior_gap_is_rejected():
    """A hole between two quoted months cannot be interpolated silently."""
    qs = MarketQuoteSet(
        valuation_utc=pd.Timestamp("2025-12-31T20:00:00Z"),
        quotes=[MonthlyBaseloadQuote("EBM0226", 2026, 2, 2900.99),
                MonthlyBaseloadQuote("EBM0426", 2026, 4, 2500.66)])
    assert qs.gap_months() == ["2026-03"]
    with pytest.raises(MarketDataError, match="interior delivery months"):
        qs.validate(strict_interior=True)


def test_adding_a_january_quote_removes_the_extrapolation(quotes):
    """Schema extension test: dropping in EBM0126 makes January a constraint."""
    extended = MarketQuoteSet(
        valuation_utc=quotes.valuation_utc,
        quotes=list(quotes.quotes) + [MonthlyBaseloadQuote("EBM0126", 2026, 1, 3000.0)],
        spot_price_TRY_MWh=quotes.spot_price_TRY_MWh)
    assert extended.missing_months() == []
    assert extended.has(2026, 1)


def test_duplicate_months_rejected():
    with pytest.raises(MarketDataError, match="duplicate"):
        MarketQuoteSet(valuation_utc=pd.Timestamp("2025-12-31T20:00:00Z"),
                       quotes=[MonthlyBaseloadQuote("EBM0226", 2026, 2, 2900.99),
                               MonthlyBaseloadQuote("EBM0226", 2026, 2, 2800.0)])


def test_expired_contract_rejected():
    with pytest.raises(MarketDataError, match="before valuation"):
        MarketQuoteSet(valuation_utc=pd.Timestamp("2026-06-01T00:00:00Z"),
                       quotes=[MonthlyBaseloadQuote("EBM0226", 2026, 2, 2900.99)])


@pytest.mark.parametrize("kwargs,msg", [
    ({"price_TRY_MWh": -5.0}, "non-positive"),
    ({"price_TRY_MWh": float("nan")}, "NaN"),
    ({"quote_type": "daily_baseload"}, "unsupported quote_type"),
    ({"weight": 0.0}, "weight"),
])
def test_malformed_quotes_rejected(kwargs, msg):
    base = dict(contract_name="EBM0226", year=2026, month=2, price_TRY_MWh=2900.99)
    base.update(kwargs)
    with pytest.raises(MarketDataError, match=msg):
        MonthlyBaseloadQuote(**base)


def test_delivery_average_requires_full_coverage(quotes, curve_smooth):
    q = quotes.get(2026, 2)
    assert abs(q.delivery_average(curve_smooth.values) - q.price_TRY_MWh) < 1e-8
    truncated = curve_smooth.values.iloc[:100]
    with pytest.raises(MarketDataError, match="does not cover"):
        q.delivery_average(truncated)


def test_quote_frame_schema(quotes):
    df = quotes.to_frame()
    for col in ("contract_name", "delivery_month_label", "price_TRY_MWh",
                "n_delivery_hours", "delivery_start_utc", "delivery_end_utc"):
        assert col in df.columns
    assert len(df) == 6
