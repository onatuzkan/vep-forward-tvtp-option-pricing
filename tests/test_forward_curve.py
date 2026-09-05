"""Forward curve: exact monthly averages, schema, flags, smoothing safety."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pde_option_model.calendar_tr import DeliveryMonth
from pde_option_model.forward_curve import (CURVE_COLUMNS, ForwardCurveError,
                                            NearTermAnchor, build_forward_curve,
                                            load_forward_curve_csv)
from pde_option_model.market_data import MarketQuoteSet, MonthlyBaseloadQuote

MODES = ("piecewise_constant", "smooth_constrained")


@pytest.mark.parametrize("mode", MODES)
def test_monthly_average_constraint_holds_exactly(quotes, params, mode):
    """(1/N_m) sum_h F(h) == VEP_m for every quoted month, both modes."""
    c = build_forward_curve(quotes, mode=mode,
                            spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    assert c.max_abs_monthly_error() < 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_exact_quote_reproduction_per_contract(quotes, params, mode):
    c = build_forward_curve(quotes, mode=mode,
                            spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    for _, row in c.fit_table().iterrows():
        assert abs(row["residual_TRY_MWh"]) < 1e-6, row["contract_name"]
        assert abs(row["relative_error_pct"]) < 1e-9


def test_monthly_average_recomputed_from_utc_delivery_hours(quotes, curve_smooth):
    """Recompute independently from the curve's own hourly UTC index."""
    for q in quotes.quotes:
        hrs = DeliveryMonth(q.year, q.month).hours_utc()
        avg = float(curve_smooth.values.reindex(hrs).to_numpy().mean())
        assert len(hrs) == q.n_delivery_hours
        assert abs(avg - q.price_TRY_MWh) < 1e-6


def test_smoothing_cannot_break_the_monthly_constraints(quotes, params):
    """Constraints are hard equalities: any smoothness weight keeps them exact."""
    for w_smooth, w_level in [(1e-6, 1.0), (1.0, 1e-4), (1e4, 1e-8)]:
        c = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=params.spot_price_TRY_MWh,
                                smoothness_weight=w_smooth, level_weight=w_level)
        assert c.max_abs_monthly_error() < 1e-6, (w_smooth, w_level)


def test_smooth_curve_is_smoother_than_piecewise(curve_pwc, curve_smooth):
    d2p = np.diff(curve_pwc.values.to_numpy(), 2)
    d2s = np.diff(curve_smooth.values.to_numpy(), 2)
    assert np.sum(d2s ** 2) < np.sum(d2p ** 2)


@pytest.mark.parametrize("mode", MODES)
def test_curve_schema_columns(quotes, params, mode):
    c = build_forward_curve(quotes, mode=mode,
                            spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    assert list(c.frame.columns) == list(CURVE_COLUMNS)
    assert len(c.frame) == len(c.values)
    assert c.frame["hourly_forward_TRY_MWh"].notna().all()
    for col in ("interpolation_flag", "extrapolation_flag", "near_term_anchor_flag"):
        assert c.frame[col].dtype == bool


def test_january_rows_are_flagged_as_extrapolation(curve_smooth):
    f = curve_smooth.frame
    jan = f[f["delivery_month"] == "2026-01"]
    feb = f[f["delivery_month"] == "2026-02"]
    assert len(jan) == 744 and len(feb) == 672
    assert jan["extrapolation_flag"].all()
    assert jan["near_term_anchor_flag"].all()
    assert (jan["contract_name"].fillna("") == "").all()
    assert not feb["extrapolation_flag"].any()
    assert not feb["near_term_anchor_flag"].any()
    assert (feb["contract_name"] == "EBM0226").all()
    assert "2026-01" in curve_smooth.extrapolated_months
    assert "2026-02" in curve_smooth.constrained_months


def test_quoted_rows_carry_their_contract_name(curve_smooth, quotes):
    f = curve_smooth.frame
    for q in quotes.quotes:
        rows = f[f["delivery_month"] == q.label]
        assert (rows["contract_name"] == q.contract_name).all()
        assert (rows["source"] == f"EPIAS_VEP:{q.contract_name}").all()


def test_time_turkey_column_is_three_hours_ahead(curve_smooth):
    f = curve_smooth.frame.head(50)
    u = pd.DatetimeIndex(pd.to_datetime(f["time_utc"], utc=True))
    t = pd.DatetimeIndex(pd.to_datetime(f["time_turkey"], utc=True))
    assert (u == t).all()
    local_hours = pd.to_datetime(f["time_turkey"]).dt.hour.to_numpy()
    utc_hours = u.hour.to_numpy()
    assert np.all((local_hours - utc_hours) % 24 == 3)


@pytest.mark.parametrize("mode,pins", [("spot_flat", True),
                                       ("spot_to_next_linear", True),
                                       ("flat_next_month", False)])
def test_anchor_modes(quotes, params, mode, pins):
    a = NearTermAnchor(mode=mode)
    c = build_forward_curve(quotes, mode="smooth_constrained", anchor=a,
                            spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    assert c.max_abs_monthly_error() < 1e-6
    assert a.pins_spot is pins
    if pins:
        assert abs(float(c.values.iloc[0]) - params.spot_price_TRY_MWh) < 1e-6


def test_explicit_level_anchor_requires_a_level():
    with pytest.raises(ForwardCurveError, match="explicit_level"):
        NearTermAnchor(mode="explicit_level")


def test_explicit_level_anchor_moves_only_january(quotes, params):
    c = build_forward_curve(
        quotes, mode="piecewise_constant",
        anchor=NearTermAnchor(mode="explicit_level", level_TRY_MWh=3300.0),
        spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    jan = c.frame[c.frame["delivery_month"] == "2026-01"]
    assert np.allclose(jan["hourly_forward_TRY_MWh"].to_numpy(), 3300.0)
    assert c.max_abs_monthly_error() < 1e-6


def test_curve_query_and_horizon_guard(curve_smooth, params):
    v = curve_smooth.at_hours(params.valuation_utc, np.array([0.0, 72.0, 720.0]))
    assert v.shape == (3,) and np.all(np.isfinite(v))
    assert abs(v[0] - params.spot_price_TRY_MWh) < 1e-6
    with pytest.raises(ForwardCurveError, match="outside the curve horizon"):
        curve_smooth.at_hours(params.valuation_utc, np.array([100000.0]))


def test_shape_profile_preserves_monthly_average(quotes, params):
    idx = pd.date_range(quotes.valuation_utc, quotes.last_delivery_utc,
                        freq="h", inclusive="left", tz="UTC")
    shape = pd.Series(1.0 + 0.25 * np.sin(2 * np.pi * idx.hour / 24.0), index=idx)
    c = build_forward_curve(quotes, mode="piecewise_constant", shape_profile=shape,
                            spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    assert c.max_abs_monthly_error() < 1e-6
    assert c.values.std() > 0


def test_interior_gap_curve_is_refused(params):
    qs = MarketQuoteSet(
        valuation_utc=pd.Timestamp("2025-12-31T20:00:00Z"),
        quotes=[MonthlyBaseloadQuote("EBM0226", 2026, 2, 2900.99),
                MonthlyBaseloadQuote("EBM0426", 2026, 4, 2500.66)],
        spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    with pytest.raises(ForwardCurveError, match="no quote and no anchor"):
        build_forward_curve(qs, mode="piecewise_constant",
                            spot_price_TRY_MWh=params.spot_price_TRY_MWh)


def test_curve_csv_roundtrip(tmp_path, curve_smooth):
    p = tmp_path / "hourly_forward_curve.csv"
    curve_smooth.frame.to_csv(p, index=False)
    back = load_forward_curve_csv(p)
    assert len(back) == len(curve_smooth.values)
    assert np.allclose(back.to_numpy(), curve_smooth.values.to_numpy())


def test_curve_values_are_economically_plausible(curve_smooth):
    v = curve_smooth.values.to_numpy()
    assert np.all(np.isfinite(v))
    assert v.min() > 0.0
    assert v.max() < 1.0e5
