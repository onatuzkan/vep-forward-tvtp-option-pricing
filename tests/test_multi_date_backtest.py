"""Tests for the multi-date forward-curve backtest harness.

Every fixture here is built from the REAL quote file shipped in the repository
(`inputs/market/vep_monthly_quotes.csv`, the observed 2025-12-31 VEP strip).
Nothing is synthetic: the interior-gap case is the real strip with one observed
month removed, which is exactly the situation a historical VEP strip can present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.path.insert(0, str(REPO / "scripts" / "backtest"))
import multi_date_forward_backtest as mdb  # noqa: E402

QUOTES = REPO / "inputs" / "market" / "vep_monthly_quotes.csv"


@pytest.fixture(scope="module")
def strip() -> pd.DataFrame:
    if not QUOTES.exists():                                   # pragma: no cover
        pytest.skip(f"{QUOTES} not present")
    return pd.read_csv(QUOTES)


# ---------------------------------------------------------------------------
def test_add_months_wraps_the_year():
    assert mdb._add_months(2025, 12, 1) == (2026, 1)
    assert mdb._add_months(2025, 12, 7) == (2026, 7)
    assert mdb._add_months(2023, 6, 7) == (2024, 1)
    assert mdb._add_months(2022, 12, 12) == (2023, 12)


def test_contiguous_strip_keeps_the_whole_real_strip(strip):
    kept, dropped = mdb._contiguous_strip(strip, 2025, 12, max_horizon=7)
    assert list(kept["tau"]) == [2, 3, 4, 5, 6, 7]
    assert dropped == []


def test_contiguous_strip_starts_at_the_first_quoted_month(strip):
    """The real 2025-12-31 strip has no January contract; tau=1 is absent."""
    kept, _ = mdb._contiguous_strip(strip, 2025, 12, max_horizon=7)
    assert kept["tau"].min() == 2, "January 2026 must not appear as a quote"


def test_contiguous_strip_truncates_at_an_interior_hole(strip):
    """Remove the observed April contract: May onward must be dropped, not filled."""
    holed = strip[strip["contract_name"] != "EBM0426"]
    kept, dropped = mdb._contiguous_strip(holed, 2025, 12, max_horizon=7)
    assert list(kept["tau"]) == [2, 3]
    assert dropped == ["2026-05", "2026-06", "2026-07"]


def test_contiguous_strip_respects_the_horizon(strip):
    kept, _ = mdb._contiguous_strip(strip, 2025, 12, max_horizon=4)
    assert list(kept["tau"]) == [2, 3, 4]


def test_contiguous_strip_empty_when_nothing_is_in_horizon(strip):
    kept, dropped = mdb._contiguous_strip(strip, 2025, 12, max_horizon=1)
    assert kept.empty and dropped == []


# ---------------------------------------------------------------------------
def test_realized_month_matches_the_published_backtest():
    """February 2026 baseload mean must equal the published realised figure."""
    ptf = mdb.load_realized_hourly()
    mean, coverage, n_hours = mdb.realized_month(ptf, 2026, 2)
    assert coverage == pytest.approx(1.0)
    assert n_hours == 672
    assert mean == pytest.approx(2078.20, abs=0.01)


def test_realized_month_rejects_a_month_outside_the_archive():
    ptf = mdb.load_realized_hourly()
    mean, coverage, _ = mdb.realized_month(ptf, 2030, 1)
    assert mean is None and coverage < mdb.MIN_REALIZED_COVERAGE


# ---------------------------------------------------------------------------
def test_rebuilt_curve_reproduces_the_published_2025_curve():
    """The harness must build the production curve, not an approximation of it."""
    published = REPO / "outputs" / "market_calibration_final" / "hourly_forward_curve.csv"
    if not published.exists():                                # pragma: no cover
        pytest.skip("published 2025-12-31 curve not present")
    out = mdb.verify_2025(mdb._cfg())
    assert out["status"] == "ok"
    assert out["max_abs_difference_TRY_MWh"] < 1.0e-4


def test_hour_weighted_mean_uses_delivery_hours():
    df = pd.DataFrame({"bias_TRY_MWh": [-100.0, -200.0],
                       "n_delivery_hours": [744, 672]})
    expected = (-100.0 * 744 - 200.0 * 672) / (744 + 672)
    assert mdb._hour_weighted_mean(df) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# quote-set builder: anchor check and the as-of rule (real downloaded GGF)
# ---------------------------------------------------------------------------
import build_vep_quote_sets as bq  # noqa: E402

HIST = REPO / "inputs" / "market" / "historical_vep" / "vep_history.csv"


@pytest.fixture(scope="module")
def hist() -> pd.DataFrame:
    if not HIST.exists():                                     # pragma: no cover
        pytest.skip(f"{HIST} not present (run build_vep_quote_sets.py)")
    return pd.read_csv(HIST)


def test_anchor_check_passes_on_the_downloaded_series(hist):
    """EBM0226 on 2025-12-31 must be 2900.99 and every production contract
    must be reproduced; otherwise the downloaded series is not the series the
    manuscript is built on."""
    rep = bq.anchor_check(hist)
    assert rep["status"] == "ok"
    assert rep["contracts"]["EBM0226"]["downloaded"] == pytest.approx(2900.99, abs=1e-9)
    assert rep["max_abs_diff_TRY_MWh"] <= 1e-9
    assert len(rep["contracts"]) == 6


def test_anchor_check_stops_when_the_anchor_price_differs(hist):
    """Shift the observed anchor row by more than the tolerance: hard stop."""
    shifted = hist.copy()
    m = (shifted["valuation_date"] == "2025-12-31") & (shifted["contract_name"] == "EBM0226")
    shifted.loc[m, "price_TRY_MWh"] += 1.5
    with pytest.raises(bq.AnchorMismatchError):
        bq.anchor_check(shifted)


def test_anchor_check_stops_when_the_anchor_row_is_missing(hist):
    without = hist[~((hist["valuation_date"] == "2025-12-31")
                     & (hist["contract_name"] == "EBM0226"))]
    with pytest.raises(bq.AnchorMismatchError):
        bq.anchor_check(without)


def test_asof_rule_uses_same_day_when_published(hist):
    assert bq.asof_quotation_date(hist, "2025-12-31", strict=False) == "2025-12-31"
    assert bq.asof_quotation_date(hist, "2025-12-31", strict=True) == "2025-12-31"


def test_asof_rule_carries_the_last_publication_forward_never_backward(hist):
    """2023-06-30 is Kurban Bayramı: the observed set is the 2023-06-26 GGF,
    and strict mode reports nothing rather than borrowing 2023-07-03."""
    assert bq.asof_quotation_date(hist, "2023-06-30", strict=False) == "2023-06-26"
    assert bq.asof_quotation_date(hist, "2023-06-30", strict=True) is None
    assert bq.asof_quotation_date(hist, "2023-12-31", strict=False) == "2023-12-29"


def test_asof_rule_refuses_to_reach_too_far_back(hist):
    """Before the first GGF publication there is nothing to carry forward."""
    assert bq.asof_quotation_date(hist, "2021-12-31", strict=False) is None


# ---------------------------------------------------------------------------
# backtest analyses on the produced multi-date table
# ---------------------------------------------------------------------------
BACKTEST = REPO / "outputs" / "multi_date" / "multi_date_backtest.csv"


@pytest.fixture(scope="module")
def backtest() -> pd.DataFrame:
    if not BACKTEST.exists():                                 # pragma: no cover
        pytest.skip(f"{BACKTEST} not present (run multi_date_forward_backtest.py)")
    return pd.read_csv(BACKTEST)


def test_non_overlapping_subsets_are_pairwise_disjoint(backtest):
    fams = mdb.non_overlapping_subsets(backtest)
    assert fams, "at least one family expected"
    for dates in fams.values():
        wins = (backtest[backtest["valuation_date"].isin(dates)]
                .groupby("valuation_date")["delivery_month"].agg(["min", "max"])
                .sort_index())
        for (_, a), (_, b) in zip(wins.iterrows(), wins.iloc[1:].iterrows()):
            assert b["min"] > a["max"]


def test_every_month_end_valuation_is_anchored(backtest):
    """The VEP front contract is delisted before month end, so horizon 1 is
    never quoted at a month-end valuation date in this sample."""
    h1 = backtest[backtest["horizon_months"] == 1]
    assert h1["near_term_anchored"].all()
    assert h1["quoted"].eq(False).all()
    assert (backtest[backtest["horizon_months"] > 1]["quoted"]).all()


def test_backtest_rows_carry_the_quotation_provenance(backtest):
    assert {"ggf_quotation_date", "ggf_staleness_days"} <= set(backtest.columns)
    assert backtest["ggf_staleness_days"].between(0, bq.MAX_STALENESS_DAYS).all()
    same_day = backtest[backtest["ggf_staleness_days"] == 0]
    assert (same_day["ggf_quotation_date"] == same_day["valuation_date"]).all()


def test_2025_12_31_rows_match_the_published_single_date_backtest(backtest):
    """The multi-date table must reproduce the published 2026 hour-weighted mean."""
    sub = backtest[backtest["valuation_date"] == "2025-12-31"]
    assert len(sub) == 7
    assert mdb._hour_weighted_mean(sub) == pytest.approx(mdb.REFERENCE_2026_MEAN_BIAS, abs=0.01)


def test_min_quotes_rule_rejects_a_two_contract_strip(tmp_path, strip):
    """A quote file with two contracts must be reported, not analysed."""
    two = strip.head(2).copy()
    qdir = tmp_path / "2025-12-31"
    qdir.mkdir()
    two.to_csv(qdir / "vep_monthly_quotes.csv", index=False)
    orig = mdb.VEP_DIR
    mdb.VEP_DIR = tmp_path
    try:
        res = mdb.build_one_date("2025-12-31", mdb.load_realized_hourly(),
                                 mdb._cfg(), 7, tmp_path / "out")
    finally:
        mdb.VEP_DIR = orig
    assert res.status == "TOO_FEW_QUOTES"
    assert res.rows == []
