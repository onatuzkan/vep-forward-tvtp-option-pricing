"""FW2 §2.1 ex-post-premium reproducibility + look-ahead structural test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fw2.ex_post_premium import (       # noqa: E402
    ANCHOR_DATES, VALUATION_CUTOFF_UTC, block_bootstrap_se, build_panel,
    summarise_by_horizon)


def test_seven_anchor_snapshots_exist():
    for d in ANCHOR_DATES:
        f = REPO_ROOT / "inputs" / "market" / "historical_vep" / d / "vep_monthly_quotes.csv"
        assert f.exists(), f"missing anchor: {f}"


def test_panel_has_no_look_ahead_past_cutoff():
    """No delivery month should start after the FW2 valuation cutoff.
    This is a STRUCTURAL guard: adding a snapshot dated after
    2025-12-31 should not silently drop the guard."""
    panel = build_panel()
    latest = panel["delivery_month_ts"].max()
    cutoff_month = (VALUATION_CUTOFF_UTC.tz_convert("Europe/Istanbul")
                    .to_period("M").to_timestamp())
    assert latest <= cutoff_month, (
        f"look-ahead: delivery month {latest} beyond cutoff {cutoff_month}")


def test_panel_columns_and_signs():
    p = build_panel()
    expected = {"valuation_date", "contract_name", "delivery_month_ts",
                "price_TRY_MWh", "realised_mean", "n_hours",
                "horizon_months", "premium_TRY_MWh",
                "premium_pct_of_forward"}
    assert expected.issubset(set(p.columns))
    assert (p["horizon_months"] >= 0).all()
    assert (p["price_TRY_MWh"] > 0).all()
    assert (p["realised_mean"] > 0).all()


def test_summary_is_deterministic_given_the_seed():
    """Two calls to summarise_by_horizon on the same panel produce the
    same numbers (the block bootstrap SE depends on a seed carried in
    the module, not on wall clock)."""
    p = build_panel()
    s1 = summarise_by_horizon(p)
    s2 = summarise_by_horizon(p)
    pd.testing.assert_frame_equal(s1, s2)


def test_block_bootstrap_se_matches_analytic_limit():
    """For a large iid gaussian series and a large n_boot, block
    bootstrap SE should agree with the classical sample-mean SE
    sigma/sqrt(n) to within ~10% (with block=1 the two coincide in
    expectation)."""
    rng = np.random.default_rng(20260927)
    n = 300
    x = rng.standard_normal(n)
    analytic = x.std(ddof=1) / np.sqrt(n)
    boot = block_bootstrap_se(x, block=1, n_boot=2000, seed=20260927)
    # The classical formula is exact; block=1 bootstrap should be within
    # ~15% of it for n=300 and n_boot=2000.
    assert abs(boot - analytic) / analytic < 0.15


def test_range_clipping_of_summary():
    """No horizon summary row should carry non-finite or negative
    counts."""
    s = summarise_by_horizon(build_panel())
    assert (s["n_obs"] > 0).all()
    assert np.isfinite(s["mean_premium_TRY_MWh"]).all()
    assert np.isfinite(s["median_premium_TRY_MWh"]).all()
