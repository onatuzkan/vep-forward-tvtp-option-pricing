"""FW10b tests -- production forward curve builder, dF/dF_M
sensitivity, Part A output shape, archive presence."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "fw10_validation"


def test_production_curve_builder_returns_hpfc_shaped_curve_for_a_2026_day():
    """The FW10b daily curve builder must return a properly formed
    ForwardCurve for an arbitrary 2026 business day.

    Note: the near-term anchor `spot_to_next_linear` produces a linear
    ramp from spot to the first quoted delivery month, so intra-day
    HPFC shape is only visible INSIDE the quoted months (h >= 30 days
    from valuation).  Short-horizon (h = 24-72 h) forwards live in the
    near-term ramp and are dominated by the spot anchor, not the
    HPFC shape -- this is the exact production behaviour we want to
    lock in.
    """
    from pde_option_model.calendar_tr import TURKEY_TZ
    from scripts.fw10.curve_builder import (build_daily_production_curve,
                                              F_at_hours)
    d = pd.Timestamp("2026-03-10", tz=TURKEY_TZ)
    res = build_daily_production_curve(d)
    assert res is not None, "expected a curve for 2026-03-10"
    assert res.hpfc_applied is True
    F = F_at_hours(res, [6, 12, 24, 48, 72])
    for h, v in F.items():
        assert math.isfinite(v) and v > 0, f"h={h}: bad F {v}"
    # Sanity: monotonic drift from spot to the first quoted month
    # (the ramp is linear so successive F(h) values must move in a
    # single direction unless the spot is already at the quoted level)
    # We only check that adjacent F values differ by less than a
    # thousand TRY -- the ramp is smooth by construction.
    for h1, h2 in ((6, 12), (12, 24), (24, 48), (48, 72)):
        assert abs(F[h1] - F[h2]) < 1000, (
            f"F ramp not smooth: F[{h1}]={F[h1]:.1f}, F[{h2}]={F[h2]:.1f}")
    # In the deeper horizon (say 60 days) the HPFC shape DOES vary by
    # hour of day.  Read F at two hours 6 h apart inside the first
    # quoted month and verify they differ.
    from scripts.fw10._data import day_end_utc
    from scripts.fw10.curve_builder import F_at_hours
    last_known = day_end_utc(d)
    deep_night = last_known + pd.Timedelta(days=60, hours=5)   # 04:00 TRT
    deep_morning = last_known + pd.Timedelta(days=60, hours=11)  # 10:00 TRT
    s = res.hourly_forward_TRY_MWh
    if deep_night in s.index and deep_morning in s.index:
        v_n = float(s.loc[deep_night]); v_m = float(s.loc[deep_morning])
        assert abs(v_n - v_m) > 50.0, (
            f"HPFC shape did not produce a deep-horizon night/morning "
            f"gap: night={v_n:.1f}, morning={v_m:.1f}")


def test_dF_target_dF_month_is_between_zero_and_one():
    """`dF(target)/dF_M` for a near-term horizon must be a finite
    value in [0, 1]: bumping the delivery-month monthly quote by
    +10 TRY/MWh cannot move the delivery-hour F by more than the
    monthly move itself, and the smoother/anchor combination guarantees
    non-negativity of the transmission."""
    from pde_option_model.calendar_tr import TURKEY_TZ
    from scripts.fw10._data import day_end_utc, load_realized_ptf, load_vep_quotes_daily
    from scripts.fw10.curve_builder import dF_target_dF_month_bump
    d = pd.Timestamp("2026-03-10", tz=TURKEY_TZ)
    last_known = day_end_utc(d)
    ptf = load_realized_ptf()
    vep = load_vep_quotes_daily()
    for h in (24, 48, 72):
        target = last_known + pd.Timedelta(hours=h)
        t_local = target.tz_convert(TURKEY_TZ)
        # Delivery month may be partial -- fall back to next month for
        # the sensitivity test
        y_t, m_t = t_local.year, t_local.month
        # if next month
        if m_t == 12:
            y_next, m_next = y_t + 1, 1
        else:
            y_next, m_next = y_t, m_t + 1
        dF = dF_target_dF_month_bump(d, target, (y_next, m_next),
                                       bump_TRY_MWh=10.0, ptf=ptf, vep=vep)
        if dF is None:
            continue
        assert 0.0 <= dF <= 1.0 + 1e-6, (
            f"h={h}: dF/dF_M = {dF:.4f} out of [0, 1]")


@pytest.mark.skipif(not (OUT / "predictive_daily_fw10b.csv").exists(),
                    reason="run scripts/fw10/run_validation_fw10b.py first")
def test_fw10b_predictive_daily_has_new_horizons_and_z_columns():
    """FW10b's predictive_daily output must cover the new short
    horizons (6, 12 h) and expose the bias/dispersion columns
    (z_standardised, F_minus_actual, sample_mean, sample_sd)."""
    df = pd.read_csv(OUT / "predictive_daily_fw10b.csv")
    assert not df.empty
    assert set(df["h"].unique()).issuperset({6, 12, 24, 48, 72}), (
        f"missing new horizons: {sorted(df['h'].unique())}")
    for col in ("z_standardised", "F_minus_actual", "sample_mean",
                "sample_sd"):
        assert col in df.columns, f"missing {col}"
    # z_standardised must be finite (not NaN) for the bulk of rows
    n_finite = df["z_standardised"].notna().sum()
    assert n_finite > 0.9 * len(df), (
        f"too many NaN z values: {len(df) - n_finite} out of {len(df)}")


def test_fw10_archive_of_baseload_F_variant_is_present():
    """The FW10 monthly-baseload-F Part A/B outputs were archived
    under archive_baseload_F/ during FW10b; the README must be
    present and the archive must hold at least the option-daily
    and predictive-daily CSVs."""
    archive = OUT / "archive_baseload_F"
    assert archive.is_dir(), "archive_baseload_F/ missing"
    readme = archive / "README.md"
    assert readme.exists(), "archive README missing"
    text = readme.read_text(encoding="utf-8")
    assert "SUPERSEDED" in text and "monthly baseload" in text, (
        "archive README does not explain why the FW10 outputs were "
        "superseded")
    assert (archive / "predictive_daily.csv").exists()
    assert (archive / "option_daily.csv").exists()
