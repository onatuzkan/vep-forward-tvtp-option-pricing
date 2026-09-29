"""FW11 tests -- variance stability reproduction of FW9 numbers,
look-ahead guard, sd_prod formula sanity."""
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

OUT = REPO / "outputs" / "fw11_variance_stability"


# --------------------------------------------------------------------
# 1. 2025-12-31 row reproduces FW9 numbers.
# --------------------------------------------------------------------
@pytest.mark.skipif(not (OUT / "variance_stability.csv").exists(),
                    reason="run scripts/fw11/variance_stability.py first")
def test_2025_12_31_row_reproduces_fw9_values():
    """The 2025-12-31 row must reproduce:
      - sd_pooled = 624 (matches stationary_variance_check.md)
      - sd_12m_only = 531 (matches tail_validation.md)
      - kappa_per_hour_pooled ~ 0.20 (matches
        yearly_kappa_A3_TRY.csv 2025 row = 0.2013)
    """
    df = pd.read_csv(OUT / "variance_stability.csv")
    row = df[df["label"] == "2025-12-31"]
    assert not row.empty, "missing 2025-12-31 row"
    r = row.iloc[0]
    assert abs(float(r["sd_pooled"]) - 624.0) < 1.0, (
        f"sd_pooled = {r['sd_pooled']:.3f} deviates from FW9 = 624")
    assert abs(float(r["sd_12m_only"]) - 531.0) < 1.0, (
        f"sd_12m_only = {r['sd_12m_only']:.3f} deviates from "
        f"FW9 tail_validation = 531")
    kappa = float(r["kappa_per_hour_pooled"])
    assert abs(kappa - 0.2013) < 0.01, (
        f"kappa = {kappa:.4f} deviates from FW9 yearly A3 = 0.2013")
    # L must match: 2025 12-month mean
    L = float(r["L_mean_TRY_MWh"])
    assert 2500.0 < L < 2700.0, f"L={L:.1f} out of expected 2025 range"
    # sd_prod at that L: 0.1987 * sqrt(L^2 + 282.48^2)
    expected_sd_prod = 0.198734 * math.sqrt(L * L + 282.48 * 282.48)
    assert abs(float(r["sd_prod_TRY"]) - expected_sd_prod) < 0.5, (
        f"sd_prod inconsistent: got {r['sd_prod_TRY']:.3f}, "
        f"expected {expected_sd_prod:.3f}")


# --------------------------------------------------------------------
# 2. Look-ahead guard: eval_row for a valuation date must not touch
#    any PTF hour past that date.
# --------------------------------------------------------------------
def test_eval_row_never_uses_future_hours():
    """Set the valuation to 2024-06-30 and check that the returned
    statistics are identical whether or not we feed the loader
    extra 2025-2026 hours.  If the loader touched future hours,
    the residual sd would differ."""
    from scripts.fw11.variance_stability import (DateSpec, eval_row,
                                                    _end_utc_of_date,
                                                    _twelve_month_start,
                                                    load_ptf_hourly)
    ptf_all = load_ptf_hourly()
    end_utc = _end_utc_of_date("2024-06-30")
    start_utc = _twelve_month_start(end_utc)
    spec = DateSpec(label="2024-06-30", valuation_utc=end_utc,
                      start_utc=start_utc)
    r_full = eval_row(ptf_all, spec, scale_P=282.48)
    ptf_truncated = ptf_all[ptf_all.index <= end_utc]
    r_trunc = eval_row(ptf_truncated, spec, scale_P=282.48)
    for k in ("sd_pooled", "sd_12m_only", "kappa_per_hour_pooled",
              "L_mean_TRY_MWh"):
        assert abs(float(r_full[k]) - float(r_trunc[k])) < 1e-9, (
            f"look-ahead detected in field {k}: "
            f"full={r_full[k]}, truncated={r_trunc[k]}")


# --------------------------------------------------------------------
# 3. sd_prod formula correctness at known reference points.
# --------------------------------------------------------------------
def test_sd_prod_formula_at_shipped_spot():
    """At L = 2917.78 the sd_prod formula must return 582.48 TRY/MWh
    (the FW9 stationary_variance_check.md headline number)."""
    from scripts.fw11.variance_stability import sd_prod_at_L
    L = 2917.78
    scale_P = 282.48
    got = sd_prod_at_L(L, scale_P)
    assert abs(got - 582.48) < 0.5, (
        f"sd_prod(L=2917.78) = {got:.3f}, expected ~582.48")
    # At L = 0 the formula reduces to 0.1987 * scale_P
    got_at_zero = sd_prod_at_L(0.0, scale_P)
    assert abs(got_at_zero - 0.198734 * scale_P) < 1e-9
    # Linear scaling: at large L the ratio sd_prod/L -> 0.1987
    got_at_huge = sd_prod_at_L(1e6, scale_P)
    assert abs(got_at_huge / 1e6 - 0.198734) < 1e-4
