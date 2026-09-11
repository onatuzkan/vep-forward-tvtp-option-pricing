"""Tests for the Q1 forward risk-premium module -- REAL data only.

Data used: inputs/market/vep_monthly_quotes.csv (EPİAŞ VEP strip of
2025-12-31), inputs/market/realized_ptf_2026.csv and, when present,
inputs/historical/ptf_raw/ptf_2019.csv (EPİAŞ Şeffaflık export).  The only
non-data inputs are the closed-form identities being verified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pde_option_model.forward_centered import ResidualSpec, residual_moments
from pde_option_model.premium import (drift_from_premium, fit_premium_curve,
                                      load_epias_ptf_csv, load_vep_history, ou_mean_with_drift,
                                      physical_mean, premium_path_hours,
                                      realized_monthly_means, realized_premium_panel,
                                      stress_premium_variance_uplift)

# Fast-factor values estimated on EPİAŞ PTF 2023-2025 (scripts/residual/fit_residual_v2.py):
# phi = 0.6764 -> kappa = -ln(phi); P(0->1), P(1->0) at z = ramp = 0.
KAPPA_FAST = 0.3909
Q01_REAL = 1.0 / (1.0 + np.exp(2.0004))
Q10_REAL = 1.0 / (1.0 + np.exp(2.3511))


@pytest.fixture(scope="module")
def real_panel(repo_root):
    vep = load_vep_history(repo_root / "inputs/market/vep_monthly_quotes.csv")
    real = realized_monthly_means(
        load_epias_ptf_csv(repo_root / "inputs/market/realized_ptf_2026.csv"))
    return vep, real


@pytest.fixture(scope="module")
def real_curve(repo_root):
    c = pd.read_csv(repo_root / "outputs/market_calibration_final/hourly_forward_curve.csv")
    t = pd.to_datetime(c["time_utc"], utc=True)
    return ((t - t.iloc[0]).dt.total_seconds().to_numpy() / 3600.0,
            c["hourly_forward_TRY_MWh"].to_numpy())


# --------------------------------------------------------------------------
# realised premium panel on the actual 2025-12-31 VEP strip
# --------------------------------------------------------------------------
def test_realized_premium_matches_hand_computation(real_panel):
    vep, real = real_panel
    p = realized_premium_panel(vep, real)
    feb = p[(p.delivery_year == 2026) & (p.delivery_month == 2)].iloc[0]
    assert feb["tau_months"] == 2
    assert feb["rel_premium"] == pytest.approx((2900.99 - 2078.195) / 2900.99, abs=1e-4)
    may = p[p.delivery_month == 5].iloc[0]
    assert may["rel_premium"] == pytest.approx((2506.25 - 590.90) / 2506.25, abs=1e-4)
    assert len(p) == 6 and (p["rel_premium"] > 0).all()        # VEP above realised every month


def test_look_ahead_guard_on_real_data(real_panel):
    vep, real = real_panel
    cutoff = pd.Timestamp("2026-04-30 21:00", tz="UTC")          # end of April delivery
    p1 = realized_premium_panel(vep, real, cutoff_utc=cutoff)
    assert sorted(p1["delivery_month"]) == [2, 3, 4]
    corrupted = real.copy()
    corrupted.loc[corrupted["delivery_end_utc"] > cutoff, "realized_TRY_MWh"] *= 10.0
    p2 = realized_premium_panel(vep, corrupted, cutoff_utc=cutoff)
    pd.testing.assert_frame_equal(p1, p2)
    early = realized_premium_panel(vep, real, cutoff_utc=pd.Timestamp("2026-01-15", tz="UTC"))
    assert early.empty                                           # nothing delivered yet


def test_fit_without_penalty_reproduces_raw_means_and_ridge_shrinks(real_panel):
    vep, real = real_panel
    p = realized_premium_panel(vep, real, max_tau=7)
    raw = fit_premium_curve(p, max_tau=7, smoothness=0.0, ridge=0.0)
    obs = ~np.isnan(raw.raw_mean)
    np.testing.assert_allclose(raw.rel_premium[obs], raw.raw_mean[obs], atol=1e-10)
    tight = fit_premium_curve(p, max_tau=7, smoothness=0.0, ridge=1e3)
    assert np.nanmax(np.abs(tight.rel_premium)) < 0.01
    assert any("prior-dominated" in n for n in raw.notes)          # one month per tau


# --------------------------------------------------------------------------
# pi(t) -> a(t): exact on the real VEP curve with the real 2026 premia
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [KAPPA_FAST, 0.0309, 4.108274e-06])
def test_drift_from_premium_maps_physical_mean_onto_forward(real_panel, real_curve, kappa):
    vep, real = real_panel
    p = realized_premium_panel(vep, real, max_tau=7)
    curve = fit_premium_curve(p, max_tau=7, smoothness=1.0, ridge=0.0)
    t, F = real_curve
    pi = premium_path_hours(t, F, curve)
    a = drift_from_premium(t, pi, kappa)
    mu_p = ou_mean_with_drift(t, -a, kappa, x0=-pi[0])
    assert np.max(np.abs(mu_p + pi)) < 0.5                      # TRY/MWh
    assert pi[0] == 0.0 and pi.max() > 100.0


def test_physical_mean_leaves_q_mean_untouched(real_curve):
    t, F = real_curve
    pi = 0.1 * F
    np.testing.assert_allclose(physical_mean(F, pi), 0.9 * F)


# --------------------------------------------------------------------------
# regime-dependent Q1 with the ESTIMATED fast-factor dynamics
# --------------------------------------------------------------------------
def test_stress_premium_uplift_matches_moment_ode():
    delta_a = 100.0
    spec0 = ResidualSpec(kappa_per_hour=KAPPA_FAST, sigma_y=np.array([1e-3, 2e-3]),
                         scale_P=282.48)
    spec1 = ResidualSpec(kappa_per_hour=KAPPA_FAST, sigma_y=np.array([1e-3, 2e-3]),
                         scale_P=282.48, drift_shift_per_hour=np.array([0.0, delta_a]))
    t = np.arange(0.0, 200.0, 0.5)
    n = t.size
    p_stat = np.array([Q10_REAL, Q01_REAL]) / (Q01_REAL + Q10_REAL)
    kw = dict(times_hours=t, q01=np.full(n, Q01_REAL), q10=np.full(n, Q10_REAL), pi0=p_stat)
    v0 = residual_moments(spec0, **kw).variance[-1]
    v1 = residual_moments(spec1, **kw).variance[-1]
    assert v0 == pytest.approx(0.0, abs=1e-9)
    assert v1 == pytest.approx(stress_premium_variance_uplift(delta_a, KAPPA_FAST,
                                                               Q01_REAL, Q10_REAL), rel=0.02)


def test_uplift_below_naive_formula_with_estimated_switching():
    q = Q01_REAL + Q10_REAL
    p0, p1 = Q10_REAL / q, Q01_REAL / q
    naive = p0 * p1 * (100.0 / KAPPA_FAST) ** 2
    assert stress_premium_variance_uplift(100.0, KAPPA_FAST, Q01_REAL, Q10_REAL) < naive


# --------------------------------------------------------------------------
# loaders on the real EPİAŞ files
# --------------------------------------------------------------------------
def test_epias_loader_on_realized_file(repo_root):
    s = load_epias_ptf_csv(repo_root / "inputs/market/realized_ptf_2026.csv")
    assert s.iloc[0] == pytest.approx(2799.98)                 # 31.12.2025 00:00 TRT
    assert str(s.index[0]) == "2025-12-30 21:00:00+00:00"
    mm = realized_monthly_means(s)
    feb = mm[(mm.delivery_year == 2026) & (mm.delivery_month == 2)]
    assert feb["realized_TRY_MWh"].iloc[0] == pytest.approx(2078.20, abs=0.05)
    assert feb["n_hours"].iloc[0] == 672


def test_epias_loader_on_downloaded_history(repo_root):
    f = repo_root / "inputs/historical/ptf_raw/ptf_2019.csv"
    if not f.exists():
        pytest.skip("ptf_raw not downloaded (scripts/data/download_epias.py)")
    s = load_epias_ptf_csv(f)
    assert len(s) == 8760 and s.isna().sum() == 0
    assert s.iloc[0] == pytest.approx(100.38)
    assert s.mean() == pytest.approx(260.32, abs=0.01)
