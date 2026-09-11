"""Tests for the re-estimated residual dynamics -- REAL EPİAŞ data only.

Uses inputs/historical/ptf_raw (EPİAŞ PTF 2019-2025), the repository's
rd_standardized.csv covariate and the accepted VEP forward curve.  The model is
estimated once on 2025 (8 760 h) to keep the suite fast; the production run
(scripts/residual/fit_residual_v2.py) uses 2023-2025.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pde_option_model.premium import load_epias_ptf_dir
from pde_option_model.residual_v2 import (ResidualModelV2, build_residual_panel,
                                          fit_fast_msar, fit_slow_daily, level_uncertainty,
                                          load_rd_covariates, mc_option_price, msar_negloglik,
                                          price_limits, simulate_prices)

VAL = pd.Timestamp("2025-12-31 20:00", tz="UTC")


@pytest.fixture(scope="module")
def ptf(repo_root):
    d = repo_root / "inputs/historical/ptf_raw"
    if not d.exists() or not list(d.glob("ptf_*.csv")):
        pytest.skip("ptf_raw not downloaded (scripts/data/download_epias.py)")
    s = load_epias_ptf_dir(d)
    return s[s.index <= VAL]


@pytest.fixture(scope="module")
def cov(repo_root):
    return load_rd_covariates(repo_root / "inputs/historical/rd_standardized.csv")


@pytest.fixture(scope="module")
def panel_2025(ptf, cov):
    p = build_residual_panel(ptf, 2025, 2025)
    return p.join(cov, how="left").dropna(subset=["u", "z1", "r1", "L"])


@pytest.fixture(scope="module")
def fast_2025(panel_2025):
    return fit_fast_msar(panel_2025["u"].to_numpy(), panel_2025["z1"].to_numpy(),
                         panel_2025["r1"].to_numpy(), with_se=False)


@pytest.fixture(scope="module")
def model_2025(ptf, panel_2025, fast_2025):
    slow = fit_slow_daily(panel_2025.groupby("day")["x"].mean())
    loc = ptf.copy()
    loc.index = ptf.index.tz_convert("Europe/Istanbul").tz_localize(None)
    L = float(loc.groupby(loc.index.to_period("M")).mean().iloc[-12:].mean())
    return ResidualModelV2(fast=fast_2025, slow=slow, level=level_uncertainty(ptf), scale_L=L)


def _covariate_paths(cov, times, n):
    """Real z / ramp of the same calendar hours one year earlier (2025), all paths."""
    prev = times - pd.DateOffset(years=1)
    sub = cov.reindex(prev).ffill().bfill()
    return (np.repeat(sub["z1"].to_numpy()[:, None], n, 1),
            np.repeat(sub["r1"].to_numpy()[:, None], n, 1))


# --------------------------------------------------------------------------
# estimation on EPİAŞ 2025
# --------------------------------------------------------------------------
def test_fast_factor_estimate_is_economically_sensible(fast_2025, panel_2025):
    p = fast_2025.params
    assert fast_2025.converged
    assert 0.5 < p["phi"] < 0.85                          # half-life 1-4 h
    assert fast_2025.sigma[1] > 1.5 * fast_2025.sigma[0]   # regime 1 = high vol
    assert p["g01"] < 0 < p["g10"]                         # high RD -> calmer regime
    assert 0.2 < fast_2025.stationary_stress_share < 0.8
    assert fast_2025.n_obs == len(panel_2025)


def test_fast_factor_optimum_is_identified(panel_2025, fast_2025):
    """Re-starting from a different point (labels swapped) reaches the same optimum."""
    u, Z, R = (panel_2025[c].to_numpy() for c in ("u", "z1", "r1"))
    swapped = fast_2025.theta[[1, 0, 2, 4, 3, 8, 9, 10, 5, 6, 7]]
    refit = fit_fast_msar(u, Z, R, theta0=swapped, with_se=False)
    assert refit.sigma[1] > refit.sigma[0]
    assert refit.loglik == pytest.approx(fast_2025.loglik, abs=1.0)
    assert msar_negloglik(fast_2025.theta, u, Z, R) < msar_negloglik(
        fast_2025.theta * np.r_[1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1], u, Z, R)


def test_slow_factor_on_real_daily_means(panel_2025):
    s = fit_slow_daily(panel_2025.groupby("day")["x"].mean())
    assert 0.2 < s["phi_daily"] < 0.8 and s["n_days"] >= 360
    assert 0.3 < s["half_life_days"] < 3.0


def test_panel_shape_is_ex_ante(ptf):
    a = build_residual_panel(ptf, 2025, 2025)
    y25 = ptf.index >= pd.Timestamp("2025-01-01", tz="Europe/Istanbul")
    corrupted = ptf.copy()
    corrupted[y25] = corrupted[y25] * 3.0
    b = build_residual_panel(corrupted, 2025, 2025)
    np.testing.assert_allclose(a["S"].to_numpy(), b["S"].to_numpy())
    assert not np.allclose(a["P"].to_numpy(), b["P"].to_numpy())


def test_level_uncertainty_grows_with_horizon(ptf):
    lv = level_uncertainty(ptf)
    assert (np.diff(lv["sigma_log"].to_numpy()) > 0).all()


# --------------------------------------------------------------------------
# regulatory limits and Monte Carlo on the real VEP curve
# --------------------------------------------------------------------------
def test_price_limits_schedule():
    t = pd.DatetimeIndex(["2026-01-15 12:00", "2026-04-03 20:00", "2026-04-04 12:00"],
                         tz="UTC")
    np.testing.assert_array_equal(price_limits(t), [3400.0, 3400.0, 4500.0])


def test_simulated_prices_respect_floor_and_cap(model_2025, cov, repo_root):
    curve = pd.read_csv(repo_root / "outputs/market_calibration_final/hourly_forward_curve.csv")
    t = pd.DatetimeIndex(pd.to_datetime(curve["time_utc"], utc=True))
    m = (t >= pd.Timestamp("2026-03-29", tz="UTC")) & (t < pd.Timestamp("2026-04-08", tz="UTC"))
    times, F = t[m], curve["hourly_forward_TRY_MWh"].to_numpy()[m]
    Z, R = _covariate_paths(cov, times, 400)
    sim = simulate_prices(model_2025, times, F, VAL, Z, R, n_paths=400, keep_paths=True)
    p, cap = sim["paths"], sim["cap"]
    assert p.min() >= 0.0 and np.all(p.max(axis=1) <= cap + 1e-6)
    assert (cap == 3400.0).any() and (cap == 4500.0).any()


def test_option_prices_satisfy_no_arbitrage_bounds(model_2025, cov, repo_root):
    curve = pd.read_csv(repo_root / "outputs/market_calibration_final/hourly_forward_curve.csv")
    t = pd.DatetimeIndex(pd.to_datetime(curve["time_utc"], utc=True))
    m = (t > VAL) & (t <= VAL + pd.Timedelta(hours=72))
    times, F = t[m], curve["hourly_forward_TRY_MWh"].to_numpy()[m]
    Z, R = _covariate_paths(cov, times, 4000)
    sim = simulate_prices(model_2025, times, F, VAL, Z, R, n_paths=4000, keep_paths=True, seed=3)
    PT = sim["paths"][-1].astype(float)
    strikes = (2000, 2600, 3000, 3400, 4000)
    calls = [mc_option_price(PT, K, 72)["value"] for K in strikes]
    df = math.exp(-0.40 * 72 / 8760)
    for K, c in zip(strikes, calls):
        assert 0.0 <= c <= df * max(3400.0 - K, 0.0) + 1e-9
    assert all(a >= b for a, b in zip(calls, calls[1:]))
    assert mc_option_price(PT, 3000, 72, kind="put")["value"] <= df * 3000.0


def test_continuous_time_mapping(model_2025):
    c = model_2025.continuous()
    phi = model_2025.fast.params["phi"]
    assert c["half_life_fast_h"] == pytest.approx(math.log(2) / -math.log(phi))
    assert c["half_life_slow_h"] == pytest.approx(
        24 * math.log(2) / -math.log(model_2025.slow["phi_daily"]))
    assert c["sigma_fast_stress_TRY"] > c["sigma_fast_normal_TRY"] > 0
