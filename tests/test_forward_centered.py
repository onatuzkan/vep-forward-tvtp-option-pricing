"""Forward-centered model: centering, finite moments, PDE pricing, parity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pde_option_model.contracts import EuropeanOption
from pde_option_model.forward_centered import (ForwardCenteredError,
                                               ResidualGridSettings, ResidualSpec,
                                               price_forward_centered,
                                               residual_moments,
                                               simulate_forward_centered)

HZ = np.array([72.0, 168.0, 336.0, 720.0])


# --------------------------------------------------------------------------
# centering
# --------------------------------------------------------------------------
def test_residual_expectation_is_exactly_zero(model):
    t = np.arange(0.0, 721.0, 1.0)
    mom = model.moments(t)
    assert np.max(np.abs(mom.mean)) < 1e-9


def test_expected_spot_equals_the_market_forward(model):
    t = np.arange(0.0, 721.0, 1.0)
    s = model.residual_summary(t)
    dev = np.max(np.abs(s["expected_spot_TRY_MWh"] - s["forward_TRY_MWh"]))
    assert dev < 1e-9


def test_monthly_average_of_expected_spot_equals_the_vep_quote(model, quotes):
    """The acceptance identity, evaluated over true UTC delivery hours."""
    for q in quotes.quotes:
        hrs = q.delivery.hours_utc()
        h = np.asarray((hrs - model.valuation_utc).total_seconds(), float) / 3600.0
        avg = float(np.mean(model.expected_spot(h)))
        assert abs(avg - q.price_TRY_MWh) < 0.10, q.contract_name
        assert abs(avg - q.price_TRY_MWh) < 1e-6, q.contract_name


def test_centering_works_with_nonzero_regime_means(curve_smooth, params, make_model):
    """E[P_t] = F(t) must hold even when the conditional regime means are non-zero.

    Under the M9-CSV-reconciled kappa (~4.1e-6/h, half-life ~19 years) the
    residual mean moves much more slowly than under the previous fallback
    kappa; the "non-zero E[X] before centering" check is therefore run on
    a longer 5000 h horizon so the drift toward the regime means is
    observable.  The centering invariant itself is horizon-independent.
    """
    m = make_model(curve_smooth, params, regime_means=(150.0, -400.0))
    t = np.arange(0.0, 5001.0, 5.0)
    mom = m.moments(t)
    assert np.max(np.abs(mom.mean)) > 0.1, "regime means should move E[X] before centering"
    s = m.residual_summary(t)
    assert np.max(np.abs(s["expected_spot_TRY_MWh"] - s["forward_TRY_MWh"])) < 1e-8


def test_centering_works_with_nonzero_initial_residual(curve_smooth, params, make_model):
    m = make_model(curve_smooth, params, x0_mode="spot_minus_curve")
    t = np.arange(0.0, 241.0, 1.0)
    s = m.residual_summary(t)
    assert np.max(np.abs(s["expected_spot_TRY_MWh"] - s["forward_TRY_MWh"])) < 1e-8


def test_monthly_fit_is_invariant_to_regime_volatility(curve_smooth, params,
                                                       make_model, quotes):
    """mu_X(t) contains no sigma term, so volatility cannot break the fit."""
    q = quotes.get(2026, 4)
    hrs = q.delivery.hours_utc()
    for mult in ((1.0, 1.0), (0.2, 0.2), (5.0, 5.0)):
        m = make_model(curve_smooth, params)
        m.spec.sigma_multipliers = np.asarray(mult, dtype=float)
        h = np.asarray((hrs - m.valuation_utc).total_seconds(), float) / 3600.0
        assert abs(float(np.mean(m.expected_spot(h))) - q.price_TRY_MWh) < 1e-6


def test_multiplicative_mode_is_also_centred(curve_smooth, params, make_model):
    m = make_model(curve_smooth, params, mode="multiplicative")
    t = np.arange(0.0, 169.0, 1.0)
    s = m.residual_summary(t)
    assert np.max(np.abs(s["expected_spot_TRY_MWh"] - s["forward_TRY_MWh"])) < 1e-6


@pytest.mark.parametrize("a0,a1", [
    (0.01, -0.02),          # asymmetric small
    (0.05, 0.05),           # symmetric positive
    (-0.03, 0.02),          # asymmetric mixed-sign
])
def test_q1_drift_shift_preserves_centering(curve_smooth, params, make_model,
                                            a0, a1):
    """A non-zero Q1 drift shift must NOT break E^Q[P_t] = F(t).

    The drift shift enters both the moment ODE (via + a_i p_i on u_i and
    + 2 a_i u_i on w_i) and the pricing PDE drift (via + a_i in
    price_forward_centered.drift_fn) symmetrically, so mu_X(t) absorbs it
    exactly and the residual mean cancels in F(t) + X_t - mu_X(t).
    Also verifies that the mean residual E[X_t] is meaningfully non-zero
    before centering (otherwise the test would be vacuous).
    """
    m = make_model(curve_smooth, params,
                   drift_shift_per_hour=(a0, a1))
    t = np.arange(0.0, 721.0, 1.0)
    mom = m.moments(t)
    assert np.max(np.abs(mom.mean)) > 1e-3, (
        "drift shift should move E[X] meaningfully before centering")
    s = m.residual_summary(t)
    assert np.max(np.abs(s["expected_spot_TRY_MWh"]
                         - s["forward_TRY_MWh"])) < 1e-8


# --------------------------------------------------------------------------
# finite moments
# --------------------------------------------------------------------------
def test_forward_centered_moments_are_finite_at_all_horizons(model):
    t = np.arange(0.0, 5001.0, 5.0)
    mom = model.moments(t)
    assert np.all(np.isfinite(mom.mean))
    assert np.all(np.isfinite(mom.variance))
    assert np.all(mom.variance >= 0.0)


def test_residual_variance_grows_at_most_linearly(model):
    """No exp(v) inflation: variance is bounded by sigma_bar^2 * t."""
    t = np.arange(0.0, 721.0, 1.0)
    v = model.moments(t).variance
    sigma_max = float(np.max(model.sigma_path(t)))
    assert np.all(v <= sigma_max ** 2 * t + 1e-6)


def test_expected_spot_never_reaches_implausible_magnitudes(model):
    v = model.expected_spot(np.arange(1.0, 5001.0, 10.0))
    assert np.all(np.isfinite(v))
    assert np.max(np.abs(v)) < 1.0e5


def test_regime_probabilities_stay_on_the_simplex(model):
    mom = model.moments(np.arange(0.0, 2001.0, 1.0))
    assert np.max(np.abs(mom.p.sum(axis=0) - 1.0)) < 1e-10
    assert np.all(mom.p >= -1e-12)


def test_moments_reject_a_grid_that_does_not_start_at_zero(model):
    with pytest.raises(ForwardCenteredError, match="starting at t=0"):
        model.moments(np.array([10.0, 20.0, 30.0]))


def test_moment_ode_matches_the_analytic_single_regime_solution():
    """m_i = m, x0 != 0 -> E[X_t] = m + (x0 - m) e^{-kappa t} exactly."""
    spec = ResidualSpec(kappa_per_hour=0.01, sigma_y=np.array([0.01, 0.2]),
                        scale_P=282.48, regime_means=np.array([50.0, 50.0]))
    t = np.linspace(0.0, 200.0, 401)
    q = np.full(t.size, 0.02)
    mom = residual_moments(spec, t, q, q, np.array([0.5, 0.5]), x0=500.0)
    exact = 50.0 + (500.0 - 50.0) * np.exp(-0.01 * t)
    assert np.max(np.abs(mom.mean - exact)) < 1e-8


def test_moment_ode_matches_analytic_variance_single_regime():
    spec = ResidualSpec(kappa_per_hour=0.01, sigma_y=np.array([1.0, 1.0]),
                        scale_P=0.0 + 1.0)
    t = np.linspace(0.0, 100.0, 501)
    q = np.full(t.size, 0.0)
    sig = np.ones((2, t.size)) * 2.0
    mom = residual_moments(spec, t, q, q, np.array([0.5, 0.5]), x0=0.0,
                           sigma_price_path=sig)
    exact = 4.0 * (1.0 - np.exp(-2 * 0.01 * t)) / (2 * 0.01)
    assert np.max(np.abs(mom.variance - exact)) < 1e-6


# --------------------------------------------------------------------------
# option pricing
# --------------------------------------------------------------------------
def test_pde_matches_monte_carlo_at_short_horizon(model, params, fast_grid):
    T = params.valuation_utc + pd.Timedelta(hours=72)
    for otype in ("call", "put"):
        c = EuropeanOption(otype, 3000.0, params.valuation_utc, T, 0.40)
        pde = price_forward_centered(model, c, fast_grid)
        mc = simulate_forward_centered(model, c, n_paths=40_000, seed=11)
        z = abs(pde.value - mc["value"]) / max(mc["std_error"], 1e-12)
        assert z < 3.5, f"{otype}: PDE {pde.value} vs MC {mc['value']}+-{mc['std_error']}"


def test_monte_carlo_reports_seed_and_standard_error(model, params):
    T = params.valuation_utc + pd.Timedelta(hours=48)
    mc = simulate_forward_centered(
        model, EuropeanOption("call", 3000.0, params.valuation_utc, T, 0.40),
        n_paths=20_000, seed=4242)
    assert mc["seed"] == 4242 and mc["std_error"] > 0 and mc["n_paths"] == 20_000
    same = simulate_forward_centered(
        model, EuropeanOption("call", 3000.0, params.valuation_utc, T, 0.40),
        n_paths=20_000, seed=4242)
    assert mc["value"] == same["value"], "seeded Monte Carlo must be reproducible"


def test_monte_carlo_mean_price_matches_the_forward(model, params):
    T = params.valuation_utc + pd.Timedelta(hours=72)
    mc = simulate_forward_centered(
        model, EuropeanOption("call", 3000.0, params.valuation_utc, T, 0.40),
        n_paths=60_000, seed=7)
    z = abs(mc["mean_price_T"] - mc["analytic_expected_spot_T"]) / mc["mean_price_T_se"]
    assert z < 3.5


def test_call_is_decreasing_in_strike(model, params, fast_grid):
    T = params.valuation_utc + pd.Timedelta(hours=72)
    vals = [price_forward_centered(
        model, EuropeanOption("call", K, params.valuation_utc, T, 0.40),
        fast_grid).value for K in (2000., 2500., 3000., 3500., 4000.)]
    assert np.all(np.diff(vals) < 0)
    assert np.all(np.array(vals) >= -1e-8)


def test_put_is_increasing_in_strike(model, params, fast_grid):
    T = params.valuation_utc + pd.Timedelta(hours=72)
    vals = [price_forward_centered(
        model, EuropeanOption("put", K, params.valuation_utc, T, 0.40),
        fast_grid).value for K in (2000., 2500., 3000., 3500., 4000.)]
    assert np.all(np.diff(vals) > 0)
    assert np.all(np.array(vals) >= -1e-8)


def test_put_call_parity(model, params, fast_grid):
    """C - P = e^{-r tau} (F(T) - K) exactly, since E^Q[P_T] = F(T)."""
    T = params.valuation_utc + pd.Timedelta(hours=72)
    disc = float(np.exp(-0.40 / 8760.0 * 72.0))
    for K in (2000., 2500., 3000., 3500., 4000.):
        c = price_forward_centered(
            model, EuropeanOption("call", K, params.valuation_utc, T, 0.40), fast_grid)
        p = price_forward_centered(
            model, EuropeanOption("put", K, params.valuation_utc, T, 0.40), fast_grid)
        assert abs((c.value - p.value) - disc * (c.forward_at_expiry - K)) < 1e-4


def test_payoff_uses_the_price_not_the_residual_state(model, params, fast_grid):
    """A deep-OTM strike above the whole price grid must be worthless."""
    T = params.valuation_utc + pd.Timedelta(hours=24)
    v = price_forward_centered(
        model, EuropeanOption("call", 5.0e4, params.valuation_utc, T, 0.40),
        fast_grid).value
    assert 0.0 <= v < 1e-3


def test_price_from_state_maps_residual_to_price(model):
    x = np.array([-500.0, 0.0, 500.0])
    p = model.price_from_state(x, forward_level=2900.0, centering=0.0)
    assert np.allclose(p, np.array([2400.0, 2900.0, 3400.0]))


def test_option_is_on_the_expiry_hour_spot_not_a_monthly_average(model, params,
                                                                 fast_grid, quotes):
    """Documented contract semantics: a single expiry hour, not a baseload strip."""
    q = quotes.get(2026, 2)
    T = q.delivery_start_utc + pd.Timedelta(hours=360)
    res = price_forward_centered(
        model, EuropeanOption("call", 2900.99, params.valuation_utc, T, 0.40),
        fast_grid)
    f_hour = float(model.curve.values.reindex([T]).iloc[0])
    assert abs(res.forward_at_expiry - f_hour) < 1e-6
    assert abs(res.forward_at_expiry - q.price_TRY_MWh) > 1e-6


def test_grid_boundary_guard(model, params):
    T = params.valuation_utc + pd.Timedelta(hours=72)
    tight = ResidualGridSettings(n_space_nodes=101, x_min=1.0e5, x_max=1.1e5)
    with pytest.raises(ForwardCenteredError, match="too close to the residual grid"):
        price_forward_centered(model, EuropeanOption(
            "call", 3000.0, params.valuation_utc, T, 0.40), tight)


def test_spec_rejects_inverted_regime_volatilities():
    with pytest.raises(ForwardCenteredError):
        ResidualSpec(kappa_per_hour=1e-4, sigma_y=np.array([-1.0, 0.1]), scale_P=282.48)


def test_sigma_transfer_preserves_the_regime_ratio(model, params):
    sig = model.spec.sigma_price(np.array([2500.0, 3500.0]))
    ratio = sig[1] / sig[0]
    expected = params.sigma_stress / params.sigma_normal
    assert np.allclose(ratio, expected)
    assert sig[0][1] > sig[0][0], "higher forward level -> higher price-space vol"
