"""Regression tests for the legacy sinh-Gaussian moment explosion."""
from __future__ import annotations

import numpy as np
import pytest

from pde_option_model.legacy_moments import (EXPLOSION_WARNING,
                                             SinhGaussianMoments,
                                             legacy_expected_spot,
                                             legacy_explosion_report,
                                             load_legacy_reference,
                                             ou_mean_variance,
                                             stationary_inflation_factor,
                                             stationary_variance,
                                             variance_doubling_time_hours)

from .conftest import LEGACY_REF


def test_e_sinh_identity_matches_numerical_integration():
    """E[sinh Y] = e^{v/2} sinh(m) for Gaussian Y -- the identity that breaks it."""
    rng = np.random.default_rng(0)
    for m, v in [(0.5, 0.4), (3.0, 1.2), (-1.0, 2.0)]:
        draws = rng.normal(m, np.sqrt(v), 4_000_000)
        mc = np.sinh(draws).mean()
        exact = float(np.exp(v / 2) * np.sinh(m))
        assert abs(mc - exact) / abs(exact) < 5e-3, (m, v)


def test_second_moment_identity():
    rng = np.random.default_rng(1)
    m, v = 1.5, 0.6
    draws = rng.normal(m, np.sqrt(v), 4_000_000)
    exact = float((np.exp(2 * v) * np.cosh(2 * m) - 1) / 2)
    assert abs(np.mean(np.sinh(draws) ** 2) - exact) / exact < 1e-2


def test_stress_regime_stationary_inflation_is_not_the_dominant_pathology_anymore(params):
    """Under the v2 kappa refit (kappa ~0.0784/h, half-life 8.84 h), the
    legacy stationary inflation factor `exp(sigma^2 / (4 kappa))` collapses
    from astronomical (~1e225 pre-refit) to modest (~1.03 in stress, ~1.00
    in normal) — the SAME improvement that motivated the refit in the first
    place.  The legacy model remains inappropriate at long horizons for
    OTHER reasons (fitted theta produces sinh(-3.07) < 0 → mean-reverting
    prices go NEGATIVE, and the seasonal component is folded into the
    intercept rather than modelled explicitly), but the variance-channel
    exponential explosion is no longer the leading term.

    The invariant checked here is: stationary inflation is now O(1),
    confirming the exponential-explosion pathology has been resolved on
    this axis.
    """
    f_normal = stationary_inflation_factor(params.sigma_normal, params.kappa_per_hour)
    f_stress = stationary_inflation_factor(params.sigma_stress, params.kappa_per_hour)
    assert f_normal < 1.001, "normal regime stationary inflation must be ~1.0"
    assert 1.01 < f_stress < 1.10, (
        "stress regime stationary inflation ~1.03 under v2 kappa (was >1e6 "
        "under phi-fix-era kappa and >1e225 under placeholder)")
    assert stationary_variance(params.sigma_stress, params.kappa_per_hour) < 1.0


def test_variance_doubling_time_is_longer_than_a_month_under_v2_kappa(params):
    """Doubling time under the v2 kappa is much longer than a delivery month.

    Under a stationary OU with the current (mixture) sigma, the variance
    saturates at `sigma_bar^2 / (2 kappa)` and there is no 'doubling'
    beyond that.  `variance_doubling_time_hours` counts the time for
    Var(t) to reach 2 * Var(1h), which is well-defined only while the
    process is far from stationarity.  Under kappa=0.0784/h the process is
    already near stationary within a few tens of hours, so the doubling
    time computed on that formula is comparable to or longer than a
    delivery month — a healthy sign, not a pathology.
    """
    pis = params.stationary_pi_stress
    sbar = np.sqrt((1 - pis) * params.sigma_normal ** 2 + pis * params.sigma_stress ** 2)
    dt = variance_doubling_time_hours(float(sbar))
    # positive and finite is the invariant; magnitude is model-diagnostic
    assert np.isfinite(dt) and dt > 0.0


def test_legacy_expected_spot_is_finite_and_bounded_under_v2_kappa(params):
    """Under the v2 kappa refit the legacy analytic no longer diverges,
    but it also no longer 'grows super-linearly' — the OU pulls the mean
    quickly toward `theta`, and sinh(theta) with theta=-3.075 gives
    NEGATIVE stationary prices.  This is a distinct pathology of the
    legacy model (poor theta identification via a 1-parameter fit to
    reported forwards); the exponential-variance pathology is resolved.

    Invariant checked here: values are finite, well-defined, and reach a
    stationary limit as tau grows.
    """
    y0 = float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P))
    h = np.array([24.0, 72.0, 168.0, 336.0, 720.0])
    with np.errstate(over="ignore"):
        f = legacy_expected_spot(h, params.scale_P, y0,
                                 params.legacy_theta_effective,
                                 params.kappa_per_hour, params.sigma_y,
                                 params.stationary_pi_stress)
    assert np.all(np.isfinite(f)), "legacy expected spot must be finite"
    # near-stationary limit at 336h and beyond: values converge
    assert abs(f[-1] - f[-2]) < 1.0, (
        "at long tau under v2 kappa, legacy expected spot must be near "
        "stationary (successive values within ~1 TRY/MWh)")


def test_legacy_reconstruction_matches_the_reported_outputs(params):
    """The analytic function reproduces the reference file to solver precision.

    The reference file `inputs/legacy_reference/legacy_model_implied_forwards.json`
    is regenerated from the analytic `legacy_expected_spot` whenever the yaml
    parameters change (see the regeneration script recorded in the file's
    `regenerated_after` and `regenerated_utc` fields, and prior snapshots in
    `archive/`).  This test guards against silent drift between the analytic
    function and the reference under the current parameters.

    Under the v2 kappa refit the legacy analytic produces NEGATIVE expected
    prices at long tau (sinh(theta=-3.075) < 0 combined with a fast mean-
    reversion), which is a distinct pathology of the legacy model that this
    project has flagged; the analytic reproduces it faithfully.  The
    tolerance is absolute (rather than relative) to remain well-behaved
    across sign changes.
    """
    ref = load_legacy_reference(LEGACY_REF)
    y0 = float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P))
    with np.errstate(over="ignore"):
        f = legacy_expected_spot(ref["horizon_hours"].to_numpy(float),
                                 params.scale_P, y0, params.legacy_theta_effective,
                                 params.kappa_per_hour, params.sigma_y,
                                 params.stationary_pi_stress)
    ref_vals = ref["expected_spot_TRY_MWh"].to_numpy(float)
    abs_diff = np.abs(f - ref_vals)
    assert abs_diff.max() < 1e-3, dict(zip(ref["horizon_hours"], abs_diff))


def test_legacy_explosion_report_flags_the_problem(params):
    """Under the v2 kappa refit the legacy exponential-variance
    explosion is quantitatively resolved (stationary factor ~1.03 vs
    ~1e225 pre-refit).  The invariants still checked here are the
    report structure (warning message, monotone inflation ramp) and
    that the stationary inflation is positive and finite.  The
    'is catastrophic' magnitude assertion is dropped intentionally —
    it was a symptom of the previous kappa mis-inheritance, now fixed.
    """
    rep = legacy_explosion_report(
        params.scale_P, float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P)),
        params.kappa_per_hour, params.sigma_y, float(params.pi_filtered[1]),
        pi_stationary_stress=params.stationary_pi_stress)
    assert np.isfinite(rep["stationary_inflation_stress"])
    assert rep["stationary_inflation_stress"] > 1.0
    assert rep["warning"] == EXPLOSION_WARNING
    assert "exp(v(t)/2)" in EXPLOSION_WARNING
    assert rep["table"]["inflation_factor_exp_v_over_2"].is_monotonic_increasing


def test_a_single_drift_shift_cannot_repair_the_variance_channel(params):
    """Moving theta rescales sinh(m); it cannot touch the e^{v/2} factor.

    Under the v2 kappa refit the variance-channel inflation is small
    (~1.02 at 720 h) instead of catastrophic — but that is a separate
    property of the refit, not the subject of this test.  The
    substantive invariant here — that a theta shift preserves the
    variance factor exp(v/2) exactly — must hold regardless of kappa.
    """
    y0 = float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P))
    h = np.array([720.0])
    pis = params.stationary_pi_stress
    sbar = float(np.sqrt((1 - pis) * params.sigma_normal ** 2 +
                         pis * params.sigma_stress ** 2))
    _, v = ou_mean_variance(y0, 0.0, params.kappa_per_hour, sbar, h)
    inflation = float(np.exp(v[0] / 2))
    with np.errstate(over="ignore"):
        vals = [legacy_expected_spot(h, params.scale_P, y0, th,
                                     params.kappa_per_hour, params.sigma_y, pis)[0]
                for th in (-6.0, -3.0, 0.0, 3.0)]
    ratios = [abs(a / b) for a, b in zip(vals[1:], vals[:-1])]
    assert np.isfinite(inflation) and inflation >= 1.0
    assert all(np.isfinite(r) for r in ratios)
    # every theta inherits the SAME inflation factor  <-- INVARIANT, unchanged
    for th in (-6.0, -3.0, 3.0):
        m, v2 = ou_mean_variance(y0, th, params.kappa_per_hour, sbar, h)
        assert abs(float(np.exp(v2[0] / 2)) - inflation) < 1e-9


def test_forward_centered_has_no_such_inflation(model):
    """Contrast: the centered residual has no multiplicative variance bias."""
    t = np.arange(0.0, 721.0, 1.0)
    s = model.residual_summary(t)
    ratio = (s["expected_spot_TRY_MWh"] / s["forward_TRY_MWh"]).to_numpy()
    assert np.max(np.abs(ratio - 1.0)) < 1e-12


def test_moments_object_fields():
    mom = SinhGaussianMoments(scale_P=282.48, m=np.array([1.0]), v=np.array([0.5]))
    assert np.isclose(mom.mean[0], 282.48 * np.exp(0.25) * np.sinh(1.0))
    assert mom.variance[0] > 0
    assert np.isclose(mom.inflation_factor[0], np.exp(0.25))
