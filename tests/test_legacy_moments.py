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


def test_stress_regime_stationary_inflation_is_catastrophic(params):
    """The core regression: sigma_stress with a near-unit-root kappa explodes."""
    f_normal = stationary_inflation_factor(params.sigma_normal, params.kappa_per_hour)
    f_stress = stationary_inflation_factor(params.sigma_stress, params.kappa_per_hour)
    assert f_normal < 1.1, "the normal regime alone is harmless"
    assert f_stress > 1.0e6, "the stress regime must be flagged as explosive"
    assert stationary_variance(params.sigma_stress, params.kappa_per_hour) > 30.0


def test_variance_doubling_time_is_short_relative_to_a_month(params):
    pis = params.stationary_pi_stress
    sbar = np.sqrt((1 - pis) * params.sigma_normal ** 2 + pis * params.sigma_stress ** 2)
    assert variance_doubling_time_hours(float(sbar)) < 200.0


def test_legacy_expected_spot_grows_super_linearly(params):
    y0 = float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P))
    h = np.array([72.0, 168.0, 336.0, 720.0])
    with np.errstate(over="ignore"):
        f = legacy_expected_spot(h, params.scale_P, y0,
                                 params.legacy_theta_effective,
                                 params.kappa_per_hour, params.sigma_y,
                                 params.stationary_pi_stress)
    assert np.all(np.diff(f) > 0)
    assert f[-1] / params.spot_price_TRY_MWh > 20.0, "720h must exceed 20x spot"
    growth = f[1:] / f[:-1]
    assert np.all(growth > 1.5)


def test_legacy_reconstruction_matches_the_reported_outputs(params):
    """The analytic diagnosis reproduces the reported legacy numbers within 5%."""
    ref = load_legacy_reference(LEGACY_REF)
    y0 = float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P))
    with np.errstate(over="ignore"):
        f = legacy_expected_spot(ref["horizon_hours"].to_numpy(float),
                                 params.scale_P, y0, params.legacy_theta_effective,
                                 params.kappa_per_hour, params.sigma_y,
                                 params.stationary_pi_stress)
    rel = np.abs(f / ref["expected_spot_TRY_MWh"].to_numpy(float) - 1.0)
    assert rel.max() < 0.05, dict(zip(ref["horizon_hours"], rel))


def test_legacy_explosion_report_flags_the_problem(params):
    rep = legacy_explosion_report(
        params.scale_P, float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P)),
        params.kappa_per_hour, params.sigma_y, float(params.pi_filtered[1]),
        pi_stationary_stress=params.stationary_pi_stress)
    assert rep["stationary_inflation_stress"] > 1e6
    assert rep["warning"] == EXPLOSION_WARNING
    assert "exp(v(t)/2)" in EXPLOSION_WARNING
    assert rep["table"]["inflation_factor_exp_v_over_2"].is_monotonic_increasing


def test_a_single_drift_shift_cannot_repair_the_variance_channel(params):
    """Moving theta rescales sinh(m); it cannot touch the e^{v/2} factor."""
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
    assert inflation > 100.0
    assert all(np.isfinite(r) for r in ratios)
    # every theta inherits the SAME inflation factor
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
