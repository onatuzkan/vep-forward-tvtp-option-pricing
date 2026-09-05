"""The legacy stack must keep working: imports, invariants, solver compatibility."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pde_option_model import (boundaries, contracts, dynamics, forward_calibration,
                              generator, grid, risk_neutral, scenarios, solver,
                              transformations)


def test_every_legacy_module_still_imports():
    for mod in (boundaries, contracts, dynamics, forward_calibration, generator,
                grid, risk_neutral, scenarios, solver, transformations):
        assert mod is not None


def test_price_transform_roundtrip():
    tr = transformations.PriceTransform(scale_P=282.48)
    p = np.array([100.0, 2917.78, 9000.0])
    assert np.allclose(tr.price_from_y(tr.y_from_price(p)), p)
    assert np.isclose(float(tr.y_from_price(2917.78)), np.arcsinh(2917.78 / 282.48))


def test_ar_to_ou_mapping_matches_the_frozen_parameters(params):
    ou = transformations.ar_to_ou(params.phi, (1e-3, -1e-3), (0.0075, 0.1730))
    assert abs(ou.kappa_per_hour + np.log(params.phi)) < 1e-12
    assert abs(ou.half_life_hours - 1800.0) < 5.0


def test_generator_matrix_log_roundtrip():
    p01 = np.array([0.002, 0.01, 0.05])
    p10 = np.array([0.05, 0.08, 0.2])
    gen = generator.probs_to_generator(p01, p10)
    b01, b10 = generator.generator_to_probs(gen.q01, gen.q10)
    assert np.allclose(b01, p01) and np.allclose(b10, p10)
    assert gen.n_clipped == 0
    assert np.allclose(gen.q_matrix.sum(axis=2), 0.0, atol=1e-12)


def test_generator_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        generator.probs_to_generator(np.array([1.5]), np.array([0.1]))


def test_stationary_distribution():
    pi = generator.stationary_distribution(0.01, 0.09)
    assert np.isclose(pi.sum(), 1.0) and np.isclose(pi[1], 0.1)


def test_european_option_validation():
    v = pd.Timestamp("2025-12-31T20:00:00Z")
    c = contracts.EuropeanOption("call", 3000.0, v, v + pd.Timedelta(hours=72), 0.40)
    assert c.tau_hours == 72.0
    assert np.isclose(c.r_per_hour, 0.40 / 8760.0)
    assert np.allclose(c.payoff_from_price(np.array([2000.0, 3500.0])), [0.0, 500.0])
    with pytest.raises(ValueError):
        contracts.EuropeanOption("straddle", 3000.0, v, v + pd.Timedelta(hours=1), 0.4)
    with pytest.raises(ValueError):
        contracts.EuropeanOption("call", -1.0, v, v + pd.Timedelta(hours=1), 0.4)


def test_space_and_time_grids():
    g = grid.SpaceGrid(-5.0, 5.0, 101)
    assert g.y.size == 101 and np.isclose(g.h, 0.1)
    assert g.contains(0.0) and not g.contains(4.99)
    v = pd.Timestamp("2025-12-31T20:00:00Z")
    tg = grid.TimeGrid(v, v + pd.Timedelta(hours=72), 144)
    assert tg.tau_hours == 72.0 and np.isclose(tg.dt_hours, 0.5)
    with pytest.raises(ValueError):
        grid.TimeGrid(v, v - pd.Timedelta(hours=1), 10)


def test_risk_neutral_specs():
    adj = risk_neutral.baseline_q1(drift_shift_per_hour=(-0.004, -0.004))
    assert adj.spec == "Q1" and not adj.calibrated
    assert "NOT calibrated" in adj.label()
    q2 = risk_neutral.extended_q2(eta=(0.5, -0.5))
    a, b = q2.adjust_generator(np.array([0.01]), np.array([0.05]))
    assert np.isclose(a[0], 0.01 * np.exp(0.5))
    with pytest.raises(ValueError):
        risk_neutral.MeasureAdjustment(spec="Q1", eta=np.array([0.3, 0.0]))


def test_solver_is_backward_compatible_without_sigma_fn():
    """The legacy call signature must behave exactly as before."""
    g = grid.SpaceGrid(-4.0, 4.0, 201)
    v = pd.Timestamp("2025-12-31T20:00:00Z")
    tg = grid.TimeGrid(v, v + pd.Timedelta(hours=24), 48)
    payoff = np.maximum(g.y, 0.0)
    term = np.vstack([payoff, payoff])
    res = solver.solve_coupled_pde(
        g, tg, term, lambda t: np.zeros((2, g.y.size)), np.array([0.1, 0.1]),
        0.0, lambda t: (0.0, 0.0))
    assert res.V.shape == (2, 201) and np.all(np.isfinite(res.V))
    assert np.allclose(res.V[0], res.V[1])


def test_solver_sigma_fn_reproduces_constant_sigma():
    """sigma_fn returning a constant must equal the constant-sigma path exactly."""
    g = grid.SpaceGrid(-4.0, 4.0, 201)
    v = pd.Timestamp("2025-12-31T20:00:00Z")
    tg = grid.TimeGrid(v, v + pd.Timedelta(hours=24), 48)
    payoff = np.maximum(g.y, 0.0)
    term = np.vstack([payoff, payoff])
    sig = np.array([0.1, 0.3])
    kw = dict(drift_fn=lambda t: np.zeros((2, g.y.size)), sigma=sig,
              r_per_hour=0.0, generator_fn=lambda t: (0.01, 0.05))
    a = solver.solve_coupled_pde(g, tg, term, **kw)
    b = solver.solve_coupled_pde(g, tg, term, **kw, sigma_fn=lambda t: sig)
    assert np.allclose(a.V, b.V, atol=1e-12)


def test_solver_sigma_fn_is_actually_used():
    g = grid.SpaceGrid(-4.0, 4.0, 201)
    v = pd.Timestamp("2025-12-31T20:00:00Z")
    tg = grid.TimeGrid(v, v + pd.Timedelta(hours=24), 48)
    payoff = np.maximum(g.y, 0.0) ** 2
    term = np.vstack([payoff, payoff])
    kw = dict(drift_fn=lambda t: np.zeros((2, g.y.size)), sigma=np.array([0.1, 0.1]),
              r_per_hour=0.0, generator_fn=lambda t: (0.0, 0.0))
    a = solver.solve_coupled_pde(g, tg, term, **kw)
    b = solver.solve_coupled_pde(g, tg, term, **kw,
                                 sigma_fn=lambda t: np.array([0.5, 0.5]))
    assert not np.allclose(a.V, b.V)


def test_solver_rejects_bad_sigma_fn():
    g = grid.SpaceGrid(-4.0, 4.0, 51)
    v = pd.Timestamp("2025-12-31T20:00:00Z")
    tg = grid.TimeGrid(v, v + pd.Timedelta(hours=4), 8)
    term = np.zeros((2, 51))
    with pytest.raises(ValueError, match="invalid volatility vector"):
        solver.solve_coupled_pde(
            g, tg, term, lambda t: np.zeros((2, 51)), np.array([0.1, 0.1]), 0.0,
            lambda t: (0.0, 0.0), sigma_fn=lambda t: np.array([-1.0, 0.1]))


def test_single_regime_reference_solver_runs():
    g = grid.SpaceGrid(-3.0, 3.0, 121)
    v = pd.Timestamp("2025-12-31T20:00:00Z")
    tg = grid.TimeGrid(v, v + pd.Timedelta(hours=12), 24)
    out = solver.solve_single_regime(
        g, tg, np.maximum(g.y, 0.0), lambda t: np.zeros(g.y.size), 0.2, 0.0)
    assert out.shape == (121,) and np.all(np.isfinite(out))


def test_forward_calibration_still_refuses_empty_quotes():
    with pytest.raises(ValueError, match="no forward quotes"):
        forward_calibration.calibrate(None, None, [])


def test_legacy_scenario_path_object():
    t = np.linspace(0, 10, 11)
    sp = scenarios.ScenarioPath("t", t, np.zeros(11), np.zeros(11))
    assert sp.z_for_theta().shape == (11,)
    assert sp.z_for_transitions().shape == (11,)


def test_seasonal_model_modes():
    betas = np.zeros(len(transformations.SEASONAL_COLUMNS))
    betas[transformations.SEASONAL_COLUMNS.index("D_weekend")] = 0.7
    sm = transformations.SeasonalModel(betas=betas, mode="mean")
    assert np.isclose(sm.mean_value(), 0.7 * 2 / 7)
    sm.mode = "zero"
    idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    assert np.allclose(sm.value(idx), 0.0)


def test_gamma_zero_boundary_object():
    bc = boundaries.GammaZeroY()
    assert bc.kind() == "gamma_zero"
    with pytest.raises(NotImplementedError):
        bc.dirichlet_value("lower", 0, 1.0, 0.0, 0.1)
