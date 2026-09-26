"""Two-covariate TVTP in the pricing stack: identities, centering, PDE vs MC.

Validation items 1 (zero-ramp reproduction of the production prices),
4 (E^Q[P_t] = F(t), monthly VEP averages and put-call parity, including
NON-circular Monte Carlo checks with m != 0 so that mu_X is not identically
zero), 5 (PDE vs Monte Carlo on the same two-covariate path, with standard
errors) and 6 (controlled 1D vs 2D comparison: everything but the transition
law held fixed by object).  Repository data only.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import run_pde
from pde_option_model.contracts import EuropeanOption
from pde_option_model.forward_centered import (ForwardCenteredError, ResidualGridSettings,
                                               ResidualSpec, price_forward_centered,
                                               simulate_forward_centered,
                                               simulate_residual_at_hours)
from pde_option_model.generator import (EmbeddabilityError, TVTP2Coefficients,
                                        TVTPCoefficients)
from pde_option_model.market_calibration import run_market_calibration
from pde_option_model.scenarios import ScenarioSpec

from .conftest import M9_BUNDLE, RD_HISTORY, TRY_TRAIN_END

R_ANNUAL = 0.40
NONZERO_MEANS = (150.0, -400.0)          # makes mu_X(t) path-dependent (non-vacuous)
DRIFT_SHIFT = (0.05, -0.03)


def _opt(params, otype, K, T):
    return EuropeanOption(otype, float(K), params.valuation_utc,
                          params.valuation_utc + pd.Timedelta(hours=T), R_ANNUAL)


def _model2d(model, tvtp2_params, coef=None, **spec_kw):
    spec = model.spec if not spec_kw else ResidualSpec(
        kappa_per_hour=model.spec.kappa_per_hour, sigma_y=model.spec.sigma_y,
        scale_P=model.spec.scale_P, **spec_kw)
    return dataclasses.replace(model, spec=spec,
                               tvtp=coef if coef is not None else tvtp2_params.coefficients,
                               expected_ramp_scaler=tvtp2_params.ramp_scaler)


@pytest.fixture(scope="module")
def paths72(path_builder, params, production_grid):
    return path_builder.build(ScenarioSpec("clim"), params.valuation_utc,
                              maturity_utc=params.valuation_utc + pd.Timedelta(hours=72),
                              grid_settings=production_grid)


# ---------------------------------------------------------------------------
# 1. zero ramp slopes reproduce the production (single-covariate) prices
# ---------------------------------------------------------------------------
def test_zero_ramp_reproduces_the_production_price_bit_for_bit(model, params, tvtp2_params,
                                                               paths72, production_grid):
    """2D code, h = 0, production intercepts == old 1D code path (run_pde)."""
    cfg = run_pde._load_config(None)
    ns = type("A", (), {"scenario_history": None, "scenario_mode": None,
                        "scenario_offset": None, "scenario_custom_csv": None})()
    zero = TVTP2Coefficients.from_single_covariate(model.tvtp)
    m2 = _model2d(model, tvtp2_params, coef=zero)
    for otype in ("call", "put"):
        c = _opt(params, otype, 3000.0, 72)
        _, zfn = run_pde._build_tvtp_scenario(c, production_grid, cfg, ns)
        old = price_forward_centered(model, c, production_grid, z_lagged_fn=zfn)
        new = price_forward_centered(m2, c, production_grid, covariate_path=paths72)
        assert new.value == old.value
        assert new.residual_std_at_expiry == old.residual_std_at_expiry
        assert new.diagnostics["p_stress_at_expiry"] == old.diagnostics["p_stress_at_expiry"]
        assert np.array_equal(new.V_regime, old.V_regime)
    assert abs(price_forward_centered(model, _opt(params, "call", 3000.0, 72), production_grid,
                                      z_lagged_fn=zfn).value - 166.7477) < 5e-5   # paper Table 3


def test_zero_ramp_reproduces_the_production_monte_carlo(model, params, tvtp2_params, paths72):
    m2 = _model2d(model, tvtp2_params, coef=TVTP2Coefficients.from_single_covariate(model.tvtp))
    c = _opt(params, "call", 3000.0, 72)
    a = simulate_forward_centered(model, c, n_paths=8000, dt_hours=0.25, seed=3,
                                  z_lagged_fn=paths72.z_lagged)
    b = simulate_forward_centered(m2, c, n_paths=8000, dt_hours=0.25, seed=3,
                                  covariate_path=paths72)
    assert a["value"] == b["value"] and a["mean_price_T"] == b["mean_price_T"]


# ---------------------------------------------------------------------------
# guards: the ramp is never dropped or zero-filled, units are checked
# ---------------------------------------------------------------------------
def test_two_covariate_model_refuses_to_run_without_the_ramp(model, params, tvtp2_params,
                                                             paths72, fast_grid):
    m2 = _model2d(model, tvtp2_params)
    c = _opt(params, "call", 3000.0, 72)
    with pytest.raises(ForwardCenteredError, match="drop the ramp"):
        price_forward_centered(m2, c, fast_grid, z_lagged_fn=paths72.z_lagged)
    with pytest.raises(ForwardCenteredError, match="refusing to default"):
        m2.moments(np.arange(0.0, 73.0))
    t = np.arange(0.0, 73.0)
    with pytest.raises(ForwardCenteredError, match="BOTH"):
        m2.moments(t, paths72.z_lagged(t))
    with pytest.raises(ForwardCenteredError, match="single-covariate"):
        model.generator_path(t, paths72.z_lagged(t), paths72.ramp_lagged(t))


def test_a_path_with_a_different_ramp_standardization_is_refused(model, params, tvtp2_params,
                                                                 z_history, fast_grid):
    from pde_option_model.scenarios import CovariatePathBuilder
    from pde_option_model.tvtp2 import fit_ramp_scaler
    sT = fit_ramp_scaler(z_history, TRY_TRAIN_END, "W_T")
    other = CovariatePathBuilder(z_history, TRY_TRAIN_END, 1.0, sT).build(
        ScenarioSpec("wt"), params.valuation_utc, horizon_hours=72.0)
    with pytest.raises(ForwardCenteredError, match="mismatch"):
        price_forward_centered(_model2d(model, tvtp2_params), _opt(params, "call", 3000.0, 72),
                               fast_grid, covariate_path=other)


def test_nonembeddable_scenario_rejects_pricing(model, params, tvtp2_params, path_builder,
                                                fast_grid):
    path = path_builder.build(ScenarioSpec("x", "climatology", -4.5), params.valuation_utc,
                              horizon_hours=720.0)
    with pytest.raises(EmbeddabilityError) as ei:
        price_forward_centered(_model2d(model, tvtp2_params), _opt(params, "call", 3000.0, 720),
                               fast_grid, covariate_path=path)
    assert ei.value.report["n_s_ge_1"] > 0 and ei.value.report["max_s"] > 1.0


# ---------------------------------------------------------------------------
# 4. centering, VEP months and parity -- non-circular
# ---------------------------------------------------------------------------
def test_centering_holds_on_the_two_covariate_path(model, tvtp2_params, path_builder, params):
    path = path_builder.build(ScenarioSpec("c"), params.valuation_utc, horizon_hours=720.0)
    for kw in ({}, {"regime_means": np.array(NONZERO_MEANS)},
               {"drift_shift_per_hour": np.array(DRIFT_SHIFT)}):
        m2 = _model2d(model, tvtp2_params, **kw)
        t = np.arange(0.0, 721.0)
        mom = m2.moments(t, covariate_path=path)
        s = m2.residual_summary(t, covariate_path=path)
        assert np.max(np.abs(s["expected_spot_TRY_MWh"] - s["forward_TRY_MWh"])) < 1e-8
        if kw:
            assert np.max(np.abs(mom.mean)) > 1e-3, "non-vacuous: mu_X must move"


def test_independent_monte_carlo_mean_matches_the_ode_centering(model, params, tvtp2_params,
                                                                paths72):
    """MC sample mean of X_T vs the ODE mu_X(T) on the SAME 2D q path, m != 0."""
    m2 = _model2d(model, tvtp2_params, regime_means=np.array(NONZERO_MEANS),
                  drift_shift_per_hour=np.array(DRIFT_SHIFT))
    mc = simulate_forward_centered(m2, _opt(params, "call", 3000.0, 72), n_paths=40_000,
                                   dt_hours=0.05, seed=20260924, covariate_path=paths72)
    assert abs(mc["analytic_mean_residual_T"]) > 50.0          # non-vacuous
    z = (mc["mean_residual_T"] - mc["analytic_mean_residual_T"]) / mc["mean_residual_T_se"]
    assert abs(z) < 3.5, z
    zp = (mc["p_stress_T_mc"] - mc["analytic_p_stress_T"]) / mc["p_stress_T_se"]
    assert abs(zp) < 3.5, zp
    zf = (mc["mean_price_T"] - mc["analytic_expected_spot_T"]) / mc["mean_price_T_se"]
    assert abs(zf) < 3.5, zf


def test_monthly_vep_average_holds_by_simulation(model, params, quotes, tvtp2_params,
                                                 path_builder):
    """Average over February 2026 delivery hours of simulated P_h = VEP quote (m != 0)."""
    q = quotes.get(2026, 2)
    hrs = q.delivery.hours_utc()
    h = np.asarray((hrs - params.valuation_utc).total_seconds(), float) / 3600.0
    path = path_builder.build(ScenarioSpec("feb"), params.valuation_utc,
                              horizon_hours=float(np.ceil(h.max())))
    m2 = _model2d(model, tvtp2_params, regime_means=np.array(NONZERO_MEANS))
    sim = simulate_residual_at_hours(m2, h, n_paths=3000, dt_hours=0.25, seed=11,
                                     covariate_path=path)
    prices = sim["forward"][None, :] + sim["x"] - sim["mu_x"][None, :]
    per_path = prices.mean(axis=1)                      # monthly average per path
    se = per_path.std(ddof=1) / np.sqrt(per_path.size)
    assert abs(float(np.mean(sim["mu_x"]))) > 50.0      # non-vacuous
    assert abs(per_path.mean() - q.price_TRY_MWh) / se < 3.5
    # the deterministic curve reproduces the quote exactly (calibration identity)
    assert abs(float(np.mean(sim["forward"])) - q.price_TRY_MWh) < 1e-6


def test_put_call_parity_on_the_two_covariate_path(model, params, tvtp2_params, paths72,
                                                   fast_grid):
    """Production setting (m = a = 0): parity to solver precision."""
    disc = float(np.exp(-R_ANNUAL / 8760.0 * 72.0))
    m2 = _model2d(model, tvtp2_params)
    for K in (2500.0, 3000.0, 3500.0):
        c = price_forward_centered(m2, _opt(params, "call", K, 72), fast_grid,
                                   covariate_path=paths72)
        p = price_forward_centered(m2, _opt(params, "put", K, 72), fast_grid,
                                   covariate_path=paths72)
        assert abs((c.value - p.value) - disc * (c.forward_at_expiry - K)) < 1e-4


def _parity_error(m, params, path, K, n_steps):
    gs = ResidualGridSettings(n_space_nodes=801, n_time_steps=n_steps)
    disc = float(np.exp(-R_ANNUAL / 8760.0 * 72.0))
    c = price_forward_centered(m, _opt(params, "call", K, 72), gs, covariate_path=path)
    p = price_forward_centered(m, _opt(params, "put", K, 72), gs, covariate_path=path)
    return (c.value - p.value) - disc * (c.forward_at_expiry - K), c.centering_at_expiry


def test_parity_with_nonzero_regime_means_is_a_real_pde_vs_ode_check(model, params,
                                                                     tvtp2_params, paths72):
    """With m != 0 the PDE's implied first moment (linear payoff C - P) is compared
    with the moment-ODE mu_X -- two different numerical routes.  The gap is the
    PDE time-discretization error: strike-independent, second order in dt, of
    the same size in the single-covariate model (not a two-covariate artefact)."""
    kw = {"regime_means": np.array(NONZERO_MEANS),
          "drift_shift_per_hour": np.array(DRIFT_SHIFT)}
    m2 = _model2d(model, tvtp2_params, **kw)
    e144, mu = _parity_error(m2, params, paths72, 3000.0, 144)
    e144b, _ = _parity_error(m2, params, paths72, 2500.0, 144)
    e288, _ = _parity_error(m2, params, paths72, 3000.0, 288)
    e576, _ = _parity_error(m2, params, paths72, 3000.0, 576)
    assert abs(mu) > 50.0                                       # non-vacuous
    assert abs(e144 - e144b) < 1e-6                             # first-moment error only
    assert abs(e144) > abs(e288) > abs(e576)
    assert abs(e144 / e288) > 2.5 and abs(e288 / e576) > 2.5   # ~second order
    assert abs(e576) < 0.01 and abs(e576 / mu) < 1e-4
    m1 = dataclasses.replace(model, spec=m2.spec)               # same spec, 1D law
    e1, _ = _parity_error(m1, params, paths72, 3000.0, 576)
    assert 0.5 < abs(e576 / e1) < 2.0


def test_two_covariate_market_calibration_keeps_the_vep_fit(quotes, params, tvtp2_params,
                                                            path_builder):
    horizon = (quotes.last_delivery_utc - params.valuation_utc).total_seconds() / 3600.0
    path = path_builder.build(ScenarioSpec("cal"), params.valuation_utc,
                              horizon_hours=float(np.ceil(horizon)))
    res = run_market_calibration(quotes, params, tvtp2_params=tvtp2_params, covariate_path=path)
    assert res.calibration_accepted and res.metrics["maximum_absolute_monthly_error"] < 1e-6
    assert res.tvtp_mode == "rd_ramp_2d_experimental"
    tv = res.parameter_provenance["2_inherited_from_historical_M2_fit"]["tvtp_coefficients"]
    assert tv["h01"] == tvtp2_params.coefficients.h01 and tv["status"] == "experimental_reconstructed"
    assert res.tvtp_provenance["covariate_path"]["ramp_scaler"]["window_name"] == "W9"
    assert any("EXPERIMENTAL" in w for w in res.warnings)
    with pytest.raises(ValueError, match="covariate_path"):
        run_market_calibration(quotes, params, tvtp2_params=tvtp2_params)


# ---------------------------------------------------------------------------
# 5. PDE vs Monte Carlo on the same two-covariate path
# ---------------------------------------------------------------------------
def test_pde_matches_monte_carlo_on_the_same_2d_path(model, params, tvtp2_params, paths72,
                                                     production_grid):
    m2 = _model2d(model, tvtp2_params)
    for otype in ("call", "put"):
        c = _opt(params, otype, 3000.0, 72)
        pde = price_forward_centered(m2, c, production_grid, covariate_path=paths72)
        mc = simulate_forward_centered(m2, c, n_paths=40_000, dt_hours=0.05, seed=20260808,
                                       covariate_path=paths72)
        assert mc["std_error"] > 0 and mc["n_paths"] == 40_000
        z = (mc["value"] - pde.value) / mc["std_error"]
        assert abs(z) < 3.5, f"{otype}: PDE {pde.value} vs MC {mc['value']} +- {mc['std_error']}"


# ---------------------------------------------------------------------------
# 6. controlled 1D vs 2D comparison
# ---------------------------------------------------------------------------
def test_controlled_comparison_changes_only_the_transition_law(model, params, tvtp2_params,
                                                               z_history, production_grid):
    from pde_option_model import tvtp2_compare as C
    ctx = C.build_context(model, params, tvtp2_params, z_history, str(M9_BUNDLE),
                          TRY_TRAIN_END, str(RD_HISTORY), include_sensitivities=False)
    runs = {r.name: r for r in ctx["runs"]}
    assert [r.name for r in ctx["runs"]] == list(C.LADDER)
    paths = C.build_paths(ctx, params, [72.0], production_grid)
    grids = C.common_grids(ctx, params, paths, [72.0], [3000.0], production_grid, R_ANNUAL)
    models = {n: C.model_for(r, model) for n, r in runs.items()}
    for m in models.values():                                 # same objects, not copies
        assert m.curve is model.curve and m.spec is model.spec
        assert m.pi_filtered is model.pi_filtered
    rows = {n: C.price_one(runs[n], model, params, paths[("base", 72.0)], 72.0, 3000.0,
                           R_ANNUAL, production_grid, grids[72.0]) for n in ("R1", "R2", "R3")}
    for r in rows.values():
        assert (r["grid_x_min"], r["grid_x_max"]) == grids[72.0]
        assert abs(r["parity_error"]) < 1e-4
    assert rows["R2"]["call_pde"] == rows["R1"]["call_pde"]          # identity
    assert rows["R2"]["put_pde"] == rows["R1"]["put_pde"]
    effect = rows["R3"]["call_pde"] - rows["R1"]["call_pde"]
    assert 0.0 < abs(effect) < 0.05 * rows["R1"]["call_pde"]
    assert rows["R3"]["forward_T"] == rows["R1"]["forward_T"]
    # R1 intercepts are the single-covariate roots on the SAME sample as R3
    assert runs["R1"].coef.gamma01 == runs["R3"].coef.gamma01
    assert runs["R4"].coef.alpha01 == runs["R1"].coef.alpha01
    assert runs["R4"].coef.h10 == runs["R3"].coef.h10
