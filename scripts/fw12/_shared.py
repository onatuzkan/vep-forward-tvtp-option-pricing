"""Shared helpers for the FW12 numerical-convergence work package.

The production CLI (``run_pde.py price``) builds a climatology
``z(t-1)`` covariate path from ``inputs/historical/rd_standardized.csv``
(train_end 2022-12-31 20:00 UTC, lag 1 h) before pricing.  Calling
``price_forward_centered(model, contract, ...)`` from a Python script
WITHOUT that path silently defaults to ``z = 0`` (see
``ForwardCenteredModel.resolve_covariates``), which produces a
DIFFERENT set of q01(t), q10(t) intensities and therefore a
different price.

That is the actual reason the FW2 baseline of 179.65 differs from the
CLI's 166.75 -- 601 vs 1201 nodes account for a small part of it
(~0.05 TRY/MWh at ATM 72 h in the two runs), and z=0 vs climatology
z accounts for the rest.  FW12 fixes that ambiguity: every FW12 sweep
uses the SAME climatology path the CLI uses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from pde_option_model.contracts import EuropeanOption
from pde_option_model.forward_centered import (ForwardCenteredModel,
                                               ResidualGridSettings,
                                               ResidualSpec,
                                               price_forward_centered)
from pde_option_model.forward_curve import build_forward_curve
from pde_option_model.generator import TVTPCoefficients
from pde_option_model.market_data import load_quotes
from pde_option_model.params_frozen import load_frozen_parameters
from pde_option_model.scenarios import ScenarioBuilder, ScenarioSpec
from pde_option_model.tvtp2 import load_hourly_z_history
from pde_option_model.grid import TimeGrid

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_YAML = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
QUOTES_CSV = REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv"
Z_HISTORY_CSV = REPO_ROOT / "inputs" / "historical" / "rd_standardized.csv"
TRAIN_END_UTC = pd.Timestamp("2022-12-31 20:00:00+00:00")
COVARIATE_LAG_HOURS = 1.0

R_ANNUAL = 0.40


def build_production_model() -> ForwardCenteredModel:
    """Assemble the production forward-centered model (matches the
    accepted `run_pde.py calibrate-market` output)."""
    params = load_frozen_parameters(PARAMS_YAML)
    quotes = load_quotes(QUOTES_CSV)
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    return ForwardCenteredModel(
        curve=curve, spec=ResidualSpec.from_frozen(params),
        tvtp=TVTPCoefficients(params.alpha01, params.gamma01,
                              params.alpha10, params.gamma10),
        pi_filtered=params.pi_filtered,
        valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh)


def climatology_scenario_builder() -> ScenarioBuilder:
    z_hist = load_hourly_z_history(str(Z_HISTORY_CSV))
    return ScenarioBuilder(z_hist, TRAIN_END_UTC,
                           covariate_lag_hours=COVARIATE_LAG_HOURS)


def climatology_z_lagged_fn(model: ForwardCenteredModel,
                            contract: EuropeanOption,
                            gs: ResidualGridSettings):
    """Return a callable ``t_array -> z_lag(t_array)`` matching the
    production CLI's climatology scenario.  Because the FW12 sweep
    varies n_time_steps, the callable is a closure that produces the
    z_lag values for whatever solver grid the pricer builds.
    """
    builder = climatology_scenario_builder()
    # We pre-build a long dense z(t) climatology so that resampling to
    # any solver grid is a simple linear-interpolate lookup.
    horizon = float(contract.tau_hours)
    dense_t = np.linspace(0.0, horizon, int(horizon * 60) + 1)
    tgrid = TimeGrid(contract.valuation_utc, contract.maturity_utc,
                     dense_t.size - 1)
    path = builder.build(ScenarioSpec(name="clim72h", mode="climatology"),
                         tgrid)
    def _fn(t_solver: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(t_solver, dtype=float),
                         path.times_hours, path.z_lagged)
    return _fn


def price_at(model: ForwardCenteredModel, strike: float, tau_hours: int,
             kind: str, n_space_nodes: int, n_time_steps: Optional[int],
             n_std: float = 6.0, eta_ij=None
             ) -> Tuple[float, dict]:
    contract = EuropeanOption(kind, strike, model.valuation_utc,
                              model.valuation_utc + pd.Timedelta(hours=int(tau_hours)),
                              r_annual=R_ANNUAL)
    gs = ResidualGridSettings(n_space_nodes=int(n_space_nodes),
                              n_time_steps=n_time_steps, n_std=float(n_std))
    z_fn = climatology_z_lagged_fn(model, contract, gs)
    res = price_forward_centered(model, contract, grid_settings=gs,
                                 z_lagged_fn=z_fn, eta_ij=eta_ij)
    info = {"n_space_nodes": int(n_space_nodes),
            "n_time_steps": int(res.diagnostics["n_time_steps"]),
            "n_std": float(n_std),
            "grid_x_min": float(res.diagnostics["grid_x_min"]),
            "grid_x_max": float(res.diagnostics["grid_x_max"]),
            "residual_sd_T": float(res.residual_std_at_expiry),
            "expected_spot_T": float(res.expected_spot_at_expiry),
            "forward_T": float(res.forward_at_expiry),
            "p_stress_at_expiry": float(res.diagnostics.get(
                "p_stress_at_expiry", np.nan))}
    return float(res.value), info


def observed_order(v_h: float, v_h2: float, v_h4: float) -> float:
    """Observed convergence order p from three refinements h, h/2, h/4.

        p = log2(|V_h - V_{h/2}| / |V_{h/2} - V_{h/4}|)

    Returns NaN if either difference is smaller than ~1e-9 TRY/MWh
    (in the noise floor of a Crank-Nicolson PDE at this grid).
    """
    d1 = abs(v_h - v_h2)
    d2 = abs(v_h2 - v_h4)
    if d1 < 1e-9 or d2 < 1e-9:
        return float("nan")
    return float(np.log2(d1 / d2))


def richardson_estimate(v_h2: float, v_h4: float, order: float) -> float:
    """Richardson extrapolation on two refinements to estimate the
    converged value, assuming the observed order ``order``.

        V_star = (2^p * V_{h/4} - V_{h/2}) / (2^p - 1)
    """
    f = 2.0 ** order
    return (f * v_h4 - v_h2) / (f - 1.0)
