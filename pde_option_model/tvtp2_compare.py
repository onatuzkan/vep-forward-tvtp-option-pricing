"""Controlled 1D vs 2D TVTP comparison (ramp-effect ladder R0-R4 + sensitivities).

Everything except the transition law is held fixed and shared BY OBJECT:
the forward curve F(t), kappa, sigma, the initial regime law pi0, the
discount rate, the contracts, the residual space grid (one fixed x-range per
maturity = union over all runs and strikes), the time grid, the ODE sub-step
and the Monte Carlo seed / paths / step.

Ladder (all on the same climatology covariate path)::

    R0  1D, production intercepts (repo yaml)                 reference
    R1  1D, intercepts re-derived on D = W9                    R1 - R0 = sample effect
    R2  2D code, R1 intercepts, h = (0, 0)                     identity: R2 == R1
    R3  2D, intercepts re-derived on D = W9 with the M9 h      R3 - R1 = total ramp effect
    R4  2D, R1 intercepts, M9 h (diagnostic; breaks targets)   R4 - R1 slope part,
                                                               R3 - R4 intercept compensation

Sensitivities change ONE thing and report the 2D-minus-1D ramp effect under
that change (see :func:`sensitivity_specs`).  EXPERIMENTAL: the ramp is a
reconstruction; nothing here replaces accepted results.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import EuropeanOption
from .forward_centered import (ForwardCenteredModel, ResidualGridSettings,
                               build_residual_grid, price_forward_centered,
                               simulate_residual_at_hours)
from .generator import TVTP2Coefficients, TVTPCoefficients
from .scenarios import CovariatePathBuilder, ScenarioSpec, TVTPCovariatePath
from .tvtp2 import (build_covariate_panel, derive_intercepts, fit_ramp_scaler,
                    load_m9_bundle, m9_raw_to_production)

MATURITIES_H: Tuple[int, ...] = (24, 72, 168, 336)
STRIKES: Tuple[float, ...] = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0)
SENS_STRIKES: Tuple[float, ...] = (2500.0, 3000.0, 3500.0)
LADDER = ("R0", "R1", "R2", "R3", "R4")


@dataclass
class PathVariant:
    key: str
    spec: ScenarioSpec
    builder: CovariatePathBuilder
    description: str


@dataclass
class RunSpec:
    name: str
    family: str
    coef: Any                           # TVTPCoefficients | TVTP2Coefficients
    path_key: str
    ramp_scaler: Any                    # expected scaler (2D) or None
    description: str

    @property
    def is_2d(self) -> bool:
        return isinstance(self.coef, TVTP2Coefficients)

    def coefficients_dict(self) -> Dict[str, Any]:
        c = self.coef
        d = {"alpha01": c.alpha01, "gamma01": c.gamma01, "alpha10": c.alpha10,
             "gamma10": c.gamma10, "h01": getattr(c, "h01", 0.0), "h10": getattr(c, "h10", 0.0)}
        return d


def curve_hash(model: ForwardCenteredModel) -> str:
    v = model.curve.values
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(v.to_numpy(dtype=float)).tobytes())
    h.update(np.asarray(v.index.asi8).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
def build_context(base_model: ForwardCenteredModel, params, tvtp2, z_history: pd.Series,
                  bundle_dir: str, train_end: pd.Timestamp,
                  history_file: str = "inputs/historical/rd_standardized.csv",
                  include_sensitivities: bool = True) -> Dict[str, Any]:
    """Coefficient sets, path variants and run specs for the comparison."""
    prod = m9_raw_to_production(load_m9_bundle(bundle_dir))
    dn, ds = prod["duration_normal_h"], prod["duration_stress_h"]
    c3 = tvtp2.coefficients
    g01, g10, h01, h10 = c3.gamma01, c3.gamma10, c3.h01, c3.h10
    sc9 = tvtp2.ramp_scaler
    D_end = pd.Timestamp(tvtp2.derivation["sample"]["last_valid_transition_utc"])
    panel = build_covariate_panel(z_history, sc9)
    one = derive_intercepts(panel, g01, 0.0, g10, 0.0, dn, ds, D_end, use_ramp=False)
    c0 = TVTPCoefficients(params.alpha01, params.gamma01, params.alpha10, params.gamma10)
    c1 = TVTPCoefficients(one["alpha01"], g01, one["alpha10"], g10)
    c2 = TVTP2Coefficients(one["alpha01"], g01, 0.0, one["alpha10"], g10, 0.0, name="R2")
    c4 = TVTP2Coefficients(one["alpha01"], g01, h01, one["alpha10"], g10, h10, name="R4")

    lag = float(tvtp2.lag_hours)
    b_base = CovariatePathBuilder(z_history, train_end, lag, sc9, history_file)
    variants: Dict[str, PathVariant] = {
        "base": PathVariant("base", ScenarioSpec("base", "climatology", 0.0), b_base,
                            "climatology (TRY window), offset 0, lag 1 h, ramp scaler W9"),
    }
    runs: List[RunSpec] = [
        RunSpec("R0", "ladder", c0, "base", None, "1D, production intercepts (repo yaml)"),
        RunSpec("R1", "ladder", c1, "base", None, "1D, intercepts re-derived on D = W9"),
        RunSpec("R2", "ladder", c2, "base", sc9, "2D code, R1 intercepts, h = 0 (identity)"),
        RunSpec("R3", "ladder", c3, "base", sc9, "2D primary: W9 intercepts with M9 h"),
        RunSpec("R4", "ladder", c4, "base", sc9, "2D, R1 intercepts, M9 h (diagnostic)"),
    ]
    derived: Dict[str, Any] = {"R1": {"alpha01": one["alpha01"], "alpha10": one["alpha10"]}}

    if include_sensitivities:
        # (a) ramp-scaler window W_T (2D only; 1D twin = R1)
        scT = fit_ramp_scaler(z_history, pd.Timestamp("2022-12-31T20:00:00+00:00"), "W_T")
        pT = build_covariate_panel(z_history, scT)
        dT = derive_intercepts(pT, g01, h01, g10, h10, dn, ds, D_end)
        variants["rampWT"] = PathVariant(
            "rampWT", ScenarioSpec("rampWT", "climatology", 0.0),
            CovariatePathBuilder(z_history, train_end, lag, scT, history_file),
            "as base, ramp standardized on W_T")
        runs.append(RunSpec("S_rampWT_2D", "ramp_scaler_W_T",
                            TVTP2Coefficients(dT["alpha01"], g01, h01, dT["alpha10"], g10, h10),
                            "rampWT", scT, "2D, ramp scaler W_T, intercepts re-derived"))
        # (b) intercept sample = full sample
        full_end = panel.grid_end
        of = derive_intercepts(panel, g01, 0.0, g10, 0.0, dn, ds, full_end, use_ramp=False)
        tf = derive_intercepts(panel, g01, h01, g10, h10, dn, ds, full_end)
        runs.append(RunSpec("S_alphaFull_1D", "alpha_sample_full",
                            TVTPCoefficients(of["alpha01"], g01, of["alpha10"], g10),
                            "base", None, "1D, intercepts on the full sample"))
        runs.append(RunSpec("S_alphaFull_2D", "alpha_sample_full",
                            TVTP2Coefficients(tf["alpha01"], g01, h01, tf["alpha10"], g10, h10),
                            "base", sc9, "2D, intercepts on the full sample"))
        # (c) z re-standardized on the M9 window (affects 1D too)
        w9 = z_history.loc[:D_end].to_numpy(dtype=float)
        mu9, sd9 = float(w9.mean()), float(w9.std(ddof=1))
        z9 = (z_history - mu9) / sd9
        sc9z = fit_ramp_scaler(z9, sc9.window_end_utc, "W9(z rescaled)")
        p9 = build_covariate_panel(z9, sc9z)
        o9 = derive_intercepts(p9, g01, 0.0, g10, 0.0, dn, ds, D_end, use_ramp=False)
        t9 = derive_intercepts(p9, g01, h01, g10, h10, dn, ds, D_end)
        variants["zW9"] = PathVariant(
            "zW9", ScenarioSpec("zW9", "climatology", 0.0),
            CovariatePathBuilder(z9, train_end, lag, sc9z, history_file + " (rescaled on W9)"),
            f"z re-standardized on W9 (mean {mu9:.6f}, sd {sd9:.6f}); climatology of the rescaled z")
        runs.append(RunSpec("S_zW9_1D", "z_scale_W9",
                            TVTPCoefficients(o9["alpha01"], g01, o9["alpha10"], g10),
                            "zW9", None, "1D, z rescaled on W9, intercepts re-derived"))
        runs.append(RunSpec("S_zW9_2D", "z_scale_W9",
                            TVTP2Coefficients(t9["alpha01"], g01, h01, t9["alpha10"], g10, h10),
                            "zW9", sc9z, "2D, z rescaled on W9, intercepts re-derived"))
        # (d) time alignment +/- 1 h
        for lg, key in ((lag - 1.0, "lag0"), (lag + 1.0, "lag2")):
            variants[key] = PathVariant(
                key, ScenarioSpec(key, "climatology", 0.0),
                CovariatePathBuilder(z_history, train_end, lg, sc9, history_file),
                f"alignment l(tau) = floor(t_v + tau - {lg:g} h)")
            runs.append(RunSpec(f"S_{key}_1D", f"alignment_{key}", c1, key, None, "R1 coefficients"))
            runs.append(RunSpec(f"S_{key}_2D", f"alignment_{key}", c3, key, sc9, "R3 coefficients"))
        # (e) scenario modes
        for key, spec, desc in (
                ("constant0", ScenarioSpec("constant0", "constant", 0.0),
                 "constant z = 0 (ramp = -m_r/s_r)"),
                ("offm2", ScenarioSpec("offm2", "climatology", -2.0), "climatology - 2 sd"),
                ("offp2", ScenarioSpec("offp2", "climatology", 2.0), "climatology + 2 sd"),
                ("observed", ScenarioSpec("observed", "climatology", 0.0,
                                          initial_hours="observed"),
                 "climatology, observed z at t_v - 2 h .. t_v")):
            variants[key] = PathVariant(key, spec, b_base, desc)
            runs.append(RunSpec(f"S_{key}_1D", f"scenario_{key}", c1, key, None, "R1 coefficients"))
            runs.append(RunSpec(f"S_{key}_2D", f"scenario_{key}", c3, key, sc9, "R3 coefficients"))
        derived.update({"S_rampWT_2D": dT, "S_alphaFull": {"1D": of, "2D": tf},
                        "S_zW9": {"1D": o9, "2D": t9, "mu9": mu9, "sd9": sd9}})
    return {"runs": runs, "variants": variants, "derived": derived, "c0": c0,
            "base_model": base_model}


def model_for(run: RunSpec, base_model: ForwardCenteredModel) -> ForwardCenteredModel:
    """Same curve / spec / pi0 objects; only the transition law differs."""
    return dataclasses.replace(base_model, tvtp=run.coef, covariate_path=None,
                               expected_ramp_scaler=run.ramp_scaler)


def contract(params, T_hours: float, K: float, otype: str, r_annual: float) -> EuropeanOption:
    return EuropeanOption(otype, float(K), params.valuation_utc,
                          params.valuation_utc + pd.Timedelta(hours=float(T_hours)), r_annual)


def build_paths(ctx: Dict[str, Any], params, maturities: Sequence[float],
                gs: ResidualGridSettings) -> Dict[Tuple[str, float], TVTPCovariatePath]:
    out = {}
    for key, v in ctx["variants"].items():
        for T in maturities:
            mat = params.valuation_utc + pd.Timedelta(hours=float(T))
            out[(key, float(T))] = v.builder.build(v.spec, params.valuation_utc,
                                                   maturity_utc=mat, grid_settings=gs)
    return out


def common_grids(ctx, params, paths, maturities, strikes, gs: ResidualGridSettings,
                 r_annual: float) -> Dict[float, Tuple[float, float]]:
    """One x-range per maturity covering every run and strike (union)."""
    grids = {}
    for T in maturities:
        lo, hi = np.inf, -np.inf
        for run in ctx["runs"]:
            m = model_for(run, ctx["base_model"])
            path = paths[(run.path_key, float(T))]
            for K in strikes:
                c = contract(params, T, K, "call", r_annual)
                t = np.linspace(0.0, c.tau_hours, gs.n_steps(c.tau_hours) + 1)
                z, r = m.resolve_covariates(t, covariate_path=path)
                g = (build_residual_grid(m, c, gs, z) if r is None
                     else build_residual_grid(m, c, gs, z, r))
                lo, hi = min(lo, g.y_min), max(hi, g.y_max)
        grids[float(T)] = (float(lo), float(hi))
    return grids


def price_one(run: RunSpec, base_model, params, path, T: float, K: float,
              r_annual: float, gs: ResidualGridSettings, x_range: Tuple[float, float],
              with_put: bool = True, n_time_steps: Optional[int] = None
              ) -> Dict[str, Any]:
    """Call (+ put) on the fixed grid; returns values, parity and transition stats."""
    m = model_for(run, base_model)
    gfix = dataclasses.replace(gs, x_min=x_range[0], x_max=x_range[1])
    if n_time_steps is not None:
        gfix = dataclasses.replace(gfix, n_time_steps=int(n_time_steps))
    c = price_forward_centered(m, contract(params, T, K, "call", r_annual), gfix,
                               covariate_path=path)
    p = (price_forward_centered(m, contract(params, T, K, "put", r_annual), gfix,
                                covariate_path=path) if with_put else None)
    disc = float(np.exp(-r_annual / 8760.0 * T))
    row = {"run": run.name, "family": run.family, "tvtp": "2D" if run.is_2d else "1D",
           "path": run.path_key, "maturity_h": float(T), "strike": float(K),
           "call_pde": c.value, "put_pde": (p.value if p is not None else float("nan")),
           "parity_error": ((c.value - p.value - disc * (c.forward_at_expiry - K))
                            if p is not None else float("nan")),
           "forward_T": c.forward_at_expiry, "residual_sd_T": c.residual_std_at_expiry,
           "grid_x_min": c.diagnostics["grid_x_min"], "grid_x_max": c.diagnostics["grid_x_max"],
           "n_space_nodes": c.diagnostics["n_space_nodes"],
           "n_time_steps": c.diagnostics["n_time_steps"]}
    for k in ("p_stress_at_expiry", "mean_p_stress", "mean_q01_per_hour", "mean_q10_per_hour",
              "mean_p01_one_hour", "mean_p10_one_hour", "min_s", "max_s",
              "n_nodes_s_ge_0.95", "expected_transitions"):
        row[k] = c.diagnostics[k]
    return row


def mc_one(run: RunSpec, base_model, params, path, T: float, strikes: Sequence[float],
           r_annual: float, n_paths: int, dt_hours: float, seed: int) -> List[Dict[str, Any]]:
    """One Monte Carlo per (run, maturity); every strike from the SAME paths."""
    m = model_for(run, base_model)
    sim = simulate_residual_at_hours(m, np.array([float(T)]), n_paths=n_paths,
                                     dt_hours=dt_hours, seed=seed, covariate_path=path)
    x = sim["x"][:, 0]
    ps = float(sim["regime"][:, 0].mean())
    prices = m.price_from_state(x, float(sim["forward"][0]), float(sim["mu_x"][0]))
    disc = float(np.exp(-r_annual / 8760.0 * T))
    # first-moment control variate: E[X_T] = mu_X(T) is known from the ODE, so the
    # mean-matched sample isolates the discrepancy in the SHAPE of the law
    x_mm = x - x.mean() + float(sim["mu_x"][0])
    prices_mm = m.price_from_state(x_mm, float(sim["forward"][0]), float(sim["mu_x"][0]))
    rows = []
    for K in strikes:
        pay_c = np.maximum(prices - K, 0.0)
        pay_p = np.maximum(K - prices, 0.0)
        rows.append({"run": run.name, "maturity_h": float(T), "strike": float(K),
                     "call_mc": disc * float(pay_c.mean()),
                     "call_mc_se": disc * float(pay_c.std(ddof=1) / np.sqrt(n_paths)),
                     "put_mc": disc * float(pay_p.mean()),
                     "put_mc_se": disc * float(pay_p.std(ddof=1) / np.sqrt(n_paths)),
                     "call_mc_mean_matched": disc * float(np.maximum(prices_mm - K, 0.0).mean()),
                     "put_mc_mean_matched": disc * float(np.maximum(K - prices_mm, 0.0).mean()),
                     "mc_mean_residual_T": float(x.mean()),
                     "mc_mean_residual_T_se": float(x.std(ddof=1) / np.sqrt(n_paths)),
                     "ode_mean_residual_T": float(sim["mu_x"][0]),
                     "mc_residual_sd_T": float(x.std(ddof=1)),
                     "ode_residual_sd_T_mc_grid": float(np.sqrt(sim["var_x"][0])),
                     "mc_p_stress_T": ps,
                     "mc_p_stress_T_se": float(np.sqrt(ps * (1.0 - ps) / n_paths)),
                     "ode_p_stress_T_mc_grid": float(sim["p_stress"][0]),
                     "mc_n_paths": n_paths, "mc_dt_hours": sim["dt_hours"], "mc_seed": seed})
    return rows
