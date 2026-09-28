"""FW9 §5, §6 -- LR test + parameter comparison + price impact at 1201 nodes.

Reads the pickled fit results, produces:

* ``parameter_comparison.csv/md`` -- inherited vs re-estimated with SE,
  t-stat, and whether the inherited value sits inside the FW9 95 % CI;
* ``lr_test_TVTP_vs_constant.csv/md`` -- FW7 nested LR test;
* ``price_impact_grid.csv/md`` -- K x T = {2000..4000} x {24, 48, 72}
  under both parameter sets, at 1201 spatial nodes with the
  production climatology z path (mandatory per FW12b rule);
* ``fw9_reestimated_parameters.yaml`` candidate yaml.

No production output tree is touched; ``m2_frozen_parameters.yaml``
is not modified.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption               # noqa: E402
from pde_option_model.forward_centered import (                     # noqa: E402
    ForwardCenteredModel, ResidualGridSettings, ResidualSpec,
    price_forward_centered)
from pde_option_model.forward_curve import build_forward_curve       # noqa: E402
from pde_option_model.generator import TVTPCoefficients              # noqa: E402
from pde_option_model.market_data import load_quotes                 # noqa: E402
from pde_option_model.params_frozen import load_frozen_parameters    # noqa: E402
from scripts.fw12._shared import climatology_z_lagged_fn             # noqa: E402
from pde_option_model.msar_estimation import unpack                  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "fw9_self_estimation"
YAML_PATH = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"


# --------------------------------------------------------------------------
# Inherited (yaml) -> canonical (index 0 = normal) mapping
# --------------------------------------------------------------------------
def inherited_yaml_params() -> Dict[str, float]:
    """Yaml values under the (index 0 = normal) label convention."""
    p = load_frozen_parameters(YAML_PATH)
    return {
        "mu_normal": 0.0,             # yaml regime_means = [0, 0]
        "mu_stress": 0.0,
        "sigma_normal": float(p.sigma_y[0]),
        "sigma_stress": float(p.sigma_y[1]),
        "phi": float(p.phi),
        "alpha01": float(p.alpha01),
        "gamma01": float(p.gamma01),
        "alpha10": float(p.alpha10),
        "gamma10": float(p.gamma10),
    }


# --------------------------------------------------------------------------
# Parameter comparison table
# --------------------------------------------------------------------------
def _se_transform_hessian(se_hessian: np.ndarray, params) -> Dict[str, float]:
    """Approximate SE in CANONICAL parameter space.  The Hessian SE
    returned by fit_msar is in the PACKED space (log_sigma, atanh(phi),
    softplus^-1(delta)); this converts to the canonical natural-space
    SEs by the delta method.

    Packed layout:
      [0] mu_normal
      [1] mu_stress
      [2] log_sigma_normal        -> sigma_normal = exp(x_2)
      [3] log_sigma_delta_raw     -> sigma_stress = sigma_normal + softplus(x_3)
      [4] phi_raw                 -> phi = tanh(x_4)
      [5..8] alpha/gamma pairs
    """
    mu_n_se, mu_s_se = se_hessian[0], se_hessian[1]
    # sigma_normal: dsigma/dx_2 = sigma_normal
    sigma_n_se = params.sigma_normal * se_hessian[2]
    # sigma_stress = sigma_normal + softplus(x_3); d/dx_3 = sigmoid(x_3)
    # x_3 satisfies softplus(x_3) = sigma_stress - sigma_normal
    diff = params.sigma_stress - params.sigma_normal
    dsig = 1.0 - math.exp(-diff)     # sigmoid'un softplus^-1'i
    # Also propagate SE from x_2 (linear addition in sigma_normal)
    sigma_s_se = math.hypot(params.sigma_normal * se_hessian[2],
                            dsig * se_hessian[3])
    # phi = tanh(x_4); dphi/dx_4 = 1 - phi^2
    phi_se = (1.0 - params.phi ** 2) * se_hessian[4]
    return {
        "mu_normal_se": mu_n_se,
        "mu_stress_se": mu_s_se,
        "sigma_normal_se": sigma_n_se,
        "sigma_stress_se": sigma_s_se,
        "phi_se": phi_se,
        "alpha01_se": se_hessian[5],
        "gamma01_se": se_hessian[6],
        "alpha10_se": se_hessian[7],
        "gamma10_se": se_hessian[8],
    }


def build_comparison(res_tvtp, res_const=None) -> pd.DataFrame:
    inh = inherited_yaml_params()
    est = res_tvtp.params.as_dict()
    se_map = _se_transform_hessian(res_tvtp.se_hessian, res_tvtp.params)
    rows = []
    for k, v in inh.items():
        est_k = est[k]
        se_k = se_map[f"{k}_se"]
        t_stat = (est_k - v) / se_k if se_k > 0 else float("nan")
        ci_lo = est_k - 1.96 * se_k
        ci_hi = est_k + 1.96 * se_k
        in_ci = ci_lo <= v <= ci_hi
        rows.append({
            "parameter": k,
            "inherited_yaml": v,
            "fw9_estimate": est_k,
            "fw9_se": se_k,
            "t_stat_vs_inherited": t_stat,
            "fw9_95pct_low": ci_lo,
            "fw9_95pct_high": ci_hi,
            "inherited_in_fw9_95pct": bool(in_ci),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# LR test
# --------------------------------------------------------------------------
def lr_test(res_tvtp, res_const) -> Dict[str, float]:
    """FW7 nested LR: TVTP vs constant-transition, df = 2."""
    lr = 2.0 * (res_tvtp.loglik - res_const.loglik)
    df = res_tvtp.n_params - res_const.n_params   # = 2 (gamma01, gamma10)
    p = 1.0 - stats.chi2.cdf(lr, df) if lr > 0 else 1.0
    return {"loglik_TVTP": res_tvtp.loglik,
            "loglik_constant": res_const.loglik,
            "LR_stat": lr, "df": df, "p_value": p,
            "AIC_TVTP": res_tvtp.aic, "AIC_constant": res_const.aic,
            "BIC_TVTP": res_tvtp.bic, "BIC_constant": res_const.bic}


# --------------------------------------------------------------------------
# Price impact -- 1201 nodes, climatology z
# --------------------------------------------------------------------------
STRIKES = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0)
MATURITIES_H = (24, 48, 72)


def _build_model_with_params(sigma_normal: float, sigma_stress: float,
                             phi: float, alpha01: float, gamma01: float,
                             alpha10: float, gamma10: float,
                             scale_P: float = 282.48) -> ForwardCenteredModel:
    params_yaml = load_frozen_parameters(YAML_PATH)
    quotes = load_quotes(REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv")
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=params_yaml.spot_price_TRY_MWh)
    kappa_per_hour = -math.log(phi)   # discrete AR -> continuous OU rate
    spec = ResidualSpec(
        kappa_per_hour=kappa_per_hour,
        sigma_y=np.array([sigma_normal, sigma_stress]),
        scale_P=scale_P, regime_means=np.zeros(2),
        mode="additive",
    )
    return ForwardCenteredModel(
        curve=curve, spec=spec,
        tvtp=TVTPCoefficients(alpha01, gamma01, alpha10, gamma10),
        pi_filtered=params_yaml.pi_filtered,
        valuation_utc=params_yaml.valuation_utc,
        spot_price_TRY_MWh=params_yaml.spot_price_TRY_MWh)


def reprice_ladder(sigma_normal: float, sigma_stress: float, phi: float,
                   alpha01: float, gamma01: float,
                   alpha10: float, gamma10: float,
                   label: str) -> pd.DataFrame:
    model = _build_model_with_params(sigma_normal, sigma_stress, phi,
                                     alpha01, gamma01, alpha10, gamma10)
    rows = []
    for K in STRIKES:
        for T in MATURITIES_H:
            contract = EuropeanOption(
                "call", float(K), model.valuation_utc,
                model.valuation_utc + pd.Timedelta(hours=int(T)),
                r_annual=0.40)
            gs = ResidualGridSettings(n_space_nodes=1201)
            # FW12b: MANDATORY climatology z path -- no fallback.
            z_fn = climatology_z_lagged_fn(model, contract, gs)
            res = price_forward_centered(model, contract, grid_settings=gs,
                                         z_lagged_fn=z_fn)
            rows.append({
                "label": label, "strike": K, "maturity_h": T,
                "V_call": float(res.value),
                "residual_sd_T": float(res.residual_std_at_expiry),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Candidate yaml
# --------------------------------------------------------------------------
def write_candidate_yaml(res_tvtp) -> Path:
    est = res_tvtp.params.as_dict()
    se_map = _se_transform_hessian(res_tvtp.se_hessian, res_tvtp.params)
    phi = est["phi"]
    kappa = -math.log(phi) if 0 < phi < 1 else float("nan")
    half_life = math.log(2.0) / kappa if kappa > 0 else float("nan")
    yml_path = REPO_ROOT / "inputs" / "historical" / "fw9_reestimated_parameters.yaml"
    body = [
        "# =============================================================================",
        "# FW9 self-estimated MS-AR(1) TVTP parameters (CANDIDATE, not for production)",
        "# =============================================================================",
        "# These values are the outcome of an INDEPENDENT re-estimation of the M9",
        "# specification on the hourly PTF series 2019-01-01 -> 2025-12-31 20:00 UTC",
        "# (n = 61 368 hourly observations, look-ahead guard at the valuation instant).",
        "# See docs/fw9_self_estimation/README.md for the derivation.",
        "#",
        "# NOT PROMOTED to production.  The frozen yaml at",
        "# inputs/historical/m2_frozen_parameters.yaml stays as the shipped model.",
        "# =============================================================================",
        f'description: "FW9 independent re-estimation of the M9 spec on the TRY hourly series"',
        f'model: "MS-AR(1) with TVTP; 1 covariate = RD_lag1; index 0 = normal, index 1 = stress"',
        "",
        "# --- price transform --------------------------------------------------------",
        f"scale_P: 282.48                # inherited yaml value; FW9 §3 diagnoses this",
        "",
        "# --- mean reversion ---------------------------------------------------------",
        f"phi: {phi:.10f}",
        f"kappa_per_hour: {kappa:.6e}",
        f"half_life_hours: {half_life:.6f}",
        "",
        "# --- regime volatilities (per sqrt(hour), in y = asinh(P/scale_P) units) ---",
        f"sigma_y_normal: {est['sigma_normal']:.10f}",
        f"sigma_y_stress: {est['sigma_stress']:.10f}",
        "",
        "# --- TVTP logistic transition coefficients ---------------------------------",
        "tvtp:",
        f"  alpha01: {est['alpha01']:.10f}",
        f"  gamma01: {est['gamma01']:.10f}",
        f"  alpha10: {est['alpha10']:.10f}",
        f"  gamma10: {est['gamma10']:.10f}",
        "  covariate: RD_WS_standardized_lag1h",
        "  placeholder: false",
        "",
        "# --- provenance / standard errors ------------------------------------------",
        "provenance:",
        f"  fitting_method: L-BFGS-B, 20 random starts, seed 20260927",
        f"  n_obs: {res_tvtp.n_obs}",
        f"  loglik: {res_tvtp.loglik:.6f}",
        f"  AIC: {res_tvtp.aic:.6f}",
        f"  BIC: {res_tvtp.bic:.6f}",
        f"  grad_norm_at_optimum: {res_tvtp.grad_norm:.6e}",
        f"  converged_flag: {res_tvtp.converged}",
        "  provenance_tag: Calibrated (FW9 independent MLE)",
        "",
        "standard_errors:",
        f"  mu_normal_se: {se_map['mu_normal_se']:.6e}",
        f"  mu_stress_se: {se_map['mu_stress_se']:.6e}",
        f"  sigma_normal_se: {se_map['sigma_normal_se']:.6e}",
        f"  sigma_stress_se: {se_map['sigma_stress_se']:.6e}",
        f"  phi_se: {se_map['phi_se']:.6e}",
        f"  alpha01_se: {se_map['alpha01_se']:.6e}",
        f"  gamma01_se: {se_map['gamma01_se']:.6e}",
        f"  alpha10_se: {se_map['alpha10_se']:.6e}",
        f"  gamma10_se: {se_map['gamma10_se']:.6e}",
        "",
        "estimation_window:",
        '  start_utc: "2018-12-31T21:00:00+00:00"',
        '  end_utc:   "2025-12-31T20:00:00+00:00"',
    ]
    yml_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return yml_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "TVTP_1cov.pkl", "rb") as f:
        res_tvtp = pickle.load(f)

    print("Building parameter comparison table ...")
    cmp = build_comparison(res_tvtp)
    cmp.to_csv(OUT_DIR / "parameter_comparison.csv", index=False)
    print(cmp.round(6).to_string(index=False))
    (OUT_DIR / "parameter_comparison.md").write_text(
        "# FW9 §6.1 -- Parameter comparison\n\n"
        "| par | inherited | FW9 est | FW9 SE | t vs inherited | 95 % low | 95 % high | in FW9 95 %? |\n"
        "|---|---:|---:|---:|---:|---:|---:|:-:|\n"
        + "\n".join(
            f"| {r.parameter} | {r.inherited_yaml:.6g} | {r.fw9_estimate:.6g} | "
            f"{r.fw9_se:.3g} | {r.t_stat_vs_inherited:.3g} | {r.fw9_95pct_low:.6g} | "
            f"{r.fw9_95pct_high:.6g} | {'yes' if r.inherited_in_fw9_95pct else '**NO**'} |"
            for r in cmp.itertuples()) + "\n",
        encoding="utf-8")

    # LR test if constant_trans available
    lr_path = OUT_DIR / "constant_trans.pkl"
    if lr_path.exists():
        with open(lr_path, "rb") as f:
            res_const = pickle.load(f)
        lr = lr_test(res_tvtp, res_const)
        pd.DataFrame([lr]).to_csv(OUT_DIR / "lr_test_TVTP_vs_constant.csv",
                                  index=False)
        print("\nLR test (TVTP vs constant-transition):")
        print(pd.DataFrame([lr]).T.to_string(header=False))
    else:
        print("constant_trans.pkl not yet available; skipping LR test.")

    # Price impact under FW9 params (climatology z, 1201 nodes)
    print("\nRe-pricing K x T ladder under FW9 parameters ...")
    t0 = time.time()
    fw9 = reprice_ladder(sigma_normal=res_tvtp.params.sigma_normal,
                         sigma_stress=res_tvtp.params.sigma_stress,
                         phi=res_tvtp.params.phi,
                         alpha01=res_tvtp.params.alpha01,
                         gamma01=res_tvtp.params.gamma01,
                         alpha10=res_tvtp.params.alpha10,
                         gamma10=res_tvtp.params.gamma10,
                         label="FW9_reestimated")
    print(f"  fw9 ladder time {time.time()-t0:.1f} s")

    inh = inherited_yaml_params()
    print("Re-pricing under inherited yaml parameters (climatology z, 1201 nodes) ...")
    t0 = time.time()
    inh_grid = reprice_ladder(sigma_normal=inh["sigma_normal"],
                              sigma_stress=inh["sigma_stress"],
                              phi=inh["phi"],
                              alpha01=inh["alpha01"], gamma01=inh["gamma01"],
                              alpha10=inh["alpha10"], gamma10=inh["gamma10"],
                              label="inherited_yaml")
    print(f"  inherited ladder time {time.time()-t0:.1f} s")

    both = pd.concat([inh_grid, fw9], ignore_index=True)
    both.to_csv(OUT_DIR / "price_impact_grid.csv", index=False)

    # Pivot to side-by-side
    pivot = both.pivot_table(index=["strike", "maturity_h"],
                             columns="label", values="V_call").reset_index()
    pivot["delta_TRY"] = pivot["FW9_reestimated"] - pivot["inherited_yaml"]
    pivot["delta_pct"] = 100.0 * pivot["delta_TRY"] / pivot["inherited_yaml"]
    pivot.to_csv(OUT_DIR / "price_impact_pivot.csv", index=False)
    print("\nPrice impact pivot (inherited vs FW9 reestimated):")
    print(pivot.round(3).to_string(index=False))

    # Candidate yaml
    yml_path = write_candidate_yaml(res_tvtp)
    print("wrote candidate yaml:", yml_path)


if __name__ == "__main__":
    main()
