"""FW9b §3 + §5 + §6 -- Rebuild the comparison table, reprice with a
consistent FW9 set, decompose the price gap, and rewrite the candidate
yaml.

Reads:
  * outputs/fw9_self_estimation/profile_phi_TVTP_1cov_argmax.pkl
    (SE at the phi-conditional argmax; sigmas, alphas, gammas)
  * outputs/fw9_self_estimation/deseasonalized_ar1.json
    (phi_deseasonalized, kappa)
  * inputs/historical/archive/calibration_bundle/parameter_estimates.csv
    (raw M9 CSV row; the correct comparison target for the MS-AR
    sigmas / phi_raw etc.)
  * inputs/historical/archive/calibration_bundle/transition_coefficients.csv
    (raw M9 alpha / gamma; but only gammas are in this file -- alphas
    are DERIVED in the yaml)
  * inputs/historical/m2_frozen_parameters.yaml
    (production values under the yaml label convention)

Writes:
  * parameter_comparison_v2.csv/md  (object-matched T8)
  * price_impact_v2_grid.csv       (consistent FW9 set)
  * price_impact_v2_decomposition.csv (Δ contributions per parameter)
  * fw9_reestimated_parameters.yaml  (rewritten)
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption
from pde_option_model.forward_centered import (ForwardCenteredModel,
                                               ResidualGridSettings,
                                               ResidualSpec,
                                               price_forward_centered)
from pde_option_model.forward_curve import build_forward_curve
from pde_option_model.generator import TVTPCoefficients
from pde_option_model.market_data import load_quotes
from pde_option_model.params_frozen import load_frozen_parameters
from scripts.fw12._shared import climatology_z_lagged_fn

OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"
YAML_PATH = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
M9_CSV = REPO_ROOT / "inputs" / "historical" / "archive" / "calibration_bundle" / "parameter_estimates.csv"
M9_TRANS = REPO_ROOT / "inputs" / "historical" / "archive" / "calibration_bundle" / "transition_coefficients.csv"


def _m9_csv_row() -> Dict[str, float]:
    df = pd.read_csv(M9_CSV)
    row = df[df["model"] == "M9"].iloc[0]
    # Raw M9 labelling: sigma0 = HIGH-VOL (=> stress), sigma1 = LOW-VOL (=> normal).
    return {
        "mu_normal": float(row["mu1"]),
        "mu_stress": float(row["mu0"]),
        "sigma_normal": float(row["sigma1"]),
        "sigma_stress": float(row["sigma0"]),
        "phi": float(row["phi"]),
    }


def _m9_transition_row() -> Dict[str, float]:
    df = pd.read_csv(M9_TRANS)
    rd = df[df["covariate"] == "RD_lag1"].iloc[0]
    # Raw M9 (index 0 = stress): gamma01 = transition stress -> normal
    # Under yaml (index 0 = normal): gamma01 = transition normal -> stress
    # So yaml_gamma01 = raw_gamma10, yaml_gamma10 = raw_gamma01
    return {
        "raw_gamma01_stress_to_normal": float(rd["gamma01"]),
        "raw_gamma10_normal_to_stress": float(rd["gamma10"]),
    }


def build_comparison_v2(res_profile) -> pd.DataFrame:
    """FW9b object-matched T8.

    Column ``inherited_from`` says which OBJECT the inherited value is
    the natural comparison for (MS-AR CSV row, or the deseasonalised
    stationarity table, or the DERIVED-from-diagnostics yaml alpha).
    """
    yaml_p = load_frozen_parameters(YAML_PATH)
    m9 = _m9_csv_row()
    m9_trans = _m9_transition_row()
    des_path = OUT / "deseasonalized_ar1.json"
    des = json.loads(des_path.read_text()) if des_path.exists() else None

    est = res_profile.params.as_dict()
    # SE layout in profile: phi slot (index 4) is NaN by construction
    se9 = res_profile.se_hessian if res_profile.se_hessian is not None else np.full(9, np.nan)
    slot = {"mu_normal": 0, "mu_stress": 1, "sigma_normal": 2,
            "sigma_stress": 3, "phi": 4, "alpha01": 5, "gamma01": 6,
            "alpha10": 7, "gamma10": 8}

    def _row(par, inherited, fw9_val, se, inherited_from, boundary=False,
             comment=""):
        if boundary:
            return {
                "parameter": par, "inherited_from": inherited_from,
                "inherited": inherited, "fw9_estimate": fw9_val,
                "fw9_se": None, "t_stat_vs_inherited": None,
                "fw9_95pct_low": None, "fw9_95pct_high": None,
                "inherited_in_fw9_95pct": None,
                "boundary_flag": True, "comment": comment,
            }
        if se is None or not math.isfinite(se) or se <= 0:
            return {
                "parameter": par, "inherited_from": inherited_from,
                "inherited": inherited, "fw9_estimate": fw9_val,
                "fw9_se": None, "t_stat_vs_inherited": None,
                "fw9_95pct_low": None, "fw9_95pct_high": None,
                "inherited_in_fw9_95pct": None,
                "boundary_flag": False, "comment": comment,
            }
        t = (fw9_val - inherited) / se
        lo = fw9_val - 1.96 * se
        hi = fw9_val + 1.96 * se
        return {
            "parameter": par, "inherited_from": inherited_from,
            "inherited": inherited, "fw9_estimate": fw9_val,
            "fw9_se": se, "t_stat_vs_inherited": t,
            "fw9_95pct_low": lo, "fw9_95pct_high": hi,
            "inherited_in_fw9_95pct": bool(lo <= inherited <= hi),
            "boundary_flag": False, "comment": comment,
        }

    def _se(name):
        v = float(se9[slot[name]])
        return v if math.isfinite(v) else None

    rows = []
    # sigma_normal / sigma_stress -> MS-AR CSV row
    rows.append(_row("sigma_normal", m9["sigma_normal"], est["sigma_normal"],
                     _transform_se_sigma_normal(res_profile),
                     "M9 CSV row (parameter_estimates.csv, sigma1 low-vol)"))
    rows.append(_row("sigma_stress", m9["sigma_stress"], est["sigma_stress"],
                     _transform_se_sigma_stress(res_profile),
                     "M9 CSV row (parameter_estimates.csv, sigma0 high-vol)"))
    # phi -- MS-AR object comparison
    rows.append(_row("phi_MS_AR", m9["phi"], est["phi"], None,
                     "M9 CSV row (parameter_estimates.csv, phi)", boundary=True,
                     comment="Both fits estimated on the unit-root boundary; "
                             "Hessian-based SE not interpretable."))
    # phi -- deseasonalised object comparison
    if des is not None:
        rows.append(_row("phi_deseasonalized", 0.9246, des["phi"],
                         des["se_phi"],
                         "yaml (v2 kappa refit; deseasonalized single-regime AR(1))",
                         comment=("FW9 pipeline: hour-of-week + month-of-year "
                                  "additive seasonal dummies + OLS AR(1); the "
                                  "M9 original deseasonalisation pipeline is "
                                  "not shipped with the repo -- see preprocessing_audit.md.")))
    # mu -- MS-AR CSV row (yaml intentionally zeros)
    rows.append(_row("mu_normal", m9["mu_normal"], est["mu_normal"],
                     _se("mu_normal"),
                     "M9 CSV row (mu1); yaml explicitly zeros mu_i (centering "
                     "identity absorbs them)"))
    rows.append(_row("mu_stress", m9["mu_stress"], est["mu_stress"],
                     _se("mu_stress"),
                     "M9 CSV row (mu0); yaml explicitly zeros mu_i (centering "
                     "identity absorbs them)"))
    # alphas -- DERIVED, so inherited from the yaml root-finding output
    rows.append(_row("alpha01", float(yaml_p.alpha01), est["alpha01"],
                     _se("alpha01"),
                     "yaml DERIVED (occupancy/duration root-finding); "
                     "no MLE counterpart in the bundle"))
    rows.append(_row("alpha10", float(yaml_p.alpha10), est["alpha10"],
                     _se("alpha10"),
                     "yaml DERIVED (occupancy/duration root-finding); "
                     "no MLE counterpart in the bundle"))
    # gammas -- MS-AR transition CSV row (label swapped for yaml convention)
    rows.append(_row("gamma01", float(yaml_p.gamma01), est["gamma01"],
                     _se("gamma01"),
                     "M9 transition_coefficients.csv (yaml swap of raw gamma10, "
                     "normal->stress direction)"))
    rows.append(_row("gamma10", float(yaml_p.gamma10), est["gamma10"],
                     _se("gamma10"),
                     "M9 transition_coefficients.csv (yaml swap of raw gamma01, "
                     "stress->normal direction)"))
    return pd.DataFrame(rows)


def _transform_se_sigma_normal(res) -> Optional[float]:
    """Delta-method SE for sigma_normal in natural space."""
    se = res.se_hessian
    if se is None or not math.isfinite(se[2]):
        return None
    return float(res.params.sigma_normal * se[2])


def _transform_se_sigma_stress(res) -> Optional[float]:
    se = res.se_hessian
    if se is None or not math.isfinite(se[3]):
        return None
    diff = res.params.sigma_stress - res.params.sigma_normal
    dsig = 1.0 - math.exp(-diff)
    return float(math.hypot(res.params.sigma_normal * se[2],
                            dsig * se[3]))


# --------------------------------------------------------------------------
# Price impact reprice + parameter-wise decomposition
# --------------------------------------------------------------------------
STRIKES = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0)
MATURITIES_H = (24, 48, 72)


def _model_with(sigma_normal: float, sigma_stress: float, kappa: float,
                alpha01: float, gamma01: float, alpha10: float,
                gamma10: float) -> ForwardCenteredModel:
    params_yaml = load_frozen_parameters(YAML_PATH)
    quotes = load_quotes(REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv")
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=params_yaml.spot_price_TRY_MWh)
    spec = ResidualSpec(
        kappa_per_hour=kappa,
        sigma_y=np.array([sigma_normal, sigma_stress]),
        scale_P=282.48, regime_means=np.zeros(2),
        mode="additive",
    )
    return ForwardCenteredModel(
        curve=curve, spec=spec,
        tvtp=TVTPCoefficients(alpha01, gamma01, alpha10, gamma10),
        pi_filtered=params_yaml.pi_filtered,
        valuation_utc=params_yaml.valuation_utc,
        spot_price_TRY_MWh=params_yaml.spot_price_TRY_MWh)


def _price_grid(model, label: str) -> pd.DataFrame:
    rows = []
    for K in STRIKES:
        for T in MATURITIES_H:
            contract = EuropeanOption(
                "call", float(K), model.valuation_utc,
                model.valuation_utc + pd.Timedelta(hours=int(T)),
                r_annual=0.40)
            gs = ResidualGridSettings(n_space_nodes=1201)
            z_fn = climatology_z_lagged_fn(model, contract, gs)
            res = price_forward_centered(model, contract, grid_settings=gs,
                                         z_lagged_fn=z_fn)
            rows.append({"label": label, "strike": K, "maturity_h": T,
                         "V_call": float(res.value),
                         "residual_sd_T": float(res.residual_std_at_expiry)})
    return pd.DataFrame(rows)


def reprice_and_decompose(res_profile, kappa_from_des: float) -> Tuple[pd.DataFrame,
                                                                        pd.DataFrame]:
    """Reprice under production, FW9-consistent, and intermediate sets to
    decompose the FW9 - production delta into sigma / kappa / transition
    contributions."""
    yaml_p = load_frozen_parameters(YAML_PATH)

    # inherited (production) params
    inh_sigma_n = float(yaml_p.sigma_y[0])
    inh_sigma_s = float(yaml_p.sigma_y[1])
    inh_kappa = float(yaml_p.kappa_per_hour)
    inh_a01 = float(yaml_p.alpha01); inh_g01 = float(yaml_p.gamma01)
    inh_a10 = float(yaml_p.alpha10); inh_g10 = float(yaml_p.gamma10)

    # FW9-consistent params
    fw9_sigma_n = float(res_profile.params.sigma_normal)
    fw9_sigma_s = float(res_profile.params.sigma_stress)
    fw9_kappa = float(kappa_from_des)
    fw9_a01 = float(res_profile.params.alpha01)
    fw9_g01 = float(res_profile.params.gamma01)
    fw9_a10 = float(res_profile.params.alpha10)
    fw9_g10 = float(res_profile.params.gamma10)

    variants = [
        ("A_inherited", inh_sigma_n, inh_sigma_s, inh_kappa,
         inh_a01, inh_g01, inh_a10, inh_g10),
        # swap SIGMA to FW9
        ("B_inh_kappa_inh_trans_fw9_sigma", fw9_sigma_n, fw9_sigma_s, inh_kappa,
         inh_a01, inh_g01, inh_a10, inh_g10),
        # swap KAPPA to FW9 (from des)
        ("C_inh_sigma_inh_trans_fw9_kappa", inh_sigma_n, inh_sigma_s, fw9_kappa,
         inh_a01, inh_g01, inh_a10, inh_g10),
        # swap TRANSITION to FW9
        ("D_inh_sigma_inh_kappa_fw9_trans", inh_sigma_n, inh_sigma_s, inh_kappa,
         fw9_a01, fw9_g01, fw9_a10, fw9_g10),
        # full FW9
        ("E_fw9_consistent", fw9_sigma_n, fw9_sigma_s, fw9_kappa,
         fw9_a01, fw9_g01, fw9_a10, fw9_g10),
    ]
    frames = []
    for label, sn, ss, kappa, a01, g01, a10, g10 in variants:
        print(f"pricing variant {label} (kappa={kappa:.5g}) ...")
        t0 = time.time()
        m = _model_with(sn, ss, kappa, a01, g01, a10, g10)
        frames.append(_price_grid(m, label))
        print(f"  {label} time {time.time()-t0:.1f}s")
    grid = pd.concat(frames, ignore_index=True)
    grid.to_csv(OUT / "price_impact_v2_grid.csv", index=False)

    pivot = grid.pivot_table(index=["strike", "maturity_h"],
                             columns="label", values="V_call").reset_index()
    for c in ("B_inh_kappa_inh_trans_fw9_sigma",
              "C_inh_sigma_inh_trans_fw9_kappa",
              "D_inh_sigma_inh_kappa_fw9_trans",
              "E_fw9_consistent"):
        pivot[f"delta_{c}_vs_A"] = pivot[c] - pivot["A_inherited"]
    pivot.to_csv(OUT / "price_impact_v2_decomposition.csv", index=False)
    return grid, pivot


# --------------------------------------------------------------------------
# Candidate yaml rewrite
# --------------------------------------------------------------------------
def write_candidate_yaml(res_profile, kappa_from_des: float, phi_des: float,
                         phi_des_se: float) -> Path:
    est = res_profile.params.as_dict()
    yml = REPO_ROOT / "inputs" / "historical" / "fw9_reestimated_parameters.yaml"
    body = [
        "# =============================================================================",
        "# FW9 self-estimated MS-AR(1) TVTP parameters (CANDIDATE; not for production)",
        "# =============================================================================",
        "# Consistent parameter set assembled from the FW9b work package:",
        "#   * sigmas / alphas / gammas: profile fit at phi = argmax with proper SEs",
        "#     (outputs/fw9_self_estimation/profile_phi_TVTP_1cov_argmax.pkl)",
        "#   * phi and kappa: from the deseasonalised single-regime AR(1) fit",
        "#     (outputs/fw9_self_estimation/deseasonalized_ar1.json)",
        "# The frozen production yaml at inputs/historical/m2_frozen_parameters.yaml",
        "# is UNCHANGED; this file is an audit artefact of the FW9b re-estimation.",
        "# =============================================================================",
        'description: "FW9b independent re-estimation with proper boundary handling and object-matched inheritance"',
        'model: "MS-AR(1) with TVTP; 1 covariate = RD_lag1; index 0 = normal, index 1 = stress"',
        "",
        "scale_P: 282.48                # inherited yaml value; FW9 diagnoses window sensitivity",
        "",
        "# --- mean reversion (from the deseasonalised AR(1) fit, FW9 §2) ------------",
        f"phi: {phi_des:.10f}",
        f"kappa_per_hour: {kappa_from_des:.10f}",
        f"half_life_hours: {math.log(2.0) / kappa_from_des:.6f}",
        "",
        "# --- regime volatilities (profile fit at phi=argmax, FW9 §1c) --------------",
        f"sigma_y_normal: {est['sigma_normal']:.10f}",
        f"sigma_y_stress: {est['sigma_stress']:.10f}",
        "",
        "# --- TVTP logistic transition coefficients (profile fit) -------------------",
        "tvtp:",
        f"  alpha01: {est['alpha01']:.10f}",
        f"  gamma01: {est['gamma01']:.10f}",
        f"  alpha10: {est['alpha10']:.10f}",
        f"  gamma10: {est['gamma10']:.10f}",
        "  covariate: RD_WS_standardized_lag1h",
        "  placeholder: false",
        "",
        "# --- boundary flag ---------------------------------------------------------",
        "# phi_MS_AR from the free-phi MLE lands on the unit-root boundary in every start.",
        "# The value shipped in this candidate yaml is the DESEASONALISED phi, which sits",
        "# in the interior and has a proper Hessian-based standard error.",
        "boundary_flags:",
        "  phi_MS_AR_boundary: true",
        "  phi_deseasonalized_boundary: false",
        "",
        "# --- provenance ------------------------------------------------------------",
        "provenance:",
        "  sigmas_source: profile fit at phi=argmax (FW9b §1c)",
        "  alphas_source: MLE at phi=argmax profile node (FW9b §1c)",
        "  gammas_source: MLE at phi=argmax profile node (FW9b §1c)",
        "  phi_source: deseasonalised single-regime AR(1) (FW9b §2)",
        "  kappa_source: transformed from deseasonalised phi",
        "  n_obs: 61368",
        '  estimation_window_start_utc: "2018-12-31T21:00:00+00:00"',
        '  estimation_window_end_utc: "2025-12-31T20:00:00+00:00"',
        "  provenance_tag: Calibrated (FW9b independent MLE)",
        "",
        "standard_errors:",
    ]
    slot = {"mu_normal": 0, "mu_stress": 1, "sigma_normal": 2,
            "sigma_stress": 3, "phi": 4, "alpha01": 5, "gamma01": 6,
            "alpha10": 7, "gamma10": 8}
    se_map = {
        "mu_normal_se": res_profile.se_hessian[slot["mu_normal"]],
        "mu_stress_se": res_profile.se_hessian[slot["mu_stress"]],
        "sigma_normal_se": _transform_se_sigma_normal(res_profile),
        "sigma_stress_se": _transform_se_sigma_stress(res_profile),
        "phi_deseasonalized_se": phi_des_se,
        "alpha01_se": res_profile.se_hessian[slot["alpha01"]],
        "gamma01_se": res_profile.se_hessian[slot["gamma01"]],
        "alpha10_se": res_profile.se_hessian[slot["alpha10"]],
        "gamma10_se": res_profile.se_hessian[slot["gamma10"]],
    }
    for k, v in se_map.items():
        if v is None or not math.isfinite(v):
            body.append(f"  {k}: null")
        else:
            body.append(f"  {k}: {v:.6e}")
    yml.write_text("\n".join(body) + "\n", encoding="utf-8")
    return yml


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "profile_phi_TVTP_1cov_argmax.pkl"
    if not p.exists():
        raise SystemExit(f"Missing {p}; run scripts/fw9/profile_phi.py --only tvtp1cov first.")
    with open(p, "rb") as f:
        blob = pickle.load(f)
    res_profile = blob["res"]
    print(f"Profile argmax phi = {blob['best_phi']}")
    print(f"Profile fit converged = {res_profile.converged}, "
          f"grad_norm = {res_profile.grad_norm:.3g}")

    des = json.loads((OUT / "deseasonalized_ar1.json").read_text())
    kappa_des = float(des["kappa_per_hour"])
    phi_des = float(des["phi"])
    phi_des_se = float(des["se_phi"])

    cmp = build_comparison_v2(res_profile)
    cmp.to_csv(OUT / "parameter_comparison_v2.csv", index=False)

    md_lines = ["# FW9b §3 -- Object-matched parameter comparison (T8)\n",
                "* `inherited_from` says which OBJECT the inherited value is",
                "  the natural comparison for; the FW9 estimate is compared",
                "  against the OBJECT-MATCHED inherited value.",
                "* Boundary-flagged rows (phi_MS_AR) do not carry SE/t/CI --",
                "  the Hessian-based estimator is not interpretable at the",
                "  unit-root boundary; see FW9b §1 profile analysis.",
                ""]
    md_lines.append("| par | inherited from | inherited | FW9 est | SE | t | 95 % low | 95 % high | in FW9 95 %? | comment |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|:-:|---|")
    for r in cmp.itertuples():
        def _fmt(v, fmt="{:.6g}"):
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return "--"
            if isinstance(v, bool):
                return "yes" if v else "**NO**"
            try:
                return fmt.format(v)
            except (ValueError, TypeError):
                return str(v)
        md_lines.append(
            f"| {r.parameter} | {r.inherited_from} | {_fmt(r.inherited)} | "
            f"{_fmt(r.fw9_estimate)} | {_fmt(r.fw9_se)} | "
            f"{_fmt(r.t_stat_vs_inherited, '{:.3g}')} | "
            f"{_fmt(r.fw9_95pct_low)} | {_fmt(r.fw9_95pct_high)} | "
            f"{_fmt(r.inherited_in_fw9_95pct)} | "
            f"{(r.comment or '').replace('|','/')} |"
        )
    (OUT / "parameter_comparison_v2.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8")

    grid, pivot = reprice_and_decompose(res_profile, kappa_des)
    print("\nprice_impact_v2 decomposition (pivot):")
    print(pivot.round(3).to_string(index=False))

    yml = write_candidate_yaml(res_profile, kappa_des, phi_des, phi_des_se)
    print("\nwrote", yml)


if __name__ == "__main__":
    main()
