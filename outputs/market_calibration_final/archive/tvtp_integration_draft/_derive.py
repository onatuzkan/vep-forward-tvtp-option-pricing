"""Derive yaml-convention TVTP parameters from the M9 raw fit + duration targets.

Nothing outside outputs/tvtp_integration_draft/ is touched.

  Raw M9 (parameter_estimates.csv, transition_coefficients.csv, run_summary
  block inside metadata JSON):

    raw state 0 = HIGH-vol, sigma0_raw = 0.09240665,
      occupancy_raw_0 = 0.6746,  mean_duration_raw_0 = 7.559716 h
    raw state 1 = LOW-vol,  sigma1_raw = 0.00353481,
      occupancy_raw_1 = 0.3254,  mean_duration_raw_1 = 3.759199 h

    RD_lag1 slopes on the RAW state indexing:
      gamma01_raw = +0.07769839  (logit slope of raw state 0 -> state 1)
      gamma10_raw = -0.58377808  (logit slope of raw state 1 -> state 0)

  Yaml convention (m2_frozen_parameters.yaml / params_frozen.py invariant):
    index 0 = normal (low sigma),  index 1 = stress (high sigma).

  Mapping (Step 1 in the user's brief):
    new_sigma_normal = sigma1_raw   = 0.00353481
    new_sigma_stress = sigma0_raw   = 0.09240665
    new_occupancy_normal = 0.3254   new_duration_normal = 3.759199 h
    new_occupancy_stress = 0.6746   new_duration_stress = 7.559716 h
    new_gamma01 (normal->stress) = gamma10_raw = -0.58377808
    new_gamma10 (stress->normal) = gamma01_raw = +0.07769839

  Step 2: solve
    E_z[logistic(alpha01 + new_gamma01 * z)] = 1 / new_duration_normal
    E_z[logistic(alpha10 + new_gamma10 * z)] = 1 / new_duration_stress
  with z the historical rd_standardized.csv column (87665 hours).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

# --- inputs from the M9 bundle -------------------------------------------------
SIG0_RAW = 0.09240665439799690     # parameter_estimates.csv M9, sigma0
SIG1_RAW = 0.00353480696422237     # parameter_estimates.csv M9, sigma1
OCC0_RAW = 0.6745780174292473      # metadata json run_summary.occupancy[0]
OCC1_RAW = 0.32542198257074967
DUR0_RAW = 7.559715945771465       # mean_duration_state0
DUR1_RAW = 3.7591994835377665      # mean_duration_state1
GAMMA01_RAW = 0.07769839363017603  # transition_coefficients.csv M9, RD_lag1
GAMMA10_RAW = -0.5837780825414551

# yaml-convention swap
NEW_SIGMA_NORMAL = SIG1_RAW
NEW_SIGMA_STRESS = SIG0_RAW
NEW_OCC_NORMAL   = OCC1_RAW
NEW_OCC_STRESS   = OCC0_RAW
NEW_DUR_NORMAL   = DUR1_RAW
NEW_DUR_STRESS   = DUR0_RAW
NEW_GAMMA01 = GAMMA10_RAW          # normal -> stress
NEW_GAMMA10 = GAMMA01_RAW          # stress -> normal

# --- load z series -------------------------------------------------------------
z_series = pd.read_csv("inputs/historical/rd_standardized.csv")
z = z_series["z"].to_numpy(dtype=float)
assert z.size == 87665, f"z has {z.size} rows, expected 87665"


def mean_p(alpha: float, gamma: float) -> float:
    """Sample average of logistic(alpha + gamma * z) over the historical z."""
    return float(np.mean(expit(alpha + gamma * z)))


def solve_alpha(gamma: float, target_rate: float) -> tuple[float, float]:
    """Return (alpha, achieved_mean) such that mean_p(alpha, gamma) = target_rate."""
    f = lambda a: mean_p(a, gamma) - target_rate  # noqa: E731
    # mean_p is strictly increasing in alpha; wide bracket is fine.
    a = brentq(f, -20.0, 20.0, xtol=1e-12, rtol=1e-12)
    return a, mean_p(a, gamma)


target_p01_mean = 1.0 / NEW_DUR_NORMAL          # normal -> stress avg hourly prob
target_p10_mean = 1.0 / NEW_DUR_STRESS          # stress -> normal

alpha01_new, achieved_p01 = solve_alpha(NEW_GAMMA01, target_p01_mean)
alpha10_new, achieved_p10 = solve_alpha(NEW_GAMMA10, target_p10_mean)

# --- cross-check: stationary occupancy along the empirical z path -----------
# 1. Local (time-varying) average-rate approximation
p01_series = expit(alpha01_new + NEW_GAMMA01 * z)
p10_series = expit(alpha10_new + NEW_GAMMA10 * z)
mean_p01 = float(np.mean(p01_series))
mean_p10 = float(np.mean(p10_series))
# ergodic pi for average-rate Markov chain
implied_pi_stress_ergodic = mean_p01 / (mean_p01 + mean_p10)
implied_pi_normal_ergodic = 1.0 - implied_pi_stress_ergodic

# 2. Chain simulation along the actual z time series (10 replicates)
rng = np.random.default_rng(20260906)
sim_occ_stress = []
n_reps = 10
for _ in range(n_reps):
    s = 1 if rng.random() < NEW_OCC_STRESS else 0
    time_in_stress = 0
    for i in range(z.size):
        if s == 0:
            if rng.random() < p01_series[i]:
                s = 1
        else:
            if rng.random() < p10_series[i]:
                s = 0
        if s == 1:
            time_in_stress += 1
    sim_occ_stress.append(time_in_stress / z.size)
implied_pi_stress_sim = float(np.mean(sim_occ_stress))
implied_pi_stress_sim_sd = float(np.std(sim_occ_stress, ddof=1))

# --- consistency invariant check --------------------------------------------
sig_normal_lt_sig_stress = NEW_SIGMA_NORMAL < NEW_SIGMA_STRESS

# --- write yaml + report -----------------------------------------------------
out_dir = Path("outputs/tvtp_integration_draft")
yaml_body = textwrap.dedent(f"""\
    # =============================================================================
    # DRAFT — Derived TVTP parameters under the yaml convention (index0=normal,
    # index1=stress).  Produced by outputs/tvtp_integration_draft/_derive.py.
    # NOT a production artefact.  Do not merge into inputs/historical/ without an
    # explicit review sign-off.
    # =============================================================================

    model: M9_student_t_tvtp_TVTP-2 (single-covariate reduction: RD_lag1 only)
    convention: "index 0 = normal (low sigma), index 1 = stress (high sigma)"
    covariate: RD_WS_standardized_lag1h

    # --- regime volatilities (per sqrt(hour), y = asinh(P / scale_P)) ----------
    # Discrete AR(1) sigma_eps; near-unit-root kappa makes continuous OU sigma
    # numerically identical to two decimals.
    sigma_y_normal: {NEW_SIGMA_NORMAL:.10f}       # = raw M9 sigma1 (low-vol state)
    sigma_y_stress: {NEW_SIGMA_STRESS:.10f}       # = raw M9 sigma0 (high-vol state)

    # --- TVTP logistic transition coefficients (yaml convention) --------------
    tvtp:
      alpha01: {alpha01_new:.10f}      # derived by root-finding
      gamma01: {NEW_GAMMA01:.10f}      # = raw M9 gamma10 (state1->state0 raw = normal->stress in yaml)
      alpha10: {alpha10_new:.10f}      # derived by root-finding
      gamma10: {NEW_GAMMA10:.10f}      # = raw M9 gamma01 (state0->state1 raw = stress->normal in yaml)
      placeholder: false

    # --- yaml-convention diagnostics (from raw M9, swap-applied) --------------
    occupancy_normal: {NEW_OCC_NORMAL:.6f}     # = raw M9 occupancy_state1
    occupancy_stress: {NEW_OCC_STRESS:.6f}     # = raw M9 occupancy_state0
    mean_duration_normal_hours: {NEW_DUR_NORMAL:.6f}
    mean_duration_stress_hours: {NEW_DUR_STRESS:.6f}

    # --- consistency check (target 0 <= diff <= 0.05 for a good fit) -----------
    consistency_check:
      target_avg_p01_normal_to_stress: {target_p01_mean:.6f}
      achieved_avg_p01: {achieved_p01:.6f}
      target_avg_p10_stress_to_normal: {target_p10_mean:.6f}
      achieved_avg_p10: {achieved_p10:.6f}
      implied_pi_stress_ergodic_avg_rate: {implied_pi_stress_ergodic:.6f}   # target {NEW_OCC_STRESS:.6f}
      implied_pi_stress_simulated_mean:  {implied_pi_stress_sim:.6f}
      implied_pi_stress_simulated_sd:    {implied_pi_stress_sim_sd:.6f}
      simulation_reps: {n_reps}
      simulation_seed: 20260906

    provenance:
      sigma_y_normal:  "raw M9 parameter_estimates.csv sigma1 (state1=low-vol)"
      sigma_y_stress:  "raw M9 parameter_estimates.csv sigma0 (state0=high-vol)"
      gamma01:         "raw M9 transition_coefficients.csv gamma10 (RD_lag1); swap enforces yaml convention"
      gamma10:         "raw M9 transition_coefficients.csv gamma01 (RD_lag1); swap enforces yaml convention"
      alpha01:         "DERIVED: brentq root-finding, target E_z[logistic(a+gamma01*z)] = 1/mean_duration_normal, z = inputs/historical/rd_standardized.csv (87665 hours)"
      alpha10:         "DERIVED: brentq root-finding, target E_z[logistic(a+gamma10*z)] = 1/mean_duration_stress"
      occupancy_normal: "raw M9 occupancy[1] (yaml swap)"
      occupancy_stress: "raw M9 occupancy[0] (yaml swap)"

    invariant_check:
      sigma_normal_lt_sigma_stress: {str(sig_normal_lt_sig_stress).lower()}
      # params_frozen.py FrozenM2Parameters.__post_init__ requires True.
""")

(out_dir / "derived_tvtp_parameters.yaml").write_text(yaml_body, encoding="utf-8")

summary = {
    "raw_m9": {
        "sigma0_state0_high_vol":  SIG0_RAW,
        "sigma1_state1_low_vol":   SIG1_RAW,
        "occupancy_state0":        OCC0_RAW,
        "occupancy_state1":        OCC1_RAW,
        "mean_duration_state0_h":  DUR0_RAW,
        "mean_duration_state1_h":  DUR1_RAW,
        "gamma01_RD_lag1_raw":     GAMMA01_RAW,
        "gamma10_RD_lag1_raw":     GAMMA10_RAW,
    },
    "yaml_convention": {
        "sigma_y_normal":  NEW_SIGMA_NORMAL,
        "sigma_y_stress":  NEW_SIGMA_STRESS,
        "occupancy_normal": NEW_OCC_NORMAL,
        "occupancy_stress": NEW_OCC_STRESS,
        "gamma01":  NEW_GAMMA01,
        "gamma10":  NEW_GAMMA10,
        "alpha01_derived":  alpha01_new,
        "alpha10_derived":  alpha10_new,
    },
    "cross_check": {
        "target_avg_p01": target_p01_mean,
        "achieved_avg_p01": achieved_p01,
        "target_avg_p10": target_p10_mean,
        "achieved_avg_p10": achieved_p10,
        "implied_pi_stress_ergodic": implied_pi_stress_ergodic,
        "implied_pi_stress_simulated": implied_pi_stress_sim,
        "implied_pi_stress_simulated_sd": implied_pi_stress_sim_sd,
        "target_pi_stress": NEW_OCC_STRESS,
        "abs_pct_error_ergodic": 100.0 * abs(implied_pi_stress_ergodic - NEW_OCC_STRESS) / NEW_OCC_STRESS,
        "abs_pct_error_simulated": 100.0 * abs(implied_pi_stress_sim - NEW_OCC_STRESS) / NEW_OCC_STRESS,
    },
    "sign_comparison_with_old_placeholder": {
        "old_gamma01": -0.680174,
        "new_gamma01": NEW_GAMMA01,
        "sign_agrees_gamma01": (NEW_GAMMA01 < 0) == (-0.680174 < 0),
        "abs_ratio_gamma01":  abs(NEW_GAMMA01) / abs(-0.680174),
        "old_gamma10": 0.203006,
        "new_gamma10": NEW_GAMMA10,
        "sign_agrees_gamma10": (NEW_GAMMA10 > 0) == (0.203006 > 0),
        "abs_ratio_gamma10":  abs(NEW_GAMMA10) / abs(0.203006),
    },
    "invariant": {
        "sigma_normal_lt_sigma_stress": sig_normal_lt_sig_stress,
        "params_frozen_invariant_would_pass": sig_normal_lt_sig_stress,
    },
}

with open(out_dir / "derivation_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

print("=== KEY NUMBERS ===")
print(f"  new_sigma_normal = {NEW_SIGMA_NORMAL:.8f}   (raw sigma1)")
print(f"  new_sigma_stress = {NEW_SIGMA_STRESS:.8f}   (raw sigma0)")
print(f"  new_gamma01      = {NEW_GAMMA01:+.6f}  (from raw gamma10)")
print(f"  new_gamma10      = {NEW_GAMMA10:+.6f}  (from raw gamma01)")
print(f"  alpha01 derived  = {alpha01_new:+.6f}  target E[p01]={target_p01_mean:.5f}  achieved={achieved_p01:.5f}")
print(f"  alpha10 derived  = {alpha10_new:+.6f}  target E[p10]={target_p10_mean:.5f}  achieved={achieved_p10:.5f}")
print()
print(f"  invariant sigma_normal < sigma_stress: {sig_normal_lt_sig_stress}")
print()
print("=== IMPLIED STATIONARY STRESS OCCUPANCY (target 0.6746) ===")
print(f"  ergodic avg-rate approx   = {implied_pi_stress_ergodic:.4f}   ({100*(implied_pi_stress_ergodic-NEW_OCC_STRESS)/NEW_OCC_STRESS:+.2f}%)")
print(f"  Monte Carlo simulation    = {implied_pi_stress_sim:.4f} +/- {implied_pi_stress_sim_sd:.4f}   ({100*(implied_pi_stress_sim-NEW_OCC_STRESS)/NEW_OCC_STRESS:+.2f}%)")
print()
print("=== SIGN vs OLD PLACEHOLDER ===")
print(f"  gamma01: old {-0.680174:+.5f}   new {NEW_GAMMA01:+.5f}   |new|/|old| = {abs(NEW_GAMMA01)/0.680174:.3f}")
print(f"  gamma10: old {0.203006:+.5f}   new {NEW_GAMMA10:+.5f}   |new|/|old| = {abs(NEW_GAMMA10)/0.203006:.3f}")
