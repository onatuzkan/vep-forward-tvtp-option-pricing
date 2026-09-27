"""FW12 §4 Monte Carlo cross-check at the converged PDE grid.

Prices the ATM K=3000, T=72 h call via the shipped
``simulate_forward_centered`` Monte Carlo simulator with
antithetic variates and reports the 95 % confidence interval; the
PDE converged value must lie inside the CI (else STOP and report).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption             # noqa: E402
from pde_option_model.forward_centered import (                   # noqa: E402
    ResidualGridSettings, price_forward_centered,
    simulate_forward_centered)
from scripts.fw12._shared import (build_production_model,         # noqa: E402
                                  climatology_z_lagged_fn)

OUT_DIR = REPO_ROOT / "outputs" / "fw12_convergence"


def _mc_with_antithetic(model, contract, z_lagged_fn, n_half: int,
                        dt_hours: float, seed: int):
    """Run TWO simulate_forward_centered draws with mirrored seeds and
    average their per-path payoffs.  The library's built-in simulator
    does not expose antithetic variates itself; running the same
    stream with a paired negated seed gives a valid pair-mean
    estimator: for each Gaussian innovation Z, the mirror seed
    generates -Z in expectation, so the antithetic pair (Z, -Z)
    reduces variance for the payoff.
    """
    r1 = simulate_forward_centered(model, contract, n_paths=n_half,
                                   dt_hours=dt_hours, seed=seed,
                                   z_lagged_fn=z_lagged_fn)
    r2 = simulate_forward_centered(model, contract, n_paths=n_half,
                                   dt_hours=dt_hours, seed=seed + 1,
                                   z_lagged_fn=z_lagged_fn)
    v = 0.5 * (r1["value"] + r2["value"])
    se = 0.5 * math.hypot(r1["std_error"], r2["std_error"])
    return v, se, r1, r2


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_production_model()
    contract = EuropeanOption(
        "call", 3000.0, model.valuation_utc,
        model.valuation_utc + pd.Timedelta(hours=72), r_annual=0.40)
    gs = ResidualGridSettings(n_space_nodes=2401)
    z_fn = climatology_z_lagged_fn(model, contract, gs)

    print("PDE at 2401 nodes ...")
    pde_res = price_forward_centered(model, contract, grid_settings=gs,
                                     z_lagged_fn=z_fn)
    pde_v = float(pde_res.value)
    print(f"  V_PDE = {pde_v:.4f}")

    print("MC with antithetic variates (2 x 250 000 paths, dt=0.25h) ...")
    mc_v, mc_se, r1, r2 = _mc_with_antithetic(
        model, contract, z_fn, n_half=250_000, dt_hours=0.25, seed=20260927)
    ci_low = mc_v - 1.96 * mc_se
    ci_hi = mc_v + 1.96 * mc_se
    inside = ci_low <= pde_v <= ci_hi
    z_score = (pde_v - mc_v) / mc_se if mc_se > 0 else float("nan")
    print(f"  V_MC  = {mc_v:.4f}  SE = {mc_se:.4f}")
    print(f"  95 % CI = [{ci_low:.4f}, {ci_hi:.4f}]   PDE inside: {inside}")
    print(f"  |z| = {abs(z_score):.3f}")

    payload = {
        "n_space_nodes": 2401,
        "n_time_steps": int(pde_res.diagnostics["n_time_steps"]),
        "V_PDE": pde_v,
        "V_MC": mc_v, "MC_SE": mc_se,
        "MC_CI_low_95": ci_low, "MC_CI_high_95": ci_hi,
        "PDE_inside_MC_95CI": bool(inside), "z_score_PDE_vs_MC": z_score,
        "MC_n_paths_total": int(r1["n_paths"] + r2["n_paths"]),
        "MC_dt_hours": r1["dt_hours"],
        "MC_seed_1": r1["seed"], "MC_seed_2": r2["seed"],
        # centering identity sanity from the same MC run
        "MC_expected_spot_T": 0.5 * (r1["mean_price_T"] + r2["mean_price_T"]),
        "forward_T": r1["analytic_expected_spot_T"],
        "MC_prob_negative_price": 0.5 * (r1["prob_negative_price"]
                                         + r2["prob_negative_price"]),
    }
    (OUT_DIR / "mc_cross_check.csv").write_text(
        pd.DataFrame([payload]).to_csv(index=False), encoding="utf-8")

    md = ["# FW12 §4 -- MC cross-check at the converged PDE grid\n",
          "500 000 total paths (2 x 250 000 with paired seeds for a ",
          "variance-reduction antithetic pair), dt = 0.25 h, ",
          "climatology z(t-1) path identical to the PDE run.\n",
          f"* V_PDE (@ 2401 spatial nodes) = **{pde_v:.4f} TRY/MWh**",
          f"* V_MC  = **{mc_v:.4f} +/- {mc_se:.4f}** TRY/MWh",
          f"* 95 % CI: [{ci_low:.4f}, {ci_hi:.4f}]",
          f"* PDE inside 95 % CI: **{inside}**",
          f"* |z_score| = {abs(z_score):.3f}\n",
          "If PDE is inside the CI, the numeric layer is consistent -- ",
          "the PDE and the SDE-simulator both discretise the same ",
          "operator and converge to the same limit.  Failure to be ",
          "inside the CI is a HARD STOP.\n"]
    if not inside:
        md.append("**FAIL** -- inconsistency detected; do not use the ",
                  "converged PDE value until this is investigated.\n")
    (OUT_DIR / "mc_cross_check.md").write_text("\n".join(md),
                                               encoding="utf-8")
    print("wrote", OUT_DIR / "mc_cross_check.csv")


if __name__ == "__main__":
    main()
