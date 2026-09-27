"""FW12 §1 spatial-convergence sweep at production settings.

Fixes climatology z path, r=0.40, n_std=6, boundary=gamma_zero (defaults);
varies ``n_space_nodes`` over {301, 601, 1201, 2401, 4801}; prices ATM
K=3000, T={24, 48, 72} h call.  Reports V_h, the successive
differences, the observed order p = log2(|Δ_h| / |Δ_{h/2}|), and a
Richardson extrapolation to the converged limit; the tabulated
relative error at 1201 nodes is what the manuscript's Appendix C will
cite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fw12._shared import (build_production_model,        # noqa: E402
                                  observed_order, price_at,
                                  richardson_estimate)

OUT_DIR = REPO_ROOT / "outputs" / "fw12_convergence"
STRIKES = [3000.0]
MATURITIES_H = [24, 48, 72]
NODES = [301, 601, 1201, 2401, 4801]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_production_model()
    rows = []
    for K in STRIKES:
        for T in MATURITIES_H:
            prev = None
            per_node = {}
            for n in NODES:
                v, info = price_at(model, K, T, "call", n, None, n_std=6.0)
                diff = None if prev is None else v - prev
                per_node[n] = v
                rows.append({
                    "strike": K, "maturity_h": T, "n_space_nodes": n,
                    "n_time_steps": info["n_time_steps"],
                    "V_call": v,
                    "diff_from_prev": diff,
                    "residual_sd_T": info["residual_sd_T"],
                    "p_stress_T": info["p_stress_at_expiry"],
                })
                prev = v
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "spatial_convergence.csv", index=False)

    md = ["# FW12 §1 -- Spatial convergence sweep\n",
          "Same climatology z(t-1) path as the production `run_pde.py",
          "price` command (train_end 2022-12-31 20:00 UTC, lag 1 h),",
          "same `n_std = 6.0`, same discount `r_annual = 0.40`, and no",
          "measure adjustment.  Only `n_space_nodes` varies.  The",
          "grid halves each step so the observed order `p =",
          "log2(|V_h - V_{h/2}| / |V_{h/2} - V_{h/4}|)` is meaningful.\n",
          "## Raw values\n",
          df.round(6).to_markdown(index=False)]

    md.append("\n\n## Order + Richardson estimate per maturity\n")
    md.append("| K | T (h) | V_301 | V_601 | V_1201 | V_2401 | V_4801 | "
              "p_601->1201 | p_1201->2401 | V_star (Richardson from 2401/4801) | "
              "|V_1201 - V_star| | rel err at 1201 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for K in STRIKES:
        for T in MATURITIES_H:
            sub = df[(df["strike"] == K) & (df["maturity_h"] == T)] \
                .sort_values("n_space_nodes")
            vs = sub["V_call"].tolist()
            V_301, V_601, V_1201, V_2401, V_4801 = vs
            p1 = observed_order(V_301, V_601, V_1201)
            p2 = observed_order(V_601, V_1201, V_2401)
            # Richardson uses the finest available pair with the observed
            # order at that end.  Use p_1201->2401 with the pair (2401, 4801).
            V_star = richardson_estimate(V_2401, V_4801, p2)
            rel = abs(V_1201 - V_star) / max(abs(V_star), 1e-9)
            md.append(f"| {K:.0f} | {T} | {V_301:.4f} | {V_601:.4f} | "
                      f"{V_1201:.4f} | {V_2401:.4f} | {V_4801:.4f} | "
                      f"{p1:.3f} | {p2:.3f} | {V_star:.4f} | "
                      f"{abs(V_1201 - V_star):.4f} | {rel * 100:.4f} % |")
    md.append("\n\nTheoretical order for Crank-Nicolson on a uniform "
              "residual grid is O(h^2) in space.  The observed order is "
              "reported above; departures from 2.0 typically reflect the "
              "gamma-zero boundary contribution (first-order in the far "
              "field for OTM payoffs), the mixture of an irregular payoff "
              "kink at K, and the Crank-Nicolson scheme's O(k^2) time "
              "component intermingling with the spatial one at this "
              "solver's default time step.  §2 fixes the spatial grid at "
              "the finest available level and varies time steps to "
              "separate the two.\n")
    (OUT_DIR / "spatial_convergence.md").write_text("\n".join(md),
                                                    encoding="utf-8")
    print("wrote", OUT_DIR / "spatial_convergence.csv")
    print("wrote", OUT_DIR / "spatial_convergence.md")


if __name__ == "__main__":
    main()
