"""FW12 §2 time-step convergence sweep + spatial-time cross check."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fw12._shared import (build_production_model,        # noqa: E402
                                  observed_order, price_at)

OUT_DIR = REPO_ROOT / "outputs" / "fw12_convergence"
FIXED_NODES = 2401
STRIKE = 3000.0
MATURITIES_H = [24, 48, 72]
# n_time_steps = k * tau_hours means k solver steps per contract hour.
# Production default is k=2 (via `n_steps(tau) = max(96, ceil(2*tau))`),
# floored at 96 for tau <= 48 h.  We sweep k in {1, 2, 4, 8, 16} scaled
# by tau to give an even four-decade log2 span.
K_STEP_MULTS = [1, 2, 4, 8, 16]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_production_model()
    rows = []
    for T in MATURITIES_H:
        for k in K_STEP_MULTS:
            n_time = max(96, k * T)
            v, info = price_at(model, STRIKE, T, "call", FIXED_NODES,
                               n_time_steps=n_time)
            rows.append({"strike": STRIKE, "maturity_h": T,
                         "n_space_nodes": FIXED_NODES,
                         "n_time_steps": info["n_time_steps"],
                         "steps_per_hour": info["n_time_steps"] / T,
                         "V_call": v,
                         "residual_sd_T": info["residual_sd_T"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "time_convergence.csv", index=False)

    md = ["# FW12 §2 -- Time-step convergence\n",
          f"Spatial grid fixed at {FIXED_NODES} nodes; climatology z path; "
          "boundary and other settings as production.  Sweep multiplies "
          f"the base steps/hour by {K_STEP_MULTS}, so k=2 reproduces the "
          "shipped default and k=16 gives an 8x refinement of the "
          "time grid.\n",
          df.round(6).to_markdown(index=False)]
    md.append("\n\n## Observed time order per maturity\n")
    md.append("| T (h) | V_k1 | V_k2 | V_k4 | V_k8 | V_k16 | p_k2->k8 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for T in MATURITIES_H:
        sub = df[df["maturity_h"] == T].sort_values("n_time_steps")
        vs = sub["V_call"].tolist()
        v1, v2, v4, v8, v16 = vs
        p = observed_order(v2, v4, v8)
        md.append(f"| {T} | {v1:.5f} | {v2:.5f} | {v4:.5f} | {v8:.5f} | "
                  f"{v16:.5f} | {p:.3f} |")
    md.append("\n\nCrank-Nicolson time discretisation is theoretically O(k^2). "
              "If the observed order deviates materially from 2, the space "
              "error is likely masking the time error (§1 residual dominates).\n")
    (OUT_DIR / "time_convergence.md").write_text("\n".join(md),
                                                 encoding="utf-8")
    print("wrote", OUT_DIR / "time_convergence.csv")


if __name__ == "__main__":
    main()
