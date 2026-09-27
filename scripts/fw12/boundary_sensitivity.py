"""FW12 §3 boundary-location sensitivity: does the far-field spec matter?"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fw12._shared import build_production_model, price_at  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "fw12_convergence"

STRIKE = 3000.0
MATURITIES_H = [24, 48, 72]
N_STDS = [4.0, 5.0, 6.0, 7.5, 9.0]     # 6.0 is the production default
FIXED_NODES = 2401


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_production_model()
    rows = []
    for T in MATURITIES_H:
        for n_std in N_STDS:
            v, info = price_at(model, STRIKE, T, "call", FIXED_NODES, None,
                               n_std=n_std)
            rows.append({
                "strike": STRIKE, "maturity_h": T, "n_std": n_std,
                "V_call": v,
                "grid_x_min": info["grid_x_min"],
                "grid_x_max": info["grid_x_max"],
                "grid_halfwidth_TRY_MWh":
                    0.5 * (info["grid_x_max"] - info["grid_x_min"])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "boundary_sensitivity.csv", index=False)

    md = ["# FW12 §3 -- Boundary-location sensitivity\n",
          f"Spatial nodes fixed at {FIXED_NODES}; time steps default; "
          "climatology z; only `n_std` (grid half-width in analytic "
          "residual standard deviations) varies.  Production default is "
          "6.0; the manuscript's Appendix C threshold is that shifting "
          "the boundary should not move the price by more than 0.1 % of "
          "the converged value.\n",
          df.round(6).to_markdown(index=False)]

    md.append("\n\n## % effect vs the n_std=6.0 production value\n")
    md.append("| T (h) | n_std | V | Δ vs V(n_std=6) | % vs V(n_std=6) |")
    md.append("|---:|---:|---:|---:|---:|")
    for T in MATURITIES_H:
        sub = df[df["maturity_h"] == T].sort_values("n_std")
        v_ref = float(sub[sub["n_std"] == 6.0]["V_call"].iloc[0])
        for _, r in sub.iterrows():
            d = r["V_call"] - v_ref
            md.append(f"| {T} | {r['n_std']:.1f} | {r['V_call']:.5f} | "
                      f"{d:+.5f} | {100 * d / max(abs(v_ref), 1e-9):+.4f} % |")
    md.append("\n\nIf the |% vs V(n_std=6)| column exceeds 0.1 % for any "
              "row, the far-field specification is affecting the reported "
              "prices materially -- flagged as a boundary-spec issue.\n")
    (OUT_DIR / "boundary_sensitivity.md").write_text("\n".join(md),
                                                     encoding="utf-8")
    print("wrote", OUT_DIR / "boundary_sensitivity.csv")


if __name__ == "__main__":
    main()
