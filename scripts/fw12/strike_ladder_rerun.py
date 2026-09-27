"""FW12 §5 strike ladder at the converged spatial grid.

Reprices the strike ladder K in {2000, 2500, 3000, 3500, 4000} at
T in {24, 48, 72} h under the SAME climatology z path and the
production settings otherwise, but with 2401 spatial nodes instead of
1201; put next to the values in
`outputs/market_calibration_final/strike_maturity_grid.csv` (the
production 1201-node run) and report the per-cell relative
difference.  The manuscript needs this table to decide whether the
2401 (or Richardson-extrapolated) value should replace the 1201
value in Appendix C.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fw12._shared import build_production_model, price_at   # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "fw12_convergence"

STRIKES = [2000.0, 2500.0, 3000.0, 3500.0, 4000.0]
MATURITIES_H = [24, 48, 72]
PRODUCTION_GRID_CSV = REPO_ROOT / "outputs" / "market_calibration_final" / "strike_maturity_grid.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prod = pd.read_csv(PRODUCTION_GRID_CSV)
    prod = prod[prod["maturity_h"].isin(MATURITIES_H)
                & prod["strike_TRY_MWh"].isin(STRIKES)]

    model = build_production_model()
    rows = []
    for K in STRIKES:
        for T in MATURITIES_H:
            v_call, info = price_at(model, K, T, "call", 2401, None)
            v_put, _ = price_at(model, K, T, "put", 2401, None)
            prod_row = prod[(prod["strike_TRY_MWh"] == K)
                            & (prod["maturity_h"] == T)]
            prod_call = float(prod_row["call_TRY_MWh"].iloc[0]) if len(prod_row) else float("nan")
            prod_put = float(prod_row["put_TRY_MWh"].iloc[0]) if len(prod_row) else float("nan")
            rows.append({
                "strike": K, "maturity_h": T,
                "call_2401": v_call, "call_prod_1201": prod_call,
                "call_diff_TRY_MWh": v_call - prod_call,
                "call_diff_pct": (100.0 * (v_call - prod_call) / prod_call
                                  if abs(prod_call) > 1e-9 else float("nan")),
                "put_2401": v_put, "put_prod_1201": prod_put,
                "put_diff_TRY_MWh": v_put - prod_put,
                "put_diff_pct": (100.0 * (v_put - prod_put) / prod_put
                                 if abs(prod_put) > 1e-9 else float("nan")),
                "residual_sd_T": info["residual_sd_T"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "strike_ladder_rerun.csv", index=False)

    md = ["# FW12 §5 -- Strike ladder at 2401 nodes vs production 1201\n",
          "Same climatology z, same n_std=6, same default time steps; ",
          "only spatial nodes 2401 vs 1201.  Production values come ",
          "straight from `outputs/market_calibration_final/",
          "strike_maturity_grid.csv`.  Watch the K=4000 (deep-OTM) ",
          "row: FW2's largest Q1 effect was at that strike, and any ",
          "numerical bias there would poison the interpretation of the ",
          "risk-premium envelope.\n",
          df.round({"call_2401": 4, "call_prod_1201": 4,
                    "call_diff_TRY_MWh": 4, "call_diff_pct": 3,
                    "put_2401": 4, "put_prod_1201": 4,
                    "put_diff_TRY_MWh": 4, "put_diff_pct": 3,
                    "residual_sd_T": 2}).to_markdown(index=False)]
    (OUT_DIR / "strike_ladder_rerun.md").write_text("\n".join(md),
                                                    encoding="utf-8")
    print("wrote", OUT_DIR / "strike_ladder_rerun.csv")


if __name__ == "__main__":
    main()
