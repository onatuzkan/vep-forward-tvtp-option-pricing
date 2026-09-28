"""FW9d §4 -- Kappa sensitivity curve, paper-ready.

Sweep kappa across a 15-point grid spanning the range of estimates
found in FW9/9b/9c/9d plus the yaml value 0.078394.  For each kappa,
price the ATM K=3000 call at T in {24, 48, 72} h with the shipped
1201-node grid and the climatology z path.  All other parameters at
production values.

The output is the paper figure argued for in FW9d §4: rather than
claiming a point estimate for kappa (which cannot be defended in the
face of the FW9d isolation), the paper reports a MAP of price vs
kappa with every FW9-family estimate marked on it.
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
YAML_KAPPA = 0.078394

# Kappa grid: dense from 0.01 to 0.25, including the yaml value and
# every FW9-family estimate.  Log-spaced 12 points + yaml + 3 estimate
# markers = 16 unique.
BASE_GRID = list(np.geomspace(0.01, 0.25, 12))
MARK_KAPPAS = {
    "FW9b_R1_asinh_S0": 0.019599,
    "FW9b_R1_asinh_S5": 0.016729,
    "yaml": YAML_KAPPA,
    "FW9d_R3_S5": 0.016690,
    "FW9c_R2_TRY_S0": 0.211560,
    "FW9c_R2_TRY_S5": 0.224634,
}
GRID = sorted(set(BASE_GRID + list(MARK_KAPPAS.values())))
MATURITIES = (24, 48, 72)


def _price_atm(kappa: float, maturities=MATURITIES) -> dict:
    yaml_p = load_frozen_parameters(YAML_PATH)
    quotes = load_quotes(REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv")
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    spec = ResidualSpec(kappa_per_hour=kappa,
                        sigma_y=np.array([yaml_p.sigma_y[0],
                                          yaml_p.sigma_y[1]]),
                        scale_P=282.48, regime_means=np.zeros(2),
                        mode="additive")
    model = ForwardCenteredModel(
        curve=curve, spec=spec,
        tvtp=TVTPCoefficients(yaml_p.alpha01, yaml_p.gamma01,
                              yaml_p.alpha10, yaml_p.gamma10),
        pi_filtered=yaml_p.pi_filtered,
        valuation_utc=yaml_p.valuation_utc,
        spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    out = {}
    for T in maturities:
        contract = EuropeanOption("call", 3000.0, model.valuation_utc,
                                  model.valuation_utc + pd.Timedelta(hours=int(T)),
                                  r_annual=0.40)
        gs = ResidualGridSettings(n_space_nodes=1201)
        z_fn = climatology_z_lagged_fn(model, contract, gs)
        r = price_forward_centered(model, contract, grid_settings=gs,
                                   z_lagged_fn=z_fn)
        out[T] = float(r.value)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Sweep kappa on {len(GRID)} points: {[round(x, 5) for x in GRID]}")
    rows = []
    for k in GRID:
        px = _price_atm(k)
        # markers -- one row per kappa can carry multiple marker names
        marker_hits = [m for m, kv in MARK_KAPPAS.items() if abs(kv - k) < 1e-8]
        marker = "|".join(marker_hits) if marker_hits else ""
        rows.append({
            "kappa_per_hour": k,
            "half_life_hours": math.log(2.0) / k,
            "call_T24": px[24], "call_T48": px[48], "call_T72": px[72],
            "marker": marker,
        })
        print(f"  kappa={k:.5f}  HL={math.log(2.0)/k:5.2f}h  "
              f"call_T72={px[72]:7.3f}  {marker}")
    df = pd.DataFrame(rows).sort_values("kappa_per_hour").reset_index(drop=True)
    df.to_csv(OUT / "kappa_sensitivity.csv", index=False)

    # figure-ready CSV (columns matched to paper/make_figures.py style)
    fig_df = df.copy()
    fig_df["kappa_per_hour"] = fig_df["kappa_per_hour"].round(6)
    fig_df["log10_kappa"] = np.log10(fig_df["kappa_per_hour"])
    fig_df.to_csv(OUT / "figure_F14_kappa_sensitivity.csv", index=False)

    md = ["# FW9d §4 -- ATM call value vs kappa (paper figure)\n",
          "Sweep of the mean-reversion rate kappa; all other parameters"
          " at production values; 1201 spatial nodes; climatology z"
          " covariate path.  ATM K=3000 call priced at T = 24, 48, 72 h"
          " at each kappa.\n",
          "**Markers** on the curve show the estimates from FW9b (raw"
          " asinh, R1), FW9c (TRY monthly-anchor, R2), FW9d (asinh"
          " hour-of-week anchor, R3), and the shipped yaml value"
          " 0.078394.  None of the FW9-family estimates land on the"
          " yaml value; yaml sits **between** the two clusters"
          " (R1/R3 ~ 0.017/h in asinh space, R2 ~ 0.21/h in TRY-space).\n",
          df.round({"kappa_per_hour": 5, "half_life_hours": 3,
                    "call_T24": 3, "call_T48": 3, "call_T72": 3})
              .to_markdown(index=False)]

    md.append("\n\n## Reading the curve\n")
    md.append("* Between 0.01 and 0.25 /h the ATM 72 h call sweeps"
              " roughly {:.1f} to {:.1f} TRY/MWh -- a ~{:.0f} % dynamic"
              " range.".format(df["call_T72"].max(), df["call_T72"].min(),
                                100 * (df["call_T72"].max() - df["call_T72"].min()) / df["call_T72"].min()))
    md.append("* The three FW9 estimation branches produce point"
              " estimates at ~0.017 (asinh space), ~0.21 (TRY space),"
              " and yaml 0.078 (deseasonalised in the M9 handoff).")
    md.append("* The paper's honest conclusion: kappa is not sharply"
              " identified from this dataset without knowing the"
              " M9 deseasonalisation pipeline; the option price at any"
              " strike / maturity is a KNOWN FUNCTION of kappa (this"
              " table), and the paper reports the family of prices"
              " rather than a single point.")
    md.append("\n\n## Files\n\n"
              "* `kappa_sensitivity.csv` -- 3-maturity price grid over"
              " kappa.\n"
              "* `figure_F14_kappa_sensitivity.csv` -- same, with"
              " log10(kappa) column added for the paper figure"
              " (DejaVu Serif 9.5 pt, W=5.5 in, no top/right spines).\n"
              "* Markers to overlay: FW9b_R1_asinh_S0/S5 (asinh cluster),"
              " yaml (0.078394), FW9c_R2_TRY_S0/S5 (TRY cluster),"
              " FW9d_R3_S5 (asinh how-climatology cluster).")
    (OUT / "kappa_sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print("\nwrote kappa_sensitivity.csv/md and figure_F14_kappa_sensitivity.csv")


if __name__ == "__main__":
    main()
