"""FW12 §6 -- FW2 (a, eta) manuscript rows re-priced at the converged grid.

Restricted to the rows that will appear in the paper:
  * Q1: a_stress in {0, 10, 25, 50} TRY/MWh/h (a_normal = 0)
  * Q2: eta_ij in {(0, 0), (+/-0.5, -/+0.5), (+/-0.75, -/+0.75)}
ATM K=3000 at T = 24 / 48 / 72 h.  Percent effects are computed on
BOTH bases -- the FW2 179.65 baseline and the FW12-converged
baseline -- so the deltas the manuscript reports are the ones from
a converged base, and the reader can see the size of the
scenario/discretisation correction as a footnote.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fw12._shared import build_production_model, price_at    # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "fw12_convergence"

STRIKE = 3000.0
MATURITIES_H = [24, 48, 72]
A_VALUES = [0.0, 10.0, 25.0, 50.0]
ETA_VALUES: List[Tuple[float, float]] = [
    (0.0, 0.0),
    (+0.5, -0.5), (-0.5, +0.5),
    (+0.75, -0.75), (-0.75, +0.75),
]
N_SPACE = 2401
FW2_BASELINE_179_65 = 179.6485          # canonical FW2 601-node z=0 baseline


def _price_pair(model, K: int, T: int, a: float,
                eta: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if eta is not None and a != 0.0:
        family = f"joint(a={a}, eta={eta})"
    elif eta is not None:
        family = f"Q2 eta={eta}"
    else:
        family = f"Q1 a_stress={a}"
    _ = family
    # apply a_stress via drift_shift_per_hour override; the shipped
    # price_at helper does not carry that knob, so we build the shifted
    # model here.
    from pde_option_model.contracts import EuropeanOption
    from pde_option_model.forward_centered import (
        ForwardCenteredModel, ResidualGridSettings, ResidualSpec,
        price_forward_centered)
    import numpy as np
    spec = model.spec
    shifted = ResidualSpec(
        kappa_per_hour=spec.kappa_per_hour, sigma_y=spec.sigma_y,
        scale_P=spec.scale_P, regime_means=spec.regime_means,
        mode=spec.mode, x0_mode=spec.x0_mode,
        sigma_multipliers=spec.sigma_multipliers,
        drift_shift_per_hour=np.array([0.0, a]),
    )
    m2 = ForwardCenteredModel(
        curve=model.curve, spec=shifted, tvtp=model.tvtp,
        pi_filtered=model.pi_filtered,
        valuation_utc=model.valuation_utc,
        spot_price_TRY_MWh=model.spot_price_TRY_MWh)
    contract = EuropeanOption(
        "call", float(K), model.valuation_utc,
        model.valuation_utc + pd.Timedelta(hours=int(T)), r_annual=0.40)
    gs = ResidualGridSettings(n_space_nodes=N_SPACE)
    from scripts.fw12._shared import climatology_z_lagged_fn
    z_fn = climatology_z_lagged_fn(m2, contract, gs)
    r = price_forward_centered(m2, contract, grid_settings=gs,
                               z_lagged_fn=z_fn, eta_ij=eta)
    return float(r.value), float(r.residual_std_at_expiry)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_production_model()
    # baseline at converged grid, no premium
    base = {}
    for T in MATURITIES_H:
        v, _ = _price_pair(model, STRIKE, T, 0.0, None)
        base[T] = v
        print(f"baseline @ 2401 z_clim, K={STRIKE}, T={T}h: {v:.4f}")

    rows = []
    for T in MATURITIES_H:
        for a in A_VALUES:
            for eta in ETA_VALUES:
                if a == 0.0 and eta == (0.0, 0.0):
                    continue      # baseline handled separately
                family = ("Q1_only" if eta == (0.0, 0.0) else
                          "Q2_only" if a == 0.0 else "joint")
                v, sd_T = _price_pair(model, STRIKE, T, a,
                                      None if eta == (0.0, 0.0) else eta)
                d_conv = v - base[T]
                rows.append({
                    "strike": STRIKE, "maturity_h": T, "family": family,
                    "a_stress": a, "eta_01": eta[0], "eta_10": eta[1],
                    "V_call_2401_clim": v,
                    "baseline_2401_clim": base[T],
                    "delta_conv_TRY_MWh": d_conv,
                    "delta_conv_pct_of_2401_base":
                        100.0 * d_conv / max(abs(base[T]), 1e-9),
                    "delta_from_FW2_179p65_pct":
                        100.0 * (v - FW2_BASELINE_179_65) / FW2_BASELINE_179_65,
                    "residual_sd_T": sd_T,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "fw2_at_converged.csv", index=False)

    md = ["# FW12 §6 -- FW2 (a, eta) sensitivity at the converged grid\n",
          "Same rows the manuscript will cite, re-priced at 2401 spatial ",
          "nodes with the production climatology z path.  Percent columns:",
          "* `delta_conv_pct_of_2401_base` is the CORRECT number for the ",
          "manuscript (base = 2401-node, climatology-z, no premium)",
          "* `delta_from_FW2_179p65_pct` is provided so the reader can see ",
          "how FW2's raw table would appear if computed from its 179.65 baseline. ",
          "Do not cite this column in the paper; it exists only to reconcile ",
          "against FW2 §4.\n",
          "## Baselines\n"]
    for T in MATURITIES_H:
        md.append(f"* K=3000, T={T}h, baseline @ 2401 climatology = "
                  f"**{base[T]:.4f} TRY/MWh**")
    md.append("\n")
    md.append(df.round({"V_call_2401_clim": 4, "baseline_2401_clim": 4,
                        "delta_conv_TRY_MWh": 4,
                        "delta_conv_pct_of_2401_base": 3,
                        "delta_from_FW2_179p65_pct": 3,
                        "residual_sd_T": 2}).to_markdown(index=False))
    (OUT_DIR / "fw2_at_converged.md").write_text("\n".join(md),
                                                 encoding="utf-8")
    print("wrote", OUT_DIR / "fw2_at_converged.csv")


if __name__ == "__main__":
    main()
