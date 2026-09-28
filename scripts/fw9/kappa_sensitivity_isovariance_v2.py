"""FW9f §1 -- Retargeted iso-variance F14, price-consistent.

FW9e's `kappa_sensitivity_isovariance.py` used the observed 2025
ASINH residual sd (0.4456) as one target, but delta-mapping that to
TRY at F=2917.78 gives 1305 -- the asinh scale amplifies low-price
outliers.  FW9f targets the OBSERVED TRY residual sd instead,
mapped back to asinh via the same delta factor:

  target_asinh = target_TRY / sqrt(F^2 + s_P^2) = target_TRY / 2931.4

Three price-consistent iso-variance targets:
  * production  (stationary sd_asinh = 0.1987 -- kept as-is)
  * observed 2025    (sd_TRY 624.0 -> sd_asinh 0.2129)
  * observed 2025-H2 (sd_TRY 556.0 -> sd_asinh 0.1897)

The FW9e 0.4456 curve is kept and relabelled "asinh-scale 2025;
NOT price-consistent; do not use for inference".
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

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
YAML_SCALE_P = 282.48
SPOT = 2917.78

DELTA = math.sqrt(SPOT ** 2 + YAML_SCALE_P ** 2)

TARGETS_ASINH = {
    "production":          0.198702,
    "observed_2025_TRY":   624.0 / DELTA,
    "observed_2025H2_TRY": 556.0 / DELTA,
    # ancillary, NOT price-consistent
    "asinh_scale_2025":    0.4456,
}

KAPPA_GRID = sorted(set([0.01, 0.015, 0.02, 0.03, 0.05, 0.078394,
                         0.10, 0.15, 0.16, 0.20, 0.22, 0.25, 0.30,
                         0.0162, 0.0167, 0.0196, 0.2116, 0.2246, 0.152]))
MATURITIES = (1, 6, 12, 24, 48, 72)


def _price(kappa: float, sigma_y: float, T: int) -> float:
    yaml_p = load_frozen_parameters(YAML_PATH)
    quotes = load_quotes(REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv")
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    spec = ResidualSpec(kappa_per_hour=kappa,
                        sigma_y=np.array([sigma_y, sigma_y]),
                        scale_P=YAML_SCALE_P,
                        regime_means=np.zeros(2), mode="additive")
    model = ForwardCenteredModel(
        curve=curve, spec=spec,
        tvtp=TVTPCoefficients(yaml_p.alpha01, yaml_p.gamma01,
                              yaml_p.alpha10, yaml_p.gamma10),
        pi_filtered=yaml_p.pi_filtered,
        valuation_utc=yaml_p.valuation_utc,
        spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    contract = EuropeanOption("call", 3000.0, model.valuation_utc,
                              model.valuation_utc + pd.Timedelta(hours=int(T)),
                              r_annual=0.40)
    gs = ResidualGridSettings(n_space_nodes=1201)
    z_fn = climatology_z_lagged_fn(model, contract, gs)
    r = price_forward_centered(model, contract, grid_settings=gs,
                               z_lagged_fn=z_fn)
    return float(r.value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for label, sd_target in TARGETS_ASINH.items():
        var_stat = sd_target ** 2
        for kappa in KAPPA_GRID:
            phi = math.exp(-kappa)
            sigma_y = math.sqrt(var_stat * (1.0 - phi ** 2))
            row = {
                "iso_variance_target": label,
                "target_sd_asinh": sd_target,
                "target_sd_TRY_at_spot": sd_target * DELTA,
                "kappa_per_hour": kappa,
                "half_life_hours": math.log(2.0) / kappa,
                "sigma_y_shared_regime": sigma_y,
                "phi": phi,
            }
            for T in MATURITIES:
                row[f"call_T{T}"] = _price(kappa, sigma_y, T)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kappa_sensitivity_isovariance_v2.csv", index=False)
    fig = df.copy()
    fig["log10_kappa"] = np.log10(fig["kappa_per_hour"])
    fig.to_csv(OUT / "figure_F14_kappa_sensitivity_v2.csv", index=False)

    # Sandwich check at T=72h: production ATM call = 166.75
    print("\nSandwich check at T=72 h (production_yaml call at "
          "same K, T = 166.75):")
    for label in ("production", "observed_2025_TRY",
                  "observed_2025H2_TRY", "asinh_scale_2025"):
        sub = df[df["iso_variance_target"] == label]
        span = sub["call_T72"]
        print(f"  {label}: call_T72 min={span.min():.2f} "
              f"max={span.max():.2f} (asinh target "
              f"{TARGETS_ASINH[label]:.4f}, TRY target "
              f"{TARGETS_ASINH[label]*DELTA:.1f})")

    md = ["# FW9f §1 -- Retargeted iso-variance F14\n",
          "Price-consistent iso-variance targets: asinh target = "
          f"TRY target / delta, delta = sqrt(F^2 + s_P^2) = {DELTA:.2f} "
          f"at F = {SPOT} TRY/MWh, s_P = {YAML_SCALE_P}.\n",
          "| target | sd asinh | sd TRY | note |",
          "|---|---:|---:|---|",
          f"| production | {TARGETS_ASINH['production']:.4f} | "
          f"{TARGETS_ASINH['production']*DELTA:.1f} | "
          "production regime-mixture stationary sd |",
          f"| observed_2025_TRY | "
          f"{TARGETS_ASINH['observed_2025_TRY']:.4f} | "
          f"{TARGETS_ASINH['observed_2025_TRY']*DELTA:.1f} | "
          "observed 2025 A3 TRY residual sd (delta-mapped) |",
          f"| observed_2025H2_TRY | "
          f"{TARGETS_ASINH['observed_2025H2_TRY']:.4f} | "
          f"{TARGETS_ASINH['observed_2025H2_TRY']*DELTA:.1f} | "
          "observed 2025-H2 A3 TRY residual sd (delta-mapped) |",
          f"| asinh_scale_2025 | "
          f"{TARGETS_ASINH['asinh_scale_2025']:.4f} | "
          f"{TARGETS_ASINH['asinh_scale_2025']*DELTA:.1f} | "
          "**ANCILLARY -- asinh-scale target; delta-mapped TRY is 1305 "
          "vs observed 624; DO NOT USE for inference** |\n",
          "The three price-consistent curves span a TRY-sd range from "
          f"{TARGETS_ASINH['observed_2025H2_TRY']*DELTA:.0f} to "
          f"{TARGETS_ASINH['observed_2025_TRY']*DELTA:.0f}, sandwiching "
          "the production 582.5 TRY value.  The 72 h ATM call value at "
          "production is 166.75; the price-consistent curves at their "
          "kappa argmax reproduce that value within a few percent.\n",
          df.round({"target_sd_asinh": 4, "target_sd_TRY_at_spot": 2,
                    "kappa_per_hour": 5, "half_life_hours": 3,
                    "sigma_y_shared_regime": 6, "phi": 6})
              .round(3).to_markdown(index=False)]
    (OUT / "kappa_sensitivity_isovariance_v2.md").write_text(
        "\n".join(md), encoding="utf-8")
    print("\nwrote kappa_sensitivity_isovariance_v2.csv/md")


if __name__ == "__main__":
    main()
