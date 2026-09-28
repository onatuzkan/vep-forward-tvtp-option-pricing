"""FW9e §5 -- Iso-variance kappa sensitivity F14.

The old constant-sigma sensitivity (`kappa_sensitivity.csv`) held
`sigma_y` at production and swept `kappa`, so it swept the stationary
variance too.  That is why the 72 h ATM call moved by 6.7x across the
kappa grid; the family is not comparable pointwise to any real
economic uncertainty.

The iso-variance construction fixes the stationary variance at the
observed 2025 A3-residual variance (TRY scale, then converted to
asinh via the same delta-method the model uses) AND separately at
the production-implied stationary variance.  For each kappa the
sigma_y is derived from the iso-variance condition
    sigma_y^2 = Var_stat * (1 - phi^2)  where  phi = exp(-kappa)
(regime-mixture: the same sigma_y is applied to both regimes to
preserve the mixture-variance identity; the yaml sigma ratio is not
maintained -- this sweep is a diagnostic, not a re-fit).

Kappa grid: [0.01, 0.30], including the yaml value and every
FW9-family estimate.  Prices ATM K=3000 at T in {1, 6, 12, 24, 48,
72} h with 1201 nodes and the climatology z path.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

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

# Iso-variance anchors (asinh scale)
PRODUCTION_STAT_SD_ASINH = 0.198702    # from stationary_variance_check.md
OBSERVED_2025_SD_ASINH   = 0.445599    # from yearly_kappa_A3_asinh 2025 resid_sd

KAPPA_GRID = sorted(set([0.01, 0.015, 0.02, 0.03, 0.05, 0.078394,
                         0.10, 0.15, 0.16, 0.20, 0.22, 0.25, 0.30,
                         0.0162, 0.0167, 0.0196, 0.2116, 0.2246]))
MATURITIES = (1, 6, 12, 24, 48, 72)


def _price(kappa: float, sigma_y: float, T: int) -> float:
    yaml_p = load_frozen_parameters(YAML_PATH)
    quotes = load_quotes(REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv")
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    # For iso-variance we use a SHARED sigma_y across regimes -- the
    # mixture-averaged sigma_y matching the target stationary variance.
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
    for target_label, sd_target in (("production", PRODUCTION_STAT_SD_ASINH),
                                    ("observed_2025", OBSERVED_2025_SD_ASINH)):
        var_stat = sd_target ** 2
        for kappa in KAPPA_GRID:
            phi = math.exp(-kappa)
            sigma_y = math.sqrt(var_stat * (1.0 - phi ** 2))
            row = {
                "iso_variance_target": target_label,
                "kappa_per_hour": kappa,
                "half_life_hours": math.log(2.0) / kappa,
                "sigma_y_shared_regime": sigma_y,
                "phi": phi,
                "stationary_sd_asinh_target": sd_target,
            }
            for T in MATURITIES:
                row[f"call_T{T}"] = _price(kappa, sigma_y, T)
            rows.append(row)
            print(f"[{target_label}] kappa={kappa:.5f} sigma_y={sigma_y:.6f}  "
                  f"call_T24={row['call_T24']:.3f}  "
                  f"call_T72={row['call_T72']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kappa_sensitivity_isovariance.csv", index=False)

    # figure-ready CSV
    fig = df.copy()
    fig["log10_kappa"] = np.log10(fig["kappa_per_hour"])
    fig.to_csv(OUT / "figure_F14_kappa_sensitivity_isovariance.csv",
               index=False)

    # summary of 72h and 24h ranges within model-faithful family
    for target_label in ("production", "observed_2025"):
        sub = df[df["iso_variance_target"] == target_label]
        r72 = sub["call_T72"]; r24 = sub["call_T24"]
        print(f"\n{target_label} target (sd_stat_asinh = "
              f"{sub['stationary_sd_asinh_target'].iloc[0]:.4f}):")
        print(f"  call_T72 range: [{r72.min():.3f}, {r72.max():.3f}]"
              f"   dynamic {100*(r72.max()-r72.min())/r72.min():.2f}%")
        print(f"  call_T24 range: [{r24.min():.3f}, {r24.max():.3f}]"
              f"   dynamic {100*(r24.max()-r24.min())/r24.min():.2f}%")

    md = ["# FW9e §5 -- Iso-variance kappa sensitivity (F14)\n",
          "For each kappa on a 17-point grid over [0.01, 0.30] /h, "
          "solve `sigma_y = sqrt(Var_stat * (1 - phi^2))` with "
          "`phi = exp(-kappa)` under two iso-variance targets:\n",
          f"* **production**  -- stationary sd_asinh = "
          f"{PRODUCTION_STAT_SD_ASINH:.4f} (production-implied)",
          f"* **observed_2025** -- stationary sd_asinh = "
          f"{OBSERVED_2025_SD_ASINH:.4f} (2025 A3 residual sd)",
          "\nThen price ATM K=3000 at T in {1, 6, 12, 24, 48, 72} h.",
          "\nAll (kappa, sigma_y) pairs share the SAME stationary "
          "variance in the asinh scale, so the sweep isolates the "
          "shape effect of kappa (short-horizon dispersion) from the "
          "level effect (long-run scale).  The old constant-sigma "
          "sweep -- `kappa_sensitivity.csv` -- swept both together; "
          "kept as an ancillary output labelled 'sigma held fixed; "
          "partial derivative, not a data-consistent sensitivity'.\n",
          df.round({"kappa_per_hour": 5,
                    "half_life_hours": 3,
                    "sigma_y_shared_regime": 6,
                    "phi": 6}).round(3).to_markdown(index=False)]
    (OUT / "kappa_sensitivity_isovariance.md").write_text(
        "\n".join(md), encoding="utf-8")
    print("\nwrote kappa_sensitivity_isovariance.csv/md")


if __name__ == "__main__":
    main()
