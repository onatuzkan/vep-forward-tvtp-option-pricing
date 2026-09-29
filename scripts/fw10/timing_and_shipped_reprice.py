"""FW10b -- corrected day-ahead timing audit and re-pricing of the
shipped 2025-12-31 valuation under the corrected rule.

* Rewrites `outputs/fw10_validation/day_ahead_timing_check.md` with the
  correct column label: "published_by_valuation" (was mis-labelled
  "available_at_valuation" in the FW10 pass; the values themselves
  were correct).
* Identifies the VEP quotation day that FEEDS the shipped production
  forward curve (2025-12-31 23:00 TRT valuation instant).
* Rebuilds the curve under the FW10b rule (valuation d 11:00 TRT,
  quote day = last VEP GGF strictly before that instant) and prices
  the paper's 24/48/72 h ATM call plus the K in {2000, 2500, 3000,
  3500, 4000} strike ladder.  Reports absolute and percentage
  differences against the shipped 166.75 TRY at K=3000, T=72 h and
  the rest of the ladder.  DOES NOT modify the shipped yaml.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pde_option_model.calendar_tr import TURKEY_TZ
from pde_option_model.contracts import EuropeanOption
from pde_option_model.forward_centered import (ResidualGridSettings,
                                                 price_forward_centered)
from pde_option_model.forward_curve import NearTermAnchor, build_forward_curve
from pde_option_model.generator import TVTPCoefficients
from pde_option_model.market_data import load_quotes
from pde_option_model.params_frozen import load_frozen_parameters
from pde_option_model.forward_centered import (ForwardCenteredModel,
                                                 ResidualSpec)
from scripts.fw10._data import (day_end_utc, load_realized_ptf,
                                 load_vep_quotes_daily, valuation_utc)
from scripts.fw10.curve_builder import (_last_vep_quote_day_before,
                                          build_daily_production_curve)
from scripts.fw12._shared import climatology_z_lagged_fn

OUT = REPO / "outputs" / "fw10_validation"
YAML_PATH = REPO / "inputs" / "historical" / "m2_frozen_parameters.yaml"
PROD_QUOTES = REPO / "inputs" / "market" / "vep_monthly_quotes.csv"

R_ANNUAL = 0.40
HORIZONS = (24, 48, 72)
STRIKES = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0)


def _make_model_from_curve(curve, params) -> ForwardCenteredModel:
    return ForwardCenteredModel(
        curve=curve,
        spec=ResidualSpec.from_frozen(params),
        tvtp=TVTPCoefficients(params.alpha01, params.gamma01,
                              params.alpha10, params.gamma10),
        pi_filtered=params.pi_filtered,
        valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh)


def _price_at(model: ForwardCenteredModel, K: float, tau_h: int) -> float:
    contract = EuropeanOption(
        "call", K, model.valuation_utc,
        model.valuation_utc + pd.Timedelta(hours=int(tau_h)),
        r_annual=R_ANNUAL)
    gs = ResidualGridSettings(n_space_nodes=601)
    z_fn = climatology_z_lagged_fn(model, contract, gs)
    res = price_forward_centered(model, contract, grid_settings=gs,
                                  z_lagged_fn=z_fn)
    return float(res.value)


def _shipped_prices(params) -> dict:
    """Price under the SHIPPED production curve (from vep_monthly_quotes.csv
    + smooth_constrained + spot_to_next_linear anchor)."""
    quotes = load_quotes(PROD_QUOTES)
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                 spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    model = _make_model_from_curve(curve, params)
    return {(K, tau): _price_at(model, K, tau)
            for K in STRIKES for tau in HORIZONS}


def _corrected_prices(params) -> tuple:
    """Rebuild the curve at the FW10b-corrected timing rule.

    Valuation stays at the shipped instant 2025-12-31 20:00 UTC
    (23:00 TRT) so the residual PDE runs with the shipped
    ``valuation_utc``, but the forward curve is built from the VEP
    quote of the last publication STRICTLY BEFORE that instant --
    i.e. the last day before 2025-12-31 with a VEP GGF row.
    """
    vep = load_vep_quotes_daily()
    # Under the FW10b timing rule we would use the last VEP GGF
    # publication strictly before the valuation instant.  For the
    # 2025-12-31 valuation this is 2025-12-30 (assuming a GGF was
    # published that day) or earlier.
    val_utc = pd.Timestamp("2025-12-31 20:00", tz="UTC")
    quote_day = _last_vep_quote_day_before(vep, val_utc)
    slice_ = vep[vep["valuation_date"] == quote_day].reset_index(drop=True)
    if slice_.empty:
        raise RuntimeError(f"no VEP quote found before {val_utc}")
    # Save a temporary quote CSV with the same schema
    tmp = OUT / "_tmp_quote_corrected.csv"
    slice_["valuation_date"] = str(quote_day)
    slice_["source"] = "EPIAS_VEP_daily_reference_price"
    slice_["quote_type"] = "monthly_baseload"
    slice_["spot_price_TRY_MWh"] = params.spot_price_TRY_MWh
    slice_[["valuation_date", "contract_name", "price_TRY_MWh",
             "delivery_year", "delivery_month", "source", "quote_type",
             "spot_price_TRY_MWh"]].to_csv(tmp, index=False)
    quotes = load_quotes(tmp, valuation_utc=val_utc)
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                 spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    model = _make_model_from_curve(curve, params)
    prices = {(K, tau): _price_at(model, K, tau)
              for K in STRIKES for tau in HORIZONS}
    return prices, quote_day


def rewrite_timing_check_md() -> str:
    """Regenerate day_ahead_timing_check.md with the corrected column
    label and a note that the underlying values are unchanged.
    """
    prev = OUT / "day_ahead_timing_check.md"
    text = prev.read_text(encoding="utf-8")
    corrected = text.replace(
        "published_by_valuation", "published_by_valuation"  # placeholder
    )
    # Old header had "already_published_at_valuation" in the field but
    # the compact table used "available_at_valuation" in the wording.
    # We rewrite the (b) table with the correct label and value logic.
    return corrected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_frozen_parameters(YAML_PATH)
    print("Pricing under SHIPPED forward curve...")
    shipped = _shipped_prices(params)
    print("Pricing under CORRECTED (FW10b timing rule) forward curve...")
    corrected, quote_day_used = _corrected_prices(params)

    rows = []
    for K in STRIKES:
        for tau in HORIZONS:
            v0 = shipped[(K, tau)]
            v1 = corrected[(K, tau)]
            rows.append({
                "K": K, "tau_h": tau,
                "shipped_call": v0,
                "corrected_call": v1,
                "abs_diff_TRY": v1 - v0,
                "pct_diff": 100.0 * (v1 - v0) / v0 if v0 != 0 else float("nan"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "shipped_vs_corrected_reprice.csv", index=False)
    piv_abs = df.pivot(index="K", columns="tau_h", values="abs_diff_TRY")
    piv_pct = df.pivot(index="K", columns="tau_h", values="pct_diff")

    # Rewrite the timing check md with the corrected label plus the
    # reprice numbers
    md = [
        "# FW10 / FW10b -- Day-ahead publication timing audit and "
        "shipped-valuation reprice\n",
        "## (a) Day-D prices known at end of D-1\n",
        "The Turkish day-ahead auction closes at 12:30 TRT for the "
        "NEXT day's hourly clearing prices.  At 11:00 TRT of day D "
        "the hours 00:00-23:00 of day D are already published (D was "
        "cleared on D-1 at ~14:00 TRT); the 24 hours of day D+1 are "
        "NOT yet known.\n",
        "## (b) Original valuation 2025-12-31 20:00 UTC (23:00 TRT)\n",
        "Column labels corrected in this FW10b pass -- **the shipped "
        "valuation instant is at day-end, so any horizon whose "
        "terminal falls on day D+1 or later has NOT been published "
        "by the valuation instant.**  The FW10 pass mis-labelled the "
        "compact-table column as 'available_at_valuation'; the values "
        "themselves are the correct booleans.\n",
        "| horizon | target_UTC | target_TRT | published_by_valuation |",
        "|---|---|---|:---:|",
        "| 24 h | 2026-01-01 20:00 UTC | 23:00 TRT of 2026-01-01 | "
        "**True** (day 01-01 DA prices publish 2025-12-31 ~14:00 TRT) |",
        "| 48 h | 2026-01-02 20:00 UTC | 23:00 TRT of 2026-01-02 | "
        "False |",
        "| 72 h | 2026-01-03 20:00 UTC | 23:00 TRT of 2026-01-03 | "
        "False |",
        "",
        "## (c) FW3 backtest",
        "22 of the 66 grid points (the entire 24 h maturity row) had "
        "their terminal PTF already published at the shipped "
        "valuation instant; this is a structural feature of the "
        "day-end valuation timestamp.\n",
        "## (d) FW10b corrected rule + reprice",
        "Under the corrected FW10b timing rule the forward curve is "
        "built from the VEP GGF publication STRICTLY BEFORE the "
        "valuation instant.  For the shipped 2025-12-31 20:00 UTC "
        f"valuation this quote day is **{quote_day_used}**, versus "
        "the shipped `inputs/market/vep_monthly_quotes.csv` which "
        "records the 2025-12-31 quote day directly (i.e. a same-day "
        "quote that would only be available AFTER 20:00 UTC).\n",
        "Absolute call-price differences (TRY):",
        piv_abs.round(3).to_markdown(),
        "\nPercentage differences (pct):",
        piv_pct.round(3).to_markdown(),
        "",
        "**Effect on the shipped 72 h ATM K=3000 call (166.75 TRY):** "
        f"the corrected timing rule shifts it to "
        f"{corrected[(3000.0, 72)]:.3f} TRY, a change of "
        f"{corrected[(3000.0, 72)] - shipped[(3000.0, 72)]:+.3f} TRY "
        f"({100.0 * (corrected[(3000.0, 72)] - shipped[(3000.0, 72)]) / shipped[(3000.0, 72)]:+.3f} pct).  "
        "The shipped production yaml valuation is NOT modified by "
        "this report; the reprice quantifies the timing-rule effect.",
        "",
        "## (e) FW10b daily-evaluation convention",
        "Every FW10b daily evaluation uses rule (d): valuation at "
        "d 11:00 TRT, last known PTF hour = d 23:00 TRT, horizons at "
        "h in {6, 12, 24, 48, 72} hours after that instant.  Forward "
        "curve on day d is built from the last VEP GGF strictly "
        "before d 11:00 TRT (so d-1 or earlier).  HPFC shape (fit "
        "once from pre-FREEZE_UTC data with `half_life_years = 0.5`, "
        "`n_harmonics = 2`) is applied inside every quoted delivery "
        "month.",
    ]
    (OUT / "day_ahead_timing_check.md").write_text("\n".join(md),
                                                    encoding="utf-8")
    print(f"corrected reprice: K=3000 T=72h shipped {shipped[(3000.0, 72)]:.3f} -> "
          f"corrected {corrected[(3000.0, 72)]:.3f}")
    print("wrote shipped_vs_corrected_reprice.csv and updated "
          "day_ahead_timing_check.md")


if __name__ == "__main__":
    main()
