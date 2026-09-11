"""Review analysis for market_calibration_review.

Produces:
  * option_value_comparison.csv  — call K=3000 at 24/72/168h for each anchor mode
  * hourly_curve_shape_january.csv — January-window curve for each mode
  * hourly_jumps_<mode>.csv — every hour-to-hour |dF|>5 TRY/MWh, flagged as month-boundary or not
  * near_term_anchor_sensitivity_fixed.csv — sensitivity table using
    spot_to_next_linear (level swept via the assumed spot); shape preserved
  * near_term_anchor_sensitivity_comparison.csv — old (explicit_level) vs fixed side-by-side
  * summary_metrics.json — per-mode fit stats + option values + jump counts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pde_option_model.contracts import EuropeanOption
from pde_option_model.forward_centered import (ForwardCenteredModel,
                                                ResidualGridSettings,
                                                ResidualSpec,
                                                price_forward_centered)
from pde_option_model.forward_curve import (NearTermAnchor,
                                             build_forward_curve,
                                             load_forward_curve_csv)
from pde_option_model.generator import TVTPCoefficients
from pde_option_model.market_calibration import REPORTING_HORIZONS_HOURS
from pde_option_model.market_data import load_quotes
from pde_option_model.params_frozen import load_frozen_parameters

REVIEW = Path("outputs/market_calibration_review")
MODES = ("spot_flat", "spot_to_next_linear", "flat_next_month")
STRIKE = 3000.0
MATURITIES_HOURS = (24, 72, 168)


def _load_model(params, quotes, curve):
    return ForwardCenteredModel(
        curve=curve,
        spec=ResidualSpec.from_frozen(params),
        tvtp=TVTPCoefficients(params.alpha01, params.gamma01,
                              params.alpha10, params.gamma10),
        pi_filtered=params.pi_filtered,
        valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh,
        covariate_lag_hours=params.covariate_lag_hours,
        allow_spot_mismatch=True,
    )


def _price_calls(model, params, grid_settings):
    rows = []
    for h in MATURITIES_HOURS:
        opt = EuropeanOption(
            option_type="call", strike=STRIKE,
            valuation_utc=params.valuation_utc,
            maturity_utc=params.valuation_utc + pd.Timedelta(hours=int(h)),
            r_annual=0.40,
        )
        pr = price_forward_centered(model, opt, grid_settings)
        rows.append({
            "maturity_h": h,
            "strike_TRY_MWh": STRIKE,
            "forward_at_expiry_TRY_MWh": float(pr.forward_at_expiry),
            "expected_spot_at_expiry_TRY_MWh": float(pr.expected_spot_at_expiry),
            "residual_std_at_expiry_TRY_MWh": float(pr.residual_std_at_expiry),
            "call_value_TRY_MWh": float(pr.value),
        })
    return rows


def _hourly_jumps(df, threshold=5.0):
    times = pd.to_datetime(df["time_utc"], utc=True)
    values = df["hourly_forward_TRY_MWh"].to_numpy(dtype=float)
    month_labels = df["delivery_month"].astype(str).to_numpy()
    jumps = []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        if abs(delta) > threshold:
            is_boundary = month_labels[i] != month_labels[i - 1]
            jumps.append({
                "time_utc_from": times.iloc[i - 1].isoformat(),
                "time_utc_to": times.iloc[i].isoformat(),
                "month_from": month_labels[i - 1],
                "month_to": month_labels[i],
                "F_prev_TRY_MWh": float(values[i - 1]),
                "F_next_TRY_MWh": float(values[i]),
                "delta_TRY_MWh": float(delta),
                "is_month_boundary": bool(is_boundary),
            })
    return jumps


def main() -> None:
    quotes = load_quotes("inputs/market/vep_monthly_quotes.csv")
    params = load_frozen_parameters("inputs/historical/m2_frozen_parameters.yaml")
    gs = ResidualGridSettings(n_space_nodes=1201, n_std=6.0)

    # 1 & 2 --- per-mode: option prices, curve shape, hourly jumps
    per_mode = {}
    all_option_rows = []
    jan_frames = []
    for mode in MODES:
        d = REVIEW / mode
        curve_vals = load_forward_curve_csv(d / "hourly_forward_curve.csv")
        anchor = NearTermAnchor(mode=mode)
        curve = build_forward_curve(quotes, mode="smooth_constrained",
                                    anchor=anchor,
                                    spot_price_TRY_MWh=params.spot_price_TRY_MWh)
        # numerical parity check
        pmax = float(np.max(np.abs(curve.values.to_numpy() - curve_vals.to_numpy())))
        model = _load_model(params, quotes, curve)
        rows = _price_calls(model, params, gs)
        for r in rows:
            r["anchor_mode"] = mode
        all_option_rows.extend(rows)

        # January window (first 31*24 = 744 hours after valuation)
        january_frame = pd.read_csv(d / "hourly_forward_curve.csv").head(744).copy()
        january_frame["anchor_mode"] = mode
        jan_frames.append(january_frame[["time_utc", "delivery_month",
                                          "hourly_forward_TRY_MWh",
                                          "anchor_mode"]])

        # hourly jumps
        full = pd.read_csv(d / "hourly_forward_curve.csv")
        jumps = _hourly_jumps(full, threshold=5.0)
        pd.DataFrame(jumps).to_csv(REVIEW / f"hourly_jumps_{mode}.csv", index=False)
        per_mode[mode] = {
            "reproduction_max_abs_diff_TRY_MWh": pmax,
            "n_jumps_gt_5": int(len(jumps)),
            "n_boundary_jumps": int(sum(j["is_month_boundary"] for j in jumps)),
            "n_intra_month_jumps": int(sum(not j["is_month_boundary"] for j in jumps)),
            "call_values": rows,
            "F_at_first_hour_TRY_MWh": float(curve.values.iloc[0]),
            "F_at_month_start_Feb_TRY_MWh": float(curve.at(
                pd.Timestamp("2026-01-31T21:00:00+00:00"))),
            "spot_consistent_at_t0": bool(model.spot_consistent),
        }

    pd.DataFrame(all_option_rows).to_csv(
        REVIEW / "option_value_comparison.csv", index=False)
    pd.concat(jan_frames).to_csv(
        REVIEW / "hourly_curve_shape_january.csv", index=False)

    # 3 --- Fixed sensitivity: sweep the ASSUMED SPOT under spot_to_next_linear.
    #       Shape (linear ramp to F_Feb) preserved; only the anchor level moves.
    base_spot = params.spot_price_TRY_MWh
    factors = (0.80, 0.90, 1.00, 1.10, 1.20)
    opt_ref = EuropeanOption(
        option_type="call", strike=STRIKE,
        valuation_utc=params.valuation_utc,
        maturity_utc=params.valuation_utc + pd.Timedelta(hours=72),
        r_annual=0.40,
    )
    fixed_rows = []
    for f in factors:
        s = round(base_spot * f, 2)
        anchor = NearTermAnchor(mode="spot_to_next_linear")
        curve = build_forward_curve(quotes, mode="smooth_constrained",
                                    anchor=anchor, spot_price_TRY_MWh=s)
        m_sens = _load_model(params, quotes, curve)
        es = m_sens.expected_spot(np.array(REPORTING_HORIZONS_HOURS, dtype=float))
        opt = opt_ref
        pr = price_forward_centered(m_sens, opt, gs)
        row = {
            "assumed_spot_TRY_MWh": s,
            "spot_vs_actual_pct": round(100.0 * (f - 1.0), 4),
            "F_at_t0_TRY_MWh": float(curve.values.iloc[0]),
            "F_at_Feb_start_TRY_MWh": float(curve.at(
                pd.Timestamp("2026-01-31T21:00:00+00:00"))),
            "max_abs_monthly_error_TRY_MWh": float(curve.max_abs_monthly_error()),
            "spot_consistent_at_t0": bool(m_sens.spot_consistent),
        }
        for h, v in zip(REPORTING_HORIZONS_HOURS, es):
            row[f"expected_spot_{h}h_TRY_MWh"] = float(v)
        row["call_K3000_72h_TRY_MWh"] = float(pr.value)
        fixed_rows.append(row)
    fixed = pd.DataFrame(fixed_rows)
    base_val = float(fixed["call_K3000_72h_TRY_MWh"].iloc[len(fixed) // 2])
    fixed["call_value_pct_vs_mid"] = (
        100.0 * (fixed["call_K3000_72h_TRY_MWh"] / base_val - 1.0))
    fixed.to_csv(REVIEW / "near_term_anchor_sensitivity_fixed.csv", index=False)

    # Comparison with the accepted (explicit_level) sensitivity table
    old = pd.read_csv("outputs/market_calibration_final/near_term_anchor_sensitivity.csv")
    old_slim = old[["january_anchor_TRY_MWh", "anchor_vs_spot_pct",
                    "spot_consistent_at_t0",
                    "expected_spot_72h_TRY_MWh",
                    "expected_spot_168h_TRY_MWh",
                    "expected_spot_336h_TRY_MWh",
                    "expected_spot_720h_TRY_MWh",
                    "option_value_TRY_MWh"]].copy()
    old_slim.columns = ["OLD_anchor_TRY_MWh", "OLD_anchor_vs_spot_pct",
                        "OLD_spot_consistent_at_t0",
                        "OLD_ES_72h", "OLD_ES_168h",
                        "OLD_ES_336h", "OLD_ES_720h",
                        "OLD_call_K3000_72h"]
    new_slim = fixed[["assumed_spot_TRY_MWh", "spot_vs_actual_pct",
                       "spot_consistent_at_t0",
                       "expected_spot_72h_TRY_MWh",
                       "expected_spot_168h_TRY_MWh",
                       "expected_spot_336h_TRY_MWh",
                       "expected_spot_720h_TRY_MWh",
                       "call_K3000_72h_TRY_MWh"]].copy()
    new_slim.columns = ["NEW_spot_TRY_MWh", "NEW_spot_vs_actual_pct",
                        "NEW_spot_consistent_at_t0",
                        "NEW_ES_72h", "NEW_ES_168h",
                        "NEW_ES_336h", "NEW_ES_720h",
                        "NEW_call_K3000_72h"]
    comparison = pd.concat([old_slim.reset_index(drop=True),
                             new_slim.reset_index(drop=True)], axis=1)
    comparison["delta_ES_72h"] = comparison["NEW_ES_72h"] - comparison["OLD_ES_72h"]
    comparison["delta_call_K3000_72h"] = (
        comparison["NEW_call_K3000_72h"] - comparison["OLD_call_K3000_72h"])
    comparison.to_csv(
        REVIEW / "near_term_anchor_sensitivity_comparison.csv", index=False)

    with open(REVIEW / "summary_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(per_mode, fh, indent=2)

    print("=== per-mode option values (K=3000 call) ===")
    for row in all_option_rows:
        print(f"  {row['anchor_mode']:22s} T={row['maturity_h']:>3d}h  "
              f"F(T)={row['forward_at_expiry_TRY_MWh']:8.2f}  "
              f"call={row['call_value_TRY_MWh']:8.2f}")
    print("\n=== hourly jumps > 5 TRY/MWh ===")
    for mode, m in per_mode.items():
        print(f"  {mode:22s} n={m['n_jumps_gt_5']:>3d}  "
              f"(boundary {m['n_boundary_jumps']}, intra {m['n_intra_month_jumps']})")
    print("\n=== files written ===")
    for p in sorted(REVIEW.glob("*")):
        if p.is_file():
            print(f"  {p}")


if __name__ == "__main__":
    main()
