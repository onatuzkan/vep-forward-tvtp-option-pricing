"""Fit the recency-weighted HPFC shape, apply it to the VEP curve, test on 2026.

Usage (repository root, venv active):

    python scripts/hpfc/fit_hpfc.py
    python scripts/hpfc/fit_hpfc.py --half-life 1.0 --harmonics 2     # skip CV

Steps
  1. Load hourly PTF 2019-2025 from inputs/historical/ptf_raw/.
  2. Rolling-origin CV (fit before 1 Jan Y, score year Y; Y = 2023, 2024, 2025)
     over half-life H and number of seasonal harmonics K.
  3. Fit the chosen (H, K) on all data before the valuation 2025-12-31 23:00 TRT
     (no look-ahead into 2026).
  4. Multiply the accepted smooth VEP curve by the normalised shape; every
     monthly average (hence every VEP quote) is preserved exactly.
  5. Compare smooth vs HPFC against the realised 2026 PTF.

Outputs -> outputs/hpfc/
  hpfc_cv_results.csv, hpfc_selection.json, hourly_forward_curve_hpfc.csv,
  hpfc_profile_2026.csv, hpfc_eval_2026.csv, hpfc_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", message="Converting to PeriodArray")

from pde_option_model.hpfc import (apply_shape_to_curve, evaluate_against_realized,  # noqa: E402
                                   fit_shape, select_half_life)
from pde_option_model.premium import load_epias_ptf_csv, load_epias_ptf_dir       # noqa: E402

VALUATION_UTC = pd.Timestamp("2025-12-31 20:00", tz="UTC")   # 23:00 TRT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ptf-dir", type=Path, default=REPO / "inputs/historical/ptf_raw")
    ap.add_argument("--curve", type=Path,
                    default=REPO / "outputs/market_calibration_final/hourly_forward_curve.csv")
    ap.add_argument("--realized", type=Path, default=REPO / "inputs/market/realized_ptf_2026.csv")
    ap.add_argument("--half-life", type=float, default=None,
                    help="years; 0 = equal weights; omit to select by CV")
    ap.add_argument("--harmonics", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=REPO / "outputs/hpfc")
    a = ap.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)

    ptf = load_epias_ptf_dir(a.ptf_dir)
    ptf = ptf[ptf.index <= VALUATION_UTC]
    print(f"PTF history: {len(ptf)} h, {ptf.index.min()} -> {ptf.index.max()}")

    sel = {}
    if a.half_life is None or a.harmonics is None:
        print("rolling-origin CV (test years 2023-2025) ...")
        res, best = select_half_life(ptf)
        res.to_csv(a.outdir / "hpfc_cv_results.csv", index=False)
        table = (res.groupby(["n_harmonics", "half_life_years"])["ratio_rmse"].mean()
                 .unstack(0).round(4))
        print("mean out-of-sample ratio RMSE (rows: half-life years, cols: harmonics)")
        print(table.to_string())
        sel = dict(best, method="rolling-origin CV")
        H, K = best["half_life_years"], best["n_harmonics"]
    else:
        H = None if a.half_life == 0 else a.half_life
        K = a.harmonics
        sel = dict(half_life_years=H, n_harmonics=K, method="user-specified")
    print(f"\nselected: half-life = {H if H is not None else 'inf (equal)'} y, harmonics = {K}")

    cutoff = VALUATION_UTC + pd.Timedelta(hours=1)
    model = fit_shape(ptf, cutoff, H, K)
    curve = pd.read_csv(a.curve)
    hp = apply_shape_to_curve(curve, model)
    hp.to_csv(a.outdir / "hourly_forward_curve_hpfc.csv", index=False)
    model.profile_table(2026).to_csv(a.outdir / "hpfc_profile_2026.csv", index=False)

    ym = (pd.to_datetime(hp["time_utc"], utc=True).dt.tz_convert("Europe/Istanbul")
          .dt.strftime("%Y-%m"))
    g = hp.groupby(ym)[["hourly_forward_TRY_MWh", "hpfc_TRY_MWh"]].mean()
    max_month_err = float((g.iloc[:, 0] - g.iloc[:, 1]).abs().max())
    print(f"max |monthly avg HPFC - monthly avg VEP curve| = {max_month_err:.2e} TRY/MWh")

    realized = load_epias_ptf_csv(a.realized)
    ev = evaluate_against_realized(hp, realized).round(2)
    ev.to_csv(a.outdir / "hpfc_eval_2026.csv", index=False)
    eq = evaluate_against_realized(apply_shape_to_curve(curve, fit_shape(ptf, cutoff, None, K)),
                                   realized)
    print("\n2026 out-of-sample (TRY/MWh). *_demeaned removes each month's VEP level miss:")
    print(ev.to_string(index=False))
    print(f"\nequal-weight shape, same K: variance reduction "
          f"{eq['var_reduction_pct'].iloc[-1]:.1f} % (recency: {ev['var_reduction_pct'].iloc[-1]:.1f} %)")

    sel.update(max_abs_monthly_error_TRY_MWh=max_month_err,
               var_reduction_2026_pct=float(ev["var_reduction_pct"].iloc[-1]),
               var_reduction_2026_equal_weight_pct=float(eq["var_reduction_pct"].iloc[-1]),
               cutoff_utc=str(cutoff), n_obs=model.n_obs)
    (a.outdir / "hpfc_selection.json").write_text(json.dumps(sel, indent=2, default=str),
                                                  encoding="utf-8")
    (a.outdir / "hpfc_report.md").write_text(
        "# Recency-weighted HPFC\n\n"
        f"* selection: {sel['method']}; half-life = {H} y; harmonics = {K}\n"
        f"* fitted on {model.n_obs} hours before {cutoff} (no 2026 data)\n"
        f"* VEP monthly averages preserved to {max_month_err:.1e} TRY/MWh\n\n"
        "## 2026 out-of-sample\n\n" + ev.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"\nwritten to {a.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
