"""Calibrate the Q1 forward risk-premium curve pi(tau) and test it on 2026.

Usage (from the repository root):

    python scripts/premium/calibrate_premium.py ^
        --vep-history inputs/market/vep_history/vep_history.csv ^
        --ptf-dir inputs/historical/ptf_raw ^
        --cutoff 2025-12-31

Outputs (default outdir: outputs/premium/):
    premium_panel.csv        realised premia, one row per (valuation date, delivery month)
    premium_curve.csv        smoothed relative premium by months-to-delivery (+ SE, n)
    premium_backtest_2026.csv / .md
                             2026 hourly backtest: Q mean F(t) vs P mean F(t)-pi(t)
                             (only when --cutoff <= 2025-12-31, i.e. no look-ahead)

Q-measure option prices are NOT changed by this script: E^Q[P_t] = F(t) holds
by construction.  The premium only defines the real-world (P) mean.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pde_option_model.premium import (drift_from_premium, fit_premium_curve,  # noqa: E402
                                      load_epias_ptf_csv, load_epias_ptf_dir,
                                      load_vep_history, premium_path_hours,
                                      realized_monthly_means, realized_premium_panel)


def _backtest_2026(curve, kappa: float, outdir: Path, tag: str) -> pd.DataFrame:
    fwd = pd.read_csv(REPO / "outputs/market_calibration_final/hourly_forward_curve.csv",
                      parse_dates=["time_utc"])
    real = load_epias_ptf_csv(REPO / "inputs/market/realized_ptf_2026.csv").rename("realized")
    df = fwd.merge(real, left_on="time_utc", right_index=True, how="inner")
    t0 = df["time_utc"].iloc[0]
    t = (df["time_utc"] - t0).dt.total_seconds().to_numpy() / 3600.0
    F = df["hourly_forward_TRY_MWh"].to_numpy()
    pi = premium_path_hours(t, F, curve)
    df["pi_TRY_MWh"] = pi
    df["P_mean_TRY_MWh"] = F - pi
    df["a_drift_TRY_MWh_per_h"] = drift_from_premium(t, pi, kappa)
    df = df[df["delivery_month"] != df["delivery_month"].iloc[0]]     # drop partial Dec
    rows = []
    for m, g in df.groupby("delivery_month"):
        eq = g["realized"] - g["hourly_forward_TRY_MWh"]
        ep = g["realized"] - g["P_mean_TRY_MWh"]
        rows.append(dict(month=m, n=len(g), mean_pi=g["pi_TRY_MWh"].mean(),
                         bias_Q=eq.mean(), bias_P=ep.mean(),
                         MAE_Q=eq.abs().mean(), MAE_P=ep.abs().mean(),
                         RMSE_Q=np.sqrt((eq ** 2).mean()), RMSE_P=np.sqrt((ep ** 2).mean())))
    eq = df["realized"] - df["hourly_forward_TRY_MWh"]
    ep = df["realized"] - df["P_mean_TRY_MWh"]
    rows.append(dict(month="ALL", n=len(df), mean_pi=df["pi_TRY_MWh"].mean(),
                     bias_Q=eq.mean(), bias_P=ep.mean(), MAE_Q=eq.abs().mean(),
                     MAE_P=ep.abs().mean(), RMSE_Q=np.sqrt((eq ** 2).mean()),
                     RMSE_P=np.sqrt((ep ** 2).mean())))
    res = pd.DataFrame(rows).round(2)
    res.to_csv(outdir / "premium_backtest_2026.csv", index=False)
    (outdir / "premium_backtest_2026.md").write_text(
        f"# 2026 backtest: Q mean F(t) vs P mean F(t) - pi(t)  [{tag}]\n\n"
        "bias = realised - mean.  Q-option prices are unaffected by pi.\n\n"
        + res.to_markdown(index=False) + "\n", encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vep-history", type=Path)
    ap.add_argument("--ptf-dir", type=Path, default=REPO / "inputs" / "historical" / "ptf_raw")
    ap.add_argument("--cutoff", default="2025-12-31",
                    help="information cutoff (Turkish local date, end of day)")
    ap.add_argument("--max-tau", type=int, default=12)
    ap.add_argument("--smoothness", type=float, default=5.0)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--kappa", type=float, default=0.01,
                    help="OU rate of the factor carrying the premium (slow factor), 1/h")
    ap.add_argument("--outdir", type=Path, default=REPO / "outputs" / "premium")
    a = ap.parse_args(argv)

    a.outdir.mkdir(parents=True, exist_ok=True)
    cutoff = (pd.Timestamp(a.cutoff) + pd.Timedelta(hours=23)).tz_localize(
        "Europe/Istanbul").tz_convert("UTC")
    if a.vep_history is None:
        ap.error("--vep-history is required (EPİAŞ VEP history in the repository schema)")
    tag = "ESTIMATED"
    vep = load_vep_history(a.vep_history)
    real = realized_monthly_means(load_epias_ptf_dir(a.ptf_dir))

    panel = realized_premium_panel(vep, real, cutoff_utc=cutoff, max_tau=a.max_tau)
    if panel.empty:
        print("No usable (quote, realised month) pairs before the cutoff: the VEP history "
              "must contain quotes whose delivery months ended before --cutoff.")
        return 1
    curve = fit_premium_curve(panel, max_tau=a.max_tau, smoothness=a.smoothness,
                              ridge=a.ridge)
    curve.label = tag
    panel.to_csv(a.outdir / "premium_panel.csv", index=False)
    cf = curve.to_frame()
    cf.insert(0, "label", tag)
    cf.to_csv(a.outdir / "premium_curve.csv", index=False)

    print(f"\nQ1 forward risk premium  [{tag}]  cutoff {a.cutoff}")
    print(f"panel rows: {len(panel)}   delivery months: "
          f"{panel[['delivery_year', 'delivery_month']].drop_duplicates().shape[0]}")
    print(cf.round(4).to_string(index=False))
    for n in curve.notes:
        print("NOTE:", n)

    if cutoff <= pd.Timestamp("2025-12-31 20:00", tz="UTC"):
        res = _backtest_2026(curve, a.kappa, a.outdir, tag)
        print("\n2026 backtest (realised - mean), TRY/MWh:")
        print(res.to_string(index=False))
    else:
        print("\n2026 backtest skipped: cutoff after 2025-12-31 would be look-ahead.")
    print(f"\nwritten to {a.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
