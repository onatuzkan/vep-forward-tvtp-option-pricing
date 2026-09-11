"""Re-estimate the residual dynamics (v2) and test them on 2026.

Usage (repository root, venv active; run scripts/hpfc/fit_hpfc.py first):

    python scripts/residual/fit_residual_v2.py
    python scripts/residual/fit_residual_v2.py --train-start 2021 --paths 4000

Steps
  1. Hourly panel 2023-2025 (default): ex-ante HPFC shape, trailing 12-month
     scale L, x = (P - M*S)/L split into daily mean d and hourly deviation u.
  2. Fast factor: Markov-switching AR(1) with TVTP on z_{t-1} and ramp_{t-1}
     (exact Hamilton-filter MLE, Hessian standard errors).
  3. Slow factor: AR(1) on the daily mean.  Level factor: naive proxy.
  4. Monte Carlo of clipped prices [0, cap] over the 2026 backtest horizon with
     historical-year covariate bootstrap; coverage / sd / RMSE vs realised PTF,
     side by side with the v1 model.
  5. 72 h European call K=3000 (and a strike strip) under v2 vs v1.

Outputs -> outputs/residual_v2/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", message="Converting to PeriodArray")

from pde_option_model.premium import load_epias_ptf_csv, load_epias_ptf_dir   # noqa: E402
from pde_option_model.residual_v2 import (ResidualModelV2, build_residual_panel,  # noqa: E402
                                          coverage_table, fit_fast_msar, fit_slow_daily,
                                          level_uncertainty, load_rd_covariates,
                                          mc_option_price, simulate_prices)

VALUATION_UTC = pd.Timestamp("2025-12-31 20:00", tz="UTC")


def _covariate_bootstrap(cov: pd.DataFrame, times_utc: pd.DatetimeIndex, n_paths: int,
                         seed: int, years=range(2016, 2026)):
    """For each path draw one historical year and use its z / ramp on the same
    calendar hour (Feb-29 and missing hours forward-filled)."""
    rng = np.random.default_rng(seed)
    loc_cov = cov.index.tz_convert("Europe/Istanbul")
    key = lambda ix: (ix.month * 100 + ix.day) * 100 + ix.hour          # noqa: E731
    tgt = key(times_utc.tz_convert("Europe/Istanbul"))
    Z, R = {}, {}
    for y in years:
        m = loc_cov.year == y
        if not m.any():
            continue
        sub = cov[m].copy()
        sub.index = key(loc_cov[m])
        sub = sub[~sub.index.duplicated()]
        Z[y] = sub["z1"].reindex(tgt).ffill().bfill().to_numpy()
        R[y] = sub["r1"].reindex(tgt).ffill().bfill().to_numpy()
    ys = rng.choice(sorted(Z), n_paths)
    return np.stack([Z[y] for y in ys], 1), np.stack([R[y] for y in ys], 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ptf-dir", type=Path, default=REPO / "inputs/historical/ptf_raw")
    ap.add_argument("--rd", type=Path, default=REPO / "inputs/historical/rd_standardized.csv")
    ap.add_argument("--hpfc", type=Path, default=REPO / "outputs/hpfc/hourly_forward_curve_hpfc.csv")
    ap.add_argument("--realized", type=Path, default=REPO / "inputs/market/realized_ptf_2026.csv")
    ap.add_argument("--backtest", type=Path,
                    default=REPO / "outputs/market_calibration_final/realized_2026_backtest.csv")
    ap.add_argument("--train-start", type=int, default=2023)
    ap.add_argument("--train-end", type=int, default=2025)
    ap.add_argument("--paths", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--outdir", type=Path, default=REPO / "outputs/residual_v2")
    a = ap.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)
    if not a.hpfc.exists():
        sys.exit(f"{a.hpfc} not found -- run scripts/hpfc/fit_hpfc.py first")

    ptf = load_epias_ptf_dir(a.ptf_dir)
    ptf = ptf[ptf.index <= VALUATION_UTC]
    cov = load_rd_covariates(a.rd)

    # ---- 1-3 estimation --------------------------------------------------
    t0 = time.time()
    panel = build_residual_panel(ptf, a.train_start, a.train_end)
    panel = panel.join(cov, how="left").dropna(subset=["u", "z1", "r1", "L"])
    print(f"panel {a.train_start}-{a.train_end}: {len(panel)} h   sd(x)={panel['x'].std():.3f} L-units")
    fast = fit_fast_msar(panel["u"].to_numpy(), panel["z1"].to_numpy(), panel["r1"].to_numpy())
    daily = panel.groupby("day")["x"].mean()
    slow = fit_slow_daily(daily)
    lev = level_uncertainty(ptf)
    loc = ptf.copy(); loc.index = ptf.index.tz_convert("Europe/Istanbul").tz_localize(None)
    mm = loc.groupby(loc.index.to_period("M")).mean()
    scale_L = float(mm.iloc[-12:].mean())                 # trailing 12 months at valuation
    model = ResidualModelV2(fast=fast, slow=slow, level=lev, scale_L=scale_L,
                            estimation_window=f"{a.train_start}-{a.train_end}")
    print(f"estimation done in {time.time() - t0:.1f}s  converged={fast.converged}")
    print("\nfast factor (MS-AR(1), TVTP on z_{t-1}, ramp_{t-1}):")
    for k, v in fast.params.items():
        print(f"  {k:7s} = {v: .4f}   (se {fast.se[k]:.4f})")
    s0, s1 = fast.sigma
    print(f"  sigma normal/stress = {s0:.4f} / {s1:.4f} L-units per hour; "
          f"stress share {fast.stationary_stress_share:.2f}")
    print(f"slow factor: phi_daily={slow['phi_daily']:.3f} (half-life {slow['half_life_days']:.2f} d), "
          f"sigma_daily={slow['sigma_daily']:.4f}")
    print("level proxy sigma_log by tau:", dict(zip(lev.tau_months, lev.sigma_log.round(3))))
    cont = model.continuous()
    print("continuous-time (TRY, scale L = %.0f):" % scale_L,
          {k: round(v, 4) for k, v in cont.items()})

    # ---- 4 2026 backtest -------------------------------------------------
    hp = pd.read_csv(a.hpfc)
    hp["t"] = pd.to_datetime(hp["time_utc"], utc=True)
    real = load_epias_ptf_csv(a.realized).rename("P")
    df = hp.merge(real, left_on="t", right_index=True)
    df = df[df["t"] > VALUATION_UTC].reset_index(drop=True)
    times = pd.DatetimeIndex(df["t"])
    Z, R = _covariate_bootstrap(cov, times, a.paths, a.seed)
    t0 = time.time()
    sim = simulate_prices(model, times, df["hpfc_TRY_MWh"].to_numpy(), VALUATION_UTC, Z, R,
                          n_paths=a.paths, seed=a.seed, keep_paths=True)
    print(f"\nMonte Carlo {a.paths} paths x {len(times)} h in {time.time() - t0:.1f}s")
    months = df["delivery_month"].to_numpy()
    cov_v2 = coverage_table(df["P"].to_numpy(), sim["paths"], months,
                            df["hpfc_TRY_MWh"].to_numpy())

    # v1 coverage from the shipped backtest (Gaussian with analytic sd)
    from scipy.stats import norm
    bt = pd.read_csv(a.backtest, parse_dates=["time_utc"])
    bt = bt[bt["delivery_month"] != "2025-12"]
    zz = (bt["realized_TRY_MWh"] - bt["hourly_forward_TRY_MWh"]) / bt["model_residual_sd_TRY_MWh"]
    pit1 = norm.cdf(zz)
    v1 = []
    for m, g in list(pd.DataFrame(dict(m=bt["delivery_month"], p=pit1,
                                       sd=bt["model_residual_sd_TRY_MWh"])).groupby("m")) \
            + [("ALL", pd.DataFrame(dict(p=pit1, sd=bt["model_residual_sd_TRY_MWh"])))]:
        v1.append(dict(month=m, v1_model_sd=g["sd"].mean(),
                       v1_cov50=100 * np.mean((g["p"] > .25) & (g["p"] < .75)),
                       v1_cov90=100 * np.mean((g["p"] > .05) & (g["p"] < .95))))
    tab = cov_v2.merge(pd.DataFrame(v1), on="month").round(1)
    print("\n2026 out-of-sample: v2 (HPFC + re-estimated residual + level proxy + cap) vs v1")
    print(tab[["month", "n", "rmse", "model_sd", "cov50", "cov80", "cov90",
               "v1_model_sd", "v1_cov50", "v1_cov90"]].to_string(index=False))

    # ---- 5 option prices -------------------------------------------------
    rows = []
    for h in (24, 72, 168, 336):
        T = VALUATION_UTC + pd.Timedelta(hours=h)
        j = int(np.searchsorted(times, T))
        for K in (2000, 2600, 3000, 3400, 4000):
            c = mc_option_price(sim["paths"][j].astype(float), K, h)
            p = mc_option_price(sim["paths"][j].astype(float), K, h, kind="put")
            rows.append(dict(maturity_h=h, strike=K, F_T=float(df["hpfc_TRY_MWh"].iloc[j]),
                             cap=float(sim["cap"][j]), call_v2=c["value"], call_se=c["mc_se"],
                             put_v2=p["value"]))
    opt = pd.DataFrame(rows)
    grid = REPO / "outputs/market_calibration_final/strike_maturity_grid.csv"
    if grid.exists():
        g = pd.read_csv(grid).rename(columns={"strike_TRY_MWh": "strike",
                                              "call_TRY_MWh": "call_v1"})
        opt = opt.merge(g[["maturity_h", "strike", "call_v1"]], on=["maturity_h", "strike"],
                        how="left")
    opt["call_upper_bound"] = np.exp(-0.40 * opt["maturity_h"] / 8760) * np.clip(
        opt["cap"] - opt["strike"], 0, None)
    print("\nEuropean options on the expiry-hour PTF (v2 Monte Carlo, clipped to [0, cap]):")
    print(opt.round(1).to_string(index=False))

    # ---- write -------------------------------------------------------------
    tab.to_csv(a.outdir / "coverage_2026_v2_vs_v1.csv", index=False)
    opt.to_csv(a.outdir / "option_prices_v2_vs_v1.csv", index=False)
    (a.outdir / "residual_v2_params.json").write_text(
        json.dumps(model.summary(), indent=2, default=float), encoding="utf-8")
    (a.outdir / "residual_v2_report.md").write_text(
        "# Residual dynamics v2\n\n"
        f"* estimation window {a.train_start}-{a.train_end}, {fast.n_obs} hours, "
        f"converged={fast.converged}\n"
        f"* scale L at valuation = {scale_L:.0f} TRY/MWh\n"
        "* level factor = naive monthly-change proxy [ASSUMED]\n\n"
        "## 2026 coverage (v2 vs v1)\n\n" + tab.to_markdown(index=False) +
        "\n\n## Options\n\n" + opt.round(2).to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"\nwritten to {a.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
