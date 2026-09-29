"""FW10b -- corrected out-of-sample validation with the production
forward curve and the added short-horizon set h in {6, 12} hours.

Key differences from the FW10 pass:

  * F(target hour) comes from the PRODUCTION hourly forward curve
    (smooth constrained QP + spot_to_next_linear near-term anchor +
    HPFC shape) built for the valuation day, not the target-hour's
    delivery-month VEP quote.
  * The VEP quotation day used on day d is the last publication
    STRICTLY BEFORE d 11:00 TRT.
  * Horizons are h in {6, 12, 24, 48, 72} hours.  h = 6 and h = 12
    are new: FW9's headline reservation was that production
    underestimates variance BELOW 12 hours, and this is the direct
    out-of-sample test.  Under the day-ahead rule at 11:00 TRT
    of day d, no hour of day d+1 has been published, so the 6-72 h
    horizons that all fall between d 23:00 TRT and d+3 23:00 TRT are
    all forward-looking.
  * Bias / dispersion decomposition: per (d, h, model) we record
    z = (actual - sample_mean) / sample_sd; aggregated mean(z) is
    the centre bias and var(z) is the dispersion calibration
    (ideal 1); PIT sd (ideal 0.289 for a uniform).
  * Forward-level bias table F(h) - actual by horizon.

Every 3rd Turkish business day is retained to keep the paired-CRPS
DM residuals approximately non-overlapping at h = 72 (three trading
days apart).
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pde_option_model.calendar_tr import TURKEY_TZ
from pde_option_model.params_frozen import load_frozen_parameters
from scripts.fw10._data import (business_day_universe, day_end_utc,
                                 load_realized_ptf, load_vep_quotes_daily,
                                 valuation_utc)
from scripts.fw10.curve_builder import (F_at_hours,
                                          build_daily_production_curve)
from scripts.fw10.run_validation import (berkowitz_test,
                                          call_price_from_sample,
                                          christoffersen_indep,
                                          climatology_z_cycle, crps_sample,
                                          coverage_indicator, fit_hist_vol,
                                          kupiec_test, load_M0_production,
                                          load_M1_fw9e_A3, MSARParamSet,
                                          pinball_loss, pit_value,
                                          predict_bachelier, predict_black76,
                                          predict_lucia_schwartz,
                                          simulate_msar_paths, z_for_utc,
                                          COVERAGE_ALPHAS, QUANTILES,
                                          R_PER_HOUR, YAML_PATH)

OUT = REPO / "outputs" / "fw10_validation"
HORIZONS_H = (6, 12, 24, 48, 72)
MONEYNESS = (0.8, 0.9, 1.0, 1.1, 1.2)


def predict_msar_fw10b(params: MSARParamSet, last_known_utc: pd.Timestamp,
                        F_by_hour: Dict[int, float], scale_P: float,
                        z_cycle: np.ndarray, n_paths: int, seed: int
                        ) -> Dict[int, np.ndarray]:
    horizons = sorted(HORIZONS_H)
    n_steps = horizons[-1]
    step_times = [last_known_utc + pd.Timedelta(hours=k)
                  for k in range(1, n_steps + 1)]
    z_seq = np.array([z_for_utc(t, z_cycle) for t in step_times],
                     dtype=float)
    y_paths = simulate_msar_paths(params, z_seq, n_paths=n_paths, seed=seed)
    out = {}
    for h in horizons:
        y_T = y_paths[:, h]
        F = F_by_hour[h]
        delta = math.sqrt(F * F + scale_P * scale_P)
        out[h] = F + y_T * delta
    return out


def run_daily_evaluation(ptf: pd.Series, vep: pd.DataFrame,
                          days: List[pd.Timestamp], n_paths: int,
                          seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    yaml_p = load_frozen_parameters(YAML_PATH)
    scale_P = float(yaml_p.scale_P)
    m0 = load_M0_production()
    m1 = load_M1_fw9e_A3()
    z_cycle = climatology_z_cycle()
    pi_arr = np.asarray(yaml_p.m9_stationary_pi, dtype=float)
    sig_y_pooled = float(math.sqrt(pi_arr @ (np.asarray(yaml_p.sigma_y) ** 2)))
    kappa_prod = float(yaml_p.kappa_per_hour)

    rows_dist: List[dict] = []
    rows_opt: List[dict] = []

    for i, d in enumerate(days):
        # Build the production curve for this day
        try:
            res = build_daily_production_curve(d, ptf, vep, hpfc=True)
        except Exception:
            continue
        if res is None:
            continue
        # Collect F at each horizon and the realised terminal price
        try:
            F_by_hour = F_at_hours(res, list(HORIZONS_H))
        except Exception:
            continue
        actuals: Dict[int, float] = {}
        skip = False
        for h in HORIZONS_H:
            tgt = res.last_known_utc + pd.Timedelta(hours=h)
            if tgt not in ptf.index or not np.isfinite(F_by_hour[h]):
                skip = True
                break
            actuals[h] = float(ptf.loc[tgt])
        if skip:
            continue

        # MSAR samples
        try:
            samples_m0 = predict_msar_fw10b(m0, res.last_known_utc,
                                             F_by_hour, scale_P, z_cycle,
                                             n_paths, seed + i)
            samples_m1 = predict_msar_fw10b(m1, res.last_known_utc,
                                             F_by_hour, scale_P, z_cycle,
                                             n_paths, seed + i + 500_000)
        except Exception:
            continue
        # Historical vol at last_known
        vol = fit_hist_vol(ptf, cutoff_utc=res.last_known_utc)

        for h in HORIZONS_H:
            F = F_by_hour[h]
            actual = actuals[h]
            tau_h = float(h)
            samp_b1 = predict_black76(F, vol["sigma_log_d2h"], tau_h,
                                       n_paths, seed + i + 1_000_000 + h)
            samp_b2 = predict_bachelier(F, vol["sigma_abs_d2h"], tau_h,
                                         n_paths, seed + i + 2_000_000 + h)
            sigma_price = sig_y_pooled * math.sqrt(F * F + scale_P * scale_P)
            samp_b3 = predict_lucia_schwartz(F, sigma_price, kappa_prod,
                                              tau_h, n_paths,
                                              seed + i + 3_000_000 + h)
            for label, samp in [("M0", samples_m0[h]),
                                ("M1", samples_m1[h]),
                                ("B1", samp_b1), ("B2", samp_b2),
                                ("B3", samp_b3)]:
                pit = pit_value(samp, actual)
                crps = crps_sample(samp, actual)
                sd = float(samp.std(ddof=1))
                mean = float(samp.mean())
                z = (actual - mean) / sd if sd > 0 else float("nan")
                rec = {
                    "day": d.strftime("%Y-%m-%d"),
                    "quote_day": res.vep_quote_day,
                    "h": h, "model": label, "F": F, "actual": actual,
                    "sample_mean": mean, "sample_sd": sd,
                    "z_standardised": z, "pit": pit, "crps": crps,
                    "F_minus_actual": F - actual,
                }
                for q in QUANTILES:
                    rec[f"pin_q{int(100*q):02d}"] = pinball_loss(samp, actual, q)
                for a in COVERAGE_ALPHAS:
                    hit, lo, hi = coverage_indicator(samp, actual, a)
                    rec[f"cover_{int(100*a):02d}"] = int(hit)
                rows_dist.append(rec)
                for mm in MONEYNESS:
                    K = mm * F
                    price = call_price_from_sample(samp, K, tau_h, R_PER_HOUR)
                    payoff = max(actual - K, 0.0) * math.exp(
                        -R_PER_HOUR * tau_h)
                    rows_opt.append({
                        "day": d.strftime("%Y-%m-%d"), "h": h,
                        "model": label, "moneyness": mm, "K": K, "F": F,
                        "actual": actual, "call_price": price,
                        "discounted_payoff": payoff,
                        "error": price - payoff,
                    })
    return pd.DataFrame(rows_dist), pd.DataFrame(rows_opt)


def build_bias_dispersion_table(dist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS_H:
        for model in ("M0", "M1", "B1", "B2", "B3"):
            sub = dist[(dist["h"] == h) & (dist["model"] == model)]
            if sub.empty:
                continue
            z = sub["z_standardised"].to_numpy()
            z = z[np.isfinite(z)]
            F_minus_actual = sub["F_minus_actual"].to_numpy()
            pit = sub["pit"].to_numpy()
            crps = sub["crps"].to_numpy()
            rows.append({
                "h": h, "model": model, "n_days": int(sub.shape[0]),
                "crps_mean": float(crps.mean()),
                "mean_z_center_bias": float(z.mean()) if z.size else float("nan"),
                "var_z_dispersion_calib": float(z.var(ddof=1)) if z.size > 1 else float("nan"),
                "pit_mean": float(pit.mean()),
                "pit_sd": float(pit.std(ddof=1)) if pit.size > 1 else float("nan"),
                "pit_sd_ideal": 1.0 / math.sqrt(12.0),
                "forward_bias_TRY": float(F_minus_actual.mean()),
                "forward_bias_sd_TRY": float(F_minus_actual.std(ddof=1)) if F_minus_actual.size > 1 else float("nan"),
            })
    return pd.DataFrame(rows)


def build_calibration_summary(dist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS_H:
        for model in ("M0", "M1", "B1", "B2", "B3"):
            sub = dist[(dist["h"] == h) & (dist["model"] == model)]
            if sub.empty:
                continue
            pit = sub["pit"].to_numpy()
            crps = sub["crps"].to_numpy()
            rec = {
                "h": h, "model": model, "n_days": int(sub.shape[0]),
                "crps_mean": float(crps.mean()),
                "pit_mean": float(pit.mean()),
                "pit_sd": float(pit.std(ddof=1)) if pit.size > 1 else float("nan"),
            }
            from scipy.stats import kstest
            try:
                ks_stat, ks_p = kstest(pit, "uniform")
                rec["ks_stat"] = float(ks_stat); rec["ks_p"] = float(ks_p)
            except Exception:
                rec["ks_stat"] = float("nan"); rec["ks_p"] = float("nan")
            berk_lr, berk_p = berkowitz_test(pit)
            rec["berk_lr"] = berk_lr; rec["berk_p"] = berk_p
            for a in COVERAGE_ALPHAS:
                hits = sub[f"cover_{int(100*a):02d}"].to_numpy()
                rec[f"cover_{int(100*a):02d}_freq"] = float(hits.mean())
                _, kup_p = kupiec_test(hits, a)
                _, cc_p = christoffersen_indep(hits)
                rec[f"cover_{int(100*a):02d}_kupiec_p"] = kup_p
                rec[f"cover_{int(100*a):02d}_indep_p"] = cc_p
            rows.append(rec)
    return pd.DataFrame(rows)


def build_dm_table(dist: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import t as tdist
    rows = []
    for h in HORIZONS_H:
        base = dist[(dist["h"] == h) & (dist["model"] == "M0")] \
            .set_index("day")["crps"]
        for alt in ("M1", "B1", "B2", "B3"):
            other = dist[(dist["h"] == h) & (dist["model"] == alt)] \
                .set_index("day")["crps"]
            paired = pd.concat([base, other], axis=1, keys=["m0", "alt"]) \
                .dropna()
            if paired.empty:
                continue
            diff = (paired["m0"] - paired["alt"]).to_numpy()
            n = diff.size
            if n < 3:
                continue
            mean_d = float(diff.mean())
            sd_d = float(diff.std(ddof=1))
            se_d = sd_d / math.sqrt(n) if sd_d > 0 else float("nan")
            t_stat = mean_d / se_d if se_d > 0 else float("nan")
            p = float(2.0 * (1.0 - tdist.cdf(abs(t_stat), df=n - 1))) \
                if math.isfinite(t_stat) else float("nan")
            rows.append({
                "h": h, "alt": alt, "n": n,
                "mean_diff_M0_minus_alt": mean_d,
                "sd_diff": sd_d, "se_diff": se_d,
                "t_stat": t_stat, "p_value": p,
                "M0_beats_alt": bool(mean_d < 0),
            })
    return pd.DataFrame(rows)


def build_option_bias_table(opt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, h, mm), sub in opt.groupby(["model", "h", "moneyness"]):
        err = sub["error"].to_numpy()
        rows.append({
            "model": model, "h": h, "moneyness": mm,
            "n_days": sub.shape[0],
            "mean_error_TRY": float(err.mean()),
            "MAE_TRY": float(np.mean(np.abs(err))),
            "mean_call": float(sub["call_price"].mean()),
            "mean_payoff": float(sub["discounted_payoff"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["model", "h", "moneyness"]) \
        .reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", type=int, default=3)
    ap.add_argument("--n-paths", type=int, default=10_000)
    ap.add_argument("--start", type=str, default="2026-01-05")
    ap.add_argument("--end", type=str, default="2026-09-24")
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("loading data...")
    ptf = load_realized_ptf(include_history=True)
    vep = load_vep_quotes_daily()
    print(f"  PTF total: n={len(ptf)}, first {ptf.index[0]}, last {ptf.index[-1]}")
    print(f"  VEP: n={len(vep)}, days={vep['valuation_date'].nunique()}, "
          f"last quote day = {vep['valuation_date'].max()}")

    universe = business_day_universe(args.start, args.end)
    days = universe[::args.cadence]
    print(f"eval universe: {len(universe)} biz days -> thinned to "
          f"{len(days)} (cadence {args.cadence})")

    t0 = time.time()
    dist, opt = run_daily_evaluation(ptf, vep, days, args.n_paths,
                                       args.seed)
    dt = time.time() - t0
    print(f"daily loop finished in {dt:.1f}s: dist={dist.shape}, "
          f"opt={opt.shape}")

    dist.to_csv(OUT / "predictive_daily_fw10b.csv", index=False)
    opt.to_csv(OUT / "option_daily_fw10b.csv", index=False)

    bias_disp = build_bias_dispersion_table(dist)
    bias_disp.to_csv(OUT / "bias_dispersion_summary.csv", index=False)
    print("\n=== Bias / dispersion decomposition ===")
    print(bias_disp[["h", "model", "n_days", "crps_mean",
                     "mean_z_center_bias", "var_z_dispersion_calib",
                     "pit_mean", "pit_sd", "forward_bias_TRY"]]
          .round(3).to_string(index=False))

    cal = build_calibration_summary(dist)
    cal.to_csv(OUT / "pit_coverage_summary_fw10b.csv", index=False)
    print("\n=== PIT & coverage (compact) ===")
    print(cal[["h", "model", "n_days", "crps_mean", "pit_mean",
               "ks_p", "berk_p",
               "cover_50_freq", "cover_50_kupiec_p",
               "cover_90_freq", "cover_90_kupiec_p"]]
          .round(4).to_string(index=False))

    dm = build_dm_table(dist)
    dm.to_csv(OUT / "dm_crps_M0_vs_alt_fw10b.csv", index=False)
    print("\n=== Diebold-Mariano on CRPS (M0 vs each) ===")
    print(dm.round(4).to_string(index=False))

    opt_bias = build_option_bias_table(opt)
    opt_bias.to_csv(OUT / "option_bias_by_moneyness_fw10b.csv", index=False)
    print("\n=== Option-level bias (mean error TRY) ===")
    piv = opt_bias.pivot_table(index=["model", "h"], columns="moneyness",
                                values="mean_error_TRY")
    print(piv.round(1).to_string())

    # Fig CSVs
    pit_hist_rows = []
    bins = np.linspace(0.0, 1.0, 11)
    for h in HORIZONS_H:
        for model in ("M0", "M1", "B1", "B2", "B3"):
            sub = dist[(dist["h"] == h) & (dist["model"] == model)]
            if sub.empty:
                continue
            hist, _ = np.histogram(sub["pit"].to_numpy(), bins=bins)
            for k in range(hist.size):
                pit_hist_rows.append({
                    "h": h, "model": model,
                    "bin_lo": bins[k], "bin_hi": bins[k + 1],
                    "count": int(hist[k]),
                    "freq": float(hist[k]) / max(1, sub.shape[0]),
                })
    pd.DataFrame(pit_hist_rows).to_csv(OUT / "fig_pit_histogram_fw10b.csv",
                                         index=False)
    opt_bias[["model", "h", "moneyness", "mean_error_TRY", "MAE_TRY"]] \
        .to_csv(OUT / "fig_option_bias_fw10b.csv", index=False)

    # MD summary
    md = ["# FW10b Part A -- Predictive-distribution validation "
          "with the production forward curve\n",
          f"* Evaluation window: {args.start} -> {args.end}, business "
          f"days thinned every {args.cadence} days -> {len(days)} "
          f"valuation dates; N_PATHS = {args.n_paths}.\n",
          "* Forward: production hourly curve rebuilt on each "
          "valuation day (smooth constrained QP + spot_to_next_linear "
          "anchor + HPFC shape).  VEP quote day = last publication "
          "strictly before d 11:00 TRT.\n",
          "* New horizons h in {6, 12}: FW9's headline reservation "
          "that production underestimates variance below 12 h is "
          "tested here for the first time out of sample.\n",
          "## Bias / dispersion decomposition\n",
          bias_disp[["h", "model", "n_days", "crps_mean",
                      "mean_z_center_bias", "var_z_dispersion_calib",
                      "pit_mean", "pit_sd", "forward_bias_TRY"]]
          .round(3).to_markdown(index=False),
          "\n\n## PIT / coverage / DM (compact)\n",
          cal[["h", "model", "n_days", "crps_mean", "pit_mean",
                "ks_p", "berk_p", "cover_50_freq", "cover_90_freq"]]
          .round(4).to_markdown(index=False),
          "\n\n## Diebold-Mariano on CRPS (M0 vs alternatives)\n",
          dm.round(4).to_markdown(index=False),
          "\n\n## Option-level bias by moneyness (mean error TRY)\n",
          piv.round(1).to_markdown()]
    (OUT / "fw10b_partA_summary.md").write_text("\n".join(md),
                                                  encoding="utf-8")
    print("wrote outputs/fw10_validation/*_fw10b.csv, *.md")


if __name__ == "__main__":
    main()
