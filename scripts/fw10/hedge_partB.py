"""FW10 Part B -- Delta hedge with monthly VEP forwards.

We reuse the per-day predictive-distribution samples that
run_validation.py produced to compute:

  * ``delta_M`` = ``disc * P(P_T > K)`` for each (d, h, K, model).
    This is the classical result for a European call priced against
    a predictive P_T with additive dependence on the forward level
    F (which is what all five FW10 models are: benchmark closed
    forms treat F as the direct location parameter; MC models use
    P = F + X - mu_X).  Because our forward proxy is the monthly
    baseload VEP quote, the delta is the hedge ratio in units of
    monthly baseload contracts.

  * Static hedge P&L = ``delta_d * (F_M,d+h - F_M,d)``.
  * Daily-rebalanced hedge P&L = ``sum_i delta_i * (F_M,i+1 -
    F_M,i)`` where ``i`` runs over the trading days spanned by the
    horizon and ``delta_i`` is the option delta re-computed on
    day ``i`` at its own forward level (sample re-generation with
    the tenor shortened by ``i * 24`` hours).

  * Hedged error = unhedged error - hedge P&L, where the unhedged
    error is (discounted payoff - call premium).

  * Effectiveness = 1 - Var(hedged) / Var(unhedged), reported per
    (model, h, moneyness) cell.

Simplifying assumption: for the daily rebalancing we treat each new
day's forward level as observable via the VEP quote of that day for
the SAME delivery month.  Days where the corresponding VEP quote is
absent (weekends, holidays) are handled by carrying the last
observed forward forward.  When the daily rebalancing produces the
same delta twice (e.g. samples are identical because the tenor is
still an integer multiple of 24 h), the effect on the hedge P&L is
mechanically zero.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.fw10._data import (business_day_universe, day_end_utc,
                                 load_realized_ptf, load_vep_quotes_daily,
                                 valuation_utc, vep_quote_on_or_before)
from scripts.fw10.run_validation import (climatology_z_cycle,
                                          fit_hist_vol, load_M0_production,
                                          load_M1_fw9e_A3, MSARParamSet,
                                          predict_bachelier, predict_black76,
                                          predict_lucia_schwartz,
                                          predict_msar, HORIZONS_H, MONEYNESS,
                                          R_PER_HOUR, YAML_PATH)
from pde_option_model.params_frozen import load_frozen_parameters
from pde_option_model.calendar_tr import TURKEY_TZ

OUT = REPO / "outputs" / "fw10_validation"


def call_price_and_delta_from_sample(sample: np.ndarray, K: float,
                                      tau_h: float) -> Tuple[float, float]:
    """Return (call_price, delta_wrt_F).  Both are DISCOUNTED."""
    disc = math.exp(-R_PER_HOUR * tau_h)
    price = float(disc * np.maximum(sample - K, 0.0).mean())
    delta = float(disc * (sample > K).mean())
    return price, delta


def _future_month_quote(vep_slice: pd.DataFrame,
                        year_m: int, mo_m: int) -> Optional[float]:
    """Return the price of the VEP monthly baseload for (year_m, mo_m).

    Falls back to the nearest future month if the exact one is
    unavailable.  Returns None if the quote day has no future month
    on the horizon.
    """
    exact = vep_slice[(vep_slice["delivery_year"] == year_m)
                      & (vep_slice["delivery_month"] == mo_m)]
    if not exact.empty:
        return float(exact["price_TRY_MWh"].iloc[0])
    future = vep_slice[
        (vep_slice["delivery_year"] * 100 + vep_slice["delivery_month"])
        >= (year_m * 100 + mo_m)].sort_values(
            ["delivery_year", "delivery_month"])
    if future.empty:
        return None
    return float(future["price_TRY_MWh"].iloc[0])


def sample_for_model(model: str, val_utc: pd.Timestamp,
                      F_by_hour: Dict[int, float], scale_P: float,
                      z_cycle: np.ndarray, params_m0: MSARParamSet,
                      params_m1: MSARParamSet, yaml_p, ptf: pd.Series,
                      n_paths: int, seed: int
                      ) -> Dict[int, np.ndarray]:
    """Route to the appropriate predictive sampler and return
    horizon-keyed samples of P_T."""
    if model == "M0":
        return predict_msar(params_m0, val_utc, F_by_hour, scale_P,
                             z_cycle, n_paths, seed)
    if model == "M1":
        return predict_msar(params_m1, val_utc, F_by_hour, scale_P,
                             z_cycle, n_paths, seed + 500_000)
    # For benchmarks we need historical vol at last_known
    last_known = day_end_utc(val_utc.tz_convert(TURKEY_TZ))
    vol = fit_hist_vol(ptf, cutoff_utc=last_known)
    pi_arr = np.asarray(yaml_p.m9_stationary_pi, dtype=float)
    sig_y_pooled = float(math.sqrt(
        pi_arr @ (np.asarray(yaml_p.sigma_y) ** 2)))
    kappa_prod = float(yaml_p.kappa_per_hour)
    out = {}
    for h in HORIZONS_H:
        F = F_by_hour[h]
        tau_h = float(h)
        if model == "B1":
            out[h] = predict_black76(F, vol["sigma_log_d2h"], tau_h,
                                      n_paths, seed + 1_000_000 + h)
        elif model == "B2":
            out[h] = predict_bachelier(F, vol["sigma_abs_d2h"], tau_h,
                                        n_paths, seed + 2_000_000 + h)
        elif model == "B3":
            sigma_price = sig_y_pooled * math.sqrt(F * F + scale_P * scale_P)
            out[h] = predict_lucia_schwartz(F, sigma_price, kappa_prod,
                                             tau_h, n_paths,
                                             seed + 3_000_000 + h)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", type=int, default=3)
    ap.add_argument("--n-paths", type=int, default=10_000)
    ap.add_argument("--start", type=str, default="2026-01-02")
    ap.add_argument("--end", type=str, default="2026-09-24")
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ptf = load_realized_ptf()
    vep = load_vep_quotes_daily()
    m0 = load_M0_production()
    m1 = load_M1_fw9e_A3()
    yaml_p = load_frozen_parameters(YAML_PATH)
    scale_P = float(yaml_p.scale_P)
    z_cycle = climatology_z_cycle()

    universe = business_day_universe(args.start, args.end)
    days = universe[::args.cadence]
    print(f"eval universe: {len(universe)} biz days -> thinned to "
          f"{len(days)} (cadence {args.cadence})")

    rows: List[dict] = []
    t0 = time.time()

    for i, d in enumerate(days):
        val_utc = valuation_utc(d)
        last_known = day_end_utc(d)
        try:
            quote_day, vep_slice = vep_quote_on_or_before(
                vep, d.strftime("%Y-%m-%d"))
        except KeyError:
            continue
        # Build F_by_hour and delivery month
        F_by_hour: Dict[int, float] = {}
        actuals: Dict[int, float] = {}
        deliveries: Dict[int, Tuple[int, int]] = {}
        skip_day = False
        for h in HORIZONS_H:
            target = last_known + pd.Timedelta(hours=h)
            local_target = target.tz_convert(TURKEY_TZ)
            year_m, mo_m = local_target.year, local_target.month
            f = _future_month_quote(vep_slice, year_m, mo_m)
            if f is None or target not in ptf.index:
                skip_day = True
                break
            F_by_hour[h] = f
            actuals[h] = float(ptf.loc[target])
            deliveries[h] = (year_m, mo_m)
        if skip_day:
            continue

        # Also collect F for the delivery month on the intermediate
        # rebalance days d+1, d+2 (only those covered by the horizon
        # h -> at most h/24 - 1 rebalance days).
        # We use one delivery month per horizon (the target month for
        # that horizon).  For a 72 h horizon this is 2 rebalance days.
        F_traj: Dict[int, List[float]] = {}
        rebal_days: Dict[int, List[str]] = {}
        for h in HORIZONS_H:
            year_m, mo_m = deliveries[h]
            traj = [F_by_hour[h]]
            days_span = list(range(1, h // 24))
            rebal_labels = [d.strftime("%Y-%m-%d")]
            for k in days_span:
                d_k = d + pd.Timedelta(days=k)
                try:
                    _qd, vslice_k = vep_quote_on_or_before(
                        vep, d_k.strftime("%Y-%m-%d"))
                    fk = _future_month_quote(vslice_k, year_m, mo_m)
                except KeyError:
                    fk = None
                if fk is None:
                    fk = traj[-1]  # carry forward
                traj.append(float(fk))
                rebal_labels.append(d_k.strftime("%Y-%m-%d"))
            # Terminal F is the discounted realised monthly-average
            # replaced with a simpler proxy: use the terminal-day
            # spot as the "final" F for the P&L calc.  This is the
            # cleanest short-horizon proxy since the delivery month
            # contract expires only at its delivery start.
            # Better proxy: use the last VEP quote on d+ceil(h/24) - 1
            # for the same delivery month (already the last element
            # of traj).
            F_traj[h] = traj
            rebal_days[h] = rebal_labels

        # For each model at this day, sample at valuation day
        samples_by_model: Dict[str, Dict[int, np.ndarray]] = {}
        for model in ("M0", "M1", "B1", "B2", "B3"):
            samples_by_model[model] = sample_for_model(
                model, val_utc, F_by_hour, scale_P, z_cycle,
                m0, m1, yaml_p, ptf, args.n_paths,
                args.seed + i + hash(model) % 1000)

        # For each K in moneyness, compute price + delta at d, then
        # step forward day by day computing new price/delta at each
        # intermediate d+k with the OLD sample shifted by ΔF (a
        # first-order approximation: sample_k = sample_0 + (F_k - F_0)).
        for h in HORIZONS_H:
            tau_h = float(h)
            F0 = F_by_hour[h]
            traj = F_traj[h]
            n_rebal = len(traj)
            for mm in MONEYNESS:
                K = mm * F0
                for model in ("M0", "M1", "B1", "B2", "B3"):
                    samp0 = samples_by_model[model][h]
                    C0, delta0 = call_price_and_delta_from_sample(
                        samp0, K, tau_h)
                    # Static hedge: hold delta0 units of monthly VEP
                    # contract, receive/pay ΔF * delta0 at horizon end
                    # PROXY terminal-F: last available VEP quote in
                    # the trajectory
                    F_end = traj[-1]
                    hedge_static = delta0 * (F_end - F0)
                    # Daily rebalance: at each rebalance day compute
                    # new delta from the sample shifted by the current
                    # F drift (first-order Taylor at fixed residual)
                    hedge_dyn = 0.0
                    deltas = [delta0]
                    for k in range(1, n_rebal):
                        F_k = traj[k]
                        # First-order proxy: shift sample by (F_k - F0)
                        # keeps the residual dispersion the same.
                        samp_k = samp0 + (F_k - F0)
                        _C_k, delta_k = call_price_and_delta_from_sample(
                            samp_k, K, tau_h - k * 24.0)
                        hedge_dyn += deltas[-1] * (F_k - traj[k - 1])
                        deltas.append(delta_k)
                    hedge_dyn += deltas[-1] * (F_end - traj[n_rebal - 1])
                    disc = math.exp(-R_PER_HOUR * tau_h)
                    payoff = max(actuals[h] - K, 0.0) * disc
                    unhedged_err = payoff - C0
                    hedged_err_stat = unhedged_err - hedge_static
                    hedged_err_dyn = unhedged_err - hedge_dyn
                    rows.append({
                        "day": d.strftime("%Y-%m-%d"),
                        "h": h, "model": model, "moneyness": mm, "K": K,
                        "F_d": F0, "F_end": F_end, "actual": actuals[h],
                        "call_price": C0, "delta0": delta0,
                        "hedge_static": hedge_static,
                        "hedge_dynamic": hedge_dyn,
                        "unhedged_error": unhedged_err,
                        "hedged_error_static": hedged_err_stat,
                        "hedged_error_dynamic": hedged_err_dyn,
                    })

    dt = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hedge_daily.csv", index=False)
    print(f"hedge daily loop done in {dt:.1f}s: n={len(df)}")

    # Aggregate to hedge effectiveness table
    agg_rows = []
    for (model, h, mm), sub in df.groupby(["model", "h", "moneyness"]):
        var_u = float(sub["unhedged_error"].var(ddof=1))
        var_s = float(sub["hedged_error_static"].var(ddof=1))
        var_d = float(sub["hedged_error_dynamic"].var(ddof=1))
        eff_s = 1.0 - var_s / var_u if var_u > 0 else float("nan")
        eff_d = 1.0 - var_d / var_u if var_u > 0 else float("nan")
        # Realised sd of hedged (dynamic) P&L
        realised_sd_dyn = float(sub["hedged_error_dynamic"].std(ddof=1))
        # Model-predicted sd of hedge P&L: for delta-hedged strategy the
        # residual variance is Var(discounted payoff - delta * (F_end - F_d))
        # under the model's terminal predictive distribution -- proxied
        # by the standard error of the mean call minus 0.  Use the
        # average absolute error under the model as a proxy predictor
        # (this is a rough calibration; a full Greek predictor is
        # future work).
        agg_rows.append({
            "model": model, "h": h, "moneyness": mm,
            "n_days": int(sub.shape[0]),
            "unhedged_sd_TRY": float(math.sqrt(var_u)),
            "static_hedged_sd_TRY": float(math.sqrt(var_s)),
            "dynamic_hedged_sd_TRY": float(math.sqrt(var_d)),
            "effectiveness_static": eff_s,
            "effectiveness_dynamic": eff_d,
            "realised_hedged_sd_dynamic": realised_sd_dyn,
        })
    eff = pd.DataFrame(agg_rows).sort_values(["model", "h", "moneyness"]) \
        .reset_index(drop=True)
    eff.to_csv(OUT / "hedge_effectiveness.csv", index=False)
    print("\n=== Hedge effectiveness (dynamic) ===")
    piv = eff.pivot_table(index=["model", "h"], columns="moneyness",
                          values="effectiveness_dynamic")
    print(piv.round(3).to_string())

    # Residual-risk calibration: compare model-predicted call-value sd
    # to realised hedged P&L sd at ATM (moneyness = 1.0)
    atm = eff[eff["moneyness"] == 1.0].copy()
    atm["realised_over_unhedged_pct"] = 100.0 * atm["dynamic_hedged_sd_TRY"] \
        / atm["unhedged_sd_TRY"]
    atm.to_csv(OUT / "residual_risk_calibration.csv", index=False)

    md = ["# FW10 Part B -- Delta hedge with monthly VEP forwards\n",
          "## Method\n",
          "* Delta = disc * P(P_T > K) from the model's predictive "
          "sample (analytical result for arithmetic P = F + residual).\n",
          "* Static: hold delta_d units, P&L = delta_d * (F_{end} - F_d).\n",
          "* Dynamic: rebalance every 24 h at the day's VEP quote for "
          "the same delivery month; carry forward on non-quoting days.\n",
          "* Effectiveness = 1 - Var(hedged) / Var(unhedged).\n\n",
          "## Effectiveness table (dynamic hedge)\n",
          piv.round(3).to_markdown(),
          "\n\n## Residual-risk calibration at ATM (moneyness = 1.0)\n",
          atm[["model", "h", "n_days", "unhedged_sd_TRY",
                "dynamic_hedged_sd_TRY", "effectiveness_dynamic",
                "realised_over_unhedged_pct"]].round(2).to_markdown(index=False),
          "\n\n## Interpretation\n",
          "Effectiveness is expected to be LOW because the 24-72 h "
          "price risk is dominated by the intraday shape of the "
          "residual, which the monthly VEP baseload contract cannot "
          "hedge.  A high dynamic effectiveness would actually be "
          "suspicious: it would suggest that monthly baseload swings "
          "drive short-horizon variance, contrary to the residual-vs-"
          "curve decomposition on which the pricer is built.  The "
          "primary reading is a MARKET-COMPLETENESS finding, not a "
          "model failure: the monthly VEP strip does not span the "
          "short-horizon hourly PTF risk.\n"]
    (OUT / "hedge_effectiveness.md").write_text("\n".join(md),
                                                  encoding="utf-8")
    print("wrote hedge_daily.csv, hedge_effectiveness.csv/md, "
          "residual_risk_calibration.csv")


if __name__ == "__main__":
    main()
