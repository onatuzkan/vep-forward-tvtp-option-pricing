"""FW3 orchestrator: benchmark model comparison on the F2.8 grid.

Prices three closed-form benchmarks -- Black-76 (B1), Bachelier (B2), and
Lucia & Schwartz single-factor OU (B3) -- on the SAME European call/put
contracts as the accepted PDE run, using the SAME forward level F(T) and
the SAME discount factor.  Every benchmark input is sourced from real
repo data (no synthetic values):

* option/model prices, F(T) and put values come from
  ``outputs/market_calibration_final/strike_maturity_grid.csv``;
* discount rate ``r = 0.40`` (annualised) matches the calibration config;
* historical volatilities are computed on ``inputs/historical/ptf_raw/``
  (real EPIAS PTF, TRY/MWh) with no look-ahead past 2025-12-31 20:00 UTC;
* ``sigma_y`` and ``kappa`` for the OU baseline are the accepted values
  from ``inputs/historical/m2_frozen_parameters.yaml`` (pooled by the M9
  stationary occupancy for the single-regime collapse; NOT re-fitted);
* the realized backtest uses ``inputs/market/realized_ptf_2026.csv`` for
  the maturity-hour spot -- a single-path realization, so results are
  reported with explicit ``n_hours`` and ``std_error`` and interpreted
  cautiously.

Outputs land in ``outputs/fw3_benchmarks/`` (CSV + Markdown).  Accepted
project artefacts (m2/tvtp2 yaml, outputs/market_calibration_final,
outputs/forward_centered_diagnostics, outputs/scenario_sweep,
outputs/tvtp2_experimental) are not touched.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.benchmarks import (      # noqa: E402
    bachelier, black76, implied_vol_bachelier, implied_vol_black76,
    lucia_schwartz, ou_terminal_variance, parity_error, price_bachelier_pair,
    price_black76_pair, price_lucia_schwartz_pair,
)
from pde_option_model.params_frozen import load_frozen_parameters   # noqa: E402

GRID_CSV = REPO_ROOT / "outputs" / "market_calibration_final" / "strike_maturity_grid.csv"
HOURLY_FWD_CSV = REPO_ROOT / "outputs" / "market_calibration_final" / "hourly_forward_curve.csv"
PARAMS_YAML = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
PTF_HISTORY_DIR = REPO_ROOT / "inputs" / "historical" / "ptf_raw"
REALIZED_CSV = REPO_ROOT / "inputs" / "market" / "realized_ptf_2026.csv"
OUT_DIR = REPO_ROOT / "outputs" / "fw3_benchmarks"

# Discount rate assumed by the accepted grid (see strike_maturity_grid.md).
R_ANNUAL = 0.40
HOURS_PER_YEAR = 8760.0
R_PER_HOUR = R_ANNUAL / HOURS_PER_YEAR
# The valuation instant matches the yaml/CSV header.
VALUATION_UTC = pd.Timestamp("2025-12-31 20:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Historical volatility (real EPIAS PTF, no look-ahead)
# ---------------------------------------------------------------------------
def load_ptf_history() -> pd.DataFrame:
    """Load hourly PTF 2019-2025 in Turkey local time.

    The raw CSV carries ``dd.mm.YYYY`` in ``Tarih`` and ``HH:MM`` in
    ``Saat`` with Turkish-locale thousands / decimals, which pandas
    handles with ``sep=';', decimal=',', thousands='.'`` and an explicit
    string dtype for the date column so that leading zeros are not lost.
    """
    frames = []
    for year in range(2019, 2026):
        f = PTF_HISTORY_DIR / f"ptf_{year}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep=";", decimal=",", thousands=".",
                         dtype={"Tarih": str})
        ts = pd.to_datetime(df["Tarih"].str.zfill(8) + " " + df["Saat"],
                            format="%d%m%Y %H:%M", errors="coerce")
        alt = ts.isna()
        if alt.any():
            ts = ts.where(~alt, pd.to_datetime(
                df.loc[alt, "Tarih"] + " " + df.loc[alt, "Saat"],
                format="%d.%m.%Y %H:%M", errors="coerce"))
        frames.append(pd.DataFrame({
            "ts_local": ts, "ptf_TRY_MWh": df["PTF (TL/MWh)"].astype(float),
        }))
    d = pd.concat(frames, ignore_index=True)
    d["ts_utc"] = (d["ts_local"].dt.tz_localize("Europe/Istanbul",
                                                ambiguous="infer",
                                                nonexistent="shift_forward")
                   .dt.tz_convert("UTC"))
    d = d.sort_values("ts_utc").drop_duplicates("ts_utc").reset_index(drop=True)
    return d


def historical_vol(hist: pd.DataFrame, cutoff_utc: pd.Timestamp,
                   window_hours: int, price_floor: float = 50.0,
                   ) -> Dict[str, float]:
    """Historical volatilities up to ``cutoff_utc`` (no look-ahead).

    Two sampling frequencies are reported side by side:

    * **Hourly returns.**  ``sigma_log_hourly`` is std of
      ``log(P_t / P_{t-1})`` over the window (with ``price_floor``
      guarding against occasional near-zero clearings).  On the Turkish
      day-ahead market the hourly log-return is DOMINATED by the
      intraday cycle rather than by shock noise (a peak-hour vs pre-
      dawn move is a signal, not a random innovation), so hourly vol
      is very large and NOT the natural input for a Black-76 baseline
      on a multi-day option -- reported here only for transparency.

    * **Daily returns.**  ``sigma_log_daily`` uses the daily-average
      spot ``P_d = mean(P_h in day d)``, log-differenced across
      consecutive Turkey-local calendar days.  This is the natural
      no-look-ahead volatility for a 1-30 day option benchmark: the
      intraday seasonality is stripped by averaging.  Converted to a
      per-sqrt(hour) sigma via ``sigma_log_daily / sqrt(24)`` so the
      benchmark accepts it directly.

    Bachelier gets the analogous arithmetic sigma (in TRY/MWh) from the
    same daily series.
    """
    s = hist[hist["ts_utc"] <= cutoff_utc].tail(window_hours + 1).copy()
    p = s["ptf_TRY_MWh"].to_numpy(dtype=float)
    mask = p[:-1] >= price_floor
    log_ret_h = np.log(np.where(p[1:] >= price_floor, p[1:], np.nan)
                       / np.where(mask, p[:-1], np.nan))
    abs_ret_h = np.diff(p)
    log_ret_h = log_ret_h[np.isfinite(log_ret_h)]

    # daily-average spot in Turkey local time
    s2 = s.copy()
    s2["date_local"] = (s2["ts_utc"].dt.tz_convert("Europe/Istanbul")
                        .dt.date)
    daily = (s2.groupby("date_local", as_index=False)
             .agg(ptf_daily=("ptf_TRY_MWh", "mean"),
                  n_hours=("ptf_TRY_MWh", "size"))
             .query("n_hours == 24")
             .sort_values("date_local"))
    p_d = daily["ptf_daily"].to_numpy(dtype=float)
    mask_d = p_d[:-1] >= price_floor
    log_ret_d = np.log(np.where(p_d[1:] >= price_floor, p_d[1:], np.nan)
                       / np.where(mask_d, p_d[:-1], np.nan))
    log_ret_d = log_ret_d[np.isfinite(log_ret_d)]
    abs_ret_d = np.diff(p_d)

    sigma_log_daily = float(np.std(log_ret_d, ddof=1))
    sigma_abs_daily = float(np.std(abs_ret_d, ddof=1))
    return {
        "cutoff_utc": str(cutoff_utc),
        "window_hours": int(window_hours),
        "price_floor_TRY_MWh": float(price_floor),
        "n_price_floor_rejected_hourly": int((~mask).sum()),
        "n_log_return_hourly": int(log_ret_h.size),
        "n_log_return_daily": int(log_ret_d.size),
        "sigma_log_hourly_per_sqrt_h": float(np.std(log_ret_h, ddof=1)),
        "sigma_abs_hourly_per_sqrt_h_TRY_MWh": float(np.std(abs_ret_h, ddof=1)),
        "sigma_log_daily_per_sqrt_d": sigma_log_daily,
        "sigma_abs_daily_per_sqrt_d_TRY_MWh": sigma_abs_daily,
        # per-sqrt-hour view derived from daily (divide by sqrt(24));
        # this is the PRIMARY input for the closed-form benchmarks
        "sigma_log_daily_to_per_sqrt_h": sigma_log_daily / math.sqrt(24.0),
        "sigma_abs_daily_to_per_sqrt_h_TRY_MWh": sigma_abs_daily / math.sqrt(24.0),
        "mean_price_TRY_MWh": float(np.nanmean(p)),
        "median_price_TRY_MWh": float(np.nanmedian(p)),
    }


# ---------------------------------------------------------------------------
# Lucia-Schwartz pooled single-regime sigma from the accepted yaml
# ---------------------------------------------------------------------------
def ls_sigma_price(F: float, params) -> Tuple[float, float]:
    """(pooled sigma_y, sigma_price at F) for the single-regime OU baseline.

    Sigma-y is pooled by the M9 stationary occupancy (index 0 = normal,
    index 1 = stress) so the single-regime variance rate equals the
    stress-weighted mixture variance rate the two-regime model reaches
    in the long run.  This matches the way the existing pooled baseline
    (model_comparison_pooled_vs_M9.md) collapses two regimes into one.
    The sigma is then mapped from y-space to price space at ``F`` via
    the delta-method transfer ``sigma_price = sigma_y * sqrt(F^2 +
    scale_P^2)`` used throughout ``ResidualSpec.sigma_price``.
    """
    pi = np.asarray(params.m9_stationary_pi, dtype=float)
    sig_y = np.asarray(params.sigma_y, dtype=float)
    pooled_sigma_y = float(np.sqrt(pi @ (sig_y ** 2)))
    scale_P = float(params.scale_P)
    sigma_price = pooled_sigma_y * math.sqrt(F * F + scale_P * scale_P)
    return pooled_sigma_y, sigma_price


# ---------------------------------------------------------------------------
# Grid comparison
# ---------------------------------------------------------------------------
def _atm_history_sigma(F_ref: float, sigma_log_per_h: float,
                       sigma_abs_per_h: float) -> Tuple[float, float]:
    """Return (sigma_B76_per_sqrt_h, sigma_Bach_per_sqrt_h) for one F ref.

    Black-76 takes a log-return sigma directly.  For Bachelier the
    historical arithmetic-return sigma is used as-is (TRY/MWh per
    sqrt(hour)); it already reflects the price level of the sample
    window, so no re-scaling by ``F`` is applied.
    """
    return float(sigma_log_per_h), float(sigma_abs_per_h)


def build_grid_comparison(hist_vol: Dict[str, float], params) -> pd.DataFrame:
    """Row-wise benchmark comparison against the accepted 66-point grid.

    For every (K, T) row we produce three columns per benchmark: the
    (i) historical-vol price, (ii) model-implied Black-76 vol from the
    CALL price, (iii) model-implied Bachelier vol from the CALL price.
    Lucia-Schwartz has no implied-vol column of its own -- it is a
    price-only benchmark with the (sigma, kappa) fixed from the yaml.
    """
    df = pd.read_csv(GRID_CSV)
    kappa = float(params.kappa_per_hour)
    pooled_sigma_y, _ = ls_sigma_price(float(df["F_T_TRY_MWh"].iloc[0]), params)

    sigma_log_h = float(hist_vol["sigma_log_hourly_per_sqrt_h"])
    sigma_abs_h = float(hist_vol["sigma_abs_hourly_per_sqrt_h_TRY_MWh"])
    sigma_log_d2h = float(hist_vol["sigma_log_daily_to_per_sqrt_h"])
    sigma_abs_d2h = float(hist_vol["sigma_abs_daily_to_per_sqrt_h_TRY_MWh"])

    rows: List[Dict[str, float]] = []
    for _, r in df.iterrows():
        F = float(r["F_T_TRY_MWh"])
        K = float(r["strike_TRY_MWh"])
        T = float(r["maturity_h"])
        model_call = float(r["call_TRY_MWh"])
        model_put = float(r["put_TRY_MWh"])

        # (i) benchmarks with historical volatility.  PRIMARY = daily
        # returns rescaled to per-sqrt(hour) (intraday cycle stripped);
        # HOURLY reported alongside as a naive-baseline reference.
        b76c, b76p = price_black76_pair(F, K, T, sigma_log_d2h, R_PER_HOUR)
        bacc, bacp = price_bachelier_pair(F, K, T, sigma_abs_d2h, R_PER_HOUR)
        b76c_h, _ = price_black76_pair(F, K, T, sigma_log_h, R_PER_HOUR)
        bacc_h, _ = price_bachelier_pair(F, K, T, sigma_abs_h, R_PER_HOUR)

        # Lucia-Schwartz: pooled sigma_y in y-space -> sigma_price at F
        _, sigma_price = ls_sigma_price(F, params)
        lsc, lsp = price_lucia_schwartz_pair(F, K, T, sigma_price, kappa,
                                             R_PER_HOUR)

        # (ii) implied vols from the MODEL call (never from the benchmarks)
        try:
            iv_b76 = implied_vol_black76(model_call, F, K, T, R_PER_HOUR, "call")
        except ValueError:
            iv_b76 = float("nan")
        try:
            iv_bac = implied_vol_bachelier(model_call, F, K, T, R_PER_HOUR, "call")
        except ValueError:
            iv_bac = float("nan")

        rows.append({
            "strike": K, "maturity_h": T, "F_T": F,
            "model_call": model_call, "model_put": model_put,
            # historical-vol benchmark prices (PRIMARY = daily-derived)
            "B1_black76_call_hist": b76c, "B1_black76_put_hist": b76p,
            "B2_bachelier_call_hist": bacc, "B2_bachelier_put_hist": bacp,
            "B3_lucia_schwartz_call": lsc, "B3_lucia_schwartz_put": lsp,
            # HOURLY-derived variants (reported for transparency only;
            # inflated by the Turkish market's intraday cycle)
            "B1_black76_call_hourly": b76c_h,
            "B2_bachelier_call_hourly": bacc_h,
            # % differences (benchmark vs model, on the call)
            "B1_vs_model_call_pct": _pct_diff(b76c, model_call),
            "B2_vs_model_call_pct": _pct_diff(bacc, model_call),
            "B3_vs_model_call_pct": _pct_diff(lsc, model_call),
            # implied vols extracted from the MODEL call
            "iv_black76_from_model_per_h": iv_b76,
            "iv_bachelier_from_model_per_h": iv_bac,
            # annualised versions for readability
            "iv_black76_annual": iv_b76 * math.sqrt(HOURS_PER_YEAR),
            "iv_bachelier_annual_TRY_MWh": iv_bac * math.sqrt(HOURS_PER_YEAR),
            # put-call parity on every benchmark
            "parity_err_B1": parity_error(b76c, b76p, F, K, T, R_PER_HOUR),
            "parity_err_B2": parity_error(bacc, bacp, F, K, T, R_PER_HOUR),
            "parity_err_B3": parity_error(lsc, lsp, F, K, T, R_PER_HOUR),
        })

    return pd.DataFrame(rows)


def _pct_diff(bench: float, model: float) -> float:
    return float("nan") if model == 0 else 100.0 * (bench - model) / model


# ---------------------------------------------------------------------------
# Realized 2026 backtest on the F2.8 contracts
# ---------------------------------------------------------------------------
def load_realized() -> pd.DataFrame:
    """Realized hourly PTF (TRY/MWh) in UTC, from the tracked EPIAS CSV."""
    df = pd.read_csv(REALIZED_CSV, sep=";", decimal=",", thousands=".",
                     dtype={"Tarih": str})
    ts_local = pd.to_datetime(df["Tarih"] + " " + df["Saat"],
                              format="%d.%m.%Y %H:%M", errors="coerce")
    ts_utc = (ts_local.dt.tz_localize("Europe/Istanbul", ambiguous="infer",
                                      nonexistent="shift_forward")
              .dt.tz_convert("UTC"))
    return pd.DataFrame({"ts_utc": ts_utc,
                         "ptf_TRY_MWh": df["PTF (TL/MWh)"].astype(float)})


def realized_backtest(grid: pd.DataFrame, hist_vol: Dict[str, float],
                      params) -> pd.DataFrame:
    """Per-model realized discounted payoff error on the F2.8 contracts.

    One realized payoff per contract -- so per-contract error is a
    single-path draw and only aggregation across contracts is meaningful.
    The header of the resulting Markdown report calls this out.
    """
    realized = load_realized()
    kappa = float(params.kappa_per_hour)
    sigma_log = float(hist_vol["sigma_log_daily_to_per_sqrt_h"])
    sigma_abs = float(hist_vol["sigma_abs_daily_to_per_sqrt_h_TRY_MWh"])

    rows: List[Dict[str, float]] = []
    for _, r in grid.iterrows():
        F = float(r["F_T"]); K = float(r["strike"]); T = float(r["maturity_h"])
        mat_utc = VALUATION_UTC + pd.Timedelta(hours=T)
        sel = realized[realized["ts_utc"] == mat_utc]
        if sel.empty:
            continue
        p_real = float(sel["ptf_TRY_MWh"].iloc[0])
        disc = math.exp(-R_PER_HOUR * T)
        realized_call = disc * max(p_real - K, 0.0)
        realized_put = disc * max(K - p_real, 0.0)

        _, sigma_price = ls_sigma_price(F, params)
        rows.append({
            "strike": K, "maturity_h": T, "F_T": F, "realized_P_T": p_real,
            "realized_call": realized_call, "realized_put": realized_put,
            "model_call": float(r["model_call"]),
            "B1_black76_call": black76(F, K, T, sigma_log, R_PER_HOUR, "call"),
            "B2_bachelier_call": bachelier(F, K, T, sigma_abs, R_PER_HOUR, "call"),
            "B3_lucia_schwartz_call": lucia_schwartz(F, K, T, sigma_price, kappa,
                                                     R_PER_HOUR, "call"),
        })
    df = pd.DataFrame(rows)
    for col in ("model_call", "B1_black76_call", "B2_bachelier_call",
                "B3_lucia_schwartz_call"):
        df[f"{col}_minus_realized"] = df[col] - df["realized_call"]
    return df


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """Aggregate model/benchmark error against realized discounted payoff."""
    stats: List[Dict[str, float]] = []
    for label, col in [("Model_PDE", "model_call"),
                       ("B1_Black76", "B1_black76_call"),
                       ("B2_Bachelier", "B2_bachelier_call"),
                       ("B3_Lucia_Schwartz", "B3_lucia_schwartz_call")]:
        err = bt[f"{col}_minus_realized"].to_numpy(dtype=float)
        stats.append({
            "model": label, "n_contracts": int(err.size),
            "mean_bias": float(err.mean()),
            "std_error_of_bias": float(err.std(ddof=1) / math.sqrt(err.size)),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
        })
    return pd.DataFrame(stats)


# ---------------------------------------------------------------------------
# Implied vol surface / ATM term structure
# ---------------------------------------------------------------------------
def implied_vol_surface(grid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wide-format smile (per maturity) + ATM term structure.

    'ATM' is defined per-maturity as the strike whose absolute
    log-moneyness ``|ln(K / F(T))|`` is smallest across the 11
    tabulated strikes.  Since F(T) is 2917 -> 2901 TRY/MWh across the
    six maturities and the grid contains K = 3000 at every maturity,
    ATM lands on K = 3000 for every row (moneyness ~ 0.03), which
    lets the ATM curve be read off a single strike.
    """
    piv = grid.pivot_table(index="strike", columns="maturity_h",
                           values="iv_black76_annual")
    atm_rows = []
    for T, sub in grid.groupby("maturity_h"):
        F_T = float(sub["F_T"].iloc[0])
        moneyness = np.abs(np.log(sub["strike"].to_numpy(dtype=float) / F_T))
        i_atm = int(np.argmin(moneyness))
        atm_rows.append({
            "maturity_h": float(T), "F_T": F_T,
            "K_ATM": float(sub["strike"].iloc[i_atm]),
            "iv_black76_annual_ATM": float(sub["iv_black76_annual"].iloc[i_atm]),
            "iv_bachelier_annual_ATM_TRY_MWh": float(
                sub["iv_bachelier_annual_TRY_MWh"].iloc[i_atm]),
        })
    return piv, pd.DataFrame(atm_rows).sort_values("maturity_h").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_markdown(hist_vol: Dict[str, float], grid_cmp: pd.DataFrame,
                   iv_smile: pd.DataFrame, atm: pd.DataFrame,
                   backtest: pd.DataFrame, summary: pd.DataFrame,
                   params, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "grid_comparison.csv").write_text(
        grid_cmp.to_csv(index=False), encoding="utf-8")
    (out_dir / "implied_vol_surface_black76.csv").write_text(
        iv_smile.to_csv(), encoding="utf-8")
    (out_dir / "implied_vol_atm_term_structure.csv").write_text(
        atm.to_csv(index=False), encoding="utf-8")
    (out_dir / "realized_backtest.csv").write_text(
        backtest.to_csv(index=False), encoding="utf-8")
    (out_dir / "realized_backtest_summary.csv").write_text(
        summary.to_csv(index=False), encoding="utf-8")
    (out_dir / "historical_vol.json").write_text(
        json.dumps(hist_vol, indent=2), encoding="utf-8")

    pooled_sigma_y, _ = ls_sigma_price(float(grid_cmp["F_T"].iloc[0]), params)

    md = []
    md.append("# FW3 benchmark model comparison\n")
    md.append("Closed-form benchmark option prices (Black-76, Bachelier, "
              "Lucia-Schwartz 2002) evaluated on the SAME contracts, F(T) and "
              "discount factor as the accepted PDE model.  See "
              "`docs/fw3_benchmark_methodology.md` for the full methodology.\n")
    md.append("## Setup\n")
    md.append(f"* Valuation: {VALUATION_UTC.isoformat()}")
    md.append(f"* Discount: `r_annual = {R_ANNUAL:.2f}`  (`r_per_hour = "
              f"{R_PER_HOUR:.6e}`)")
    md.append(f"* Contracts: {len(grid_cmp)} rows from "
              "`outputs/market_calibration_final/strike_maturity_grid.csv`")
    md.append("* Real repo data only: PTF history from "
              "`inputs/historical/ptf_raw/`; realized PTF from "
              "`inputs/market/realized_ptf_2026.csv`.  No synthetic values.\n")

    md.append("## Historical volatility inputs (no look-ahead)\n")
    md.append(f"* Window: last {hist_vol['window_hours']} hours "
              f"(~365 days) ending {hist_vol['cutoff_utc']}; source "
              "`inputs/historical/ptf_raw/ptf_2019..2025.csv`")
    md.append(f"* Price floor: {hist_vol['price_floor_TRY_MWh']:.0f} "
              "TRY/MWh (guards log-returns against near-zero clearings)\n")
    md.append("**Two sampling frequencies computed side by side:**")
    md.append(f"* HOURLY log-return sample: {hist_vol['n_log_return_hourly']} "
              "obs -> `sigma_log_hourly = "
              f"{hist_vol['sigma_log_hourly_per_sqrt_h']:.4f}/sqrt(h)` "
              f"(annualised ~"
              f"{hist_vol['sigma_log_hourly_per_sqrt_h'] * math.sqrt(HOURS_PER_YEAR):.2f}). "
              "Dominated by the intraday demand cycle; NOT the primary "
              "input for a multi-day option.")
    md.append(f"* DAILY (calendar-day mean) log-return sample: "
              f"{hist_vol['n_log_return_daily']} obs -> "
              f"`sigma_log_daily = "
              f"{hist_vol['sigma_log_daily_per_sqrt_d']:.4f}/sqrt(d)`, "
              "rescaled to per-sqrt(hour) via "
              f"`/sqrt(24)` = "
              f"{hist_vol['sigma_log_daily_to_per_sqrt_h']:.4f}. "
              "Annualised = "
              f"{hist_vol['sigma_log_daily_to_per_sqrt_h'] * math.sqrt(HOURS_PER_YEAR):.2f}. "
              "PRIMARY input for B1 (Black-76).")
    md.append(f"* DAILY abs-return -> `sigma_abs_daily = "
              f"{hist_vol['sigma_abs_daily_per_sqrt_d_TRY_MWh']:.2f} "
              "TRY/MWh/sqrt(d)`, per-sqrt(hour) = "
              f"{hist_vol['sigma_abs_daily_to_per_sqrt_h_TRY_MWh']:.2f}. "
              "PRIMARY input for B2 (Bachelier).\n")

    md.append("## Lucia-Schwartz single-factor OU inputs (from the yaml, "
              "NOT re-fitted)\n")
    md.append(f"* `kappa_per_hour = {params.kappa_per_hour:.6f}` "
              f"(half-life {params.half_life_hours:.2f} h) -- v2 reconciled "
              "kappa in the accepted yaml")
    md.append(f"* Pooled `sigma_y = {pooled_sigma_y:.6f}` (M9 stationary "
              "occupancy weights "
              f"{tuple(np.round(np.asarray(params.m9_stationary_pi), 4))})")
    md.append("* Mapped to price space at each contract's F(T) via the same "
              "delta-method transfer `sigma_price = sigma_y * sqrt(F^2 + "
              "scale_P^2)` used by `ResidualSpec.sigma_price`\n")

    md.append("## Grid comparison summary (ATM K=3000)\n")
    atm_grid = grid_cmp[grid_cmp["strike"] == 3000.0]
    md.append(atm_grid[["maturity_h", "F_T", "model_call",
                        "B1_black76_call_hist", "B2_bachelier_call_hist",
                        "B3_lucia_schwartz_call",
                        "B1_vs_model_call_pct", "B2_vs_model_call_pct",
                        "B3_vs_model_call_pct"]]
              .round({"F_T": 2, "model_call": 2, "B1_black76_call_hist": 2,
                      "B2_bachelier_call_hist": 2, "B3_lucia_schwartz_call": 2,
                      "B1_vs_model_call_pct": 2, "B2_vs_model_call_pct": 2,
                      "B3_vs_model_call_pct": 2})
              .to_markdown(index=False))
    md.append("\n\nPositive `%` = benchmark above model.  Deep-OTM and long-"
              "dated rows can appear extreme in % terms because the model call "
              "value is small in the denominator; see `grid_comparison.csv` "
              "for absolute levels.\n")

    md.append("## ATM implied vol term structure (extracted from the MODEL "
              "call, per maturity)\n")
    md.append(atm.round({"F_T": 2, "K_ATM": 2,
                         "iv_black76_annual_ATM": 4,
                         "iv_bachelier_annual_ATM_TRY_MWh": 2})
              .to_markdown(index=False))
    md.append("\n\nBoth columns are annualised (multiplied by sqrt(8760)) so "
              "readers can compare against literature vols directly.\n")

    md.append("## Implied Black-76 vol smile (annualised, per maturity)\n")
    md.append(iv_smile.round(4).to_markdown())
    md.append("\n\nThe columns are maturities in hours.  If the model were "
              "pure Black-76, every row/column combination would show the "
              "same number; departures encode the skew induced by the "
              "regime-switching mixture (see methodology).\n")

    md.append("## Put-call parity on every benchmark\n")
    max_pe = grid_cmp[["parity_err_B1", "parity_err_B2",
                       "parity_err_B3"]].abs().max().to_dict()
    md.append(f"* max |C - P - e^-rT(F-K)| across all {len(grid_cmp)} rows: "
              f"B1 = {max_pe['parity_err_B1']:.2e}, B2 = "
              f"{max_pe['parity_err_B2']:.2e}, B3 = "
              f"{max_pe['parity_err_B3']:.2e}\n\n")

    md.append("## Realized 2026 discounted-payoff backtest\n")
    md.append(f"Each contract has EXACTLY ONE realized draw of P_T; the "
              f"per-contract error is single-path noise.  Only the "
              f"cross-contract aggregate is interpretable.  {len(backtest)} "
              "contracts matched (contracts whose maturity hour fell outside "
              "the realized CSV window are dropped).\n")
    md.append(summary.round({"mean_bias": 2, "std_error_of_bias": 2,
                             "MAE": 2, "RMSE": 2}).to_markdown(index=False))
    md.append("\n\nInterpretation: with a single realized draw per "
              "contract, `mean_bias` is dominated by the systematic "
              "over-forecast of the VEP forward curve documented in "
              "`realized_2026_backtest.md` (F2.9).  MAE ranking across "
              "the four pricers is more informative than absolute values; "
              "small differences within a single std_error should not be "
              "over-interpreted.\n")

    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")


def write_paper_table(grid_cmp: pd.DataFrame, out_dir: Path) -> None:
    """EK2: a compact table of representative (K, T) points, ready for LaTeX.

    Rows: ATM K=3000 across the six maturities plus one deep-OTM
    (K=4000, T=168h) and one deep-ITM (K=2000, T=168h).  Columns:
    Model / B1 / B2 / B3 call values in TRY/MWh, plus % differences.
    """
    subset = grid_cmp[(grid_cmp["strike"] == 3000.0)
                      | ((grid_cmp["strike"] == 4000.0)
                         & (grid_cmp["maturity_h"] == 168.0))
                      | ((grid_cmp["strike"] == 2000.0)
                         & (grid_cmp["maturity_h"] == 168.0))].copy()
    subset = subset.sort_values(["strike", "maturity_h"]).reset_index(drop=True)
    view = subset[["strike", "maturity_h", "F_T", "model_call",
                   "B1_black76_call_hist", "B2_bachelier_call_hist",
                   "B3_lucia_schwartz_call", "B1_vs_model_call_pct",
                   "B2_vs_model_call_pct", "B3_vs_model_call_pct"]]
    (out_dir / "paper_table.csv").write_text(view.to_csv(index=False),
                                             encoding="utf-8")
    (out_dir / "paper_table.md").write_text(
        "# FW3 paper-ready benchmark table\n\n"
        "Representative (K, T) points for the manuscript.  Columns:\n"
        "Model = accepted PDE call value; B1 = Black-76 (hist sigma); "
        "B2 = Bachelier (hist sigma); B3 = Lucia-Schwartz (yaml sigma, kappa).\n"
        "% diff = 100*(bench-model)/model.\n\n"
        + view.round({"F_T": 2, "model_call": 2, "B1_black76_call_hist": 2,
                      "B2_bachelier_call_hist": 2, "B3_lucia_schwartz_call": 2,
                      "B1_vs_model_call_pct": 2, "B2_vs_model_call_pct": 2,
                      "B3_vs_model_call_pct": 2}).to_markdown(index=False),
        encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = load_frozen_parameters(PARAMS_YAML)
    print("Loading real EPIAS PTF history ...")
    hist = load_ptf_history()
    print(f"  {len(hist):,} hourly rows, "
          f"{hist['ts_utc'].min()} -> {hist['ts_utc'].max()}")

    hist_vol = historical_vol(hist, cutoff_utc=VALUATION_UTC,
                              window_hours=365 * 24)
    print(f"Historical vol (last 365 d): sigma_log_daily = "
          f"{hist_vol['sigma_log_daily_per_sqrt_d']:.4f}/sqrt(d) -> "
          f"{hist_vol['sigma_log_daily_to_per_sqrt_h']:.4f}/sqrt(h), "
          f"sigma_abs_daily = "
          f"{hist_vol['sigma_abs_daily_per_sqrt_d_TRY_MWh']:.2f} TRY/MWh/sqrt(d)")

    print("Building grid comparison (66 rows) ...")
    grid_cmp = build_grid_comparison(hist_vol, params)
    piv, atm = implied_vol_surface(grid_cmp)
    print("Building realized backtest ...")
    backtest = realized_backtest(grid_cmp, hist_vol, params)
    summary = summarize_backtest(backtest)

    print(f"Writing outputs to {OUT_DIR} ...")
    write_markdown(hist_vol, grid_cmp, piv, atm, backtest, summary, params,
                   OUT_DIR)
    write_paper_table(grid_cmp, OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
