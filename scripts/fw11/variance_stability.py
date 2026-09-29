"""FW11 -- Stationary variance stability over time.

Rolling 12-month evaluation of the model-faithful A3 TRY residual
statistics at eight valuation dates (the seven FW4 dates plus
2026-09-27 for the twelve-month out-of-sample window), with two
alternative hour-of-week shape definitions:

  (i)  pooled       -- HOW mean estimated from ALL history observed up
                       to and including the valuation date.
  (ii) 12m-only     -- HOW mean estimated from the trailing 12-month
                       window only.

The monthly-mean removal step of the A3 residual always uses the
observation's OWN calendar-month mean (invariant under either shape
definition), so the pooled-vs-12m comparison isolates the effect of
the hour-of-week seasonality window on the observed residual sd.

Metrics per date, per shape variant:
  * n       -- hours in the trailing 12-month window
  * sd      -- standard deviation of the residual (TRY)
  * kappa   -- OLS AR(1) mean-reversion rate (per hour), fitted on
               the same residual with the shared model AR(1)
               likelihood.  No new MLE fit -- OLS on the residual
               series only.
  * q05/q95 -- 5 and 95 percent quantiles of the residual
  * L       -- 12-month mean of the raw PTF (TRY/MWh)
  * sd_prod -- production stationary residual sd at level L:
               0.1987 * sqrt(L**2 + 282.48**2)
  * ratio   -- observed sd / sd_prod
  * norm_sd -- observed sd / L (normalised residual dispersion)
  * norm_sd_prod = sd_prod / L (approaches 0.1987 for large L)
  * share_lt_50    -- share of hours in the window with PTF < 50 TRY
  * sd_ex_lt_50    -- residual sd excluding those hours
  * ratio_ex_lt_50 -- (sd_ex_lt_50 / sd_prod)

All statistics are computed under the strict-no-look-ahead rule:
only PTF observations with UTC timestamp <= valuation_utc feed any
calculation.  The production reference constants (asinh scale
0.1987, s_P = 282.48) are read from the frozen yaml and are not
re-estimated.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pde_option_model.calendar_tr import TURKEY_TZ
from pde_option_model.params_frozen import load_frozen_parameters

YAML_PATH = REPO / "inputs" / "historical" / "m2_frozen_parameters.yaml"
PTF_ARCHIVE = REPO / "inputs" / "historical" / "ptf_raw"
PTF_2026 = REPO / "inputs" / "market" / "realized_ptf_2025-12-31_2026-09-27.csv"
OUT = REPO / "outputs" / "fw11_variance_stability"

# Production reference: sd_asinh_stat = 0.1987, s_P = 282.48
# (loaded from yaml at run time to keep provenance).
PROD_SD_ASINH = 0.198734  # matches stationary_variance_check.md 0.1987


# ----------------------------------------------------------------------
# PTF loading (historical archive + 2026 realised)
# ----------------------------------------------------------------------
def _read_epias_csv(path: Path, col: str = "PTF (TL/MWh)") -> pd.Series:
    df = pd.read_csv(path, sep=";", decimal=",", thousands=".",
                     dtype={"Tarih": str, "Saat": str})
    if col not in df.columns:
        for c in df.columns:
            if "PTF" in c and "TL" in c:
                col = c; break
    ts = pd.to_datetime(df["Tarih"].str.zfill(8) + " " + df["Saat"],
                        format="%d%m%Y %H:%M", errors="coerce")
    alt = ts.isna()
    if alt.any():
        ts = ts.where(~alt, pd.to_datetime(
            df.loc[alt, "Tarih"] + " " + df.loc[alt, "Saat"],
            format="%d.%m.%Y %H:%M", errors="coerce"))
    m = pd.DataFrame({"ts_local": ts,
                       "ptf_TRY_MWh": df[col].astype(float)}).dropna()
    m["ts_utc"] = (m["ts_local"].dt.tz_localize(
        "Europe/Istanbul", ambiguous="infer",
        nonexistent="shift_forward").dt.tz_convert("UTC"))
    return (m.sort_values("ts_utc").drop_duplicates("ts_utc")
            .set_index("ts_utc")["ptf_TRY_MWh"])


def load_ptf_hourly() -> pd.Series:
    """Concatenated hourly PTF (TL/MWh) on a UTC index, 2019-2019 to
    2026-09-27.  Duplicate-hour drops are handled inside `_read_epias_csv`.
    """
    parts = []
    for year in range(2019, 2026):
        f = PTF_ARCHIVE / f"ptf_{year}.csv"
        if f.exists():
            parts.append(_read_epias_csv(f))
    if PTF_2026.exists():
        parts.append(_read_epias_csv(PTF_2026))
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


# ----------------------------------------------------------------------
# A3 residual computation with configurable HOW shape window
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class A3Result:
    residual: np.ndarray
    sd: float
    q05: float
    q95: float
    n: int
    n_lt_50: int
    sd_excluding_lt_50: float
    L_mean: float


def _hour_of_week(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    """Vector of (dow * 24 + hour) computed in Europe/Istanbul local time."""
    loc = ts_utc.tz_convert(TURKEY_TZ)
    return (loc.dayofweek * 24 + loc.hour).to_numpy(dtype=int)


def _month_ts(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    """Vector of (year * 100 + month) local Istanbul month key."""
    loc = ts_utc.tz_convert(TURKEY_TZ)
    return (loc.year * 100 + loc.month).to_numpy(dtype=int)


def compute_a3(price_window: pd.Series,
                shape_source: pd.Series) -> A3Result:
    """Compute the A3 model-faithful residual on ``price_window`` with the
    hour-of-week climatology estimated on ``shape_source``.

    Step 1 -- subtract the OWN calendar-month mean from every hour.
    Step 2 -- subtract the hour-of-week climatology of
              (price minus its own calendar-month mean) computed on
              ``shape_source`` (which is either the same window or the
              full history up to the valuation date, depending on the
              shape-source choice).

    Both `price_window` and `shape_source` must be tz-aware UTC series
    of hourly TRY/MWh values.  `shape_source` MUST be a superset of
    `price_window` in time.
    """
    if price_window.empty:
        raise ValueError("empty price window")
    # Step 1 on the price window
    p = price_window.to_numpy(dtype=float)
    win_month_key = _month_ts(price_window.index)
    m_mean_win = pd.Series(p).groupby(win_month_key).transform("mean").to_numpy()
    r1_window = p - m_mean_win
    # Step 1 on the shape source (needed for HOW estimation)
    sp = shape_source.to_numpy(dtype=float)
    src_month_key = _month_ts(shape_source.index)
    m_mean_src = pd.Series(sp).groupby(src_month_key).transform("mean").to_numpy()
    r1_source = sp - m_mean_src
    # Step 2: HOW climatology from the shape source
    src_how = _hour_of_week(shape_source.index)
    h_lookup = (pd.Series(r1_source).groupby(src_how).mean())
    win_how = _hour_of_week(price_window.index)
    h_mean_win = h_lookup.reindex(win_how).to_numpy()
    if np.isnan(h_mean_win).any():
        # Any HOW bucket missing in the source -- fall back to overall
        # mean of r1_source to keep the residual finite.
        fill = float(np.mean(r1_source))
        h_mean_win = np.where(np.isnan(h_mean_win), fill, h_mean_win)
    residual = r1_window - h_mean_win

    sd = float(residual.std(ddof=1))
    q05 = float(np.quantile(residual, 0.05))
    q95 = float(np.quantile(residual, 0.95))
    n = int(residual.size)
    low_mask = p < 50.0
    n_lt_50 = int(low_mask.sum())
    if n_lt_50 < n - 10:
        sd_ex = float(residual[~low_mask].std(ddof=1))
    else:
        sd_ex = float("nan")
    return A3Result(residual=residual, sd=sd, q05=q05, q95=q95, n=n,
                    n_lt_50=n_lt_50, sd_excluding_lt_50=sd_ex,
                    L_mean=float(np.mean(p)))


# ----------------------------------------------------------------------
# OLS AR(1) on the residual series
# ----------------------------------------------------------------------
def ols_ar1(x: np.ndarray) -> Dict[str, float]:
    """Fit r_{t+1} = c + phi * r_t + eps by OLS.  Returns phi, kappa
    per hour, and half-life.  No MLE, no TVTP, no regime switch."""
    y0 = x[:-1]; y1 = x[1:]
    X = np.column_stack([np.ones_like(y0), y0])
    beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
    c, phi = float(beta[0]), float(beta[1])
    eps = y1 - X @ beta
    sigma = float(eps.std(ddof=2))
    kappa = -math.log(phi) if 0 < phi < 1 else float("nan")
    half_life = math.log(2.0) / kappa if kappa > 0 else float("nan")
    return {"phi": phi, "sigma_innov": sigma,
            "kappa_per_hour": kappa, "half_life_hours": half_life}


# ----------------------------------------------------------------------
# Production sd_prod(L) mapping
# ----------------------------------------------------------------------
def sd_prod_at_L(L: float, scale_P: float,
                  sd_asinh: float = PROD_SD_ASINH) -> float:
    """Production stationary residual sd on the TRY scale at level L.

    The production model has asinh-scale stationary sd 0.1987, and the
    delta mapping from asinh to TRY is d/dP (asinh(P/s_P))^{-1}
    evaluated at P = L, which gives sqrt(L^2 + s_P^2).  Hence
        sd_prod_TRY(L) = 0.1987 * sqrt(L^2 + scale_P^2).
    """
    return float(sd_asinh * math.sqrt(L * L + scale_P * scale_P))


# ----------------------------------------------------------------------
# Windowed evaluation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DateSpec:
    label: str
    valuation_utc: pd.Timestamp
    start_utc: pd.Timestamp   # window start (inclusive)


def _end_utc_of_date(d_local_str: str) -> pd.Timestamp:
    """Return the UTC hour that corresponds to 23:00 TRT of the given
    local date; this is the last hour whose PTF is fully known.
    """
    return pd.Timestamp(f"{d_local_str} 23:00", tz=TURKEY_TZ).tz_convert("UTC")


def _twelve_month_start(end_utc: pd.Timestamp) -> pd.Timestamp:
    """Return the UTC timestamp exactly 12 local calendar months before
    ``end_utc``.  Uses TR-local subtraction to handle DST cleanly.
    """
    end_local = end_utc.tz_convert(TURKEY_TZ)
    y, m = end_local.year, end_local.month
    y_start = y - 1
    start_local = pd.Timestamp(f"{y_start:04d}-{m:02d}-{end_local.day:02d} "
                                 f"{end_local.hour:02d}:00",
                                 tz=TURKEY_TZ)
    return start_local.tz_convert("UTC")


def eval_row(ptf: pd.Series, spec: DateSpec, scale_P: float) -> Dict:
    """Compute the FW11 statistics for one valuation date spec.

    Uses ONLY PTF observations with UTC timestamp <= spec.valuation_utc.
    """
    # All history up to and including the valuation date
    history = ptf[ptf.index <= spec.valuation_utc]
    # 12-month rolling window
    window = ptf[(ptf.index >= spec.start_utc)
                 & (ptf.index <= spec.valuation_utc)]
    if len(window) < 24 * 30:
        raise ValueError(f"{spec.label}: window too short "
                          f"({len(window)} hours)")

    # (i) pooled shape (from full history <= valuation_utc)
    a_pooled = compute_a3(window, history)
    ols_pooled = ols_ar1(a_pooled.residual)

    # (ii) 12m-only shape (from the same window)
    a_12m = compute_a3(window, window)
    ols_12m = ols_ar1(a_12m.residual)

    L = a_pooled.L_mean       # same across variants (it's the window mean)
    sd_prod_val = sd_prod_at_L(L, scale_P)

    def _pack(a: A3Result, o: Dict, suffix: str) -> Dict:
        return {
            f"n_hours_{suffix}": a.n,
            f"sd_{suffix}": a.sd,
            f"kappa_per_hour_{suffix}": o["kappa_per_hour"],
            f"half_life_hours_{suffix}": o["half_life_hours"],
            f"q05_{suffix}": a.q05,
            f"q95_{suffix}": a.q95,
            f"ratio_over_sd_prod_{suffix}": a.sd / sd_prod_val,
            f"normalized_sd_over_L_{suffix}": a.sd / L,
            f"sd_ex_lt_50_{suffix}": a.sd_excluding_lt_50,
            f"ratio_ex_lt_50_{suffix}": (a.sd_excluding_lt_50 / sd_prod_val
                                          if not math.isnan(
                                              a.sd_excluding_lt_50)
                                          else float("nan")),
        }

    row = {
        "label": spec.label,
        "valuation_utc": str(spec.valuation_utc),
        "window_start_utc": str(spec.start_utc),
        "L_mean_TRY_MWh": L,
        "sd_prod_TRY": sd_prod_val,
        "norm_sd_prod_over_L": sd_prod_val / L,
        "share_pct_lt_50_TRY": 100.0 * a_pooled.n_lt_50 / a_pooled.n,
    }
    row.update(_pack(a_pooled, ols_pooled, "pooled"))
    row.update(_pack(a_12m, ols_12m, "12m_only"))
    return row


# ----------------------------------------------------------------------
# Date universe
# ----------------------------------------------------------------------
FW4_DATES = ("2022-12-31", "2023-06-30", "2023-12-31",
             "2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31")


def build_specs() -> List[DateSpec]:
    """FW4 seven dates + 2026-09-27 (last twelve months, nine of which
    are out of sample under the frozen calibration).  A ninth row
    covers only 2026-01-01 to 2026-09-27 so the reader can isolate the
    2026 sub-window from the mixed 2025 + 2026 twelve-month window.
    """
    specs: List[DateSpec] = []
    for d in FW4_DATES + ("2026-09-27",):
        end_utc = _end_utc_of_date(d)
        start_utc = _twelve_month_start(end_utc)
        specs.append(DateSpec(label=d, valuation_utc=end_utc,
                                start_utc=start_utc))
    # 2026-only variant
    y2026_start_local = pd.Timestamp("2026-01-01 00:00", tz=TURKEY_TZ)
    y2026_end_local = pd.Timestamp("2026-09-27 23:00", tz=TURKEY_TZ)
    specs.append(DateSpec(
        label="2026-01-01_to_2026-09-27",
        valuation_utc=y2026_end_local.tz_convert("UTC"),
        start_utc=y2026_start_local.tz_convert("UTC")))
    return specs


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    yaml_p = load_frozen_parameters(YAML_PATH)
    scale_P = float(yaml_p.scale_P)
    ptf = load_ptf_hourly()
    print(f"PTF loaded: n={len(ptf)}, first={ptf.index[0]}, "
          f"last={ptf.index[-1]}")

    specs = build_specs()
    rows = []
    for s in specs:
        try:
            r = eval_row(ptf, s, scale_P)
            rows.append(r)
        except Exception as exc:
            print(f"  {s.label}: SKIP ({exc})")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "variance_stability.csv", index=False)
    print(f"\nwrote variance_stability.csv with {len(df)} rows")

    # Compact preview
    cols_show = ["label", "L_mean_TRY_MWh", "sd_pooled", "sd_12m_only",
                 "sd_prod_TRY", "ratio_over_sd_prod_pooled",
                 "ratio_over_sd_prod_12m_only",
                 "normalized_sd_over_L_pooled",
                 "kappa_per_hour_pooled",
                 "share_pct_lt_50_TRY", "sd_ex_lt_50_pooled",
                 "ratio_ex_lt_50_pooled"]
    show = df[cols_show].copy()
    show = show.rename(columns={
        "L_mean_TRY_MWh": "L", "sd_pooled": "sd_p", "sd_12m_only": "sd_12",
        "sd_prod_TRY": "sd_prod", "ratio_over_sd_prod_pooled": "r_p",
        "ratio_over_sd_prod_12m_only": "r_12",
        "normalized_sd_over_L_pooled": "norm_sd_p",
        "kappa_per_hour_pooled": "kappa_p",
        "share_pct_lt_50_TRY": "pct_lt50",
        "sd_ex_lt_50_pooled": "sd_ex50",
        "ratio_ex_lt_50_pooled": "r_ex50"})
    print("\n=== compact table ===")
    print(show.round(3).to_string(index=False))

    md_lines = ["# FW11 -- Stationary variance stability across dates\n"]
    md_lines.append("Rows: seven FW4 valuation dates, 2026-09-27, and a "
                    "2026-only sub-window.  All statistics use ONLY data "
                    "with UTC timestamp <= the row's valuation.  No new "
                    "MLE fit; only OLS AR(1) and descriptive stats.  "
                    "Production reference: sd_asinh_stat = 0.1987, "
                    f"scale_P = {scale_P:.2f}, so sd_prod(L) = "
                    "0.1987 * sqrt(L^2 + 282.48^2).")
    md_lines.append("\n## Compact\n")
    md_lines.append(show.round(3).to_markdown(index=False))
    md_lines.append("\n\n## Full detail (variance_stability.csv)\n")
    md_lines.append(df.round(4).to_markdown(index=False))
    (OUT / "variance_stability.md").write_text("\n".join(md_lines),
                                                encoding="utf-8")
    print("wrote variance_stability.md")

    # Figure-source CSV (paper make_figures style)
    fig_rows = []
    for r in rows:
        fig_rows.append({
            "date_label": r["label"],
            "L_TRY_MWh": r["L_mean_TRY_MWh"],
            "sd_pooled_TRY": r["sd_pooled"],
            "sd_12m_only_TRY": r["sd_12m_only"],
            "sd_prod_TRY": r["sd_prod_TRY"],
            "ratio_pooled": r["ratio_over_sd_prod_pooled"],
            "ratio_12m_only": r["ratio_over_sd_prod_12m_only"],
            "norm_sd_pooled": r["normalized_sd_over_L_pooled"],
            "norm_sd_12m_only": r["normalized_sd_over_L_12m_only"],
            "norm_sd_prod": r["norm_sd_prod_over_L"],
            "share_pct_lt_50": r["share_pct_lt_50_TRY"],
            "sd_ex_lt_50_pooled": r["sd_ex_lt_50_pooled"],
            "ratio_ex_lt_50_pooled": r["ratio_ex_lt_50_pooled"],
            "kappa_pooled": r["kappa_per_hour_pooled"],
        })
    fig_df = pd.DataFrame(fig_rows)
    # Paper-figure style header note: DejaVu Serif, 9.5 pt, W = 5.5 in,
    # top and right frames off.  These are style hints for the plotter,
    # NOT graphic output here.
    header = ["# FW11 figure source table -- style: DejaVu Serif, "
              "9.5 pt, width 5.5 in, top and right spines off.",
              "# One row per valuation date.  L: 12-month mean PTF "
              "TRY/MWh.  sd_pooled/sd_12m_only: observed A3 residual sd, "
              "TRY.  sd_prod: production stationary sd at L.",
              "# ratio: observed / production.  norm_sd: observed sd / L.",
              "# share_pct_lt_50: percent of hours below 50 TRY.  "
              "sd_ex_lt_50: sd excluding those hours."]
    with open(OUT / "figure_source.csv", "w", encoding="utf-8") as f:
        for h in header:
            f.write(h + "\n")
        fig_df.to_csv(f, index=False, lineterminator="\n")
    print("wrote figure_source.csv")


if __name__ == "__main__":
    main()
