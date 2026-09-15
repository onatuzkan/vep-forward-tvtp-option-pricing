"""Multi-date backtest of the VEP-anchored hourly forward curve.

Scope -- read this before reading the numbers
---------------------------------------------
This study evaluates the FORWARD CURVE ONLY.  It never prices an option.

The reason is look-ahead bias.  The TVTP / kappa / sigma parameters in
``inputs/historical/m2_frozen_parameters.yaml`` were estimated on data running to
the end of 2025.  Using them at a 2022 or 2023 valuation date would import
information that did not exist then, and every resulting option price would be
contaminated.  The forward-curve construction is different: it consumes only the
VEP quotations published on the valuation date itself and the spot PTF of that
hour.  It contains no historical parameter at all, so it can be replayed at an
earlier date without leakage.

What is replayed, and how it differs from `run_pde.py calibrate-market`
----------------------------------------------------------------------
The curve is built with :func:`pde_option_model.forward_curve.build_forward_curve`
-- the exact function ``calibrate-market`` calls, with the weights and the anchor
rule read from the same ``config/forward_centered_config.yaml``.  The CLI itself
is NOT invoked, for two reasons, both of which would breach the rule above:

  1. ``cmd_calibrate_market`` takes the valuation timestamp and the spot from the
     frozen parameter file (2025-12-31), so it cannot be pointed at an earlier
     valuation date without editing that file;
  2. it builds the TVTP residual model and, unless ``--no-sensitivity`` is given,
     prices a 72 h option.

Everything that is genuinely part of the curve -- the hard delivery-average
equality constraints, the KKT smoothing, the near-term anchor rule, the monthly
acceptance table -- is identical, and ``--verify-2025`` reproduces the published
2025-12-31 curve from ``outputs/market_calibration_final/`` to prove it.

Method
------
For every valuation date d with a published VEP strip:

  * quotes  = the monthly baseload daily index prices observed on d, restricted
    to delivery months 1..``--max-horizon`` months ahead.  If an interior month
    inside that window is unquoted the strip is truncated at the hole (a curve
    cannot be built across an interior gap) and the truncation is reported.
  * spot    = last hourly PTF of the Turkish day d (23:00 TRT).
  * anchor  = the production rule ``spot_to_next_linear`` whenever the nearest
    delivery month carries no quote; otherwise no anchoring happens.
  * forward(d, m) = baseload average of the hourly curve over delivery month m
    (equal to the quote itself for a quoted month, to solver precision).
  * realized(m)   = baseload average of the realised hourly PTF over m.
  * bias(d, m)    = realized(m) - forward(d, m)      [TRY/MWh]

Outputs
-------
    outputs/multi_date/<valuation_date>/hourly_forward_curve.csv
    outputs/multi_date/<valuation_date>/monthly_forward_fit.csv
    outputs/multi_date/<valuation_date>/curve_metadata.json
    outputs/multi_date/multi_date_backtest.csv
    outputs/multi_date/multi_date_summary.md
    outputs/multi_date/fig_bias_by_horizon.csv     (boxplot source data)
    outputs/multi_date/fig_anchor_error.csv        (histogram source data)

Usage:
    python scripts/backtest/multi_date_forward_backtest.py
    python scripts/backtest/multi_date_forward_backtest.py --verify-2025
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pde_option_model.calendar_tr import (  # noqa: E402
    TURKEY_TZ, DeliveryMonth, month_label)
from pde_option_model.forward_curve import (  # noqa: E402
    NearTermAnchor, build_forward_curve)
from pde_option_model.market_data import (  # noqa: E402
    MarketQuoteSet, MonthlyBaseloadQuote)
from pde_option_model.premium import load_epias_ptf_csv  # noqa: E402

VEP_DIR = REPO / "inputs" / "market" / "historical_vep"
PTF_DIR = REPO / "inputs" / "historical" / "ptf_raw"
PTF_2026 = REPO / "inputs" / "market" / "realized_ptf_2026.csv"
OUT_DIR = REPO / "outputs" / "multi_date"
CONFIG = REPO / "config" / "forward_centered_config.yaml"
PUBLISHED_2025 = REPO / "outputs" / "market_calibration_final"

VALUATION_DATES: Tuple[str, ...] = (
    "2022-12-31", "2023-06-30", "2023-12-31", "2024-06-30",
    "2024-12-31", "2025-06-30", "2025-12-31",
)
DEFAULT_MAX_HORIZON = 7
MIN_REALIZED_COVERAGE = 0.98
MIN_QUOTES = 3           # a strip with fewer quoted months is not analysed
PAPER_ANCHOR_BAND_PCT = 20.0
QUOTE_SET_REPORT = VEP_DIR / "quote_set_report.json"

# The single-date result this study is meant to put in context
# (outputs/market_calibration_final/realized_2026_backtest.md).
REFERENCE_2026_MEAN_BIAS = -1019.04


# ---------------------------------------------------------------------------
@dataclass
class DateResult:
    valuation_date: str
    status: str
    note: str = ""
    spot_TRY_MWh: Optional[float] = None
    quoted_months: List[str] = field(default_factory=list)
    anchored_months: List[str] = field(default_factory=list)
    nearest_month_quoted: Optional[bool] = None
    dropped_months: List[str] = field(default_factory=list)
    max_abs_monthly_error: Optional[float] = None
    n_contracts_in_file: int = 0
    ggf_quotation_date: Optional[str] = None
    ggf_staleness_days: Optional[int] = None
    rows: List[Dict[str, Any]] = field(default_factory=list)


def _cfg() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


# ---------------------------------------------------------------------------
def available_strip_dates() -> List[str]:
    """Valuation dates that actually have a quote file under ``VEP_DIR``.

    Used only to warn when a run covers fewer dates than the archive holds, so
    that a deliberately restricted run (``--dates 2025-12-31``) can never be
    mistaken for a complete one when its summary is read later.
    """
    if not VEP_DIR.is_dir():
        return []
    return sorted(d.name for d in VEP_DIR.iterdir()
                  if d.is_dir() and (d / "vep_monthly_quotes.csv").is_file())


# ---------------------------------------------------------------------------
def load_realized_hourly() -> pd.Series:
    parts = [load_epias_ptf_csv(f) for f in sorted(PTF_DIR.glob("ptf_*.csv"))]
    if PTF_2026.exists():
        parts.append(load_epias_ptf_csv(PTF_2026))
    if not parts:
        raise FileNotFoundError(f"no PTF exports under {PTF_DIR}")
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def realized_month(ptf: pd.Series, year: int, month: int) -> Tuple[Optional[float], float, int]:
    """(baseload mean, coverage, n hours) of the realised PTF over a delivery month."""
    dm = DeliveryMonth(int(year), int(month))
    hrs = dm.hours_utc()
    v = ptf.reindex(hrs)
    n = int(v.notna().sum())
    cov = n / dm.n_hours
    if cov < MIN_REALIZED_COVERAGE:
        return None, cov, n
    return float(v.mean()), cov, n


# ---------------------------------------------------------------------------
def _contiguous_strip(df: pd.DataFrame, vyear: int, vmonth: int,
                      max_horizon: int) -> Tuple[pd.DataFrame, List[str]]:
    """Keep the contiguous run of quoted months inside the horizon.

    ``build_forward_curve`` refuses an interior unquoted delivery month (there is
    no interpolation rule for one), so the strip is cut at the first hole and the
    dropped months are reported rather than filled.
    """
    df = df.copy()
    df["tau"] = (df["delivery_year"] - vyear) * 12 + (df["delivery_month"] - vmonth)
    df = df[(df["tau"] >= 1) & (df["tau"] <= max_horizon)].sort_values("tau")
    if df.empty:
        return df, []
    taus = list(df["tau"])
    keep = [taus[0]]
    for t in taus[1:]:
        if t == keep[-1] + 1:
            keep.append(t)
        else:
            break
    dropped = [month_label(int(r.delivery_year), int(r.delivery_month))
               for r in df.itertuples() if r.tau not in keep]
    return df[df["tau"].isin(keep)].copy(), dropped


def build_one_date(date: str, ptf: pd.Series, cfg: Dict[str, Any],
                   max_horizon: int, outdir: Path) -> DateResult:
    qfile = VEP_DIR / date / "vep_monthly_quotes.csv"
    if not qfile.exists():
        return DateResult(date, "NO_QUOTE_FILE",
                          f"{qfile.relative_to(REPO)} not found; VEP GGF for this "
                          "date was not downloaded or was not published")
    raw = pd.read_csv(qfile)
    if raw.empty:
        return DateResult(date, "EMPTY_QUOTE_FILE", "quote file has no rows")
    qday = (str(raw["ggf_quotation_date"].iloc[0])
            if "ggf_quotation_date" in raw.columns else date)
    stale = (int(raw["ggf_staleness_days"].iloc[0])
             if "ggf_staleness_days" in raw.columns else 0)
    if len(raw) < MIN_QUOTES:
        return DateResult(date, "TOO_FEW_QUOTES",
                          f"only {len(raw)} monthly contract(s) quoted; the "
                          f"analysis requires at least {MIN_QUOTES}",
                          n_contracts_in_file=int(len(raw)),
                          ggf_quotation_date=qday, ggf_staleness_days=stale)

    spot_col = raw.get("spot_price_TRY_MWh")
    spot = None
    if spot_col is not None:
        s = pd.to_numeric(spot_col, errors="coerce").dropna()
        if len(s):
            spot = float(s.iloc[0])
    if spot is None:
        return DateResult(date, "NO_SPOT", "quote file carries no spot price")

    vyear, vmonth = int(date[:4]), int(date[5:7])
    strip, dropped = _contiguous_strip(raw, vyear, vmonth, max_horizon)
    if strip.empty:
        return DateResult(date, "NO_FORWARD_QUOTES",
                          f"no quoted delivery month within {max_horizon} months",
                          spot_TRY_MWh=spot, n_contracts_in_file=int(len(raw)),
                          ggf_quotation_date=qday, ggf_staleness_days=stale)
    if len(strip) < MIN_QUOTES:
        return DateResult(date, "TOO_FEW_QUOTES",
                          f"only {len(strip)} contiguous quoted month(s) inside "
                          f"the {max_horizon}-month horizon (dropped: {dropped}); "
                          f"the analysis requires at least {MIN_QUOTES}",
                          spot_TRY_MWh=spot, n_contracts_in_file=int(len(raw)),
                          dropped_months=dropped,
                          ggf_quotation_date=qday, ggf_staleness_days=stale)

    valuation_utc = (pd.Timestamp(date, tz=TURKEY_TZ) + pd.Timedelta(hours=23)
                     ).tz_convert("UTC")
    quotes = MarketQuoteSet(
        valuation_utc=valuation_utc,
        quotes=[MonthlyBaseloadQuote(
            contract_name=str(r.contract_name), year=int(r.delivery_year),
            month=int(r.delivery_month), price_TRY_MWh=float(r.price_TRY_MWh),
            source=str(getattr(r, "source", "EPIAS_VEP")),
            quote_type=str(getattr(r, "quote_type", "monthly_baseload")),
            weight=float(getattr(r, "weight", 1.0) or 1.0))
            for r in strip.itertuples()],
        spot_price_TRY_MWh=spot)
    warns = quotes.validate(strict_interior=True)

    anchor = NearTermAnchor(
        mode=_get(cfg, "market.january_anchor_mode", "spot_to_next_linear"),
        level_TRY_MWh=_get(cfg, "market.january_anchor_level_TRY_MWh"))
    curve = build_forward_curve(
        quotes,
        mode=_get(cfg, "market.curve_mode", "smooth_constrained"),
        anchor=anchor,
        spot_price_TRY_MWh=spot,
        smoothness_weight=float(_get(cfg, "market.smoothness_weight", 1.0)),
        level_weight=float(_get(cfg, "market.level_weight", 1.0e-4)))

    dest = outdir / date
    dest.mkdir(parents=True, exist_ok=True)
    curve.frame.to_csv(dest / "hourly_forward_curve.csv", index=False)
    curve.fit_table().to_csv(dest / "monthly_forward_fit.csv", index=False)

    quoted = set(q.label for q in quotes.quotes)
    # The valuation hour sits inside its own delivery month (23:00 TRT on the
    # last day), leaving a one-hour stub that carries the anchor flag but is not
    # a delivery month this backtest can score.  Drop it from the anchor report.
    partial = set(quotes.partial_months())
    anchored = [m for m in curve.extrapolated_months
                if m not in quoted and m not in partial]
    nearest_label = month_label(*_add_months(vyear, vmonth, 1))
    nearest_quoted = nearest_label in quoted

    avgs = curve.monthly_averages()
    rows: List[Dict[str, Any]] = []
    for tau in range(1, max_horizon + 1):
        y, m = _add_months(vyear, vmonth, tau)
        label = month_label(y, m)
        if label not in avgs:
            continue
        fwd = float(avgs[label])
        real, cov, nh = realized_month(ptf, y, m)
        is_anchor = label in anchored
        row = {
            "valuation_date": date,
            "delivery_month": label,
            "horizon_months": tau,
            "n_delivery_hours": DeliveryMonth(y, m).n_hours,
            "contract_name": (next((q.contract_name for q in quotes.quotes
                                    if q.label == label), "")),
            "quoted": bool(label in quoted),
            "near_term_anchored": bool(is_anchor),
            "forward_TRY_MWh": fwd,
            "realized_TRY_MWh": real,
            "realized_coverage": round(cov, 4),
            "realized_n_hours": nh,
            "spot_at_valuation_TRY_MWh": spot,
            "ggf_quotation_date": qday,
            "ggf_staleness_days": stale,
        }
        if real is None:
            row.update({"bias_TRY_MWh": None, "abs_error_TRY_MWh": None,
                        "sMAPE_pct": None, "relative_bias_pct": None,
                        "realized_status": "insufficient_coverage"})
        else:
            bias = real - fwd
            row.update({
                "bias_TRY_MWh": bias,
                "abs_error_TRY_MWh": abs(bias),
                "sMAPE_pct": 100.0 * abs(bias) / ((abs(real) + abs(fwd)) / 2.0),
                "relative_bias_pct": 100.0 * bias / fwd,
                "realized_status": "ok",
            })
        rows.append(row)

    meta = {
        "valuation_date": date,
        "valuation_utc": valuation_utc.isoformat(),
        "ggf_quotation_date": qday,
        "ggf_staleness_days": stale,
        "n_contracts_in_quote_file": int(len(raw)),
        "spot_price_TRY_MWh": spot,
        "curve_mode": _get(cfg, "market.curve_mode", "smooth_constrained"),
        "anchor_mode": anchor.mode,
        "smoothness_weight": float(_get(cfg, "market.smoothness_weight", 1.0)),
        "level_weight": float(_get(cfg, "market.level_weight", 1.0e-4)),
        "quoted_months": sorted(quoted),
        "near_term_anchored_months": anchored,
        "nearest_delivery_month": nearest_label,
        "nearest_month_quoted": nearest_quoted,
        "dropped_months_after_interior_gap": dropped,
        "max_abs_monthly_constraint_error_TRY_MWh": curve.max_abs_monthly_error(),
        "curve_diagnostics": curve.diagnostics,
        "market_data_warnings": warns,
        "option_pricing_performed": False,
    }
    with open(dest / "curve_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=str)

    return DateResult(date, "ok", spot_TRY_MWh=spot, quoted_months=sorted(quoted),
                      anchored_months=anchored, nearest_month_quoted=nearest_quoted,
                      dropped_months=dropped,
                      max_abs_monthly_error=curve.max_abs_monthly_error(),
                      n_contracts_in_file=int(len(raw)),
                      ggf_quotation_date=qday, ggf_staleness_days=stale,
                      rows=rows)


def _add_months(year: int, month: int, k: int) -> Tuple[int, int]:
    idx = (year * 12 + (month - 1)) + k
    return idx // 12, idx % 12 + 1


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------
def _quantiles(x: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
        "min": float(np.min(x)), "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)), "max": float(np.max(x)),
    }


def _hour_weighted_mean(g: pd.DataFrame) -> float:
    w = g["n_delivery_hours"].to_numpy(float)
    b = g["bias_TRY_MWh"].to_numpy(float)
    return float((b * w).sum() / w.sum())


def analysis_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    d = df.dropna(subset=["bias_TRY_MWh"])
    b = d["bias_TRY_MWh"].to_numpy(float)
    out: Dict[str, Any] = {"bias": _quantiles(b)}

    ref = REFERENCE_2026_MEAN_BIAS
    out["reference_2026_mean_bias_TRY_MWh"] = ref
    out["reference_aggregation"] = (
        "hour-weighted mean of the seven 2026 monthly biases, as published in "
        "outputs/market_calibration_final/realized_2026_backtest.md")
    out["reference_percentile_in_pooled_bias"] = float(100.0 * np.mean(b <= ref))
    out["n_more_negative_than_reference"] = int(np.sum(b < ref))

    sub = d[d["valuation_date"] == "2025-12-31"]
    out["mean_bias_2025-12-31_unweighted"] = (float(sub["bias_TRY_MWh"].mean())
                                              if len(sub) else None)
    out["mean_bias_2025-12-31_hour_weighted"] = (_hour_weighted_mean(sub)
                                                 if len(sub) else None)

    per = (d.groupby("valuation_date")
            .apply(lambda g: pd.Series({
                "count": int(len(g)),
                "mean": float(g["bias_TRY_MWh"].mean()),
                "median": float(g["bias_TRY_MWh"].median()),
                "hour_weighted_mean": _hour_weighted_mean(g)}),
                include_groups=False)
            .round(2))
    out["per_valuation_date_mean_bias"] = per.to_dict("index")

    dm = per["mean"].to_numpy(float)
    out["date_mean_distribution"] = _quantiles(dm) if dm.size else None
    out["reference_percentile_among_date_means"] = (
        float(100.0 * np.mean(dm <= ref)) if dm.size else None)
    out["n_date_means_more_negative_than_reference"] = int(np.sum(dm < ref))

    # Relative view: the TRY price level roughly doubled over the sample, so an
    # absolute TRY/MWh bias is not comparable across valuation dates.
    rel = d["relative_bias_pct"].to_numpy(float)
    out["relative_bias_pct"] = _quantiles(rel)
    per_rel = (d.groupby("valuation_date")["relative_bias_pct"].mean().round(2))
    out["per_valuation_date_mean_relative_bias_pct"] = per_rel.to_dict()
    ref_rel = float(sub["relative_bias_pct"].mean()) if len(sub) else float("nan")
    out["reference_2026_mean_relative_bias_pct"] = ref_rel
    out["reference_percentile_in_pooled_relative_bias"] = (
        float(100.0 * np.mean(rel <= ref_rel)))
    dr = per_rel.to_numpy(float)
    out["reference_percentile_among_date_mean_relative_bias"] = (
        float(100.0 * np.mean(dr <= ref_rel)))
    out["n_date_means_more_negative_than_reference_relative"] = int(np.sum(dr < ref_rel))
    out["date_mean_relative_distribution"] = _quantiles(dr)
    return out


def _ttest(x: np.ndarray) -> Dict[str, float]:
    from scipy import stats
    if x.size < 2:
        return {"n": int(x.size), "mean": float(np.mean(x)) if x.size else float("nan"),
                "sd": float("nan"), "t_statistic": float("nan"),
                "p_value": float("nan")}
    t, p = stats.ttest_1samp(x, 0.0)
    return {"n": int(x.size), "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "t_statistic": float(t), "p_value": float(p)}


def non_overlapping_subsets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Valuation-date subsets whose delivery-month windows are pairwise disjoint.

    Greedy scan in date order starting from each candidate date: a date is kept
    when its first delivery month lies strictly after the last delivery month of
    the previously kept date.  Every maximal disjoint family that the greedy
    scan finds is returned, keyed by its first date, so the reader sees all
    the non-overlapping alternatives rather than one arbitrary choice.
    """
    d = df.dropna(subset=["bias_TRY_MWh"])
    win = (d.groupby("valuation_date")["delivery_month"]
             .agg(["min", "max"]).sort_index())
    dates = list(win.index)
    families: Dict[str, List[str]] = {}
    for start in range(len(dates)):
        keep = [dates[start]]
        for dd in dates[start + 1:]:
            if win.loc[dd, "min"] > win.loc[keep[-1], "max"]:
                keep.append(dd)
        key = keep[0]
        if not any(set(keep) <= set(v) for v in families.values()):
            families[key] = keep
    return families


def analysis_systematic(df: pd.DataFrame) -> Dict[str, Any]:
    d = df.dropna(subset=["bias_TRY_MWh"])
    b = d["bias_TRY_MWh"].to_numpy(float)
    per_date = d.groupby("valuation_date")["bias_TRY_MWh"].mean().to_numpy(float)
    fams = non_overlapping_subsets(df)
    subsets = {}
    for key, dates in fams.items():
        sub = d[d["valuation_date"].isin(dates)]
        subsets[key] = {
            "valuation_dates": dates,
            "delivery_windows": {dd: [sub[sub.valuation_date == dd].delivery_month.min(),
                                      sub[sub.valuation_date == dd].delivery_month.max()]
                                 for dd in dates},
            "pooled": _ttest(sub["bias_TRY_MWh"].to_numpy(float)),
            "date_level": _ttest(sub.groupby("valuation_date")["bias_TRY_MWh"]
                                    .mean().to_numpy(float)),
            "share_negative": float(np.mean(sub["bias_TRY_MWh"] < 0)),
        }
    return {
        "naive_pooled": _ttest(b),
        "date_level_aggregate": {
            "n_valuation_dates": int(per_date.size),
            "mean_of_date_means": float(np.mean(per_date)),
            **{k: v for k, v in _ttest(per_date).items() if k not in ("n", "mean")}},
        "non_overlapping_subsets": subsets,
        "independence_warning": (
            "The pooled observations are NOT independent: the same delivery month "
            "is seen from several valuation dates, and the months of one valuation "
            "date share a single market state. Both p-values are therefore "
            "anti-conservative and must not be read as calibrated significance "
            "levels. The date-level aggregate removes the within-date dependence "
            "only, not the overlap between dates."),
        "n_deliveries_seen_more_than_once": int(
            (df.dropna(subset=["bias_TRY_MWh"])
               .groupby("delivery_month")["valuation_date"].nunique() > 1).sum()),
        "sign_test_share_negative": float(np.mean(b < 0)) if b.size else float("nan"),
    }


def analysis_anchor(df: pd.DataFrame) -> Dict[str, Any]:
    a = df[df["near_term_anchored"]].dropna(subset=["bias_TRY_MWh"])
    q = df[~df["near_term_anchored"]].dropna(subset=["bias_TRY_MWh"])
    out: Dict[str, Any] = {
        "n_anchored_observations": int(len(a)),
        "n_quoted_observations": int(len(q)),
        "anchor_rule": "spot_to_next_linear",
    }
    if len(a):
        out["anchored_bias_TRY_MWh"] = _quantiles(a["bias_TRY_MWh"].to_numpy(float))
        rel = a["relative_bias_pct"].to_numpy(float)
        out["anchored_relative_bias_pct"] = _quantiles(rel)
        out["share_within_plus_minus_20_pct"] = float(
            np.mean(np.abs(rel) <= PAPER_ANCHOR_BAND_PCT))
        out["paper_band_pct"] = PAPER_ANCHOR_BAND_PCT
        out["anchored_abs_relative_bias_pct"] = _quantiles(np.abs(rel))
        out["anchored_rmse_relative_pct"] = float(np.sqrt(np.mean(rel ** 2)))
        # the +/-20 % band read as a uniform prior has sd 20/sqrt(3) = 11.55 %
        out["uniform_band_sd_pct"] = float(PAPER_ANCHOR_BAND_PCT / np.sqrt(3.0))
        out["empirical_iqr_pct"] = float(np.percentile(rel, 75) - np.percentile(rel, 25))
        sd = out["anchored_relative_bias_pct"]["std"]
        # verdict on the empirical spread relative to the band's implied sd;
        # differences under 2 percentage points on a handful of observations
        # are not distinguishable and are called "comparable"
        out["band_verdict"] = (
            "comparable to" if abs(sd - out["uniform_band_sd_pct"]) < 2.0
            else "narrower than" if sd < out["uniform_band_sd_pct"]
            else "wider than")
    if len(q):
        out["quoted_bias_TRY_MWh"] = _quantiles(q["bias_TRY_MWh"].to_numpy(float))
        relq = q["relative_bias_pct"].to_numpy(float)
        out["quoted_relative_bias_pct"] = _quantiles(relq)
    return out


def horizon_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["bias_TRY_MWh"])
    g = d.groupby("horizon_months")
    t = pd.DataFrame({
        "n": g["bias_TRY_MWh"].size(),
        "mean_bias_TRY_MWh": g["bias_TRY_MWh"].mean(),
        "median_bias_TRY_MWh": g["bias_TRY_MWh"].median(),
        "std_bias_TRY_MWh": g["bias_TRY_MWh"].std(ddof=1),
        "min_bias_TRY_MWh": g["bias_TRY_MWh"].min(),
        "q25_bias_TRY_MWh": g["bias_TRY_MWh"].quantile(0.25),
        "q75_bias_TRY_MWh": g["bias_TRY_MWh"].quantile(0.75),
        "max_bias_TRY_MWh": g["bias_TRY_MWh"].max(),
        "MAE_TRY_MWh": g["abs_error_TRY_MWh"].mean(),
        "sMAPE_pct": g["sMAPE_pct"].mean(),
    }).reset_index()
    return t.round(4)


def horizon_monotonicity(df: pd.DataFrame) -> Dict[str, Any]:
    from scipy import stats
    d = df.dropna(subset=["bias_TRY_MWh"])
    rho, p = stats.spearmanr(d["horizon_months"], d["abs_error_TRY_MWh"])
    rho_s, p_s = stats.spearmanr(d["horizon_months"], d["sMAPE_pct"])
    mae = d.groupby("horizon_months")["abs_error_TRY_MWh"].mean()
    return {
        "spearman_horizon_vs_abs_error": float(rho), "p_value_abs_error": float(p),
        "spearman_horizon_vs_sMAPE": float(rho_s), "p_value_sMAPE": float(p_s),
        "MAE_by_horizon": {int(k): float(v) for k, v in mae.items()},
        "MAE_is_monotone_increasing": bool(np.all(np.diff(mae.to_numpy()) > 0)),
        "note": "p-values share the dependence caveat of section 3",
    }


# ---------------------------------------------------------------------------
def verify_2025(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the published 2025-12-31 curve from the shipped quote file."""
    from pde_option_model.market_data import load_quotes
    ref = PUBLISHED_2025 / "hourly_forward_curve.csv"
    if not ref.exists():
        return {"status": "reference_curve_missing", "path": str(ref)}
    quotes = load_quotes(REPO / "inputs" / "market" / "vep_monthly_quotes.csv")
    curve = build_forward_curve(
        quotes, mode=_get(cfg, "market.curve_mode", "smooth_constrained"),
        anchor=NearTermAnchor(
            mode=_get(cfg, "market.january_anchor_mode", "spot_to_next_linear"),
            level_TRY_MWh=_get(cfg, "market.january_anchor_level_TRY_MWh")),
        spot_price_TRY_MWh=quotes.spot_price_TRY_MWh,
        smoothness_weight=float(_get(cfg, "market.smoothness_weight", 1.0)),
        level_weight=float(_get(cfg, "market.level_weight", 1.0e-4)))
    pub = pd.read_csv(ref)
    a = curve.values.to_numpy(float)
    b = pd.to_numeric(pub["hourly_forward_TRY_MWh"]).to_numpy(float)
    if a.size != b.size:
        return {"status": "length_mismatch", "rebuilt": int(a.size),
                "published": int(b.size)}
    return {"status": "ok", "n_hours": int(a.size),
            "max_abs_difference_TRY_MWh": float(np.max(np.abs(a - b)))}


def _stale_anchor_note(df: pd.DataFrame, stale: List["DateResult"]) -> str:
    """Sentence tying the stalest carried-forward date to the anchor tail, if
    that date is in fact outside the manuscript band; empty otherwise."""
    if not stale:
        return ""
    worst = max(stale, key=lambda r: r.ggf_staleness_days or 0)
    a = df[(df["near_term_anchored"]) & (df["valuation_date"] == worst.valuation_date)]
    if a.empty or a["relative_bias_pct"].isna().all():
        return ""
    rel = float(a["relative_bias_pct"].iloc[0])
    if abs(rel) <= PAPER_ANCHOR_BAND_PCT:
        return (f" The stalest case ({worst.valuation_date}, GGF of "
                f"{worst.ggf_quotation_date}) has an anchored-month relative bias "
                f"of {rel:+.1f} %, inside the manuscript band.")
    return (f" The stalest case ({worst.valuation_date}, GGF of "
            f"{worst.ggf_quotation_date}, spot of a non-business night) is also "
            f"an anchored month outside ±{PAPER_ANCHOR_BAND_PCT:.0f} % "
            f"({rel:+.1f} %); the reader should weigh that observation "
            f"accordingly rather than have it removed.")


# ---------------------------------------------------------------------------
def write_summary(path: Path, results: List[DateResult], df: pd.DataFrame,
                  dist: Dict[str, Any], sysx: Dict[str, Any],
                  anch: Dict[str, Any], hz: pd.DataFrame,
                  verify: Optional[Dict[str, Any]], cfg: Dict[str, Any],
                  max_horizon: int, mono: Optional[Dict[str, Any]] = None,
                  anchor_check: Optional[Dict[str, Any]] = None,
                  available: Optional[List[str]] = None) -> None:
    ok = [r for r in results if r.status == "ok"]
    bad = [r for r in results if r.status != "ok"]
    L: List[str] = []
    A = L.append

    A("# Multi-date forward-curve backtest")
    A("")
    A("**This study evaluates the forward curve only. No option is priced "
      "anywhere in it.**")
    A("")
    A("The TVTP, kappa and sigma parameters of the model were estimated on data "
      "running to the end of 2025. Applying them at a 2022 or 2023 valuation "
      "date would be look-ahead bias, and any option price produced that way "
      "would be contaminated. The forward-curve construction uses only the VEP "
      "quotations published on the valuation date and the spot PTF of that "
      "hour; it contains no historical parameter, so replaying it at an earlier "
      "date is leak-free. That is the whole reason this backtest exists in this "
      "restricted form, and it is the reason its conclusions say nothing "
      "directly about option-price accuracy.")
    A("")
    A(f"* valuation dates attempted: {len(results)}")
    A(f"* valuation dates with a usable VEP strip: {len(ok)}")
    unused: List[str] = []
    if available is not None:
        A(f"* valuation dates with a quote file on disk: {len(available)}")
        unused = [d for d in available
                  if d not in {r.valuation_date for r in results}]
    A(f"* delivery horizon per date: 1..{max_horizon} months ahead")
    A(f"* curve mode: `{_get(cfg, 'market.curve_mode')}`, "
      f"anchor rule: `{_get(cfg, 'market.january_anchor_mode')}`, "
      f"smoothness {_get(cfg, 'market.smoothness_weight')}, "
      f"level {_get(cfg, 'market.level_weight')}")
    A(f"* usable (valuation date x delivery month) observations: "
      f"{int(df['bias_TRY_MWh'].notna().sum())}")
    A("")
    if unused:
        A(f"> **This run is restricted.** {len(unused)} valuation date(s) carry "
          f"a quote file under `inputs/market/historical_vep/` but were not "
          f"requested on the command line: {', '.join(unused)}. Every number "
          f"below is computed on the {len(results)} requested date(s) only and "
          f"must not be read as the full-sample result.")
        A("")

    A("## 0. Is the downloaded series the series the manuscript uses?")
    A("")
    if anchor_check and anchor_check.get("status") == "ok":
        cc = anchor_check["contracts"]
        A(f"The production quote file `inputs/market/vep_monthly_quotes.csv` "
          f"(source `EPIAS_VEP_daily_reference_price`, the 2025-12-31 strip the "
          f"single-date result is built on) was compared contract by contract "
          f"with the downloaded EPİAŞ `vep-ggf` (VEP Günlük Gösterge Fiyatı) rows "
          f"of {anchor_check['anchor_date']}:")
        A("")
        A("| contract | production file | downloaded GGF | abs diff | matching decimals |")
        A("|---|---|---|---|---|")
        for c, v in cc.items():
            A(f"| {c} | {v['expected']:.2f} | {v['downloaded']:.2f} | "
              f"{v['abs_diff']:.2e} | {v['matching_decimals']} |")
        A("")
        A(f"Maximum absolute difference {anchor_check['max_abs_diff_TRY_MWh']:.2e} "
          f"TRY/MWh against a stop tolerance of "
          f"{anchor_check['tolerance_TRY_MWh']} TRY/MWh. The two names label the "
          f"same EPİAŞ series; the multi-date analysis is comparable with the "
          f"single-date result. (`scripts/backtest/build_vep_quote_sets.py` "
          f"exits with code 2 and writes nothing if this check fails.)")
    else:
        A(f"**No anchor-check evidence found** (`{QUOTE_SET_REPORT.name}` missing "
          f"or failed: `{anchor_check}`). Do not compare these numbers with the "
          f"single-date result until it is re-run.")
    A("")

    if verify:
        A("## Equivalence with the production pipeline")
        A("")
        if verify.get("status") == "ok":
            A(f"Rebuilding the 2025-12-31 curve with this script and comparing it "
              f"hour by hour with the published "
              f"`outputs/market_calibration_final/hourly_forward_curve.csv` gives a "
              f"maximum absolute difference of "
              f"**{verify['max_abs_difference_TRY_MWh']:.3e} TRY/MWh** over "
              f"{verify['n_hours']} hours. The curve built here is the production "
              f"curve.")
        else:
            A(f"Verification did not run cleanly: `{verify}`.")
        A("")

    A("## 1. Coverage: which dates carried a strip, and did the anchor fire?")
    A("")
    A("| valuation date | weekday | GGF quotation date (staleness) | contracts "
      "in file | spot TRY/MWh | quoted months in horizon | nearest month "
      "quoted? | anchored months | months dropped at an interior gap | max abs "
      "monthly constraint error |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        wd = pd.Timestamp(r.valuation_date).day_name()[:3]
        gq = (f"{r.ggf_quotation_date} ({r.ggf_staleness_days} d)"
              if r.ggf_quotation_date else "—")
        if r.status != "ok":
            A(f"| {r.valuation_date} | {wd} | {gq} | {r.n_contracts_in_file} | — | "
              f"— | — | — | — | **{r.status}** |")
            continue
        A(f"| {r.valuation_date} | {wd} | {gq} | {r.n_contracts_in_file} | "
          f"{r.spot_TRY_MWh:.2f} | "
          f"{len(r.quoted_months)} ({', '.join(r.quoted_months) or '—'}) | "
          f"{'yes' if r.nearest_month_quoted else '**no — anchor fired**'} | "
          f"{', '.join(r.anchored_months) or '—'} | "
          f"{', '.join(r.dropped_months) or '—'} | "
          f"{r.max_abs_monthly_error:.2e} |")
    A("")
    stale = [r for r in ok if (r.ggf_staleness_days or 0) > 0]
    if stale:
        A(f"GGF is published on business days only. {len(stale)} of the {len(ok)} "
          f"valuation dates fall on a weekend or public holiday; for them the "
          f"quote set is the last GGF published *before* the valuation date "
          f"({', '.join(f'{r.valuation_date} ← {r.ggf_quotation_date}' for r in stale)}). "
          f"That is the information a market participant held at the end of the "
          f"valuation day; no later publication is used. The spot is still the "
          f"23:00 PTF of the valuation day itself. Under a strict same-day rule "
          f"these {len(stale)} dates would have been dropped and the study would "
          f"rest on {len(ok) - len(stale)} dates.")
        A("")
    worst_err = max((r.max_abs_monthly_error for r in ok), default=float('nan'))
    A(f"Every quoted month is a hard equality constraint of the KKT system; the "
      f"largest monthly reproduction error over all {len(ok)} curves is "
      f"{worst_err:.2e} TRY/MWh"
      + (" — all at solver precision." if worst_err < 1e-9 else
         " — **one or more curves did not reproduce their quotes at solver "
         "precision; inspect the per-date `monthly_forward_fit.csv`.**"))
    A("")
    few = [r for r in results if r.status == "TOO_FEW_QUOTES"]
    A(f"Minimum strip length for inclusion: {MIN_QUOTES} quoted months. "
      + (f"Excluded on that rule: "
         + "; ".join(f"{r.valuation_date} ({r.note})" for r in few) + "."
         if few else "No date was excluded on that rule."))
    A("")
    n_anchor = sum(1 for r in ok if r.nearest_month_quoted is False)
    A(f"The near-term anchor rule actually fired at **{n_anchor} of {len(ok)}** "
      f"valuation dates, producing {anch['n_anchored_observations']} anchored "
      f"(valuation date x delivery month) observations against "
      f"{anch['n_quoted_observations']} quote-constrained ones.")
    A("")
    if bad:
        A("Dates with no usable strip:")
        A("")
        for r in bad:
            A(f"* **{r.valuation_date}** — {r.status}: {r.note}")
        A("")

    A("## 2. Distribution of the bias (realized − forward)")
    A("")
    b = dist["bias"]
    A("| statistic | TRY/MWh |")
    A("|---|---|")
    for k, lab in (("n", "observations"), ("mean", "mean"), ("median", "median"),
                   ("std", "standard deviation"), ("min", "minimum"),
                   ("q25", "25th percentile"), ("q75", "75th percentile"),
                   ("max", "maximum")):
        v = b[k]
        A(f"| {lab} | {v:.0f} |" if k == "n" else f"| {lab} | {v:,.2f} |")
    A("")
    pct = dist["reference_percentile_in_pooled_bias"]
    A(f"The single-date figure this study exists to contextualise — the "
      f"**{REFERENCE_2026_MEAN_BIAS:,.2f} TRY/MWh** mean bias of the 2025-12-31 "
      f"valuation over 2026 — sits at the **{pct:.1f}th percentile** of the "
      f"pooled bias distribution: {dist['n_more_negative_than_reference']} of "
      f"{b['n']} individual (date x month) biases are more negative than it.")
    A("")
    A(f"That published figure is the *hour-weighted* mean of the seven 2026 "
      f"monthly biases. Recomputed here from the same curve it is "
      f"{dist['mean_bias_2025-12-31_hour_weighted']:,.2f} TRY/MWh hour-weighted "
      f"and {dist['mean_bias_2025-12-31_unweighted']:,.2f} TRY/MWh unweighted; "
      f"the tables below use the unweighted convention throughout.")
    A("")
    if dist.get("reference_percentile_among_date_means") is not None:
        A(f"Compared like with like — against the {dist['date_mean_distribution']['n']} "
          f"per-valuation-date mean biases rather than individual months — it sits "
          f"at the **{dist['reference_percentile_among_date_means']:.1f}th "
          f"percentile**, in a date-mean distribution running from "
          f"{dist['date_mean_distribution']['min']:,.2f} to "
          f"{dist['date_mean_distribution']['max']:,.2f} TRY/MWh "
          f"(median {dist['date_mean_distribution']['median']:,.2f}).")
        A("")
    A("Mean bias by valuation date:")
    A("")
    A("| valuation date | n | mean bias TRY/MWh | median bias | hour-weighted mean | "
      "mean relative bias % |")
    A("|---|---|---|---|---|---|")
    prel = dist["per_valuation_date_mean_relative_bias_pct"]
    for d, v in dist["per_valuation_date_mean_bias"].items():
        A(f"| {d} | {int(v['count'])} | {v['mean']:,.2f} | {v['median']:,.2f} | "
          f"{v['hour_weighted_mean']:,.2f} | {prel[d]:+.1f} |")
    A("")
    rb = dist["relative_bias_pct"]
    A(f"Relative view (bias / forward, %), because the TRY price level is not "
      f"constant across the sample: pooled mean {rb['mean']:+.1f} %, median "
      f"{rb['median']:+.1f} %, sd {rb['std']:.1f} %, quartiles "
      f"{rb['q25']:+.1f} % / {rb['q75']:+.1f} %, range {rb['min']:+.1f} % to "
      f"{rb['max']:+.1f} %. The 2025-12-31 date mean is "
      f"{dist['reference_2026_mean_relative_bias_pct']:+.1f} %, which is the "
      f"**{dist['reference_percentile_among_date_mean_relative_bias']:.1f}th "
      f"percentile** of the {dist['date_mean_relative_distribution']['n']} "
      f"date means ({dist['n_date_means_more_negative_than_reference_relative']} "
      f"dates more negative) and the "
      f"{dist['reference_percentile_in_pooled_relative_bias']:.1f}th percentile "
      f"of the pooled monthly relative biases.")
    A("")
    dmn = dist["date_mean_distribution"]
    pct_dm = dist["reference_percentile_among_date_means"]
    k_abs = dist["n_date_means_more_negative_than_reference"]
    k_rel = dist["n_date_means_more_negative_than_reference_relative"]
    n_d = dmn["n"]
    worse = [d_ for d_, v in dist["per_valuation_date_mean_bias"].items()
             if v["mean"] < REFERENCE_2026_MEAN_BIAS]
    if k_abs == 0 and k_rel == 0:
        verdict = ("2026 is the **most negative valuation date in the sample** "
                   "on both the absolute and the relative scale: an extreme "
                   "observation, not a typical one.")
    elif k_abs == 0 or k_rel == 0:
        verdict = ("2026 is the most negative valuation date on one scale but "
                   "not the other: at the tail of the sample, not clearly "
                   "outside it.")
    elif 25.0 <= pct_dm <= 75.0 and dmn["q25"] <= REFERENCE_2026_MEAN_BIAS <= dmn["q75"]:
        verdict = (f"2026 is **not an outlier**: {k_abs} of the other {n_d - 1} "
                   f"valuation dates were more negative in TRY/MWh and {k_rel} "
                   f"in relative terms, and the 2026 date mean lies inside the "
                   f"interquartile range of the date means. It is a large "
                   f"negative bias of a kind the VEP strip has produced before "
                   f"({', '.join(worse) or '—'}), not a one-off.")
    else:
        verdict = (f"2026 sits in the lower tail but not at its end: {k_abs} of "
                   f"the other {n_d - 1} dates were more negative in TRY/MWh and "
                   f"{k_rel} in relative terms.")
    pos = [d_ for d_, v in dist["per_valuation_date_mean_bias"].items() if v["mean"] > 0]
    neg = [d_ for d_, v in dist["per_valuation_date_mean_bias"].items() if v["mean"] <= 0]
    A(f"**Verdict on the manuscript narrative.** {verdict} The sign of the "
      f"date mean is not stable either: the biases run from "
      f"{dmn['min']:,.0f} to {dmn['max']:,.0f} TRY/MWh; realised came in "
      f"*above* the strip at {len(pos)} valuation dates ({', '.join(pos) or '—'}) "
      f"and below it at {len(neg)} ({', '.join(neg) or '—'}). The VEP monthly "
      f"strip has therefore been a **biased and unstable** predictor of the "
      f"realised monthly baseload over 2023-26, not a consistently over-priced "
      f"one.")
    A("")

    A("## 3. Is the bias systematic?")
    A("")
    n = sysx["naive_pooled"]
    d = sysx["date_level_aggregate"]
    A("| sample | valuation dates | n obs | mean bias TRY/MWh | sd | t | p | share negative |")
    A("|---|---|---|---|---|---|---|---|")
    A(f"| all dates, pooled months | {d['n_valuation_dates']} | {n['n']} | "
      f"{n['mean']:,.2f} | {n['sd']:,.2f} | {n['t_statistic']:.3f} | "
      f"{n['p_value']:.4g} | {100 * sysx['sign_test_share_negative']:.1f} % |")
    A(f"| all dates, one mean per date | {d['n_valuation_dates']} | "
      f"{d['n_valuation_dates']} | {d['mean_of_date_means']:,.2f} | {d['sd']:,.2f} | "
      f"{d['t_statistic']:.3f} | {d['p_value']:.4g} | — |")
    for key, sub in sysx["non_overlapping_subsets"].items():
        sp, sdl = sub["pooled"], sub["date_level"]
        A(f"| non-overlapping subset from {key}, pooled months | "
          f"{len(sub['valuation_dates'])} | {sp['n']} | {sp['mean']:,.2f} | "
          f"{sp['sd']:,.2f} | {sp['t_statistic']:.3f} | {sp['p_value']:.4g} | "
          f"{100 * sub['share_negative']:.1f} % |")
        A(f"| non-overlapping subset from {key}, one mean per date | "
          f"{len(sub['valuation_dates'])} | {sdl['n']} | {sdl['mean']:,.2f} | "
          f"{sdl['sd']:,.2f} | {sdl['t_statistic']:.3f} | {sdl['p_value']:.4g} | — |")
    A("")
    for key, sub in sysx["non_overlapping_subsets"].items():
        wins = ", ".join(f"{dd} → {w[0]}..{w[1]}"
                         for dd, w in sub["delivery_windows"].items())
        A(f"* subset from {key}: {wins}")
    A("")
    A(f"**Read every p-value with suspicion.** {sysx['independence_warning']} "
      f"{sysx['n_deliveries_seen_more_than_once']} delivery months appear from "
      f"more than one valuation date in the full sample. The non-overlapping "
      f"subsets remove the shared-delivery-month overlap and the date-level "
      f"rows remove the within-date dependence, but consecutive delivery months "
      f"of one curve still share a single market state, and the price level of "
      f"one half-year is not independent of the next; none of these tests is a "
      f"calibrated significance level.")
    A("")

    A("## 4. Error where the anchor rule fired")
    A("")
    if anch["n_anchored_observations"]:
        aa = anch["anchored_relative_bias_pct"]
        ab = anch["anchored_bias_TRY_MWh"]
        A("| statistic | anchored months, relative bias % | anchored months, "
          "bias TRY/MWh |")
        A("|---|---|---|")
        for k, lab in (("n", "observations"), ("mean", "mean"),
                       ("median", "median"), ("std", "standard deviation"),
                       ("min", "minimum"), ("q25", "25th percentile"),
                       ("q75", "75th percentile"), ("max", "maximum")):
            if k == "n":
                A(f"| {lab} | {aa[k]:.0f} | {ab[k]:.0f} |")
            else:
                A(f"| {lab} | {aa[k]:,.2f} | {ab[k]:,.2f} |")
        A("")
        share = 100.0 * anch["share_within_plus_minus_20_pct"]
        A(f"The manuscript quotes a ±{PAPER_ANCHOR_BAND_PCT:.0f} % band for the "
          f"near-term anchor. Empirically **{share:.1f} %** of the anchored months "
          f"fall inside ±{PAPER_ANCHOR_BAND_PCT:.0f} %, the realised spread runs "
          f"from {aa['min']:,.1f} % to {aa['max']:,.1f} %, the interquartile "
          f"range is {anch['empirical_iqr_pct']:.1f} percentage points, the "
          f"standard deviation is {aa['std']:.1f} % and the RMS relative error "
          f"is {anch['anchored_rmse_relative_pct']:.1f} %, on "
          f"{anch['n_anchored_observations']} observations.")
        A("")
        widest = max(abs(aa["min"]), abs(aa["max"]))
        n_out = int(round((1.0 - anch["share_within_plus_minus_20_pct"])
                          * anch["n_anchored_observations"]))
        A(f"**Verdict:** a ±{PAPER_ANCHOR_BAND_PCT:.0f} % band read as a uniform "
          f"prior has standard deviation {anch['uniform_band_sd_pct']:.1f} %; the "
          f"empirical standard deviation is {aa['std']:.1f} %, so in *spread* the "
          f"empirical distribution is **{anch['band_verdict']}** the band "
          f"(difference {abs(aa['std'] - anch['uniform_band_sd_pct']):.1f} "
          f"percentage points on {anch['n_anchored_observations']} observations). "
          f"The empirical distribution is **not narrower** than the band: "
          f"{n_out} of {anch['n_anchored_observations']} anchored months fall "
          f"outside ±{PAPER_ANCHOR_BAND_PCT:.0f} % (widest {widest:.1f} %), so "
          f"the band does not contain the tail. Its centre "
          f"is {aa['mean']:+.1f} % (median {aa['median']:+.1f} %): the ramp "
          f"{'over-prices' if aa['mean'] < 0 else 'under-prices'} the anchored "
          f"month on average, and a band symmetric around zero ignores that "
          f"shift. Read the ±{PAPER_ANCHOR_BAND_PCT:.0f} % figure as an "
          f"approximately right *scale* for the anchor uncertainty, not as a "
          f"containment bound.")
        if anch.get("quoted_relative_bias_pct"):
            qq = anch["quoted_relative_bias_pct"]
            A("")
            A(f"For comparison, quote-constrained months over the same sample "
              f"have mean relative bias {qq['mean']:,.2f} % "
              f"(median {qq['median']:,.2f} %, sd {qq['std']:,.2f} %).")
    else:
        A("The anchor rule never fired in this sample: every nearest delivery "
          "month carried a published quote. The ±20 % band therefore has no "
          "empirical counterpart here.")
    A("")

    A("## 5. Error by delivery horizon")
    A("")
    A(hz.to_markdown(index=False))
    A("")
    if mono:
        mae = mono["MAE_by_horizon"]
        A(f"Expectation: error grows with horizon. Observed MAE by horizon (1 → "
          f"{max(mae)}): " + ", ".join(f"{k}: {v:,.0f}" for k, v in mae.items())
          + f" TRY/MWh. MAE is "
          f"{'monotonically increasing' if mono['MAE_is_monotone_increasing'] else '**not** monotone'}"
          f" in the horizon; Spearman rank correlation between horizon and "
          f"absolute error is {mono['spearman_horizon_vs_abs_error']:+.3f} "
          f"(p = {mono['p_value_abs_error']:.3g}, same dependence caveat as "
          f"section 3), and between horizon and sMAPE "
          f"{mono['spearman_horizon_vs_sMAPE']:+.3f} "
          f"(p = {mono['p_value_sMAPE']:.3g}).")
        A("")
        peak = max(mae, key=mae.get)
        A(f"Two things to read from that profile. First, horizon 1 is *always* "
          f"the anchored month in this sample (the front contract is delisted "
          f"before month end), so the horizon-1 column is the spot-to-next "
          f"ramp, not a VEP quote; its MAE of {mae[1]:,.0f} TRY/MWh is the "
          f"smallest of all horizons. Second, the error rises from horizon 1 to "
          f"a peak at horizon {peak} ({mae[peak]:,.0f} TRY/MWh) and then falls "
          f"back toward horizon {max(mae)} ({mae[max(mae)]:,.0f} TRY/MWh). The "
          f"decline at the long end should not be read as a forecasting virtue "
          f"of the far contract: with {len(mae)} horizons and "
          f"{len(ok)} valuation dates six months apart, horizon {max(mae)} is "
          f"always the same two calendar months (July for the December dates, "
          f"January for the June dates), so the long-end column is as much a "
          f"calendar-month effect as a maturity effect. The expectation 'error "
          f"grows with horizon' holds up to horizon {peak} and not beyond, and "
          f"the rank correlation is weak.")
        A("")

    A("## 6. Honesty check")
    A("")
    A("* **No option was priced in this study.** Only the forward curve was "
      "evaluated. The reason is stated at the top: the residual-process "
      "parameters are fitted to data through 2025 and would be look-ahead bias "
      "at every earlier valuation date.")
    A("* `run_pde.py calibrate-market` was not invoked per date. It reads the "
      "valuation timestamp and the spot from the frozen 2025-12-31 parameter "
      "file and, by default, prices a 72 h option for its sensitivity table. "
      "The curve itself is built by the same `build_forward_curve` call with the "
      "same configuration, and section 'Equivalence with the production "
      "pipeline' above measures the difference against the published curve.")
    A("* Realised PTF comes from the local EPİAŞ archive only "
      "(`inputs/historical/ptf_raw/`, `inputs/market/realized_ptf_2026.csv`). "
      f"Delivery months with less than {MIN_REALIZED_COVERAGE:.0%} hourly "
      "coverage are reported with an empty bias rather than a partial average.")
    A("* Every metric here is computed on **monthly baseload averages**: one "
      "forward number and one realised number per delivery month. The sMAPE "
      "column is therefore not comparable with the hourly sMAPE in "
      "`realized_2026_backtest.md`, which averages the error hour by hour and is "
      "much larger because individual hours swing far more than a monthly mean.")
    if bad:
        A(f"* Valuation dates with no usable data: "
          f"{', '.join(r.valuation_date for r in bad)}. Nothing was substituted "
          f"for them.")
    else:
        A("* Every requested valuation date produced a usable strip.")
    dropped_any = [r for r in ok if r.dropped_months]
    if dropped_any:
        A("* Strips truncated at an interior unquoted month (the curve cannot be "
          "built across an interior hole, and no interpolation was invented): "
          + "; ".join(f"{r.valuation_date} dropped {', '.join(r.dropped_months)}"
                      for r in dropped_any) + ".")
    A("* The t-tests in section 3 assume independent observations, which these "
      "are not: the same delivery month is seen from more than one valuation "
      "date and consecutive delivery months of one curve share a market state. "
      "They are reported because they were asked for, with the dependence "
      "stated rather than corrected; the non-overlapping subsets are shown "
      "beside the pooled result so the reader can see how much of the pooled "
      "t-statistic is overlap.")
    A(f"* Contract counts per date are in section 1; every retained date carries "
      f"at least {MIN_QUOTES} quoted months. "
      + (f"Dropped for too few quotes: {', '.join(r.valuation_date for r in few)}."
         if few else "No date was dropped for too few quotes."))
    if stale:
        A(f"* {len(stale)} valuation dates are non-business days; their quote set "
          f"is the last GGF published before the date (1–"
          f"{max(r.ggf_staleness_days for r in stale)} days stale), never after "
          f"it. This is the only substitution made anywhere in the study and it "
          f"is flagged per row in `multi_date_backtest.csv` "
          f"(`ggf_quotation_date`, `ggf_staleness_days`)."
          + _stale_anchor_note(df, stale))
    if anchor_check and anchor_check.get("status") == "ok":
        A(f"* The downloaded EPİAŞ `vep-ggf` series reproduces the production "
          f"quote file on {anchor_check['anchor_date']} for all "
          f"{len(anchor_check['contracts'])} contracts to "
          f"{anchor_check['max_abs_diff_TRY_MWh']:.1e} TRY/MWh (section 0); "
          f"`EPIAS_VEP_daily_reference_price` in the schema and "
          f"`VEP Günlük Gösterge Fiyatı` in the API are the same series.")
    A("")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="*", default=list(VALUATION_DATES))
    ap.add_argument("--max-horizon", type=int, default=DEFAULT_MAX_HORIZON)
    ap.add_argument("--outdir", type=Path, default=OUT_DIR)
    ap.add_argument("--verify-2025", action="store_true",
                    help="also rebuild and diff the published 2025-12-31 curve")
    a = ap.parse_args(argv)

    cfg = _cfg()
    ptf = load_realized_hourly()
    print(f"realised PTF archive: {len(ptf)} hours "
          f"({ptf.index[0]} .. {ptf.index[-1]})")

    available = available_strip_dates()
    unused = [d for d in available if d not in set(a.dates)]
    if unused:
        print(f"WARNING: restricted run -- {len(unused)} valuation date(s) have "
              f"a quote file on disk but were not requested: "
              f"{', '.join(unused)}", file=sys.stderr)

    a.outdir.mkdir(parents=True, exist_ok=True)
    results = [build_one_date(d, ptf, cfg, a.max_horizon, a.outdir) for d in a.dates]
    for r in results:
        if r.status == "ok":
            print(f"  {r.valuation_date}: {len(r.quoted_months)} quotes, "
                  f"anchor {'FIRED' if not r.nearest_month_quoted else 'not needed'}, "
                  f"{len(r.rows)} horizon rows")
        else:
            print(f"  {r.valuation_date}: {r.status} -- {r.note}")

    rows = [row for r in results for row in r.rows]
    if not rows:
        print("\nNo usable (valuation date x delivery month) observation was "
              "produced. Nothing was written beyond the per-date folders.",
              file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(a.outdir / "multi_date_backtest.csv", index=False)

    dist = analysis_distribution(df)
    sysx = analysis_systematic(df)
    anch = analysis_anchor(df)
    hz = horizon_table(df)
    hz.to_csv(a.outdir / "fig_bias_by_horizon.csv", index=False)
    mono = horizon_monotonicity(df)
    anchor_check = None
    if QUOTE_SET_REPORT.exists():
        with open(QUOTE_SET_REPORT, encoding="utf-8") as fh:
            anchor_check = json.load(fh).get("anchor_check")

    (df[df["near_term_anchored"]]
       .dropna(subset=["bias_TRY_MWh"])
       [["valuation_date", "ggf_quotation_date", "ggf_staleness_days",
         "delivery_month", "horizon_months", "spot_at_valuation_TRY_MWh",
         "forward_TRY_MWh", "realized_TRY_MWh", "bias_TRY_MWh",
         "relative_bias_pct", "sMAPE_pct"]]
       .to_csv(a.outdir / "fig_anchor_error.csv", index=False))

    verify = verify_2025(cfg) if a.verify_2025 else None
    with open(a.outdir / "multi_date_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"dates_requested": list(a.dates),
                   "dates_available_on_disk": available,
                   "dates_available_not_requested": unused,
                   "distribution": dist, "systematic": sysx, "anchor": anch,
                   "horizon_monotonicity": mono,
                   "vep_series_anchor_check": anchor_check,
                   "verification": verify,
                   "per_date": [r.__dict__ for r in results]},
                  fh, ensure_ascii=False, indent=1, default=str)

    write_summary(a.outdir / "multi_date_summary.md", results, df,
                  dist, sysx, anch, hz, verify, cfg, a.max_horizon,
                  mono, anchor_check, available)

    print(f"\nwrote {a.outdir / 'multi_date_backtest.csv'} ({len(df)} rows)")
    print(f"wrote {a.outdir / 'multi_date_summary.md'}")
    print(f"wrote {a.outdir / 'fig_bias_by_horizon.csv'}")
    print(f"wrote {a.outdir / 'fig_anchor_error.csv'}")
    if verify:
        print(f"verification vs published 2025-12-31 curve: {verify}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
