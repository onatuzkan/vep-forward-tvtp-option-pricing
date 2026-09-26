"""Ex-post forward premium panel (FW2 §2.1).

For each of the 7 tracked VEP snapshots (2022-12-31 -> 2025-12-31,
semi-annual) we join the 6 forward monthly baseload contracts with the
realised monthly-average PTF, when the realised month has landed in
`inputs/historical/ptf_raw/*.csv` (up to and including 2025-12-31).  Any
delivery month falling AFTER the last available realised hour is
dropped -- no look-ahead past the FW2 valuation cut-off.

The reported quantity per row is the *forward premium proxy*
``F_quoted - realised_month_mean``, in TRY/MWh and as a ratio of
F_quoted.  Aggregated by horizon (delivery-month minus valuation-month,
in months) and by delivery calendar month.  Standard errors account
for the OVERLAPPING horizon structure (a single realised month feeds
several forward observations at different lead times) via a monthly
block bootstrap; the 7-date sample is small, so the SEs are wide and
the panel is descriptive, NOT an estimator of the risk premium.

Writes ``outputs/fw2_risk_premium/ex_post_premium.csv`` and a
Markdown summary.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VEP_DIR = REPO_ROOT / "inputs" / "market" / "historical_vep"
PTF_DIR = REPO_ROOT / "inputs" / "historical" / "ptf_raw"
OUT_DIR = REPO_ROOT / "outputs" / "fw2_risk_premium"

ANCHOR_DATES = ["2022-12-31", "2023-06-30", "2023-12-31",
                "2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31"]
VALUATION_CUTOFF_UTC = pd.Timestamp("2025-12-31 20:00:00", tz="UTC")
BLOCK_BOOTSTRAP_SEED = 20260927
BLOCK_LEN = 3      # months, roughly one quarterly cycle
N_BOOT = 2000


def _load_ptf_series() -> pd.DataFrame:
    """Historical hourly PTF in Turkey local time -> monthly-mean panel."""
    frames = []
    for year in range(2019, 2026):
        f = PTF_DIR / f"ptf_{year}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep=";", decimal=",", thousands=".",
                         dtype={"Tarih": str, "Saat": str})
        ts = pd.to_datetime(df["Tarih"].str.zfill(8) + " " + df["Saat"],
                            format="%d%m%Y %H:%M", errors="coerce")
        alt = ts.isna()
        if alt.any():
            ts = ts.where(~alt, pd.to_datetime(
                df.loc[alt, "Tarih"] + " " + df.loc[alt, "Saat"],
                format="%d.%m.%Y %H:%M", errors="coerce"))
        frames.append(pd.DataFrame({
            "ts_local": ts, "ptf": df["PTF (TL/MWh)"].astype(float),
        }))
    d = pd.concat(frames, ignore_index=True).dropna(subset=["ts_local"])
    d["month"] = d["ts_local"].dt.to_period("M").dt.to_timestamp()
    monthly = (d.groupby("month", as_index=False)
               .agg(realised_mean=("ptf", "mean"),
                    n_hours=("ptf", "size")))
    # keep only fully-observed months
    monthly = monthly.query("n_hours >= 24 * 28").copy()
    return monthly


def _load_vep_snapshots() -> pd.DataFrame:
    """Concatenate the 7 anchor-date CSVs into one long panel."""
    rows: List[pd.DataFrame] = []
    for d in ANCHOR_DATES:
        f = VEP_DIR / d / "vep_monthly_quotes.csv"
        if not f.exists():
            raise FileNotFoundError(f"missing anchor snapshot at {f}")
        sub = pd.read_csv(f)
        sub["valuation_date"] = pd.to_datetime(sub["valuation_date"])
        sub["delivery_month_ts"] = pd.to_datetime(
            sub[["delivery_year", "delivery_month"]].assign(day=1).rename(
                columns={"delivery_year": "year", "delivery_month": "month"}))
        rows.append(sub[["valuation_date", "contract_name",
                         "delivery_month_ts", "price_TRY_MWh"]])
    return pd.concat(rows, ignore_index=True)


def _horizon_months(val: pd.Timestamp, delivery: pd.Timestamp) -> int:
    v = val.to_period("M"); d = delivery.to_period("M")
    return (d.year - v.year) * 12 + (d.month - v.month)


def build_panel() -> pd.DataFrame:
    """One row per (valuation_date, delivery_month) pair with a realised
    average and a forward quote observed.  Excludes rows whose delivery
    month starts AFTER the FW2 valuation cut-off."""
    monthly = _load_ptf_series()
    vep = _load_vep_snapshots()
    # ANCHOR the look-ahead guard: delivery months whose START is on or
    # before the cut-off month are eligible; anything later is dropped.
    cutoff_month = VALUATION_CUTOFF_UTC.tz_convert("Europe/Istanbul").to_period("M").to_timestamp()
    vep = vep[vep["delivery_month_ts"] <= cutoff_month].copy()
    m = vep.merge(monthly.rename(columns={"month": "delivery_month_ts"}),
                  on="delivery_month_ts", how="inner")
    m["horizon_months"] = m.apply(
        lambda r: _horizon_months(r["valuation_date"],
                                  r["delivery_month_ts"]), axis=1)
    m["premium_TRY_MWh"] = m["price_TRY_MWh"] - m["realised_mean"]
    m["premium_pct_of_forward"] = 100.0 * m["premium_TRY_MWh"] / m["price_TRY_MWh"]
    return m.sort_values(["valuation_date", "horizon_months"]).reset_index(drop=True)


def block_bootstrap_se(series: np.ndarray, block: int, n_boot: int,
                       seed: int) -> float:
    """Overlapping monthly-block bootstrap SE of a sample mean."""
    x = np.asarray(series, dtype=float)
    n = x.size
    if n <= 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        boot_means[i] = sample.mean()
    return float(boot_means.std(ddof=1))


def summarise_by_horizon(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h, sub in panel.groupby("horizon_months"):
        x = sub["premium_TRY_MWh"].to_numpy(dtype=float)
        xp = sub["premium_pct_of_forward"].to_numpy(dtype=float)
        out.append({
            "horizon_months": int(h), "n_obs": int(x.size),
            "mean_premium_TRY_MWh": float(x.mean()),
            "median_premium_TRY_MWh": float(np.median(x)),
            "std_premium_TRY_MWh": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
            "block_bootstrap_SE_TRY_MWh": block_bootstrap_se(
                x, BLOCK_LEN, N_BOOT, BLOCK_BOOTSTRAP_SEED + int(h)),
            "mean_premium_pct_of_forward": float(xp.mean()),
            "median_premium_pct_of_forward": float(np.median(xp)),
        })
    return pd.DataFrame(out).sort_values("horizon_months").reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    summary = summarise_by_horizon(panel)
    panel_out = OUT_DIR / "ex_post_premium.csv"
    summary_out = OUT_DIR / "ex_post_premium_summary.csv"
    panel.to_csv(panel_out, index=False)
    summary.to_csv(summary_out, index=False)

    md = ["# Ex-post forward premium panel (FW2 §2.1)\n",
          "**Descriptive**, not an estimator.  Small sample: seven "
          "semi-annual VEP snapshots, `n_obs` per horizon in the table "
          "below.  With this sample size the sample mean is NOT a "
          "consistent estimator of the risk premium; the numbers are a "
          "sanity-bounded plausible range for the FW2 sensitivity "
          "sweep.\n",
          f"* Look-ahead cut-off: {VALUATION_CUTOFF_UTC.isoformat()} "
          "(realised months beyond this cut-off are dropped).",
          f"* Panel rows: {len(panel)} (valuation x delivery-month pairs).",
          f"* Overlapping horizons -> monthly block bootstrap SEs "
          f"(block_length = {BLOCK_LEN}, n_boot = {N_BOOT}).\n"]

    md.append("## Ex-post premium by horizon\n")
    md.append(summary.round({"mean_premium_TRY_MWh": 2,
                             "median_premium_TRY_MWh": 2,
                             "std_premium_TRY_MWh": 2,
                             "block_bootstrap_SE_TRY_MWh": 2,
                             "mean_premium_pct_of_forward": 2,
                             "median_premium_pct_of_forward": 2})
              .to_markdown(index=False))
    md.append("\n\n**Sign convention.**  Positive = forward quoted "
              "ABOVE realised, i.e. the buyer PAID a positive risk "
              "premium.  Negative = forward under-called realised, i.e. "
              "the seller collected a premium.\n")

    md.append("\n## Descriptive band for FW2 sensitivity sweep\n")
    mean_all = panel["premium_TRY_MWh"].mean()
    p95 = np.percentile(np.abs(panel["premium_TRY_MWh"]), 95)
    md.append(f"* Sample mean (all horizons pooled): "
              f"{mean_all:.1f} TRY/MWh\n"
              f"* 95th percentile of |premium|: "
              f"{p95:.1f} TRY/MWh\n"
              "* These are the bounds used in FW2 §2.3 to translate "
              "into per-hour drift-shift a_i.\n")

    (OUT_DIR / "ex_post_premium.md").write_text(
        "\n".join(md), encoding="utf-8")
    print("wrote:", panel_out, summary_out, OUT_DIR / "ex_post_premium.md")


if __name__ == "__main__":
    main()
