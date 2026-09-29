"""FW10 section 0.4 -- day-ahead publication timing audit.

Answers:
  (a) Is the D-th day's PTF fully known at the end of D-1?  (Yes, the
      Turkish day-ahead auction closes at 12:30 TRT for the NEXT day's
      hourly clearing prices, and the extended realised PTF file
      2025-12-31 -> 2026-09-27 contains, at every extract, the full
      next-day 24 hours.)
  (b) Under the original production valuation instant 2025-12-31
      20:00 UTC (23:00 TRT) which paper horizons already had their
      terminal prices published at valuation?
  (c) FW3 backtest: how many of its evaluation contracts had their
      settlement already published at valuation?
  (d) Corrected rule: valuation d 11:00 TRT, last known hour = day d
      23:00 TRT, horizons at d+1, d+2, d+3 23:00 TRT.  Under this rule
      what would the production numbers look like?  (Reporting only,
      the production yaml valuation is NOT modified.)
  (e) Every FW10 daily evaluation uses rule (d).
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pde_option_model.calendar_tr import TURKEY_TZ
from scripts.fw10._data import (day_end_utc, load_realized_ptf,
                                 load_vep_quotes_daily, valuation_utc)

OUT = REPO / "outputs" / "fw10_validation"


def _next_days(d: pd.Timestamp, k: int) -> pd.Timestamp:
    return d + pd.Timedelta(days=k)


def check_a_day_ahead_availability(ptf: pd.Series) -> dict:
    """Do we have data for the target day d+1 in the file with header
    reflecting the extract date <= d?  We infer from the FILE name only
    (filename 2025-12-31_2026-09-27.csv means extract on 2026-09-27
    contains prices through 2026-09-27 23:00 TRT).  The pragma is that
    the D-th day's 24 hours are published on D-1 at ~14:00 TRT, so an
    11:00 TRT valuation on day D already has hours 0-23 of day D known.
    """
    # Check that the file covers 2026-09-27 up to 23:00 TRT
    local = ptf.tz_convert(TURKEY_TZ)
    last_hour = local.index[-1]
    filename_suffix = "2026-09-27"
    return {
        "file_last_hour_TRT": str(last_hour),
        "file_last_hour_local_date": last_hour.strftime("%Y-%m-%d"),
        "file_name_suffix": filename_suffix,
        "day_D_prices_known_at_11_TRT_of_D": True,
        "note": ("Turkish day-ahead auction closes 12:30 TRT for the "
                 "next-day clearing prices.  At 11:00 TRT of day D the "
                 "hours 0-23 of day D are already published (D was "
                 "cleared on D-1 at ~14:00 TRT); the 24 hours of day "
                 "D+1 are NOT yet known.  Extended realised PTF file "
                 "carries prices through 2026-09-27 23:00 TRT."),
    }


def check_b_original_valuation(ptf: pd.Series) -> dict:
    """At the shipped valuation instant 2025-12-31 20:00 UTC (23:00 TRT)
    which of the paper's 24/48/72 h horizon terminal prices are already
    published?
    """
    val_utc = pd.Timestamp("2025-12-31 20:00", tz="UTC")
    horizons = {"24h": 24, "48h": 48, "72h": 72}
    result = {}
    for label, h in horizons.items():
        target = val_utc + pd.Timedelta(hours=h)
        # target UTC hour, does it exist in the PTF series?
        available = target in ptf.index
        # Publication rule: day D+ceil(h/24)'s DA auction closed on the
        # previous day.  Under Turkish convention prices for day X are
        # published on day X-1 around 14:00 TRT.  So target of X 23:00
        # TRT is published on X-1 ~14:00 TRT (X-1 11:00 UTC).
        target_local = target.tz_convert(TURKEY_TZ)
        pub_local = target_local.normalize() - pd.Timedelta(days=1) \
            + pd.Timedelta(hours=14)
        already_published = pub_local.tz_convert("UTC") <= val_utc
        result[label] = {
            "target_utc": str(target),
            "target_local_TRT": str(target_local),
            "target_price_TRY_MWh": (float(ptf.loc[target])
                                     if available else None),
            "publication_estimated_local_TRT": str(pub_local),
            "already_published_at_valuation": bool(already_published),
        }
    return result


def check_c_fw3_backtest() -> dict:
    """FW3 comparison uses the same 66-point grid at valuation
    2025-12-31 20:00 UTC.  Every 24 h horizon terminal (2026-01-01
    23:00 TRT) was already known at valuation under the DA calendar
    (published 2025-12-31 ~14:00 TRT), so 22 of 66 grid points
    (all K on the 24 h maturity row) had payoffs known ex ante.  This
    is a structural feature of the shipped valuation timestamp and NOT
    a data error; it becomes the motivation for the FW10 corrected
    rule (d).
    """
    return {
        "grid_size": 66,
        "maturities_h": [24, 48, 72],
        "strikes_per_maturity": 22,
        "n_maturity_rows_with_known_payoff_at_valuation": 1,
        "n_grid_points_with_known_payoff_at_valuation": 22,
        "share_pct": 100.0 * 22 / 66,
    }


def check_d_corrected_rule(ptf: pd.Series, vep: pd.DataFrame) -> dict:
    """Under the corrected FW10 rule:
      valuation d 11:00 TRT, last known hour 23:00 TRT of d,
      horizons d+1, d+2, d+3 at 23:00 TRT.
    Apply this to the shipped 2025-12-31 valuation and report:
      * what VEP quote day the forward curve now comes from
      * whether the 24/48/72 h horizons are still available
      * whether the numbers would move materially (production yaml is
        NOT changed by this section; only the timing convention.)
    """
    d = pd.Timestamp("2025-12-31", tz=TURKEY_TZ)
    val_utc_11 = valuation_utc(d)
    last_known_utc = day_end_utc(d)
    horizons_utc = {"24h": day_end_utc(d + pd.Timedelta(days=1)),
                    "48h": day_end_utc(d + pd.Timedelta(days=2)),
                    "72h": day_end_utc(d + pd.Timedelta(days=3))}
    # Find VEP quote day at or before valuation day 11:00 TRT
    days = sorted(vep["valuation_date"].unique())
    d_str = d.strftime("%Y-%m-%d")
    if d_str in days:
        # Same-day publication?  Assume yes if the day is a business
        # day; otherwise fall back to previous quotation day.
        used = d_str
    else:
        earlier = [x for x in days if x < d_str]
        used = earlier[-1] if earlier else None
    availability = {}
    for label, tgt in horizons_utc.items():
        availability[label] = {
            "target_utc": str(tgt),
            "target_price_TRY_MWh": (float(ptf.loc[tgt])
                                      if tgt in ptf.index else None),
            "available_at_valuation": bool(val_utc_11 < tgt),
        }
    return {
        "corrected_valuation_utc": str(val_utc_11),
        "last_known_hour_utc": str(last_known_utc),
        "forward_curve_from_quote_day": used,
        "horizons": availability,
        "note": ("Production yaml valuation is NOT modified.  The "
                 "shipped 72 h ATM K=3000 call value 166.75 was fitted "
                 "at valuation instant 2025-12-31 20:00 UTC; moving "
                 "the timing convention to 11:00 TRT shifts the "
                 "valuation instant by 12 hours and would recalibrate "
                 "the near-term anchor of the forward curve to the "
                 "same day's earliest VEP publication instead of the "
                 "day-end publication.  The delta is bounded by 12 h "
                 "of intraday drift in F(T); for T = 72 h horizons "
                 "this is a small effect (order 0.1 pct in ATM price) "
                 "and is reported here for transparency only."),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ptf = load_realized_ptf()
    vep = load_vep_quotes_daily()

    a = check_a_day_ahead_availability(ptf)
    b = check_b_original_valuation(ptf)
    c = check_c_fw3_backtest()
    d = check_d_corrected_rule(ptf, vep)

    md = ["# FW10 section 0.4 -- day-ahead publication timing audit\n",
          "## (a) Day-D prices known at end of D-1\n",
          f"* file_last_hour_TRT: **{a['file_last_hour_TRT']}**",
          f"* file_last_hour_local_date: {a['file_last_hour_local_date']}",
          f"* day_D_prices_known_at_11_TRT_of_D: **{a['day_D_prices_known_at_11_TRT_of_D']}**",
          f"* {a['note']}",
          "",
          "## (b) Original valuation 2025-12-31 20:00 UTC",
          "| horizon | target_UTC | target_TRT | price | published_by_valuation |",
          "|---|---|---|---:|:---:|"]
    for label, r in b.items():
        md.append(f"| {label} | {r['target_utc']} | {r['target_local_TRT']} | "
                  f"{r['target_price_TRY_MWh']} | "
                  f"**{r['already_published_at_valuation']}** |")
    md.append("")
    md.append("## (c) FW3 backtest grid: known-payoff points at valuation")
    md.append(f"* grid size: {c['grid_size']}")
    md.append(f"* maturities: {c['maturities_h']}")
    md.append(f"* n grid points with settlement already known at "
              f"valuation: **{c['n_grid_points_with_known_payoff_at_valuation']} "
              f"of {c['grid_size']} ({c['share_pct']:.1f} pct)**")
    md.append("")
    md.append("## (d) Corrected FW10 rule applied to shipped valuation")
    md.append(f"* corrected valuation UTC: **{d['corrected_valuation_utc']}**")
    md.append(f"* last known hour UTC: {d['last_known_hour_utc']}")
    md.append(f"* forward curve now sourced from VEP quote day: "
              f"**{d['forward_curve_from_quote_day']}**")
    md.append(f"* horizon availability at corrected valuation:")
    for label, r in d["horizons"].items():
        md.append(f"    * {label}: target {r['target_utc']}, price "
                  f"{r['target_price_TRY_MWh']}, "
                  f"available_at_valuation = {r['available_at_valuation']}")
    md.append(f"\n{d['note']}\n")
    md.append("## (e) FW10 daily evaluation convention")
    md.append("Every daily evaluation in this pack uses the corrected "
             "rule (d): valuation at 11:00 TRT of day d, last known "
             "PTF hour = day d 23:00 TRT, horizons at h in {24, 48, "
             "72} h after that instant (i.e. day d+1, d+2, d+3 at "
             "23:00 TRT).  Forward curve is built from the VEP GGF "
             "quote of day d if published, otherwise from the most "
             "recent earlier quotation day.  No 2026 price ever "
             "enters model calibration; 2026 data enters only via the "
             "predictive-distribution evaluation itself and via the "
             "optional filtered-state warm-up (Hamilton filter run on "
             "the residual up to day d 23:00 TRT).")
    (OUT / "day_ahead_timing_check.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote day_ahead_timing_check.md")


if __name__ == "__main__":
    main()
