"""Turn raw EPİAŞ VEP GGF downloads into per-valuation-date quote sets.

Input   inputs/market/historical_vep/raw/vep_ggf_<YYYY-MM>.json
        (written verbatim by scripts/data/download_vep_history.py)

Output  inputs/market/historical_vep/vep_history.csv
            every observed monthly-baseload daily index price, in the schema
            pde_option_model.premium.load_vep_history expects
            (valuation_date, delivery_year, delivery_month, price_TRY_MWh, ...)

        inputs/market/historical_vep/<valuation_date>/vep_monthly_quotes.csv
            one quote set per backtest valuation date, in the schema
            pde_option_model.market_data.load_quotes_csv expects, with the spot
            column filled from the local PTF archive (no network needed)

        inputs/market/historical_vep/quote_set_report.json
            what was found, what was missing, and every rule that fired

Nothing is invented.  GGF is published on business days only, so a valuation
date that falls on a weekend or a public holiday (2022-12-31, 2023-06-30 Kurban
Bayramı, 2023-12-31, 2024-06-30) has no row of its own.  For such a date the
quote set is the LAST GGF published ON OR BEFORE the valuation date -- the
information a market participant actually held at that day's end, never a later
publication -- and the quotation date plus its staleness in days are written
into the quote file and the report.  ``--strict-same-day`` disables the
carry-forward and reports those dates as missing instead.  The spot price
attached to each quote set is the LAST hourly PTF of the valuation day itself,
read from inputs/historical/ptf_raw/ and inputs/market/realized_ptf_2026.csv.

Only monthly baseload contracts (EBM<MM><YY>) are kept: those are the only
instruments the forward-curve construction can use as delivery-average
constraints.

Usage:
    python scripts/backtest/build_vep_quote_sets.py
    python scripts/backtest/build_vep_quote_sets.py --dates 2024-12-31 2025-06-30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pde_option_model.calendar_tr import (  # noqa: E402
    TURKEY_TZ, month_label, parse_vep_contract_code)
from pde_option_model.premium import load_epias_ptf_csv  # noqa: E402

RAW_DIR = REPO / "inputs" / "market" / "historical_vep" / "raw"
OUT_DIR = REPO / "inputs" / "market" / "historical_vep"
PTF_DIR = REPO / "inputs" / "historical" / "ptf_raw"
PTF_2026 = REPO / "inputs" / "market" / "realized_ptf_2026.csv"

# The seven valuation dates of the multi-date forward backtest.
VALUATION_DATES: Tuple[str, ...] = (
    "2022-12-31", "2023-06-30", "2023-12-31", "2024-06-30",
    "2024-12-31", "2025-06-30", "2025-12-31",
)

MONTHLY_BASELOAD_PREFIX = "EBM"
QUOTE_TYPE = "monthly_baseload"
SOURCE = "EPIAS_VEP_daily_reference_price"

# HARD STOP.  The production quote file (the 2025-12-31 strip the manuscript is
# built on) must be reproduced by the downloaded GGF series; otherwise the
# downloaded series is a different EPİAŞ product and nothing built from it can
# be compared with the single-date result.  Tolerance in TRY/MWh.
PRODUCTION_QUOTES = REPO / "inputs" / "market" / "vep_monthly_quotes.csv"
ANCHOR_CONTRACT = "EBM0226"
ANCHOR_DATE = "2025-12-31"
ANCHOR_PRICE = 2900.99
ANCHOR_TOL_TRY = 1.0


class AnchorMismatchError(RuntimeError):
    """Raised when the downloaded series does not reproduce the production strip."""

# Candidate response field names.  EPİAŞ has renamed PFM fields before, so the
# detection is explicit and fails loudly rather than guessing silently.
DATE_FIELDS = ("date", "ggfDate", "priceDate", "tarih", "gunlukGostergeFiyatiTarihi")
CONTRACT_FIELDS = ("contractName", "contractCode", "contract", "kontratAdi", "name")
PRICE_FIELDS = ("price", "ggf", "dailyIndexPrice", "indexPrice", "gostergeFiyat",
                "fiyat", "priceTl", "priceTL")


class RawSchemaError(RuntimeError):
    """Raised when the downloaded GGF records do not expose the needed fields."""


# ---------------------------------------------------------------------------
def _load_raw_rows() -> List[Dict[str, Any]]:
    files = sorted(RAW_DIR.glob("vep_ggf_*.json"))
    if not files:
        raise FileNotFoundError(
            f"no raw GGF files in {RAW_DIR}\n"
            "run:  python scripts/data/download_vep_history.py ggf "
            "--start 2022-01 --end 2026-09")
    rows: List[Dict[str, Any]] = []
    for f in files:
        blob = json.load(open(f, encoding="utf-8"))
        rows.extend(blob.get("response", []))
    return rows


def _detect(rows: List[Dict[str, Any]], candidates: Tuple[str, ...],
            what: str) -> str:
    keys = set()
    for r in rows[:500]:
        keys.update(r.keys())
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise RawSchemaError(
        f"cannot find the {what} field in the GGF response.\n"
        f"  tried   : {list(candidates)}\n"
        f"  observed: {sorted(keys)}\n"
        f"Add the correct name to the *_FIELDS tuple in {Path(__file__).name}.")


def parse_ggf(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Raw GGF records -> tidy monthly-baseload frame."""
    if not rows:
        raise RawSchemaError("raw GGF files contain zero records")
    f_date = _detect(rows, DATE_FIELDS, "quotation date")
    f_contract = _detect(rows, CONTRACT_FIELDS, "contract name")
    f_price = _detect(rows, PRICE_FIELDS, "daily index price")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "valuation_date": pd.to_datetime(df[f_date].astype(str).str[:10],
                                         errors="coerce").dt.strftime("%Y-%m-%d"),
        "contract_name": df[f_contract].astype(str).str.strip().str.upper(),
        "price_TRY_MWh": pd.to_numeric(df[f_price], errors="coerce"),
    })
    out["_detected_fields"] = f"{f_date}|{f_contract}|{f_price}"
    out = out[out["contract_name"].str.startswith(MONTHLY_BASELOAD_PREFIX)]
    out = out.dropna(subset=["valuation_date", "price_TRY_MWh"])
    out = out[out["price_TRY_MWh"] > 0]

    ym = out["contract_name"].map(_safe_parse_code)
    out = out[ym.notna()].copy()
    out["delivery_year"] = [v[0] for v in ym[ym.notna()]]
    out["delivery_month"] = [v[1] for v in ym[ym.notna()]]
    out["delivery_month_label"] = [month_label(y, m) for y, m
                                   in zip(out["delivery_year"], out["delivery_month"])]
    out["quote_type"] = QUOTE_TYPE
    out["source"] = SOURCE
    out = (out.sort_values(["valuation_date", "delivery_year", "delivery_month"])
              .drop_duplicates(subset=["valuation_date", "contract_name"], keep="last")
              .reset_index(drop=True))
    return out


def _safe_parse_code(code: str) -> Optional[Tuple[int, int]]:
    try:
        return parse_vep_contract_code(code)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
def anchor_check(hist: pd.DataFrame) -> Dict[str, Any]:
    """Compare the downloaded GGF rows of ANCHOR_DATE with the production strip.

    Returns a report dict; raises :class:`AnchorMismatchError` if the anchor
    contract is absent or any production contract differs by more than
    ANCHOR_TOL_TRY.  The number of matching decimals is reported per contract
    (``None`` when the values are not identical to 12 decimals).
    """
    day = hist[hist["valuation_date"] == ANCHOR_DATE].set_index("contract_name")
    report: Dict[str, Any] = {
        "anchor_date": ANCHOR_DATE, "anchor_contract": ANCHOR_CONTRACT,
        "expected_price_TRY_MWh": ANCHOR_PRICE, "tolerance_TRY_MWh": ANCHOR_TOL_TRY,
        "contracts": {}, "status": "ok",
    }
    if ANCHOR_CONTRACT not in day.index:
        report["status"] = "ANCHOR_CONTRACT_MISSING"
        report["observed_contracts"] = sorted(day.index)
        raise AnchorMismatchError(
            f"{ANCHOR_CONTRACT} not present on {ANCHOR_DATE}; "
            f"observed {sorted(day.index)}")
    expected = {ANCHOR_CONTRACT: ANCHOR_PRICE}
    if PRODUCTION_QUOTES.exists():
        prod = pd.read_csv(PRODUCTION_QUOTES)
        prod = prod[prod["valuation_date"].astype(str) == ANCHOR_DATE]
        expected.update({str(r.contract_name): float(r.price_TRY_MWh)
                         for r in prod.itertuples()})
    worst = 0.0
    for c, exp in expected.items():
        if c not in day.index:
            report["contracts"][c] = {"expected": exp, "downloaded": None,
                                      "abs_diff": None, "matching_decimals": None}
            report["status"] = "PRODUCTION_CONTRACT_MISSING"
            continue
        got = float(day.loc[c, "price_TRY_MWh"])
        diff = abs(got - exp)
        worst = max(worst, diff)
        dec: Optional[int] = None
        for k in range(12, -1, -1):
            if round(got, k) == round(exp, k):
                dec = k
                break
        report["contracts"][c] = {"expected": exp, "downloaded": got,
                                  "abs_diff": diff, "matching_decimals": dec}
    report["max_abs_diff_TRY_MWh"] = worst
    if worst > ANCHOR_TOL_TRY or report["status"] != "ok":
        report["status"] = report["status"] if report["status"] != "ok" else "MISMATCH"
        raise AnchorMismatchError(
            f"downloaded GGF series does not reproduce the production strip on "
            f"{ANCHOR_DATE}: {json.dumps(report['contracts'])}")
    return report


def load_local_ptf() -> pd.Series:
    """Hourly PTF (TRY/MWh, UTC index) from every local EPİAŞ export."""
    parts = [load_epias_ptf_csv(f) for f in sorted(PTF_DIR.glob("ptf_*.csv"))]
    if PTF_2026.exists():
        parts.append(load_epias_ptf_csv(PTF_2026))
    if not parts:
        raise FileNotFoundError(f"no PTF exports under {PTF_DIR}")
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def day_end_spot(ptf: pd.Series, date: str) -> Optional[float]:
    """Last hourly PTF of the Turkish day ``date`` (23:00 TRT)."""
    local = ptf.tz_convert(TURKEY_TZ)
    day = local[local.index.normalize() == pd.Timestamp(date, tz=TURKEY_TZ)]
    if day.empty:
        return None
    return float(day.iloc[-1])


# ---------------------------------------------------------------------------
MAX_STALENESS_DAYS = 7   # carry-forward never reaches further back than this


def asof_quotation_date(hist: pd.DataFrame, date: str,
                        strict: bool) -> Optional[str]:
    """Quotation day whose GGF is 'observed' at ``date``: same day, or the last
    published day before it (at most MAX_STALENESS_DAYS back).  None if absent."""
    days = sorted(hist["valuation_date"].unique())
    if date in days:
        return date
    if strict:
        return None
    earlier = [x for x in days if x < date]
    if not earlier:
        return None
    last = earlier[-1]
    stale = (pd.Timestamp(date) - pd.Timestamp(last)).days
    return last if stale <= MAX_STALENESS_DAYS else None


def write_quote_sets(hist: pd.DataFrame, ptf: pd.Series,
                     dates: Tuple[str, ...], strict_same_day: bool = False
                     ) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "valuation_dates_requested": list(dates),
        "quotation_rule": ("same-day GGF only" if strict_same_day else
                           "last GGF published on or before the valuation date "
                           f"(max {MAX_STALENESS_DAYS} days back)"),
        "detected_fields": (sorted(hist["_detected_fields"].unique())[0]
                            if len(hist) else None),
        "raw_rows_monthly_baseload": int(len(hist)),
        "raw_date_range": [hist["valuation_date"].min(), hist["valuation_date"].max()]
                          if len(hist) else None,
        "per_date": {},
        "missing_dates": [],
    }
    for d in dates:
        qday = asof_quotation_date(hist, d, strict_same_day)
        sub = hist[hist["valuation_date"] == qday] if qday else hist.iloc[0:0]
        stale = (pd.Timestamp(d) - pd.Timestamp(qday)).days if qday else None
        entry: Dict[str, Any] = {
            "n_quotes": int(len(sub)),
            "ggf_quotation_date": qday,
            "staleness_days": stale,
            "valuation_weekday": pd.Timestamp(d).day_name(),
        }
        if sub.empty:
            entry["status"] = "NO_GGF_PUBLISHED"
            entry["note"] = ("no VEP daily index price row on this date"
                             + (" (strict same-day rule)" if strict_same_day else
                                f" or within {MAX_STALENESS_DAYS} days before it")
                             + "; GGF is published on business days only")
            report["missing_dates"].append(d)
            report["per_date"][d] = entry
            continue
        spot = day_end_spot(ptf, d)
        if spot is None:
            entry["status"] = "NO_SPOT_PTF"
            entry["note"] = "no hourly PTF for this Turkish day in the local archive"
            report["missing_dates"].append(d)
            report["per_date"][d] = entry
            continue

        vyear, vmonth = int(d[:4]), int(d[5:7])
        sub = sub.copy()
        sub["tau_months"] = ((sub["delivery_year"] - vyear) * 12
                             + (sub["delivery_month"] - vmonth))
        sub = sub[sub["tau_months"] >= 1].sort_values("tau_months")

        out = pd.DataFrame({
            "contract_name": sub["contract_name"],
            "delivery_year": sub["delivery_year"].astype(int),
            "delivery_month": sub["delivery_month"].astype(int),
            "price_TRY_MWh": sub["price_TRY_MWh"].astype(float),
            "quote_type": QUOTE_TYPE,
            "source": SOURCE,
            "valuation_date": d,
            "weight": 1.0,
            "spot_price_TRY_MWh": "",
            "ggf_quotation_date": qday,
            "ggf_staleness_days": stale,
        })
        if len(out):
            out.iloc[0, out.columns.get_loc("spot_price_TRY_MWh")] = round(spot, 2)
        dest = OUT_DIR / d
        dest.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest / "vep_monthly_quotes.csv", index=False)

        entry.update({
            "status": "ok",
            "n_forward_quotes": int(len(out)),
            "spot_price_TRY_MWh": round(spot, 2),
            "quoted_months": [month_label(y, m) for y, m
                              in zip(out["delivery_year"], out["delivery_month"])],
            "tau_months_quoted": [int(t) for t in sub["tau_months"]],
            "nearest_month_quoted": bool(1 in set(int(t) for t in sub["tau_months"])),
            "file": str((dest / "vep_monthly_quotes.csv").relative_to(REPO)).replace("\\", "/"),
        })
        report["per_date"][d] = entry
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="*", default=list(VALUATION_DATES))
    ap.add_argument("--strict-same-day", action="store_true",
                    help="do not carry the last published GGF forward to a "
                         "non-business valuation date")
    a = ap.parse_args(argv)

    rows = _load_raw_rows()
    hist = parse_ggf(rows)

    print("ANCHOR CHECK (downloaded GGF vs inputs/market/vep_monthly_quotes.csv, "
          f"{ANCHOR_DATE})")
    try:
        anchor = anchor_check(hist)
    except AnchorMismatchError as exc:
        print(f"  FAILED: {exc}")
        print("  HARD STOP -- the downloaded series is not the series the "
              "manuscript is built on. No quote set written.")
        return 2
    for c, v in anchor["contracts"].items():
        print(f"  {c}: expected {v['expected']:.2f}  downloaded {v['downloaded']:.2f}"
              f"  |diff| {v['abs_diff']:.2e}  matching decimals {v['matching_decimals']}")
    print(f"  max |diff| {anchor['max_abs_diff_TRY_MWh']:.2e} TRY/MWh "
          f"(tolerance {ANCHOR_TOL_TRY}) -> PASS")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist.drop(columns=["_detected_fields"]).to_csv(
        OUT_DIR / "vep_history.csv", index=False)
    print(f"vep_history.csv: {len(hist)} monthly-baseload quotes, "
          f"{hist['valuation_date'].nunique()} quotation days "
          f"({hist['valuation_date'].min()} .. {hist['valuation_date'].max()})")

    ptf = load_local_ptf()
    print(f"local PTF archive: {len(ptf)} hours "
          f"({ptf.index[0]} .. {ptf.index[-1]})")

    report = write_quote_sets(hist, ptf, tuple(a.dates), a.strict_same_day)
    report["anchor_check"] = anchor
    with open(OUT_DIR / "quote_set_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    print("\nper valuation date")
    for d, e in report["per_date"].items():
        if e["status"] != "ok":
            print(f"  {d}: {e['status']} -- {e.get('note', '')}")
            continue
        print(f"  {d} ({e['valuation_weekday'][:3]}): {e['n_forward_quotes']} quotes "
              f"{e['quoted_months'][0]}..{e['quoted_months'][-1]}"
              f"  GGF of {e['ggf_quotation_date']} (stale {e['staleness_days']} d)"
              f"  spot {e['spot_price_TRY_MWh']}  "
              f"nearest month quoted: {e['nearest_month_quoted']}")
    if report["missing_dates"]:
        print(f"\nMISSING: {report['missing_dates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
