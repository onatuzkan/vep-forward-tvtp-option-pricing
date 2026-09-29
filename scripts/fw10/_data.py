"""FW10 shared data loaders for the out-of-sample validation.

All data windows are frozen at 2025-12-31 20:00 UTC (23:00 TRT) for
model calibration.  2026 data is used ONLY for evaluation and for
the day-of filtered-state warm-up permitted by the FW10 rules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pde_option_model.calendar_tr import TURKEY_TZ, parse_vep_contract_code

PTF_OLD = REPO / "inputs" / "market" / "realized_ptf_2026.csv"
PTF_NEW = REPO / "inputs" / "market" / "realized_ptf_2025-12-31_2026-09-27.csv"
PTF_HISTORY_DIR = REPO / "inputs" / "historical" / "ptf_raw"
VEP_RAW = REPO / "inputs" / "market" / "historical_vep" / "raw"

# Fixed frozen calibration boundary
FREEZE_UTC = pd.Timestamp("2025-12-31 20:00:00", tz="UTC")
# FW10 evaluation valuation-hour: 11:00 TRT (08:00 UTC)
VAL_HOUR_UTC = 8


# ----------------------------------------------------------------------
# Realised PTF
# ----------------------------------------------------------------------
def load_realized_ptf(prefer_extended: bool = True,
                      include_history: bool = False) -> pd.Series:
    """Concatenated PTF series (TL/MWh, UTC hourly index).

    The extended file (`realized_ptf_2025-12-31_2026-09-27.csv`)
    includes 2025-12-31 and every observed hour up to 2026-09-27.
    Turkish EPIAS CSV format: semicolon-separated, comma decimals.

    If ``include_history=True`` the 2019-2025 archive under
    ``inputs/historical/ptf_raw/`` is also concatenated (needed for
    HPFC shape fitting and historical-vol windows).
    """
    path = PTF_NEW if prefer_extended and PTF_NEW.exists() else PTF_OLD

    def _parse_num(s: str) -> float:
        # "2.799,98" -> 2799.98
        return float(s.replace(".", "").replace(",", "."))

    def _read_epias_csv(p: Path) -> pd.Series:
        df = pd.read_csv(p, sep=";", encoding="utf-8-sig")
        # detect the price column
        price_col = None
        for c in df.columns:
            if "PTF" in c and "TL" in c:
                price_col = c
                break
        if price_col is None:
            raise ValueError(f"cannot find TL price column in {p.name}")
        prices = df[price_col].astype(str).map(_parse_num).to_numpy()
        dates = df["Tarih"].astype(str).to_numpy()
        hours = df["Saat"].astype(str).to_numpy()
        # parse "31.12.2025 00:00" as Europe/Istanbul local time
        stamps = pd.to_datetime(
            [f"{d} {h}" for d, h in zip(dates, hours)],
            format="%d.%m.%Y %H:%M")
        stamps = stamps.tz_localize(TURKEY_TZ, nonexistent="shift_forward",
                                    ambiguous=False)
        s = pd.Series(prices, index=stamps.tz_convert("UTC"), name="ptf_TRY_MWh")
        return s[~s.index.duplicated(keep="last")].sort_index()

    latest = _read_epias_csv(path)
    if not include_history:
        return latest
    parts = [_read_epias_csv(f) for f in sorted(PTF_HISTORY_DIR.glob("ptf_*.csv"))]
    parts.append(latest)
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def day_end_utc(date_local: pd.Timestamp) -> pd.Timestamp:
    """23:00 local TRT of the given local date, returned in UTC."""
    return pd.Timestamp(f"{date_local.date()} 23:00", tz=TURKEY_TZ).tz_convert("UTC")


def valuation_utc(date_local: pd.Timestamp) -> pd.Timestamp:
    """11:00 TRT of the valuation day, in UTC."""
    return pd.Timestamp(f"{date_local.date()} 11:00", tz=TURKEY_TZ).tz_convert("UTC")


# ----------------------------------------------------------------------
# VEP daily quotes
# ----------------------------------------------------------------------
def load_vep_quotes_daily() -> pd.DataFrame:
    """Every observed VEP GGF row (monthly baseload EBM contracts).

    One row per (quotation_day, contract).  Columns:
      valuation_date (str YYYY-MM-DD, Turkish local calendar day),
      contract_name, price_TRY_MWh, delivery_year, delivery_month.
    """
    rows = []
    for f in sorted(VEP_RAW.glob("vep_ggf_*.json")):
        blob = json.loads(f.read_text(encoding="utf-8"))
        rows.extend(blob.get("response", []))
    if not rows:
        raise FileNotFoundError(f"no VEP GGF files under {VEP_RAW}")
    dfr = pd.DataFrame(rows)
    # Turkish date string YYYY-MM-DDT00:00:00+03:00 -> "YYYY-MM-DD"
    dfr["valuation_date"] = pd.to_datetime(
        dfr["date"].astype(str).str[:10]).dt.strftime("%Y-%m-%d")
    dfr["contract_name"] = dfr["contractName"].astype(str).str.strip().str.upper()
    dfr["price_TRY_MWh"] = pd.to_numeric(dfr["price"], errors="coerce")
    dfr = dfr[dfr["contract_name"].str.startswith("EBM")]
    dfr = dfr.dropna(subset=["valuation_date", "price_TRY_MWh"])
    dfr = dfr[dfr["price_TRY_MWh"] > 0].copy()
    ym: List[Optional[Tuple[int, int]]] = []
    for c in dfr["contract_name"]:
        try:
            ym.append(parse_vep_contract_code(c))
        except ValueError:
            ym.append(None)
    keep = [v is not None for v in ym]
    dfr = dfr[keep].copy()
    ym2 = [v for v in ym if v is not None]
    dfr["delivery_year"] = [v[0] for v in ym2]
    dfr["delivery_month"] = [v[1] for v in ym2]
    dfr = (dfr[["valuation_date", "contract_name", "price_TRY_MWh",
                "delivery_year", "delivery_month"]]
           .drop_duplicates(subset=["valuation_date", "contract_name"], keep="last")
           .sort_values(["valuation_date", "delivery_year", "delivery_month"])
           .reset_index(drop=True))
    return dfr


def vep_quote_on_or_before(vep: pd.DataFrame, target_date_str: str) -> Tuple[str, pd.DataFrame]:
    """Return (quote_day_used, dataframe-slice).

    If VEP is published on target_date_str, use that day; otherwise fall
    back to the latest earlier quotation day (weekend/holiday
    carry-forward).  Raises KeyError if no quote is available.
    """
    days = sorted(vep["valuation_date"].unique())
    if target_date_str in days:
        used = target_date_str
    else:
        earlier = [x for x in days if x < target_date_str]
        if not earlier:
            raise KeyError(f"no VEP quote on or before {target_date_str}")
        used = earlier[-1]
    return used, vep[vep["valuation_date"] == used].reset_index(drop=True)


# ----------------------------------------------------------------------
# Evaluation day universe
# ----------------------------------------------------------------------
def business_day_universe(start: str, end: str) -> List[pd.Timestamp]:
    """Turkish business days (Mon-Fri, excluding a few well-known
    holidays) in the closed interval [start, end], TRT calendar."""
    all_days = pd.date_range(start, end, freq="D", tz=TURKEY_TZ)
    # Weekend filter
    all_days = all_days[all_days.dayofweek < 5]
    # Known TR public holidays 2026 (subset that fall on weekdays)
    tr_holidays_2026 = {
        "2026-01-01",   # New Year
        "2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22", "2026-03-23",
        # Ramazan Bayrami approx
        "2026-04-23",   # Ulusal Egemenlik
        "2026-05-01",   # Isci Bayrami
        "2026-05-19",   # Ataturk'u Anma
        "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30", "2026-05-31",
        # Kurban Bayrami approx
        "2026-07-15",   # Demokrasi Bayrami
        "2026-08-30",   # Zafer Bayrami
    }
    keep = [d for d in all_days if d.strftime("%Y-%m-%d") not in tr_holidays_2026]
    return keep
