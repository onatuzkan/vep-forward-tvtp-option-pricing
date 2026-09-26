"""Loader for the TUIK CPI monthly series (TP.GENENDEKS.T1, base 2003=100).

Source file: ``inputs/macro/tuik_cpi_monthly_2016_2026.xlsx`` (EVDS export).
The raw workbook carries the 128 monthly index observations followed by a
few blank rows and then an EVDS "Seri Aciklamalari" + "Notlar" footer with
the series code, base year, and the TUIK data-source URLs.  That footer is
kept in the file on purpose -- the manuscript data appendix cites the
series code and base year from it -- but naive ``pd.read_excel`` will
happily emit the footer rows as data rows.  This loader guards against
that with a strict ``^\\d{4}-\\d{2}$`` regex on the date column, matching
only rows whose ``Tarih`` looks like ``YYYY-MM``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Structural regex.  Any row whose Tarih does NOT match is treated as a
# non-data row and dropped -- covers the trailing blank rows plus the
# EVDS series-description block that starts at "Seri Aciklamalari".
_TARIH_RE = re.compile(r"^\d{4}-\d{2}$")

DEFAULT_PATH = Path("inputs/macro/tuik_cpi_monthly_2016_2026.xlsx")
_SERIES_COL = "TP_GENENDEKS_T1"     # header emitted by EVDS

PROVENANCE = {
    "series_code": "TP.GENENDEKS.T1",
    "series_label": "Genel Endeks (2003=100) - Duzey",
    "base_year": "2003 = 100",
    "source": "TUIK (via EVDS)",
    "provenance_tag": "Inherited",
}


def load_cpi(path: Optional[Path] = None) -> pd.DataFrame:
    """Return a clean CPI panel with columns ``date`` and ``cpi``.

    * ``date``: month-start ``pd.Timestamp`` (tz-naive)
    * ``cpi``:  base 2003=100 index level (float)

    Rows whose ``Tarih`` string does not match ``YYYY-MM`` are dropped;
    the loader raises if the resulting panel is empty, has NaNs, or has
    gaps in the monthly cadence.  Order guaranteed ascending by date.
    """
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"CPI workbook not found at {p}")
    raw = pd.read_excel(p, sheet_name=0, dtype={"Tarih": str,
                                                _SERIES_COL: str})
    mask = raw["Tarih"].astype(str).str.match(_TARIH_RE).fillna(False)
    df = raw.loc[mask, ["Tarih", _SERIES_COL]].copy()
    if df.empty:
        raise ValueError(f"no YYYY-MM data rows found in {p}")
    df["cpi"] = df[_SERIES_COL].astype(float)
    df["date"] = pd.to_datetime(df["Tarih"], format="%Y-%m")
    df = df.sort_values("date").reset_index(drop=True)
    if df["cpi"].isna().any():
        raise ValueError("CPI series contains NaN after filtering")
    # gap check: every consecutive month present
    gaps = df["date"].diff().dropna().unique()
    ok = pd.Timedelta(days=28)
    if any(g < ok or g > pd.Timedelta(days=31) for g in gaps):
        raise ValueError(f"CPI series has non-monthly gaps: {gaps}")
    return df[["date", "cpi"]]


def yoy_december_inflation(cpi: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year December/December inflation (%) per calendar year.

    Uses each year's December index divided by the previous December's
    index, minus 1.  Returned as a small DataFrame with columns
    ``year`` and ``yoy_dec_dec_pct``, one row per year for which both
    Decembers are present in the panel.
    """
    dec = cpi[cpi["date"].dt.month == 12].copy()
    dec["year"] = dec["date"].dt.year
    dec = dec.sort_values("year").reset_index(drop=True)
    dec["prev"] = dec["cpi"].shift(1)
    out = dec.dropna(subset=["prev"]).copy()
    out["yoy_dec_dec_pct"] = 100.0 * (out["cpi"] / out["prev"] - 1.0)
    return out[["year", "yoy_dec_dec_pct"]].reset_index(drop=True)


def deflator(cpi: pd.DataFrame, base_month: pd.Timestamp) -> pd.DataFrame:
    """Return ``deflator = base_cpi / cpi`` for use in real-price rescaling.

    Multiply a nominal TRY/MWh series in month M by ``deflator.loc[M]``
    to express it in constant-purchasing-power TRY/MWh of ``base_month``.
    """
    base_ts = pd.Timestamp(base_month).to_period("M").to_timestamp()
    if base_ts not in set(cpi["date"]):
        raise ValueError(f"base month {base_ts:%Y-%m} not in CPI panel")
    base_cpi = float(cpi.loc[cpi["date"] == base_ts, "cpi"].iloc[0])
    out = cpi.copy()
    out["deflator"] = base_cpi / out["cpi"]
    return out[["date", "cpi", "deflator"]]


if __name__ == "__main__":     # pragma: no cover - manual smoke test
    import json
    d = load_cpi()
    print("first:", d.iloc[0].to_dict())
    print("last:", d.iloc[-1].to_dict())
    print("n =", len(d))
    print("YoY Dec/Dec:")
    print(yoy_december_inflation(d).to_string(index=False))
    print(json.dumps(PROVENANCE, indent=2))
