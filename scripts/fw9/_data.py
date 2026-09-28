"""Shared data loading for FW9: hourly PTF joined with lagged z."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PTF_DIR = REPO_ROOT / "inputs" / "historical" / "ptf_raw"
RD_LAG1_CSV = REPO_ROOT / "inputs" / "historical" / "rd_lag1_standardized.csv"
RD_CSV = REPO_ROOT / "inputs" / "historical" / "rd_standardized.csv"
YAML = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"

# Look-ahead cut-off = valuation instant of the accepted calibration.
CUTOFF_UTC = pd.Timestamp("2025-12-31 20:00:00", tz="UTC")


def load_ptf_hourly(currency: str = "TRY") -> pd.DataFrame:
    """Hourly PTF (TRY/MWh) in UTC, from ptf_2019..2025.csv.  No
    look-ahead: rows past CUTOFF_UTC are dropped.  ``currency='TRY'``
    (default) uses the ``PTF (TL/MWh)`` column; ``currency='USD'``
    uses ``PTF (USD/MWh)`` (FW9b §0a decisive test)."""
    col = {"TRY": "PTF (TL/MWh)", "USD": "PTF (USD/MWh)",
           "EUR": "PTF (EUR/MWh)"}[currency]
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
        frames.append(pd.DataFrame({"ts_local": ts,
                                    "ptf_TRY_MWh": df[col].astype(float)}))
    d = pd.concat(frames, ignore_index=True).dropna(subset=["ts_local"])
    d["ts_utc"] = (d["ts_local"].dt.tz_localize("Europe/Istanbul",
                                                ambiguous="infer",
                                                nonexistent="shift_forward")
                   .dt.tz_convert("UTC"))
    d = (d.sort_values("ts_utc").drop_duplicates("ts_utc")
         .reset_index(drop=True))
    d = d[d["ts_utc"] <= CUTOFF_UTC].reset_index(drop=True)
    return d


def load_z_lag1() -> pd.Series:
    """Standardised RD, lag 1 h (production covariate)."""
    z = pd.read_csv(RD_LAG1_CSV, parse_dates=["datetime"])
    z["ts_utc"] = pd.to_datetime(z["datetime"], utc=True)
    return z.set_index("ts_utc")["z"]


def load_z_now() -> pd.Series:
    z = pd.read_csv(RD_CSV, parse_dates=["datetime"])
    z["ts_utc"] = pd.to_datetime(z["datetime"], utc=True)
    return z.set_index("ts_utc")["z"]


def build_fit_frame(scale_P: Optional[float] = None,
                    currency: str = "TRY",
                    start_utc: Optional[pd.Timestamp] = None,
                    end_utc: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """PTF joined with rd_lag1 on the UTC hour, with y = asinh(P/scale_P).

    If ``scale_P is None``, uses the median absolute price of the
    training window as the FW9 estimator (§3.1) -- this is the
    definition the yaml provenance describes.
    """
    ptf = load_ptf_hourly(currency=currency)
    z_lag = load_z_lag1().to_frame("z_lag1")
    z_lag.index = z_lag.index.tz_convert("UTC")
    m = ptf.merge(z_lag, left_on="ts_utc", right_index=True, how="inner")
    end_ = end_utc if end_utc is not None else CUTOFF_UTC
    m = m[m["ts_utc"] <= end_].sort_values("ts_utc").reset_index(drop=True)
    if start_utc is not None:
        m = m[m["ts_utc"] >= start_utc].reset_index(drop=True)
    if scale_P is None:
        scale_P = float(m["ptf_TRY_MWh"].abs().median())
    m["scale_P"] = scale_P
    m["y"] = np.arcsinh(m["ptf_TRY_MWh"].to_numpy() / scale_P)
    return m
