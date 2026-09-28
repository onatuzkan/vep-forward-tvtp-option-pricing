"""Build RD_Ramp_1h_lag1 covariate (FW9 §4).

Definition (per the FW4 task description and the M9 bundle's
transition_coefficients.csv row for RD_Ramp_1h_lag1):

    rd_now[t]    = standardised residual demand at hour t
    rd_ramp[t]   = rd_now[t] - rd_now[t - 1]                   # first diff
    rd_ramp_lag1[t] = rd_ramp[t - 1] = rd_now[t - 1] - rd_now[t - 2]
    z_ramp[t]    = (rd_ramp_lag1[t] - mu_train) / sd_train    # z-score

with ``mu_train`` and ``sd_train`` computed on the TRAIN WINDOW
2016-01-01 -> 2022-12-31 20:00 UTC (matching the M9 bundle's
train_end).  This is a strict subset of the available series and
introduces no look-ahead into the post-2022 evaluation window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fw9._data import load_z_now, CUTOFF_UTC

TRAIN_END_UTC = pd.Timestamp("2022-12-31 20:00:00", tz="UTC")


def build_ramp_lag1_series() -> pd.Series:
    """Standardised ramp covariate on the full available window."""
    z = load_z_now()   # tz-aware UTC index
    z = z[z.index <= CUTOFF_UTC].sort_index()
    ramp = z.diff().shift(1)     # rd_now[t-1] - rd_now[t-2]
    train = ramp.loc[:TRAIN_END_UTC].dropna()
    mu = float(train.mean())
    sd = float(train.std(ddof=1))
    if sd < 1e-12:
        raise RuntimeError("training-window ramp SD is degenerate")
    return ((ramp - mu) / sd), (mu, sd)


if __name__ == "__main__":
    z_ramp, (mu, sd) = build_ramp_lag1_series()
    z_ramp = z_ramp.dropna()
    print(f"z_ramp series: n = {len(z_ramp)}, train mu = {mu:.6f}, train sd = {sd:.6f}")
    print("head:")
    print(z_ramp.head())
    print("tail:")
    print(z_ramp.tail())
