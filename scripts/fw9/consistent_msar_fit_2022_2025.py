"""FW9f §3 -- Consistent MS-AR fit on the A3 asinh residual restricted
to the 2022-01-01 -> 2025-12-31 20:00 UTC window.

Purpose: full-window (FW9e) fit pooled 2019-2021 (low nominal price)
and 2022-2025 (high nominal price) hours; the implied stationary
sd_TRY at F=2917.78 is 927, well above the observed 2025 sd_TRY 624.
Restricting to 2022-2025 removes that regime-pooling artefact.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.msar_estimation import fit_msar
from scripts.fw9._data import build_fit_frame

OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"
YAML_SCALE_P = 282.48
WIN_START = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")


def _model_faithful_asinh_2022() -> tuple:
    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df[df["ts_utc"] >= WIN_START].reset_index(drop=True)
    ts_local = df["ts_utc"].dt.tz_convert("Europe/Istanbul")
    df["hour"] = ts_local.dt.hour
    df["dow"] = ts_local.dt.dayofweek
    df["how"] = df["dow"] * 24 + df["hour"]
    df["month_ts"] = ts_local.dt.to_period("M").dt.to_timestamp()
    y = df["y"].to_numpy()
    m_mean = df.groupby("month_ts")["y"].transform("mean").to_numpy()
    r1 = y - m_mean
    df["_r1"] = r1
    how_mean = df.groupby("how")["_r1"].transform("mean").to_numpy()
    r3 = r1 - how_mean
    return r3, df["z_lag1"].to_numpy(), df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y, z, df = _model_faithful_asinh_2022()
    print(f"regime-matched A3 asinh: n={len(y)}, "
          f"mean={y.mean():.4f}, std={y.std():.4f}, "
          f"window {df['ts_utc'].min()} -> {df['ts_utc'].max()}")

    for tag, tvtp in [("TVTP_1cov_A3_2022_2025", True),
                      ("constant_trans_A3_2022_2025", False)]:
        print(f"\n=== {tag} (tvtp={tvtp}) ===")
        t0 = time.time()
        res = fit_msar(y, z, n_starts=20, seed=20260928, tvtp=tvtp,
                       compute_se=True, compute_opg=False)
        dt = time.time() - t0
        print(f"time {dt:.1f}s  LL {res.loglik:.4f}  "
              f"grad {res.grad_norm:.3g}  iters {res.n_iter}  "
              f"converged {res.converged}")
        print(f"  params: {res.params.as_dict()}")
        with open(OUT / f"{tag}.pkl", "wb") as f:
            pickle.dump(res, f)
        print(f"wrote {tag}.pkl")


if __name__ == "__main__":
    main()
