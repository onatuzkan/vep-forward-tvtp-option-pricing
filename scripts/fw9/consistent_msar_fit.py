"""FW9e §4 -- Consistent MS-AR(1) TVTP MLE on the model-faithful
A3 asinh residual (asinh(P/scale_P) minus month-mean, minus month-
detrended hour-of-week shape).

Expectation: phi lands in the interior (not on the unit-root
boundary that FW9b's raw-asinh fit hit), because the multi-year
trend has been removed.  SEs from the ordinary Hessian route are
then interpretable without profile-mode workarounds.
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


def _model_faithful_asinh() -> tuple:
    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY")
    ts_local = df["ts_utc"].dt.tz_convert("Europe/Istanbul")
    df["hour"] = ts_local.dt.hour
    df["dow"] = ts_local.dt.dayofweek
    df["how"] = df["dow"] * 24 + df["hour"]
    df["month_ts"] = ts_local.dt.to_period("M").dt.to_timestamp()
    # A3 on asinh
    y = df["y"].to_numpy()
    m_mean = df.groupby("month_ts")["y"].transform("mean").to_numpy()
    r1 = y - m_mean
    df["_r1"] = r1
    how_mean = df.groupby("how")["_r1"].transform("mean").to_numpy()
    r3 = r1 - how_mean
    return r3, df["z_lag1"].to_numpy(), df["ts_utc"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y, z, ts = _model_faithful_asinh()
    print(f"model-faithful asinh residual: n={len(y)}, "
          f"mean={y.mean():.4f}, std={y.std():.4f}")

    for tag, tvtp in [("TVTP_1cov_A3", True), ("constant_trans_A3", False)]:
        print(f"\n=== {tag} (tvtp={tvtp}) ===")
        t0 = time.time()
        res = fit_msar(y, z, n_starts=20, seed=20260928, tvtp=tvtp,
                       compute_se=True, compute_opg=False)
        dt = time.time() - t0
        print(f"time {dt:.1f}s  LL {res.loglik:.4f}  "
              f"grad {res.grad_norm:.3g}  iters {res.n_iter}  "
              f"converged {res.converged}")
        print(f"  params: {res.params.as_dict()}")
        print(f"  SE (Hess) 9-slot: {res.se_hessian}")
        with open(OUT / f"{tag}.pkl", "wb") as f:
            pickle.dump(res, f)
        print(f"wrote {tag}.pkl")


if __name__ == "__main__":
    main()
