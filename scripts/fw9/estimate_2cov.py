"""FW9 §4 -- 2-covariate TVTP fit (RD_lag1 + RD_Ramp_1h_lag1)."""
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

from pde_option_model.msar_estimation import fit_msar_2cov     # noqa: E402
from scripts.fw9._data import build_fit_frame, CUTOFF_UTC      # noqa: E402
from scripts.fw9.build_ramp import build_ramp_lag1_series      # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "fw9_self_estimation"
YAML_SCALE_P = 282.48


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_fit_frame(scale_P=YAML_SCALE_P)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    z_ramp, (mu_r, sd_r) = build_ramp_lag1_series()
    ramp_df = z_ramp.dropna().to_frame("ramp_lag1")
    m = df.merge(ramp_df, left_on="ts_utc", right_index=True, how="inner")
    m = m.sort_values("ts_utc").reset_index(drop=True)
    print(f"2cov fit sample: n = {len(m)}, "
          f"train mu = {mu_r:.6f}, train sd = {sd_r:.6f}")
    y = m["y"].to_numpy()
    z = m["z_lag1"].to_numpy()
    r = m["ramp_lag1"].to_numpy()
    print("Running 20-start MLE ...")
    t0 = time.time()
    res, params_dict = fit_msar_2cov(y, z, r, n_starts=20, seed=20260927)
    print(f"time {time.time()-t0:.1f} s, best log-lik {res.loglik:.4f}, "
          f"grad {res.grad_norm:.3g}, iters {res.n_iter}, "
          f"converged {res.converged}")
    print("params:", params_dict)
    with open(OUT_DIR / "TVTP_2cov.pkl", "wb") as f:
        pickle.dump({"res": res, "params_dict": params_dict,
                     "ramp_scaler": {"mu": mu_r, "sd": sd_r},
                     "n_obs": int(len(m))}, f)
    print("wrote TVTP_2cov.pkl")


if __name__ == "__main__":
    main()
