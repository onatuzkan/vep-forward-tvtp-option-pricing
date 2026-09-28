"""FW9b §0c: fit TVTP_1cov on the 2019-2021 sub-window only, where
median|TRY PTF|/282.48 is much closer to 1 than on the full 2019-2025
window.  If sigmas move toward the yaml values, the "window
explanation" for the FW9 sigma gap is quantitatively supported.
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

from pde_option_model.msar_estimation import fit_msar         # noqa: E402
from scripts.fw9._data import build_fit_frame                 # noqa: E402

YAML_SCALE_P = 282.48
OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"
START = pd.Timestamp("2018-12-31 21:00:00", tz="UTC")
END = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")   # exclusive; up to 2021-12-31 23 UTC


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY",
                         start_utc=START, end_utc=END)
    y = df["y"].to_numpy()
    z = df["z_lag1"].to_numpy()
    print(f"2019-2021 sub-window fit: n={len(y)}, "
          f"y mean={y.mean():.4f} std={y.std():.4f}, "
          f"ts_utc window {df['ts_utc'].min()} -> {df['ts_utc'].max()}")
    t0 = time.time()
    res = fit_msar(y, z, n_starts=20, seed=20260927, tvtp=True,
                   compute_se=True, compute_opg=False)
    print(f"time {time.time()-t0:.1f}s  LL {res.loglik:.4f}  "
          f"grad {res.grad_norm:.3g}  iters {res.n_iter}  "
          f"converged {res.converged}")
    print(f"  sigma_normal = {res.params.sigma_normal:.10f}  "
          f"(yaml = 0.0035348070)")
    print(f"  sigma_stress = {res.params.sigma_stress:.10f}  "
          f"(yaml = 0.0924066544)")
    with open(OUT / "TVTP_1cov_2019_2021.pkl", "wb") as f:
        pickle.dump(res, f)
    print("wrote TVTP_1cov_2019_2021.pkl")


if __name__ == "__main__":
    main()
