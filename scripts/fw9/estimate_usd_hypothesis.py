"""FW9b §0a decisive test of the USD-fit hypothesis.

Refit the TVTP_1cov model on ``y = asinh(PTF_USD / 282.48)`` with the
same 20-start settings and the same z(t-1) series.  If the fitted
sigmas land on yaml's (0.0035348070, 0.0924066544), the hypothesis is
proved; otherwise it is killed.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.msar_estimation import fit_msar         # noqa: E402
from scripts.fw9._data import build_fit_frame                 # noqa: E402

YAML_SCALE_P = 282.48
OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="USD")
    y = df["y"].to_numpy()
    z = df["z_lag1"].to_numpy()
    print(f"USD PTF fit: n={len(y)}, y mean={y.mean():.4f} std={y.std():.4f}")
    print("Running 20-start MLE (USD)...")
    t0 = time.time()
    res = fit_msar(y, z, n_starts=20, seed=20260927, tvtp=True,
                   compute_se=True, compute_opg=False)
    print(f"time {time.time()-t0:.1f}s  best log-lik {res.loglik:.4f}  "
          f"grad {res.grad_norm:.3g}  iters {res.n_iter}  "
          f"converged {res.converged}")
    print(f"  sigma_normal = {res.params.sigma_normal:.10f}  "
          f"(yaml = 0.0035348070)")
    print(f"  sigma_stress = {res.params.sigma_stress:.10f}  "
          f"(yaml = 0.0924066544)")
    with open(OUT / "TVTP_1cov_USD.pkl", "wb") as f:
        pickle.dump(res, f)
    print("wrote TVTP_1cov_USD.pkl")


if __name__ == "__main__":
    main()
