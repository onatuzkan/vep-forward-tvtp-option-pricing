"""FW9b §1: profile the log-likelihood in phi.

Fixes phi on the ladder {0.99, 0.995, 0.999, 0.9999, 0.99999} and
fits the remaining parameters at every node with 15 random starts.
Reports grad norms, iters, and converged flags.  Standard errors are
computed at the argmax profile node ("phi-conditional") from a full
Hessian on the 8 (or 6) free parameters.

Also runs the same procedure for TVTP_2cov and constant_trans if
requested via CLI flag.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.msar_estimation import fit_msar_fixed_phi
from scripts.fw9._data import build_fit_frame

YAML_SCALE_P = 282.48
OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"
PHI_GRID = [0.99, 0.995, 0.999, 0.9999, 0.99999]


def profile(y: np.ndarray, z: np.ndarray, tvtp: bool, tag: str,
            n_starts: int = 15, seed: int = 20260927) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    fits = {}
    for phi in PHI_GRID:
        t0 = time.time()
        res = fit_msar_fixed_phi(y, z, fixed_phi=phi, n_starts=n_starts,
                                 seed=seed, tvtp=tvtp, compute_se=False)
        dt = time.time() - t0
        rows.append({
            "phi": phi, "loglik": res.loglik, "grad_norm": res.grad_norm,
            "n_iter": res.n_iter, "converged": res.converged,
            "sigma_normal": res.params.sigma_normal,
            "sigma_stress": res.params.sigma_stress,
            "mu_normal": res.params.mu_normal,
            "mu_stress": res.params.mu_stress,
            "alpha01": res.params.alpha01,
            "gamma01": res.params.gamma01,
            "alpha10": res.params.alpha10,
            "gamma10": res.params.gamma10,
            "elapsed_s": dt,
        })
        fits[phi] = res
        print(f"  [{tag}] phi={phi}: LL={res.loglik:.4f}  grad={res.grad_norm:.3g}  "
              f"iters={res.n_iter}  converged={res.converged}  ({dt:.1f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"profile_phi_{tag}.csv", index=False)
    # SE at argmax: usually the largest phi (boundary), but pick the
    # profile argmax to be safe.
    best_phi = float(df.loc[df["loglik"].idxmax(), "phi"])
    print(f"  [{tag}] argmax phi = {best_phi};  computing SE ...")
    t0 = time.time()
    res_se = fit_msar_fixed_phi(y, z, fixed_phi=best_phi, n_starts=n_starts,
                                seed=seed, tvtp=tvtp, compute_se=True)
    print(f"  [{tag}] SE fit: {time.time()-t0:.1f}s  "
          f"converged={res_se.converged}  grad={res_se.grad_norm:.3g}")
    with open(OUT / f"profile_phi_{tag}_argmax.pkl", "wb") as f:
        pickle.dump({"best_phi": best_phi, "res": res_se,
                     "profile_table": df,
                     "fits_by_phi": fits}, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["tvtp1cov", "constant", "tvtp2cov", "all"],
                    default="all")
    args = ap.parse_args()

    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY")
    y = df["y"].to_numpy()
    z = df["z_lag1"].to_numpy()
    print(f"Profile input: n={len(y)}, y mean={y.mean():.4f} std={y.std():.4f}")

    if args.only in ("tvtp1cov", "all"):
        print("\n=== Profile TVTP_1cov ===")
        profile(y, z, tvtp=True, tag="TVTP_1cov")
    if args.only in ("constant", "all"):
        print("\n=== Profile constant_trans ===")
        profile(y, z, tvtp=False, tag="constant_trans")


if __name__ == "__main__":
    main()
