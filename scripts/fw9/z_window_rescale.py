"""FW9b §4 -- z standardisation window diagnostic.

Re-standardise the RD covariate on the FW9 estimation window
(2019-01-01 → 2025-12-31 20:00 UTC) rather than the M9 train window
(2016-01-01 → 2022-12-31 20:00 UTC), then refit TVTP_1cov via the
profile procedure at phi = argmax.  Report the shift in
alpha01/alpha10 vs the FW9 fit that used the M9-window-standardised z.

Neither version is picked as "main" -- the shift is a documented
diagnostic.
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

from pde_option_model.msar_estimation import fit_msar_fixed_phi
from scripts.fw9._data import build_fit_frame, load_z_lag1, CUTOFF_UTC

YAML_SCALE_P = 282.48
OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"
FW9_WINDOW_START = pd.Timestamp("2018-12-31 21:00:00", tz="UTC")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Build FW9-window-rescaled z from rd_standardized (contemporaneous),
    # then take its 1-hour lag.
    from scripts.fw9._data import load_z_now
    z_now = load_z_now()
    z_now = z_now[(z_now.index >= FW9_WINDOW_START)
                  & (z_now.index <= CUTOFF_UTC)].sort_index()
    # Reconstruct standardisation-free RD z: undo the M9 standardisation is
    # unavailable (only standardised series ships).  We can only RE-standardise
    # the ALREADY-standardised series to have mean 0, std 1 on the FW9 window,
    # which corresponds to a LOCATION/SCALE SHIFT vs the M9-window scaling.
    fw9_mu = float(z_now.mean())
    fw9_sd = float(z_now.std(ddof=1))
    z_rescaled_now = (z_now - fw9_mu) / fw9_sd
    z_rescaled_lag = z_rescaled_now.shift(1).dropna()

    print(f"FW9-window z rescale: mu={fw9_mu:.6f}, sd={fw9_sd:.6f}")
    print(f"n after lag = {len(z_rescaled_lag)}")

    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY",
                         start_utc=FW9_WINDOW_START,
                         end_utc=CUTOFF_UTC)
    # merge on ts_utc
    merged = df.merge(z_rescaled_lag.rename("z_lag1_fw9").to_frame(),
                      left_on="ts_utc", right_index=True, how="inner")
    y = merged["y"].to_numpy()
    z = merged["z_lag1_fw9"].to_numpy()
    print(f"fit sample: n={len(y)}")

    # Use phi = 0.99999 (near the profile argmax from §1)
    print("Fitting at phi = 0.99999 (5 starts) ...")
    t0 = time.time()
    res = fit_msar_fixed_phi(y, z, fixed_phi=0.99999, n_starts=5,
                             seed=20260927, tvtp=True, compute_se=True)
    print(f"time {time.time()-t0:.1f}s LL {res.loglik:.4f}  "
          f"grad {res.grad_norm:.3g}  iters {res.n_iter}  "
          f"converged {res.converged}")
    print(f"  alpha01 = {res.params.alpha01:.6f}  "
          f"(FW9 M9-window z: -0.678)")
    print(f"  alpha10 = {res.params.alpha10:.6f}  "
          f"(FW9 M9-window z: -1.651)")
    print(f"  gamma01 = {res.params.gamma01:.6f}  "
          f"(FW9 M9-window z: -0.443)")
    print(f"  gamma10 = {res.params.gamma10:.6f}  "
          f"(FW9 M9-window z: +0.176)")
    with open(OUT / "z_window_rescale_fit.pkl", "wb") as f:
        pickle.dump({"res": res, "rescale": {"mu": fw9_mu, "sd": fw9_sd,
                                             "window_start": str(FW9_WINDOW_START),
                                             "window_end": str(CUTOFF_UTC)}}, f)
    print("wrote z_window_rescale_fit.pkl")


if __name__ == "__main__":
    main()
