"""FW9c §3 -- Deflated TVTP profile fit for FW5.

Fits the TVTP_1cov specification on the CPI-deflated PTF series
(base 2025-12, scale_P = 3092.05) with the profile procedure at
phi = 0.99999.  The purpose is FW5: to check whether the regime
sigmas are stable in REAL terms across the 2019-2025 window (which
spans a large TRY depreciation event).
"""
from __future__ import annotations

import math
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
from scripts.data.load_cpi import deflator, load_cpi
from scripts.fw9._data import CUTOFF_UTC, load_ptf_hourly, load_z_lag1

OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"
DEFLATED_SCALE_P = 3092.05           # from scale_P.json (base 2025-12)


def _y_deflated_and_z():
    ptf = load_ptf_hourly()          # nominal TRY
    cpi = load_cpi()
    dfl = deflator(cpi, base_month=pd.Timestamp("2025-12-01"))
    dfl = dfl.set_index("date")["deflator"]
    ptf["month"] = ptf["ts_utc"].dt.tz_convert("Europe/Istanbul").dt.to_period(
        "M").dt.to_timestamp()
    ptf["deflator"] = ptf["month"].map(dfl)
    ptf["ptf_real"] = ptf["ptf_TRY_MWh"] * ptf["deflator"]
    z = load_z_lag1().to_frame("z_lag1")
    z.index = z.index.tz_convert("UTC")
    m = ptf.merge(z, left_on="ts_utc", right_index=True, how="inner")
    m = m[m["ts_utc"] <= CUTOFF_UTC].sort_values("ts_utc").reset_index(drop=True)
    y = np.arcsinh(m["ptf_real"].to_numpy() / DEFLATED_SCALE_P)
    return y, m["z_lag1"].to_numpy(), m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y, z, m = _y_deflated_and_z()
    print(f"Deflated fit sample: n = {len(y)}, "
          f"y mean = {y.mean():.4f}, y std = {y.std():.4f}")
    print("Running profile MLE at phi = 0.99999, 5 starts ...")
    t0 = time.time()
    res = fit_msar_fixed_phi(y, z, fixed_phi=0.99999, n_starts=5,
                             seed=20260928, tvtp=True, compute_se=True)
    dt = time.time() - t0
    print(f"time {dt:.1f}s  LL {res.loglik:.4f}  "
          f"grad {res.grad_norm:.3g}  iters {res.n_iter}  "
          f"converged {res.converged}")
    print("Deflated fit params:")
    for k, v in res.params.as_dict().items():
        print(f"  {k}: {v:.6f}")
    # Compare with nominal profile fit
    with open(OUT / "profile_phi_TVTP_1cov_argmax.pkl", "rb") as f:
        nominal = pickle.load(f)
    nom = nominal["res"]
    print("\nNominal vs deflated (sigmas):")
    print(f"  sigma_normal:  nominal = {nom.params.sigma_normal:.6f}  "
          f"deflated = {res.params.sigma_normal:.6f}  "
          f"ratio = {res.params.sigma_normal / nom.params.sigma_normal:.3f}")
    print(f"  sigma_stress:  nominal = {nom.params.sigma_stress:.6f}  "
          f"deflated = {res.params.sigma_stress:.6f}  "
          f"ratio = {res.params.sigma_stress / nom.params.sigma_stress:.3f}")
    with open(OUT / "TVTP_1cov_deflated.pkl", "wb") as f:
        pickle.dump({"res": res, "scale_P": DEFLATED_SCALE_P,
                     "deflation_base": "2025-12-01"}, f)
    print("wrote TVTP_1cov_deflated.pkl")


if __name__ == "__main__":
    main()
