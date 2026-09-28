"""FW9 §2-§5 orchestrator: MLE of MS-AR(1) TVTP models on the hourly
asinh(PTF) series, plus constant-transition and deflated variants.

Runs four models and pickles the results to
``outputs/fw9_self_estimation/``:

  * ``TVTP_1cov``       -- single covariate z_{t-1} = RD_lag1 (production spec)
  * ``constant_trans``  -- gamma01 = gamma10 = 0 (nested inside TVTP_1cov)
  * ``TVTP_2cov``       -- adds RD_Ramp_1h_lag1 (FW4)
  * ``TVTP_deflated``   -- TVTP_1cov on CPI-deflated y (FW5)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.msar_estimation import (fit_msar, hamilton_loglik,   # noqa: E402
                                              MSARParams, pack)
from scripts.fw9._data import build_fit_frame, CUTOFF_UTC       # noqa: E402
from scripts.data.load_cpi import deflator, load_cpi            # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "fw9_self_estimation"
YAML_SCALE_P = 282.48                       # from m2_frozen_parameters.yaml
N_STARTS = 20
SEED = 20260927


def _pickle(obj, name: str) -> None:
    with open(OUT_DIR / name, "wb") as f:
        pickle.dump(obj, f)


def _run(name: str, y: np.ndarray, z_lag: np.ndarray,
         tvtp: bool = True, seed: int = SEED,
         n_starts: int = N_STARTS) -> None:
    print(f"\n=== {name} (n_obs={y.size}, n_starts={n_starts}, tvtp={tvtp}) ===")
    t0 = time.time()
    res = fit_msar(y, z_lag, n_starts=n_starts, seed=seed, tvtp=tvtp,
                   compute_se=True, compute_opg=False)
    dt = time.time() - t0
    print(f"  fit time {dt:.1f} s -- best log-lik {res.loglik:.4f}, "
          f"grad {res.grad_norm:.3g}, iters {res.n_iter}, "
          f"converged {res.converged}")
    print(f"  params: {res.params.as_dict()}")
    print(f"  SE (Hess): {res.se_hessian}")
    _pickle(res, f"{name}.pkl")


def _deflated_y(scale_P: float) -> tuple:
    """Return (y_deflated, z_lag) with prices CPI-deflated to a base
    month (2025-12 chosen so the December series is at 1.0 real TRY)."""
    df = build_fit_frame(scale_P=None)   # unused: rebuild inline
    from scripts.fw9._data import load_ptf_hourly, load_z_lag1
    ptf = load_ptf_hourly()
    cpi = load_cpi()
    dfl = deflator(cpi, base_month=pd.Timestamp("2025-12-01"))
    dfl = dfl.set_index("date")["deflator"]
    ptf["month"] = ptf["ts_utc"].dt.tz_convert("Europe/Istanbul").dt.to_period(
        "M").dt.to_timestamp()
    ptf["deflator"] = ptf["month"].map(dfl)
    ptf["ptf_real"] = ptf["ptf_TRY_MWh"] * ptf["deflator"]
    z_lag = load_z_lag1().to_frame("z_lag1")
    z_lag.index = z_lag.index.tz_convert("UTC")
    m = ptf.merge(z_lag, left_on="ts_utc", right_index=True, how="inner")
    m = m[m["ts_utc"] <= CUTOFF_UTC].sort_values("ts_utc").reset_index(drop=True)
    y = np.arcsinh(m["ptf_real"].to_numpy() / scale_P)
    return y, m["z_lag1"].to_numpy(), float(m["ptf_real"].abs().median())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- primary spec: y = asinh(PTF / yaml scale_P = 282.48) --------------
    df = build_fit_frame(scale_P=YAML_SCALE_P)
    y = df["y"].to_numpy()
    z = df["z_lag1"].to_numpy()
    print("PRIMARY FIT SETUP  scale_P =", YAML_SCALE_P,
          " y stats mean=%.4f std=%.4f" % (y.mean(), y.std()),
          " n =", len(y))

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["tvtp", "constant", "deflated", "all"],
                    default="all")
    args, _ = ap.parse_known_args()

    if args.only in ("tvtp", "all"):
        _run("TVTP_1cov", y, z, tvtp=True)
    if args.only in ("constant", "all"):
        _run("constant_trans", y, z, tvtp=False)
    if args.only in ("deflated", "all"):
        # -- deflated variant (FW5) ------------------------------------
        y_def, z_def, scale_P_deflated_median = _deflated_y(scale_P=YAML_SCALE_P)
        print(f"\nDEFLATED SETUP (base 2025-12): "
              f"y stats mean={y_def.mean():.4f} std={y_def.std():.4f}, "
              f"scale_P_deflated_median = {scale_P_deflated_median:.2f}")
        _run("TVTP_deflated", y_def, z_def, tvtp=True)

    # write scale_P outputs (FW9 §3)
    if args.only in ("deflated", "all"):
        scale_P_own = float(df["ptf_TRY_MWh"].abs().median())
        with open(OUT_DIR / "scale_P.json", "w", encoding="utf-8") as f:
            json.dump({
                "yaml_scale_P_TRY_MWh": YAML_SCALE_P,
                "fw9_median_abs_PTF_full_window": scale_P_own,
                "fw9_deflated_median_abs_PTF_base_2025_12": scale_P_deflated_median,
                "window_start_utc": str(df["ts_utc"].min()),
                "window_end_utc": str(df["ts_utc"].max()),
                "n_obs": int(len(df)),
            }, f, indent=2)
        print("\nscale_P.json written.")


if __name__ == "__main__":
    main()
