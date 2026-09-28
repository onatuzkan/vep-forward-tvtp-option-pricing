"""FW9b §2 -- deseasonalised single-regime AR(1) on the same PTF data.

The reference (metadata/deseasonalized_stationarity_summary.csv from the
M9 bundle) reports ``AR_phi = 0.924597, half_life_hours = 8.842`` on
78 905 hourly observations.  The yaml carries phi = 0.9246,
kappa_per_hour = 0.078394 from this fit (per the v2 kappa refit note in
`m2_frozen_parameters.yaml`).

The M9 bundle does NOT ship the actual deseasonalisation pipeline
source; the audit note in `outputs/fw9_self_estimation/preprocessing_audit.md`
lists this as an item not fully reproducible from the repo alone.  This
script applies a standard hour-of-week + month-of-year seasonal dummy
regression to ``y = asinh(PTF/scale_P)`` and fits AR(1) to the residual,
documenting the exact pipeline as an ASSUMPTION.

Outputs:
* deseasonalized_ar1.json (phi, SE, kappa, half-life, n, AIC, BIC)
* deseasonalized_ar1_residuals.csv (t, y, y_deseasonalised, residual)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fw9._data import build_fit_frame

YAML_SCALE_P = 282.48
OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"


def deseasonalise(df: pd.DataFrame) -> pd.DataFrame:
    """Assumed pipeline: y_t = mean(hour-of-week, month-of-year) + eps_t.

    The M9 bundle's actual deseasonalisation is not shipped with the
    repo (see preprocessing_audit.md); this is the closest documentable
    reproduction.  The two effects taken as ADDITIVE and estimated
    LEAST-SQUARES on the full FW9 window (no look-ahead beyond the
    valuation instant).
    """
    df = df.copy()
    ts_local = df["ts_utc"].dt.tz_convert("Europe/Istanbul")
    df["how"] = ts_local.dt.dayofweek * 24 + ts_local.dt.hour
    df["moy"] = ts_local.dt.month
    how_mean = df.groupby("how")["y"].transform("mean")
    moy_mean = df.groupby("moy")["y"].transform("mean")
    grand = df["y"].mean()
    df["y_deseasonalised"] = df["y"] - (how_mean - grand) - (moy_mean - grand)
    df["residual"] = df["y_deseasonalised"] - df["y_deseasonalised"].mean()
    return df


def fit_ar1(y: np.ndarray) -> dict:
    """Simple AR(1) MLE:  y_t = c + phi * y_{t-1} + eps_t, eps ~ N(0, sigma^2).

    Closed-form OLS gives (c_hat, phi_hat); SE of phi from OLS covariance
    of ((y_{t-1}, 1)^T (y_{t-1}, 1))^-1 * sigma^2.
    """
    y0 = y[:-1]
    y1 = y[1:]
    X = np.column_stack([np.ones_like(y0), y0])
    beta, res, rank, sv = np.linalg.lstsq(X, y1, rcond=None)
    c_hat, phi_hat = float(beta[0]), float(beta[1])
    n = int(y1.size)
    eps = y1 - X @ beta
    sigma = float(eps.std(ddof=2))
    cov = np.linalg.inv(X.T @ X) * (sigma ** 2)
    se_phi = float(np.sqrt(cov[1, 1]))
    se_c = float(np.sqrt(cov[0, 0]))
    loglik = float(-0.5 * n * (np.log(2 * np.pi) + 1) - n * np.log(sigma))
    aic = -2 * loglik + 2 * 3
    bic = -2 * loglik + np.log(n) * 3
    return {"phi": phi_hat, "c": c_hat, "sigma_eps": sigma,
            "se_phi": se_phi, "se_c": se_c,
            "n": n, "loglik": loglik, "AIC": aic, "BIC": bic}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY")
    df = deseasonalise(df)
    y_des = df["y_deseasonalised"].to_numpy()
    print(f"y stats: mean={y_des.mean():.6f} std={y_des.std():.6f}")
    result = fit_ar1(y_des)
    kappa = -np.log(result["phi"]) if result["phi"] > 0 else float("nan")
    half_life = np.log(2.0) / kappa if kappa > 0 else float("nan")
    result["kappa_per_hour"] = kappa
    result["half_life_hours"] = half_life
    result["yaml_phi"] = 0.9246
    result["yaml_kappa_per_hour"] = 0.078394
    result["yaml_half_life_hours"] = 8.842
    result["pipeline"] = ("hour-of-week + month-of-year additive seasonal "
                          "dummies on y = asinh(PTF/scale_P); AR(1) on the "
                          "residual (OLS)")
    print(json.dumps(result, indent=2))
    with open(OUT / "deseasonalized_ar1.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
