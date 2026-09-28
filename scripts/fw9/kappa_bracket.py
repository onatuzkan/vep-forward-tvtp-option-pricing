"""FW9c §1 -- Bracket the OU mean-reversion rate kappa as a function
of seasonal-model richness on the (P - F) residual, so the yaml value
kappa = 0.078394/h can be placed on the resulting curve.

Ladder:
    S0:  no seasonality removed
    S1:  hour-of-day (24 dummies)
    S2:  S1 + day-of-week (7 dummies)
    S3:  S2 + month-of-year (12 dummies)
    S4:  S3 + annual + semi-annual Fourier terms (>= 2 harmonics)
    S5:  S4 + hour x day-of-week interaction (168 dummies)

For each spec:  fit AR(1) via OLS on the residual, report phi, SE,
kappa = -ln(phi), half-life, R^2 of the seasonal projection.  Then
reprice ATM K = 3000 at T in {24, 48, 72} h with that kappa (all
other parameters at production), climatology z, 1201 nodes.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption
from pde_option_model.forward_centered import (ForwardCenteredModel,
                                               ResidualGridSettings,
                                               ResidualSpec,
                                               price_forward_centered)
from pde_option_model.forward_curve import build_forward_curve
from pde_option_model.generator import TVTPCoefficients
from pde_option_model.market_data import load_quotes
from pde_option_model.params_frozen import load_frozen_parameters
from scripts.fw12._shared import climatology_z_lagged_fn
from scripts.fw9._data import build_fit_frame

OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"
YAML_PATH = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
YAML_KAPPA = 0.078394


# ------------------------------------------------------------------- residual y
def _y_series() -> pd.DataFrame:
    """y = P - F(t) on the FW9 window.  F(t) is the shipped hourly VEP-
    anchored forward curve; on-window PTF minus forward gives the
    price-space residual the F2.12 refit fits."""
    df = build_fit_frame(scale_P=282.48, currency="TRY")
    hourly_fwd = pd.read_csv(
        REPO_ROOT / "outputs" / "market_calibration_final"
        / "hourly_forward_curve.csv", parse_dates=["time_utc"])
    hourly_fwd["ts_utc"] = pd.to_datetime(hourly_fwd["time_utc"], utc=True)
    # Only the pre-valuation forwards are relevant; the shipped curve
    # starts at the valuation instant, so the FW9 window is BEFORE the
    # curve.  Fallback: use a same-hour-average climatology of PTF as F(t)
    # (documented as an assumption).
    # A cleaner approach: use the empirical monthly-mean PTF as F for
    # each hour's delivery month; this matches the "level absorbs slow
    # dynamics" spirit of the forward-centered model.  Compute F(t) as
    # the delivery-month mean PTF over the training window.
    df2 = df.copy()
    ts_local = df2["ts_utc"].dt.tz_convert("Europe/Istanbul")
    df2["month"] = ts_local.dt.to_period("M").dt.to_timestamp()
    df2["F_t"] = df2.groupby("month")["ptf_TRY_MWh"].transform("mean")
    df2["resid"] = df2["ptf_TRY_MWh"] - df2["F_t"]
    df2["hour"] = ts_local.dt.hour
    df2["dow"] = ts_local.dt.dayofweek
    df2["moy"] = ts_local.dt.month
    df2["how"] = df2["dow"] * 24 + df2["hour"]
    return df2


def _fit_ols_ar1(resid: np.ndarray) -> Dict[str, float]:
    y0 = resid[:-1]; y1 = resid[1:]
    X = np.column_stack([np.ones_like(y0), y0])
    beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
    c, phi = float(beta[0]), float(beta[1])
    eps = y1 - X @ beta
    sigma = float(eps.std(ddof=2))
    cov = np.linalg.inv(X.T @ X) * (sigma ** 2)
    se_phi = float(np.sqrt(cov[1, 1]))
    kappa = -math.log(phi) if 0 < phi < 1 else float("nan")
    half_life = math.log(2.0) / kappa if kappa > 0 else float("nan")
    return {"phi": phi, "se_phi": se_phi, "sigma_eps": sigma,
            "kappa_per_hour": kappa, "half_life_hours": half_life,
            "n": int(y1.size)}


def _r2_of_seasonal(y: np.ndarray, X: np.ndarray) -> float:
    """R^2 of an OLS projection of y on the seasonal design X."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ b
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12) if ss_tot > 0 else 0.0


def _make_dummies(labels: np.ndarray, drop_first: bool = True) -> np.ndarray:
    """Column-per-category (with the first category dropped, so the
    intercept absorbs the reference level)."""
    uniq = np.unique(labels)
    if drop_first:
        uniq = uniq[1:]
    return np.column_stack([(labels == u).astype(float) for u in uniq])


def _fourier(t_hours: np.ndarray, harmonics: int, period_h: float) -> np.ndarray:
    """[sin, cos] pairs for `harmonics` multiples of the base period."""
    cols = []
    for k in range(1, harmonics + 1):
        omega = 2.0 * math.pi * k / period_h
        cols.append(np.sin(omega * t_hours))
        cols.append(np.cos(omega * t_hours))
    return np.column_stack(cols)


def _spec_design(y_df: pd.DataFrame, spec: str) -> tuple:
    """Return (design matrix INCLUDING intercept column, dof of the
    seasonal projection excluding intercept)."""
    n = len(y_df)
    ones = np.ones((n, 1))
    if spec == "S0":
        X = ones
        dof = 0
    elif spec == "S1":
        h = _make_dummies(y_df["hour"].to_numpy())
        X = np.hstack([ones, h])
        dof = h.shape[1]
    elif spec == "S2":
        h = _make_dummies(y_df["hour"].to_numpy())
        d = _make_dummies(y_df["dow"].to_numpy())
        X = np.hstack([ones, h, d])
        dof = h.shape[1] + d.shape[1]
    elif spec == "S3":
        h = _make_dummies(y_df["hour"].to_numpy())
        d = _make_dummies(y_df["dow"].to_numpy())
        m = _make_dummies(y_df["moy"].to_numpy())
        X = np.hstack([ones, h, d, m])
        dof = h.shape[1] + d.shape[1] + m.shape[1]
    elif spec == "S4":
        h = _make_dummies(y_df["hour"].to_numpy())
        d = _make_dummies(y_df["dow"].to_numpy())
        m = _make_dummies(y_df["moy"].to_numpy())
        t_h = (y_df["ts_utc"] - y_df["ts_utc"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        fyr = _fourier(t_h, harmonics=2, period_h=8766.0)   # annual + semi-annual
        X = np.hstack([ones, h, d, m, fyr])
        dof = h.shape[1] + d.shape[1] + m.shape[1] + fyr.shape[1]
    elif spec == "S5":
        how = _make_dummies(y_df["how"].to_numpy())
        m = _make_dummies(y_df["moy"].to_numpy())
        t_h = (y_df["ts_utc"] - y_df["ts_utc"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        fyr = _fourier(t_h, harmonics=2, period_h=8766.0)
        X = np.hstack([ones, how, m, fyr])
        dof = how.shape[1] + m.shape[1] + fyr.shape[1]
    else:
        raise ValueError(spec)
    return X, dof


# --------------------------------------------------------------- price impact
def _price_atm_at_kappa(kappa: float) -> Dict[int, float]:
    """Reprice ATM K=3000, T in {24,48,72}h with the given kappa,
    every other production parameter fixed."""
    yaml_p = load_frozen_parameters(YAML_PATH)
    quotes = load_quotes(REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv")
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    spec = ResidualSpec(kappa_per_hour=kappa,
                        sigma_y=np.array([yaml_p.sigma_y[0],
                                          yaml_p.sigma_y[1]]),
                        scale_P=282.48, regime_means=np.zeros(2),
                        mode="additive")
    model = ForwardCenteredModel(
        curve=curve, spec=spec,
        tvtp=TVTPCoefficients(yaml_p.alpha01, yaml_p.gamma01,
                              yaml_p.alpha10, yaml_p.gamma10),
        pi_filtered=yaml_p.pi_filtered,
        valuation_utc=yaml_p.valuation_utc,
        spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    out = {}
    for T in (24, 48, 72):
        contract = EuropeanOption("call", 3000.0, model.valuation_utc,
                                  model.valuation_utc + pd.Timedelta(hours=int(T)),
                                  r_annual=0.40)
        gs = ResidualGridSettings(n_space_nodes=1201)
        z_fn = climatology_z_lagged_fn(model, contract, gs)
        r = price_forward_centered(model, contract, grid_settings=gs,
                                   z_lagged_fn=z_fn)
        out[T] = float(r.value)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y_df = _y_series()
    print(f"y_series: n = {len(y_df)}, F mean = {y_df['F_t'].mean():.2f}, "
          f"resid mean = {y_df['resid'].mean():.4f}, "
          f"resid std = {y_df['resid'].std():.2f}")

    rows: List[dict] = []
    for spec in ("S0", "S1", "S2", "S3", "S4", "S5"):
        X, dof = _spec_design(y_df, spec)
        # Project resid on X (with intercept), get residual
        y = y_df["resid"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid_seasonal = y - X @ beta
        r2 = _r2_of_seasonal(y, X)
        ar = _fit_ols_ar1(resid_seasonal)
        px = _price_atm_at_kappa(ar["kappa_per_hour"]) if ar["kappa_per_hour"] > 0 else {24: float("nan"), 48: float("nan"), 72: float("nan")}
        rows.append({
            "spec": spec,
            "seasonal_dof": dof,
            "seasonal_R2": r2,
            "ar1_phi": ar["phi"],
            "ar1_se_phi": ar["se_phi"],
            "ar1_kappa_per_hour": ar["kappa_per_hour"],
            "ar1_half_life_hours": ar["half_life_hours"],
            "ar1_sigma_eps": ar["sigma_eps"],
            "atm_call_T24": px[24],
            "atm_call_T48": px[48],
            "atm_call_T72": px[72],
        })
        print(f"  {spec}: dof={dof:3d}  R2={r2:.4f}  phi={ar['phi']:.5f}  "
              f"kappa={ar['kappa_per_hour']:.4f}  HL={ar['half_life_hours']:.2f}h  "
              f"call_T72={px[72]:.3f}")

    # yaml reference row
    px_yaml = _price_atm_at_kappa(YAML_KAPPA)
    rows.append({
        "spec": "YAML",
        "seasonal_dof": None, "seasonal_R2": None,
        "ar1_phi": math.exp(-YAML_KAPPA),
        "ar1_se_phi": None,
        "ar1_kappa_per_hour": YAML_KAPPA,
        "ar1_half_life_hours": math.log(2.0) / YAML_KAPPA,
        "ar1_sigma_eps": None,
        "atm_call_T24": px_yaml[24],
        "atm_call_T48": px_yaml[48],
        "atm_call_T72": px_yaml[72],
    })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "kappa_bracket.csv", index=False)

    # Markdown
    md = ["# FW9c §1 -- Kappa bracket by seasonal-model richness\n",
          "Ladder from S0 (no seasonality removed) to S5 (hour x dow "
          "+ month-of-year + annual/semi-annual Fourier).  For each "
          "spec: project residual = P - F(t) on the seasonal design, "
          "fit AR(1) on the projection residual (OLS), report kappa "
          "and the ATM K=3000 call price with that kappa (all other "
          "parameters at production, climatology z, 1201 nodes).\n",
          "The **YAML** row shows the shipped `kappa_per_hour = "
          f"{YAML_KAPPA}` for reference.\n",
          df_out.round({"seasonal_R2": 4, "ar1_phi": 6, "ar1_se_phi": 6,
                        "ar1_kappa_per_hour": 6, "ar1_half_life_hours": 3,
                        "ar1_sigma_eps": 4,
                        "atm_call_T24": 3, "atm_call_T48": 3,
                        "atm_call_T72": 3}).to_markdown(index=False)]

    kappa_vals = df_out.dropna(subset=["seasonal_dof"])["ar1_kappa_per_hour"].tolist()
    md.append(f"\n## kappa aralığı (S0..S5): [{min(kappa_vals):.5f}, "
              f"{max(kappa_vals):.5f}]  /  yaml = {YAML_KAPPA}")
    if min(kappa_vals) <= YAML_KAPPA <= max(kappa_vals):
        md.append("Yaml kappa aralığın **İÇİNDE**.\n")
    else:
        md.append("Yaml kappa aralığın **DIŞINDA**.\n")

    (OUT / "kappa_bracket.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote kappa_bracket.csv/md")


if __name__ == "__main__":
    main()
