"""FW9d §1 + §3 -- kappa bracket rebuilt on a single defensible
residual definition, with axis-by-axis isolation of the FW9b/FW9c gap.

Two axes drove the 13x kappa discrepancy between FW9b's 0.017/h and
FW9c's 0.21/h:

  * SCALE (asinh vs raw TRY/MWh)
  * ANCHOR (none, monthly mean, hour-of-week climatology)

FW9d isolates them by rerunning the same S0..S5 seasonal ladder on
three residual definitions and reporting all three ranges.  The
"chosen" definition for §3 is the shock-around-anchor residual in
ASINH space with hour-of-week climatology as F, which is the closest
match to the forward-centered production spec (spec-space asinh,
residual centred around a slow anchor).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
YAML_SCALE_P = 282.48


# ------------------------------------------------------------------------
# Residual definitions
# ------------------------------------------------------------------------
def _base_frame() -> pd.DataFrame:
    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY")
    ts_local = df["ts_utc"].dt.tz_convert("Europe/Istanbul")
    df["hour"] = ts_local.dt.hour
    df["dow"] = ts_local.dt.dayofweek
    df["moy"] = ts_local.dt.month
    df["how"] = df["dow"] * 24 + df["hour"]
    df["month_ts"] = ts_local.dt.to_period("M").dt.to_timestamp()
    return df


def _residual(df: pd.DataFrame, kind: str) -> pd.Series:
    """Return a Pandas Series of the residual under a given definition."""
    if kind == "R1_asinh_raw":
        # FW9b §2: raw asinh, no anchor.
        return pd.Series(df["y"].to_numpy(), index=df.index, name="R1")
    if kind == "R2_TRY_minus_monthly":
        # FW9c §1: TRY-space minus within-month mean (moy variance absorbed).
        F = df.groupby("month_ts")["ptf_TRY_MWh"].transform("mean")
        return pd.Series(df["ptf_TRY_MWh"].to_numpy() - F.to_numpy(),
                         index=df.index, name="R2")
    if kind == "R3_asinh_minus_how_clim":
        # FW9d chosen: asinh space minus hour-of-week climatology mean.
        # Anchor absorbs the intraday+weekly cycle without eating monthly
        # drift.  This is a "shock around anchor" in the same asinh scale
        # the shipped model uses.
        y = df["y"].to_numpy()
        how_mean = df.groupby("how")["y"].transform("mean").to_numpy()
        return pd.Series(y - how_mean, index=df.index, name="R3")
    raise ValueError(kind)


# ------------------------------------------------------------------------
# Seasonal design ladder (S0..S5)  -- same as FW9c §1 but with the S3 fix
# ------------------------------------------------------------------------
def _make_dummies(labels: np.ndarray) -> np.ndarray:
    uniq = np.unique(labels)[1:]     # drop reference to avoid rank issues
    return np.column_stack([(labels == u).astype(float) for u in uniq])


def _fourier(t_h: np.ndarray, harmonics: int, period_h: float) -> np.ndarray:
    cols = []
    for k in range(1, harmonics + 1):
        omega = 2.0 * math.pi * k / period_h
        cols.append(np.sin(omega * t_h))
        cols.append(np.cos(omega * t_h))
    return np.column_stack(cols)


def _spec_design(df: pd.DataFrame, spec: str) -> Tuple[np.ndarray, int]:
    n = len(df)
    ones = np.ones((n, 1))
    if spec == "S0":
        return ones, 0
    h = _make_dummies(df["hour"].to_numpy())
    if spec == "S1":
        return np.hstack([ones, h]), h.shape[1]
    d = _make_dummies(df["dow"].to_numpy())
    if spec == "S2":
        return np.hstack([ones, h, d]), h.shape[1] + d.shape[1]
    m = _make_dummies(df["moy"].to_numpy())
    if spec == "S3":
        return np.hstack([ones, h, d, m]), h.shape[1] + d.shape[1] + m.shape[1]
    t_h = (df["ts_utc"] - df["ts_utc"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    fyr = _fourier(t_h, harmonics=2, period_h=8766.0)
    if spec == "S4":
        return (np.hstack([ones, h, d, m, fyr]),
                h.shape[1] + d.shape[1] + m.shape[1] + fyr.shape[1])
    if spec == "S5":
        how = _make_dummies(df["how"].to_numpy())
        return (np.hstack([ones, how, m, fyr]),
                how.shape[1] + m.shape[1] + fyr.shape[1])
    raise ValueError(spec)


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


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ b
    return 1.0 - ((y - y_hat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(),
                                                1e-12)


# ------------------------------------------------------------------------
# Price impact
# ------------------------------------------------------------------------
def _price_atm_at_kappa(kappa: float, maturities=(24, 48, 72)) -> Dict[int, float]:
    yaml_p = load_frozen_parameters(YAML_PATH)
    quotes = load_quotes(REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv")
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    spec = ResidualSpec(kappa_per_hour=kappa,
                        sigma_y=np.array([yaml_p.sigma_y[0],
                                          yaml_p.sigma_y[1]]),
                        scale_P=YAML_SCALE_P, regime_means=np.zeros(2),
                        mode="additive")
    model = ForwardCenteredModel(
        curve=curve, spec=spec,
        tvtp=TVTPCoefficients(yaml_p.alpha01, yaml_p.gamma01,
                              yaml_p.alpha10, yaml_p.gamma10),
        pi_filtered=yaml_p.pi_filtered,
        valuation_utc=yaml_p.valuation_utc,
        spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    out = {}
    for T in maturities:
        contract = EuropeanOption("call", 3000.0, model.valuation_utc,
                                  model.valuation_utc + pd.Timedelta(hours=int(T)),
                                  r_annual=0.40)
        gs = ResidualGridSettings(n_space_nodes=1201)
        z_fn = climatology_z_lagged_fn(model, contract, gs)
        r = price_forward_centered(model, contract, grid_settings=gs,
                                   z_lagged_fn=z_fn)
        out[T] = float(r.value)
    return out


# ------------------------------------------------------------------------
# Full ladder x residual matrix
# ------------------------------------------------------------------------
def run_ladder(df: pd.DataFrame, resid_kind: str) -> pd.DataFrame:
    resid = _residual(df, resid_kind).to_numpy()
    rows = []
    for spec in ("S0", "S1", "S2", "S3", "S4", "S5"):
        X, dof = _spec_design(df, spec)
        b, *_ = np.linalg.lstsq(X, resid, rcond=None)
        r_after = resid - X @ b
        r2 = _r2(resid, X)
        ar = _fit_ols_ar1(r_after)
        rows.append({
            "residual_definition": resid_kind,
            "spec": spec,
            "seasonal_dof": dof,
            "seasonal_R2": r2,
            "phi": ar["phi"], "se_phi": ar["se_phi"],
            "kappa_per_hour": ar["kappa_per_hour"],
            "half_life_hours": ar["half_life_hours"],
            "sigma_eps": ar["sigma_eps"],
            "n_obs": ar["n"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = _base_frame()
    print(f"n = {len(df)}, y mean = {df['y'].mean():.4f}, "
          f"y std = {df['y'].std():.4f}")

    tables = []
    for kind in ("R1_asinh_raw", "R2_TRY_minus_monthly",
                 "R3_asinh_minus_how_clim"):
        print(f"\n=== residual = {kind} ===")
        t = run_ladder(df, kind)
        # Print quick summary
        for _, r in t.iterrows():
            print(f"  {r['spec']}: dof={int(r['seasonal_dof']):3d}  "
                  f"R2={r['seasonal_R2']:.4f}  phi={r['phi']:.5f}  "
                  f"kappa={r['kappa_per_hour']:.4f}  HL={r['half_life_hours']:.2f}h")
        tables.append(t)

    combined = pd.concat(tables, ignore_index=True)
    # Diagnose S2/S3 for the monthly-mean residual: sum of resid per month is zero
    R2_df = combined[combined["residual_definition"] == "R2_TRY_minus_monthly"]
    print("\nS2/S3 fingerprint (R2 residual): identical because within-month "
          "sum is 0 -> moy dummies span the null space:")
    print(R2_df[R2_df["spec"].isin(["S2", "S3"])][
        ["spec", "kappa_per_hour", "half_life_hours"]].to_string(index=False))

    # kappa range and yaml placement per residual
    summary_rows = []
    for kind in combined["residual_definition"].unique():
        sub = combined[combined["residual_definition"] == kind]
        kmin = float(sub["kappa_per_hour"].min())
        kmax = float(sub["kappa_per_hour"].max())
        summary_rows.append({
            "residual_definition": kind,
            "kappa_min": kmin, "kappa_max": kmax,
            "yaml_kappa": YAML_KAPPA,
            "yaml_inside": bool(kmin <= YAML_KAPPA <= kmax),
        })
    summary = pd.DataFrame(summary_rows)
    print("\n=== summary ===")
    print(summary.to_string(index=False))

    combined.to_csv(OUT / "kappa_bracket_v2.csv", index=False)
    summary.to_csv(OUT / "kappa_bracket_v2_summary.csv", index=False)

    # Markdown report
    md = ["# FW9d §1 + §3 -- kappa bracket rebuilt across residual definitions\n",
          "Three residual definitions, six seasonal specifications (S0..S5)"
          " each; total 18 (residual, spec) pairs.  This makes the FW9b vs"
          " FW9c gap explicit: the same seasonal ladder gives very different"
          " kappa depending on the residual.\n",
          "## Residual definitions\n",
          "* **R1_asinh_raw**: y = asinh(P/282.48).  No anchor subtraction. "
          "  Matches FW9b §2 (the raw-asinh single-regime AR that gave"
          " 0.017/h).",
          "* **R2_TRY_minus_monthly**: P - mean(P over calendar month). "
          " TRY-space, monthly anchor.  Matches FW9c §1 (the 0.21/h fit).",
          "* **R3_asinh_minus_how_clim**: asinh(P/282.48) minus hour-of-week"
          " climatology mean over the training window.  The"
          " 'shock-around-anchor' residual on the same asinh scale the"
          " shipped forward-centered model uses -- the FW9d preferred"
          " definition for reproducing the yaml phi.\n",
          "## Full ladder\n",
          combined.round({"seasonal_R2": 4, "phi": 5, "se_phi": 5,
                          "kappa_per_hour": 4, "half_life_hours": 3,
                          "sigma_eps": 4}).to_markdown(index=False),
          "\n\n## kappa range per residual definition (with yaml placement)\n",
          summary.round(4).to_markdown(index=False),
          "\n\n## S2/S3 identity on R2 (FW9c anomaly explanation)\n",
          "For R2_TRY_minus_monthly the within-month sum of the residual"
          " is zero by construction (P - monthly-mean subtracts the monthly"
          " mean).  Month-of-year dummies lie in the null space of the"
          " residual, so adding them from S2 to S3 does not change the fit."
          "  This is not a bug in the ladder; it is a specification"
          " consequence of the R2 anchor choice.  R1 and R3, which do NOT"
          " subtract a monthly mean, DO show a non-degenerate S2 -> S3"
          " step.\n",
          "## FW9d picked definition and why\n",
          "R3 is the natural analog of the shipped model's residual in the"
          " same variable and on the same scale as the yaml phi:"
          " the yaml v2 kappa refit comment attributes phi=0.9246 to a"
          " 'shock-around-anchor' single-regime AR(1) on the residual the"
          " forward-centered model actually prices.  R1 has no anchor (so"
          " persistence carries the seasonal cycle); R2 anchors on a"
          " monthly mean (so moy is degenerate).  R3 anchors on hour-of-week"
          " climatology, which is the intraday+weekly cycle that both the"
          " forward curve and the M9 deseasonalisation would remove first.\n"]
    (OUT / "kappa_bracket_v2.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
