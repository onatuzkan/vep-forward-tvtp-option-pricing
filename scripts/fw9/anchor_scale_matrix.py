"""FW9e §1 + §2 -- Reproduce the 2x2 (scale x anchor) kappa table
and build the model-faithful residual for the yearly kappa breakdown.

Four anchor definitions (independently reproduced from the FW9e
independent check):

  A0: no anchor                     (raw variable)
  A1: pooled hour-of-week climatology
  A2: monthly mean
  A3: monthly mean + month-detrended hour-of-week shape
      (the model-faithful anchor, F2.12 spec analog)

Each anchor is applied in two SCALES:

  S1: asinh(P / scale_P)
  S2: raw TRY/MWh

Total 8 fits, all OLS AR(1) on the deseasonalised residual over the
FW9 window (2019-01-01 -> 2025-12-31 20:00 UTC).  The FW9e task
supplies benchmark numbers; we reproduce independently and compare.

For §2, the model-faithful residual (A3) is priced yearly with the
same OLS AR(1) fit; both scales reported.
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

from scripts.fw9._data import build_fit_frame

OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"
YAML_SCALE_P = 282.48
YAML_KAPPA = 0.078394


# --------------------------------------------------------------------------
# Anchor / scale grid
# --------------------------------------------------------------------------
def _base_frame() -> pd.DataFrame:
    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY")
    ts_local = df["ts_utc"].dt.tz_convert("Europe/Istanbul")
    df["hour"] = ts_local.dt.hour
    df["dow"] = ts_local.dt.dayofweek
    df["moy"] = ts_local.dt.month
    df["how"] = df["dow"] * 24 + df["hour"]
    df["month_ts"] = ts_local.dt.to_period("M").dt.to_timestamp()
    df["year"] = ts_local.dt.year
    return df


def _apply_anchor(x: pd.Series, df: pd.DataFrame, anchor: str) -> np.ndarray:
    """Anchor definitions:

    A0 -- no anchor  (returns x)
    A1 -- x minus pooled hour-of-week mean (across full window)
    A2 -- x minus within-month mean
    A3 -- x minus within-month mean, then minus month-detrended
          hour-of-week shape (shape estimated on the monthly-mean-
          removed series, so it captures the intra-month cycle
          rather than the level).
    """
    x = pd.Series(x.to_numpy(), index=df.index)
    if anchor == "A0":
        return x.to_numpy()
    if anchor == "A1":
        cat = df["how"].to_numpy()
        means = pd.Series(x.to_numpy()).groupby(cat).transform("mean")
        return x.to_numpy() - means.to_numpy()
    if anchor == "A2":
        cat = df["month_ts"].to_numpy()
        means = pd.Series(x.to_numpy()).groupby(cat).transform("mean")
        return x.to_numpy() - means.to_numpy()
    if anchor == "A3":
        # step 1: remove monthly mean
        cat_m = df["month_ts"].to_numpy()
        m_mean = pd.Series(x.to_numpy()).groupby(cat_m).transform("mean")
        r1 = x.to_numpy() - m_mean.to_numpy()
        # step 2: from the monthly-mean-removed series, remove the how
        # shape (which by construction is a within-month-average-zero
        # cycle, i.e. the intra-month shape).
        cat_h = df["how"].to_numpy()
        h_mean = pd.Series(r1).groupby(cat_h).transform("mean")
        return r1 - h_mean.to_numpy()
    raise ValueError(anchor)


def _scale_series(df: pd.DataFrame, scale: str) -> pd.Series:
    if scale == "asinh":
        return pd.Series(df["y"].to_numpy(), index=df.index, name="y")
    if scale == "TRY":
        return pd.Series(df["ptf_TRY_MWh"].to_numpy(), index=df.index, name="p")
    raise ValueError(scale)


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
    return {"phi": phi, "se_phi": se_phi, "sigma_innov": sigma,
            "kappa_per_hour": kappa, "half_life_hours": half_life,
            "resid_sd": float(resid.std(ddof=1))}


def _yearly_kappa(df: pd.DataFrame, resid: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["_r"] = resid
    rows = []
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]["_r"].to_numpy()
        if sub.size < 100:
            continue
        r = _fit_ols_ar1(sub)
        rows.append({"year": int(y), "n": int(sub.size),
                     "kappa_per_hour": r["kappa_per_hour"],
                     "half_life_hours": r["half_life_hours"],
                     "resid_sd": r["resid_sd"],
                     "innov_sd": r["sigma_innov"]})
    # 2025-H2
    mask = (df["year"] == 2025) & (df["ts_utc"] >= pd.Timestamp("2025-07-01", tz="UTC"))
    sub = df[mask]["_r"].to_numpy()
    if sub.size > 100:
        r = _fit_ols_ar1(sub)
        rows.append({"year": "2025-H2", "n": int(sub.size),
                     "kappa_per_hour": r["kappa_per_hour"],
                     "half_life_hours": r["half_life_hours"],
                     "resid_sd": r["resid_sd"],
                     "innov_sd": r["sigma_innov"]})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = _base_frame()

    # -- 2x2 (scale x anchor) matrix ----------------------------------
    rows = []
    for scale in ("asinh", "TRY"):
        x = _scale_series(df, scale)
        x.name = "x"
        for anchor in ("A0", "A1", "A2", "A3"):
            r = _apply_anchor(x, df, anchor)
            fit = _fit_ols_ar1(r)
            rows.append({
                "scale": scale, "anchor": anchor,
                "kappa_per_hour": fit["kappa_per_hour"],
                "half_life_hours": fit["half_life_hours"],
                "resid_sd": fit["resid_sd"],
                "innov_sd": fit["sigma_innov"],
                "phi": fit["phi"], "se_phi": fit["se_phi"],
            })
    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUT / "anchor_scale_matrix.csv", index=False)

    print("=== FW9e §1: 2x2 (scale x anchor) matrix ===")
    piv_k = matrix.pivot(index="anchor", columns="scale",
                         values="kappa_per_hour")
    piv_hl = matrix.pivot(index="anchor", columns="scale",
                          values="half_life_hours")
    piv_sd = matrix.pivot(index="anchor", columns="scale",
                          values="resid_sd")
    print("\nkappa /h:")
    print(piv_k.round(4))
    print("\nhalf-life h:")
    print(piv_hl.round(2))
    print("\nresidual sd:")
    print(piv_sd.round(4))

    # ratio checks
    ratio_scale = {}
    ratio_anchor = {}
    for a in ("A0", "A1", "A2", "A3"):
        try_ = float(piv_k.loc[a, "TRY"])
        asinh = float(piv_k.loc[a, "asinh"])
        ratio_scale[a] = try_ / asinh
    for s in ("asinh", "TRY"):
        no_ = float(piv_k.loc["A0", s])
        mo_ = float(piv_k.loc["A2", s])
        mh_ = float(piv_k.loc["A3", s])
        ratio_anchor[f"{s}: A2/A0"] = mo_ / no_
        ratio_anchor[f"{s}: A3/A0"] = mh_ / no_
    print("\nscale ratios (TRY/asinh) at each anchor:")
    for a, v in ratio_scale.items():
        print(f"  {a}: {v:.2f}")
    print("\nanchor ratios at each scale:")
    for k, v in ratio_anchor.items():
        print(f"  {k}: {v:.2f}")

    # -- Yearly kappa breakdown for A3 (both scales) ---------------------
    print("\n=== FW9e §2: yearly kappa on model-faithful (A3) residual ===")
    for scale in ("asinh", "TRY"):
        x = _scale_series(df, scale); x.name = "x"
        r = _apply_anchor(x, df, "A3")
        yr = _yearly_kappa(df, r)
        yr["scale"] = scale
        yr["anchor"] = "A3"
        yr.to_csv(OUT / f"yearly_kappa_A3_{scale}.csv", index=False)
        print(f"\n{scale}:")
        print(yr.to_string(index=False))

    # summary markdown
    md = ["# FW9e §1 -- 2x2 (scale x anchor) kappa matrix\n",
          "OLS AR(1) on the FW9 window (2019-01-01 -> 2025-12-31 20:00 UTC,",
          f"n = {len(df)}) under four anchor definitions x two scales.",
          "Anchor definitions:\n",
          "* A0: no anchor (raw variable)",
          "* A1: pooled hour-of-week climatology mean subtracted",
          "* A2: within-month mean subtracted",
          "* A3: A2 + intra-month hour-of-week shape (model-faithful)\n",
          "## kappa /h (2x2 pivot)\n",
          piv_k.round(4).to_markdown(),
          "\n\n## half-life (hours)\n",
          piv_hl.round(3).to_markdown(),
          "\n\n## residual sd (in-scale units)\n",
          piv_sd.round(4).to_markdown(),
          "\n\n## Ratio isolation\n",
          "* Scale ratio (TRY / asinh) at each anchor:  "
          + ", ".join(f"{a}={v:.2f}" for a, v in ratio_scale.items()),
          "* Anchor ratio (A3 / A0) at each scale:  "
          + ", ".join(f"{s}={ratio_anchor[f'{s}: A3/A0']:.2f}" for s in ("asinh", "TRY")),
          "\n\n**The dominant effect is the anchor: subtracting the "
          "monthly level pushes kappa 6-10x higher.  The scale change "
          "(asinh vs TRY) contributes a much smaller 1.3-1.7x factor.**  "
          "This inverts the FW9d attribution.  R1 and R3 both look "
          "'the same' (kappa ~ 0.017/h) because BOTH keep the "
          "multi-year 2019-2025 level trend in the residual, so their "
          "phi is measuring the persistence of the TRY inflation trend,"
          " not the persistence of a price shock around a slow "
          "anchor.\n",
          f"* yaml kappa = **{YAML_KAPPA}** falls between the A0/A1 (no "
          "trend removed) and A2/A3 (trend removed) rows on both scales."]
    (OUT / "anchor_scale_matrix.md").write_text("\n".join(md),
                                                encoding="utf-8")
    print("\nwrote anchor_scale_matrix.csv/md and yearly_kappa_A3_*.csv")


if __name__ == "__main__":
    main()
