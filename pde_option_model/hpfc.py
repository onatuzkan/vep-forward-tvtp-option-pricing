"""Hourly price forward curve (HPFC) with recency-weighted shape estimation.

Problem it solves
-----------------
The VEP-calibrated forward curve is smooth inside each delivery month, so the
deterministic intraday pattern (night trough, solar midday dip, evening peak)
ends up in the residual and is treated as random noise (~40 % of the 2026
residual variance).  An HPFC multiplies the monthly level by a normalised
hourly shape:

    F_hpfc(t) = F_month(t) * S(t),    (1/N_m) sum_{t in m} S(t) = 1

so every VEP monthly baseload average is preserved exactly.

Shape model
-----------
The target is the ratio  r_t = P_t / (monthly mean of P)  on Turkish local
delivery months -- scale-free, so TRY inflation drops out.  For each
(hour-of-day h, day type d) cell the ratio is regressed on a seasonal Fourier
basis of the day of year,

    r_t ~ beta_{h,d,0} + sum_{k=1..K} beta_{h,d,k}^c cos(k w) + beta_{h,d,k}^s sin(k w),
    w = 2 pi doy / 365.25,

by weighted least squares with a small ridge.  Day types: 0 = working day,
1 = Saturday, 2 = Sunday or public holiday.

Recency weighting ("ağırlığı son yıllara ver")
----------------------------------------------
Observation weights decay exponentially with age relative to the estimation
cutoff:

    w_t = 2^( -age_t / H ),   age in years,  H = half-life in years.

H -> infinity is the equal-weight fit.  Because the solar build-out keeps
deepening the midday dip, recent years are more informative; H is chosen by
rolling-origin out-of-sample validation (fit on data before year Y, score on
year Y) with `select_half_life`.

Public holidays
---------------
Fixed national holidays plus Ramazan/Kurban Bayramı dates for 2019-2026 are
hard-coded below ([ASSUMED] -- verify against the official calendar; arife
half-days are ignored).  Extra dates can be passed in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

TURKEY_TZ = "Europe/Istanbul"

_FIXED = ("01-01", "04-23", "05-01", "05-19", "07-15", "08-30", "10-29")
# (first day, number of days) -- Ramazan Bayramı (3 days), Kurban Bayramı (4 days)
_RELIGIOUS = {
    2019: [("06-04", 3), ("08-11", 4)],
    2020: [("05-24", 3), ("07-31", 4)],
    2021: [("05-13", 3), ("07-20", 4)],
    2022: [("05-02", 3), ("07-09", 4)],
    2023: [("04-21", 3), ("06-28", 4)],
    2024: [("04-10", 3), ("06-16", 4)],
    2025: [("03-30", 3), ("06-06", 4)],
    2026: [("03-20", 3), ("05-27", 4)],
}

__all__ = ["turkish_holidays", "day_type", "recency_weights", "ShapeModel",
           "fit_shape", "select_half_life", "apply_shape_to_curve",
           "evaluate_against_realized"]


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------
def turkish_holidays(years: Iterable[int], extra: Sequence[str] = ()) -> set:
    """Set of local dates (naive ``pd.Timestamp`` at midnight)."""
    out = set()
    for y in years:
        for f in _FIXED:
            out.add(pd.Timestamp(f"{y}-{f}"))
        for start, n in _RELIGIOUS.get(y, []):
            d0 = pd.Timestamp(f"{y}-{start}")
            out.update(d0 + pd.Timedelta(days=k) for k in range(n))
    out.update(pd.Timestamp(d).normalize() for d in extra)
    return out


def _local_naive(index_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index_utc.tz is None:
        raise ValueError("index must be tz-aware")
    return index_utc.tz_convert(TURKEY_TZ).tz_localize(None)


def day_type(index_local: pd.DatetimeIndex, holidays: Optional[set] = None) -> np.ndarray:
    """0 = working day, 1 = Saturday, 2 = Sunday / public holiday."""
    if holidays is None:
        holidays = turkish_holidays(range(index_local.year.min(), index_local.year.max() + 1))
    dow = index_local.dayofweek
    hol = index_local.normalize().isin(list(holidays))
    return np.where(hol | (dow == 6), 2, np.where(dow == 5, 1, 0)).astype(int)


def recency_weights(index_local: pd.DatetimeIndex, cutoff_local: pd.Timestamp,
                    half_life_years: Optional[float]) -> np.ndarray:
    """w = 2^(-age/H); H=None or inf -> equal weights."""
    if half_life_years is None or not np.isfinite(half_life_years):
        return np.ones(len(index_local))
    age = (cutoff_local - index_local).total_seconds().to_numpy() / (365.25 * 86400.0)
    return np.power(2.0, -np.maximum(age, 0.0) / half_life_years)


def _basis(index_local: pd.DatetimeIndex, K: int) -> np.ndarray:
    w = 2.0 * np.pi * index_local.dayofyear.to_numpy() / 365.25
    cols = [np.ones_like(w)]
    for k in range(1, K + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# shape model
# ---------------------------------------------------------------------------
@dataclass
class ShapeModel:
    coef: np.ndarray                  # (24, 3, 2K+1)
    n_harmonics: int
    half_life_years: Optional[float]
    cutoff_utc: pd.Timestamp
    n_obs: int
    extra_holidays: tuple = ()
    notes: list = field(default_factory=list)

    def raw(self, index_utc: pd.DatetimeIndex) -> np.ndarray:
        loc = _local_naive(index_utc)
        hol = turkish_holidays(range(loc.year.min(), loc.year.max() + 1), self.extra_holidays)
        dt = day_type(loc, hol)
        B = _basis(loc, self.n_harmonics)
        c = self.coef[loc.hour.to_numpy(), dt]            # (n, 2K+1)
        return np.einsum("ij,ij->i", B, c)

    def shape(self, index_utc: pd.DatetimeIndex) -> np.ndarray:
        """Shape normalised to mean 1 within each local delivery month present."""
        raw = self.raw(index_utc)
        ym = _local_naive(index_utc).to_period("M")
        s = pd.Series(raw)
        return (s / s.groupby(np.asarray(ym)).transform("mean")).to_numpy()

    def profile_table(self, year: int = 2026) -> pd.DataFrame:
        """Average normalised shape by month x hour x day type (for plots)."""
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h",
                            tz=TURKEY_TZ).tz_convert("UTC")
        loc = _local_naive(idx)
        df = pd.DataFrame(dict(month=loc.month, hour=loc.hour,
                               day_type=day_type(loc, turkish_holidays([year], self.extra_holidays)),
                               shape=self.shape(idx)))
        return df.groupby(["month", "day_type", "hour"])["shape"].mean().reset_index()


def _ratio_frame(ptf_hourly: pd.Series, min_coverage: float = 0.98,
                 min_month_mean: float = 1.0) -> pd.DataFrame:
    loc = _local_naive(ptf_hourly.index)
    df = pd.DataFrame(dict(P=ptf_hourly.to_numpy()), index=loc)
    df["ym"] = df.index.to_period("M")
    g = df.groupby("ym")["P"]
    df["m_mean"] = g.transform("mean")
    cover = g.transform("count") / (df.index.days_in_month * 24)
    df = df[(cover >= min_coverage) & (df["m_mean"] > min_month_mean)].copy()
    df["r"] = df["P"] / df["m_mean"]
    return df.dropna(subset=["r"])


def fit_shape(ptf_hourly: pd.Series, cutoff_utc: pd.Timestamp,
              half_life_years: Optional[float] = 0.5, n_harmonics: int = 2,
              ridge: float = 1e-6, extra_holidays: Sequence[str] = ()) -> ShapeModel:
    """Weighted-LS shape fit on hours strictly before ``cutoff_utc``.

    Only complete months that END at or before the cutoff are used (a month's
    ratio needs its full monthly mean -- no look-ahead).
    """
    cutoff_utc = pd.Timestamp(cutoff_utc)
    s = ptf_hourly[ptf_hourly.index < cutoff_utc]
    df = _ratio_frame(s)
    if df.empty:
        raise ValueError("no complete months before the cutoff")
    cutoff_local = cutoff_utc.tz_convert(TURKEY_TZ).tz_localize(None)
    hol = turkish_holidays(range(df.index.year.min(), df.index.year.max() + 1), extra_holidays)
    dt = day_type(df.index, hol)
    w = recency_weights(df.index, cutoff_local, half_life_years)
    B = _basis(df.index, n_harmonics)
    hours = df.index.hour.to_numpy()
    y = df["r"].to_numpy()
    p = B.shape[1]
    coef = np.zeros((24, 3, p))
    notes = []
    for h in range(24):
        for d in range(3):
            m = (hours == h) & (dt == d)
            if m.sum() < p + 1:
                coef[h, d, 0] = 1.0
                notes.append(f"cell h={h} d={d}: {m.sum()} obs -> flat")
                continue
            X, ww, yy = B[m], w[m], y[m]
            A = X.T @ (X * ww[:, None]) + ridge * ww.sum() * np.eye(p)
            coef[h, d] = np.linalg.solve(A, X.T @ (ww * yy))
    return ShapeModel(coef=coef, n_harmonics=n_harmonics, half_life_years=half_life_years,
                      cutoff_utc=cutoff_utc, n_obs=int(len(df)),
                      extra_holidays=tuple(extra_holidays), notes=notes)


# ---------------------------------------------------------------------------
# half-life selection (rolling origin)
# ---------------------------------------------------------------------------
def select_half_life(ptf_hourly: pd.Series, test_years: Sequence[int] = (2023, 2024, 2025),
                     half_lives: Sequence[Optional[float]] = (0.15, 0.25, 0.35, 0.5, 0.75,
                                                             1.0, 2.0, None),
                     harmonics: Sequence[int] = (1, 2, 3), tie_tol: float = 0.005):
    """Fit on data before 1 Jan Y, score the ratio RMSE on year Y.

    Returns (results DataFrame, best dict).  Among configurations within
    ``tie_tol`` (relative) of the best mean score the one with the LONGEST
    half-life and FEWEST harmonics is chosen (simplest / most stable).
    """
    rows = []
    full = _ratio_frame(ptf_hourly)
    for K in harmonics:
        for H in half_lives:
            for Y in test_years:
                cut = pd.Timestamp(f"{Y}-01-01", tz=TURKEY_TZ).tz_convert("UTC")
                model = fit_shape(ptf_hourly, cut, H, K)
                te = full[full.index.year == Y]
                if te.empty:
                    continue
                idx = te.index.tz_localize(TURKEY_TZ).tz_convert("UTC")
                S = model.shape(idx)
                rows.append(dict(n_harmonics=K, half_life_years=np.inf if H is None else H,
                                 test_year=Y,
                                 ratio_rmse=float(np.sqrt(np.mean((te["r"].to_numpy() - S) ** 2))),
                                 rmse_TRY=float(np.sqrt(np.mean((te["P"].to_numpy()
                                                                 - te["m_mean"].to_numpy() * S) ** 2))),
                                 flat_rmse_TRY=float(np.sqrt(np.mean((te["P"] - te["m_mean"]) ** 2)))))
    res = pd.DataFrame(rows)
    agg = res.groupby(["n_harmonics", "half_life_years"])["ratio_rmse"].mean().reset_index()
    best_score = agg["ratio_rmse"].min()
    ok = agg[agg["ratio_rmse"] <= best_score * (1 + tie_tol)]
    ok = ok.sort_values(["half_life_years", "n_harmonics"], ascending=[False, True])
    pick = ok.iloc[0]
    H = None if not np.isfinite(pick["half_life_years"]) else float(pick["half_life_years"])
    eq = agg[(agg["n_harmonics"] == pick["n_harmonics"]) & ~np.isfinite(agg["half_life_years"])]
    best = dict(half_life_years=H, n_harmonics=int(pick["n_harmonics"]),
                mean_ratio_rmse=float(pick["ratio_rmse"]), best_score=float(best_score),
                equal_weight_ratio_rmse=float(eq["ratio_rmse"].iloc[0]) if len(eq) else np.nan)
    return res, best


# ---------------------------------------------------------------------------
# apply to the VEP-calibrated curve and evaluate
# ---------------------------------------------------------------------------
def apply_shape_to_curve(curve: pd.DataFrame, model: ShapeModel,
                         time_col: str = "time_utc",
                         fwd_col: str = "hourly_forward_TRY_MWh") -> pd.DataFrame:
    """F_hpfc = F_smooth * S', with S' rescaled so every local month keeps the
    SAME average as the smooth VEP curve (hence the VEP quotes) exactly."""
    out = curve.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out[time_col], utc=True))
    S = model.shape(idx)
    F = out[fwd_col].to_numpy(dtype=float)
    ym = np.asarray(_local_naive(idx).to_period("M"))
    prod = pd.Series(F * S)
    scale = (pd.Series(F).groupby(ym).transform("mean")
             / prod.groupby(ym).transform("mean")).to_numpy()
    out["shape"] = S * scale
    out["hpfc_TRY_MWh"] = F * out["shape"].to_numpy()
    return out


def evaluate_against_realized(curve_hpfc: pd.DataFrame, realized: pd.Series,
                              time_col: str = "time_utc",
                              fwd_col: str = "hourly_forward_TRY_MWh") -> pd.DataFrame:
    """Per-month and overall errors of the smooth curve vs the HPFC.

    ``*_demeaned`` removes each month's mean error (the VEP level miss, which no
    shape can fix) and isolates what the shape is responsible for.
    """
    df = curve_hpfc.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.merge(realized.rename("realized"), left_on=time_col, right_index=True, how="inner")
    df["month"] = np.asarray(_local_naive(pd.DatetimeIndex(df[time_col])).to_period("M").astype(str))
    rows = []

    def _row(name, g):
        e0 = g["realized"] - g[fwd_col]
        e1 = g["realized"] - g["hpfc_TRY_MWh"]
        d0 = e0 - e0.mean()
        d1 = e1 - e1.mean()
        return dict(month=name, n=len(g),
                    rmse_smooth=np.sqrt((e0 ** 2).mean()), rmse_hpfc=np.sqrt((e1 ** 2).mean()),
                    rmse_smooth_demeaned=np.sqrt((d0 ** 2).mean()),
                    rmse_hpfc_demeaned=np.sqrt((d1 ** 2).mean()),
                    var_reduction_pct=100 * (1 - (d1 ** 2).mean() / (d0 ** 2).mean()))

    for m, g in df.groupby("month"):
        if len(g) > 24:
            rows.append(_row(m, g))
    full = df[df["month"].isin([r["month"] for r in rows])]
    e0 = full["realized"] - full[fwd_col]
    e1 = full["realized"] - full["hpfc_TRY_MWh"]
    d0 = e0 - e0.groupby(full["month"]).transform("mean")
    d1 = e1 - e1.groupby(full["month"]).transform("mean")
    rows.append(dict(month="ALL", n=len(full),
                     rmse_smooth=np.sqrt((e0 ** 2).mean()), rmse_hpfc=np.sqrt((e1 ** 2).mean()),
                     rmse_smooth_demeaned=np.sqrt((d0 ** 2).mean()),
                     rmse_hpfc_demeaned=np.sqrt((d1 ** 2).mean()),
                     var_reduction_pct=100 * (1 - (d1 ** 2).mean() / (d0 ** 2).mean())))
    return pd.DataFrame(rows)
