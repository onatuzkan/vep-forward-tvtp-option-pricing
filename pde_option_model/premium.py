"""Q1 as a forward risk-premium term structure (P <-> Q level bridge).

Why this module exists
----------------------
In the forward-centered model the pricing measure is pinned by
``E^Q[P_t] = F(t)``.  A Q1 drift shift ``a`` on an additive Gaussian residual
changes only the mean (Girsanov leaves the diffusion untouched) and the
centering ODE absorbs that mean, so Q-option prices are invariant to ``a`` up
to an O(a^2) regime-mixing term.  This is not a bug; it is the reason the old
``--risk-premium-a0/a1`` knobs looked inert.

The economically meaningful job of Q1 is therefore to carry the *forward risk
premium* between the two measures:

    E^Q[P_t] = F(t)                         (pricing, unchanged)
    E^P[P_t] = F(t) - pi(t)                 (real-world forecast)

For an OU residual ``dX = -kappa X dt + b(t) dt + sigma dW`` the drift change
that maps the P-mean ``-pi(t)`` onto the Q-mean ``0`` is, in closed form,

    a(t) = pi'(t) + kappa * pi(t)           [TRY/MWh per hour]
    lambda(t) = a(t) / sigma(t)             [market price of risk]

This module provides

* EPİAŞ PTF CSV loading (Turkish number format, local time),
* the realised-premium panel  pi_hat(d, m) = F_VEP(d, m) - realised mean(m),
  with a strict look-ahead guard,
* a Tikhonov-regularised (second-difference + ridge) premium curve over
  months-to-delivery -- the MAP estimator of a Gaussian smoothness prior, the
  same construction as Gupta & Reisinger (2012) eq. (4)-(11),
* the pi(t) -> a(t) map and an exact OU mean integrator used to verify it,
* the regime-dependent ("stress premium") variance uplift of a Markov-modulated
  drift, used to size what Q1 *can* do to option prices.

Nothing here changes the Q-measure pricer; it is additive and optional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

TURKEY_TZ = "Europe/Istanbul"   # fixed UTC+3 since 2016

__all__ = [
    "PremiumCurve",
    "load_epias_ptf_csv",
    "load_epias_ptf_dir",
    "realized_monthly_means",
    "load_vep_history",
    "realized_premium_panel",
    "fit_premium_curve",
    "premium_path_hours",
    "drift_from_premium",
    "ou_mean_with_drift",
    "physical_mean",
    "stress_premium_variance_uplift",
]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def _tr_number(s: pd.Series) -> pd.Series:
    """'2.799,98' -> 2799.98 ; already-numeric columns pass through."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s = s.astype(str).str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_epias_ptf_csv(path: str | Path, column: str = "PTF (TL/MWh)") -> pd.Series:
    """Load one EPİAŞ Şeffaflık PTF export.

    Expected header (semicolon separated, Turkish locale):
        ``Tarih;Saat;PTF (TL/MWh);PTF (USD/MWh);PTF (EUR/MWh)``
    Returns an hourly ``pd.Series`` (TRY/MWh) with a tz-aware UTC index.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = {"Tarih", "Saat", column} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}; "
                         f"found {list(df.columns)}")
    ts = pd.to_datetime(df["Tarih"].str.strip() + " " + df["Saat"].str.strip(),
                        format="%d.%m.%Y %H:%M")
    idx = ts.dt.tz_localize(TURKEY_TZ).dt.tz_convert("UTC")
    out = pd.Series(_tr_number(df[column]).to_numpy(), index=pd.DatetimeIndex(idx),
                    name="ptf_TRY_MWh")
    return out.sort_index()


def load_epias_ptf_dir(folder: str | Path, pattern: str = "*.csv") -> pd.Series:
    """Concatenate every PTF export in ``folder``; duplicates keep the last file."""
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {folder}")
    parts = [load_epias_ptf_csv(f) for f in files]
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def realized_monthly_means(ptf_hourly: pd.Series, min_coverage: float = 0.98) -> pd.DataFrame:
    """Baseload monthly averages on Turkish local delivery months.

    Returns columns ``delivery_year, delivery_month, realized_TRY_MWh,
    n_hours, coverage, delivery_end_utc``; months with less than
    ``min_coverage`` of their delivery hours are dropped.
    """
    if ptf_hourly.index.tz is None:
        raise ValueError("ptf_hourly must have a tz-aware index")
    local = ptf_hourly.tz_convert(TURKEY_TZ)
    g = local.groupby([local.index.year, local.index.month])
    rows = []
    for (y, m), v in g:
        n_exp = pd.Period(year=y, month=m, freq="M").days_in_month * 24
        cov = v.notna().sum() / n_exp
        end_local = (pd.Timestamp(year=y, month=m, day=1, tz=TURKEY_TZ)
                     + pd.offsets.MonthBegin(1))
        rows.append(dict(delivery_year=int(y), delivery_month=int(m),
                         realized_TRY_MWh=float(v.mean()), n_hours=int(v.notna().sum()),
                         coverage=float(cov),
                         delivery_end_utc=end_local.tz_convert("UTC")))
    df = pd.DataFrame(rows)
    return df[df["coverage"] >= min_coverage].reset_index(drop=True)


def load_vep_history(path: str | Path) -> pd.DataFrame:
    """Multi-date VEP monthly baseload history in the repository schema.

    Required columns (see ``inputs/market/vep_quotes_schema.md``):
    ``valuation_date, delivery_year, delivery_month, price_TRY_MWh``.
    ``contract_name`` / ``quote_type`` are optional here.  ``valuation_date``
    is a bare date = end of that Turkish day (23:00 TRT = 20:00 UTC).
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"valuation_date", "delivery_year", "delivery_month", "price_TRY_MWh"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"VEP history missing columns {sorted(missing)}")
    if "quote_type" in df.columns:
        df = df[df["quote_type"].fillna("monthly_baseload") == "monthly_baseload"]
    df = df.copy()
    df["price_TRY_MWh"] = _tr_number(df["price_TRY_MWh"])
    d = pd.to_datetime(df["valuation_date"].astype(str).str[:10])
    df["valuation_utc"] = (d + pd.Timedelta(hours=23)).dt.tz_localize(TURKEY_TZ).dt.tz_convert("UTC")
    df["delivery_year"] = df["delivery_year"].astype(int)
    df["delivery_month"] = df["delivery_month"].astype(int)
    return df.dropna(subset=["price_TRY_MWh"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# realised premium panel (with look-ahead guard)
# ---------------------------------------------------------------------------
def realized_premium_panel(vep: pd.DataFrame, realized: pd.DataFrame,
                           cutoff_utc: Optional[pd.Timestamp] = None,
                           min_tau: int = 1, max_tau: int = 12) -> pd.DataFrame:
    """pi_hat(d, m) = F_VEP(d, m) - realised mean(m), relative and absolute.

    ``tau_months`` = delivery month index minus valuation month index (local).
    Look-ahead guard: when ``cutoff_utc`` is given, only quotes observed at or
    before the cutoff AND delivery months that ended at or before the cutoff
    are used -- i.e. exactly the information available at the cutoff.
    """
    df = vep.merge(realized, on=["delivery_year", "delivery_month"], how="inner")
    if cutoff_utc is not None:
        cutoff_utc = pd.Timestamp(cutoff_utc)
        if cutoff_utc.tz is None:
            cutoff_utc = cutoff_utc.tz_localize("UTC")
        df = df[(df["valuation_utc"] <= cutoff_utc) & (df["delivery_end_utc"] <= cutoff_utc)]
    vl = df["valuation_utc"].dt.tz_convert(TURKEY_TZ)
    df = df.assign(
        tau_months=(df["delivery_year"] - vl.dt.year) * 12 + (df["delivery_month"] - vl.dt.month))
    df = df[(df["tau_months"] >= min_tau) & (df["tau_months"] <= max_tau)].copy()
    df["premium_TRY_MWh"] = df["price_TRY_MWh"] - df["realized_TRY_MWh"]
    df["rel_premium"] = df["premium_TRY_MWh"] / df["price_TRY_MWh"]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# regularised premium curve (MAP / Tikhonov over months-to-delivery)
# ---------------------------------------------------------------------------
@dataclass
class PremiumCurve:
    tau_months: np.ndarray            # 1..max_tau
    rel_premium: np.ndarray           # smoothed pi/F
    raw_mean: np.ndarray              # unsmoothed per-tau mean (NaN if no data)
    se: np.ndarray                    # delivery-month-clustered standard error
    n_deliveries: np.ndarray          # independent delivery months per tau
    smoothness: float
    ridge: float
    label: str = "ESTIMATED"
    notes: list = field(default_factory=list)

    def rel_at(self, tau_months: np.ndarray | float) -> np.ndarray:
        """Relative premium at (fractional) months-to-delivery; flat outside."""
        return np.interp(np.asarray(tau_months, float), self.tau_months, self.rel_premium,
                         left=self.rel_premium[0], right=self.rel_premium[-1])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(dict(tau_months=self.tau_months, rel_premium=self.rel_premium,
                                 raw_mean=self.raw_mean, se=self.se,
                                 n_deliveries=self.n_deliveries))


def fit_premium_curve(panel: pd.DataFrame, max_tau: int = 12,
                      smoothness: float = 5.0, ridge: float = 0.1) -> PremiumCurve:
    """Penalised least squares  min_b sum_tau w_tau (ybar_tau - b_tau)^2
    + smoothness * ||D2 b||^2 + ridge * ||b||^2.

    * Observations of the same delivery month from many valuation dates are
      strongly dependent, so each (tau, delivery month) cell is averaged first
      and each delivery month counts once per tau (clustered SE).
    * ``w_tau = n_deliveries / se_tau^2`` (inverse-variance weights, pooled
      variance where a tau has < 2 deliveries).
    * The ridge term shrinks toward zero premium (the no-premium prior); the
      second-difference term is the discrete H^1-type smoothness prior.
    """
    taus = np.arange(1, max_tau + 1)
    cell = (panel.groupby(["tau_months", "delivery_year", "delivery_month"])["rel_premium"]
            .mean().reset_index())
    stats = cell.groupby("tau_months")["rel_premium"].agg(["mean", "std", "count"])
    raw = stats["mean"].reindex(taus).to_numpy()
    n = stats["count"].reindex(taus).fillna(0).to_numpy()
    sd = stats["std"].reindex(taus).to_numpy()
    pooled = np.nanmean(sd) if np.any(np.isfinite(sd)) else 0.25
    sd = np.where(np.isfinite(sd) & (n >= 2), sd, pooled)
    se = np.where(n > 0, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    w = np.where(n > 0, 1.0 / np.maximum(se, 1e-6) ** 2, 0.0)
    w = w / max(w.max(), 1e-12)                     # scale-free penalty weights
    y = np.nan_to_num(raw)
    # fit only up to the last observed tau; beyond it hold the curve flat
    # (a second-difference prior would otherwise extrapolate the slope)
    k_obs = int(taus[n > 0].max()) if np.any(n > 0) else 1
    k = k_obs
    D2 = np.diff(np.eye(k), n=2, axis=0) if k >= 3 else np.zeros((0, k))
    A = np.diag(w[:k]) + smoothness * D2.T @ D2 + (ridge + 1e-10) * np.eye(k)   # tiny floor keeps A invertible
    b = np.full(taus.size, np.nan)
    b[:k] = np.linalg.solve(A, w[:k] * y[:k])
    b[k:] = b[k - 1]
    notes = []
    if k < taus.size:
        notes.append(f"no data beyond tau={k}; curve held flat for tau {k + 1}-{taus[-1]}")
    thin = taus[:k][n[:k] < 3]
    if thin.size:
        notes.append(f"tau with <3 delivery months: {thin.tolist()} -> prior-dominated")
    return PremiumCurve(tau_months=taus.astype(float), rel_premium=b, raw_mean=raw, se=se,
                        n_deliveries=n.astype(int), smoothness=smoothness, ridge=ridge,
                        notes=notes)


# ---------------------------------------------------------------------------
# pi(t) -> a(t) and the OU mean check
# ---------------------------------------------------------------------------
def premium_path_hours(times_hours: np.ndarray, forward_TRY_MWh: np.ndarray,
                       curve: PremiumCurve, ramp_hours: float = 24.0) -> np.ndarray:
    """Absolute premium path pi(t) [TRY/MWh] on an hourly grid.

    ``tau`` in months = t / 730.  The factor ``1 - exp(-t/ramp)`` enforces
    pi(0) = 0 (the spot is observed, so E^P[P_0] = E^Q[P_0] = spot).
    """
    t = np.asarray(times_hours, float)
    rel = curve.rel_at(np.maximum(t / 730.0, curve.tau_months[0]))
    ramp = 1.0 - np.exp(-t / ramp_hours) if ramp_hours > 0 else np.ones_like(t)
    return rel * np.asarray(forward_TRY_MWh, float) * ramp


def drift_from_premium(times_hours: np.ndarray, pi_TRY: np.ndarray,
                       kappa_per_hour: float) -> np.ndarray:
    """a(t) = pi'(t) + kappa * pi(t)   (Q drift minus P drift, TRY/MWh/h)."""
    t = np.asarray(times_hours, float)
    pi = np.asarray(pi_TRY, float)
    return np.gradient(pi, t) + kappa_per_hour * pi


def ou_mean_with_drift(times_hours: np.ndarray, extra_drift: np.ndarray,
                       kappa_per_hour: float, x0: float = 0.0) -> np.ndarray:
    """Exact integration of  mu' = -kappa mu + c(t)  with c piecewise linear."""
    t = np.asarray(times_hours, float)
    c = np.asarray(extra_drift, float)
    k = float(kappa_per_hour)
    mu = np.empty_like(t)
    mu[0] = x0
    for j in range(t.size - 1):
        h = t[j + 1] - t[j]
        e = np.exp(-k * h)
        c0, c1 = c[j], c[j + 1]
        s = (c1 - c0) / h
        # integral of e^{-k(h-s')} (c0 + s s') ds' over [0, h]
        i0 = c0 * (1 - e) / k
        i1 = s * (h / k - (1 - e) / k ** 2)
        mu[j + 1] = e * mu[j] + i0 + i1
    return mu


def physical_mean(forward_TRY_MWh: np.ndarray, pi_TRY: np.ndarray) -> np.ndarray:
    """E^P[P_t] = F(t) - pi(t).  E^Q[P_t] = F(t) is left untouched."""
    return np.asarray(forward_TRY_MWh, float) - np.asarray(pi_TRY, float)


# ---------------------------------------------------------------------------
# regime-dependent Q1 ("stress premium") -- what Q1 can do to option prices
# ---------------------------------------------------------------------------
def stress_premium_variance_uplift(delta_a: float, kappa_per_hour: float,
                                   q01: float, q10: float) -> float:
    """Stationary variance added by a Markov-modulated drift a_J.

    With a_1 - a_0 = delta_a, stationary probabilities p0 = q10/(q01+q10),
    p1 = q01/(q01+q10) and switching rate q = q01 + q10, the drift process has
    variance p0 p1 delta_a^2 and autocorrelation exp(-q t); filtering it through
    the OU kernel gives

        Var_add = p0 p1 delta_a^2 / ( kappa (kappa + q) ).

    Note the (kappa + q) factor: fast regime switching averages the premium out,
    so the uplift is much smaller than the naive p0 p1 (delta_a/kappa)^2.
    """
    q = q01 + q10
    p0, p1 = q10 / q, q01 / q
    return p0 * p1 * delta_a ** 2 / (kappa_per_hour * (kappa_per_hour + q))
