"""Residual dynamics v2: re-estimated around the HPFC on the right variable.

Why v1 was 5-10x too wide
-------------------------
v1 inherited kappa (half-life ~19 y) and sigma from an AR fit on RAW hourly
asinh(PTF) that also absorbed the deterministic intraday swing, so every hour
was treated as a random-walk shock.  v2 estimates the dynamics on

    x_t = ( P_t - M_m * S_t ) / L_m

* ``S_t``  ex-ante HPFC shape (fitted each year only on data BEFORE that year),
* ``M_m``  realised monthly mean (removes the monthly level -- the level risk
          is a separate factor, see below),
* ``L_m``  ex-ante price scale: trailing 12-month mean PTF before month m.
          Using the trailing level instead of M_m keeps x stable when the
          monthly level collapses (spring 2026: sd/M_m jumps 0.15 -> 1.70,
          sd/L_m only 0.16 -> 0.47).

Model (three independent Gaussian layers, price clipped to [floor, cap])
-----------------------------------------------------------------------
    P_t = clip( F_hpfc(t) + lev_m(t) + L * (u_t + d_t),  floor, cap(t) )

1. fast factor u_t -- 2-regime Markov-switching AR(1), hourly:
       u_t = c_{s_t} + phi u_{t-1} + sigma_{s_t} eta_t
       P(0->1) = logistic(a01 + g01 z_{t-1} + h01 ramp_{t-1})
       P(1->0) = logistic(a10 + g10 z_{t-1} + h10 ramp_{t-1})
   z = standardised residual demand, ramp = standardised dz (both covariates of
   the original M9 spec; the ramp is no longer dropped).  Estimated by exact
   Hamilton-filter MLE; alpha and gamma are estimated jointly (no derived
   intercepts) with standard errors from a numerical Hessian.
2. slow factor d_t -- AR(1) on the daily mean of x (step = 1 day).
3. level factor lev_m -- monthly level uncertainty of the forward, Gaussian with
   sd = sigma_L(tau) * F, tau = months to delivery.  sigma_L is taken from the
   dispersion of log monthly-average changes (a NAIVE forecast proxy)
   [ASSUMED] until VEP forecast errors are estimated from the VEP history.

Everything is additive and lives beside the v1 model; nothing in v1 changes.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .hpfc import fit_shape

TURKEY_TZ = "Europe/Istanbul"
PARAM_NAMES = ("c0", "c1", "phi", "log_s0", "log_s1",
               "a01", "g01", "h01", "a10", "g10", "h10")

__all__ = ["build_residual_panel", "load_rd_covariates", "MSARFit", "fit_fast_msar",
           "msar_negloglik", "fit_slow_daily", "level_uncertainty",
           "price_limits", "ResidualModelV2", "simulate_prices", "coverage_table",
           "mc_option_price"]


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------
def load_rd_covariates(path) -> pd.DataFrame:
    """z_{t-1} and standardised ramp_{t-1} on a UTC hourly index."""
    z = pd.read_csv(path, parse_dates=["datetime"])
    z["datetime"] = pd.to_datetime(z["datetime"], utc=True)
    z = z.set_index("datetime")["z"].sort_index()
    ramp = z.diff()
    ramp = (ramp - ramp.mean()) / ramp.std()
    return pd.DataFrame(dict(z1=z.shift(1), r1=ramp.shift(1)))


def build_residual_panel(ptf: pd.Series, start_year: int, end_year: int,
                         half_life_years: Optional[float] = 0.5, n_harmonics: int = 2,
                         scale_window_months: int = 12) -> pd.DataFrame:
    """Hourly panel with ex-ante shape, monthly mean, trailing scale and x.

    For each year Y in [start_year, end_year] the shape is fitted on data
    strictly before 1 Jan Y (local), so no shape information leaks forward.
    """
    parts = []
    for Y in range(start_year, end_year + 1):
        y0 = pd.Timestamp(f"{Y}-01-01", tz=TURKEY_TZ).tz_convert("UTC")
        y1 = pd.Timestamp(f"{Y + 1}-01-01", tz=TURKEY_TZ).tz_convert("UTC")
        seg = ptf[(ptf.index >= y0) & (ptf.index < y1)]
        if seg.empty:
            continue
        shape = fit_shape(ptf, y0, half_life_years, n_harmonics)
        parts.append(pd.DataFrame(dict(P=seg.to_numpy(), S=shape.shape(seg.index)),
                                  index=seg.index))
    df = pd.concat(parts)
    loc = df.index.tz_convert(TURKEY_TZ).tz_localize(None)
    mon = loc.to_period("M")
    df["M"] = df.groupby(np.asarray(mon))["P"].transform("mean").to_numpy()
    ploc = ptf.copy()
    ploc.index = ptf.index.tz_convert(TURKEY_TZ).tz_localize(None)
    mm = ploc.groupby(ploc.index.to_period("M")).mean()
    L = mm.rolling(scale_window_months, min_periods=6).mean().shift(1)
    df["L"] = L.reindex(mon).to_numpy()
    df["x"] = (df["P"] - df["M"] * df["S"]) / df["L"]
    day = loc.normalize()
    df["day"] = np.asarray(day)
    df["d"] = df.groupby("day")["x"].transform("mean")
    df["u"] = df["x"] - df["d"]
    return df


# ---------------------------------------------------------------------------
# fast factor: Markov-switching AR(1) with 2-covariate TVTP
# ---------------------------------------------------------------------------
def _probs(th, Z, R):
    a01, g01, h01, a10, g10, h10 = th[5:11]
    p01 = 1.0 / (1.0 + np.exp(-(a01 + g01 * Z + h01 * R)))
    p10 = 1.0 / (1.0 + np.exp(-(a10 + g10 * Z + h10 * R)))
    return p01, p10


def msar_negloglik(th: np.ndarray, u: np.ndarray, Z: np.ndarray, R: np.ndarray,
                   return_filter: bool = False):
    """Exact Hamilton-filter negative log-likelihood (conditional on u_0)."""
    c0, c1, phi, ls0, ls1 = th[:5]
    s0, s1 = math.exp(ls0), math.exp(ls1)
    p01, p10 = _probs(th, Z, R)
    lag = np.r_[0.0, u[:-1]]
    k = 1.0 / math.sqrt(2.0 * math.pi)
    f0 = k / s0 * np.exp(-0.5 * ((u - c0 - phi * lag) / s0) ** 2)
    f1 = k / s1 * np.exp(-0.5 * ((u - c1 - phi * lag) / s1) ** 2)
    p01l, p10l, f0l, f1l = p01.tolist(), p10.tolist(), f0.tolist(), f1.tolist()
    q0 = p10l[0] / max(p01l[0] + p10l[0], 1e-12)
    ll = 0.0
    filt = np.empty(len(u)) if return_filter else None
    if return_filter:
        filt[0] = q0
    log = math.log
    for t in range(1, len(u)):
        pr0 = q0 * (1.0 - p01l[t]) + (1.0 - q0) * p10l[t]
        a = pr0 * f0l[t]
        b = (1.0 - pr0) * f1l[t]
        s = a + b
        if s <= 1e-300:
            return (1e12, filt) if return_filter else 1e12
        ll += log(s)
        q0 = a / s
        if return_filter:
            filt[t] = q0
    return (-ll, filt) if return_filter else -ll


@dataclass
class MSARFit:
    params: Dict[str, float]
    se: Dict[str, float]
    loglik: float
    n_obs: int
    converged: bool
    message: str
    stationary_stress_share: float = float("nan")
    notes: list = field(default_factory=list)

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.params[k] for k in PARAM_NAMES])

    @property
    def sigma(self) -> np.ndarray:
        return np.exp([self.params["log_s0"], self.params["log_s1"]])


def _num_hessian(f, x, eps=1e-4):
    n = x.size
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            if i == j:
                H[i, i] = (f(x + ei) - 2 * fx + f(x - ei)) / eps ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej)
                                     - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps ** 2)
    return H


def fit_fast_msar(u: np.ndarray, Z: np.ndarray, R: np.ndarray,
                  theta0: Optional[Sequence[float]] = None, with_se: bool = True) -> MSARFit:
    """MLE of the fast factor.  Regime 1 is forced to be the high-sigma state."""
    u, Z, R = (np.asarray(v, float) for v in (u, Z, R))
    ok = np.isfinite(u) & np.isfinite(Z) & np.isfinite(R)
    u, Z, R = u[ok], Z[ok], R[ok]
    sd = float(np.std(u))
    th0 = np.array(theta0 if theta0 is not None else
                   [0.0, 0.0, 0.6, math.log(0.4 * sd), math.log(1.1 * sd),
                    -2.0, 0.0, 0.0, -2.0, 0.0, 0.0], float)
    bounds = [(-1, 1), (-1, 1), (-0.99, 0.999), (-12, 2), (-12, 2)] + [(-15, 15)] * 6
    f = lambda th: msar_negloglik(th, u, Z, R)          # noqa: E731
    res = minimize(f, th0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500))
    th = res.x.copy()
    notes = []
    if th[4] < th[3]:                                    # relabel: 1 = high-vol
        th = th[[1, 0, 2, 4, 3, 8, 9, 10, 5, 6, 7]]
        notes.append("regimes relabelled so that regime 1 = high sigma")
    se = {k: float("nan") for k in PARAM_NAMES}
    if with_se:
        try:
            H = _num_hessian(f, th)
            cov = np.linalg.inv(H)
            se = dict(zip(PARAM_NAMES, np.sqrt(np.clip(np.diag(cov), 0, None)).tolist()))
        except np.linalg.LinAlgError:
            notes.append("Hessian not invertible; no standard errors")
    p01, p10 = _probs(th, Z, R)
    share = float(np.mean(p01 / (p01 + p10)))
    return MSARFit(params=dict(zip(PARAM_NAMES, th.tolist())), se=se, loglik=float(-f(th)),
                   n_obs=int(u.size), converged=bool(res.success), message=str(res.message),
                   stationary_stress_share=share, notes=notes)


# ---------------------------------------------------------------------------
# slow factor and level factor
# ---------------------------------------------------------------------------
def fit_slow_daily(daily_mean: pd.Series) -> Dict[str, float]:
    x = np.asarray(daily_mean.dropna(), float)
    phi = float(np.dot(x[:-1], x[1:]) / np.dot(x[:-1], x[:-1]))
    innov = x[1:] - phi * x[:-1]
    return dict(phi_daily=phi, sigma_daily=float(np.std(innov, ddof=1)),
                half_life_days=float(math.log(2) / -math.log(phi)) if 0 < phi < 1 else float("inf"),
                n_days=int(x.size))


def level_uncertainty(ptf: pd.Series, since_year: int = 2022, max_tau: int = 7) -> pd.DataFrame:
    """sd of log(M_m / M_{m-tau}) -- naive-forecast proxy for the forward's level error."""
    loc = ptf.copy()
    loc.index = ptf.index.tz_convert(TURKEY_TZ).tz_localize(None)
    mm = np.log(loc.groupby(loc.index.to_period("M")).mean().clip(lower=1.0))
    rows = []
    for tau in range(1, max_tau + 1):
        d = (mm - mm.shift(tau)).dropna()
        d = d[d.index.year >= since_year]
        rows.append(dict(tau_months=tau, sigma_log=float(d.std()), n=int(d.size)))
    return pd.DataFrame(rows)


def price_limits(times_utc: pd.DatetimeIndex, schedule: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Regulatory cap per hour.  Default schedule [DATA-INFERRED from realised PTF]:
    3400 TRY/MWh up to 2026-04-04 00:00 TRT, 4500 afterwards; floor 0."""
    if schedule is None:
        schedule = pd.DataFrame(dict(valid_from_local=["2025-01-01", "2026-04-04"],
                                     cap_TRY_MWh=[3400.0, 4500.0]))
    starts = pd.to_datetime(schedule["valid_from_local"]).dt.tz_localize(TURKEY_TZ).dt.tz_convert("UTC")
    caps = schedule["cap_TRY_MWh"].to_numpy(float)
    idx = np.searchsorted(starts.to_numpy(), times_utc.to_numpy(), side="right") - 1
    return caps[np.clip(idx, 0, len(caps) - 1)]


# ---------------------------------------------------------------------------
# the assembled model and its Monte Carlo
# ---------------------------------------------------------------------------
@dataclass
class ResidualModelV2:
    fast: MSARFit
    slow: Dict[str, float]
    level: pd.DataFrame                   # tau_months, sigma_log
    scale_L: float                        # ex-ante TRY scale at valuation
    estimation_window: str = ""

    def continuous(self) -> Dict[str, float]:
        """OU equivalents (per hour) for the PDE implementation."""
        phi = self.fast.params["phi"]
        k_f = -math.log(phi)
        conv = math.sqrt(2 * k_f / (1 - phi ** 2))
        s0, s1 = self.fast.sigma
        k_s = -math.log(self.slow["phi_daily"]) / 24.0
        conv_s = math.sqrt(2 * k_s * 24 / (1 - self.slow["phi_daily"] ** 2)) / math.sqrt(24)
        return dict(kappa_fast_per_h=k_f, half_life_fast_h=math.log(2) / k_f,
                    sigma_fast_normal_TRY=s0 * conv * self.scale_L,
                    sigma_fast_stress_TRY=s1 * conv * self.scale_L,
                    kappa_slow_per_h=k_s, half_life_slow_h=math.log(2) / k_s,
                    sigma_slow_TRY=self.slow["sigma_daily"] * conv_s * self.scale_L)

    def sigma_level(self, tau_months: float) -> float:
        lv = self.level
        return float(np.interp(tau_months, np.r_[0.0, lv["tau_months"]], np.r_[0.0, lv["sigma_log"]]))

    def summary(self) -> Dict[str, object]:
        return dict(fast=asdict(self.fast), slow=self.slow, level=self.level.to_dict("records"),
                    scale_L=self.scale_L, continuous=self.continuous(),
                    estimation_window=self.estimation_window)


def simulate_prices(model: ResidualModelV2, times_utc: pd.DatetimeIndex, forward: np.ndarray,
                    valuation_utc: pd.Timestamp, z_paths: np.ndarray, r_paths: np.ndarray,
                    n_paths: int = 2000, seed: int = 1, include_level: bool = True,
                    pi0_stress: Optional[float] = None, cap: Optional[np.ndarray] = None,
                    floor: float = 0.0, keep_paths: bool = False):
    """Monte Carlo of clipped prices on ``times_utc`` (hourly, after valuation).

    ``z_paths``, ``r_paths``: (n_times, n_paths) covariate scenarios, e.g.
    historical-year bootstrap.  Returns dict with per-hour mean / sd / quantile
    helpers (and the paths when ``keep_paths``).
    """
    rng = np.random.default_rng(seed)
    th = model.fast.theta
    c0, c1, phi = th[0], th[1], th[2]
    s0, s1 = model.fast.sigma
    n = len(times_utc)
    N = n_paths
    cap = price_limits(times_utc) if cap is None else np.asarray(cap, float)
    loc = times_utc.tz_convert(TURKEY_TZ).tz_localize(None)
    day = loc.normalize()
    newday = np.r_[True, day[1:] != day[:-1]]
    vloc = valuation_utc.tz_convert(TURKEY_TZ).tz_localize(None)
    mon = loc.to_period("M")
    months = sorted(set(mon))
    lev_draw = {m: rng.standard_normal(N) for m in months}
    # months from valuation to the middle of each delivery month
    tau = {m: max(((m.start_time + (m.end_time - m.start_time) / 2) - vloc).days / 30.44, 0.0)
           for m in months}
    share = model.fast.stationary_stress_share if pi0_stress is None else pi0_stress
    s = (rng.random(N) < share).astype(np.int8)
    u = np.zeros(N)
    d = np.zeros(N)
    phid, sdd = model.slow["phi_daily"], model.slow["sigma_daily"]
    L = model.scale_L
    out_mean = np.empty(n); out_sd = np.empty(n)
    paths = np.empty((n, N), dtype=np.float32) if keep_paths else None
    for t in range(n):
        p01, p10 = _probs(th, z_paths[t], r_paths[t])
        r = rng.random(N)
        s = np.where(s == 0, (r < p01), ~(r < p10)).astype(np.int8)
        u = np.where(s == 0, c0, c1) + phi * u + np.where(s == 0, s0, s1) * rng.standard_normal(N)
        if newday[t]:
            d = phid * d + sdd * rng.standard_normal(N)
        x = L * (u + d)
        if include_level:
            m = mon[t]
            x = x + model.sigma_level(tau[m]) * forward[t] * lev_draw[m]
        p = np.clip(forward[t] + x, floor, cap[t])
        out_mean[t] = p.mean(); out_sd[t] = p.std()
        if keep_paths:
            paths[t] = p
    return dict(mean=out_mean, sd=out_sd, paths=paths, cap=cap)


def coverage_table(realized: np.ndarray, paths: np.ndarray, months: np.ndarray,
                   forward: np.ndarray) -> pd.DataFrame:
    """Central-interval coverage, predictive sd and RMSE by month (+ ALL)."""
    pit = (paths < realized[:, None]).mean(axis=1)
    sd = paths.std(axis=1)
    e = realized - forward
    df = pd.DataFrame(dict(month=months, pit=pit, sd=sd, e=e))
    rows = []

    def cov(p, a):
        return 100.0 * float(np.mean((p > (1 - a) / 2) & (p < 1 - (1 - a) / 2)))

    for m, g in list(df.groupby("month")) + [("ALL", df)]:
        rows.append(dict(month=m, n=len(g), model_sd=g["sd"].mean(),
                         rmse=float(np.sqrt((g["e"] ** 2).mean())),
                         cov50=cov(g["pit"], 0.5), cov80=cov(g["pit"], 0.8),
                         cov90=cov(g["pit"], 0.9)))
    return pd.DataFrame(rows)


def mc_option_price(paths_at_T: np.ndarray, strike: float, tau_hours: float,
                    r_annual: float = 0.40, kind: str = "call") -> Dict[str, float]:
    df = math.exp(-r_annual * tau_hours / 8760.0)
    pay = np.maximum(paths_at_T - strike, 0) if kind == "call" else np.maximum(strike - paths_at_T, 0)
    return dict(value=float(df * pay.mean()), mc_se=float(df * pay.std() / math.sqrt(pay.size)))
