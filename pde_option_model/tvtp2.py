"""Two-covariate TVTP (RD_lag1 + RD_Ramp_1h_lag1): historical covariate panel,
ramp standardization, intercept derivation and audit diagnostics.

STATUS: EXPERIMENTAL / RECONSTRUCTED
------------------------------------
The M9 ``TVTP-2`` fit that supplies the transition slopes used two covariates,
``RD_lag1`` and ``RD_Ramp_1h_lag1``.  The slopes of both are in the handoff
bundle (``transition_coefficients.csv``); the *construction* of the ramp
covariate (difference direction, lag, raw RD or z, standardization window,
ddof) is not -- it was searched for in the repository, its git history, the
SSRN manuscript and the handoff bundle and was not found.  This module
therefore implements an explicitly labelled reconstruction::

    dz_t = z_t - z_{t-1h}                 on the COMPLETE hourly UTC grid
                                          (undefined unless both labels exist)
    r_t  = (dz_t - m_r) / s_r             (m_r, s_r) on a training window W_r,
                                          ddof = 1
    x_{t-1} = (z_{t-1h}, r_{t-1h})        covariates of the S_{t-1} -> S_t step

Nothing here claims to reproduce M9; every artefact carries
:data:`EXPERIMENTAL_LABEL`.  See ``docs/tvtp2_methodology.md``.

Leakage rule: only labels <= the chosen window end enter a scaler or an
intercept; the historical transition at t only uses z at t-1h and t-2h.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .generator import (TVTP2Coefficients, TVTPCoefficients, embeddability_report,
                        logistic)

logger = logging.getLogger(__name__)

HOUR = pd.Timedelta(hours=1)

EXPERIMENTAL_LABEL = ("M9-transferred slopes + reconstructed ramp + derived "
                      "intercepts, zero transition premium")
PARAMETER_SET_NAME = "tvtp2_m9_transfer"
STATUS_EXPERIMENTAL = "experimental_reconstructed"
Z_COVARIATE = "RD_lag1"
RAMP_COVARIATE = "RD_Ramp_1h_lag1"
RAMP_DEFINITION = (
    "r_t = (dz_t - m_r) / s_r with dz_t = z_t - z_(t-1h) computed on the complete "
    "hourly UTC grid (dz_t undefined unless both labels t and t-1h are observed); "
    "transition S_(t-1) -> S_t uses x_(t-1) = (z_(t-1h), r_(t-1h))")
RAMP_STATUS = (
    "ASSUMED/RECONSTRUCTED: the original M9 construction of RD_Ramp_1h_lag1 "
    "(difference direction, lag, raw RD vs z, standardization window, ddof) was "
    "not found in the repository, its git history, the SSRN manuscript or the "
    "calibration bundle; z.diff() on the complete hourly grid is a reconstruction "
    "and M9 is NOT claimed to be reproduced")
ZERO_PREMIUM_NOTE = ("ASSUMED: zero transition premium, q^Q = q^P (eta01 = eta10 = 0); "
                     "NOT calibrated -- forward quotes cannot identify it")


class CovariateDataError(ValueError):
    """Historical / scenario covariate data violate the hourly-grid rules."""


# ---------------------------------------------------------------------------
# files and hourly grid
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path, normalize_newlines: bool = True) -> str:
    """Content hash; CRLF is normalized to LF by default so that a Windows
    (autocrlf) checkout and a Linux checkout of the same file agree."""
    data = Path(path).read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _fmt_labels(labels: Iterable[pd.Timestamp], limit: int = 12) -> str:
    lab = [pd.Timestamp(x).isoformat() for x in labels]
    more = f" ... (+{len(lab) - limit} more)" if len(lab) > limit else ""
    return ", ".join(lab[:limit]) + more


def validate_hourly_utc_index(idx: pd.DatetimeIndex, what: str) -> None:
    """Unique, tz-aware, hour-aligned labels (sub-hour stamps are rejected)."""
    if not isinstance(idx, pd.DatetimeIndex):
        raise CovariateDataError(f"{what}: index is not a DatetimeIndex")
    if idx.tz is None:
        raise CovariateDataError(f"{what}: timestamps must be timezone-aware (UTC)")
    dup = idx[idx.duplicated()]
    if len(dup):
        raise CovariateDataError(f"{what}: duplicate hour labels: {_fmt_labels(dup)}")
    off = idx[idx != idx.floor("h")]
    if len(off):
        raise CovariateDataError(
            f"{what}: sub-hour / irregular timestamps are not accepted (no "
            f"interpolation onto the hourly grid): {_fmt_labels(off)}")


def load_hourly_z_history(path: str | Path) -> pd.Series:
    """Standardized residual demand z on its UTC hour labels (not reindexed)."""
    p = Path(path)
    df = pd.read_csv(p)
    if "datetime" not in df.columns or "z" not in df.columns:
        raise CovariateDataError(f"{p}: expected columns 'datetime' and 'z'")
    idx = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    s = pd.Series(df["z"].to_numpy(dtype=float), index=idx, name="z")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    validate_hourly_utc_index(s.index, str(p))
    return s


def complete_hourly_grid(z: pd.Series, start: Optional[pd.Timestamp] = None,
                         end: Optional[pd.Timestamp] = None) -> pd.Series:
    """Reindex onto every UTC hour in [start, end]; missing hours stay NaN."""
    if z.index.tz is None:
        raise CovariateDataError("z index must be tz-aware (UTC)")
    lo = z.index[0] if start is None else pd.Timestamp(start)
    hi = z.index[-1] if end is None else pd.Timestamp(end)
    grid = pd.date_range(lo.tz_convert("UTC"), hi.tz_convert("UTC"), freq="h")
    return z.reindex(grid)


def _assert_complete_grid(s: pd.Series) -> None:
    d = np.diff(s.index.asi8)
    if d.size and not np.all(d == HOUR.value):
        raise CovariateDataError("series is not on a complete hourly grid")


def hourly_increments(z_grid: pd.Series) -> pd.Series:
    """dz_t = z_t - z_{t-1h}; NaN whenever either label is missing."""
    _assert_complete_grid(z_grid)
    return z_grid - z_grid.shift(1)


# ---------------------------------------------------------------------------
# ramp standardization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RampScaler:
    """(m_r, s_r) of the hourly increment dz on a training window."""

    mean: float
    std: float
    ddof: int
    n: int
    window_name: str
    window_end_utc: pd.Timestamp
    window_start_utc: Optional[pd.Timestamp] = None      # None -> sample start
    first_increment_utc: Optional[pd.Timestamp] = None
    last_increment_utc: Optional[pd.Timestamp] = None
    definition: str = RAMP_DEFINITION
    status: str = RAMP_STATUS

    def __post_init__(self) -> None:
        if not (np.isfinite(self.mean) and np.isfinite(self.std)) or self.std <= 0:
            raise CovariateDataError("ramp scaler needs a finite mean and a positive std")
        for k in ("window_end_utc", "window_start_utc", "first_increment_utc",
                  "last_increment_utc"):
            v = getattr(self, k)
            if v is not None:
                ts = pd.Timestamp(v)
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                object.__setattr__(self, k, ts)

    def standardize(self, dz: Any) -> Any:
        return (dz - self.mean) / self.std

    def matches(self, other: "RampScaler", rtol: float = 1e-12) -> bool:
        return (other is not None
                and abs(self.mean - other.mean) <= rtol * max(abs(self.std), 1e-300)
                and abs(self.std - other.std) <= rtol * abs(self.std)
                and int(self.ddof) == int(other.ddof))

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, pd.Timestamp):
                d[k] = v.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RampScaler":
        keys = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in keys}
        for k in ("window_end_utc", "window_start_utc", "first_increment_utc",
                  "last_increment_utc"):
            if kw.get(k) is not None:
                kw[k] = pd.Timestamp(str(kw[k]))
        kw["mean"] = float(kw["mean"])
        kw["std"] = float(kw["std"])
        kw["ddof"] = int(kw["ddof"])
        kw["n"] = int(kw["n"])
        return cls(**kw)


def fit_ramp_scaler(z_history: pd.Series, window_end: pd.Timestamp,
                    window_name: str, ddof: int = 1,
                    window_start: Optional[pd.Timestamp] = None) -> RampScaler:
    """Train-only (m_r, s_r): increments with label <= ``window_end`` only."""
    end = pd.Timestamp(window_end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    zg = complete_hourly_grid(z_history)
    dz = hourly_increments(zg)
    m = dz.notna().to_numpy() & (dz.index <= end)
    if window_start is not None:
        m &= dz.index >= pd.Timestamp(window_start)
    sel = dz[m]
    if len(sel) < 2:
        raise CovariateDataError(f"ramp scaler window {window_name!r} has < 2 increments")
    vals = sel.to_numpy(dtype=float)
    return RampScaler(mean=float(vals.mean()), std=float(vals.std(ddof=ddof)),
                      ddof=int(ddof), n=int(vals.size), window_name=window_name,
                      window_end_utc=end, window_start_utc=window_start,
                      first_increment_utc=sel.index[0], last_increment_utc=sel.index[-1])


# ---------------------------------------------------------------------------
# historical covariate panel
# ---------------------------------------------------------------------------
@dataclass
class CovariatePanel:
    """Historical hourly panel on the complete UTC grid.

    Columns: ``z``, ``dz``, ``r`` (standardized ramp), ``z_lag``, ``r_lag``
    (both shifted by the SAME calendar lag) and ``valid`` (both lagged
    covariates observed).  Nothing is filled: a missing hour propagates NaN
    into every increment and lag that touches it.
    """

    frame: pd.DataFrame
    ramp_scaler: RampScaler
    lag_hours: int
    missing_labels: pd.DatetimeIndex
    source: Dict[str, Any] = field(default_factory=dict)

    @property
    def grid_start(self) -> pd.Timestamp:
        return self.frame.index[0]

    @property
    def grid_end(self) -> pd.Timestamp:
        return self.frame.index[-1]

    def _window(self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
                ) -> np.ndarray:
        m = np.ones(len(self.frame), dtype=bool)
        if start is not None:
            m &= self.frame.index >= pd.Timestamp(start)
        if end is not None:
            m &= self.frame.index <= pd.Timestamp(end)
        return m

    def sample(self, end: Optional[pd.Timestamp] = None,
               start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Transitions t in the window whose covariates x_{t-1} are observed."""
        m = self._window(start, end) & self.frame["valid"].to_numpy(dtype=bool)
        return self.frame.loc[m]

    def accounting(self, end: Optional[pd.Timestamp] = None,
                   start: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """Counts (and labels) of transitions dropped in the window, by reason."""
        w = self._window(start, end)
        fr = self.frame.loc[w]
        invalid = fr.index[~fr["valid"].to_numpy(dtype=bool)]
        first_ok = self.grid_start + (self.lag_hours + 1) * HOUR
        head = invalid[invalid < first_ok]
        gap = invalid[invalid >= first_ok]
        own_missing_but_valid = fr.index[fr["valid"].to_numpy(dtype=bool)
                                         & fr["z"].isna().to_numpy()]
        sample = self.sample(end, start)
        return {
            "window_start_utc": fr.index[0].isoformat(),
            "window_end_utc": fr.index[-1].isoformat(),
            "n_grid_labels": int(len(fr)),
            "n_valid_transitions": int(len(sample)),
            "first_valid_transition_utc": sample.index[0].isoformat() if len(sample) else None,
            "last_valid_transition_utc": sample.index[-1].isoformat() if len(sample) else None,
            "n_dropped_sample_start": int(len(head)),
            "dropped_sample_start_labels": [x.isoformat() for x in head],
            "rule_sample_start": (
                f"the first {self.lag_hours + 1} hour(s) of the sample have no "
                "z_(t-1h) / z_(t-2h) pair, so r_(t-1) is undefined: dropped, "
                "never filled"),
            "n_dropped_missing_observation": int(len(gap)),
            "dropped_missing_observation_labels": [x.isoformat() for x in gap],
            "rule_missing_observation": (
                "a transition is dropped when z_(t-1h) or z_(t-2h) is missing on "
                "the complete hourly grid; no interpolation, no zero ramp"),
            "n_valid_with_missing_own_label": int(len(own_missing_but_valid)),
            "valid_with_missing_own_label": [x.isoformat() for x in own_missing_but_valid],
        }


def build_covariate_panel(z_history: pd.Series, ramp_scaler: RampScaler,
                          lag_hours: int = 1,
                          source: Optional[Dict[str, Any]] = None) -> CovariatePanel:
    """z_{t-1} and r_{t-1} built together on the complete hourly UTC grid."""
    if int(lag_hours) != lag_hours or lag_hours < 0:
        raise CovariateDataError("lag_hours must be a non-negative integer")
    lag = int(lag_hours)
    zg = complete_hourly_grid(z_history)
    dz = hourly_increments(zg)
    r = ramp_scaler.standardize(dz)
    z_lag = zg.shift(lag)
    r_lag = r.shift(lag)
    valid = z_lag.notna() & r_lag.notna()
    frame = pd.DataFrame({"z": zg, "dz": dz, "r": r, "z_lag": z_lag,
                          "r_lag": r_lag, "valid": valid})
    missing = zg.index[zg.isna().to_numpy()]
    return CovariatePanel(frame=frame, ramp_scaler=ramp_scaler, lag_hours=lag,
                          missing_labels=missing, source=dict(source or {}))


# ---------------------------------------------------------------------------
# intercept derivation (moment matching, NOT estimation)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InterceptRoot:
    transition: str
    alpha: float
    gamma: float
    h: float
    target_mean_p: float
    achieved_mean_p: float
    residual: float
    n: int
    bracket: Tuple[float, float]
    xtol: float
    rtol: float
    iterations: int
    function_calls: int
    converged: bool

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bracket"] = list(self.bracket)
        return d


def mean_transition_probability(alpha: float, gamma: float, h: float,
                                Z: np.ndarray, R: np.ndarray) -> float:
    return float(np.mean(logistic(alpha + gamma * Z + h * R)))


def solve_intercept(Z: np.ndarray, R: np.ndarray, gamma: float, h: float,
                    target: float, transition: str,
                    bracket: Tuple[float, float] = (-20.0, 20.0),
                    xtol: float = 1e-12, rtol: float = 1e-12) -> InterceptRoot:
    """Unique alpha with mean_D logistic(alpha + gamma z + h r) = target.

    g(alpha) is strictly increasing (derivative = mean Lambda' > 0) with limits
    -target and 1 - target, so a sign change on the bracket proves a unique root.
    """
    Z = np.asarray(Z, dtype=float)
    R = np.asarray(R, dtype=float)
    if Z.shape != R.shape or Z.ndim != 1 or Z.size == 0:
        raise CovariateDataError("Z and R must be equal-length non-empty vectors")
    if not (np.all(np.isfinite(Z)) and np.all(np.isfinite(R))):
        raise CovariateDataError("intercept sample contains missing covariates")
    if not 0.0 < target < 1.0:
        raise ValueError("target mean probability must lie in (0, 1)")

    def g(a: float) -> float:
        return mean_transition_probability(a, gamma, h, Z, R) - target

    lo, hi = bracket
    if not (g(lo) < 0.0 < g(hi)):
        raise ValueError(f"no sign change of the moment condition on {bracket}")
    a, res = brentq(g, lo, hi, xtol=xtol, rtol=rtol, full_output=True)
    ach = mean_transition_probability(a, gamma, h, Z, R)
    return InterceptRoot(transition=transition, alpha=float(a), gamma=float(gamma),
                         h=float(h), target_mean_p=float(target),
                         achieved_mean_p=float(ach), residual=float(ach - target),
                         n=int(Z.size), bracket=(float(lo), float(hi)), xtol=xtol,
                         rtol=rtol, iterations=int(res.iterations),
                         function_calls=int(res.function_calls),
                         converged=bool(res.converged))


def derive_intercepts(panel: CovariatePanel, gamma01: float, h01: float,
                      gamma10: float, h10: float, duration_normal_h: float,
                      duration_stress_h: float, sample_end: pd.Timestamp,
                      sample_start: Optional[pd.Timestamp] = None,
                      use_ramp: bool = True) -> Dict[str, Any]:
    """Both intercepts on the transition sample D (paired, simultaneous x_{t-1}).

    ``use_ramp=False`` gives the single-covariate derivation on the SAME D
    (needed for the ramp-effect ladder: identical sample, only h differs).
    """
    D = panel.sample(sample_end, sample_start)
    Z = D["z_lag"].to_numpy(dtype=float)
    R = D["r_lag"].to_numpy(dtype=float) if use_ramp else np.zeros(len(D))
    hh01, hh10 = (h01, h10) if use_ramp else (0.0, 0.0)
    r01 = solve_intercept(Z, R, gamma01, hh01, 1.0 / duration_normal_h, "p01_normal_to_stress")
    r10 = solve_intercept(Z, R, gamma10, hh10, 1.0 / duration_stress_h, "p10_stress_to_normal")
    return {"alpha01": r01.alpha, "alpha10": r10.alpha,
            "root_p01": r01.as_dict(), "root_p10": r10.as_dict(),
            "sample": panel.accounting(sample_end, sample_start)}


# ---------------------------------------------------------------------------
# validation metrics (reported, never targeted)
# ---------------------------------------------------------------------------
def forward_recursion_occupancy(labels: pd.DatetimeIndex, p01: np.ndarray,
                                p10: np.ndarray, burn_in_hours: int = 48
                                ) -> Dict[str, Any]:
    """Mean stress probability of pi_t = pi_{t-1} Pi(t) along the historical path.

    The path is split into runs of consecutive hourly labels; each run starts
    from the stationary distribution of its first Pi and its first
    ``burn_in_hours`` values are discarded.  No simulation is involved.
    """
    labels = pd.DatetimeIndex(labels)
    p01 = np.asarray(p01, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    if not (len(labels) == p01.size == p10.size):
        raise ValueError("labels and probabilities must align")
    br = np.flatnonzero(np.diff(labels.asi8) != HOUR.value) + 1
    starts = np.concatenate([[0], br])
    ends = np.concatenate([br, [len(labels)]])
    kept: List[float] = []
    for a, b in zip(starts, ends):
        s0 = p01[a] + p10[a]
        pi1 = p01[a] / s0                           # stationary stress share of Pi(t_a)
        vals = np.empty(b - a)
        for k in range(a, b):
            pi1 = (1.0 - pi1) * p01[k] + pi1 * (1.0 - p10[k])
            vals[k - a] = pi1
        kept.extend(vals[burn_in_hours:].tolist())
    arr = np.asarray(kept)
    return {"mean_stress_probability": float(arr.mean()) if arr.size else float("nan"),
            "n_hours_used": int(arr.size), "n_segments": int(len(starts)),
            "burn_in_hours_per_segment": int(burn_in_hours),
            "method": ("exact forward recursion pi_t = pi_(t-1) Pi(t) on the "
                       "historical covariate path, segments split at gaps, each "
                       "started at the stationary law of its first Pi")}


def ergodic_ratio_occupancy(duration_normal_h: float, duration_stress_h: float) -> float:
    """(1/d_n) / (1/d_n + 1/d_s): a function of the two targets ONLY."""
    a, b = 1.0 / duration_normal_h, 1.0 / duration_stress_h
    return a / (a + b)


def p01_duration_target_consistency(target_mean_p: float, m9_ame_ratio: float
                                    ) -> Dict[str, Any]:
    """E[p(1-p)] <= pbar(1-pbar): is the duration target compatible with M9's AME?"""
    bound = target_mean_p * (1.0 - target_mean_p)
    implied_min = (1.0 - np.sqrt(max(1.0 - 4.0 * m9_ame_ratio, 0.0))) / 2.0
    return {"target_mean_p": float(target_mean_p),
            "upper_bound_E_p_1mp": float(bound),
            "m9_ame_ratio": float(m9_ame_ratio),
            "compatible": bool(m9_ame_ratio <= bound),
            "implied_minimum_mean_p_of_m9_path": float(implied_min),
            "note": ("E[p(1-p)] = pbar(1-pbar) - Var(p) <= pbar(1-pbar); if the M9 "
                     "AME ratio exceeds the bound, the duration target cannot be "
                     "the mean of M9's own fitted p path")}


def transition_moment_metrics(p01: np.ndarray, p10: np.ndarray) -> Dict[str, float]:
    p01 = np.asarray(p01, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    return {"mean_p01": float(p01.mean()), "mean_p10": float(p10.mean()),
            "ame_ratio_p01": float(np.mean(p01 * (1.0 - p01))),
            "ame_ratio_p10": float(np.mean(p10 * (1.0 - p10))),
            "mean_inverse_p01": float(np.mean(1.0 / p01)),
            "mean_inverse_p10": float(np.mean(1.0 / p10))}


# ---------------------------------------------------------------------------
# M9 handoff bundle and the atomic label swap
# ---------------------------------------------------------------------------
def load_m9_bundle(bundle_dir: str | Path) -> Dict[str, Any]:
    """Raw M9 numbers (state 0 = HIGH volatility) exactly as shipped."""
    b = Path(bundle_dir)
    # round_trip parsing: the slopes must be the exact doubles written in the
    # CSV (the default C parser can be off by one ulp for 17-digit values).
    tc = pd.read_csv(b / "transition_coefficients.csv", float_precision="round_trip")
    pe = pd.read_csv(b / "parameter_estimates.csv", float_precision="round_trip")
    meta_path = b / "metadata" / "model_parameters_and_ou_mapping.json"
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    run = None
    for k, v in meta.get("parsed_parameter_files", {}).items():
        if k.replace("\\", "/").endswith("run_summary.json"):
            run = v
    if run is None:
        raise CovariateDataError("run_summary block not found in the bundle metadata")
    m9 = pe.loc[pe["model"] == "M9"].iloc[0]
    rows = {str(r["covariate"]): {"gamma01": float(r["gamma01"]), "gamma10": float(r["gamma10"]),
                                   "ame_p01": float(r["ame_p01"]), "ame_p10": float(r["ame_p10"])}
            for _, r in tc.loc[tc["model"] == "M9"].iterrows()}
    diag = run["diagnostics"]
    spec = [m for m in run["model_table"] if m["model"] == "M9"][0]
    return {
        "files": {"transition_coefficients.csv": str(b / "transition_coefficients.csv"),
                  "parameter_estimates.csv": str(b / "parameter_estimates.csv"),
                  "metadata": str(meta_path)},
        "sha256": {"transition_coefficients.csv": sha256_file(b / "transition_coefficients.csv"),
                   "parameter_estimates.csv": sha256_file(b / "parameter_estimates.csv"),
                   "metadata": sha256_file(meta_path)},
        "spec": spec["spec"], "k": int(spec["k"]),
        "n_train": int(run["n_train"]), "n_total": int(run["n_total"]),
        "sigma0": float(m9["sigma0"]), "sigma1": float(m9["sigma1"]),
        "coefficients": rows,
        "occupancy": [float(x) for x in diag["occupancy"]],
        "mean_duration_state0": float(diag["mean_duration_state0"]),
        "mean_duration_state1": float(diag["mean_duration_state1"]),
        "mean_expected_duration_state0": float(diag["mean_expected_duration_state0"]),
        "mean_expected_duration_state1": float(diag["mean_expected_duration_state1"]),
        "expected_duration_note": diag.get("expected_duration_note", ""),
    }


def m9_raw_to_production(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Atomic regime-label swap of EVERYTHING (sigma, slopes, AMEs, durations).

    Raw M9: state 0 = high volatility.  Production: 0 = normal (low sigma),
    1 = stress.  Production p01 (normal -> stress) is raw p10 (raw 1 -> raw 0),
    so the raw ``gamma10`` / ``ame_p10`` columns feed production 01 and vice
    versa.  Swapping only one group would silently attach volatilities and
    transition dynamics to different states, hence one function.
    """
    if not raw["sigma0"] > raw["sigma1"]:
        raise CovariateDataError("raw M9 state 0 is expected to be the high-volatility state")
    rd = raw["coefficients"][Z_COVARIATE]
    rp = raw["coefficients"][RAMP_COVARIATE]
    out = {
        "sigma_y_normal": raw["sigma1"], "sigma_y_stress": raw["sigma0"],
        "gamma01": rd["gamma10"], "gamma10": rd["gamma01"],
        "h01": rp["gamma10"], "h10": rp["gamma01"],
        "ame01_rd": rd["ame_p10"], "ame10_rd": rd["ame_p01"],
        "ame01_ramp": rp["ame_p10"], "ame10_ramp": rp["ame_p01"],
        "duration_normal_h": raw["mean_duration_state1"],
        "duration_stress_h": raw["mean_duration_state0"],
        "occupancy_normal": raw["occupancy"][1], "occupancy_stress": raw["occupancy"][0],
        "mean_expected_duration_normal": raw["mean_expected_duration_state1"],
        "mean_expected_duration_stress": raw["mean_expected_duration_state0"],
    }
    # consistency: AME = E[Lambda'] * slope, Lambda' > 0 -> same sign in each equation
    signs_ok = all(np.sign(out[a]) == np.sign(out[g]) for a, g in (
        ("ame01_rd", "gamma01"), ("ame10_rd", "gamma10"),
        ("ame01_ramp", "h01"), ("ame10_ramp", "h10")))
    ratio01 = (out["ame01_rd"] / out["gamma01"], out["ame01_ramp"] / out["h01"])
    ratio10 = (out["ame10_rd"] / out["gamma10"], out["ame10_ramp"] / out["h10"])
    out["checks"] = {
        "sigma_normal_lt_sigma_stress": bool(out["sigma_y_normal"] < out["sigma_y_stress"]),
        "ame_and_slope_share_sign_in_every_equation": bool(signs_ok),
        "ame_over_slope_p01": list(ratio01),
        "ame_over_slope_p10": list(ratio10),
        "ame_over_slope_abs_diff_p01": float(abs(ratio01[0] - ratio01[1])),
        "ame_over_slope_abs_diff_p10": float(abs(ratio10[0] - ratio10[1])),
        "joint_estimation_evidence": bool(abs(ratio01[0] - ratio01[1]) < 1e-12
                                          and abs(ratio10[0] - ratio10[1]) < 1e-12),
        "joint_estimation_note": (
            "AME/slope = E[Lambda'(eta)] is a property of ONE fitted p path; equal "
            "ratios for RD_lag1 and RD_Ramp_1h_lag1 (to ~1e-15) show both slopes "
            "come from the same TVTP-2 equation (joint estimation)"),
    }
    if not (out["checks"]["sigma_normal_lt_sigma_stress"] and signs_ok):
        raise CovariateDataError(f"label-swap consistency failed: {out['checks']}")
    return out


# ---------------------------------------------------------------------------
# scale / timing evidence for the reconstructed ramp
# ---------------------------------------------------------------------------
def ramp_scale_timing_evidence(panel: CovariatePanel, gamma01: float, h01: float,
                               gamma10: float, h10: float, duration_normal_h: float,
                               duration_stress_h: float, sample_end: pd.Timestamp,
                               scales: Sequence[float] = (0.0, 1.0, 1.1)
                               ) -> List[Dict[str, Any]]:
    """E[1/p] after re-deriving the intercepts for alternative ramp constructions.

    These moments depend on the marginal law of (z, r) only: they discriminate
    the SCALE of the ramp, not its direction or lag.
    """
    fr = panel.frame.loc[panel.frame.index <= pd.Timestamp(sample_end)]
    s = panel.ramp_scaler
    variants: List[Tuple[str, pd.Series]] = [
        (f"standardized ramp x {c:g}, lag 1h", c * fr["r_lag"]) for c in scales]
    variants += [
        (f"raw dz (unstandardized, sd {s.std:.4f}), lag 1h", fr["dz"].shift(panel.lag_hours)),
        ("standardized ramp, no lag (r_t)", fr["r"]),
        ("standardized ramp, lag 2h (r_(t-2))", fr["r"].shift(panel.lag_hours + 1)),
        ("standardized ramp, sign flipped", -fr["r_lag"]),
    ]
    rows: List[Dict[str, Any]] = []
    for name, rser in variants:
        ok = fr["z_lag"].notna().to_numpy() & rser.notna().to_numpy()
        Z = fr["z_lag"].to_numpy(dtype=float)[ok]
        R = rser.to_numpy(dtype=float)[ok]
        a01 = solve_intercept(Z, R, gamma01, h01, 1.0 / duration_normal_h, "p01").alpha
        a10 = solve_intercept(Z, R, gamma10, h10, 1.0 / duration_stress_h, "p10").alpha
        p01 = logistic(a01 + gamma01 * Z + h01 * R)
        p10 = logistic(a10 + gamma10 * Z + h10 * R)
        rows.append({"variant": name, "n": int(Z.size), "alpha01": float(a01),
                     "alpha10": float(a10),
                     "mean_inverse_p01": float(np.mean(1.0 / p01)),
                     "mean_inverse_p10": float(np.mean(1.0 / p10)),
                     "ame_ratio_p01": float(np.mean(p01 * (1 - p01))),
                     "ame_ratio_p10": float(np.mean(p10 * (1 - p10)))})
    return rows


# ---------------------------------------------------------------------------
# historical path audit helpers
# ---------------------------------------------------------------------------
def historical_path_probabilities(panel: CovariatePanel,
                                  coef: TVTP2Coefficients | TVTPCoefficients,
                                  end: Optional[pd.Timestamp] = None,
                                  start: Optional[pd.Timestamp] = None
                                  ) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    D = panel.sample(end, start)
    Z = D["z_lag"].to_numpy(dtype=float)
    if isinstance(coef, TVTP2Coefficients):
        p01, p10 = coef.probabilities(Z, D["r_lag"].to_numpy(dtype=float))
    else:
        p01, p10 = coef.probabilities(Z)
    return D.index, p01, p10


def historical_embeddability(panel: CovariatePanel, coef: TVTP2Coefficients,
                             end: Optional[pd.Timestamp] = None,
                             start: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    labels, p01, p10 = historical_path_probabilities(panel, coef, end, start)
    rep = embeddability_report(p01, p10, labels=list(labels))
    rep["window_start_utc"] = labels[0].isoformat()
    rep["window_end_utc"] = labels[-1].isoformat()
    return rep


# ---------------------------------------------------------------------------
# derivation driver: frozen 2D parameter set + audit (JSON / Markdown)
# ---------------------------------------------------------------------------
DEFAULT_HISTORY = "inputs/historical/rd_standardized.csv"
DEFAULT_LAG1_FILE = "inputs/historical/rd_lag1_standardized.csv"
DEFAULT_BASE_PARAMS = "inputs/historical/m2_frozen_parameters.yaml"
DEFAULT_BUNDLE = "inputs/historical/archive/calibration_bundle"
DEFAULT_TVTP2_YAML = "inputs/historical/tvtp2_frozen_parameters.yaml"
DEFAULT_AUDIT_DIR = "outputs/tvtp2_experimental/derivation"
TRY_WINDOW_END_UTC = "2022-12-31T20:00:00+00:00"     # z scaler window (markov_adapter)

SOURCE_SEARCH = [
    {"source": "repository working tree (code, docs, configs, outputs)",
     "result": "slope values and the covariate NAME only; no construction code"},
    {"source": "git history of the repository (all 27 commits, deleted files, "
               "team_share_2026_09_01.zip)",
     "result": "no ramp construction, no covariate_scaling.json, no prepared_meta.json"},
    {"source": "SSRN manuscript (ssrn-7472067, 43 pages)",
     "result": "states the ramp term is omitted in production and that joint vs "
               "separate estimation could not be determined; no definition"},
    {"source": "calibration bundle (transition_coefficients.csv, parameter_estimates.csv, "
               "metadata json)",
     "result": "slopes and AMEs of both covariates; no definition, no scaler"},
    {"source": "markov_adapter.py docstring",
     "result": "claims a logit(p) regression on 'lagged standardized RD and RD-ramp' "
               "recovers M9 to 4-5 digits; the code and pde_timeseries.parquet are "
               "absent, so the claim is not reproducible here"},
    {"source": "original estimation output directory "
               "(MarkovProject/res-markov/outputs/markov_usd_final)",
     "result": "not accessible from this repository"},
]


def _base_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _variant_metrics(panel: CovariatePanel, coef: TVTP2Coefficients | TVTPCoefficients,
                     end: pd.Timestamp) -> Dict[str, Any]:
    lab, p01, p10 = historical_path_probabilities(panel, coef, end)
    occ = forward_recursion_occupancy(lab, p01, p10)
    m = transition_moment_metrics(p01, p10)
    s = p01 + p10
    m.update({"forward_recursion_occupancy_stress": occ["mean_stress_probability"],
              "max_s": float(s.max()), "n_s_ge_1": int(np.sum(s >= 1.0))})
    return m


def run_tvtp2_derivation(repo_root: str | Path = ".",
                         history_file: str = DEFAULT_HISTORY,
                         lag1_file: str = DEFAULT_LAG1_FILE,
                         base_params_file: str = DEFAULT_BASE_PARAMS,
                         bundle_dir: str = DEFAULT_BUNDLE,
                         ramp_window: str = "W9",
                         sample_window: str = "W9") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Derive the frozen 2D parameter set and its audit from repository data only.

    Windows: ``W9`` = first ``n_train`` rows of the history (M9 training
    window inferred from run_summary n_train / n_total), ``W_T`` = the TRY
    window of the z scaler (<= 2022-12-31 20:00 UTC), ``all`` = full sample.
    Returns ``(yaml_blob, audit)``.
    """
    root = Path(repo_root)
    hist_p, lag1_p = root / history_file, root / lag1_file
    base_p, bund_p = root / base_params_file, root / bundle_dir
    z = load_hourly_z_history(hist_p)
    base = _base_yaml(base_p)
    raw = load_m9_bundle(bund_p)
    prod = m9_raw_to_production(raw)
    val = pd.Timestamp(str(base["valuation_utc"]))
    val = val.tz_localize("UTC") if val.tzinfo is None else val.tz_convert("UTC")

    # --- windows -----------------------------------------------------------
    if raw["n_total"] != len(z):
        raise CovariateDataError(
            f"history has {len(z)} rows but M9 n_total = {raw['n_total']}; the W9 "
            "window inference (first n_train rows) would not be valid")
    windows = {"W_T": pd.Timestamp(TRY_WINDOW_END_UTC),
               "W9": z.index[raw["n_train"] - 1],
               "all": z.index[-1]}
    for nm, end in windows.items():
        if end > val:
            raise CovariateDataError(f"window {nm} ends after the valuation time (leakage)")
    g01, g10 = float(base["tvtp"]["gamma01"]), float(base["tvtp"]["gamma10"])
    h01, h10 = float(prod["h01"]), float(prod["h10"])
    dn, ds = float(prod["duration_normal_h"]), float(prod["duration_stress_h"])

    # --- z scaler evidence ---------------------------------------------------
    z_stats = {}
    for nm, end in windows.items():
        v = z.loc[:end].to_numpy(dtype=float)
        z_stats[nm] = {"window_end_utc": end.isoformat(), "n_rows": int(v.size),
                       "mean": float(v.mean()), "std_ddof1": float(v.std(ddof=1))}

    # --- ramp scalers ----------------------------------------------------------
    scalers = {nm: fit_ramp_scaler(z, end, nm) for nm, end in windows.items()}
    sc = scalers[ramp_window]
    panel = build_covariate_panel(z, sc, source={"history_file": history_file})
    D_end = windows[sample_window]

    # --- primary derivation (R3) and its single-covariate twin (R1) ----------
    prim = derive_intercepts(panel, g01, h01, g10, h10, dn, ds, D_end)
    coef = TVTP2Coefficients(prim["alpha01"], g01, h01, prim["alpha10"], g10, h10,
                             name=PARAMETER_SET_NAME)
    one = derive_intercepts(panel, g01, 0.0, g10, 0.0, dn, ds, D_end, use_ramp=False)

    # --- 1D production reproduction (all rows, contemporaneous z) -------------
    zall = z.to_numpy(dtype=float)
    prod1d_01 = solve_intercept(zall, np.zeros_like(zall), g01, 0.0, 1.0 / dn, "p01").alpha
    prod1d_10 = solve_intercept(zall, np.zeros_like(zall), g10, 0.0, 1.0 / ds, "p10").alpha

    variants: List[Dict[str, Any]] = []

    def add(name: str, a01: float, a10: float, hh01: float, hh10: float,
            pnl: CovariatePanel, end: pd.Timestamp, note: str, n: Optional[int] = None) -> None:
        c: TVTP2Coefficients | TVTPCoefficients
        c = (TVTP2Coefficients(a01, g01, hh01, a10, g10, hh10)
             if (hh01 or hh10) else TVTPCoefficients(a01, g01, a10, g10))
        met = _variant_metrics(pnl, c, end)
        variants.append({"variant": name, "alpha01": float(a01), "alpha10": float(a10),
                         "h01": hh01, "h10": hh10, "n_transitions": n,
                         "note": note, **met})

    add("1D production (repo yaml; all rows, contemporaneous z)",
        float(base["tvtp"]["alpha01"]), float(base["tvtp"]["alpha10"]), 0.0, 0.0,
        panel, D_end, f"re-derived here: {prod1d_01:.10f}, {prod1d_10:.10f}", len(zall))
    add("1D, D = W9 (x_(t-1) sample)  [R1]", one["alpha01"], one["alpha10"], 0.0, 0.0,
        panel, D_end, "same D as the 2D primary", one["sample"]["n_valid_transitions"])
    add("2D, D = W9, ramp scaler W9  [R3, primary]", prim["alpha01"], prim["alpha10"],
        h01, h10, panel, D_end, "frozen in tvtp2_frozen_parameters.yaml",
        prim["sample"]["n_valid_transitions"])
    for nm in ("W_T", "all"):
        p2 = build_covariate_panel(z, scalers[nm])
        d = derive_intercepts(p2, g01, h01, g10, h10, dn, ds, D_end)
        add(f"2D, D = W9, ramp scaler {nm}", d["alpha01"], d["alpha10"], h01, h10, p2,
            D_end, "ramp-scaler window sensitivity", d["sample"]["n_valid_transitions"])
    dall = derive_intercepts(panel, g01, h01, g10, h10, dn, ds, windows["all"])
    add("2D, D = full sample, ramp scaler W9", dall["alpha01"], dall["alpha10"], h01, h10,
        panel, windows["all"], "alpha-sample sensitivity", dall["sample"]["n_valid_transitions"])
    oall = derive_intercepts(panel, g01, 0.0, g10, 0.0, dn, ds, windows["all"], use_ramp=False)
    add("1D, D = full sample (x_(t-1) sample)", oall["alpha01"], oall["alpha10"], 0.0, 0.0,
        panel, windows["all"], "alpha-sample sensitivity twin",
        oall["sample"]["n_valid_transitions"])
    # z re-standardized on the M9 window (ramp r is invariant to affine z maps)
    zst = z_stats["W9"]
    z9 = (z - zst["mean"]) / zst["std_ddof1"]
    sc9 = fit_ramp_scaler(z9, windows[ramp_window], ramp_window)
    p9 = build_covariate_panel(z9, sc9)
    d9 = derive_intercepts(p9, g01, h01, g10, h10, dn, ds, D_end)
    o9 = derive_intercepts(p9, g01, 0.0, g10, 0.0, dn, ds, D_end, use_ramp=False)
    add("2D, z re-standardized on W9  [R3']", d9["alpha01"], d9["alpha10"], h01, h10, p9,
        D_end, "z-scale sensitivity (M9 may have standardized on its own window)",
        d9["sample"]["n_valid_transitions"])
    add("1D, z re-standardized on W9  [R1']", o9["alpha01"], o9["alpha10"], 0.0, 0.0, p9,
        D_end, "z-scale sensitivity twin", o9["sample"]["n_valid_transitions"])
    r_invariance = float(np.nanmax(np.abs(p9.frame["r"].to_numpy() - panel.frame["r"].to_numpy())))

    # --- rd_lag1 file: calendar lag evidence + two extra z values ------------------
    lag1 = load_hourly_z_history(lag1_p)
    zg = complete_hourly_grid(z)
    cal_prev = zg.shift(1).reindex(lag1.index)
    row_prev = z.shift(1).reindex(lag1.index)
    lv, cv, rv = lag1.to_numpy(), cal_prev.to_numpy(), row_prev.to_numpy()
    agree_cal = np.isclose(lv, cv, atol=1e-12, rtol=0)
    cal_undefined = np.isnan(cv)
    row_contradicted = ~np.isnan(rv) & ~np.isclose(lv, rv, atol=1e-12, rtol=0)
    extra = {}
    for lab in lag1.index[cal_undefined]:
        extra[(lab - HOUR).isoformat()] = float(lag1.loc[lab])
    z_sup = pd.concat([z, pd.Series({pd.Timestamp(k): v for k, v in extra.items()})]).sort_index()
    ps = build_covariate_panel(z_sup, sc)
    dsup = derive_intercepts(ps, g01, h01, g10, h10, dn, ds, D_end)
    lag1_evidence = {
        "file": lag1_file, "sha256": sha256_file(lag1_p),
        "rows_total": int(len(lag1)),
        "rows_equal_to_calendar_lag_of_z": int(agree_cal.sum()),
        "rows_calendar_lag_not_computable_from_rd_standardized": int(cal_undefined.sum()),
        "rows_contradicting_row_order_lag": [
            {"label_utc": lab.isoformat(), "rd_lag1": float(lag1.loc[lab]),
             "row_order_value": float(row_prev.loc[lab])}
            for lab in lag1.index[row_contradicted]],
        "rows_not_derivable_from_rd_standardized": [
            {"label_utc": lab.isoformat(), "rd_lag1": float(lag1.loc[lab]),
             "row_order_value": (None if pd.isna(row_prev.loc[lab]) else float(row_prev.loc[lab]))}
            for lab in lag1.index[cal_undefined]],
        "finding": ("RD_lag1 in the shipped file is a CALENDAR one-hour lag: at "
                    "2016-03-27 03:00 UTC it carries z(02:00), an hour absent from "
                    "rd_standardized.csv, not the row-order value z(2016-03-26 23:00). "
                    "The upstream lag was therefore built on a calendar-consistent RD "
                    "series. This supports (does not prove) a calendar-time ramp "
                    "construction; the ramp itself is still unverified."),
        "extra_z_values_recovered": extra,
        "alpha_if_gap_supplemented": {"alpha01": dsup["alpha01"], "alpha10": dsup["alpha10"],
                                      "n_transitions": dsup["sample"]["n_valid_transitions"],
                                      "abs_change_alpha01": abs(dsup["alpha01"] - prim["alpha01"]),
                                      "abs_change_alpha10": abs(dsup["alpha10"] - prim["alpha10"])},
        "used_in_primary": False,
    }

    # --- validation metrics of the primary set --------------------------------
    lab, p01, p10 = historical_path_probabilities(panel, coef, D_end)
    metrics = transition_moment_metrics(p01, p10)
    occ = forward_recursion_occupancy(lab, p01, p10)
    ame01_m9, ame10_m9 = prod["checks"]["ame_over_slope_p01"][0], prod["checks"]["ame_over_slope_p10"][0]
    validation = {
        "ergodic_ratio_occupancy_stress": {
            "value": ergodic_ratio_occupancy(dn, ds), "target": prod["occupancy_stress"],
            "independent_of_parameters": True,
            "note": "(1/d_n)/(1/d_n + 1/d_s) uses only the two targets; NOT a validation"},
        "forward_recursion_occupancy_stress": {
            **occ, "reported_m9": prod["occupancy_stress"],
            "relative_error": occ["mean_stress_probability"] / prod["occupancy_stress"] - 1.0,
            "note": ("M9's value is a smoothed-probability average conditional on prices; "
                     "this is a covariate-conditional chain average -- a gap is expected "
                     "and is not forced")},
        "ame_ratio": {"p01": metrics["ame_ratio_p01"], "p10": metrics["ame_ratio_p10"],
                      "m9_p01": ame01_m9, "m9_p10": ame10_m9},
        "mean_inverse_p": {"p01": metrics["mean_inverse_p01"], "p10": metrics["mean_inverse_p10"],
                           "m9_mean_expected_duration_normal": prod["mean_expected_duration_normal"],
                           "m9_mean_expected_duration_stress": prod["mean_expected_duration_stress"],
                           "note": raw["expected_duration_note"] + " (auxiliary metric)"},
        "achieved_mean_p": {"p01": metrics["mean_p01"], "p10": metrics["mean_p10"]},
        "p01_duration_target_consistency": p01_duration_target_consistency(1.0 / dn, ame01_m9),
        "embeddability_historical_full_sample": historical_embeddability(panel, coef),
        "embeddability_historical_D": historical_embeddability(panel, coef, D_end),
    }
    evidence = ramp_scale_timing_evidence(panel, g01, h01, g10, h10, dn, ds, D_end,
                                          scales=(0.0, 1.0, 1.1))

    # --- assemble ---------------------------------------------------------------
    hist_sha = sha256_file(hist_p)
    base_sha = sha256_file(base_p)
    label_mapping = [
        {"production": "sigma_y_normal (0)", "raw_m9": "sigma1", "value": prod["sigma_y_normal"]},
        {"production": "sigma_y_stress (1)", "raw_m9": "sigma0", "value": prod["sigma_y_stress"]},
        {"production": "gamma01 (normal->stress, RD_lag1)", "raw_m9": "gamma10 RD_lag1",
         "value": prod["gamma01"], "value_used": g01},
        {"production": "gamma10 (stress->normal, RD_lag1)", "raw_m9": "gamma01 RD_lag1",
         "value": prod["gamma10"], "value_used": g10},
        {"production": "h01 (normal->stress, ramp)", "raw_m9": "gamma10 RD_Ramp_1h_lag1",
         "value": prod["h01"], "value_used": h01},
        {"production": "h10 (stress->normal, ramp)", "raw_m9": "gamma01 RD_Ramp_1h_lag1",
         "value": prod["h10"], "value_used": h10},
        {"production": "d_n mean normal duration (h)", "raw_m9": "mean_duration_state1", "value": dn},
        {"production": "d_s mean stress duration (h)", "raw_m9": "mean_duration_state0", "value": ds},
        {"production": "pi_stress occupancy", "raw_m9": "occupancy[0]", "value": prod["occupancy_stress"]},
    ]
    provenance = {
        "tvtp2.alpha01": ("derived: brentq root of mean_D logistic(a + gamma01 z_(t-1) + h01 r_(t-1)) "
                          f"= 1/d_n on D = {sample_window} (paired x_(t-1)); moment matching, NOT an "
                          "MLE estimate, no standard error"),
        "tvtp2.alpha10": ("derived: brentq root of mean_D logistic(a + gamma10 z_(t-1) + h10 r_(t-1)) "
                          f"= 1/d_s on D = {sample_window}; moment matching, NOT an MLE estimate"),
        "tvtp2.gamma01": ("estimated (transferred): M9 raw gamma10 of RD_lag1, regime-label swapped; "
                          "value exactly as in the production yaml (10 dp)"),
        "tvtp2.gamma10": ("estimated (transferred): M9 raw gamma01 of RD_lag1, regime-label swapped; "
                          "value exactly as in the production yaml (10 dp)"),
        "tvtp2.h01": ("estimated (transferred): M9 raw gamma10 of RD_Ramp_1h_lag1 (full precision), "
                      "regime-label swapped; applied to a RECONSTRUCTED ramp covariate"),
        "tvtp2.h10": ("estimated (transferred): M9 raw gamma01 of RD_Ramp_1h_lag1 (full precision), "
                      "regime-label swapped; applied to a RECONSTRUCTED ramp covariate"),
        "covariates.ramp.scaler": (f"reconstructed: train-only mean/std (ddof = 1) of hourly dz on "
                                   f"{ramp_window} (labels <= {sc.window_end_utc.isoformat()})"),
        "covariates.ramp.definition": "assumed: " + RAMP_STATUS,
        "covariates.z": ("inherited: rd_standardized.csv, TRY-window standardization reconstructed in "
                         "markov_adapter (ddof = 1 assumed)"),
        "transition_premium": ZERO_PREMIUM_NOTE,
    }
    yaml_blob: Dict[str, Any] = {
        "schema": "tvtp2_frozen_parameters/v1",
        "tvtp_mode": "rd_ramp_2d_experimental",
        "parameter_set_name": PARAMETER_SET_NAME,
        "status": STATUS_EXPERIMENTAL,
        "label": EXPERIMENTAL_LABEL,
        "verified_reproduction_of_m9": False,
        "generated_by": "scripts/tvtp_derivation/derive_tvtp2_parameters.py",
        "hash_convention": "sha256 of the file content with CRLF normalized to LF",
        "base_parameters": {
            "file": base_params_file, "sha256": base_sha,
            "valuation_utc": val.isoformat(),
            "inherited_unchanged": ["scale_P", "phi", "kappa_per_hour", "sigma_y_normal",
                                    "sigma_y_stress", "pi_filtered", "m9_stationary_pi",
                                    "spot_price_TRY_MWh", "valuation_utc"],
            "not_used_in_this_mode": ["tvtp.alpha01", "tvtp.alpha10"],
        },
        "tvtp2": {"alpha01": coef.alpha01, "gamma01": g01, "h01": h01,
                  "alpha10": coef.alpha10, "gamma10": g10, "h10": h10,
                  "covariates": [Z_COVARIATE, RAMP_COVARIATE]},
        "covariates": {
            "lag_hours": 1,
            "history_file": history_file, "history_sha256": hist_sha,
            "z": {"name": Z_COVARIATE, "units": "standardized (dimensionless)",
                  "definition": "z_t = (RD_t - m_z)/s_z, RD = Demand - Wind - Solar (MWh)",
                  "scaler_window": f"TRY window <= {TRY_WINDOW_END_UTC} (61 361 rows)",
                  "status": "inherited; scaler reconstructed (markov_adapter), ddof = 1 assumed"},
            "ramp": {"name": RAMP_COVARIATE, "units": "standardized (dimensionless)",
                     "definition": RAMP_DEFINITION, "status": RAMP_STATUS,
                     "scaler": sc.as_dict()},
        },
        "derivation": {
            "method": "scipy.optimize.brentq on [-20, 20], xtol = rtol = 1e-12, two independent "
                      "single-root equations (moment matching)",
            "sample_window": sample_window,
            "sample": prim["sample"],
            "targets": {"mean_p01": 1.0 / dn, "mean_p10": 1.0 / ds,
                        "duration_normal_h": dn, "duration_stress_h": ds},
            "roots": {"p01": prim["root_p01"], "p10": prim["root_p10"]},
            "audit": f"{DEFAULT_AUDIT_DIR}/tvtp2_derivation_audit.json",
        },
        "m9_source": {
            "bundle_dir": bundle_dir, "sha256": raw["sha256"], "spec": raw["spec"],
            "n_train": raw["n_train"], "n_total": raw["n_total"],
            "w9_window_end_utc_inferred": windows["W9"].isoformat(),
            "joint_estimation_evidence": prod["checks"]["joint_estimation_evidence"],
            "label_mapping": label_mapping,
        },
        "transition_premium": {"eta01": 0.0, "eta10": 0.0, "status": ZERO_PREMIUM_NOTE},
        "provenance": provenance,
    }
    audit: Dict[str, Any] = {
        "title": "2D TVTP derivation audit (RD_lag1 + RD_Ramp_1h_lag1)",
        "status": STATUS_EXPERIMENTAL, "label": EXPERIMENTAL_LABEL,
        "verified_reproduction_of_m9": False,
        "hash_convention": "sha256 of the file content with CRLF normalized to LF",
        "inputs": {"history_file": history_file, "history_sha256": hist_sha,
                   "base_params_file": base_params_file, "base_params_sha256": base_sha,
                   "bundle": raw["sha256"], "lag1_file": lag1_file},
        "source_search_for_ramp_definition": SOURCE_SEARCH,
        "label_mapping": label_mapping,
        "label_swap_checks": prod["checks"],
        "slope_precision": {
            "gamma01_csv": prod["gamma01"], "gamma01_used_yaml": g01,
            "gamma10_csv": prod["gamma10"], "gamma10_used_yaml": g10,
            "abs_rounding_gamma01": abs(prod["gamma01"] - g01),
            "abs_rounding_gamma10": abs(prod["gamma10"] - g10)},
        "grid": {"first_label_utc": z.index[0].isoformat(), "last_label_utc": z.index[-1].isoformat(),
                 "n_rows": int(len(z)), "n_complete_grid_labels": int(len(panel.frame)),
                 "missing_labels_utc": [x.isoformat() for x in panel.missing_labels]},
        "windows": {k: v.isoformat() for k, v in windows.items()},
        "w9_inference": (f"W9 = first n_train = {raw['n_train']} rows of the history (M9 run_summary); "
                         f"n_total = {raw['n_total']} equals the history row count"),
        "z_scaler_window_statistics": z_stats,
        "ramp_scalers": {k: v.as_dict() for k, v in scalers.items()},
        "ramp_scaler_used": ramp_window,
        "ramp_invariance_under_z_rescaling_max_abs": r_invariance,
        "primary": {"coefficients": coef.as_dict(), "sample": prim["sample"],
                    "roots": {"p01": prim["root_p01"], "p10": prim["root_p10"]}},
        "single_covariate_same_sample": {"alpha01": one["alpha01"], "alpha10": one["alpha10"],
                                         "roots": {"p01": one["root_p01"], "p10": one["root_p10"]}},
        "production_1d_rederived": {"alpha01": prod1d_01, "alpha10": prod1d_10,
                                    "yaml_alpha01": float(base["tvtp"]["alpha01"]),
                                    "yaml_alpha10": float(base["tvtp"]["alpha10"])},
        "variants": variants,
        "validation_metrics": validation,
        "ramp_scale_and_timing_evidence": {
            "rows": evidence,
            "m9_mean_expected_duration_stress": prod["mean_expected_duration_stress"],
            "interpretation": ("E[1/p10] discriminates the ramp SCALE (standardized ~8.58 vs "
                               "raw ~7.68, M9 8.585) but not its direction or lag (all timing "
                               "variants land within 8.48-8.65): scale is supported, "
                               "direction/lag are unverified")},
        "rd_lag1_file_evidence": lag1_evidence,
        "decisions": {
            "ramp_scaler_window": f"{ramp_window} (h slopes come from M9, whose training window "
                                  "W9 is inferred; the p10 moment check mildly favours it)",
            "alpha_sample": f"{sample_window} (durations are M9 diagnostics); full sample "
                            "reported as a sensitivity",
            "p01_target_conflict": "documented limitation, target kept (same as 1D production)",
            "time_alignment": "production l(tau) = t_v + floor(tau) - 1h kept; +/-1h reported "
                              "as a separate sensitivity",
            "non_embeddable_rows": "continuous-time mode REJECTS s >= 1; no silent clipping; "
                                   "discrete-time switching mode not implemented",
        },
    }
    return yaml_blob, audit


def _num(x: Any, nd: int = 6) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        ax = abs(float(x))
        if ax != 0 and (ax < 1e-4 or ax >= 1e6):
            return f"{float(x):.3e}"
        return f"{float(x):.{nd}f}"
    return str(x)


def render_derivation_markdown(audit: Dict[str, Any]) -> str:
    """Human-readable companion of the derivation audit JSON."""
    L: List[str] = [f"# {audit['title']}", "",
                    f"**Status:** `{audit['status']}` — {audit['label']}.", "",
                    "> EXPERIMENTAL. The ramp covariate definition is a reconstruction; "
                    "nothing here is a reproduction of M9 and the accepted paper results "
                    "are unchanged.", "",
                    "## 1. Search for the original ramp definition", "",
                    "| source | result |", "|---|---|"]
    L += [f"| {r['source']} | {r['result']} |" for r in audit["source_search_for_ramp_definition"]]
    L += ["", "## 2. Label mapping (raw M9 → production, one atomic swap)", "",
          "| production | raw M9 | value |", "|---|---|---|"]
    L += [f"| {r['production']} | {r['raw_m9']} | {_num(r.get('value_used', r['value']), 10)} |"
          for r in audit["label_mapping"]]
    ck = audit["label_swap_checks"]
    L += ["", f"- AME and slope share their sign in every equation: **{_num(ck['ame_and_slope_share_sign_in_every_equation'])}**",
          f"- AME/slope, p01 equation: {ck['ame_over_slope_p01'][0]:.12f} (RD) vs "
          f"{ck['ame_over_slope_p01'][1]:.12f} (ramp), |diff| = {ck['ame_over_slope_abs_diff_p01']:.1e}",
          f"- AME/slope, p10 equation: {ck['ame_over_slope_p10'][0]:.12f} (RD) vs "
          f"{ck['ame_over_slope_p10'][1]:.12f} (ramp), |diff| = {ck['ame_over_slope_abs_diff_p10']:.1e}",
          f"- Joint-estimation evidence: **{_num(ck['joint_estimation_evidence'])}** — {ck['joint_estimation_note']}",
          f"- RD slopes used are the production-yaml values (10 dp); rounding vs CSV: "
          f"{audit['slope_precision']['abs_rounding_gamma01']:.1e}, {audit['slope_precision']['abs_rounding_gamma10']:.1e}",
          "", "## 3. Hourly grid, windows and scalers", "",
          f"- History: {audit['grid']['first_label_utc']} → {audit['grid']['last_label_utc']}, "
          f"{audit['grid']['n_rows']:,} rows on {audit['grid']['n_complete_grid_labels']:,} grid labels; "
          f"missing: {', '.join(audit['grid']['missing_labels_utc'])}",
          f"- {audit['w9_inference']}", "",
          "| window | end (UTC) | z rows | z mean | z sd | dz n | m_r | s_r |",
          "|---|---|---|---|---|---|---|---|"]
    for nm, st in audit["z_scaler_window_statistics"].items():
        rs = audit["ramp_scalers"][nm]
        L.append(f"| {nm} | {st['window_end_utc']} | {st['n_rows']:,} | {st['mean']:.6f} | "
                 f"{st['std_ddof1']:.6f} | {rs['n']:,} | {rs['mean']:.3e} | {rs['std']:.6f} |")
    pr = audit["primary"]
    smp = pr["sample"]
    L += ["", f"Ramp scaler used: **{audit['ramp_scaler_used']}**. Ramp values are invariant to an "
          f"affine re-scaling of z (max |Δr| = {audit['ramp_invariance_under_z_rescaling_max_abs']:.1e}).",
          "", "## 4. Transition sample D and intercepts", "",
          f"- D: {smp['first_valid_transition_utc']} → {smp['last_valid_transition_utc']}, "
          f"{smp['n_valid_transitions']:,} transitions of {smp['n_grid_labels']:,} grid labels",
          f"- dropped (sample start, rule: {smp['rule_sample_start']}): {smp['n_dropped_sample_start']} → "
          f"{', '.join(smp['dropped_sample_start_labels'])}",
          f"- dropped (missing observation, rule: {smp['rule_missing_observation']}): "
          f"{smp['n_dropped_missing_observation']} → {', '.join(smp['dropped_missing_observation_labels'])}",
          f"- kept although the own label has no z (its x_(t-1) is observed): "
          f"{', '.join(smp['valid_with_missing_own_label']) or 'none'}", "",
          "| root | alpha | target mean p | achieved | residual | iterations |", "|---|---|---|---|---|---|"]
    for k in ("p01", "p10"):
        r = pr["roots"][k]
        L.append(f"| {r['transition']} | {r['alpha']:.10f} | {r['target_mean_p']:.8f} | "
                 f"{r['achieved_mean_p']:.8f} | {r['residual']:.1e} | {r['iterations']} |")
    L += ["", "## 5. Variants (derived, not re-estimated)", "",
          "| variant | α01 | α10 | n | occupancy (fwd rec.) | AME p01 | AME p10 | E[1/p01] | E[1/p10] | max s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for v in audit["variants"]:
        L.append(f"| {v['variant']} | {v['alpha01']:.6f} | {v['alpha10']:.6f} | {_num(v['n_transitions'])} | "
                 f"{v['forward_recursion_occupancy_stress']:.4f} | {v['ame_ratio_p01']:.4f} | "
                 f"{v['ame_ratio_p10']:.4f} | {v['mean_inverse_p01']:.3f} | {v['mean_inverse_p10']:.3f} | "
                 f"{v['max_s']:.4f} |")
    vm = audit["validation_metrics"]
    L += ["", "## 6. Validation metrics of the primary set (reported, never targeted)", "",
          "| metric | this set | M9 / target | comment |", "|---|---|---|---|",
          f"| ergodic ratio occupancy | {vm['ergodic_ratio_occupancy_stress']['value']:.5f} | "
          f"{vm['ergodic_ratio_occupancy_stress']['target']:.6f} | {vm['ergodic_ratio_occupancy_stress']['note']} |",
          f"| forward-recursion occupancy | {vm['forward_recursion_occupancy_stress']['mean_stress_probability']:.4f} | "
          f"{vm['forward_recursion_occupancy_stress']['reported_m9']:.6f} | rel. error "
          f"{100 * vm['forward_recursion_occupancy_stress']['relative_error']:+.1f}% — expected gap, not forced |",
          f"| AME ratio p01 | {vm['ame_ratio']['p01']:.4f} | {vm['ame_ratio']['m9_p01']:.6f} | |",
          f"| AME ratio p10 | {vm['ame_ratio']['p10']:.4f} | {vm['ame_ratio']['m9_p10']:.6f} | |",
          f"| E[1/p01] | {vm['mean_inverse_p']['p01']:.3f} | {vm['mean_inverse_p']['m9_mean_expected_duration_normal']:.6f} | auxiliary |",
          f"| E[1/p10] | {vm['mean_inverse_p']['p10']:.3f} | {vm['mean_inverse_p']['m9_mean_expected_duration_stress']:.6f} | auxiliary |"]
    pc = vm["p01_duration_target_consistency"]
    L += ["", f"**p01 target conflict.** Upper bound pbar(1-pbar) = {pc['upper_bound_E_p_1mp']:.5f} "
          f"< M9 AME ratio {pc['m9_ame_ratio']:.5f} → compatible: **{_num(pc['compatible'])}**; "
          f"M9's own mean p01 must be ≥ {pc['implied_minimum_mean_p_of_m9_path']:.5f}. "
          "Present in the 1D production set as well; the ramp does not resolve it.", ""]
    for key, title in (("embeddability_historical_full_sample", "full historical sample"),
                       ("embeddability_historical_D", "sample D")):
        e = vm[key]
        L.append(f"- Embeddability, {title}: n = {e['n_rows']:,}, s ≥ 1: {e['n_s_ge_1']} "
                 f"({100 * e['share_s_ge_1']:.3f}%), s ≥ 0.95: {e['n_s_ge_0.95']}, max s = "
                 f"{e['max_s']:.4f} at {e['argmax_label']}, 99.99% quantile {e['quantile_0.9999_s']:.4f}")
    ev = audit["ramp_scale_and_timing_evidence"]
    L += ["", "## 7. Ramp scale and timing evidence", "",
          "| construction | n | α10 | E[1/p10] | E[1/p01] |", "|---|---|---|---|---|"]
    L += [f"| {r['variant']} | {r['n']:,} | {r['alpha10']:.6f} | {r['mean_inverse_p10']:.3f} | "
          f"{r['mean_inverse_p01']:.3f} |" for r in ev["rows"]]
    L += ["", f"M9 mean expected stress duration: {ev['m9_mean_expected_duration_stress']:.6f}. "
          f"{ev['interpretation']}.", ""]
    le = audit["rd_lag1_file_evidence"]
    contra = "; ".join(f"{r['label_utc']}: file {r['rd_lag1']:.6f} vs row-order {r['row_order_value']:.6f}"
                       for r in le["rows_contradicting_row_order_lag"]) or "none"
    L += ["## 8. `rd_lag1_standardized.csv` evidence", "",
          f"- {le['rows_equal_to_calendar_lag_of_z']:,} / {le['rows_total']:,} rows equal the calendar "
          f"lag of z; {le['rows_calendar_lag_not_computable_from_rd_standardized']} rows need an hour "
          "that rd_standardized.csv does not contain.",
          f"- Rows contradicting a row-order lag: {contra}.",
          f"- {le['finding']}",
          f"- Recoverable extra z values: {le['extra_z_values_recovered']} (not used in the primary set; "
          f"effect on α if used: |Δα01| = {le['alpha_if_gap_supplemented']['abs_change_alpha01']:.1e}, "
          f"|Δα10| = {le['alpha_if_gap_supplemented']['abs_change_alpha10']:.1e}).", "",
          "## 9. Decisions", ""]
    L += [f"- **{k}**: {v}" for k, v in audit["decisions"].items()]
    return "\n".join(L) + "\n"
