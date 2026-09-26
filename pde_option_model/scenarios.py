"""Deterministic residual-demand scenario paths z(t).

First implementation treats forecast residual demand as a deterministic
exogenous path RD = RD(t) (two coupled 1-D PDEs); a stochastic residual-demand
state is a later 2-D extension (see docs/methodology.md section 8).

Scenario paths are specified directly in STANDARDIZED units z (TRY-train
standardization), which avoids any dependence on the unexported raw-unit
scaler beyond the reconstruction documented in markov_adapter.

Modes
-----
constant     z(t) = offset
climatology  z(t) = month-x-hour mean of the historical standardized RD over
             the TRY training window, evaluated on the contract's calendar
             hours, plus offset
custom       hourly z values from a CSV (columns: datetime, z)

The transition covariate is RD_{t-1}: `z_for_transitions` applies the
one-hour lag; the emission loading rho uses the contemporaneous path.

Two-covariate paths (EXPERIMENTAL)
----------------------------------
:class:`CovariatePathBuilder` builds z and the standardized ramp r TOGETHER on
hourly UTC labels and only then aligns them to a solver grid:

    l(tau) = floor(t_v + tau - lag)            (lag = 1 h: t_v + floor(tau) - 1 h)
    z(tau) = z_hour[l(tau)],   r(tau) = (z_hour[l] - z_hour[l - 1h] - m_r) / s_r

The label map is computed with the SAME calls as :meth:`ScenarioBuilder.build`
(``make_time_index`` on the 0.25 h master grid), so in the constant and
climatology modes the z path is bit-identical to the single-covariate one and
a two-covariate model with zero ramp slopes reproduces the production prices.
Covariates are never interpolated onto hours: the custom CSV must contain
every required label (missing labels are listed in the error), and the ramp is
never defaulted to zero.  Between master nodes the solvers interpolate the
covariates linearly, exactly as in the single-covariate production path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .grid import TimeGrid
from .transformations import make_time_index

logger = logging.getLogger(__name__)

ScenarioMode = Literal["constant", "climatology", "custom"]
InitialHours = Literal["scenario", "observed"]
_HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    mode: ScenarioMode = "climatology"
    offset: float = 0.0
    custom_csv: Optional[str] = None
    # Two-covariate paths only: "scenario" (default, production rule) takes the
    # labels <= t_v from the scenario too; "observed" uses the observed history
    # at those labels (t_v - 2h, t_v - 1h, t_v for lag 1) and never offsets them.
    initial_hours: InitialHours = "scenario"


@dataclass
class ScenarioPath:
    name: str
    times_hours: np.ndarray
    z: np.ndarray                       # contemporaneous z(t) on solver grid
    z_lagged: np.ndarray                # z(t - lag) for the transition covariate

    def z_for_theta(self) -> np.ndarray:
        return self.z

    def z_for_transitions(self) -> np.ndarray:
        return self.z_lagged


class ScenarioBuilder:
    """Builds z(t) paths aligned to a contract's solver time grid."""

    def __init__(self, z_history: pd.Series, train_end: pd.Timestamp,
                 covariate_lag_hours: float = 1.0) -> None:
        if z_history.index.tz is None:
            raise ValueError("z_history index must be tz-aware (UTC)")
        self.z_history = z_history
        self.lag = float(covariate_lag_hours)
        train = z_history.loc[:train_end]
        loc = train.index + pd.Timedelta(hours=3)   # Istanbul local time
        self._clim = train.groupby([loc.month, loc.hour]).mean()
        self._clim_global = float(train.mean())
        logger.info("scenario climatology built from %d training hours "
                    "(global mean z=%.4f)", len(train), self._clim_global)

    # ------------------------------------------------------------------
    def _climatology_values(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        loc = timestamps + pd.Timedelta(hours=3)
        keys = list(zip(loc.month, loc.hour))
        out = np.array([self._clim.get(k, self._clim_global) for k in keys],
                       dtype=float)
        return out

    def build(self, spec: ScenarioSpec, tgrid: TimeGrid) -> ScenarioPath:
        if spec.initial_hours != "scenario":
            raise ValueError("initial_hours='observed' is only supported by the "
                             "two-covariate CovariatePathBuilder")
        t = tgrid.times_hours
        ts_now = make_time_index(tgrid.valuation_utc, t)
        ts_lag = make_time_index(tgrid.valuation_utc, t - self.lag)
        if spec.mode == "constant":
            z_now = np.full(t.size, spec.offset)
            z_lag = z_now.copy()
        elif spec.mode == "climatology":
            z_now = self._climatology_values(ts_now) + spec.offset
            z_lag = self._climatology_values(ts_lag) + spec.offset
        elif spec.mode == "custom":
            if not spec.custom_csv:
                raise ValueError("custom scenario requires custom_csv")
            df = pd.read_csv(Path(spec.custom_csv))
            idx = pd.to_datetime(df["datetime"], utc=True)
            ser = pd.Series(df["z"].to_numpy(float), index=idx).sort_index()
            hours = (ser.index - tgrid.valuation_utc).total_seconds() / 3600.0
            z_now = np.interp(t, hours, ser.to_numpy()) + spec.offset
            z_lag = np.interp(t - self.lag, hours, ser.to_numpy()) + spec.offset
        else:  # pragma: no cover
            raise ValueError(f"unknown scenario mode {spec.mode!r}")
        logger.info("scenario '%s' (%s, offset %+0.2f): z in [%.3f, %.3f]",
                    spec.name, spec.mode, spec.offset, z_now.min(), z_now.max())
        return ScenarioPath(name=spec.name, times_hours=t, z=z_now, z_lagged=z_lag)


def default_scenarios(base_offset: float = 0.0,
                      high_offset: float = 1.5,
                      low_offset: float = -1.5,
                      mode: ScenarioMode = "climatology") -> list[ScenarioSpec]:
    return [
        ScenarioSpec(name="base", mode=mode, offset=base_offset),
        ScenarioSpec(name="high_rd", mode=mode, offset=high_offset),
        ScenarioSpec(name="low_rd", mode=mode, offset=low_offset),
    ]


# ---------------------------------------------------------------------------
# two-covariate (z, ramp) paths -- EXPERIMENTAL
# ---------------------------------------------------------------------------
@dataclass
class TVTPCovariatePath:
    """Hourly z / ramp path aligned to a 0.25 h master grid on [0, horizon].

    ``z_lagged(t)`` / ``ramp_lagged(t)`` evaluate the aligned covariates on any
    solver grid inside [0, horizon] (never extrapolated).  One object feeds
    the moment ODE, the PDE and the Monte Carlo, so all three see the same
    (z, r) path.
    """

    name: str
    mode: str
    offset: float
    initial_hours: str
    valuation_utc: pd.Timestamp
    horizon_hours: float
    lag_hours: float
    labels: pd.DatetimeIndex
    z_hourly: np.ndarray
    dz_hourly: np.ndarray
    ramp_hourly: np.ndarray
    hourly_source: np.ndarray
    master_times: np.ndarray
    master_labels: pd.DatetimeIndex
    z_master: np.ndarray
    ramp_master: np.ndarray
    ramp_scaler: Any = None
    source: Dict[str, Any] = field(default_factory=dict)
    units: str = "standardized"

    # -- evaluation -------------------------------------------------------
    def _check_times(self, t: np.ndarray) -> np.ndarray:
        from .generator import CovariateError
        t = np.asarray(t, dtype=float)
        if t.size and (float(t.min()) < -1e-9 or float(t.max()) > self.horizon_hours + 1e-9):
            raise CovariateError(
                f"covariate path '{self.name}' covers [0, {self.horizon_hours:g}] h but "
                f"times in [{float(t.min()):g}, {float(t.max()):g}] were requested; "
                "build a longer path instead of extrapolating")
        return t

    @property
    def has_ramp(self) -> bool:
        return self.ramp_scaler is not None

    def z_lagged(self, t: np.ndarray) -> np.ndarray:
        t = self._check_times(t)
        return np.interp(t, self.master_times, self.z_master)

    def ramp_lagged(self, t: np.ndarray) -> np.ndarray:
        from .generator import CovariateError
        if not self.has_ramp:
            raise CovariateError(f"covariate path '{self.name}' has no ramp standardization")
        t = self._check_times(t)
        return np.interp(t, self.master_times, self.ramp_master)

    def covariates(self, t: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """(z_lagged, ramp_lagged) on ``t``; the ramp is None without a scaler."""
        return self.z_lagged(t), (self.ramp_lagged(t) if self.has_ramp else None)

    def z_lagged_fn(self):
        """Callable for the single-covariate API (``z_lagged_fn=``)."""
        return self.z_lagged

    # -- reporting --------------------------------------------------------
    def used_hour_positions(self) -> np.ndarray:
        pos = ((self.master_labels - self.labels[0]) // _HOUR).to_numpy()
        return np.unique(pos.astype(int))

    def hourly_frame(self) -> pd.DataFrame:
        used = np.zeros(len(self.labels), dtype=bool)
        used[self.used_hour_positions()] = True
        return pd.DataFrame({"label_utc": [x.isoformat() for x in self.labels],
                             "z": self.z_hourly, "dz": self.dz_hourly,
                             "ramp": self.ramp_hourly, "source": self.hourly_source,
                             "used_by_solver_grid": used})

    def embeddability(self, coefficients) -> Dict[str, Any]:
        """s = p01 + p10 audit on the hour labels the solver grid uses."""
        from .generator import TVTP2Coefficients, embeddability_report
        pos = self.used_hour_positions()
        z = self.z_hourly[pos]
        if isinstance(coefficients, TVTP2Coefficients):
            p01, p10 = coefficients.probabilities(z, self.ramp_hourly[pos])
        else:
            p01, p10 = coefficients.probabilities(z)
        rep = embeddability_report(p01, p10, labels=list(self.labels[pos]))
        rep["scope"] = f"hour labels used by path '{self.name}'"
        return rep

    def describe(self) -> Dict[str, Any]:
        pos = self.used_hour_positions()
        d: Dict[str, Any] = {
            "name": self.name, "mode": self.mode, "offset": float(self.offset),
            "initial_hours": self.initial_hours, "lag_hours": float(self.lag_hours),
            "alignment": ("l(tau) = floor(t_v + tau - lag); covariates evaluated at hour "
                          "labels, sampled on a master grid, linear between master nodes "
                          "(same as the single-covariate production path)"),
            "valuation_utc": self.valuation_utc.isoformat(),
            "horizon_hours": float(self.horizon_hours),
            "n_master_nodes": int(self.master_times.size),
            "first_label_utc": self.labels[0].isoformat(),
            "last_label_utc": self.labels[-1].isoformat(),
            "n_hour_labels": int(len(self.labels)),
            "z_range_used": [float(self.z_hourly[pos].min()), float(self.z_hourly[pos].max())],
            "units": self.units,
            "n_observed_labels": int(np.sum(self.hourly_source == "observed")),
            "source": dict(self.source),
        }
        if self.has_ramp:
            d["ramp_range_used"] = [float(self.ramp_hourly[pos].min()),
                                    float(self.ramp_hourly[pos].max())]
            d["ramp_scaler"] = self.ramp_scaler.as_dict()
        return d


class CovariatePathBuilder:
    """Builds :class:`TVTPCovariatePath` objects (z and ramp on the same labels)."""

    def __init__(self, z_history: pd.Series, train_end: pd.Timestamp,
                 covariate_lag_hours: float = 1.0, ramp_scaler: Any = None,
                 history_file: Optional[str] = None) -> None:
        from .tvtp2 import validate_hourly_utc_index
        validate_hourly_utc_index(z_history.index, history_file or "z_history")
        lag = float(covariate_lag_hours)
        if lag < 0 or lag != int(lag):
            raise ValueError("the covariate lag must be a whole, non-negative number of hours")
        self.z_history = z_history
        self.lag = lag
        self.ramp_scaler = ramp_scaler
        self.history_file = history_file
        self.train_end = pd.Timestamp(train_end)
        # the single-covariate builder supplies the climatology, bit-for-bit
        self._clim = ScenarioBuilder(z_history, self.train_end, lag)

    @staticmethod
    def master_steps(tau_hours: float, grid_settings: Any = None) -> int:
        """Master-grid steps; identical rule to run_pde._build_tvtp_scenario."""
        n_pde = grid_settings.n_steps(tau_hours) if grid_settings is not None else 0
        return max(int(n_pde), int(np.ceil(tau_hours / 0.25)))

    def verify_ramp_scaler(self) -> None:
        """Recompute (m_r, s_r) from the history on the frozen window; refuse a mismatch."""
        from .generator import CovariateError
        from .tvtp2 import fit_ramp_scaler
        s = self.ramp_scaler
        if s is None:
            raise CovariateError("no ramp scaler to verify")
        fresh = fit_ramp_scaler(self.z_history, s.window_end_utc, s.window_name,
                                ddof=s.ddof, window_start=s.window_start_utc)
        if not (s.matches(fresh) and fresh.n == s.n):
            raise CovariateError(
                f"ramp standardization mismatch: frozen (m_r={s.mean:.12g}, s_r={s.std:.12g}, "
                f"n={s.n}) vs recomputed from the history (m_r={fresh.mean:.12g}, "
                f"s_r={fresh.std:.12g}, n={fresh.n}); the ramp units would be inconsistent")

    # ------------------------------------------------------------------
    def _read_custom(self, spec: ScenarioSpec, labels: pd.DatetimeIndex) -> np.ndarray:
        from .tvtp2 import CovariateDataError, validate_hourly_utc_index
        if not spec.custom_csv:
            raise ValueError("custom scenario requires custom_csv")
        p = Path(spec.custom_csv)
        # exact decimal -> double parsing of user-supplied scenario values
        df = pd.read_csv(p, float_precision="round_trip")
        if "datetime" not in df.columns or "z" not in df.columns:
            raise CovariateDataError(f"{p}: expected columns 'datetime' and 'z'")
        idx = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
        validate_hourly_utc_index(idx, str(p))
        ser = pd.Series(df["z"].to_numpy(dtype=float), index=idx).sort_index()
        missing = labels.difference(ser.index)
        if len(missing):
            shown = ", ".join(x.isoformat() for x in missing[:24])
            more = f" ... (+{len(missing) - 24} more)" if len(missing) > 24 else ""
            raise CovariateDataError(
                f"{p}: custom z path lacks {len(missing)} required hour label(s) in "
                f"[{labels[0].isoformat()}, {labels[-1].isoformat()}]: {shown}{more}. "
                "Hours are never interpolated or filled.")
        vals = ser.reindex(labels).to_numpy(dtype=float)
        bad = labels[~np.isfinite(vals)]
        if len(bad):
            raise CovariateDataError(f"{p}: non-finite z at required labels: "
                                     + ", ".join(x.isoformat() for x in bad[:24]))
        return vals

    def _hourly_z(self, spec: ScenarioSpec, labels: pd.DatetimeIndex,
                  valuation_utc: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        from .tvtp2 import CovariateDataError
        if spec.mode == "constant":
            z = np.full(len(labels), float(spec.offset))
        elif spec.mode == "climatology":
            z = self._clim._climatology_values(labels) + spec.offset
        elif spec.mode == "custom":
            z = self._read_custom(spec, labels) + spec.offset
        else:
            raise ValueError(f"unknown scenario mode {spec.mode!r}")
        src = np.full(len(labels), "scenario", dtype=object)
        if spec.initial_hours == "observed":
            m = np.asarray(labels <= valuation_utc)
            obs = self.z_history.reindex(labels[m]).to_numpy(dtype=float)
            if not np.all(np.isfinite(obs)):
                miss = labels[m][~np.isfinite(obs)]
                raise CovariateDataError(
                    "observed initial hours requested but the history lacks: "
                    + ", ".join(x.isoformat() for x in miss))
            z = np.asarray(z, dtype=float).copy()
            z[m] = obs
            src[m] = "observed"
        elif spec.initial_hours != "scenario":
            raise ValueError(f"unknown initial_hours rule {spec.initial_hours!r}")
        return np.asarray(z, dtype=float), src

    def build(self, spec: ScenarioSpec, valuation_utc: pd.Timestamp,
              maturity_utc: Optional[pd.Timestamp] = None,
              horizon_hours: Optional[float] = None,
              n_master: Optional[int] = None, grid_settings: Any = None
              ) -> TVTPCovariatePath:
        """Path on [0, tau] (tau from ``maturity_utc`` or ``horizon_hours``)."""
        if (maturity_utc is None) == (horizon_hours is None):
            raise ValueError("give exactly one of maturity_utc or horizon_hours")
        mat = (maturity_utc if maturity_utc is not None
               else valuation_utc + pd.Timedelta(hours=float(horizon_hours)))
        tau = (mat - valuation_utc).total_seconds() / 3600.0
        n = int(n_master) if n_master is not None else self.master_steps(tau, grid_settings)
        tgrid = TimeGrid(valuation_utc, mat, n)
        t = tgrid.times_hours
        # identical label map to ScenarioBuilder.build (floor of the lagged stamp)
        ts_lag = make_time_index(tgrid.valuation_utc, t - self.lag)
        master_labels = ts_lag.floor("h")
        labels = pd.date_range(master_labels.min() - _HOUR, master_labels.max(), freq="h")
        z_h, src = self._hourly_z(spec, labels, tgrid.valuation_utc)
        dz = np.full(len(labels), np.nan)
        dz[1:] = z_h[1:] - z_h[:-1]
        ramp = (self.ramp_scaler.standardize(dz) if self.ramp_scaler is not None
                else np.full(len(labels), np.nan))
        pos = ((master_labels - labels[0]) // _HOUR).to_numpy().astype(int)
        if pos.min() < 1:
            raise AssertionError("master grid maps onto the label without a predecessor")
        z_m = z_h[pos]
        r_m = ramp[pos]
        if not np.all(np.isfinite(z_m)) or (self.ramp_scaler is not None
                                            and not np.all(np.isfinite(r_m))):
            raise AssertionError("non-finite covariate on the master grid")
        source: Dict[str, Any] = {"history_file": self.history_file,
                                  "climatology_train_end_utc": self.train_end.isoformat()}
        if spec.mode == "custom":
            from .tvtp2 import sha256_file
            source["custom_csv"] = str(spec.custom_csv)
            source["custom_csv_sha256"] = sha256_file(spec.custom_csv)
        path = TVTPCovariatePath(
            name=spec.name, mode=spec.mode, offset=float(spec.offset),
            initial_hours=spec.initial_hours, valuation_utc=tgrid.valuation_utc,
            horizon_hours=float(tau), lag_hours=self.lag, labels=labels,
            z_hourly=z_h, dz_hourly=dz, ramp_hourly=ramp, hourly_source=src,
            master_times=t, master_labels=master_labels, z_master=z_m, ramp_master=r_m,
            ramp_scaler=self.ramp_scaler, source=source)
        logger.info("covariate path '%s' (%s, offset %+0.2f, initial hours %s): %d labels "
                    "%s..%s, z in [%.3f, %.3f]", spec.name, spec.mode, spec.offset,
                    spec.initial_hours, len(labels), labels[0], labels[-1],
                    z_m.min(), z_m.max())
        return path
