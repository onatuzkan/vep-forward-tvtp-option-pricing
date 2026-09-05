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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .grid import TimeGrid
from .transformations import make_time_index

logger = logging.getLogger(__name__)

ScenarioMode = Literal["constant", "climatology", "custom"]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    mode: ScenarioMode = "climatology"
    offset: float = 0.0
    custom_csv: Optional[str] = None


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
