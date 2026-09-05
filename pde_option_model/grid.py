"""Uniform space grid in the transformed price variable y and the time grid."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpaceGrid:
    """Uniform grid in y = asinh(P / scale_P)."""

    y_min: float
    y_max: float
    n_nodes: int
    y: np.ndarray = field(init=False, repr=False)
    h: float = field(init=False)

    def __post_init__(self) -> None:
        if self.n_nodes < 5:
            raise ValueError("need at least 5 spatial nodes")
        if not self.y_max > self.y_min:
            raise ValueError("y_max must exceed y_min")
        y = np.linspace(self.y_min, self.y_max, self.n_nodes)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "h", float(y[1] - y[0]))

    def contains(self, y0: float, margin_nodes: int = 3) -> bool:
        pad = margin_nodes * self.h
        return (self.y_min + pad) <= y0 <= (self.y_max - pad)

    def interp(self, values: np.ndarray, y0: float) -> float:
        """Linear interpolation of a nodal vector at y0."""
        if not (self.y_min <= y0 <= self.y_max):
            raise ValueError(f"y0={y0} outside grid [{self.y_min}, {self.y_max}]")
        return float(np.interp(y0, self.y, values))


@dataclass(frozen=True)
class TimeGrid:
    """Backward-solved time grid on [valuation, maturity], measured in hours."""

    valuation_utc: pd.Timestamp
    maturity_utc: pd.Timestamp
    n_steps: int
    times_hours: np.ndarray = field(init=False, repr=False)   # 0 .. tau, len n_steps+1
    dt_hours: float = field(init=False)
    tau_hours: float = field(init=False)

    def __post_init__(self) -> None:
        if self.valuation_utc.tzinfo is None or self.maturity_utc.tzinfo is None:
            raise ValueError("valuation and maturity timestamps must be tz-aware")
        tau = (self.maturity_utc - self.valuation_utc).total_seconds() / 3600.0
        if tau <= 0:
            raise ValueError("maturity must be after the valuation date")
        if self.n_steps < 2:
            raise ValueError("need at least 2 time steps")
        object.__setattr__(self, "tau_hours", float(tau))
        object.__setattr__(self, "times_hours",
                           np.linspace(0.0, tau, self.n_steps + 1))
        object.__setattr__(self, "dt_hours", float(tau / self.n_steps))

    def timestamps(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            [self.valuation_utc + pd.Timedelta(hours=float(h))
             for h in self.times_hours], tz="UTC")
