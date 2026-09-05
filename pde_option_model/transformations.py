"""Price transformation, AR(1)->OU mapping and the deterministic seasonal term.

Verified conventions (see docs/data_inventory.md):

* Price transform (report.md section 3, pde_export.json, prepared_meta.json):
      y_t = asinh(P_t / scale_P),  scale_P = 282.48
  hence the exact inverse is
      P(y) = scale_P * sinh(y)
  and NOT plain sinh(y).

* Emission equation of the fitted Markov-switching AR regression
  ("switching intercept and variance", report.md section 6-9):
      y_t = c_i + phi * y_{t-1} + beta' D_t [+ rho_i * z_t] + eps_t,
      eps_t ~ N(0, sigma_eps_i^2),  Delta t = 1 hour,
  i.e. mu0/mu1 in parameter_estimates.csv are AR *intercepts*
  (mapping "A"), not long-run means.  Evidence: (i) |mu_i| ~ 1e-3 while the
  observed y level is O(1)-O(3), so mapping B would imply a long-run price of
  ~0.25 TRY/MWh; (ii) the bundle's own ou_mapping file computes mapping-A
  thetas of -2.2438 / +2.2744 from exactly these values.

* AR(1) -> OU (exact, per-hour units):
      kappa      = -ln(phi) / dt
      theta_i    = (c_i + deterministic terms) / (1 - phi)      [mapping A]
      sigma_OU_i = sigma_eps_i * sqrt( 2 kappa / (1 - exp(-2 kappa dt)) )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MappingChoice = Literal["A_intercept", "B_long_run_mean"]


# ----------------------------------------------------------------------------
# price transform
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PriceTransform:
    """y = asinh(P / scale_P)  <->  P = scale_P * sinh(y)."""

    scale_P: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale_P) or self.scale_P <= 0:
            raise ValueError(f"scale_P must be positive, got {self.scale_P}")

    def y_from_price(self, price: np.ndarray | float) -> np.ndarray | float:
        return np.arcsinh(np.asarray(price, dtype=float) / self.scale_P)

    def price_from_y(self, y: np.ndarray | float) -> np.ndarray | float:
        return self.scale_P * np.sinh(np.asarray(y, dtype=float))


# ----------------------------------------------------------------------------
# AR(1) -> OU
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OUMapping:
    kappa_per_hour: float
    theta_base: np.ndarray          # shape (2,), constant part of theta_i
    sigma_ou: np.ndarray            # shape (2,), per sqrt(hour)
    one_minus_phi: float
    dt_hours: float
    mapping: MappingChoice
    half_life_hours: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "half_life_hours",
                           float(np.log(2.0) / self.kappa_per_hour))


def ar_to_ou(
    phi: float,
    intercepts_or_means: Sequence[float],
    sigma_eps: Sequence[float],
    dt_hours: float = 1.0,
    mapping: MappingChoice = "A_intercept",
) -> OUMapping:
    """Exact discrete AR(1) -> continuous OU conversion.

    mapping="A_intercept":       y_{t+dt} = c_i + phi y_t + eps  -> theta_i = c_i/(1-phi)
    mapping="B_long_run_mean":   y_{t+dt} = (1-phi) mu_i + phi y_t + eps -> theta_i = mu_i
    """
    if not (0.0 < phi < 1.0):
        raise ValueError(f"phi must be in (0,1) for a mean-reverting OU map, got {phi}")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be positive")
    c = np.asarray(intercepts_or_means, dtype=float)
    s = np.asarray(sigma_eps, dtype=float)
    if c.shape != (2,) or s.shape != (2,):
        raise ValueError("expected exactly two regimes")
    if np.any(s <= 0):
        raise ValueError("sigma_eps must be positive")

    kappa = -np.log(phi) / dt_hours
    if mapping == "A_intercept":
        theta = c / (1.0 - phi)
    elif mapping == "B_long_run_mean":
        theta = c.copy()
    else:  # pragma: no cover
        raise ValueError(f"unknown mapping {mapping!r}")
    sigma_ou = s * np.sqrt(2.0 * kappa / (1.0 - np.exp(-2.0 * kappa * dt_hours)))
    logger.info(
        "AR->OU (%s): kappa=%.9g /h (half-life %.1f h), theta_base=%s, sigma_OU=%s",
        mapping, kappa, np.log(2) / kappa, np.round(theta, 6), np.round(sigma_ou, 6),
    )
    return OUMapping(
        kappa_per_hour=float(kappa),
        theta_base=theta,
        sigma_ou=sigma_ou,
        one_minus_phi=float(1.0 - phi),
        dt_hours=float(dt_hours),
        mapping=mapping,
    )


# ----------------------------------------------------------------------------
# deterministic seasonal component  g(t) = beta' D_t
# ----------------------------------------------------------------------------
SEASONAL_COLUMNS: tuple[str, ...] = (
    "D_sin_hour_1", "D_cos_hour_1", "D_sin_hour_2", "D_cos_hour_2",
    "D_sin_year_1", "D_cos_year_1", "D_sin_year_2", "D_cos_year_2",
    "D_dow_1", "D_dow_2", "D_dow_3", "D_dow_4", "D_dow_5", "D_dow_6",
    "D_weekend", "D_holiday_tr",
)


@dataclass
class SeasonalModel:
    """Deterministic intercept component g(t) = beta' D(t) of the emission mean.

    mode:
      "mean"   (default) g(t) is replaced by its analytic training-average
               E[beta' D] (harmonics -> 0, dow dummies -> 1/7, weekend -> 2/7,
               holiday -> holiday_fraction).  Robust to the unverified exact
               construction of the D columns.
      "full"   evaluate D(t) with the ASSUMED conventions documented in
               docs/data_inventory.md (local-time hour/dow, Fourier periods
               24 h and 365.25 d, Monday-baseline dow dummies).
      "zero"   g(t) = 0.
      "custom" caller supplies a callable via `custom_fn(timestamps)->array`.

    The exact D-column construction of the original preparation code could not
    be reproduced from the uploaded artifacts (assumption A2 in
    docs/ambiguities.md), which is why "mean" is the default.
    """

    betas: np.ndarray
    mode: Literal["mean", "full", "zero", "custom"] = "mean"
    tz_offset_hours: float = 3.0          # Europe/Istanbul, no DST since 2016
    holiday_fraction: float = 0.043       # ~15.7 official holiday days / year (ASSUMED)
    holiday_dates: tuple[str, ...] = ()   # ISO dates (local) for mode="full"
    custom_fn: object | None = None

    def __post_init__(self) -> None:
        self.betas = np.asarray(self.betas, dtype=float)
        if self.betas.shape != (len(SEASONAL_COLUMNS),):
            raise ValueError(
                f"expected {len(SEASONAL_COLUMNS)} seasonal betas, got {self.betas.shape}")

    # -- analytic average -----------------------------------------------------
    def mean_value(self) -> float:
        if self.mode == "zero":
            return 0.0
        b = dict(zip(SEASONAL_COLUMNS, self.betas))
        val = sum(b[f"D_dow_{j}"] for j in range(1, 7)) / 7.0
        val += b["D_weekend"] * 2.0 / 7.0
        val += b["D_holiday_tr"] * self.holiday_fraction
        return float(val)

    # -- assumed full design matrix ------------------------------------------
    def design_matrix(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        if timestamps.tz is None:
            raise ValueError("timestamps must be timezone-aware (UTC)")
        loc = timestamps + pd.Timedelta(hours=self.tz_offset_hours)
        h = loc.hour.values + loc.minute.values / 60.0
        doy = loc.dayofyear.values - 1 + h / 24.0
        dow = loc.dayofweek.values
        cols = [
            np.sin(2 * np.pi * h / 24.0), np.cos(2 * np.pi * h / 24.0),
            np.sin(4 * np.pi * h / 24.0), np.cos(4 * np.pi * h / 24.0),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
            np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25),
        ]
        cols += [(dow == j).astype(float) for j in range(1, 7)]
        cols.append((dow >= 5).astype(float))
        hol = set(self.holiday_dates)
        cols.append(np.array([d.date().isoformat() in hol for d in loc], dtype=float))
        return np.column_stack(cols)

    def value(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """g(t) = beta' D(t) for the requested mode."""
        n = len(timestamps)
        if self.mode == "zero":
            return np.zeros(n)
        if self.mode == "mean":
            return np.full(n, self.mean_value())
        if self.mode == "full":
            return self.design_matrix(timestamps) @ self.betas
        if self.mode == "custom":
            if self.custom_fn is None:
                raise ValueError("mode='custom' requires custom_fn")
            out = np.asarray(self.custom_fn(timestamps), dtype=float)
            if out.shape != (n,):
                raise ValueError("custom_fn returned wrong shape")
            return out
        raise ValueError(f"unknown seasonal mode {self.mode!r}")  # pragma: no cover


def make_time_index(
    valuation_utc: pd.Timestamp, hours: Iterable[float]
) -> pd.DatetimeIndex:
    """Calendar timestamps for solver times expressed in hours from valuation."""
    return pd.DatetimeIndex(
        [valuation_utc + pd.Timedelta(hours=float(h)) for h in hours], tz="UTC"
    )
