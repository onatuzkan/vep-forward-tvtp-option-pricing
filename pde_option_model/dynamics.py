"""Regime-conditional mean-reverting dynamics of the transformed price.

Physical-measure (P) dynamics per regime i, conditional on the deterministic
residual-demand scenario z(t):

    dY_t = kappa * ( theta_i^P(t, z) - Y_t ) dt + sigma_OU_i dW_t

with, under AR-intercept mapping A,

    theta_i^P(t, z) = ( c_i + g(t) + rho_i * z(t) ) / (1 - phi)

where g(t) = beta' D(t) is the deterministic seasonal component and rho_i is
the regime-dependent loading of standardized residual demand in the emission
mean ("regime_price_sensitivities" in pde_export.json; assumption A3 in
docs/ambiguities.md -- can be switched off).

Pricing-measure (Q) drift adds the adjustments from risk_neutral.MeasureAdjustment:

    b_i^Q(t, y) = kappa * ( theta_i^P(t,z) + delta_i - y )
                  + a_i - lambda_i * sigma_OU_i

delta_i is the long-run-mean shift of specification Q1; a_i is an equivalent
direct drift shift (per hour), better conditioned here because kappa is tiny;
lambda_i is a market-price-of-risk parameterization.  All default to zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .risk_neutral import MeasureAdjustment
from .transformations import OUMapping, SeasonalModel, make_time_index


@dataclass
class RegimeDynamics:
    """Container binding OU parameters, seasonality and RD sensitivity."""

    ou: OUMapping
    seasonal: SeasonalModel
    rho: np.ndarray                      # (2,) loading on z in the emission mean
    include_rd_in_theta: bool = True
    sigma_multipliers: np.ndarray = field(
        default_factory=lambda: np.ones(2))  # sensitivity knob, default identity

    def __post_init__(self) -> None:
        self.rho = np.asarray(self.rho, dtype=float)
        self.sigma_multipliers = np.asarray(self.sigma_multipliers, dtype=float)
        if self.rho.shape != (2,):
            raise ValueError("rho must have shape (2,)")
        if self.sigma_multipliers.shape != (2,):
            raise ValueError("sigma_multipliers must have shape (2,)")

    # ------------------------------------------------------------------ sigma
    def sigma(self) -> np.ndarray:
        """Regime volatilities per sqrt(hour), including sensitivity multipliers."""
        return self.ou.sigma_ou * self.sigma_multipliers

    # ------------------------------------------------------------------ theta
    def theta_path(
        self,
        valuation_utc: pd.Timestamp,
        times_hours: np.ndarray,
        z_path: np.ndarray,
        adjustment: Optional[MeasureAdjustment] = None,
    ) -> np.ndarray:
        """theta_i(t) on the solver time grid; shape (2, n_times).

        z_path must already be aligned to times_hours (the caller applies the
        one-hour covariate lag where relevant -- the *transition* covariate is
        lagged, the emission loading rho uses contemporaneous z).
        """
        times_hours = np.asarray(times_hours, dtype=float)
        z_path = np.asarray(z_path, dtype=float)
        if z_path.shape != times_hours.shape:
            raise ValueError("z_path and times_hours must have identical shape")
        ts = make_time_index(valuation_utc, times_hours)
        g = self.seasonal.value(ts)                                  # (n,)
        theta = np.empty((2, times_hours.size))
        for i in range(2):
            extra = g.copy()
            if self.include_rd_in_theta:
                extra = extra + self.rho[i] * z_path
            theta[i] = self.ou.theta_base[i] + extra / self.ou.one_minus_phi
        if adjustment is not None:
            theta += adjustment.theta_shift[:, None]
        return theta

    # ------------------------------------------------------------------ drift
    def drift(
        self,
        y: np.ndarray,
        theta_t: np.ndarray,
        adjustment: Optional[MeasureAdjustment] = None,
    ) -> np.ndarray:
        """b_i(t, y) for a single time slice; theta_t shape (2,), returns (2, Ny)."""
        theta_t = np.asarray(theta_t, dtype=float).reshape(2, 1)
        b = self.ou.kappa_per_hour * (theta_t - y[None, :])
        if adjustment is not None:
            b = b + adjustment.drift_shift_per_hour[:, None]
            b = b - (adjustment.market_price_of_risk * self.sigma())[:, None]
        return b
