"""Analytic moments of the LEGACY sinh-Gaussian model, and why it explodes.

Legacy specification (model mode ``legacy_asinh_ou``)::

    y_t  ~  Ornstein-Uhlenbeck (Gaussian),   P_t = scale_P * sinh(y_t)

For Y ~ N(m, v) the moment generating function gives the two identities that
drive everything below::

    E[sinh Y]  = e^{v/2} sinh(m)
    E[sinh^2 Y] = ( e^{2v} cosh(2m) - 1 ) / 2

so the model's expected PRICE carries a multiplicative variance inflation
factor ``exp(v(t)/2)`` on top of the drift term ``sinh(m(t))``.  For a
mean-reverting y with speed kappa and variance rate sigma^2::

    m(t) = theta + (y_0 - theta) e^{-kappa t}
    v(t) = sigma^2 ( 1 - e^{-2 kappa t} ) / (2 kappa)   ->   sigma^2 / (2 kappa)

Both moments are finite for every finite t -- the failure is not infinity, it
is the SIZE of the growth.  The inflation factor at stationarity is

    exp( v_inf / 2 ) = exp( sigma^2 / (4 kappa) )

and with the reconciled M9 numbers (kappa = 4.11e-6 /h, half-life ~19 years)
this is:

    normal regime  sigma = 0.00353 ->  v_inf =  1.52   ->  factor 2.14    (already O(1))
    stress regime  sigma = 0.0924  ->  v_inf = 1039    ->  factor ~1e225  (astronomical)

The pre-2026 fallback-sourced yaml carried kappa = 3.85e-4 /h and reported
normal-regime factor ~1.04, stress-regime factor ~1e8; the reconciled
near-unit-root kappa makes the legacy sinh-Gaussian output even more
unusable at long horizons, without changing the qualitative diagnosis.

For short horizons v(t) ~ sigma^2 t, so the bias DOUBLES every
``2 ln 2 / sigma^2`` hours.  With the long-run stress occupancy of the fitted
chain (~0.67) the mixture variance rate is ~0.0202 /h and the doubling time is
about 69 hours: a monthly forward is inflated by orders of magnitude.

This is a property of the model's moment structure, not Monte Carlo noise, and
no single drift shift can repair it: shifting the drift moves ``sinh(m)``, while
the damage lives in ``e^{v/2}``, which the drift cannot touch.  That is exactly
why calibrating a common ``a0 = a1`` to six monthly VEP quotes reported
optimizer success while leaving RMSE in the millions of TRY/MWh.

Everything here is closed-form: no Monte Carlo, no PDE, no artefact bundle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPLOSION_WARNING = (
    "legacy_asinh_ou: E[P_t] = scale_P * exp(v(t)/2) * sinh(m(t)) grows "
    "exponentially in the variance v(t) = sigma^2 (1 - e^{-2 kappa t}) / (2 kappa). "
    "With the fitted near-unit-root kappa and the stress-regime sigma this makes "
    "long-horizon expected prices economically meaningless. Use the legacy mode "
    "for short-horizon benchmarking and diagnostics only.")


@dataclass(frozen=True)
class SinhGaussianMoments:
    """Closed-form moments of P = scale_P * sinh(Y), Y ~ N(m, v)."""

    scale_P: float
    m: np.ndarray
    v: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        """E[P] = scale_P e^{v/2} sinh(m)."""
        return self.scale_P * np.exp(0.5 * self.v) * np.sinh(self.m)

    @property
    def second_moment(self) -> np.ndarray:
        """E[P^2] = scale_P^2 ( e^{2v} cosh(2m) - 1 ) / 2."""
        return self.scale_P ** 2 * (np.exp(2.0 * self.v) * np.cosh(2.0 * self.m) - 1.0) / 2.0

    @property
    def variance(self) -> np.ndarray:
        return self.second_moment - self.mean ** 2

    @property
    def inflation_factor(self) -> np.ndarray:
        """The pure variance channel e^{v/2} (1.0 would mean 'no distortion')."""
        return np.exp(0.5 * self.v)


def ou_mean_variance(y0: float, theta: float, kappa: float, sigma: float,
                     t_hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian OU moments m(t), v(t) with a frozen variance rate."""
    t = np.asarray(t_hours, dtype=float)
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    m = theta + (y0 - theta) * np.exp(-kappa * t)
    v = sigma ** 2 * (1.0 - np.exp(-2.0 * kappa * t)) / (2.0 * kappa)
    return m, v


def stationary_variance(sigma: float, kappa: float) -> float:
    """v_inf = sigma^2 / (2 kappa), the y-variance the OU converges to."""
    return float(sigma ** 2 / (2.0 * kappa))


def stationary_inflation_factor(sigma: float, kappa: float) -> float:
    """exp(sigma^2 / (4 kappa)): the multiplicative bias E[P] inherits at t -> inf."""
    return float(np.exp(stationary_variance(sigma, kappa) / 2.0))


def variance_doubling_time_hours(sigma: float) -> float:
    """Hours for the inflation factor to double while v(t) ~ sigma^2 t."""
    return float(2.0 * np.log(2.0) / sigma ** 2)


def legacy_expected_spot(
    horizons_hours: Sequence[float] | np.ndarray,
    scale_P: float,
    y0: float,
    theta: float,
    kappa: float,
    sigma_y: Sequence[float],
    pi_stress: float,
) -> np.ndarray:
    """Frozen-occupancy analytic E[P_t] of the legacy model, TRY/MWh.

    The regime chain is collapsed onto its occupancy-weighted variance rate
    sigma_bar^2 = (1 - pi_stress) sigma_0^2 + pi_stress sigma_1^2.  This is an
    approximation to the exact Markov-modulated moment, adequate for an
    order-of-magnitude explosion diagnostic and used ONLY as a benchmark curve.
    """
    t = np.asarray(horizons_hours, dtype=float)
    s = np.asarray(sigma_y, dtype=float)
    if s.shape != (2,):
        raise ValueError("sigma_y must have shape (2,)")
    if not 0.0 <= pi_stress <= 1.0:
        raise ValueError("pi_stress must lie in [0, 1]")
    sbar2 = (1.0 - pi_stress) * s[0] ** 2 + pi_stress * s[1] ** 2
    m, v = ou_mean_variance(y0, theta, kappa, float(np.sqrt(sbar2)), t)
    with np.errstate(over="ignore"):
        out = SinhGaussianMoments(scale_P=scale_P, m=m, v=v).mean
    if not np.all(np.isfinite(out)):
        logger.error("legacy analytic forward overflowed at horizons %s",
                     t[~np.isfinite(out)].tolist())
    return out


def legacy_explosion_report(
    scale_P: float, y0: float, kappa: float, sigma_y: Sequence[float],
    pi_filtered_stress: float, pi_stationary_stress: Optional[float] = None,
    theta: float = 0.0,
    horizons_hours: Sequence[float] = (24, 72, 168, 336, 720, 2160, 8760),
) -> Dict[str, object]:
    """Quantified diagnosis of the legacy long-horizon moment explosion."""
    s = np.asarray(sigma_y, dtype=float)
    pis = float(pi_filtered_stress if pi_stationary_stress is None
                else pi_stationary_stress)
    sbar2 = (1.0 - pis) * s[0] ** 2 + pis * s[1] ** 2
    t = np.asarray(horizons_hours, dtype=float)
    m, v = ou_mean_variance(y0, theta, kappa, float(np.sqrt(sbar2)), t)
    mom = SinhGaussianMoments(scale_P=scale_P, m=m, v=v)
    with np.errstate(over="ignore"):
        table = pd.DataFrame({
            "horizon_hours": t,
            "variance_v_of_y": v,
            "inflation_factor_exp_v_over_2": mom.inflation_factor,
            "expected_spot_TRY_MWh": mom.mean,
        })
    return {
        "warning": EXPLOSION_WARNING,
        "kappa_per_hour": float(kappa),
        "half_life_hours": float(np.log(2.0) / kappa),
        "occupancy_stress_used": pis,
        "mixture_variance_rate_per_hour": float(sbar2),
        "variance_doubling_time_hours": variance_doubling_time_hours(float(np.sqrt(sbar2))),
        "stationary_variance_normal": stationary_variance(float(s[0]), kappa),
        "stationary_variance_stress": stationary_variance(float(s[1]), kappa),
        "stationary_inflation_normal": stationary_inflation_factor(float(s[0]), kappa),
        "stationary_inflation_stress": stationary_inflation_factor(float(s[1]), kappa),
        "stationary_inflation_mixture": stationary_inflation_factor(
            float(np.sqrt(sbar2)), kappa),
        "table": table,
    }


def load_legacy_reference(path) -> pd.DataFrame:
    """Read the recorded legacy model-implied forwards benchmark JSON."""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"legacy reference not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    df = pd.DataFrame(blob["points"])
    df.attrs["provenance"] = blob.get("provenance", "")
    df.attrs["model"] = blob.get("model", "legacy_asinh_ou")
    return df
