"""Time-varying transition probabilities and the continuous-time generator.

TVTP specification of the primary model M2_tvtp_TVTP-1 (single covariate
z = standardized residual demand RD_WS, lagged one hour):

    p01_t = logistic(alpha01 + gamma01 * z_{t-1})
    p10_t = logistic(alpha10 + gamma10 * z_{t-1})

Discrete -> continuous conversion (exact 2x2 matrix logarithm, dt = 1 h):

    s_t      = p01_t + p10_t                       (embeddability requires s_t < 1)
    lambda_t = -ln(1 - s_t) / dt
    q01_t    = lambda_t * p01_t / s_t
    q10_t    = lambda_t * p10_t / s_t

This is the same formula recorded in the uploaded bundle
(pde_model_contract.json -> generator_conversion) and it reproduces the
shipped q*_per_hour columns to machine precision (verified in
validation.run_generator_checks).  The naive approximation q = p/dt is
available only as an explicitly selected option.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

logger = logging.getLogger(__name__)

GeneratorMethod = Literal["matrix_log", "linear_approx"]


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


@dataclass(frozen=True)
class TVTPCoefficients:
    """Logistic TVTP coefficients for the two off-diagonal transitions."""

    alpha01: float
    gamma01: float
    alpha10: float
    gamma10: float
    covariate: str = "RD_lag1"

    def probabilities(self, z: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray]:
        """(p01, p10) evaluated at standardized covariate value(s) z."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        p01 = logistic(self.alpha01 + self.gamma01 * z)
        p10 = logistic(self.alpha10 + self.gamma10 * z)
        return p01, p10


@dataclass(frozen=True)
class GeneratorSeries:
    """q01(t), q10(t) per hour plus conversion diagnostics."""

    q01: np.ndarray
    q10: np.ndarray
    n_clipped: int
    method: GeneratorMethod

    @property
    def q_matrix(self) -> np.ndarray:
        """(n, 2, 2) generator matrices with zero row sums by construction."""
        n = self.q01.shape[0]
        Q = np.zeros((n, 2, 2))
        Q[:, 0, 0] = -self.q01
        Q[:, 0, 1] = self.q01
        Q[:, 1, 0] = self.q10
        Q[:, 1, 1] = -self.q10
        return Q


def probs_to_generator(
    p01: np.ndarray,
    p10: np.ndarray,
    dt_hours: float = 1.0,
    method: GeneratorMethod = "matrix_log",
    s_clip: float = 1.0 - 1e-10,
) -> GeneratorSeries:
    """Convert one-step transition probabilities into generator intensities.

    Values with p01 + p10 >= 1 are not embeddable in a continuous chain via the
    real matrix logarithm; such rows are clipped to ``s_clip`` and counted.
    """
    p01 = np.atleast_1d(np.asarray(p01, dtype=float))
    p10 = np.atleast_1d(np.asarray(p10, dtype=float))
    if p01.shape != p10.shape:
        raise ValueError("p01 and p10 must have the same shape")
    if np.any((p01 < 0) | (p01 > 1) | (p10 < 0) | (p10 > 1)):
        raise ValueError("transition probabilities must lie in [0, 1]")

    if method == "linear_approx":
        logger.warning("using the explicitly-selected linear approximation q = p/dt")
        return GeneratorSeries(q01=p01 / dt_hours, q10=p10 / dt_hours,
                               n_clipped=0, method=method)

    s = p01 + p10
    bad = s >= 1.0
    n_clipped = int(bad.sum())
    if n_clipped:
        logger.warning(
            "%d of %d transition rows have p01+p10 >= 1 (not embeddable); "
            "clipping s to %.3g before the matrix log", n_clipped, s.size, s_clip)
    s_eff = np.where(bad, s_clip, s)
    scale = np.where(bad, s_clip / s, 1.0)
    p01_eff, p10_eff = p01 * scale, p10 * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -np.log1p(-s_eff) / dt_hours
        q01 = np.where(s_eff > 0, lam * p01_eff / s_eff, 0.0)
        q10 = np.where(s_eff > 0, lam * p10_eff / s_eff, 0.0)
    out = GeneratorSeries(q01=q01, q10=q10, n_clipped=n_clipped, method=method)
    validate_generator(out.q01, out.q10)
    return out


def generator_to_probs(
    q01: np.ndarray, q10: np.ndarray, dt_hours: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact expm(Q dt) for the 2-state generator -> one-step probabilities."""
    q01 = np.atleast_1d(np.asarray(q01, dtype=float))
    q10 = np.atleast_1d(np.asarray(q10, dtype=float))
    sq = q01 + q10
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = -np.expm1(-sq * dt_hours)          # 1 - e^{-s q dt}
        p01 = np.where(sq > 0, q01 / np.where(sq > 0, sq, 1.0) * decay, 0.0)
        p10 = np.where(sq > 0, q10 / np.where(sq > 0, sq, 1.0) * decay, 0.0)
    return p01, p10


def validate_generator(q01: np.ndarray, q10: np.ndarray, atol: float = 1e-12) -> None:
    """q01, q10 >= 0 and generator rows sum to zero (holds by construction)."""
    if np.any(q01 < -atol) or np.any(q10 < -atol):
        raise ValueError("negative off-diagonal generator intensity encountered")
    Q = GeneratorSeries(np.atleast_1d(q01), np.atleast_1d(q10), 0, "matrix_log").q_matrix
    rowsum = np.abs(Q.sum(axis=2)).max()
    if rowsum > 1e-10:
        raise ValueError(f"generator rows do not sum to zero (max abs {rowsum:.3e})")


def expm_reproduction_error(
    p01: np.ndarray, p10: np.ndarray, dt_hours: float = 1.0
) -> float:
    """max | expm(Q dt) - P | over all supplied rows (embeddable rows only)."""
    gen = probs_to_generator(p01, p10, dt_hours=dt_hours)
    p01_hat, p10_hat = generator_to_probs(gen.q01, gen.q10, dt_hours=dt_hours)
    mask = (np.atleast_1d(p01) + np.atleast_1d(p10)) < 1.0
    if not mask.any():
        return float("nan")
    return float(max(np.abs(p01_hat - p01)[mask].max(),
                     np.abs(p10_hat - p10)[mask].max()))


def stationary_distribution(q01: float, q10: float) -> np.ndarray:
    """Stationary distribution of the frozen 2-state generator."""
    s = q01 + q10
    if s <= 0:
        return np.array([0.5, 0.5])
    return np.array([q10 / s, q01 / s])
