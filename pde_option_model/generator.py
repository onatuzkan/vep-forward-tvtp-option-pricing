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

Two-covariate extension (EXPERIMENTAL, FW4)
-------------------------------------------
:class:`TVTP2Coefficients` adds the residual-demand ramp covariate of the M9
``TVTP-2`` specification::

    p01_t = logistic(alpha01 + gamma01 * z_{t-1} + h01 * r_{t-1})
    p10_t = logistic(alpha10 + gamma10 * z_{t-1} + h10 * r_{t-1})

with r the standardized one-hour ramp of z.  The ramp definition could not be
verified against the original estimation code, so the mode is labelled
``rd_ramp_2d_experimental`` (see ``docs/tvtp2_methodology.md``).  With
h01 = h10 = 0 the two-covariate law reduces bit-for-bit to the single-covariate
one.  The same exact matrix logarithm is used; for the two-covariate path a
non-embeddable row (p01 + p10 >= 1) is REJECTED (``on_nonembeddable="raise"``)
rather than silently clipped, because the continuous-time chain does not exist
at that hour (the second eigenvalue 1 - s of the one-step matrix is <= 0).
The single-covariate API keeps its historical clip-and-warn default.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

GeneratorMethod = Literal["matrix_log", "linear_approx"]
NonEmbeddablePolicy = Literal["clip", "raise"]

#: TVTP mode identifiers.  The single-covariate mode is the production default.
TVTP_MODE_1D = "rd_lag1_1d"
TVTP_MODE_2D = "rd_ramp_2d_experimental"
TVTP_MODES: Tuple[str, str] = (TVTP_MODE_1D, TVTP_MODE_2D)

#: Coarse unit guard for standardized (dimensionless) covariates.  Raw residual
#: demand is O(1e4) MWh and a raw MWh ramp O(1e3); standardized values beyond
#: this bound are rejected as a probable unit error.  It is a plausibility
#: guard, not a proof of units: the ramp standardization itself is verified
#: against the frozen scaler where the covariate path is built.
MAX_ABS_STANDARDIZED_COVARIATE: float = 50.0


class CovariateError(ValueError):
    """Malformed TVTP covariate input (shape, missing values, units)."""


class EmbeddabilityError(ValueError):
    """p01 + p10 >= 1 on a row where the continuous-time chain is required."""

    def __init__(self, message: str, report: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.report: Dict[str, Any] = dict(report or {})


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

    @property
    def mode(self) -> str:
        return TVTP_MODE_1D

    @property
    def n_covariates(self) -> int:
        return 1


def validate_standardized_covariate(name: str, values: Any) -> np.ndarray:
    """Return ``values`` as a finite 1-D float array or raise :class:`CovariateError`.

    Missing covariates (NaN / inf) are an error: they are never replaced by
    zero.  Values far outside the standardized range are rejected as a
    probable unit error (raw MWh instead of standardized units).
    """
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.ndim != 1:
        raise CovariateError(f"{name} must be one-dimensional, got shape {arr.shape}")
    bad = ~np.isfinite(arr)
    if bad.any():
        idx = np.flatnonzero(bad)
        raise CovariateError(
            f"{name} contains {idx.size} missing/non-finite value(s) (first at "
            f"position {int(idx[0])}); a missing TVTP covariate is never "
            "replaced by zero -- build the path with an explicit rule instead")
    if arr.size and float(np.max(np.abs(arr))) > MAX_ABS_STANDARDIZED_COVARIATE:
        raise CovariateError(
            f"{name} reaches |value| = {float(np.max(np.abs(arr))):.6g} > "
            f"{MAX_ABS_STANDARDIZED_COVARIATE:g}; TVTP covariates must be in "
            "STANDARDIZED (dimensionless) units, not MWh")
    return arr


@dataclass(frozen=True)
class TVTP2Coefficients:
    """Two-covariate logistic TVTP (production labels: 0 = normal, 1 = stress).

    ``gamma*`` load on z_{t-1} (RD_lag1), ``h*`` on the standardized one-hour
    ramp r_{t-1} (RD_Ramp_1h_lag1).  EXPERIMENTAL: the ramp definition is a
    reconstruction, see ``docs/tvtp2_methodology.md``.
    """

    alpha01: float
    gamma01: float
    h01: float
    alpha10: float
    gamma10: float
    h10: float
    covariates: Tuple[str, str] = ("RD_lag1", "RD_Ramp_1h_lag1")
    name: str = "tvtp2"

    def __post_init__(self) -> None:
        for k in ("alpha01", "gamma01", "h01", "alpha10", "gamma10", "h10"):
            v = getattr(self, k)
            if not isinstance(v, (int, float, np.floating, np.integer)) or not np.isfinite(v):
                raise CovariateError(f"TVTP2 coefficient {k} must be a finite number, got {v!r}")
            object.__setattr__(self, k, float(v))
        if len(tuple(self.covariates)) != 2:
            raise CovariateError("TVTP2Coefficients needs exactly two covariate names")
        object.__setattr__(self, "covariates", tuple(str(c) for c in self.covariates))

    @property
    def mode(self) -> str:
        return TVTP_MODE_2D

    @property
    def n_covariates(self) -> int:
        return 2

    @property
    def reduces_to_single_covariate(self) -> bool:
        return self.h01 == 0.0 and self.h10 == 0.0

    def probabilities(self, z: np.ndarray | float,
                      r: Optional[np.ndarray | float] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """(p01, p10) at standardized z_{t-1} and standardized ramp r_{t-1}.

        Both covariates are required and must have identical shapes; there is
        no broadcasting and no default for a missing ramp.
        """
        if r is None:
            raise CovariateError(
                "two-covariate TVTP requires the ramp path r(t-1) "
                "(RD_Ramp_1h_lag1); refusing to default a missing ramp to zero")
        z_arr = validate_standardized_covariate("z (RD_lag1)", z)
        r_arr = validate_standardized_covariate("r (RD_Ramp_1h_lag1)", r)
        if z_arr.shape != r_arr.shape:
            raise CovariateError(
                f"z and r must have identical shapes (got {z_arr.shape} and "
                f"{r_arr.shape}); the two covariates must come from one path")
        # (alpha + gamma z) + h r: with h = 0 this is bit-identical to the
        # single-covariate predictor alpha + gamma z (x + 0.0 == x).
        p01 = logistic(self.alpha01 + self.gamma01 * z_arr + self.h01 * r_arr)
        p10 = logistic(self.alpha10 + self.gamma10 * z_arr + self.h10 * r_arr)
        return p01, p10

    @classmethod
    def from_single_covariate(cls, coef: TVTPCoefficients, h01: float = 0.0,
                              h10: float = 0.0, name: str = "tvtp2_from_1d"
                              ) -> "TVTP2Coefficients":
        """Embed single-covariate coefficients (ramp slopes default to zero)."""
        return cls(alpha01=coef.alpha01, gamma01=coef.gamma01, h01=h01,
                   alpha10=coef.alpha10, gamma10=coef.gamma10, h10=h10,
                   name=name)

    def single_covariate(self) -> TVTPCoefficients:
        """The equivalent single-covariate object; only valid when h01 = h10 = 0."""
        if not self.reduces_to_single_covariate:
            raise CovariateError(
                "ramp slopes are non-zero; the two-covariate law has no "
                "single-covariate equivalent")
        return TVTPCoefficients(self.alpha01, self.gamma01, self.alpha10, self.gamma10,
                                covariate=self.covariates[0])

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["covariates"] = list(self.covariates)
        return d


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


def embeddability_report(
    p01: np.ndarray,
    p10: np.ndarray,
    labels: Optional[Sequence[Any]] = None,
    near_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Audit of s = p01 + p10 on a path (count / share / max of s >= 1).

    ``labels`` (e.g. UTC hour labels or solver times) are used to name the
    location of the maximum and of the first / last violation.
    """
    p01 = np.atleast_1d(np.asarray(p01, dtype=float))
    p10 = np.atleast_1d(np.asarray(p10, dtype=float))
    if p01.shape != p10.shape:
        raise ValueError("p01 and p10 must have the same shape")
    s = p01 + p10
    n = int(s.size)
    lab = list(labels) if labels is not None else list(range(n))
    if len(lab) != n:
        raise ValueError("labels must match the probability arrays")

    def _fmt(v: Any) -> Any:
        return v.isoformat() if hasattr(v, "isoformat") else (
            float(v) if isinstance(v, (np.floating, float)) else
            int(v) if isinstance(v, (np.integer, int)) else str(v))

    viol = np.flatnonzero(s >= 1.0)
    out: Dict[str, Any] = {
        "n_rows": n,
        "n_s_ge_1": int(viol.size),
        "share_s_ge_1": float(viol.size / n) if n else 0.0,
        f"n_s_ge_{near_threshold:g}": int(np.sum(s >= near_threshold)),
        "near_threshold": float(near_threshold),
        "max_s": float(s.max()) if n else float("nan"),
        "min_s": float(s.min()) if n else float("nan"),
        "argmax_label": _fmt(lab[int(np.argmax(s))]) if n else None,
        "quantile_0.9999_s": float(np.quantile(s, 0.9999)) if n else float("nan"),
        "first_violation_label": _fmt(lab[int(viol[0])]) if viol.size else None,
        "last_violation_label": _fmt(lab[int(viol[-1])]) if viol.size else None,
        "embeddable": bool(viol.size == 0),
    }
    return out


def probs_to_generator(
    p01: np.ndarray,
    p10: np.ndarray,
    dt_hours: float = 1.0,
    method: GeneratorMethod = "matrix_log",
    s_clip: float = 1.0 - 1e-10,
    on_nonembeddable: NonEmbeddablePolicy = "clip",
) -> GeneratorSeries:
    """Convert one-step transition probabilities into generator intensities.

    Values with p01 + p10 >= 1 are not embeddable in a continuous chain via the
    real matrix logarithm.  ``on_nonembeddable="clip"`` (historical default of
    the single-covariate API) clips such rows to ``s_clip`` and counts them;
    ``on_nonembeddable="raise"`` refuses with :class:`EmbeddabilityError`
    carrying an :func:`embeddability_report`.  The two-covariate pricing path
    always uses ``"raise"``.
    """
    if on_nonembeddable not in ("clip", "raise"):
        raise ValueError(f"unknown on_nonembeddable policy {on_nonembeddable!r}")
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
    if n_clipped and on_nonembeddable == "raise":
        rep = embeddability_report(p01, p10)
        raise EmbeddabilityError(
            f"{n_clipped} of {s.size} transition rows have p01 + p10 >= 1 "
            f"(max s = {rep['max_s']:.6f}); the continuous-time generator does "
            "not exist there, so this pricing mode is REJECTED instead of "
            "silently clipping", rep)
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
