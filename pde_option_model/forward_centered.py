"""Forward-curve-centered risk-neutral model (the market pricing model).

Specification
-------------
The spot price is decomposed into a deterministic market level and a
zero-mean stochastic residual::

    P_t = F(t) + X_t - mu_X(t)                      (additive mode, default)
    P_t = F(t) * exp(X_t) / E^Q[exp(X_t)]           (multiplicative mode)

``F(t)`` is the hourly EPİAŞ VEP forward curve of
:mod:`pde_option_model.forward_curve`; ``X_t`` is a two-regime mean-reverting
residual whose regime process is the SAME TVTP Markov chain as the legacy
model::

    dX_t = kappa_X ( m_{J_t} - X_t ) dt + sigma^X_{J_t}(t) dW_t
    J_t in {0 = normal, 1 = stress},
    p01(t) = logistic(alpha01 + gamma01 z(t-1)),  p10(t) = logistic(alpha10 + gamma10 z(t-1))

with the exact 2x2 matrix-log generator q_ij(t) of :mod:`generator`.

Residual volatility in TRY/MWh
------------------------------
The historical regime volatilities are estimated in the transformed variable
y = asinh(P/scale_P).  Since dP/dy = sqrt(P^2 + scale_P^2), the delta-method
transfer of the regime structure onto the market level is::

    sigma^X_i(t) = sigma^y_i * sqrt( F(t)^2 + scale_P^2 )      [TRY/MWh / sqrt(h)]

This keeps the estimated regime volatility RATIO exactly (sigma_stress /
sigma_normal ~ 23x) while attaching it to the market's price level instead of
to a model-implied one — the separation of "residual regime volatility" from
"forward level" required by the specification.

Exact centering (why nothing explodes)
--------------------------------------
Write u_i(t) = E^Q[X_t 1{J_t = i}] and p_i(t) = Q(J_t = i).  For the linear
dynamics above these satisfy a CLOSED 4-dimensional linear ODE system::

    p_i'  = sum_{j!=i} ( q_ji p_j - q_ij p_i )
    u_i'  = kappa_X ( m_i p_i - u_i ) + sum_{j!=i} ( q_ji u_j - q_ij u_i )

so mu_X(t) = u_0(t) + u_1(t) = E^Q[X_t] is available deterministically, with no
Monte Carlo.  Subtracting it gives, exactly and for every t,

    E^Q[P_t] = F(t),

hence for every delivery month m

    (1/N_m) sum_{h in m} E^Q[P_h] = (1/N_m) sum_{h in m} F(h) = VEP_m.

Crucially mu_X(t) does not depend on sigma at all, so the monthly fit is exactly
invariant to the regime volatilities: the calibration cannot be broken by a
variance term.  Contrast the legacy model, where E[scale_P sinh(Y)] =
scale_P e^{v(t)/2} sinh(m(t)) makes the mean grow exponentially in the variance
v(t) (see :mod:`pde_option_model.legacy_moments`).

The second moments w_i(t) = E^Q[X_t^2 1{J_t = i}] follow the companion system

    w_i' = 2 kappa_X ( m_i u_i - w_i ) + sigma^X_i(t)^2 p_i
           + sum_{j!=i} ( q_ji w_j - q_ij w_i )

giving Var^Q[P_t] analytically — finite for every t by construction.

Pricing PDE
-----------
State variable is the raw residual x (TRY/MWh).  For i in {0, 1}::

    dV_i/dt + kappa_X(m_i - x) dV_i/dx + 0.5 sigma^X_i(t)^2 d2V_i/dx2 - r V_i
            + sum_{j!=i} q_ij(t) ( V_j - V_i ) = 0
    V_i(T, x) = payoff( price_from_state(T, x) )

The contract remains a European option on the **expiry-hour spot PTF**, not on
a monthly baseload average.  Only the payoff mapping changes::

    call: max( F(T) + x - mu_X(T) - K , 0 )
    put : max( K - (F(T) + x - mu_X(T)) , 0 )

Units: x, F, K, V are all TRY/MWh; kappa_X per hour; sigma^X per sqrt(hour).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import EuropeanOption
from .forward_curve import ForwardCurve
from .generator import TVTPCoefficients, generator_to_probs, probs_to_generator
from .grid import SpaceGrid, TimeGrid
from .params_frozen import FrozenM2Parameters
from .solver import SolverSettings, SolveResult, solve_coupled_pde

logger = logging.getLogger(__name__)

ResidualMode = Literal["additive", "multiplicative"]
X0Mode = Literal["zero", "spot_minus_curve"]

MAX_PLAUSIBLE_PRICE_TRY_MWh: float = 1.0e5


class ForwardCenteredError(ValueError):
    """Raised on a mis-specified or numerically unusable forward-centered model."""


# ---------------------------------------------------------------------------
@dataclass
class ResidualSpec:
    """Configuration of the two-regime residual process.

    ``drift_shift_per_hour`` is a Q1-style per-regime drift adjustment
    (``a_i`` in TRY/MWh per hour): the residual SDE becomes
    ``dX = [kappa (m_i - X) + a_i] dt + sigma_i dW``.  It is threaded
    symmetrically into both the moment ODE and the pricing PDE / MC
    simulator, so ``E^Q[P_t] = F(t)`` is preserved by construction (see
    ``docs/risk_neutral_methodology.md``).  Default (0, 0) reproduces the
    physical measure exactly.
    """

    kappa_per_hour: float
    sigma_y: np.ndarray                       # (2,) historical y-space volatilities
    scale_P: float                            # TRY/MWh
    regime_means: np.ndarray = field(         # m_i, TRY/MWh (additive) or log-units
        default_factory=lambda: np.zeros(2))
    mode: ResidualMode = "additive"
    x0_mode: X0Mode = "zero"
    sigma_multipliers: np.ndarray = field(default_factory=lambda: np.ones(2))
    drift_shift_per_hour: np.ndarray = field(  # a_i, TRY/MWh per hour, Q1 drift channel
        default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        self.sigma_y = np.asarray(self.sigma_y, dtype=float)
        self.regime_means = np.asarray(self.regime_means, dtype=float)
        self.sigma_multipliers = np.asarray(self.sigma_multipliers, dtype=float)
        self.drift_shift_per_hour = np.asarray(self.drift_shift_per_hour, dtype=float)
        for nm, arr in (("sigma_y", self.sigma_y),
                        ("regime_means", self.regime_means),
                        ("sigma_multipliers", self.sigma_multipliers),
                        ("drift_shift_per_hour", self.drift_shift_per_hour)):
            if arr.shape != (2,):
                raise ForwardCenteredError(f"{nm} must have shape (2,)")
        if np.any(self.sigma_y <= 0) or np.any(self.sigma_multipliers <= 0):
            raise ForwardCenteredError("volatilities and multipliers must be positive")
        if self.kappa_per_hour <= 0:
            raise ForwardCenteredError("kappa must be positive")
        if self.mode not in ("additive", "multiplicative"):
            raise ForwardCenteredError(f"unknown residual mode {self.mode!r}")

    def sigma_price(self, forward_level: np.ndarray | float) -> np.ndarray:
        """sigma^X_i at a market level F, TRY/MWh per sqrt(hour) (shape (2, ...))."""
        f = np.asarray(forward_level, dtype=float)
        if self.mode == "multiplicative":
            base = np.ones_like(f)                       # log-space: level-free
        else:
            base = np.sqrt(f ** 2 + self.scale_P ** 2)
        s = (self.sigma_y * self.sigma_multipliers)[:, None] * np.atleast_1d(base)[None, :]
        return s if f.ndim else s[:, 0]

    @classmethod
    def from_frozen(cls, params: FrozenM2Parameters, mode: ResidualMode = "additive",
                    kappa_per_hour: Optional[float] = None,
                    regime_means: Optional[Sequence[float]] = None,
                    x0_mode: X0Mode = "zero",
                    sigma_multipliers: Sequence[float] = (1.0, 1.0),
                    drift_shift_per_hour: Sequence[float] = (0.0, 0.0)) -> "ResidualSpec":
        return cls(
            kappa_per_hour=float(kappa_per_hour if kappa_per_hour is not None
                                 else params.kappa_per_hour),
            sigma_y=params.sigma_y.copy(), scale_P=params.scale_P,
            regime_means=np.asarray(regime_means if regime_means is not None
                                    else (0.0, 0.0), dtype=float),
            mode=mode, x0_mode=x0_mode,
            sigma_multipliers=np.asarray(sigma_multipliers, dtype=float),
            drift_shift_per_hour=np.asarray(drift_shift_per_hour, dtype=float),
        )


# ---------------------------------------------------------------------------
@dataclass
class ResidualMoments:
    """Deterministic first and second moments of the residual on a time grid."""

    times_hours: np.ndarray
    p: np.ndarray            # (2, n) regime probabilities
    u: np.ndarray            # (2, n) E[X 1{J=i}]
    w: np.ndarray            # (2, n) E[X^2 1{J=i}]

    @property
    def mean(self) -> np.ndarray:
        """mu_X(t) = E[X_t]."""
        return self.u.sum(axis=0)

    @property
    def second_moment(self) -> np.ndarray:
        return self.w.sum(axis=0)

    @property
    def variance(self) -> np.ndarray:
        v = self.second_moment - self.mean ** 2
        return np.maximum(v, 0.0)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    def check_finite(self) -> None:
        for nm, arr in (("p", self.p), ("u", self.u), ("w", self.w)):
            if not np.all(np.isfinite(arr)):
                raise ForwardCenteredError(
                    f"residual moment array '{nm}' contains non-finite values; "
                    "the residual dynamics are numerically unusable")


def residual_moments(
    spec: ResidualSpec,
    times_hours: np.ndarray,
    q01: np.ndarray,
    q10: np.ndarray,
    pi0: np.ndarray,
    x0: float = 0.0,
    sigma_price_path: Optional[np.ndarray] = None,
) -> ResidualMoments:
    """Integrate the regime/moment ODE system with classical RK4.

    All inputs are on the same time grid.  ``sigma_price_path`` has shape
    (2, n); when omitted the second moments are not meaningful (zeros).
    """
    t = np.asarray(times_hours, dtype=float)
    n = t.size
    if n < 2:
        raise ForwardCenteredError("need at least two time nodes")
    q01 = np.asarray(q01, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    if q01.shape != (n,) or q10.shape != (n,):
        raise ForwardCenteredError("generator paths must match the time grid")
    if np.any(q01 < 0) or np.any(q10 < 0):
        raise ForwardCenteredError("negative transition intensity")
    pi0 = np.asarray(pi0, dtype=float)
    if pi0.shape != (2,) or abs(pi0.sum() - 1.0) > 1e-8:
        raise ForwardCenteredError("pi0 must be a 2-vector probability")
    sig2 = (np.zeros((2, n)) if sigma_price_path is None
            else np.asarray(sigma_price_path, dtype=float) ** 2)
    if sig2.shape != (2, n):
        raise ForwardCenteredError("sigma_price_path must have shape (2, n)")

    k = spec.kappa_per_hour
    m = spec.regime_means
    a = spec.drift_shift_per_hour       # Q1 drift channel (see docs/risk_neutral_methodology.md)

    def interp(arr: np.ndarray, tt: float) -> np.ndarray:
        return np.array([np.interp(tt, t, arr[0]), np.interp(tt, t, arr[1])])

    q_stack = np.vstack([q01, q10])

    def deriv(tt: float, s: np.ndarray) -> np.ndarray:
        p, u, w = s[0:2], s[2:4], s[4:6]
        qq = interp(q_stack, tt)
        s2 = interp(sig2, tt)
        q01t, q10t = float(qq[0]), float(qq[1])
        dp = np.array([-q01t * p[0] + q10t * p[1],
                       q01t * p[0] - q10t * p[1]])
        # Q1 drift shift enters u_i via  + a_i p_i  and w_i via  + 2 a_i u_i
        # so mu_X(t) = u_0 + u_1 absorbs the shift and E^Q[P_t] = F(t) exactly.
        du = np.array([k * (m[0] * p[0] - u[0]) + a[0] * p[0] - q01t * u[0] + q10t * u[1],
                       k * (m[1] * p[1] - u[1]) + a[1] * p[1] + q01t * u[0] - q10t * u[1]])
        dw = np.array([2 * k * (m[0] * u[0] - w[0]) + 2 * a[0] * u[0]
                       + s2[0] * p[0] - q01t * w[0] + q10t * w[1],
                       2 * k * (m[1] * u[1] - w[1]) + 2 * a[1] * u[1]
                       + s2[1] * p[1] + q01t * w[0] - q10t * w[1]])
        return np.concatenate([dp, du, dw])

    state = np.concatenate([pi0, x0 * pi0, (x0 ** 2) * pi0])
    out = np.empty((6, n))
    out[:, 0] = state
    for j in range(n - 1):
        h = t[j + 1] - t[j]
        if h <= 0:
            raise ForwardCenteredError("time grid must be strictly increasing")

        max_internal_dt = 0.25
        n_sub = max(1, int(np.ceil(h / max_internal_dt)))
        hs = h / n_sub
        tt = t[j]

        for _ in range(n_sub):
            k1 = deriv(tt, state)
            k2 = deriv(tt + 0.5 * hs, state + 0.5 * hs * k1)
            k3 = deriv(tt + 0.5 * hs, state + 0.5 * hs * k2)
            k4 = deriv(tt + hs, state + hs * k3)

            state = state + (hs / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )
            tt += hs

        out[:, j + 1] = state
    mom = ResidualMoments(times_hours=t, p=out[0:2], u=out[2:4], w=out[4:6])
    mom.check_finite()
    ps = mom.p.sum(axis=0)
    if np.max(np.abs(ps - 1.0)) > 1e-8:
        raise ForwardCenteredError(
            f"regime probabilities drifted off the simplex (max dev "
            f"{np.max(np.abs(ps - 1.0)):.2e}); reduce the ODE step")
    return mom


# ---------------------------------------------------------------------------
@dataclass
class ForwardCenteredModel:
    """Bundle of forward curve + residual spec + TVTP chain, ready to price."""

    curve: ForwardCurve
    spec: ResidualSpec
    tvtp: TVTPCoefficients
    pi_filtered: np.ndarray
    valuation_utc: pd.Timestamp
    spot_price_TRY_MWh: float
    covariate_lag_hours: float = 1.0
    dt_hours: float = 1.0
    allow_spot_mismatch: bool = False

    def __post_init__(self) -> None:
        self.pi_filtered = np.asarray(self.pi_filtered, dtype=float)
        if self.pi_filtered.shape != (2,) or abs(self.pi_filtered.sum() - 1) > 1e-8:
            raise ForwardCenteredError("pi_filtered must be a 2-vector probability")
        if self.valuation_utc.tzinfo is None:
            raise ForwardCenteredError("valuation_utc must be tz-aware")
        f0 = float(self.curve.values.iloc[0])
        self.spot_consistent = abs(f0 - self.spot_price_TRY_MWh) <= 1e-6
        if not self.spot_consistent:
            # Strict centering enforces E[X_t] = 0 at EVERY t, including t = 0,
            # so E^Q[P_0] = F(t0) whatever x0 is: an assumed near-term level
            # that differs from the spot deliberately overrides the observed
            # spot.  That is exactly what a January-anchor sensitivity explores.
            (logger.info if self.allow_spot_mismatch else logger.warning)(
                "near-term anchor gives F(t0)=%.4f != spot %.4f, so the model "
                "implies E[P_0] = %.4f: the assumed near-term level overrides "
                "the observed spot. Use a spot-pinning anchor "
                "(spot_flat / spot_to_next_linear) for spot consistency.",
                f0, self.spot_price_TRY_MWh, f0)

    # -- state ------------------------------------------------------------
    @property
    def x0(self) -> float:
        """Initial residual, TRY/MWh (additive) or log-units (multiplicative)."""
        f0 = float(self.curve.values.iloc[0])
        if self.spec.x0_mode == "zero":
            return 0.0
        if self.spec.mode == "additive":
            return float(self.spot_price_TRY_MWh - f0)
        return float(np.log(self.spot_price_TRY_MWh / f0))

    def forward_at(self, times_hours: np.ndarray) -> np.ndarray:
        return self.curve.at_hours(self.valuation_utc, times_hours)

    def sigma_path(self, times_hours: np.ndarray) -> np.ndarray:
        """(2, n) residual volatilities along the solver grid."""
        return self.spec.sigma_price(self.forward_at(times_hours))

    def generator_path(self, times_hours: np.ndarray,
                       z_lagged: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """TVTP generator intensities q01(t), q10(t) per hour."""
        t = np.asarray(times_hours, dtype=float)
        z = np.zeros(t.size) if z_lagged is None else np.asarray(z_lagged, dtype=float)
        if z.shape != t.shape:
            raise ForwardCenteredError("z_lagged must match the time grid")
        p01, p10 = self.tvtp.probabilities(z)
        gen = probs_to_generator(p01, p10, dt_hours=self.dt_hours)
        if gen.n_clipped:
            logger.warning("TVTP embeddability clipping applied on %d nodes",
                           gen.n_clipped)
        return gen.q01, gen.q10

    # -- moments ----------------------------------------------------------
    def integration_grid(self, horizon_hours: float,
                         max_step_hours: float = 1.0) -> np.ndarray:
        """Dense grid on [0, horizon] for the moment ODE (must start at t=0)."""
        h = float(horizon_hours)
        if h <= 0:
            raise ForwardCenteredError("horizon must be positive")
        n = int(np.ceil(h / max_step_hours))
        return np.linspace(0.0, h, max(n, 2) + 1)

    def moments(self, times_hours: np.ndarray,
                z_lagged: Optional[np.ndarray] = None) -> ResidualMoments:
        """Residual moments on the supplied grid (which must start at t = 0)."""
        t = np.asarray(times_hours, dtype=float)
        if t.size < 2 or abs(float(t[0])) > 1e-12:
            raise ForwardCenteredError(
                "moments() needs a grid of at least two nodes starting at t=0; "
                "use moments_at() for arbitrary query times")
        q01, q10 = self.generator_path(t, z_lagged)
        return residual_moments(self.spec, t, q01, q10, self.pi_filtered,
                                x0=self.x0, sigma_price_path=self.sigma_path(t))

    def moments_at(self, query_hours: np.ndarray,
                   max_step_hours: float = 1.0) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
        """(mean, variance, p_stress, forward) at arbitrary horizons >= 0."""
        q = np.atleast_1d(np.asarray(query_hours, dtype=float))
        if np.any(q < 0):
            raise ForwardCenteredError("query horizons must be non-negative")
        grid = self.integration_grid(max(float(q.max()), max_step_hours),
                                     max_step_hours)
        mom = self.moments(grid)
        return (np.interp(q, grid, mom.mean), np.interp(q, grid, mom.variance),
                np.interp(q, grid, mom.p[1]), self.forward_at(q))

    def centering(self, times_hours: np.ndarray,
                  z_lagged: Optional[np.ndarray] = None) -> np.ndarray:
        """mu_X(t): the deterministic correction that zeroes E^Q[residual]."""
        mom = self.moments(times_hours, z_lagged)
        if self.spec.mode == "additive":
            return mom.mean
        # log-normal-mixture correction: c(t) = ln E[e^{X_t}] ~ mu + v/2
        return mom.mean + 0.5 * mom.variance

    # -- price map --------------------------------------------------------
    def price_from_state(self, x: np.ndarray | float, forward_level: float,
                         centering: float) -> np.ndarray:
        """Map the residual state to the spot price at one time slice."""
        x = np.asarray(x, dtype=float)
        if self.spec.mode == "additive":
            return forward_level + x - centering
        arg = x - centering
        if np.max(np.abs(arg)) > 300.0:
            raise ForwardCenteredError(
                "multiplicative residual exponent exceeded 300; the residual "
                "grid is far too wide for double precision")
        return forward_level * np.exp(arg)

    def expected_spot(self, query_hours: np.ndarray,
                      max_step_hours: float = 1.0) -> np.ndarray:
        """E^Q[P_t] at arbitrary horizons -- equals F(t) by construction.

        Computed as F(t) + (E[X_t] - mu_X(t)) rather than asserted, so the
        centering identity is verified numerically rather than assumed.
        """
        q = np.atleast_1d(np.asarray(query_hours, dtype=float))
        mean, var, _, f = self.moments_at(q, max_step_hours)
        if self.spec.mode == "additive":
            return f + mean - mean                       # exact zero residual mean
        return f * np.exp((mean + 0.5 * var) - (mean + 0.5 * var))

    def residual_summary(self, times_hours: np.ndarray,
                         z_lagged: Optional[np.ndarray] = None) -> pd.DataFrame:
        t = np.asarray(times_hours, dtype=float)
        mom = self.moments(t, z_lagged)
        f = self.forward_at(t)
        cen = mom.mean if self.spec.mode == "additive" else mom.mean + 0.5 * mom.variance
        if self.spec.mode == "additive":
            expected = f + mom.mean - cen
        else:
            expected = f * np.exp(mom.mean + 0.5 * mom.variance - cen)
        return pd.DataFrame({
            "hours": t,
            "forward_TRY_MWh": f,
            "residual_mean_TRY_MWh": mom.mean,
            "residual_std_TRY_MWh": mom.std,
            "p_stress": mom.p[1],
            "expected_spot_TRY_MWh": expected,
        })


# ---------------------------------------------------------------------------
# grid + pricing
# ---------------------------------------------------------------------------
@dataclass
class ResidualGridSettings:
    """Space/time grid for the residual PDE (state in TRY/MWh)."""

    n_space_nodes: int = 1201
    n_time_steps: Optional[int] = None
    n_std: float = 6.0
    min_halfwidth_TRY_MWh: float = 500.0
    max_halfwidth_TRY_MWh: float = 60000.0
    x_min: Optional[float] = None
    x_max: Optional[float] = None

    def n_steps(self, tau_hours: float) -> int:
        if self.n_time_steps is not None:
            return int(self.n_time_steps)
        return max(96, int(np.ceil(2.0 * tau_hours)))


@dataclass
class ForwardCenteredPricingResult:
    contract: EuropeanOption
    value: float
    V_regime: np.ndarray
    pi: np.ndarray
    forward_at_expiry: float
    centering_at_expiry: float
    expected_spot_at_expiry: float
    residual_std_at_expiry: float
    x0: float
    solve: SolveResult = field(repr=False)
    price_label: str = "VEP-forward-curve anchored option price"
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        return {
            "option_type": self.contract.option_type,
            "strike_TRY_MWh": self.contract.strike,
            "tau_hours": self.contract.tau_hours,
            "forward_at_expiry_TRY_MWh": self.forward_at_expiry,
            "expected_spot_at_expiry_TRY_MWh": self.expected_spot_at_expiry,
            "residual_std_at_expiry_TRY_MWh": self.residual_std_at_expiry,
            "V_regime0_normal": float(self.V_regime[0]),
            "V_regime1_stress": float(self.V_regime[1]),
            "pi_normal": float(self.pi[0]), "pi_stress": float(self.pi[1]),
            "value_TRY_MWh": self.value,
            "price_label": self.price_label,
            "max_peclet": self.solve.max_peclet,
            "upwinded_fraction": self.solve.upwinded_fraction,
        }


def build_residual_grid(model: ForwardCenteredModel, contract: EuropeanOption,
                        gs: ResidualGridSettings,
                        z_lagged: Optional[np.ndarray] = None) -> SpaceGrid:
    """Size the residual grid from the ANALYTIC residual dispersion."""
    if gs.x_min is not None and gs.x_max is not None:
        return SpaceGrid(gs.x_min, gs.x_max, gs.n_space_nodes)
    tau = contract.tau_hours
    t = model.integration_grid(tau)
    z_int = None
    if z_lagged is not None:
        z_src = np.asarray(z_lagged, dtype=float)
        t_src = np.linspace(0.0, tau, z_src.size)
        z_int = np.interp(t, t_src, z_src)
    mom = model.moments(t, z_int)
    sd = float(mom.std[-1])
    half = float(np.clip(gs.n_std * max(sd, 1e-9),
                         gs.min_halfwidth_TRY_MWh, gs.max_halfwidth_TRY_MWh))
    centre = model.x0
    f_T = float(model.forward_at(np.array([tau]))[0])
    cen_T = float((mom.mean if model.spec.mode == "additive"
                   else mom.mean + 0.5 * mom.variance)[-1])
    if model.spec.mode == "additive":
        x_strike = contract.strike - f_T + cen_T
    else:
        x_strike = float(np.log(max(contract.strike, 1e-9) / f_T)) + cen_T
    lo = min(centre, x_strike) - half
    hi = max(centre, x_strike) + half
    logger.info("residual grid: x in [%.1f, %.1f] (sd_T=%.1f, half=%.1f, %d nodes)",
                lo, hi, sd, half, gs.n_space_nodes)
    return SpaceGrid(lo, hi, gs.n_space_nodes)


def price_forward_centered(
    model: ForwardCenteredModel,
    contract: EuropeanOption,
    grid_settings: Optional[ResidualGridSettings] = None,
    solver_settings: Optional[SolverSettings] = None,
    z_lagged_fn=None,
    grid: Optional[SpaceGrid] = None,
) -> ForwardCenteredPricingResult:
    """Price a European option on the EXPIRY-HOUR spot under the centered model.

    The contract is an option on P_T at the single expiry hour T, not on a
    monthly baseload average.  Only the terminal map changes relative to the
    legacy model; the coupled two-regime PDE machinery is reused unchanged.
    """
    gs = grid_settings or ResidualGridSettings()
    tgrid = TimeGrid(contract.valuation_utc, contract.maturity_utc,
                     gs.n_steps(contract.tau_hours))
    t = tgrid.times_hours
    z_lag = None if z_lagged_fn is None else np.asarray(z_lagged_fn(t), dtype=float)

    f_path = model.forward_at(t)
    sig_path = model.spec.sigma_price(f_path)               # (2, n)
    q01, q10 = model.generator_path(t, z_lag)
    mom = residual_moments(model.spec, t, q01, q10, model.pi_filtered,
                           x0=model.x0, sigma_price_path=sig_path)
    cen = (mom.mean if model.spec.mode == "additive"
           else mom.mean + 0.5 * mom.variance)

    if grid is None:
        grid = build_residual_grid(model, contract, gs, z_lag)
    if not grid.contains(model.x0):
        raise ForwardCenteredError(
            f"x0={model.x0:.3f} too close to the residual grid boundary "
            f"[{grid.y_min:.3f}, {grid.y_max:.3f}]")

    kappa, m_i = model.spec.kappa_per_hour, model.spec.regime_means
    a_i = model.spec.drift_shift_per_hour   # Q1 drift channel; symmetric with the moment ODE

    def drift_fn(tt: float) -> np.ndarray:
        return kappa * (m_i[:, None] - grid.y[None, :]) + a_i[:, None]

    def sigma_fn(tt: float) -> np.ndarray:
        return np.array([np.interp(tt, t, sig_path[0]),
                         np.interp(tt, t, sig_path[1])])

    def generator_fn(tt: float) -> Tuple[float, float]:
        return float(np.interp(tt, t, q01)), float(np.interp(tt, t, q10))

    f_T, cen_T = float(f_path[-1]), float(cen[-1])
    prices_T = model.price_from_state(grid.y, f_T, cen_T)
    if not np.all(np.isfinite(prices_T)):
        raise ForwardCenteredError("terminal price map produced non-finite values")
    pay = contract.payoff_from_price(prices_T)
    terminal = np.vstack([pay, pay])

    res = solve_coupled_pde(
        grid, tgrid, terminal, drift_fn, sigma_fn(0.0), contract.r_per_hour,
        generator_fn, bc=None, settings=solver_settings, sigma_fn=sigma_fn)
    if not np.all(np.isfinite(res.V)):
        raise ForwardCenteredError(
            "PDE solution contains non-finite values; refusing to report a price")

    v_reg = np.array([grid.interp(res.V[0], model.x0),
                      grid.interp(res.V[1], model.x0)])
    value = float(model.pi_filtered @ v_reg)
    if not np.isfinite(value):
        raise ForwardCenteredError("option value is not finite")

    return ForwardCenteredPricingResult(
        contract=contract, value=value, V_regime=v_reg, pi=model.pi_filtered.copy(),
        forward_at_expiry=f_T, centering_at_expiry=cen_T,
        expected_spot_at_expiry=float(f_path[-1]),
        residual_std_at_expiry=float(mom.std[-1]), x0=model.x0, solve=res,
        diagnostics={
            "grid_x_min": grid.y_min, "grid_x_max": grid.y_max,
            "n_space_nodes": float(grid.y.size), "n_time_steps": float(tgrid.n_steps),
            "p_stress_at_expiry": float(mom.p[1][-1]),
        },
    )


# ---------------------------------------------------------------------------
def simulate_forward_centered(
    model: ForwardCenteredModel,
    contract: EuropeanOption,
    n_paths: int = 60_000,
    dt_hours: float = 0.25,
    seed: int = 20260808,
    z_lagged_fn=None,
) -> Dict[str, float]:
    """Time-discretized Monte Carlo cross-check of the residual PDE.

     Per step the regime jumps using the frozen-generator transition probability
and the residual advances with the exact conditional OU transition.
The joint CTMC-OU simulation is time-discretized and converges as dt decreases. Reports the random
    seed and the standard error, as required for any Monte Carlo output.
    """
    rng = np.random.default_rng(seed)
    tau = contract.tau_hours
    n_steps = max(int(np.ceil(tau / dt_hours)), 4)
    dt = tau / n_steps
    t = np.linspace(0.0, tau, n_steps + 1)
    z_lag = None if z_lagged_fn is None else np.asarray(z_lagged_fn(t), dtype=float)

    f_path = model.forward_at(t)
    sig_path = model.spec.sigma_price(f_path)
    q01, q10 = model.generator_path(t, z_lag)
    mom = residual_moments(model.spec, t, q01, q10, model.pi_filtered,
                           x0=model.x0, sigma_price_path=sig_path)
    cen = mom.mean if model.spec.mode == "additive" else mom.mean + 0.5 * mom.variance

    k = model.spec.kappa_per_hour
    m_i = model.spec.regime_means
    # Q1 drift shift a_i is equivalent to an effective mean m_eff_i = m_i + a_i/kappa,
    # so the exact conditional-OU step re-uses the closed-form transition with m_eff.
    m_eff = m_i + model.spec.drift_shift_per_hour / k
    e1 = np.exp(-k * dt)

    x = np.full(n_paths, model.x0, dtype=float)
    reg = (rng.random(n_paths) < model.pi_filtered[1]).astype(np.int8)
    for j in range(n_steps):
        p01j, p10j = generator_to_probs(np.array([q01[j]]), np.array([q10[j]]),
                                        dt_hours=dt)
        u = rng.random(n_paths)
        reg = np.where((reg == 0) & (u < p01j[0]), 1,
                       np.where((reg == 1) & (u < p10j[0]), 0, reg)).astype(np.int8)
        sig_mid = 0.5 * (sig_path[:, j] + sig_path[:, j + 1])
        sd = sig_mid * np.sqrt((1.0 - np.exp(-2 * k * dt)) / (2 * k))
        x = m_eff[reg] + (x - m_eff[reg]) * e1 + sd[reg] * rng.standard_normal(n_paths)

    prices = model.price_from_state(x, float(f_path[-1]), float(cen[-1]))
    pay = contract.payoff_from_price(prices)
    disc = np.exp(-contract.r_per_hour * tau)
    return {
        "value": float(disc * pay.mean()),
        "std_error": float(disc * pay.std(ddof=1) / np.sqrt(n_paths)),
        "mean_price_T": float(prices.mean()),
        "mean_price_T_se": float(prices.std(ddof=1) / np.sqrt(n_paths)),
        "analytic_expected_spot_T": float(f_path[-1]),
        "prob_negative_price": float((prices < 0).mean()),
        "n_paths": float(n_paths), "dt_hours": float(dt), "seed": float(seed),
    }
