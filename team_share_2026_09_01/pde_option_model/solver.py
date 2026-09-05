"""Finite-difference solver for the coupled two-regime backward PDE.

    dV_i/dt + b_i^Q(t,y) dV_i/dy + 0.5 sigma_i^2 d2V_i/dy2 - r V_i
            + sum_{j != i} q_ij^Q(t) (V_j - V_i) = 0,   i in {0, 1},
    V_i(T, y) = payoff(P(y)).

Discretization
--------------
* theta-scheme in time (theta = 1/2: Crank-Nicolson; theta = 1: implicit
  Euler), with an optional Rannacher startup (first steps fully implicit) to
  damp the payoff kink.  Time-dependent coefficients are evaluated at the
  respective time levels (trapezoidal rule).
* Central differences in space; node-wise first-order upwinding of the drift
  whenever the local cell Peclet number  |b| h / sigma^2  exceeds 1
  (M-matrix / positivity criterion).
* The two regimes are solved SIMULTANEOUSLY at every backward step as one
  sparse block system

      [ A_0 + theta dt q01 I      -theta dt q01 I    ] [V_0]   [b_0]
      [    -theta dt q10 I     A_1 + theta dt q10 I  ] [V_1] = [b_1]

  with A_i = I - theta dt L_i; the coupling is treated implicitly, never as an
  explicit source added after independent regime solves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .boundaries import BoundaryCondition, GammaZeroY
from .grid import SpaceGrid, TimeGrid

logger = logging.getLogger(__name__)

DriftFn = Callable[[float], np.ndarray]        # t_hours -> (2, Ny) drift b_i(t, y)
GeneratorFn = Callable[[float], Tuple[float, float]]  # t_hours -> (q01, q10)
SigmaFn = Callable[[float], np.ndarray]        # t_hours -> (2,) regime volatilities
ThetaBarFn = Callable[[float], np.ndarray]     # t_hours -> (2,) theta_i(t) for BC


@dataclass
class SolverSettings:
    theta_scheme: float = 0.5          # 0.5 CN, 1.0 implicit Euler
    rannacher_steps: int = 2
    upwind_peclet_threshold: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.theta_scheme <= 1.0):
            raise ValueError("theta_scheme must lie in (0, 1]")
        if self.rannacher_steps < 0:
            raise ValueError("rannacher_steps must be >= 0")


@dataclass
class SolveResult:
    V: np.ndarray                       # (2, Ny) values at the valuation date
    grid: SpaceGrid
    tgrid: TimeGrid
    max_peclet: float
    upwinded_fraction: float
    settings: SolverSettings
    snapshots: dict = field(default_factory=dict)   # optional time slices


def _operator_coeffs(
    y: np.ndarray, h: float, b: np.ndarray, sigma2: float, r: float,
    peclet_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal coefficients (lower, diag, upper) of L on interior nodes.

    Returns (lo, di, up, upwind_mask) each of length Ny (boundary entries
    unused).  L V = b V_y + 0.5 sigma^2 V_yy - r V.
    """
    n = y.size
    lo = np.zeros(n)
    di = np.zeros(n)
    up = np.zeros(n)
    D = 0.5 * sigma2
    pe = np.abs(b) * h / max(sigma2, 1e-300)
    upwind = pe > peclet_threshold

    # central
    lo_c = D / h**2 - b / (2.0 * h)
    up_c = D / h**2 + b / (2.0 * h)
    di_c = -2.0 * D / h**2 - r

    # upwind (first order, direction of b)
    pos = b >= 0
    lo_u = np.where(pos, D / h**2, D / h**2 - b / h)
    up_u = np.where(pos, D / h**2 + b / h, D / h**2)
    di_u = np.where(pos, -2.0 * D / h**2 - b / h - r,
                    -2.0 * D / h**2 + b / h - r)

    lo[:] = np.where(upwind, lo_u, lo_c)
    up[:] = np.where(upwind, up_u, up_c)
    di[:] = np.where(upwind, di_u, di_c)
    return lo, di, up, upwind


def _assemble_G(
    grid: SpaceGrid,
    b2: np.ndarray,                 # (2, Ny) drift at this time level
    sigma: np.ndarray,              # (2,)
    r: float,
    q01: float,
    q10: float,
    peclet_threshold: float,
) -> Tuple[sp.csr_matrix, float, float]:
    """Sparse 2Ny x 2Ny operator G = blockdiag(L_0, L_1) + coupling.

    Rows corresponding to boundary nodes are left at zero -- the caller
    overrides them with boundary equations.
    """
    n = grid.y.size
    rows, cols, vals = [], [], []
    max_pe = 0.0
    upw_ct = 0
    q_out = (q01, q10)
    q_in_block = (1, 0)
    for i in range(2):
        lo, di, up, upwind = _operator_coeffs(
            grid.y, grid.h, b2[i], float(sigma[i] ** 2), r, peclet_threshold)
        max_pe = max(max_pe, float((np.abs(b2[i]) * grid.h /
                                    max(sigma[i] ** 2, 1e-300))[1:-1].max(initial=0.0)))
        upw_ct += int(upwind[1:-1].sum())
        base = i * n
        idx = np.arange(1, n - 1)
        rows += list(base + idx) * 3
        cols += list(base + idx - 1) + list(base + idx) + list(base + idx + 1)
        vals += list(lo[idx]) + list(di[idx] - q_out[i]) + list(up[idx])
        # coupling to the other regime, same node
        other = q_in_block[i] * n
        rows += list(base + idx)
        cols += list(other + idx)
        vals += [q_out[i]] * idx.size
    G = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))
    frac = upw_ct / max(2 * (n - 2), 1)
    return G, max_pe, frac


def solve_coupled_pde(
    grid: SpaceGrid,
    tgrid: TimeGrid,
    terminal: np.ndarray,               # (2, Ny)
    drift_fn: DriftFn,
    sigma: np.ndarray,                  # (2,)
    r_per_hour: float,
    generator_fn: GeneratorFn,
    bc: BoundaryCondition | None = None,
    theta_bar_fn: Optional[ThetaBarFn] = None,
    settings: SolverSettings | None = None,
    snapshot_times: Tuple[float, ...] = (),
    sigma_fn: Optional[SigmaFn] = None,
) -> SolveResult:
    """March the coupled system backward from maturity to the valuation date.

    ``sigma_fn`` optionally supplies TIME-DEPENDENT regime volatilities
    sigma_i(t); when omitted the constant ``sigma`` vector is used at every
    time level, so existing callers are unaffected.  Needed by the
    forward-centered model, whose residual volatility is expressed in TRY/MWh
    and therefore tracks the market forward level F(t).
    """
    bc = bc or GammaZeroY()
    settings = settings or SolverSettings()
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (2,) or np.any(sigma <= 0):
        raise ValueError("sigma must be two positive regime volatilities")
    if terminal.shape != (2, grid.y.size):
        raise ValueError("terminal condition has wrong shape")

    def sig_at(t: float) -> np.ndarray:
        if sigma_fn is None:
            return sigma
        s_t = np.asarray(sigma_fn(float(t)), dtype=float)
        if s_t.shape != (2,) or np.any(s_t <= 0) or not np.all(np.isfinite(s_t)):
            raise ValueError(
                f"sigma_fn returned an invalid volatility vector at t={t}: {s_t}")
        return s_t
    if bc.kind() == "dirichlet_ou" and theta_bar_fn is None:
        raise ValueError("Dirichlet OU boundary requires theta_bar_fn")

    n = grid.y.size
    V = terminal.reshape(2 * n).astype(float).copy()
    times = tgrid.times_hours
    dt = tgrid.dt_hours
    eye = sp.identity(2 * n, format="csr")

    b_idx = np.array([0, n - 1, n, 2 * n - 1])          # boundary row indices
    gz_row = np.array([1.0, -2.0, 1.0])

    max_pe_seen, upw_frac_acc = 0.0, 0.0
    snapshots: dict = {}
    n_steps = tgrid.n_steps
    G_next = None
    for step in range(n_steps):
        t_new = times[n_steps - 1 - step]               # unknown level
        t_old = times[n_steps - step]                   # known level
        th = 1.0 if step < settings.rannacher_steps else settings.theta_scheme

        if G_next is None:
            q01o, q10o = generator_fn(t_old)
            G_old, _, _ = _assemble_G(grid, drift_fn(t_old), sig_at(t_old), r_per_hour,
                                      q01o, q10o, settings.upwind_peclet_threshold)
        else:
            G_old = G_next
        q01n, q10n = generator_fn(t_new)
        sigma_new = sig_at(t_new)
        G_new, mpe, ufr = _assemble_G(grid, drift_fn(t_new), sigma_new, r_per_hour,
                                      q01n, q10n, settings.upwind_peclet_threshold)
        max_pe_seen = max(max_pe_seen, mpe)
        upw_frac_acc += ufr

        M_L = (eye - th * dt * G_new).tolil()
        rhs = V + (1.0 - th) * dt * (G_old @ V)

        # boundary rows
        tau_rem = tgrid.tau_hours - t_new
        for i in range(2):
            lo_r, hi_r = i * n, i * n + n - 1
            if bc.kind() == "gamma_zero":
                M_L.rows[lo_r] = [lo_r, lo_r + 1, lo_r + 2]
                M_L.data[lo_r] = list(gz_row)
                rhs[lo_r] = 0.0
                M_L.rows[hi_r] = [hi_r - 2, hi_r - 1, hi_r]
                M_L.data[hi_r] = list(gz_row)
                rhs[hi_r] = 0.0
            else:
                thb = theta_bar_fn(t_new)               # type: ignore[misc]
                M_L.rows[lo_r] = [lo_r]
                M_L.data[lo_r] = [1.0]
                rhs[lo_r] = bc.dirichlet_value("lower", i, tau_rem,
                                               float(thb[i]), float(sigma_new[i]))
                M_L.rows[hi_r] = [hi_r]
                M_L.data[hi_r] = [1.0]
                rhs[hi_r] = bc.dirichlet_value("upper", i, tau_rem,
                                               float(thb[i]), float(sigma_new[i]))

        V = spla.spsolve(M_L.tocsc(), rhs)
        G_next = G_new
        for st in snapshot_times:
            if abs(t_new - st) < 0.5 * dt and st not in snapshots:
                snapshots[st] = V.reshape(2, n).copy()

    result = SolveResult(
        V=V.reshape(2, n),
        grid=grid,
        tgrid=tgrid,
        max_peclet=max_pe_seen,
        upwinded_fraction=upw_frac_acc / max(n_steps, 1),
        settings=settings,
        snapshots=snapshots,
    )
    logger.debug("solve done: max Peclet %.3g, mean upwinded fraction %.3g",
                 result.max_peclet, result.upwinded_fraction)
    return result


def solve_single_regime(
    grid: SpaceGrid,
    tgrid: TimeGrid,
    terminal: np.ndarray,               # (Ny,)
    drift_fn: Callable[[float], np.ndarray],   # t -> (Ny,)
    sigma_i: float,
    r_per_hour: float,
    bc: BoundaryCondition | None = None,
    theta_bar_fn: Optional[Callable[[float], float]] = None,
    settings: SolverSettings | None = None,
) -> np.ndarray:
    """Reference single-regime solve (used by limiting-case tests).

    Implemented by running the coupled solver with q01 = q10 = 0 and two
    identical regimes, then returning regime 0 -- guaranteeing an identical
    discretization to the coupled path.
    """
    term2 = np.vstack([terminal, terminal])

    def d2(t: float) -> np.ndarray:
        b = drift_fn(t)
        return np.vstack([b, b])

    tb = None
    if theta_bar_fn is not None:
        def tb(t: float) -> np.ndarray:      # type: ignore[misc]
            v = float(theta_bar_fn(t))
            return np.array([v, v])

    res = solve_coupled_pde(
        grid, tgrid, term2, d2, np.array([sigma_i, sigma_i]), r_per_hour,
        generator_fn=lambda t: (0.0, 0.0), bc=bc, theta_bar_fn=tb,
        settings=settings,
    )
    return res.V[0]
