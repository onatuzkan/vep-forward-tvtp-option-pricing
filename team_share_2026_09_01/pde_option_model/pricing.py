"""End-to-end pricing of a contract under one residual-demand scenario.

Total value at the valuation date (filtered probabilities only -- smoothed
probabilities contain future information and are never used here):

    V(t0, y0) = pi_0^filtered * V_0(t0, y0) + pi_1^filtered * V_1(t0, y0)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .boundaries import BoundaryCondition, DirichletMeanRevertingOU, GammaZeroY
from .contracts import EuropeanOption
from .dynamics import RegimeDynamics
from .generator import probs_to_generator, stationary_distribution
from .grid import SpaceGrid, TimeGrid
from .markov_adapter import MarkovInputs
from .risk_neutral import MeasureAdjustment, baseline_q1
from .scenarios import ScenarioPath
from .solver import SolverSettings, SolveResult, solve_coupled_pde

logger = logging.getLogger(__name__)


@dataclass
class GridSettings:
    n_space_nodes: int = 561
    n_time_steps: Optional[int] = None      # default: 2 steps per hour, min 96
    y_min: Optional[float] = None           # None -> automatic
    y_max: Optional[float] = None
    auto_n_std: float = 4.0
    auto_min_halfwidth: float = 1.5
    y_hard_min: float = -6.0
    y_hard_max: float = 10.0
    boundary: str = "gamma_zero"            # or "dirichlet_ou"

    def n_steps(self, tau_hours: float) -> int:
        if self.n_time_steps is not None:
            return int(self.n_time_steps)
        return max(96, int(np.ceil(2.0 * tau_hours)))


@dataclass
class PricingResult:
    contract: EuropeanOption
    scenario: str
    adjustment: MeasureAdjustment
    V_regime: np.ndarray            # (2,) values at y0 by initial regime
    pi: np.ndarray
    value: float
    y0: float
    price0: float
    solve: SolveResult = field(repr=False)
    price_label: str = ""

    def summary(self) -> dict:
        return {
            "scenario": self.scenario,
            "option_type": self.contract.option_type,
            "strike": self.contract.strike,
            "tau_hours": self.contract.tau_hours,
            "spot_price": self.price0,
            "V_regime0_normal": float(self.V_regime[0]),
            "V_regime1_stress": float(self.V_regime[1]),
            "pi_normal": float(self.pi[0]),
            "pi_stress": float(self.pi[1]),
            "value_weighted": self.value,
            "measure": self.adjustment.describe(),
            "price_label": self.price_label,
            "max_peclet": self.solve.max_peclet,
            "upwinded_fraction": self.solve.upwinded_fraction,
        }


# ---------------------------------------------------------------------------
def build_dynamics(inputs: MarkovInputs, include_rd_in_theta: bool = True,
                   sigma_multipliers: Tuple[float, float] = (1.0, 1.0)
                   ) -> RegimeDynamics:
    return RegimeDynamics(
        ou=inputs.ou,
        seasonal=inputs.seasonal,
        rho=inputs.rho,
        include_rd_in_theta=include_rd_in_theta,
        sigma_multipliers=np.asarray(sigma_multipliers, float),
    )


def effective_mixture_sigma(inputs: MarkovInputs, z_ref: float,
                            dynamics: Optional[RegimeDynamics] = None) -> float:
    """Occupancy-weighted volatility used only to size the automatic grid."""
    p01, p10 = inputs.tvtp.probabilities(np.array([z_ref]))
    gen = probs_to_generator(p01, p10, dt_hours=inputs.dt_hours)
    pi = stationary_distribution(float(gen.q01[0]), float(gen.q10[0]))
    sig = dynamics.sigma() if dynamics is not None else inputs.ou.sigma_ou
    return float(np.sqrt(pi @ (sig**2)))


def build_grid(inputs: MarkovInputs, contract: EuropeanOption,
               scenario: ScenarioPath, gs: GridSettings,
               dynamics: Optional[RegimeDynamics] = None) -> SpaceGrid:
    if gs.y_min is not None and gs.y_max is not None:
        return SpaceGrid(gs.y_min, gs.y_max, gs.n_space_nodes)
    sbar = effective_mixture_sigma(inputs, float(np.mean(scenario.z)), dynamics)
    half = max(gs.auto_n_std * sbar * np.sqrt(contract.tau_hours),
               gs.auto_min_halfwidth)
    y_strike = float(inputs.transform.y_from_price(contract.strike))
    lo = min(inputs.y0, y_strike) - half
    hi = max(inputs.y0, y_strike) + half
    lo = gs.y_min if gs.y_min is not None else max(lo, gs.y_hard_min)
    hi = gs.y_max if gs.y_max is not None else min(hi, gs.y_hard_max)
    if hi - lo < 2 * gs.auto_min_halfwidth:
        raise ValueError("automatic grid collapsed; widen the hard limits")
    logger.info("auto grid: y in [%.3f, %.3f] (sigma_mix=%.4f, tau=%.1f h)",
                lo, hi, sbar, contract.tau_hours)
    return SpaceGrid(lo, hi, gs.n_space_nodes)


def _coefficient_closures(
    inputs: MarkovInputs,
    dynamics: RegimeDynamics,
    contract: EuropeanOption,
    scenario: ScenarioPath,
    tgrid: TimeGrid,
    adjustment: MeasureAdjustment,
    grid: SpaceGrid,
):
    """Precompute theta(t) and q(t) on the solver grid; return closures."""
    t = tgrid.times_hours
    z_now = np.interp(t, scenario.times_hours, scenario.z)
    z_lag = np.interp(t, scenario.times_hours, scenario.z_lagged)

    theta = dynamics.theta_path(tgrid.valuation_utc, t, z_now, adjustment=None)
    p01, p10 = inputs.tvtp.probabilities(z_lag)
    gen = probs_to_generator(p01, p10, dt_hours=inputs.dt_hours)
    q01, q10 = adjustment.adjust_generator(gen.q01, gen.q10)

    # remaining-horizon average of theta for the Dirichlet far-field value
    rev0 = np.cumsum(theta[0][::-1])[::-1] / np.arange(t.size, 0, -1)
    rev1 = np.cumsum(theta[1][::-1])[::-1] / np.arange(t.size, 0, -1)
    theta_bar = np.vstack([rev0, rev1]) + adjustment.theta_shift[:, None]

    def drift_fn(tt: float) -> np.ndarray:
        th = np.array([np.interp(tt, t, theta[0]), np.interp(tt, t, theta[1])])
        return dynamics.drift(grid.y, th, adjustment=adjustment)

    def generator_fn(tt: float) -> Tuple[float, float]:
        return (float(np.interp(tt, t, q01)), float(np.interp(tt, t, q10)))

    def theta_bar_fn(tt: float) -> np.ndarray:
        return np.array([np.interp(tt, t, theta_bar[0]),
                         np.interp(tt, t, theta_bar[1])])

    diag = {"n_clipped_generator": gen.n_clipped,
            "q01_range": (float(q01.min()), float(q01.max())),
            "q10_range": (float(q10.min()), float(q10.max()))}
    return drift_fn, generator_fn, theta_bar_fn, diag


def make_boundary(kind: str, contract: EuropeanOption, inputs: MarkovInputs,
                  grid: SpaceGrid) -> BoundaryCondition:
    if kind == "gamma_zero":
        return GammaZeroY()
    if kind == "dirichlet_ou":
        return DirichletMeanRevertingOU(
            contract=contract, transform=inputs.transform,
            kappa=inputs.ou.kappa_per_hour, y_bounds=(grid.y_min, grid.y_max))
    raise ValueError(f"unknown boundary kind {kind!r}")


def _check_pi(pi: np.ndarray) -> None:
    """Filtered regime probabilities must be a valid distribution."""
    if pi.shape != (2,):
        raise ValueError("pi must have shape (2,)")
    if np.any(pi < -1e-12) or np.any(pi > 1 + 1e-12):
        raise ValueError("regime probabilities must lie in [0, 1]")
    if abs(float(pi.sum()) - 1.0) > 1e-8:
        raise ValueError("regime probabilities must sum to one")


def price_contract(
    inputs: MarkovInputs,
    contract: EuropeanOption,
    scenario: ScenarioPath,
    adjustment: Optional[MeasureAdjustment] = None,
    grid_settings: Optional[GridSettings] = None,
    dynamics: Optional[RegimeDynamics] = None,
    solver_settings: Optional[SolverSettings] = None,
    pi_override: Optional[np.ndarray] = None,
    terminal_override: Optional[np.ndarray] = None,
    grid: Optional[SpaceGrid] = None,
) -> PricingResult:
    adjustment = adjustment or baseline_q1()
    gs = grid_settings or GridSettings()
    dynamics = dynamics or build_dynamics(inputs)

    if grid is None:
        grid = build_grid(inputs, contract, scenario, gs, dynamics)
    if not grid.contains(inputs.y0):
        raise ValueError(f"y0={inputs.y0:.3f} too close to the grid boundary "
                         f"[{grid.y_min:.3f}, {grid.y_max:.3f}]")
    tgrid = TimeGrid(contract.valuation_utc, contract.maturity_utc,
                     gs.n_steps(contract.tau_hours))

    drift_fn, generator_fn, theta_bar_fn, cdiag = _coefficient_closures(
        inputs, dynamics, contract, scenario, tgrid, adjustment, grid)
    bc = make_boundary(gs.boundary, contract, inputs, grid)

    if terminal_override is not None:
        term = np.vstack([terminal_override, terminal_override])
    else:
        pay = contract.payoff_on_grid(grid.y, inputs.transform)
        term = np.vstack([pay, pay])

    res = solve_coupled_pde(
        grid, tgrid, term, drift_fn, dynamics.sigma(), contract.r_per_hour,
        generator_fn, bc=bc, theta_bar_fn=theta_bar_fn, settings=solver_settings)

    v_reg = np.array([grid.interp(res.V[0], inputs.y0),
                      grid.interp(res.V[1], inputs.y0)])
    pi = np.asarray(pi_override if pi_override is not None else inputs.pi_filtered,
                    dtype=float)
    _check_pi(pi)
    value = float(pi @ v_reg)
    if cdiag["n_clipped_generator"]:
        logger.warning("generator embeddability clipping was applied on %d nodes",
                       cdiag["n_clipped_generator"])

    return PricingResult(
        contract=contract, scenario=scenario.name, adjustment=adjustment,
        V_regime=v_reg, pi=pi, value=value, y0=inputs.y0, price0=inputs.price0,
        solve=res, price_label=adjustment.label(),
    )


def model_forward_price(
    inputs: MarkovInputs,
    scenario: ScenarioPath,
    maturity_utc,
    adjustment: Optional[MeasureAdjustment] = None,
    grid_settings: Optional[GridSettings] = None,
    dynamics: Optional[RegimeDynamics] = None,
) -> float:
    """Model-implied forward F(t0, T) = E^Q[P_T], via the same PDE with a
    linear payoff P(y) and zero discounting."""
    contract = EuropeanOption("call", strike=1.0,
                              valuation_utc=inputs.valuation_utc,
                              maturity_utc=maturity_utc, r_annual=0.0)
    gs = grid_settings or GridSettings()
    dynamics = dynamics or build_dynamics(inputs)
    grid = build_grid(inputs, contract, scenario, gs, dynamics)
    linear_terminal = np.asarray(inputs.transform.price_from_y(grid.y), dtype=float)
    res = price_contract(
        inputs, contract, scenario, adjustment=adjustment, grid_settings=gs,
        dynamics=dynamics, terminal_override=linear_terminal)
    return res.value
