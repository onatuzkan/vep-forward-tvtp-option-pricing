"""Boundary conditions for the coupled PDE in the transformed variable y.

Standard Black-Scholes boundaries are inappropriate here: the price is
mean-reverting, the state variable is y = asinh(P / scale_P), and the payoff
P(y) = scale_P * sinh(y) grows exponentially in |y|.  Two treatments are
provided.

1. GammaZeroY (default): impose  d^2 V / dy^2 = 0  at both ends
   (one-sided:  V_0 - 2 V_1 + V_2 = 0).
   Rationale: payoff-agnostic and model-agnostic; it lets the solution leave
   the domain linearly in y.  Because the payoff is convex in y at the upper
   end (P ~ e^y), linear extrapolation slightly UNDERSTATES deep-in-the-money
   call convexity; the error is confined near the boundary and controlled by
   padding the domain (see the domain-sensitivity check in validation).
   Trade-off: robust, no extra model assumptions, but only first-order
   consistent for convex far-field solutions.

2. DirichletMeanRevertingOU (alternative): pin the boundary value to the
   closed-form discounted expected payoff of the SINGLE-REGIME OU dynamics,
   ignoring regime switching at the boundary.  For y ~ OU with
   m(tau) = theta_bar + (y_b - theta_bar) e^{-kappa tau},
   v(tau) = sigma_i^2 (1 - e^{-2 kappa tau}) / (2 kappa),
   the asinh transform gives  E[P_T] = scale_P * e^{v/2} * sinh(m), hence

       call:  V_i(t, y_b) = e^{-r tau} * max(E[P_T] - K, 0)
       put:   V_i(t, y_b) = e^{-r tau} * max(K - E[P_T], 0).

   Rationale: matches the exact far-field mean-reverting solution when
   switching is negligible; asymptotically consistent with the convex payoff.
   Trade-off: hard-codes "no switching at the boundary" (error grows with
   q * tau) and inherits any drift misspecification into the boundary value;
   with the near-unit-root kappa of this model, v(tau) ~ sigma^2 tau grows
   almost linearly, so the upper Dirichlet value is very large -- correct for
   the model, but it drags model risk onto the boundary.

Both treatments are exercised in the validation suite; with adequate domain
padding they agree in the interior to well under the discretization error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np

from .contracts import EuropeanOption
from .transformations import PriceTransform


class BoundaryCondition(Protocol):
    """Protocol: per (time, regime, side) either a Gamma=0 row or a Dirichlet value."""

    def kind(self) -> str: ...

    def dirichlet_value(
        self, side: str, regime: int, tau_remaining_hours: float,
        theta_bar: float, sigma_i: float,
    ) -> float: ...


@dataclass(frozen=True)
class GammaZeroY:
    """d2V/dy2 = 0 at y_min and y_max (default)."""

    def kind(self) -> str:
        return "gamma_zero"

    def dirichlet_value(self, side: str, regime: int, tau_remaining_hours: float,
                        theta_bar: float, sigma_i: float) -> float:  # pragma: no cover
        raise NotImplementedError("GammaZeroY has no Dirichlet value")


@dataclass(frozen=True)
class DirichletMeanRevertingOU:
    """Closed-form no-switching OU boundary values (alternative treatment)."""

    contract: EuropeanOption
    transform: PriceTransform
    kappa: float
    y_bounds: Tuple[float, float]   # (y_min, y_max)

    def kind(self) -> str:
        return "dirichlet_ou"

    def expected_price(self, y_b: float, tau: float,
                       theta_bar: float, sigma_i: float) -> float:
        k = self.kappa
        m = theta_bar + (y_b - theta_bar) * np.exp(-k * tau)
        v = sigma_i**2 * (1.0 - np.exp(-2.0 * k * tau)) / (2.0 * k) if k > 0 \
            else sigma_i**2 * tau
        return float(self.transform.scale_P * np.exp(0.5 * v) * np.sinh(m))

    def dirichlet_value(self, side: str, regime: int, tau_remaining_hours: float,
                        theta_bar: float, sigma_i: float) -> float:
        y_b = self.y_bounds[0] if side == "lower" else self.y_bounds[1]
        ep = self.expected_price(y_b, tau_remaining_hours, theta_bar, sigma_i)
        disc = np.exp(-self.contract.r_per_hour * tau_remaining_hours)
        if self.contract.option_type == "call":
            return disc * max(ep - self.contract.strike, 0.0)
        return disc * max(self.contract.strike - ep, 0.0)
