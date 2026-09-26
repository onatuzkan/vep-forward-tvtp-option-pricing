"""Closed-form benchmark option pricers (FW3).

Three benchmarks are exposed, all evaluated on the SAME European call/put
contracts, the SAME forward level F(T) and the SAME discount factor
`exp(-r * tau)` used by the production forward-centered PDE model.  Time is
carried in HOURS throughout, matching the pricing pipeline
(`contracts.EuropeanOption.tau_hours`, `r_per_hour = r_annual / 8760`); the
sigma inputs therefore have units of ``per sqrt(hour)`` -- either the log-
return volatility (B1), the arithmetic-return volatility (B2), or the OU
diffusion coefficient (B3).

- ``black76``          -- lognormal forward model (Black, 1976).
- ``bachelier``        -- arithmetic-normal forward model.
- ``lucia_schwartz``   -- Lucia & Schwartz (2002) one-factor arithmetic
                          model: P_t = F(t) + X_t with X_t ~ OU(kappa, 0,
                          sigma).  Closed-form via the exact terminal
                          variance ``sigma^2 * (1 - exp(-2 kappa T)) /
                          (2 kappa)``, priced with the Bachelier formula.

Implied volatilities are recovered by monotone Brent root-finding on the
respective pricing map; a round-trip is exact to solver tolerance.
Put-call parity is a hard identity in every closed form here:
``C - P == exp(-r T) (F - K)``.

Nothing in this module depends on the PDE, the residual dynamics or the
TVTP chain -- it is a self-contained analytic layer that the FW3 comparison
script drives with real repo prices / forwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

# Discount / vol arithmetic operates entirely in hourly units; ``sigma`` is
# per sqrt(hour) and ``tau_hours`` is in hours, so no annualization is
# applied here.  The caller (compare_benchmarks.py) picks the vol input.


def _validate(F: float, K: float, tau_hours: float, sigma: float,
              option_type: OptionType) -> None:
    if not (F > 0 and K > 0):
        raise ValueError("F and K must be positive (TRY/MWh)")
    if tau_hours <= 0:
        raise ValueError("tau_hours must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


def _discount(r_per_hour: float, tau_hours: float) -> float:
    return math.exp(-r_per_hour * tau_hours)


# ---------------------------------------------------------------------------
# Black-76 (lognormal on the forward)
# ---------------------------------------------------------------------------
def black76(F: float, K: float, tau_hours: float, sigma: float,
            r_per_hour: float, option_type: OptionType) -> float:
    """Black (1976) undiscounted-forward option price.

    ``sigma`` is the lognormal volatility per ``sqrt(hour)``.  With
    ``sigma = 0`` the value collapses to the discounted intrinsic
    ``exp(-r T) max((F - K), 0)`` (call) or ``exp(-r T) max((K - F), 0)``
    (put).
    """
    _validate(F, K, tau_hours, sigma, option_type)
    disc = _discount(r_per_hour, tau_hours)
    if sigma == 0.0:
        intrinsic = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
        return disc * intrinsic
    v = sigma * math.sqrt(tau_hours)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if option_type == "call":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# Bachelier (arithmetic-normal on the forward)
# ---------------------------------------------------------------------------
def bachelier(F: float, K: float, tau_hours: float, sigma: float,
              r_per_hour: float, option_type: OptionType) -> float:
    """Bachelier (arithmetic-normal) option price.

    ``sigma`` is the TRY/MWh volatility per ``sqrt(hour)`` (P_T ~
    N(F, sigma^2 * tau)).  With ``sigma = 0`` collapses to the
    discounted intrinsic.  Prices are defined for arbitrary F, K > 0 and
    remain well behaved for ``F <= K`` where lognormal skew degenerates,
    which is why Bachelier is the natural baseline for a market with
    daily price caps and near-zero-price hours.
    """
    _validate(F, K, tau_hours, sigma, option_type)
    disc = _discount(r_per_hour, tau_hours)
    if sigma == 0.0:
        intrinsic = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
        return disc * intrinsic
    v = sigma * math.sqrt(tau_hours)
    d = (F - K) / v
    if option_type == "call":
        return disc * ((F - K) * norm.cdf(d) + v * norm.pdf(d))
    return disc * ((K - F) * norm.cdf(-d) + v * norm.pdf(d))


# ---------------------------------------------------------------------------
# Lucia & Schwartz (2002) one-factor arithmetic OU
# ---------------------------------------------------------------------------
def ou_terminal_variance(sigma: float, kappa_per_hour: float,
                         tau_hours: float) -> float:
    """Var[X_T | X_0 = 0] under dX_t = -kappa X_t dt + sigma dW_t.

    Closed form: ``sigma^2 * (1 - exp(-2 kappa T)) / (2 kappa)``.  In the
    ``kappa -> 0`` limit this tends to the driftless Brownian variance
    ``sigma^2 * T``; the closed form is evaluated with an expm1 rescue
    for very small kappa T to avoid catastrophic cancellation.
    """
    if kappa_per_hour <= 0:
        raise ValueError("kappa must be positive for the OU integrated variance")
    x = 2.0 * kappa_per_hour * tau_hours
    if x < 1e-6:
        # -expm1(-x) = 1 - exp(-x), stable near zero
        return sigma * sigma * (-math.expm1(-x)) / (2.0 * kappa_per_hour)
    return sigma * sigma * (1.0 - math.exp(-x)) / (2.0 * kappa_per_hour)


def lucia_schwartz(F: float, K: float, tau_hours: float, sigma: float,
                   kappa_per_hour: float, r_per_hour: float,
                   option_type: OptionType) -> float:
    """Lucia & Schwartz (2002) single-factor arithmetic-OU price.

    Under ``P_t = F(t) + X_t``, ``dX = -kappa X dt + sigma dW``,
    ``X_0 = 0``, the terminal law of ``P_T`` is exactly
    ``N(F(T), Var_T)`` with ``Var_T = sigma^2 (1 - e^{-2 kappa T}) /
    (2 kappa)``.  The option price is therefore Bachelier with an
    equivalent flat volatility ``sigma_B = sqrt(Var_T / T)``.  This is
    the "regime-switching closed" reference the FW3 comparison uses
    (single regime, no time-varying transitions).
    """
    _validate(F, K, tau_hours, sigma, option_type)
    var_T = ou_terminal_variance(sigma, kappa_per_hour, tau_hours)
    sigma_B = math.sqrt(var_T / tau_hours)
    return bachelier(F, K, tau_hours, sigma_B, r_per_hour, option_type)


# ---------------------------------------------------------------------------
# Put-call parity
# ---------------------------------------------------------------------------
def parity_error(call_price: float, put_price: float, F: float, K: float,
                 tau_hours: float, r_per_hour: float) -> float:
    """Signed error of the identity ``C - P - exp(-r T) (F - K)``.

    Any European-option pricer -- model or benchmark -- must return
    zero (up to solver precision) for this quantity.  Bachelier and
    Black-76 return zero analytically; the PDE model returns zero up
    to ~1e-6 TRY/MWh at the shipped resolution.
    """
    return (call_price - put_price) - _discount(r_per_hour, tau_hours) * (F - K)


# ---------------------------------------------------------------------------
# Implied vol -- monotone Brent root finding on price -> sigma
# ---------------------------------------------------------------------------
def _iv_bracket(pricer, F: float, K: float, tau_hours: float,
                r_per_hour: float, option_type: OptionType,
                sigma_lo: float, sigma_hi: float, price: float,
                extra_arg=None) -> Tuple[float, float]:
    """Expand ``sigma_hi`` until the target price is bracketed.

    ``pricer(sigma)`` must be a callable returning the model price at
    the trial vol; the bracket runs from ``sigma_lo`` (near intrinsic)
    upward, doubling ``sigma_hi`` until the price is exceeded.  Fails
    with ``ValueError`` if a bracket cannot be found within 60
    doublings (sigma > ~1e18).
    """
    p_lo = pricer(sigma_lo)
    if p_lo > price + 1e-12:
        raise ValueError(
            f"target option price {price:.6g} is below the intrinsic value "
            f"{p_lo:.6g}; no positive implied vol exists")
    hi = sigma_hi
    p_hi = pricer(hi)
    tries = 0
    while p_hi < price and tries < 60:
        hi *= 2.0
        p_hi = pricer(hi)
        tries += 1
    if p_hi < price:
        raise ValueError(
            f"could not bracket implied vol for target {price:.6g}; "
            f"pricer saturated at sigma={hi:.3g} with p={p_hi:.6g}")
    return sigma_lo, hi


def implied_vol_black76(price: float, F: float, K: float, tau_hours: float,
                        r_per_hour: float, option_type: OptionType,
                        tol: float = 1e-12) -> float:
    """Black-76 implied volatility (per ``sqrt(hour)``).

    Uses Brent on ``black76(sigma) - price`` with an intrinsic-price
    lower bracket and a doubling upper bracket.  ``ValueError`` if the
    target is unreachable (below intrinsic or above the ``F N(d1)``
    limit for saturated sigma).
    """
    def pricer(s: float) -> float:
        return black76(F, K, tau_hours, s, r_per_hour, option_type)
    lo, hi = _iv_bracket(pricer, F, K, tau_hours, r_per_hour, option_type,
                         sigma_lo=1e-12, sigma_hi=1.0, price=price)
    return float(brentq(lambda s: pricer(s) - price, lo, hi, xtol=tol,
                        rtol=tol, maxiter=200))


def implied_vol_bachelier(price: float, F: float, K: float, tau_hours: float,
                          r_per_hour: float, option_type: OptionType,
                          tol: float = 1e-12) -> float:
    """Bachelier implied volatility (TRY/MWh per ``sqrt(hour)``).

    Same bracketing strategy as Black-76.  The lower bound is intrinsic
    (``sigma = 0``); the upper bound doubles until the target is
    exceeded.
    """
    def pricer(s: float) -> float:
        return bachelier(F, K, tau_hours, s, r_per_hour, option_type)
    lo, hi = _iv_bracket(pricer, F, K, tau_hours, r_per_hour, option_type,
                         sigma_lo=0.0, sigma_hi=max(F, K) * 0.02,
                         price=price)
    # brentq requires the endpoints to have opposite signs; enforce a
    # strictly positive lower endpoint since pricer(0) == intrinsic.
    p_lo = pricer(lo) - price
    if p_lo == 0.0:
        return lo
    return float(brentq(lambda s: pricer(s) - price, lo + 1e-16, hi,
                        xtol=tol, rtol=tol, maxiter=200))


# ---------------------------------------------------------------------------
# Aggregators used by the FW3 comparison script
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkQuote:
    """A single benchmark price + implied-vol view of one contract."""

    F: float
    K: float
    tau_hours: float
    r_per_hour: float
    option_type: OptionType
    call: float
    put: float
    iv_black76: float           # from the CALL price
    iv_bachelier: float         # from the CALL price
    parity_error: float


def price_black76_pair(F: float, K: float, tau_hours: float, sigma: float,
                       r_per_hour: float) -> Tuple[float, float]:
    """(call, put) under Black-76 with a shared sigma."""
    c = black76(F, K, tau_hours, sigma, r_per_hour, "call")
    p = black76(F, K, tau_hours, sigma, r_per_hour, "put")
    return c, p


def price_bachelier_pair(F: float, K: float, tau_hours: float, sigma: float,
                         r_per_hour: float) -> Tuple[float, float]:
    """(call, put) under Bachelier with a shared sigma."""
    c = bachelier(F, K, tau_hours, sigma, r_per_hour, "call")
    p = bachelier(F, K, tau_hours, sigma, r_per_hour, "put")
    return c, p


def price_lucia_schwartz_pair(F: float, K: float, tau_hours: float,
                              sigma: float, kappa_per_hour: float,
                              r_per_hour: float) -> Tuple[float, float]:
    """(call, put) under Lucia-Schwartz with a shared (sigma, kappa)."""
    c = lucia_schwartz(F, K, tau_hours, sigma, kappa_per_hour, r_per_hour,
                       "call")
    p = lucia_schwartz(F, K, tau_hours, sigma, kappa_per_hour, r_per_hour,
                       "put")
    return c, p
