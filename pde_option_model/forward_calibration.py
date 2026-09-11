"""Calibration of pricing-measure parameters to observed electricity forwards.

Objective (weighted least squares with Tikhonov regularization):

    min_{Theta_Q}  sum_m w_m [ F_model(0, T_m; Theta_Q) - F_market(0, T_m) ]^2
                   + lambda_reg * || Theta_Q ||^2

Theta_Q is any subset of
    a_i      regime drift shifts per hour   (Q1; preferred -- kappa is tiny,
             so theta shifts delta_i = a_i / kappa are badly conditioned)
    delta_i  long-run-mean shifts           (Q1)
    eta_ij   transition log-premia          (Q2 only)

`calibrate` requires a list of ForwardQuote objects built from observed market
quotes (no model-generated / synthetic quotes are used anywhere).  Forward prices
are computed with the SAME coupled PDE as the option prices (linear payoff,
zero rate), so calibration and pricing are internally consistent.

Until real forward quotes are supplied, every price produced by this package
must be labelled "model-implied scenario prices".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .markov_adapter import MarkovInputs
from .pricing import GridSettings, model_forward_price
from .risk_neutral import MeasureAdjustment, baseline_q1
from .scenarios import ScenarioPath

logger = logging.getLogger(__name__)

FREE_PARAMS = ("a0", "a1", "delta0", "delta1", "eta01", "eta10")


@dataclass(frozen=True)
class ForwardQuote:
    maturity_utc: pd.Timestamp
    price: float
    weight: float = 1.0


@dataclass
class CalibrationResult:
    adjustment: MeasureAdjustment
    theta_hat: dict[str, float]
    residuals: np.ndarray
    model_forwards: np.ndarray
    market_forwards: np.ndarray
    cost: float
    success: bool
    message: str


def _vector_to_adjustment(x: np.ndarray, free: Sequence[str]) -> MeasureAdjustment:
    vals = dict(zip(free, x))
    a = np.array([vals.get("a0", 0.0), vals.get("a1", 0.0)])
    d = np.array([vals.get("delta0", 0.0), vals.get("delta1", 0.0)])
    eta = np.array([vals.get("eta01", 0.0), vals.get("eta10", 0.0)])
    spec = "Q2" if np.any(eta != 0.0) or any(p.startswith("eta") for p in free) else "Q1"
    return MeasureAdjustment(spec=spec, theta_shift=d, drift_shift_per_hour=a,
                             eta=eta, calibrated=True)


def calibrate(
    inputs: MarkovInputs,
    scenario: ScenarioPath,
    quotes: Sequence[ForwardQuote],
    free: Sequence[str] = ("a0", "a1"),
    lambda_reg: float = 1e-4,
    grid_settings: Optional[GridSettings] = None,
    x0: Optional[np.ndarray] = None,
    forward_fn: Optional[Callable[[MeasureAdjustment, pd.Timestamp], float]] = None,
) -> CalibrationResult:
    """Least-squares calibration of the selected pricing-measure parameters.

    `forward_fn` can inject an alternative forward pricer (e.g. Monte Carlo);
    by default the coupled-PDE forward is used.
    """
    if not quotes:
        raise ValueError(
            "no forward quotes supplied -- calibration interface is ready but "
            "cannot run; outputs remain 'model-implied scenario prices'")
    for p in free:
        if p not in FREE_PARAMS:
            raise ValueError(f"unknown free parameter {p!r}; choose from {FREE_PARAMS}")
    if any(p.startswith("eta") for p in free) and any(p.startswith("delta") or p.startswith("a") for p in free):
        logger.warning("drift and transition premia are being calibrated jointly; "
                       "identification typically requires option data as well")

    gs = grid_settings or GridSettings()
    mkt = np.array([q.price for q in quotes], dtype=float)
    w = np.sqrt(np.array([q.weight for q in quotes], dtype=float))

    def fwd(adj: MeasureAdjustment, T: pd.Timestamp) -> float:
        if forward_fn is not None:
            return forward_fn(adj, T)
        return model_forward_price(inputs, scenario, T, adjustment=adj,
                                   grid_settings=gs)

    def resid(x: np.ndarray) -> np.ndarray:
        adj = _vector_to_adjustment(x, free)
        model = np.array([fwd(adj, q.maturity_utc) for q in quotes])
        r = w * (model - mkt)
        reg = np.sqrt(lambda_reg) * x
        return np.concatenate([r, reg])

    x_init = np.zeros(len(free)) if x0 is None else np.asarray(x0, float)
    sol = least_squares(resid, x_init, method="lm" if len(free) <= len(quotes)
                        else "trf", xtol=1e-10, ftol=1e-10)
    adj = _vector_to_adjustment(sol.x, free)
    model = np.array([fwd(adj, q.maturity_utc) for q in quotes])
    logger.info("calibration %s: cost=%.6g, theta=%s",
                "converged" if sol.success else "FAILED", sol.cost,
                dict(zip(free, np.round(sol.x, 6))))
    return CalibrationResult(
        adjustment=adj,
        theta_hat=dict(zip(free, sol.x.tolist())),
        residuals=model - mkt,
        model_forwards=model,
        market_forwards=mkt,
        cost=float(sol.cost),
        success=bool(sol.success),
        message=str(sol.message),
    )
