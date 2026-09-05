"""Physical (P) to pricing (Q) measure change.

The uploaded Markov estimates are historical physical-measure (P) estimates;
the bundle itself carries the warning that transition intensities require a
change of measure before use in a Q-measure pricing PDE.

Two specifications are implemented (Janczura-style, cf. docs/methodology.md):

Q1 (baseline)  -- transition intensities unchanged, q_ij^Q = q_ij^P;
                  regime-dependent drift adjustments via any of
                    * theta_shift  delta_i : theta_i^Q = theta_i^P + delta_i
                    * drift_shift  a_i     : b_i^Q = b_i^P + a_i   [per hour]
                    * market price of risk lambda_i : b_i^Q = b_i^P - lambda_i sigma_i
                  Because kappa = -ln(phi) ~ 3.85e-4 / h is tiny, theta shifts
                  act on the drift only through kappa*delta_i; the direct
                  drift shift a_i (= kappa * delta_i) is the numerically
                  better-conditioned knob and is the one exposed to the
                  forward calibrator by default.

Q2 (extended)  -- additionally allows transition risk premia
                    q_ij^Q(t) = q_ij^P(t) * exp(eta_ij),
                  which preserves positive off-diagonal intensities and zero
                  generator row sums by construction.  Kept optional: drift
                  and transition premia are generally not jointly identifiable
                  without forward/option data.

Any output produced with `calibrated=False` must be labelled
"model-implied scenario prices", never market-calibrated option prices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np

Spec = Literal["P", "Q1", "Q2"]

UNCALIBRATED_LABEL = ("model-implied scenario prices "
                      "(pricing measure NOT calibrated to market forwards)")
CALIBRATED_LABEL = "forward-calibrated pricing-measure prices"


@dataclass(frozen=True)
class MeasureAdjustment:
    spec: Spec = "Q1"
    theta_shift: np.ndarray = field(default_factory=lambda: np.zeros(2))          # delta_i
    drift_shift_per_hour: np.ndarray = field(default_factory=lambda: np.zeros(2))  # a_i
    market_price_of_risk: np.ndarray = field(default_factory=lambda: np.zeros(2))  # lambda_i
    eta: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (eta01, eta10), Q2 only
    calibrated: bool = False

    def __post_init__(self) -> None:
        for name in ("theta_shift", "drift_shift_per_hour",
                     "market_price_of_risk", "eta"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (2,):
                raise ValueError(f"{name} must have shape (2,)")
            object.__setattr__(self, name, arr)
        if self.spec == "P":
            if (np.any(self.theta_shift) or np.any(self.drift_shift_per_hour)
                    or np.any(self.market_price_of_risk) or np.any(self.eta)):
                raise ValueError("spec 'P' does not admit risk adjustments")
        if self.spec == "Q1" and np.any(self.eta != 0.0):
            raise ValueError("transition premia eta require spec='Q2'")

    # ------------------------------------------------------------- generator
    def adjust_generator(
        self, q01: np.ndarray, q10: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply q_ij^Q = q_ij^P * exp(eta_ij); identity for P and Q1."""
        if self.spec != "Q2":
            return np.asarray(q01, float), np.asarray(q10, float)
        q01q = np.asarray(q01, float) * float(np.exp(self.eta[0]))
        q10q = np.asarray(q10, float) * float(np.exp(self.eta[1]))
        if np.any(q01q < 0) or np.any(q10q < 0):  # pragma: no cover - impossible
            raise ValueError("Q2 adjustment produced negative intensity")
        return q01q, q10q

    # ------------------------------------------------------------------ label
    def label(self) -> str:
        return CALIBRATED_LABEL if self.calibrated else UNCALIBRATED_LABEL

    def describe(self) -> str:
        return (f"spec={self.spec}, delta={self.theta_shift.tolist()}, "
                f"a_per_hour={self.drift_shift_per_hour.tolist()}, "
                f"lambda={self.market_price_of_risk.tolist()}, "
                f"eta={self.eta.tolist()}, calibrated={self.calibrated}")


def physical_measure() -> MeasureAdjustment:
    return MeasureAdjustment(spec="P")


def baseline_q1(
    theta_shift: Tuple[float, float] = (0.0, 0.0),
    drift_shift_per_hour: Tuple[float, float] = (0.0, 0.0),
    market_price_of_risk: Tuple[float, float] = (0.0, 0.0),
    calibrated: bool = False,
) -> MeasureAdjustment:
    return MeasureAdjustment(
        spec="Q1",
        theta_shift=np.asarray(theta_shift, float),
        drift_shift_per_hour=np.asarray(drift_shift_per_hour, float),
        market_price_of_risk=np.asarray(market_price_of_risk, float),
        calibrated=calibrated,
    )


def extended_q2(
    eta: Tuple[float, float],
    theta_shift: Tuple[float, float] = (0.0, 0.0),
    drift_shift_per_hour: Tuple[float, float] = (0.0, 0.0),
    calibrated: bool = False,
) -> MeasureAdjustment:
    return MeasureAdjustment(
        spec="Q2",
        theta_shift=np.asarray(theta_shift, float),
        drift_shift_per_hour=np.asarray(drift_shift_per_hour, float),
        eta=np.asarray(eta, float),
        calibrated=calibrated,
    )
