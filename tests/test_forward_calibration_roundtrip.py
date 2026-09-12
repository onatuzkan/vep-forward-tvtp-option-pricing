"""Round-trip sanity check for the forward-calibration optimiser.

Replaces the removed ``synthetic_self_test`` function from
`pde_option_model/forward_calibration.py` with a pytest that exercises
the same round-trip logic without depending on a full artefact bundle
(no ``MarkovInputs`` / ``ScenarioPath`` fixtures required).

The original test's intent (from its docstring):

    1. Generate target forwards under a KNOWN nonzero drift premium
       a_i = true_a (default (-0.004, -0.004), the magnitude relevant
       for flattening the near-unit-root forward drift).
    2. Calibrate starting from a = 0 and check the premium is recovered.

    A degenerate variant (calibrating to the unadjusted model's own
    forwards from x0 = 0) converges trivially at the start point and
    tests nothing.

We reproduce that exactly here by injecting a synthetic ``forward_fn``
into ``calibrate()``.  The forward under the adjustment is defined by a
minimal analytic proxy (a linear map of the drift shift into the
forward level), which is enough to exercise the least-squares
optimiser end-to-end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pde_option_model.forward_calibration import (ForwardQuote, calibrate)
from pde_option_model.risk_neutral import MeasureAdjustment, baseline_q1


VAL_UTC = pd.Timestamp("2025-12-31T20:00:00Z")


def _synthetic_forward_fn(a_true_0: float, a_true_1: float,
                          f0: float = 2916.0):
    """Return a ``forward_fn(adjustment, maturity_utc)`` that plays the role
    of ``model_forward_price(inputs, scenario, T, adjustment=adj, ...)``.

    Model: F(T; a) = f0 + tau_h * (w0(tau) * a0 + w1(tau) * a1), where
    the weights `(w0, w1)` change with maturity in a way that makes
    the two coefficients `a0` and `a1` identifiable from >= 2 quotes.
    This mimics the true PDE, where the regime-conditional drift shifts
    enter the forward's mean with different weights at different tau
    (because the TVTP chain's `p_stress(t)` depends on t).  Weights
    are `w0(tau) = 1 - 0.5 * (tau / 168)`, `w1(tau) = 0.5 * (tau / 168)`
    — normal-regime weight decays and stress-regime weight grows with
    tau, mirroring the mixing of a two-regime chain toward stationarity.
    """

    def fwd(adjustment: MeasureAdjustment, T: pd.Timestamp) -> float:
        a0, a1 = adjustment.drift_shift_per_hour
        tau_h = (T - VAL_UTC).total_seconds() / 3600.0
        w1 = 0.5 * tau_h / 168.0
        w0 = 1.0 - w1
        return f0 + tau_h * (w0 * a0 + w1 * a1)

    # Also generate the target quotes at the true premium.
    truth = MeasureAdjustment(spec="Q1",
                              drift_shift_per_hour=np.array([a_true_0, a_true_1]),
                              calibrated=True)
    return fwd, truth


@pytest.mark.parametrize("a_true", [
    (-0.004, -0.004),       # symmetric small negative — the paper reference
    (+0.005, -0.008),       # asymmetric mixed-sign
    (+0.010, +0.010),       # symmetric positive, larger magnitude
])
def test_calibrator_recovers_known_drift_premium(a_true):
    """From `x0 = 0`, the LS optimiser must recover a known Q1 drift shift.

    This is the round-trip test the removed ``synthetic_self_test``
    performed: inject a synthetic ``forward_fn`` under a KNOWN drift
    shift and verify the calibrator finds the same shift when started
    from zero.  Degenerate calibration to the unadjusted model's own
    forwards from ``x0 = 0`` would converge trivially at the start
    point and test nothing; here the truth is nonzero.
    """
    a_true_0, a_true_1 = a_true
    fwd, truth = _synthetic_forward_fn(a_true_0, a_true_1)

    # Build target quotes at maturities that give distinct (w0, w1) weights
    # so both a0 and a1 are identifiable.
    quotes = []
    for tau_h in (24.0, 168.0):
        T = VAL_UTC + pd.Timedelta(hours=tau_h)
        quotes.append(ForwardQuote(maturity_utc=T, price=fwd(truth, T)))

    result = calibrate(
        inputs=None, scenario=None, quotes=quotes, free=("a0", "a1"),
        lambda_reg=1e-10, forward_fn=fwd,
    )

    assert result.success, f"optimiser failed: {result.message}"
    assert abs(result.theta_hat["a0"] - a_true_0) < 1e-6
    assert abs(result.theta_hat["a1"] - a_true_1) < 1e-6
    # residuals should be at solver precision
    assert np.max(np.abs(result.residuals)) < 1e-6


def test_calibrator_refuses_empty_quotes():
    """Empty quote list should raise ValueError, not silently succeed."""
    with pytest.raises(ValueError, match="no forward quotes supplied"):
        calibrate(inputs=None, scenario=None, quotes=[], free=("a0",),
                  forward_fn=lambda adj, T: 0.0)


def test_calibrator_rejects_unknown_free_parameter():
    """Typos in `free` should surface as a clear error."""
    quotes = [ForwardQuote(maturity_utc=VAL_UTC + pd.Timedelta(hours=24),
                           price=2916.0)]
    with pytest.raises(ValueError, match="unknown free parameter"):
        calibrate(inputs=None, scenario=None, quotes=quotes,
                  free=("a_not_a_valid_name",),
                  forward_fn=lambda adj, T: 2916.0)


def test_calibrator_returns_calibrated_measure_adjustment():
    """The returned adjustment carries ``calibrated=True`` and the correct spec."""
    fwd, truth = _synthetic_forward_fn(-0.004, -0.004)
    quotes = [ForwardQuote(maturity_utc=VAL_UTC + pd.Timedelta(hours=h),
                           price=fwd(truth, VAL_UTC + pd.Timedelta(hours=h)))
              for h in (24.0, 48.0)]
    result = calibrate(inputs=None, scenario=None, quotes=quotes,
                       free=("a0", "a1"), lambda_reg=1e-10, forward_fn=fwd)
    assert result.adjustment.calibrated is True
    assert result.adjustment.spec == "Q1"
