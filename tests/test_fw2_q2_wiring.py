"""FW2 §3+§6 tests: Q2 eta wiring, centering under Q2, Q1 quadratic
suppression, generator validity, and price monotonicity in eta."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption               # noqa: E402
from pde_option_model.forward_centered import (                     # noqa: E402
    ForwardCenteredError, ResidualGridSettings, ResidualSpec,
    _apply_eta_to_generator, price_forward_centered)
from pde_option_model.forward_curve import build_forward_curve       # noqa: E402
from pde_option_model.generator import TVTPCoefficients              # noqa: E402
from pde_option_model.market_data import load_quotes                 # noqa: E402
from pde_option_model.params_frozen import load_frozen_parameters    # noqa: E402
from pde_option_model.forward_centered import ForwardCenteredModel   # noqa: E402


PARAMS_YAML = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
QUOTES_CSV = REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv"


@pytest.fixture(scope="module")
def _model_and_contract():
    params = load_frozen_parameters(PARAMS_YAML)
    quotes = load_quotes(QUOTES_CSV)
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    model = ForwardCenteredModel(
        curve=curve, spec=ResidualSpec.from_frozen(params),
        tvtp=TVTPCoefficients(params.alpha01, params.gamma01,
                              params.alpha10, params.gamma10),
        pi_filtered=params.pi_filtered,
        valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    contract = EuropeanOption("call", 3000.0, params.valuation_utc,
                              params.valuation_utc + pd.Timedelta(hours=72),
                              r_annual=0.40)
    return model, contract


# ---------------------------------------------------------------------------
# eta=None reproduces the shipped run bit-for-bit
# ---------------------------------------------------------------------------
def test_none_and_zero_eta_reproduce_baseline_bit_for_bit(_model_and_contract):
    model, contract = _model_and_contract
    gs = ResidualGridSettings()
    base = price_forward_centered(model, contract, grid_settings=gs)
    none_ = price_forward_centered(model, contract, grid_settings=gs,
                                   eta_ij=None)
    zero_ = price_forward_centered(model, contract, grid_settings=gs,
                                   eta_ij=(0.0, 0.0))
    assert base.value == none_.value
    assert base.value == zero_.value
    assert base.centering_at_expiry == zero_.centering_at_expiry


# ---------------------------------------------------------------------------
# Centering identity holds under Q2 (moment ODE and PDE share q^Q)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eta", [(0.3, 0.3), (-0.3, 0.3), (0.5, -0.5),
                                 (-0.5, -0.5), (0.9, -0.9)])
def test_centering_identity_under_q2(_model_and_contract, eta):
    model, contract = _model_and_contract
    gs = ResidualGridSettings()
    r = price_forward_centered(model, contract, grid_settings=gs, eta_ij=eta)
    assert abs(r.expected_spot_at_expiry - r.forward_at_expiry) < 1e-6


# ---------------------------------------------------------------------------
# Generator validity: eta-shifted generator has non-negative off-diagonals
# and zero row sums by construction (multiplicative form).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eta", [(0.0, 0.0), (0.5, 0.5), (-1.0, 1.0),
                                 (2.0, -2.0), (5.0, 5.0)])
def test_eta_shifted_generator_stays_valid(eta):
    q01_p = np.array([0.01, 0.02, 0.03], dtype=float)
    q10_p = np.array([0.05, 0.04, 0.06], dtype=float)
    q01q, q10q = _apply_eta_to_generator(q01_p, q10_p, eta)
    # off-diagonals must be non-negative
    assert (q01q >= 0).all() and (q10q >= 0).all()
    # multiplicative form check
    np.testing.assert_allclose(q01q, q01_p * math.exp(eta[0]), rtol=1e-14)
    np.testing.assert_allclose(q10q, q10_p * math.exp(eta[1]), rtol=1e-14)


def test_apply_eta_rejects_wrong_shape():
    with pytest.raises(ForwardCenteredError):
        _apply_eta_to_generator(np.array([0.01]), np.array([0.02]), (0.0,))


# ---------------------------------------------------------------------------
# Q1 quadratic suppression: log-log slope near 2 across drift-shift decades
# ---------------------------------------------------------------------------
def test_q1_variance_channel_scales_quadratically(_model_and_contract):
    """Symmetric drift shift a_i moves the terminal RESIDUAL VARIANCE
    (from the moment ODE, not the PDE call value) by O(a^2).  The
    moment ODE is analytic RK4 so this is a clean log-log slope test;
    the PDE call-value channel is O(a^2 * dVar/dK * ...) and is
    numerically noise-dominated at the grid the option pipeline uses
    (see risk_premium_sensitivity.README.md).
    """
    from pde_option_model.forward_centered import residual_moments
    model, _contract = _model_and_contract
    T_hours = 72.0
    t = np.linspace(0.0, T_hours, int(T_hours * 2) + 1)
    z_lag, _ = model.resolve_covariates(t, z_lagged=np.zeros_like(t))
    q01, q10 = model.generator_path(t, z_lag)
    sig_path = model.spec.sigma_price(model.forward_at(t))
    # Asymmetric a_stress only: symmetric shifts have zero effect at
    # this order (analytic identity for two-regime OU with equal m_i)
    # so testing quadratic suppression needs an asymmetry between
    # regimes -- exactly the pattern a market-price-of-risk would take.
    a_values = np.array([1.0, 2.0, 5.0, 10.0, 25.0, 50.0])
    base = residual_moments(model.spec, t, q01, q10, model.pi_filtered,
                            x0=model.x0, sigma_price_path=sig_path)
    dvars = []
    for a in a_values:
        shifted_spec = ResidualSpec(
            kappa_per_hour=model.spec.kappa_per_hour,
            sigma_y=model.spec.sigma_y, scale_P=model.spec.scale_P,
            regime_means=model.spec.regime_means,
            mode=model.spec.mode, x0_mode=model.spec.x0_mode,
            sigma_multipliers=model.spec.sigma_multipliers,
            drift_shift_per_hour=np.array([0.0, a]),  # asymmetric
        )
        m = residual_moments(shifted_spec, t, q01, q10, model.pi_filtered,
                             x0=model.x0, sigma_price_path=sig_path)
        dvars.append(abs(float(m.variance[-1] - base.variance[-1])))
    # log-log slope between smallest and largest a
    slope = (math.log(dvars[-1] / dvars[0])
             / math.log(a_values[-1] / a_values[0]))
    assert 1.9 < slope < 2.1, (
        f"Q1 log-log variance slope {slope:.4f} not near 2 "
        f"(dvars={dvars})")


# ---------------------------------------------------------------------------
# Q2 has a MATERIAL, sign-monotonic effect
# ---------------------------------------------------------------------------
def test_q2_price_is_monotonic_in_eta01_holding_eta10(_model_and_contract):
    """Raising eta_01 (transitions into stress become faster) shifts
    weight toward the high-vol regime -> option value increases."""
    model, contract = _model_and_contract
    gs = ResidualGridSettings()
    values = []
    for e in (-0.3, 0.0, 0.3, 0.6):
        r = price_forward_centered(model, contract, grid_settings=gs,
                                   eta_ij=(e, 0.0))
        values.append(r.value)
    for a, b in zip(values, values[1:]):
        assert b > a, f"eta_01 sweep not monotone: {values}"


def test_apply_eta_none_is_pass_through():
    q01 = np.array([0.01, 0.02])
    q10 = np.array([0.03, 0.04])
    q01q, q10q = _apply_eta_to_generator(q01, q10, None)
    np.testing.assert_array_equal(q01q, q01)
    np.testing.assert_array_equal(q10q, q10)
