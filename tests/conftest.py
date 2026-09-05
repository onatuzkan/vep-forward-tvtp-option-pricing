"""Shared fixtures for the forward-centered test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.forward_centered import (ForwardCenteredModel,          # noqa: E402
                                               ResidualGridSettings, ResidualSpec)
from pde_option_model.forward_curve import NearTermAnchor, build_forward_curve  # noqa: E402
from pde_option_model.generator import TVTPCoefficients                       # noqa: E402
from pde_option_model.market_data import load_quotes                          # noqa: E402
from pde_option_model.params_frozen import load_frozen_parameters             # noqa: E402

QUOTES_CSV = REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv"
QUOTES_JSON = REPO_ROOT / "inputs" / "market" / "vep_2025-12-31.json"
PARAMS_YAML = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
LEGACY_REF = REPO_ROOT / "inputs" / "legacy_reference" / "legacy_model_implied_forwards.json"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def quotes():
    return load_quotes(QUOTES_CSV)


@pytest.fixture(scope="session")
def params():
    return load_frozen_parameters(PARAMS_YAML)


@pytest.fixture(scope="session")
def curve_pwc(quotes, params):
    return build_forward_curve(quotes, mode="piecewise_constant",
                               spot_price_TRY_MWh=params.spot_price_TRY_MWh)


@pytest.fixture(scope="session")
def curve_smooth(quotes, params):
    return build_forward_curve(quotes, mode="smooth_constrained",
                               spot_price_TRY_MWh=params.spot_price_TRY_MWh)


def _make_model(curve, params, **spec_kwargs) -> ForwardCenteredModel:
    return ForwardCenteredModel(
        curve=curve, spec=ResidualSpec.from_frozen(params, **spec_kwargs),
        tvtp=TVTPCoefficients(params.alpha01, params.gamma01,
                              params.alpha10, params.gamma10),
        pi_filtered=params.pi_filtered, valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh)


@pytest.fixture(scope="session")
def model(curve_smooth, params):
    return _make_model(curve_smooth, params)


@pytest.fixture(scope="session")
def model_pwc(curve_pwc, params):
    return _make_model(curve_pwc, params)


@pytest.fixture(scope="session")
def fast_grid():
    """Small grid so the PDE tests stay quick but still converged."""
    return ResidualGridSettings(n_space_nodes=801, n_time_steps=144)


@pytest.fixture
def make_model():
    return _make_model
