"""FW12b §1.2 structural tests: no silent z path fallback.

The shipped forward-centered pricer refuses to run a single-covariate
call without either an explicit ``z_lagged_fn`` or the explicit opt-in
``allow_constant_transition_scenario=True``.  A caller who wants the
constant-transition limit at ``z(t-1) = 0`` for every t has to say so;
a caller who forgets a scenario path is told immediately instead of
being silently priced under a different model.

Three tests here lock the new contract:

1. missing z path (no covariate_path, no z_lagged_fn, no opt-in) --
   the pricer raises;
2. wrong-length z path -- the pricer raises;
3. explicit opt-in ``allow_constant_transition_scenario=True`` reproduces
   the constant-transition limit (a legitimate model, just not the
   silent one).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption              # noqa: E402
from pde_option_model.forward_centered import (ForwardCenteredError,  # noqa: E402
                                               ResidualGridSettings,
                                               price_forward_centered,
                                               simulate_forward_centered)
from pde_option_model.generator import TVTPCoefficients            # noqa: E402


def _contract(params, T_hours: int = 72) -> EuropeanOption:
    return EuropeanOption(
        "call", 3000.0, params.valuation_utc,
        params.valuation_utc + pd.Timedelta(hours=T_hours),
        r_annual=0.40)


def test_missing_z_path_raises_on_price(model, params, fast_grid):
    """z_lagged_fn=None, no covariate_path, no opt-in -> ForwardCenteredError.

    Guards against the FW2 sensitivity_sweep.py bug where the pricer
    silently ran the constant-transition limit at z(t-1) = 0.
    """
    with pytest.raises(ForwardCenteredError, match="silent z=0 fallback"):
        price_forward_centered(model, _contract(params), fast_grid)


def test_missing_z_path_raises_on_simulate(model, params):
    """simulate_forward_centered is guarded the same way."""
    with pytest.raises(ForwardCenteredError, match="silent z=0 fallback"):
        simulate_forward_centered(model, _contract(params), n_paths=100,
                                  dt_hours=0.5)


def test_wrong_length_z_path_raises(model, params, fast_grid):
    """A z_lagged_fn that returns an array of the wrong shape raises.

    ``_solver_covariates`` reshapes the callable output to
    ``np.asarray(...)`` -- if the caller supplies an array literal of
    the wrong length (a common bug), ``generator_path`` catches the
    shape mismatch.
    """
    T = _contract(params).tau_hours
    n_expected_steps = fast_grid.n_steps(T) + 1     # solver grid n_steps+1
    def bad_z(t_solver):
        # return a fixed-length array unrelated to t_solver.size
        return np.zeros(n_expected_steps + 7, dtype=float)
    with pytest.raises(ForwardCenteredError,
                       match="z_lagged must match the time grid"):
        price_forward_centered(model, _contract(params), fast_grid,
                               z_lagged_fn=bad_z)


def test_explicit_opt_in_reproduces_the_constant_transition_limit(
        model, params, fast_grid):
    """z_lagged_fn=None + allow_constant_transition_scenario=True is a
    LEGITIMATE model (constant transitions at z=0) and must produce a
    finite, sensible price.  It also must equal an explicit
    ``z_lagged_fn = lambda t: 0`` bit-for-bit -- the two ways of
    saying "run at z(t-1) = 0" are equivalent."""
    contract = _contract(params)
    v_opt_in = price_forward_centered(
        model, contract, fast_grid,
        allow_constant_transition_scenario=True).value
    v_explicit_zero = price_forward_centered(
        model, contract, fast_grid,
        z_lagged_fn=lambda t: np.zeros_like(t)).value
    assert v_opt_in == v_explicit_zero
    assert 0 < v_opt_in < 1e5


def test_both_covariate_path_and_opt_in_is_rejected(model, params, fast_grid):
    """Belt and braces: passing BOTH allow_constant_transition_scenario
    and a covariate_path is a caller bug (they specify different z
    paths); the pricer refuses."""
    # a covariate_path can only be built for the two-covariate model,
    # but even in the single-covariate branch the same paired flag
    # combination is meaningless.  Test via the two-covariate-model
    # branch of _solver_covariates -- exercised in test_tvtp2 tests --
    # or through the single-covariate refusal below.  Here we simply
    # confirm the two-in-one refusal exists in the single branch code
    # path via a direct call.
    from pde_option_model.forward_centered import _solver_covariates
    t = np.linspace(0.0, 72.0, 145)
    class _FakePath:
        valuation_utc = model.valuation_utc
        name = "fake"
        def covariates(self, tt):
            return np.zeros_like(tt), None
    with pytest.raises(ForwardCenteredError):
        _solver_covariates(model, t, z_lagged_fn=None,
                           covariate_path=_FakePath(),
                           allow_constant_transition_scenario=True)
