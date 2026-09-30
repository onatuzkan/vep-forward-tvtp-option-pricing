"""FW4-P -- the ramp covariate priced with FW9's ramp series and slopes.

What is pinned here:

1.  The two ramp conventions really are different objects, and FW9's is
    sourced from FW9's own code rather than re-derived by the repo's
    ``fit_ramp_scaler`` (which windows the increments before the shift and
    therefore disagrees).
2.  The intercept derivation reproduces FW4's R1 and the shipped yaml's R3,
    so the FW9 variants are built by the same rule FW4 used.
3.  The published FW4 numbers come back exactly when its own grid bounds are
    used -- the link between this work package and the accepted tvtp2 tree.
4.  The committed artefacts are internally consistent and the covariate path
    is the production climatology (FW12b), never the z = 0 limit.

The end-to-end repricing is marked ``slow``.
"""
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

from scripts.fw4p import ramp_price_impact as fw4p                   # noqa: E402
from pde_option_model.forward_centered import (                      # noqa: E402
    ForwardCenteredError, ResidualGridSettings, price_forward_centered)
from pde_option_model.generator import (                             # noqa: E402
    TVTPCoefficients, TVTP2Coefficients)
from pde_option_model.market_data import load_quotes                 # noqa: E402
from pde_option_model.params_frozen import (                         # noqa: E402
    load_frozen_parameters, load_tvtp2_parameters)
from pde_option_model.scenarios import CovariatePathBuilder          # noqa: E402
from pde_option_model.tvtp2 import (                                 # noqa: E402
    fit_ramp_scaler, load_hourly_z_history)

OUT = REPO_ROOT / "outputs" / "fw4p_ramp_price_impact"
needs_outputs = pytest.mark.skipif(
    not (OUT / "fw4p_runs.csv").exists(),
    reason="run scripts/fw4p/ramp_price_impact.py first")


@pytest.fixture(scope="module")
def params():
    return load_frozen_parameters(fw4p.PARAMS_YAML)


@pytest.fixture(scope="module")
def z_history():
    return load_hourly_z_history(fw4p.Z_HISTORY_CSV)


@pytest.fixture(scope="module")
def variants(params, z_history):
    return fw4p.build_variants(params, z_history)


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    return pd.read_csv(OUT / "fw4p_runs.csv")


# ---------------------------------------------------------------------------
# FW9's ramp
# ---------------------------------------------------------------------------
def test_fw9_ramp_scaler_matches_the_stored_fit():
    """The path scaler must be the one the slopes were estimated against."""
    s = fw4p.fw9_ramp_scaler()
    stored = fw4p.fw9_stored_scaler()
    assert s.mean == pytest.approx(stored["mu"], rel=0, abs=1e-15)
    assert s.std == pytest.approx(stored["sd"], rel=0, abs=1e-15)
    assert s.mean == pytest.approx(2.9624893381834632e-05, rel=0, abs=1e-18)
    assert s.std == pytest.approx(0.26640045488623665, rel=0, abs=1e-15)
    assert s.ddof == 1
    assert s.window_end_utc == fw4p.TRAIN_END_UTC


def test_the_two_ramp_conventions_are_genuinely_different(z_history):
    """FW9's scaler is not reproducible by the repo's windowing rule.

    ``fit_ramp_scaler`` cuts the *unshifted* increments at the window end;
    FW9 shifts first and then cuts.  The resulting units differ, which is
    exactly why the FW9 slopes cannot be dropped onto the FW4 path.
    """
    fw9 = fw4p.fw9_ramp_scaler()
    repo = fit_ramp_scaler(z_history, fw4p.TRAIN_END_UTC, "repo_rule", ddof=1)
    assert not fw9.matches(repo), (
        "the two conventions now agree; the FW4-P scaler caveat is obsolete")
    assert abs(fw9.std - repo.std) / repo.std < 1e-3, "difference is in units, not magnitude"


def test_fw9_ramp_is_wider_than_the_fw4_ramp(variants):
    fw4_sd = variants["fw4_ramp"]["scaler"].std
    fw9_sd = variants["fw9_ramp"]["scaler"].std
    assert fw4_sd == pytest.approx(0.271450116132395, rel=0, abs=1e-15)
    # A smaller divisor makes FW9's standardised ramp larger for the same dz.
    assert fw9_sd < fw4_sd
    assert fw4_sd / fw9_sd == pytest.approx(1.0190, abs=5e-4)


def test_fw9_slopes_are_the_ones_the_task_names():
    p = fw4p.fw9_fit()["params"]
    assert p["h01"] == pytest.approx(-0.0794620425383459, rel=0, abs=1e-15)
    assert p["h10"] == pytest.approx(0.38157298632138, rel=0, abs=1e-15)


def test_the_fw9_fit_is_at_the_unit_root_boundary_and_unconverged():
    """The caveat the report must carry, asserted rather than assumed."""
    f = fw4p.fw9_fit()
    assert f["params"]["phi"] > 0.999998, "fit is no longer near a unit root"
    half_life_years = math.log(2.0) / -math.log(f["params"]["phi"]) / 8766.0
    assert half_life_years > 50.0
    assert f["converged"] is False and f["grad_norm"] > 1.0


# ---------------------------------------------------------------------------
# variant wiring
# ---------------------------------------------------------------------------
def test_base_1d_w9_reproduces_fw4_R1_intercepts(variants):
    t = variants["base_1d_w9"]["tvtp"]
    assert isinstance(t, TVTPCoefficients)
    assert t.alpha01 == pytest.approx(-1.045554, abs=5e-7)
    assert t.alpha10 == pytest.approx(-1.890591, abs=5e-7)


def test_fw4_ramp_variant_is_the_shipped_yaml(variants, params):
    shipped = load_tvtp2_parameters(fw4p.TVTP2_YAML, base_params=params,
                                    base_params_path=str(fw4p.PARAMS_YAML))
    t = variants["fw4_ramp"]["tvtp"]
    for k in ("alpha01", "gamma01", "h01", "alpha10", "gamma10", "h10"):
        assert getattr(t, k) == getattr(shipped.coefficients, k)


def test_production_variant_is_untouched_production(variants, params):
    t = variants["production"]["tvtp"]
    assert isinstance(t, TVTPCoefficients)
    assert (t.alpha01, t.gamma01, t.alpha10, t.gamma10) == (
        params.alpha01, params.gamma01, params.alpha10, params.gamma10)


def test_both_fw9_variants_carry_production_gammas(variants, params):
    """Only the ramp channel and the intercepts may move."""
    for name in ("fw9_ramp", "fw9_ramp_matched"):
        t = variants[name]["tvtp"]
        assert isinstance(t, TVTP2Coefficients)
        assert t.gamma01 == params.gamma01
        assert t.gamma10 == params.gamma10
        assert t.h01 == pytest.approx(-0.0794620425383459)
        assert t.h10 == pytest.approx(0.38157298632138)
    assert variants["fw9_ramp"]["tvtp"].alpha01 == params.alpha01
    assert variants["fw9_ramp_matched"]["tvtp"].alpha01 != params.alpha01


def test_matched_intercepts_hit_the_moment_targets(variants):
    """The re-derived intercepts really do restore the mean durations."""
    for name in ("base_1d_w9", "fw9_ramp_matched"):
        roots = variants[name]["intercepts"]
        for key, duration in (("root_p01", fw4p.DURATION_NORMAL_H),
                              ("root_p10", fw4p.DURATION_STRESS_H)):
            r = roots[key]
            assert r["converged"]
            assert r["achieved_mean_p"] == pytest.approx(1.0 / duration,
                                                         abs=1e-12)


# ---------------------------------------------------------------------------
# the link to the accepted FW4 tree
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("hours", sorted(fw4p.FW4_GRID_BOUNDS))
def test_fw4_published_prices_reproduce_on_fw4_own_grid(hours):
    """R0 and R3 at K = 3000 come back exactly on FW4's published bounds.

    This is the bridge between FW4-P and the accepted tvtp2 tree: the
    machinery here is the machinery that produced those numbers.
    """
    frame = fw4p.price_all(maturities_hours=(hours,), strikes=(3000.0,),
                           grid_bounds={hours: fw4p.FW4_GRID_BOUNDS[hours]},
                           include_atm=False).set_index("variant")
    for variant, run in (("production", "R0"), ("fw4_ramp", "R3")):
        got = float(frame.loc[variant, "call_TRY_MWh"])
        assert got == pytest.approx(fw4p.FW4_PUBLISHED_CALLS[(run, hours)],
                                    abs=1e-6), f"{run} at {hours} h"


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("protected", fw4p.PROTECTED_OUTPUT_DIRS)
def test_writing_into_an_accepted_tree_is_refused(protected):
    with pytest.raises(fw4p.FW4PError):
        fw4p.assert_writable(REPO_ROOT / protected)
    with pytest.raises(fw4p.FW4PError):
        fw4p.assert_writable(REPO_ROOT / protected / "nested")


def test_the_default_outdir_is_writable(tmp_path):
    assert fw4p.assert_writable(fw4p.DEFAULT_OUTDIR) == fw4p.DEFAULT_OUTDIR
    assert fw4p.assert_writable(tmp_path) == tmp_path


def test_a_mismatched_ramp_scaler_is_refused(params, z_history, variants):
    """The unit guard that stops FW9 slopes riding on the FW4 path."""
    quotes = load_quotes(fw4p.QUOTES_CSV)
    gs = ResidualGridSettings(n_space_nodes=201, n_time_steps=48, n_std=6.0)
    contract = fw4p._contract(params, 24, 3000.0)
    builder = CovariatePathBuilder(
        z_history, fw4p.TRAIN_END_UTC, fw4p.COVARIATE_LAG_HOURS,
        ramp_scaler=variants["fw4_ramp"]["scaler"],
        history_file=str(fw4p.Z_HISTORY_CSV))
    path = fw4p.climatology_path(builder, contract, gs)
    model = fw4p.build_model(params, quotes, variants["fw9_ramp"]["tvtp"],
                             variants["fw9_ramp"]["scaler"])
    with pytest.raises(ForwardCenteredError, match="ramp standardization"):
        price_forward_centered(model, contract, gs, covariate_path=path)


def test_the_covariate_path_is_the_production_climatology(params, z_history,
                                                          variants):
    """FW12b: a real climatology path, never the constant-transition limit."""
    gs = ResidualGridSettings(n_space_nodes=fw4p.N_SPACE_NODES,
                              n_time_steps=None, n_std=fw4p.N_STD)
    contract = fw4p._contract(params, 72, 3000.0)
    builder = CovariatePathBuilder(
        z_history, fw4p.TRAIN_END_UTC, fw4p.COVARIATE_LAG_HOURS,
        ramp_scaler=variants["fw9_ramp"]["scaler"],
        history_file=str(fw4p.Z_HISTORY_CSV))
    path = fw4p.climatology_path(builder, contract, gs)
    assert path.mode == "climatology"
    assert np.ptp(path.z_master) > 0.5
    assert np.ptp(path.ramp_master) > 0.1
    assert np.all(np.isfinite(path.ramp_master))
    # The climatology train window ends well before the valuation: no look-ahead.
    assert fw4p.TRAIN_END_UTC < params.valuation_utc


# ---------------------------------------------------------------------------
# committed artefacts
# ---------------------------------------------------------------------------
@needs_outputs
def test_every_variant_is_priced_on_one_grid_per_maturity(runs):
    for hours, g in runs.groupby("maturity_h"):
        assert g["grid_x_min"].nunique() == 1, f"{hours} h"
        assert g["grid_x_max"].nunique() == 1, f"{hours} h"
        assert (g["n_space_nodes"] == fw4p.N_SPACE_NODES).all()
    assert set(runs["maturity_h"]) == set(fw4p.MATURITIES_HOURS)
    assert set(runs["variant"]) == {"production", "base_1d_w9", "fw4_ramp",
                                    "fw9_ramp", "fw9_ramp_matched"}
    assert runs.groupby(["variant", "maturity_h"]).size().eq(
        len(fw4p.STRIKE_LADDER) + 1).all()


@needs_outputs
def test_atm_rows_strike_the_forward(runs):
    atm = runs[runs["strike_kind"] == "ATM"]
    assert len(atm) == len(fw4p.MATURITIES_HOURS) * runs["variant"].nunique()
    np.testing.assert_allclose(atm["strike_TRY_MWh"], atm["F_T_TRY_MWh"],
                               rtol=0, atol=1e-5)


@needs_outputs
def test_forward_centering_holds_for_every_run(runs):
    """The ramp changes the transition law, never the centering identity."""
    np.testing.assert_allclose(runs["E_spot_T_TRY_MWh"], runs["F_T_TRY_MWh"],
                               rtol=0, atol=1e-6)


@needs_outputs
def test_the_forward_curve_is_variant_independent(runs):
    """Only the transition law differs, so F(T) must not move."""
    for hours, g in runs.groupby("maturity_h"):
        assert g["F_T_TRY_MWh"].nunique() == 1, f"F(T) moved at {hours} h"


@needs_outputs
def test_two_covariate_runs_stay_embeddable(runs):
    two_d = runs[runs["h01"] != 0.0]
    assert len(two_d) > 0
    assert (two_d["max_s"] < 1.0).all(), (
        "p01 + p10 >= 1 somewhere: the continuous-time chain does not exist")


@needs_outputs
def test_derived_tables_follow_from_the_runs(runs):
    for name, built in (("fw4p_three_way.csv", fw4p.side_by_side(runs)),
                        ("fw4p_ramp_effect.csv", fw4p.ramp_effect(runs))):
        on_disk = pd.read_csv(OUT / name)
        pd.testing.assert_frame_equal(
            on_disk, built[on_disk.columns].reset_index(drop=True),
            check_dtype=False, atol=1e-6)


@needs_outputs
def test_the_ramp_effect_stays_small(runs):
    """FW9's decomposition puts the transition channel at ~0.4 % of the
    total price difference, so a small effect is the prediction.  Both
    occupancy-controlled ramp effects must stay inside a few percent at the
    strikes where the option carries real value."""
    eff = fw4p.ramp_effect(runs)
    near = eff[(eff["occupancy_controlled"])
               & (eff["strike_TRY_MWh"] <= 3000.0)]
    assert len(near) > 0
    assert near["ramp_effect_pct"].abs().max() < 5.0, (
        near.loc[near["ramp_effect_pct"].abs().idxmax()].to_dict())


@needs_outputs
def test_fw4_variant_reproduces_its_published_ramp_effect(runs):
    """At 72 h / K = 3000 the FW4 leg must land on FW4's -1.008 %."""
    eff = fw4p.ramp_effect(runs)
    row = eff[(eff["ramp_variant"] == "fw4_ramp")
              & (eff["maturity_h"] == 72)
              & np.isclose(eff["strike_TRY_MWh"], 3000.0)]
    assert len(row) == 1
    # The grid here is the union over five variants, not FW4's union over its
    # own runs, so the agreement is close but not bit-exact.
    assert float(row["ramp_effect_pct"].iloc[0]) == pytest.approx(
        fw4p.FW4_TOTAL_RAMP_EFFECT_72H_K3000_PCT, abs=0.05)


# ---------------------------------------------------------------------------
@needs_outputs
@pytest.mark.slow
def test_committed_runs_regenerate_from_scratch(runs):
    fresh = fw4p.price_all(maturities_hours=(72,))
    on_disk = runs[runs["maturity_h"] == 72].reset_index(drop=True)
    # `fw4_run` is empty for the variants FW4 never ran; read_csv turns the
    # empty field into NaN, so normalise before comparing.
    text = ["fw4_run", "variant", "label", "strike_kind", "ramp_scaler_window"]
    for f in (fresh, on_disk):
        for col in text:
            f[col] = f[col].fillna("").astype(str)
    pd.testing.assert_frame_equal(fresh[on_disk.columns], on_disk,
                                  check_dtype=False, atol=1e-6)
