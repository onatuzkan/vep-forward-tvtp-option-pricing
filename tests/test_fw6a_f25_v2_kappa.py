"""FW6a -- the F2.5 pooled-vs-M9 comparison re-run at the production kappa.

Three things are pinned here:

1.  The pooled-baseline arithmetic and the variant wiring (pure, fast).
2.  The climatology ``z(t-1)`` path used by the FW6a runner is the one
    ``run_pde.py price`` builds -- the FW12b rule.  A drift here would
    silently reprice everything.
3.  The committed artefacts under ``outputs/f25_v2_kappa/`` reproduce
    (a) the accepted v2-era CSV in ``outputs/market_calibration_final/``
    and (b) the numbers actually printed in the F2.5 markdown, which
    were produced at the pre-v2 kappa.  (a) and (b) disagree with each
    other -- that disagreement is the defect FW6a documents -- so both
    are checked against their own source.

The heavy end-to-end regeneration is marked ``slow``; everything else
runs without touching the PDE.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fw6a import f25_v2_kappa as fw6a                      # noqa: E402
from pde_option_model.contracts import EuropeanOption              # noqa: E402
from pde_option_model.forward_centered import ResidualGridSettings  # noqa: E402
from pde_option_model.params_frozen import (                       # noqa: E402
    FrozenParameterError, load_frozen_parameters)

OUT = REPO_ROOT / "outputs" / "f25_v2_kappa"
ACCEPTED = (REPO_ROOT / "outputs" / "market_calibration_final"
            / "model_comparison_pooled_vs_M9.csv")
ACCEPTED_MD = ACCEPTED.with_suffix(".md")

VARIANTS = tuple(fw6a.VARIANT_LABELS)
needs_outputs = pytest.mark.skipif(
    not (OUT / "f25_v2_kappa.csv").exists(),
    reason="run scripts/fw6a/f25_v2_kappa.py first")


@pytest.fixture(scope="module")
def committed_new() -> pd.DataFrame:
    return pd.read_csv(OUT / "f25_v2_kappa.csv")


@pytest.fixture(scope="module")
def committed_old() -> pd.DataFrame:
    return pd.read_csv(OUT / "f25_pre_v2_reproduction.csv")


# ---------------------------------------------------------------------------
# pooled-baseline arithmetic
# ---------------------------------------------------------------------------
def test_m0_estimates_match_the_shipped_bundle():
    m0 = fw6a.load_m0_estimates()
    assert m0["sigma0"] == pytest.approx(0.006122693643334371, rel=0, abs=1e-15)
    assert m0["sigma1"] == pytest.approx(0.16425519044118927, rel=0, abs=1e-15)
    assert m0["phi"] == pytest.approx(0.9991806029135467, rel=0, abs=1e-15)
    # F2.5 reports M0's kappa as 8.20e-4 /h, half-life 846 h.
    assert m0["kappa_per_hour"] == pytest.approx(8.1973e-4, rel=1e-4)
    assert math.log(2.0) / m0["kappa_per_hour"] == pytest.approx(846.0, abs=1.0)


def test_pooled_sigma_reproduces_the_f25_value():
    """F2.5: pooled sigma = sqrt(0.325 s0^2 + 0.675 s1^2) = 0.13495."""
    params = load_frozen_parameters(fw6a.PARAMS_YAML)
    pooled = fw6a.pooled_sigma(fw6a.load_m0_estimates(), params.m9_stationary_pi)
    assert pooled == pytest.approx(0.13495, abs=5e-6)
    # It must sit between the two M0 volatilities and, because the stress
    # weight dominates, close to the larger one.
    assert 0.006122 < pooled < 0.164256


def test_pooled_sigma_rejects_a_non_simplex_weight():
    m0 = fw6a.load_m0_estimates()
    with pytest.raises(fw6a.FW6aError):
        fw6a.pooled_sigma(m0, [0.4, 0.4])
    with pytest.raises(fw6a.FW6aError):
        fw6a.pooled_sigma(m0, [1.2, -0.2])


def test_pooled_pair_keeps_regime_one_the_stress_state():
    pair = fw6a.pooled_sigma_pair(0.13495)
    assert pair[1] > pair[0]
    assert pair.mean() == pytest.approx(0.13495, rel=1e-12)
    # The split is the cosmetic 0.01 % of F2.5, not an economic difference.
    assert (pair[1] - pair[0]) / pair.mean() == pytest.approx(2e-4, rel=1e-9)


# ---------------------------------------------------------------------------
# variant wiring
# ---------------------------------------------------------------------------
def test_variant_parameters_set_the_intended_kappa_and_sigma():
    base = load_frozen_parameters(fw6a.PARAMS_YAML)
    m0 = fw6a.load_m0_estimates()
    k9 = float(base.kappa_per_hour)
    pooled = fw6a.pooled_sigma_pair(fw6a.pooled_sigma(m0, base.m9_stationary_pi))

    prod = fw6a.variant_parameters(base, m0, "M9_prod", k9)
    assert prod.kappa_per_hour == pytest.approx(k9)
    np.testing.assert_allclose(prod.sigma_y, base.sigma_y)

    p_m0 = fw6a.variant_parameters(base, m0, "pooled_M0_kappa", k9)
    assert p_m0.kappa_per_hour == pytest.approx(m0["kappa_per_hour"])
    np.testing.assert_allclose(p_m0.sigma_y, pooled)

    p_m9 = fw6a.variant_parameters(base, m0, "pooled_M9_kappa", k9)
    assert p_m9.kappa_per_hour == pytest.approx(k9)
    np.testing.assert_allclose(p_m9.sigma_y, pooled)

    # Everything that is NOT the residual spec stays production.
    for v in (prod, p_m0, p_m9):
        np.testing.assert_allclose(v.pi_filtered, base.pi_filtered)
        assert (v.alpha01, v.gamma01, v.alpha10, v.gamma10) == (
            base.alpha01, base.gamma01, base.alpha10, base.gamma10)
        assert v.scale_P == base.scale_P
        assert v.valuation_utc == base.valuation_utc
        # phi and kappa stay mutually consistent, so no yaml warning fires.
        assert -math.log(v.phi) == pytest.approx(v.kappa_per_hour, rel=1e-9)


def test_variant_parameters_rejects_an_unknown_variant():
    base = load_frozen_parameters(fw6a.PARAMS_YAML)
    with pytest.raises(fw6a.FW6aError):
        fw6a.variant_parameters(base, fw6a.load_m0_estimates(), "M0_as_fitted",
                                float(base.kappa_per_hour))


def test_equal_pooled_sigmas_would_break_the_stress_invariant():
    """Why F2.5 needed the +/- 0.01 % split at all."""
    base = load_frozen_parameters(fw6a.PARAMS_YAML)
    from dataclasses import replace
    with pytest.raises(FrozenParameterError):
        replace(base, sigma_y=np.array([0.13495, 0.13495]))


# ---------------------------------------------------------------------------
# the covariate path is the production one (FW12b)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hours", [24, 336])
def test_climatology_path_matches_the_production_cli(hours):
    """FW6a's z(t-1) must equal the path `run_pde.py price` builds."""
    import run_pde

    cfg = run_pde._load_config("config/forward_centered_config.yaml")
    args = argparse.Namespace(scenario_history=None, scenario_mode=None,
                              scenario_offset=None, scenario_custom_csv=None)
    base = load_frozen_parameters(fw6a.PARAMS_YAML)
    contract = EuropeanOption(
        option_type=fw6a.OPTION_TYPE, strike=fw6a.STRIKE_TRY_MWH,
        valuation_utc=base.valuation_utc,
        maturity_utc=base.valuation_utc + pd.Timedelta(hours=hours),
        r_annual=fw6a.R_ANNUAL)
    gs = ResidualGridSettings(n_space_nodes=fw6a.N_SPACE_NODES,
                              n_time_steps=None, n_std=fw6a.N_STD)

    _, cli_fn = run_pde._build_tvtp_scenario(contract, gs, cfg, args)
    fw_fn, path = fw6a.climatology_z_lagged_fn(contract, gs,
                                               fw6a.load_z_history())
    t = np.linspace(0.0, float(contract.tau_hours), 4 * hours + 1)
    np.testing.assert_allclose(fw_fn(t), cli_fn(t), rtol=0, atol=0)
    # A real climatology path, never the z = 0 constant-transition limit.
    assert np.ptp(path.z_lagged) > 0.5


def test_config_settings_match_the_hardcoded_production_values():
    """The FW6a constants are the production config, not a private copy."""
    import run_pde

    cfg = run_pde._load_config("config/forward_centered_config.yaml")
    assert run_pde._get(cfg, "grid.n_space_nodes") == fw6a.N_SPACE_NODES
    assert run_pde._get(cfg, "grid.n_std") == fw6a.N_STD
    assert run_pde._get(cfg, "grid.n_time_steps") is None
    assert run_pde._get(cfg, "scenario.mode") == fw6a.SCENARIO_MODE
    assert run_pde._get(cfg, "scenario.covariate_lag_hours") == fw6a.COVARIATE_LAG_HOURS
    assert run_pde._get(cfg, "market.curve_mode") == fw6a.CURVE_MODE
    assert run_pde._get(cfg, "market.january_anchor_mode") == fw6a.JANUARY_ANCHOR_MODE
    assert run_pde._get(cfg, "contract.strike") == fw6a.STRIKE_TRY_MWH
    assert run_pde._get(cfg, "contract.r_annual") == fw6a.R_ANNUAL
    assert run_pde._get(cfg, "contract.option_type") == fw6a.OPTION_TYPE


# ---------------------------------------------------------------------------
# the published F2.5 markdown (pre-v2 era)
# ---------------------------------------------------------------------------
def _markdown_tables(text: str):
    """Every pipe table in the F2.5 markdown, as lists of cleaned cells."""
    tables, current = [], None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue                        # the |---| rule row
            if current is None:
                current = []
                tables.append(current)
            current.append(cells)
        else:
            current = None
    return tables


def _num(cell: str) -> float:
    s = (cell.replace("−", "-").replace("*", "").replace("`", "")
         .replace("%", "").replace("h", "").replace(",", "").strip())
    return float(s)


def _published_f25_tables():
    """(call table, residual-sd table) keyed by maturity, from the F2.5 md."""
    tables = _markdown_tables(ACCEPTED_MD.read_text(encoding="utf-8"))
    calls, sds = {}, {}
    for tbl in tables:
        header = [c.replace("`", "").replace("*", "").strip() for c in tbl[0]]
        if header[:1] != ["maturity"]:
            continue
        if header[1:4] != ["pooled_M0_kappa", "pooled_M9_kappa", "M9_prod"]:
            continue
        target = calls if len(header) > 4 else sds
        for row in tbl[1:]:
            target[int(_num(row[0]))] = {v: _num(row[1 + i])
                                         for i, v in enumerate(VARIANTS)}
    return calls, sds


def test_the_f25_markdown_is_stale_relative_to_its_own_csv():
    """The premise of FW6a, asserted rather than assumed.

    The accepted CSV was regenerated by the v2 kappa refit; the markdown
    beside it was not.  The kappa-invariant `pooled_M0_kappa` rows still
    agree, which is what localises the staleness to the M9-kappa rows.
    """
    calls, _ = _published_f25_tables()
    csv = pd.read_csv(ACCEPTED)
    csv["variant"] = csv["model"].str.split(" ").str[0]
    got = csv.pivot(index="maturity_h", columns="variant", values="call_TRY_MWh")
    assert set(calls) == set(got.index)
    for h in sorted(calls):
        assert calls[h]["pooled_M0_kappa"] == pytest.approx(
            got.loc[h, "pooled_M0_kappa"], abs=5e-3), (
            f"{h} h: the kappa-invariant control should still agree")
        for v in ("pooled_M9_kappa", "M9_prod"):
            assert abs(calls[h][v] - got.loc[h, v]) > 1.0, (
                f"{h} h / {v}: markdown and CSV agree, so the premise of "
                "FW6a no longer holds and this work package is obsolete")


@needs_outputs
def test_pre_v2_reproduction_matches_the_published_markdown(committed_old):
    """FW6a's pre-v2 run must land on the numbers F2.5 actually prints."""
    calls, sds = _published_f25_tables()
    got = committed_old.set_index(["maturity_h", "variant"])
    for h, per_variant in calls.items():
        for variant, published in per_variant.items():
            assert got.loc[(h, variant), "call_TRY_MWh"] == pytest.approx(
                published, abs=5e-3), f"call mismatch at {h} h / {variant}"
    for h, per_variant in sds.items():
        for variant, published in per_variant.items():
            assert got.loc[(h, variant), "residual_sd_T_TRY_MWh"] == pytest.approx(
                published, abs=5e-2), f"sd mismatch at {h} h / {variant}"
    m9_rows = committed_old["variant"] != "pooled_M0_kappa"
    np.testing.assert_allclose(committed_old.loc[m9_rows, "kappa_per_hour"],
                               fw6a.PRE_V2_KAPPA_PER_HOUR, rtol=1e-12)


# ---------------------------------------------------------------------------
# the committed v2 artefacts
# ---------------------------------------------------------------------------
@needs_outputs
def test_v2_run_matches_the_accepted_csv(committed_new):
    """The accepted CSV already carries v2 numbers; FW6a must land on them."""
    ref = pd.read_csv(ACCEPTED)
    ref["variant"] = ref["model"].str.split(" ").str[0]
    ref = ref.set_index(["maturity_h", "variant"])
    got = committed_new.set_index(["maturity_h", "variant"])
    assert set(got.index) == set(ref.index)
    for col in ("F_T_TRY_MWh", "residual_sd_T_TRY_MWh", "V_regime0_TRY_MWh",
                "V_regime1_TRY_MWh", "call_TRY_MWh"):
        np.testing.assert_allclose(got[col].loc[ref.index], ref[col],
                                   rtol=0, atol=1e-4, err_msg=col)


@needs_outputs
def test_v2_run_uses_the_production_kappa_and_grid(committed_new):
    prod_kappa = float(load_frozen_parameters(fw6a.PARAMS_YAML).kappa_per_hour)
    assert prod_kappa == pytest.approx(0.078394)
    m9_rows = committed_new["variant"] != "pooled_M0_kappa"
    np.testing.assert_allclose(committed_new.loc[m9_rows, "kappa_per_hour"],
                               prod_kappa, rtol=1e-12)
    assert (committed_new["n_space_nodes"] == 1201).all()
    # n_time_steps is the production rule max(96, 2 per hour), not a constant.
    expected = committed_new["maturity_h"].map(lambda h: max(96, 2 * int(h)))
    assert (committed_new["n_time_steps"] == expected).all()


@needs_outputs
def test_pooled_M0_variant_is_invariant_across_the_two_eras(committed_old,
                                                            committed_new):
    """The control: M0's own kappa is untouched by the v2 refit."""
    cols = ["maturity_h", "F_T_TRY_MWh", "residual_sd_T_TRY_MWh",
            "call_TRY_MWh", "kappa_per_hour"]
    a = committed_old[committed_old["variant"] == "pooled_M0_kappa"][cols]
    b = committed_new[committed_new["variant"] == "pooled_M0_kappa"][cols]
    pd.testing.assert_frame_equal(a.reset_index(drop=True),
                                  b.reset_index(drop=True))


@needs_outputs
@pytest.mark.parametrize("name", ["f25_v2_kappa.csv",
                                  "f25_pre_v2_reproduction.csv"])
def test_forward_centering_identity_survives_the_kappa_change(name):
    """F2.5 caveat (d): E^Q[P_T] = F(T) in every variant and every era."""
    frame = pd.read_csv(OUT / name)
    np.testing.assert_allclose(frame["E_spot_T_TRY_MWh"], frame["F_T_TRY_MWh"],
                               rtol=0, atol=1e-6)
    assert frame["centering_abs_err_TRY_MWh"].max() < 1e-6


@needs_outputs
def test_derived_tables_are_consistent_with_the_two_runs(committed_old,
                                                         committed_new):
    """The side-by-side and gap CSVs are derivable from the two raw runs."""
    for name, built in (
            ("f25_old_vs_new.csv",
             fw6a.side_by_side(committed_old, committed_new)),
            ("f25_gap_v2.csv", fw6a.gap_table(committed_new)),
            ("f25_gap_pre_v2.csv", fw6a.gap_table(committed_old))):
        on_disk = pd.read_csv(OUT / name)
        pd.testing.assert_frame_equal(
            on_disk, built[on_disk.columns].reset_index(drop=True),
            check_dtype=False, atol=1e-6)


@needs_outputs
def test_the_headline_gap_flattens_under_the_v2_kappa(committed_old,
                                                      committed_new):
    """Pre-v2 the gap narrowed with horizon; at the v2 kappa it is flat.

    F2.5 reads that term structure as the signature of regime-conditioning.
    Both sides reach their stationary variance inside 24 h at the
    production kappa, so the shape it rests on disappears.
    """
    old, new = fw6a.gap_table(committed_old), fw6a.gap_table(committed_new)
    col = "pct_vs_pooled_M9_kappa"
    assert np.ptp(old[col]) > 3.0, "pre-v2 gap should vary with horizon"
    assert np.ptp(new[col]) < 0.5, "v2 gap should be flat in maturity"
    assert (new[col] < old[col]).all(), "v2 discount should be deeper"
    # The pooled_M0_kappa column keeps a kappa ~96x slower than production,
    # so it retains a strong maturity slope for that reason alone.
    assert np.ptp(new["pct_vs_pooled_M0_kappa"]) > 10.0


@needs_outputs
def test_the_flat_gap_is_a_volatility_level_gap(committed_new):
    """Why F2.5's regime-conditioning reading is withdrawn.

    The pooled baseline draws its volatility from M0's fitted sigmas while
    M9_prod runs on M9's.  Once the variance saturates, the residual-sd
    ratio at expiry is just the ratio of the two occupancy-weighted
    volatilities -- so the gap measures an M0-vs-M9 level disagreement,
    not the value of holding a filtered regime belief.
    """
    base = load_frozen_parameters(fw6a.PARAMS_YAML)
    pooled = fw6a.pooled_sigma(fw6a.load_m0_estimates(), base.m9_stationary_pi)
    effective = fw6a.effective_sigma(base)
    assert effective == pytest.approx(0.075923, abs=5e-6)
    ratio = effective / pooled
    assert ratio == pytest.approx(0.5626, abs=5e-4)

    sd = fw6a.gap_table(committed_new)["sd_ratio_vs_pooled_M9_kappa"]
    # The realised sd ratio sits within a few percent of the sigma ratio at
    # every maturity; the residual slack is the regime-mixture shape.
    assert np.allclose(sd, ratio, rtol=0.07), (
        f"sd ratio {sd.tolist()} should track the sigma ratio {ratio:.4f}")
    # Whereas the pure mixture effect at EQUAL variance is an order of
    # magnitude smaller -- FW9 round f, 183.4 vs 166.75 at the 72 h ATM.
    assert abs(100.0 * (166.75 / 183.4 - 1.0)) < 12.0


def test_effective_sigma_reuses_the_frozen_mixture_variance():
    base = load_frozen_parameters(fw6a.PARAMS_YAML)
    expected = math.sqrt(base.mixture_variance_rate(base.m9_stationary_pi))
    assert fw6a.effective_sigma(base) == pytest.approx(expected, rel=1e-15)
    # It must be bounded by M9's own two volatilities.
    assert base.sigma_y[0] < fw6a.effective_sigma(base) < base.sigma_y[1]


@needs_outputs
def test_the_kappa_channel_inverts_and_stops_being_secondary(committed_old,
                                                             committed_new):
    """F2.5 point 4: the kappa swap alone raises the call by 1-14 %.

    That statement reproduces exactly at the pre-v2 kappa, which is a
    second independent check on the pre-v2 leg.  At the production kappa
    the same swap is large and negative, so the mean-reversion timescale
    is no longer the secondary channel.
    """
    old = fw6a.gap_table(committed_old)["kappa_effect_pct"]
    new = fw6a.gap_table(committed_new)["kappa_effect_pct"]
    assert 1.0 <= old.min() and old.max() <= 14.5, "F2.5 reports 1-14 %"
    assert (new < -50.0).all() and new.min() > -90.0
    # ...and it now dominates the regime-conditioning channel (~53 %).
    assert new.abs().max() > 80.0


# ---------------------------------------------------------------------------
# output guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("protected", fw6a.PROTECTED_OUTPUT_DIRS)
def test_writing_into_an_accepted_tree_is_refused(protected):
    with pytest.raises(fw6a.FW6aError):
        fw6a.assert_writable(REPO_ROOT / protected)
    with pytest.raises(fw6a.FW6aError):
        fw6a.assert_writable(REPO_ROOT / protected / "nested")


def test_the_default_outdir_is_writable(tmp_path):
    assert fw6a.assert_writable(fw6a.DEFAULT_OUTDIR) == fw6a.DEFAULT_OUTDIR
    assert fw6a.assert_writable(tmp_path) == tmp_path


# ---------------------------------------------------------------------------
# end-to-end regeneration
# ---------------------------------------------------------------------------
@needs_outputs
@pytest.mark.slow
@pytest.mark.parametrize("era", ["v2", "pre_v2"])
def test_committed_artefacts_regenerate_from_scratch(era):
    """Re-price both eras and require the committed CSVs back, to 1e-4."""
    prod_kappa = float(load_frozen_parameters(fw6a.PARAMS_YAML).kappa_per_hour)
    kappa = prod_kappa if era == "v2" else fw6a.PRE_V2_KAPPA_PER_HOUR
    name = ("f25_v2_kappa.csv" if era == "v2"
            else "f25_pre_v2_reproduction.csv")
    fresh = fw6a.run_comparison(kappa, era=era)
    on_disk = pd.read_csv(OUT / name)
    pd.testing.assert_frame_equal(fresh[on_disk.columns], on_disk,
                                  check_dtype=False, atol=1e-4)
