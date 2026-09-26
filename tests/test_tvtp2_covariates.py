"""Two-covariate TVTP: hourly covariate panel, lag alignment, train-only ramp
standardization, missing observations, scenario paths and the frozen artefact.

Validation item 3 (z/r lag, train-only standardization, time-series gaps) and
the scenario-path rules (constant / climatology / custom, initial hours).
Repository data only; perturbation tests alter REAL values after a window end
(look-ahead guard) or drop REAL rows (gap guard) -- no synthetic series.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import pytest
import yaml

from pde_option_model.generator import CovariateError
from pde_option_model.params_frozen import FrozenParameterError, load_tvtp2_parameters
from pde_option_model.scenarios import CovariatePathBuilder, ScenarioBuilder, ScenarioSpec
from pde_option_model.transformations import make_time_index
from pde_option_model.tvtp2 import (EXPERIMENTAL_LABEL, CovariateDataError, RampScaler,
                                    build_covariate_panel, derive_intercepts, fit_ramp_scaler,
                                    run_tvtp2_derivation)

from .conftest import PARAMS_YAML, REPO_ROOT, TRY_TRAIN_END, TVTP2_YAML

H = pd.Timedelta(hours=1)
GAP = [pd.Timestamp(f"2016-03-27 0{h}:00", tz="UTC") for h in (0, 1, 2)]
W9_END = pd.Timestamp("2024-12-31 20:00", tz="UTC")


# ---------------------------------------------------------------------------
# historical panel
# ---------------------------------------------------------------------------
def test_history_is_mapped_to_the_complete_hourly_grid(panel, z_history):
    assert len(z_history) == 87665
    assert len(panel.frame) == 87668
    assert list(panel.missing_labels) == GAP
    assert panel.frame["z"].isna().sum() == 3            # never filled


def test_ramp_is_a_calendar_difference_and_undefined_across_the_gap(panel, z_history):
    fr = panel.frame
    t = pd.Timestamp("2016-03-27 03:00", tz="UTC")
    # a row-order diff would write the 4-hour move as a one-hour ramp
    row_order = z_history.loc[t] - z_history.loc[pd.Timestamp("2016-03-26 23:00", tz="UTC")]
    assert abs(row_order + 0.5116) < 1e-3
    assert np.isnan(fr.loc[t, "dz"]) and np.isnan(fr.loc[t, "r"])
    for lab in ("2016-03-27 01:00", "2016-03-27 02:00", "2016-03-27 03:00", "2016-03-27 04:00"):
        assert not fr.loc[pd.Timestamp(lab, tz="UTC"), "valid"]
    # nothing was replaced by zero
    assert not np.any(fr.loc["2016-03-27 00:00+00:00":"2016-03-27 04:00+00:00", "r_lag"] == 0.0)


def test_z_and_ramp_are_lagged_together_by_one_hour(panel, tvtp2_params):
    fr = panel.frame
    s = tvtp2_params.ramp_scaler
    for lab in ("2020-05-25 06:00", "2024-12-31 20:00", "2025-12-31 20:00", "2016-01-01 03:00"):
        t = pd.Timestamp(lab, tz="UTC")
        assert fr.loc[t, "z_lag"] == fr.loc[t - H, "z"]
        expect = (fr.loc[t - H, "z"] - fr.loc[t - 2 * H, "z"] - s.mean) / s.std
        assert abs(fr.loc[t, "r_lag"] - expect) < 1e-14
        assert fr.loc[t, "r_lag"] == fr.loc[t - H, "r"]


def test_first_two_hours_and_the_gap_are_dropped_and_counted(panel):
    acc = panel.accounting(W9_END)
    assert acc["n_valid_transitions"] == 78902
    assert acc["n_dropped_sample_start"] == 2
    assert acc["dropped_sample_start_labels"] == ["2016-01-01T01:00:00+00:00",
                                                  "2016-01-01T02:00:00+00:00"]
    assert acc["n_dropped_missing_observation"] == 4
    assert acc["dropped_missing_observation_labels"] == [
        "2016-03-27T01:00:00+00:00", "2016-03-27T02:00:00+00:00",
        "2016-03-27T03:00:00+00:00", "2016-03-27T04:00:00+00:00"]
    assert acc["first_valid_transition_utc"] == "2016-01-01T03:00:00+00:00"
    assert acc["last_valid_transition_utc"] == "2024-12-31T20:00:00+00:00"


def test_dropping_a_real_row_propagates_nan_instead_of_bridging(z_history, tvtp2_params):
    lab = pd.Timestamp("2023-07-01 12:00", tz="UTC")
    p = build_covariate_panel(z_history.drop(lab), tvtp2_params.ramp_scaler)
    fr = p.frame
    for k in (1, 2):                                   # t = lab+1h, lab+2h lose x_(t-1)
        assert not fr.loc[lab + k * H, "valid"]
    assert fr.loc[lab + 3 * H, "valid"]
    assert fr.loc[lab, "valid"] and np.isnan(fr.loc[lab, "z"])   # own z missing, x_(t-1) observed
    assert p.accounting(W9_END)["n_dropped_missing_observation"] == 4 + 2


# ---------------------------------------------------------------------------
# train-only standardization and intercepts (look-ahead guard)
# ---------------------------------------------------------------------------
def test_ramp_scaler_and_intercepts_ignore_data_after_the_window(z_history, tvtp2_params):
    s0 = tvtp2_params.ramp_scaler
    after = z_history.index > W9_END
    perturbed = z_history.copy()
    perturbed[after] = -3.0 * perturbed[after]                   # real values, scrambled
    for zz in (perturbed, z_history[~after]):                    # scrambled / truncated
        s1 = fit_ramp_scaler(zz, W9_END, "W9")
        assert s1.mean == s0.mean and s1.std == s0.std and s1.n == s0.n
        d = derive_intercepts(build_covariate_panel(zz, s1), tvtp2_params.coefficients.gamma01,
                              tvtp2_params.coefficients.h01, tvtp2_params.coefficients.gamma10,
                              tvtp2_params.coefficients.h10,
                              tvtp2_params.derivation["targets"]["duration_normal_h"],
                              tvtp2_params.derivation["targets"]["duration_stress_h"], W9_END)
        assert d["alpha01"] == tvtp2_params.coefficients.alpha01
        assert d["alpha10"] == tvtp2_params.coefficients.alpha10
    # the window itself does matter (the guard is not vacuous)
    sT = fit_ramp_scaler(z_history, TRY_TRAIN_END, "W_T")
    assert abs(sT.std - s0.std) > 1e-3


def test_frozen_artefact_matches_a_fresh_derivation(tvtp2_params):
    blob, audit = run_tvtp2_derivation(REPO_ROOT)
    c = tvtp2_params.coefficients
    assert blob["tvtp2"]["alpha01"] == c.alpha01 and blob["tvtp2"]["alpha10"] == c.alpha10
    assert blob["tvtp2"]["h01"] == c.h01 and blob["tvtp2"]["h10"] == c.h10
    sc = blob["covariates"]["ramp"]["scaler"]
    assert sc["mean"] == tvtp2_params.ramp_scaler.mean and sc["std"] == tvtp2_params.ramp_scaler.std
    assert sc["window_end_utc"] == "2024-12-31T20:00:00+00:00" and sc["ddof"] == 1
    assert blob["derivation"]["sample"]["n_valid_transitions"] == 78902
    for k in ("p01", "p10"):
        assert abs(blob["derivation"]["roots"][k]["residual"]) < 1e-12
    assert blob["status"] == "experimental_reconstructed"
    assert blob["verified_reproduction_of_m9"] is False
    assert blob["label"] == EXPERIMENTAL_LABEL == tvtp2_params.label
    # the same derivation machinery reproduces the 1D production intercepts
    rd = audit["production_1d_rederived"]
    assert abs(rd["alpha01"] - rd["yaml_alpha01"]) < 1e-9
    assert abs(rd["alpha10"] - rd["yaml_alpha10"]) < 1e-9


def _write_variant(tmp_path, edit) -> str:
    blob = yaml.safe_load(TVTP2_YAML.read_text(encoding="utf-8"))
    edit(blob)
    p = tmp_path / "tvtp2_variant.yaml"
    p.write_text(yaml.safe_dump(blob, sort_keys=False), encoding="utf-8")
    return str(p)


def test_loader_refuses_to_label_a_reconstructed_ramp_as_verified(tmp_path):
    p = _write_variant(tmp_path, lambda b: b.update(verified_reproduction_of_m9=True))
    with pytest.raises(FrozenParameterError, match="reconstructed"):
        load_tvtp2_parameters(p)


def test_loader_refuses_a_nonzero_transition_premium(tmp_path):
    p = _write_variant(tmp_path, lambda b: b["transition_premium"].update(eta10=0.2))
    with pytest.raises(FrozenParameterError, match="premia"):
        load_tvtp2_parameters(p)


def test_loader_refuses_look_ahead_windows(tmp_path):
    def edit(b):
        b["covariates"]["ramp"]["scaler"]["window_end_utc"] = "2026-03-01T00:00:00+00:00"
    with pytest.raises(FrozenParameterError, match="leakage"):
        load_tvtp2_parameters(_write_variant(tmp_path, edit))


def test_base_single_covariate_yaml_is_unchanged_since_the_derivation(tvtp2_params):
    assert tvtp2_params.base_parameters_match is True


# ---------------------------------------------------------------------------
# scenario paths (constant / climatology / custom) and their alignment
# ---------------------------------------------------------------------------
def _scenario_1d(z_history, params, T, spec, n_master):
    from pde_option_model.grid import TimeGrid
    tg = TimeGrid(params.valuation_utc, params.valuation_utc + pd.Timedelta(hours=T), n_master)
    return ScenarioBuilder(z_history, TRY_TRAIN_END, 1.0).build(spec, tg)


@pytest.mark.parametrize("T", [24, 72, 168])
@pytest.mark.parametrize("mode,offset", [("climatology", 0.0), ("climatology", -2.0),
                                         ("constant", 0.0), ("constant", 1.5)])
def test_path_z_is_bit_identical_to_the_single_covariate_scenario(
        path_builder, z_history, params, production_grid, T, mode, offset):
    n = CovariatePathBuilder.master_steps(float(T), production_grid)
    old = _scenario_1d(z_history, params, T, ScenarioSpec("x", mode, offset), n)
    new = path_builder.build(ScenarioSpec("x", mode, offset), params.valuation_utc,
                             horizon_hours=float(T), n_master=n)
    assert np.array_equal(old.times_hours, new.master_times)
    assert np.array_equal(old.z_lagged, new.z_master)
    t_mc = np.linspace(0.0, T, int(T / 0.05) + 1)
    assert np.array_equal(np.interp(t_mc, old.times_hours, old.z_lagged), new.z_lagged(t_mc))


def test_path_ramp_is_the_hourly_difference_mapped_by_the_label_rule(path_builder, params,
                                                                     tvtp2_params):
    s = tvtp2_params.ramp_scaler
    path = path_builder.build(ScenarioSpec("clim"), params.valuation_utc, horizon_hours=72.0)
    zh, rh = path.z_hourly, path.ramp_hourly
    assert np.isnan(rh[0])                                     # no predecessor: never used
    assert np.allclose(rh[1:], (zh[1:] - zh[:-1] - s.mean) / s.std, atol=0, rtol=0)
    tv = params.valuation_utc
    assert path.labels[0] == tv - 2 * H and path.labels[-1] == tv + 71 * H
    for tau in (0.0, 0.25, 0.75, 1.0, 1.5, 2.0, 71.75, 72.0):
        lab = make_time_index(tv, [tau - 1.0]).floor("h")[0]
        assert lab == tv + int(np.floor(tau)) * H - H         # l(tau) = t_v + floor(tau) - 1h
        k = path.labels.get_loc(lab)
        assert path.z_lagged(np.array([tau]))[0] == zh[k]
        assert path.ramp_lagged(np.array([tau]))[0] == rh[k]


def test_constant_path_ramp_is_the_standardized_zero_increment(path_builder, params, tvtp2_params):
    s = tvtp2_params.ramp_scaler
    path = path_builder.build(ScenarioSpec("c", "constant", 0.7), params.valuation_utc,
                              horizon_hours=48.0)
    assert np.all(path.ramp_master == -s.mean / s.std)
    assert np.all(path.ramp_master != 0.0)                    # not a filled zero
    assert np.all(path.z_master == 0.7)


def _real_custom_csv(z_history, params, tmp_path, hours=80, drop=None, name="custom.csv"):
    """Observed z of the same calendar hours one year earlier (2025), re-stamped."""
    tv = params.valuation_utc
    labels = pd.date_range(tv - 2 * H, tv + (hours - 1) * H, freq="h")
    src = z_history.reindex(labels - pd.Timedelta(hours=8760))
    assert src.notna().all()
    df = pd.DataFrame({"datetime": [x.isoformat() for x in labels], "z": src.to_numpy()})
    if drop is not None:
        df = df[df["datetime"] != drop.isoformat()]
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p, labels, src.to_numpy()


def test_custom_csv_path_uses_exactly_the_supplied_hours(path_builder, z_history, params,
                                                         tmp_path, tvtp2_params):
    p, labels, vals = _real_custom_csv(z_history, params, tmp_path)
    path = path_builder.build(ScenarioSpec("cust", "custom", 0.0, custom_csv=str(p)),
                              params.valuation_utc, horizon_hours=72.0)
    k = path.labels.get_indexer(labels[: len(path.labels)])
    assert np.array_equal(path.z_hourly, vals[k])
    s = tvtp2_params.ramp_scaler
    assert np.allclose(path.ramp_hourly[1:], (np.diff(vals[k]) - s.mean) / s.std, rtol=0, atol=0)
    assert path.source["custom_csv_sha256"]


def test_custom_csv_missing_hour_is_listed_not_interpolated(path_builder, z_history, params,
                                                           tmp_path):
    miss = params.valuation_utc + 30 * H
    p, _, _ = _real_custom_csv(z_history, params, tmp_path, drop=miss)
    with pytest.raises(CovariateDataError, match=miss.isoformat().replace("+", r"\+")):
        path_builder.build(ScenarioSpec("cust", "custom", 0.0, custom_csv=str(p)),
                           params.valuation_utc, horizon_hours=72.0)


def test_custom_csv_first_two_hours_are_required(path_builder, z_history, params, tmp_path):
    p, _, _ = _real_custom_csv(z_history, params, tmp_path, drop=params.valuation_utc - 2 * H)
    with pytest.raises(CovariateDataError, match="lacks 1 required hour"):
        path_builder.build(ScenarioSpec("cust", "custom", 0.0, custom_csv=str(p)),
                           params.valuation_utc, horizon_hours=72.0)


def test_custom_csv_rejects_subhour_and_duplicate_stamps(path_builder, z_history, params,
                                                        tmp_path):
    p, _, _ = _real_custom_csv(z_history, params, tmp_path)
    df = pd.read_csv(p)
    shifted = df.copy()
    shifted.loc[5, "datetime"] = pd.Timestamp(df.loc[5, "datetime"]).replace(minute=30).isoformat()
    p2 = tmp_path / "subhour.csv"
    shifted.to_csv(p2, index=False)
    with pytest.raises(CovariateDataError, match="sub-hour"):
        path_builder.build(ScenarioSpec("c", "custom", 0.0, custom_csv=str(p2)),
                           params.valuation_utc, horizon_hours=72.0)
    p3 = tmp_path / "dup.csv"
    pd.concat([df, df.iloc[[7]]]).to_csv(p3, index=False)
    with pytest.raises(CovariateDataError, match="duplicate"):
        path_builder.build(ScenarioSpec("c", "custom", 0.0, custom_csv=str(p3)),
                           params.valuation_utc, horizon_hours=72.0)


def test_observed_initial_hours_use_the_history_without_offset(path_builder, z_history, params):
    tv = params.valuation_utc
    path = path_builder.build(ScenarioSpec("o", "climatology", 1.0, initial_hours="observed"),
                              tv, horizon_hours=24.0)
    for k, lab in enumerate(path.labels):
        if lab <= tv:
            assert path.z_hourly[k] == z_history.loc[lab]
            assert path.hourly_source[k] == "observed"
        else:
            assert path.hourly_source[k] == "scenario"
    assert (path.hourly_source == "observed").sum() == 3       # t_v - 2h, t_v - 1h, t_v
    with pytest.raises(ValueError, match="only supported"):
        ScenarioBuilder(z_history, TRY_TRAIN_END, 1.0).build(
            ScenarioSpec("o", initial_hours="observed"),
            __import__("pde_option_model.grid", fromlist=["TimeGrid"]).TimeGrid(
                tv, tv + pd.Timedelta(hours=24), 96))


def test_path_refuses_to_extrapolate(path_builder, params):
    path = path_builder.build(ScenarioSpec("c"), params.valuation_utc, horizon_hours=24.0)
    with pytest.raises(CovariateError, match="extrapolating"):
        path.z_lagged(np.array([0.0, 30.0]))


def test_ramp_scaler_verification_detects_a_unit_mismatch(z_history, tvtp2_params):
    s = tvtp2_params.ramp_scaler
    wrong = RampScaler(mean=s.mean, std=1.0, ddof=1, n=s.n, window_name=s.window_name,
                       window_end_utc=s.window_end_utc)      # unstandardized dz
    b = CovariatePathBuilder(z_history, TRY_TRAIN_END, 1.0, wrong)
    with pytest.raises(CovariateError, match="mismatch"):
        b.verify_ramp_scaler()
    CovariatePathBuilder(z_history, TRY_TRAIN_END, 1.0, s).verify_ramp_scaler()
