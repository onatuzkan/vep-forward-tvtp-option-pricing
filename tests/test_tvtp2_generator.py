"""Two-covariate TVTP: coefficients, label swap, probabilities, generator, embeddability.

Validation items 1 (probability identity at h = 0) and 2 (label swap,
transition probabilities, generator row sums, expm(Q * 1h) round trip).
Repository data only: the M9 bundle, the frozen YAMLs, rd_standardized.csv
and (for the unit guard) realised PTF prices.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from pde_option_model.generator import (CovariateError, EmbeddabilityError, TVTP2Coefficients,
                                        TVTPCoefficients, embeddability_report,
                                        generator_to_probs, probs_to_generator)
from pde_option_model.tvtp2 import (load_m9_bundle, m9_raw_to_production)

from .conftest import M9_BUNDLE, REPO_ROOT


# ---------------------------------------------------------------------------
# frozen coefficients and the atomic label swap
# ---------------------------------------------------------------------------
def test_ramp_slopes_are_the_cross_mapped_m9_values(tvtp2_params, params):
    c = tvtp2_params.coefficients
    # normal -> stress (production 01) = raw M9 gamma10 of RD_Ramp_1h_lag1
    assert c.h01 == -0.06939531482057137
    # stress -> normal (production 10) = raw M9 gamma01 of RD_Ramp_1h_lag1
    assert c.h10 == 0.4055443473030015
    # RD slopes are exactly the production-yaml values
    assert c.gamma01 == params.gamma01 and c.gamma10 == params.gamma10
    # the single-covariate intercepts are NOT reused in the 2D set
    assert c.alpha01 != params.alpha01 and c.alpha10 != params.alpha10


def test_label_swap_is_atomic_and_internally_consistent():
    raw = load_m9_bundle(M9_BUNDLE)
    prod = m9_raw_to_production(raw)
    rd, rp = raw["coefficients"]["RD_lag1"], raw["coefficients"]["RD_Ramp_1h_lag1"]
    assert raw["sigma0"] > raw["sigma1"], "raw state 0 is the high-volatility state"
    assert prod["sigma_y_normal"] == raw["sigma1"] and prod["sigma_y_stress"] == raw["sigma0"]
    assert prod["gamma01"] == rd["gamma10"] and prod["gamma10"] == rd["gamma01"]
    assert prod["h01"] == rp["gamma10"] and prod["h10"] == rp["gamma01"]
    assert prod["duration_normal_h"] == raw["mean_duration_state1"]
    assert prod["duration_stress_h"] == raw["mean_duration_state0"]
    assert prod["occupancy_stress"] == raw["occupancy"][0]
    ck = prod["checks"]
    assert ck["sigma_normal_lt_sigma_stress"]
    assert ck["ame_and_slope_share_sign_in_every_equation"]
    # AME/slope = E[Lambda'] of ONE fitted p path -> equal across covariates (joint fit)
    assert ck["ame_over_slope_abs_diff_p01"] < 1e-12
    assert ck["ame_over_slope_abs_diff_p10"] < 1e-12
    assert ck["joint_estimation_evidence"] is True


def test_label_swap_refuses_an_already_swapped_bundle():
    raw = load_m9_bundle(M9_BUNDLE)
    flipped = dict(raw, sigma0=raw["sigma1"], sigma1=raw["sigma0"])
    with pytest.raises(ValueError, match="high-volatility"):
        m9_raw_to_production(flipped)


def test_frozen_label_mapping_matches_the_bundle(tvtp2_params):
    prod = m9_raw_to_production(load_m9_bundle(M9_BUNDLE))
    mapping = {r["production"]: r for r in tvtp2_params.m9_source["label_mapping"]}
    assert mapping["h01 (normal->stress, ramp)"]["value_used"] == prod["h01"]
    assert mapping["h10 (stress->normal, ramp)"]["value_used"] == prod["h10"]
    assert tvtp2_params.m9_source["joint_estimation_evidence"] is True


# ---------------------------------------------------------------------------
# probabilities
# ---------------------------------------------------------------------------
def test_zero_ramp_reproduces_single_covariate_probabilities_bitwise(params, panel):
    D = panel.sample()
    z, r = D["z_lag"].to_numpy(), D["r_lag"].to_numpy()
    one = TVTPCoefficients(params.alpha01, params.gamma01, params.alpha10, params.gamma10)
    two = TVTP2Coefficients.from_single_covariate(one)
    a1, b1 = one.probabilities(z)
    a2, b2 = two.probabilities(z, r)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    g1 = probs_to_generator(a1, b1)
    g2 = probs_to_generator(a2, b2, on_nonembeddable="raise")
    assert np.array_equal(g1.q01, g2.q01) and np.array_equal(g1.q10, g2.q10)
    assert two.single_covariate() == one


def test_two_covariate_probabilities_match_the_closed_form(tvtp2_params, panel):
    c = tvtp2_params.coefficients
    D = panel.sample()
    for lab in (pd.Timestamp("2020-05-25 06:00", tz="UTC"),       # historical max s
                pd.Timestamp("2024-12-31 20:00", tz="UTC"),       # end of W9
                pd.Timestamp("2025-12-31 20:00", tz="UTC")):      # valuation hour
        z, r = float(D.loc[lab, "z_lag"]), float(D.loc[lab, "r_lag"])
        p01, p10 = c.probabilities(z, r)
        e01 = 1.0 / (1.0 + math.exp(-(c.alpha01 + c.gamma01 * z + c.h01 * r)))
        e10 = 1.0 / (1.0 + math.exp(-(c.alpha10 + c.gamma10 * z + c.h10 * r)))
        assert abs(p01[0] - e01) < 1e-15 and abs(p10[0] - e10) < 1e-15


def test_ramp_moves_the_stress_exit_probability_in_the_m9_direction(tvtp2_params, panel):
    """h10 > 0: an upward residual-demand ramp raises stress -> normal exits."""
    c = tvtp2_params.coefficients
    D = panel.sample()
    z = D["z_lag"].to_numpy()
    r = D["r_lag"].to_numpy()
    up, dn = r > 1.0, r < -1.0
    _, p10 = c.probabilities(z, r)
    zref = np.full(z.size, float(np.median(z)))
    _, p10_ref_up = c.probabilities(zref[up], r[up])
    _, p10_ref_dn = c.probabilities(zref[dn], r[dn])
    assert p10_ref_up.mean() > p10_ref_dn.mean()
    assert c.h10 > 0 and c.h01 < 0


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
def test_generator_rows_sum_to_zero_and_expm_returns_the_one_hour_matrix(tvtp2_params, panel):
    c = tvtp2_params.coefficients
    D = panel.sample()
    p01, p10 = c.probabilities(D["z_lag"].to_numpy(), D["r_lag"].to_numpy())
    gen = probs_to_generator(p01, p10, on_nonembeddable="raise")
    assert gen.n_clipped == 0
    Q = gen.q_matrix
    assert np.all(gen.q01 >= 0) and np.all(gen.q10 >= 0)
    assert np.max(np.abs(Q.sum(axis=2))) < 1e-14
    # lambda = q01 + q10 = -ln(1 - s)
    s = p01 + p10
    assert np.max(np.abs(gen.q01 + gen.q10 + np.log1p(-s))) < 1e-12
    # closed-form round trip on every historical transition
    b01, b10 = generator_to_probs(gen.q01, gen.q10)
    assert max(np.max(np.abs(b01 - p01)), np.max(np.abs(b10 - p10))) < 1e-14
    # independent scipy expm on a spread of hours, including the max-s hour
    idx = np.unique(np.concatenate([np.linspace(0, s.size - 1, 60).astype(int),
                                    [int(np.argmax(s))]]))
    for k in idx:
        P = scipy.linalg.expm(Q[k] * 1.0)
        Pi = np.array([[1 - p01[k], p01[k]], [p10[k], 1 - p10[k]]])
        assert np.max(np.abs(P - Pi)) < 1e-13


def test_nonembeddable_hours_are_rejected_not_clipped(tvtp2_params, path_builder, params):
    """A -4.5 sd climatology path leaves the historical support of z (min z
    about -5.7 vs -3.64 observed); with the ramp the one-hour matrix has
    p01 + p10 >= 1 at some hours while the single-covariate law does not yet."""
    from pde_option_model.scenarios import ScenarioSpec
    c = tvtp2_params.coefficients
    path = path_builder.build(ScenarioSpec("extreme", "climatology", -4.5),
                              params.valuation_utc, horizon_hours=720.0)
    rep = path.embeddability(c)
    assert rep["n_s_ge_1"] > 0 and rep["max_s"] > 1.0 and not rep["embeddable"]
    assert 0.0 < rep["share_s_ge_1"] < 1.0
    assert rep["first_violation_label"] is not None
    pos = path.used_hour_positions()
    p01, p10 = c.probabilities(path.z_hourly[pos], path.ramp_hourly[pos])
    with pytest.raises(EmbeddabilityError) as ei:
        probs_to_generator(p01, p10, on_nonembeddable="raise")
    assert ei.value.report["n_s_ge_1"] == rep["n_s_ge_1"]
    # the historical single-covariate API keeps its documented clip-and-count default
    legacy = probs_to_generator(p01, p10)
    assert legacy.n_clipped == rep["n_s_ge_1"]
    # the single-covariate law on the same z path stays embeddable: the ramp causes it
    one = TVTPCoefficients(params.alpha01, params.gamma01, params.alpha10, params.gamma10)
    a, b = one.probabilities(path.z_hourly[pos])
    assert embeddability_report(a, b)["n_s_ge_1"] == 0


def test_historical_two_covariate_path_is_embeddable(tvtp2_params, panel):
    from pde_option_model.tvtp2 import historical_embeddability
    rep = historical_embeddability(panel, tvtp2_params.coefficients)
    assert rep["n_s_ge_1"] == 0 and rep["n_rows"] == 87662
    assert 0.80 < rep["max_s"] < 0.85


# ---------------------------------------------------------------------------
# input validation (shape, missing values, units)
# ---------------------------------------------------------------------------
def test_missing_ramp_is_an_error_not_a_zero(tvtp2_params, panel):
    c = tvtp2_params.coefficients
    D = panel.frame
    with pytest.raises(CovariateError, match="refusing to default"):
        c.probabilities(D["z_lag"].dropna().to_numpy()[:5])
    gap = D.loc["2016-03-27 01:00+00:00":"2016-03-27 05:00+00:00"]
    with pytest.raises(CovariateError, match="never replaced by zero"):
        c.probabilities(gap["z_lag"].fillna(0.0).to_numpy(), gap["r_lag"].to_numpy())


def test_shape_mismatch_is_rejected(tvtp2_params, panel):
    D = panel.sample()
    with pytest.raises(CovariateError, match="identical shapes"):
        tvtp2_params.coefficients.probabilities(D["z_lag"].to_numpy()[:10],
                                                D["r_lag"].to_numpy()[:9])


def test_unstandardized_inputs_are_rejected_as_a_unit_error(tvtp2_params):
    """Realised PTF prices (TRY/MWh) are the wrong unit for a standardized covariate."""
    ptf = pd.read_csv(REPO_ROOT / "inputs" / "market" / "realized_ptf_2026.csv",
                      sep=";", decimal=",", thousands=".")
    x = ptf["PTF (TL/MWh)"].to_numpy(dtype=float)[:48]
    assert np.max(np.abs(x)) > 100
    with pytest.raises(CovariateError, match="STANDARDIZED"):
        tvtp2_params.coefficients.probabilities(x, np.zeros_like(x))


def test_single_covariate_api_is_unchanged(params):
    one = TVTPCoefficients(params.alpha01, params.gamma01, params.alpha10, params.gamma10)
    z = np.array([-1.0, 0.0, 1.0])
    p01, p10 = one.probabilities(z)
    assert np.allclose(p01, 1 / (1 + np.exp(-(params.alpha01 + params.gamma01 * z))))
    assert one.mode == "rd_lag1_1d" and one.n_covariates == 1
    gen = probs_to_generator(p01, p10)          # default policy remains "clip"
    assert gen.n_clipped == 0 and gen.method == "matrix_log"
