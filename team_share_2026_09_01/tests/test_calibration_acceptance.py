"""Acceptance semantics: optimizer_success vs calibration_accepted, refusals."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pde_option_model.market_calibration import (FORBIDDEN_LABEL, PRICE_LABEL,
                                                 AcceptanceCriteria,
                                                 CalibrationRejected,
                                                 calibrated_config,
                                                 load_calibration_result_json,
                                                 near_term_anchor_sensitivity,
                                                 parameter_identification,
                                                 run_market_calibration)
from pde_option_model.forward_centered import ResidualSpec
from pde_option_model.forward_curve import NearTermAnchor
from pde_option_model.model_modes import (MODEL_MODES, validate_model_mode,
                                          write_calibration_outputs)


@pytest.fixture(scope="module")
def calibration(quotes, params):
    return run_market_calibration(quotes, params, curve_mode="smooth_constrained")


def test_calibration_is_accepted(calibration):
    assert calibration.optimizer_success is True
    assert calibration.calibration_accepted is True
    assert calibration.failed_checks == []
    assert calibration.metrics["maximum_absolute_monthly_error"] < 0.10
    assert calibration.metrics["monthly_MAPE"] < 1.0


def test_optimizer_success_and_acceptance_are_independent(quotes, params):
    """A converged solve with an economically absurd tolerance must be REJECTED."""
    impossible = AcceptanceCriteria(max_abs_monthly_error_TRY_MWh=1e-30,
                                    max_mape_pct=1e-30)
    res = run_market_calibration(quotes, params, criteria=impossible)
    assert res.optimizer_success is True, "the linear solve still converged"
    assert res.calibration_accepted is False, "but the economic check failed"
    assert "monthly_delivery_average_matches_quote" in res.failed_checks


def test_implausible_magnitude_check_rejects(quotes, params):
    res = run_market_calibration(
        quotes, params,
        criteria=AcceptanceCriteria(max_plausible_forward_TRY_MWh=100.0))
    assert res.optimizer_success is True
    assert res.calibration_accepted is False
    assert "no_implausible_forward_magnitude" in res.failed_checks


def test_rejected_calibration_refuses_to_produce_a_config(quotes, params):
    res = run_market_calibration(
        quotes, params, criteria=AcceptanceCriteria(max_abs_monthly_error_TRY_MWh=1e-30))
    with pytest.raises(CalibrationRejected):
        res.require_accepted()
    with pytest.raises(CalibrationRejected):
        calibrated_config(res, params, "hourly_forward_curve.csv", 0.40)


def test_rejected_run_writes_no_calibrated_config(tmp_path, quotes, params):
    res = run_market_calibration(
        quotes, params, criteria=AcceptanceCriteria(max_abs_monthly_error_TRY_MWh=1e-30))
    out = write_calibration_outputs(res, params, tmp_path / "rejected")
    assert out.accepted is False
    assert "calibrated_config.yaml" not in out.files
    assert "CALIBRATION_REJECTED.txt" in out.files
    assert "calibration_result.json" in out.files, "audit trail must still be written"
    blob = json.loads((out.directory / "calibration_result.json").read_text())
    assert blob["calibration_accepted"] is False
    assert blob["optimizer_success"] is True


def test_pricing_refuses_a_failed_calibration(tmp_path, quotes, params):
    """load_calibration_result_json must refuse a rejected directory."""
    res = run_market_calibration(
        quotes, params, criteria=AcceptanceCriteria(max_abs_monthly_error_TRY_MWh=1e-30))
    out = write_calibration_outputs(res, params, tmp_path / "rejected")
    with pytest.raises(CalibrationRejected, match="calibration_accepted=false"):
        load_calibration_result_json(out.directory)


def test_accepted_calibration_is_loadable(tmp_path, calibration, params):
    out = write_calibration_outputs(calibration, params, tmp_path / "ok")
    assert out.accepted is True
    assert "calibrated_config.yaml" in out.files
    blob = load_calibration_result_json(out.directory)
    assert blob["calibration_accepted"] is True
    assert blob["price_label"] == PRICE_LABEL


def test_missing_calibration_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration_result_json(tmp_path)


# --------------------------------------------------------------------------
def test_no_option_price_identification_warning(calibration, params):
    """Risk premia must be reported as NOT identified, with a stated reason."""
    grp = calibration.parameter_provenance["4_not_identified_without_option_prices"]
    for key in ("volatility_risk_premium", "sigma_risk_premium",
                "eta01_transition_premium", "eta10_transition_premium",
                "regime_market_price_of_risk_lambda_i"):
        assert key in grp and grp[key] is None
    assert "option" in grp["reason"].lower()
    assert "how_to_identify_later" in grp


def test_price_label_is_anchored_not_fully_calibrated(calibration):
    assert calibration.price_label == PRICE_LABEL
    assert "anchored" in PRICE_LABEL
    assert PRICE_LABEL != FORBIDDEN_LABEL
    blob = calibration.to_json_dict()
    assert blob["price_label"] == PRICE_LABEL
    assert FORBIDDEN_LABEL not in json.dumps(blob["parameter_provenance"]["1_calibrated_from_market_data"])


def test_provenance_has_four_disjoint_groups(params):
    ident = parameter_identification(
        params, ResidualSpec.from_frozen(params), NearTermAnchor(), 0.40)
    for g in ("1_calibrated_from_market_data",
              "2_inherited_from_historical_M2_fit",
              "3_fixed_by_assumption",
              "4_not_identified_without_option_prices"):
        assert g in ident
    assert "hourly_forward_curve_F_t" in ident["1_calibrated_from_market_data"]
    assert "kappa_per_hour" in ident["2_inherited_from_historical_M2_fit"]
    assert "january_anchor_mode" in ident["3_fixed_by_assumption"]


def test_placeholder_parameters_are_reported(calibration, params):
    if params.has_placeholders:
        assert any("PLACEHOLDER" in w for w in calibration.warnings)
        tv = calibration.parameter_provenance[
            "2_inherited_from_historical_M2_fit"]["tvtp_coefficients"]
        assert tv["placeholder"] is True and tv["placeholder_fields"]


# --------------------------------------------------------------------------
def test_january_is_flagged_as_extrapolation(calibration):
    js = calibration.january_status
    assert js["months_without_observed_quote"] == ["2026-01"]
    assert "2026-01" in js["flagged_extrapolated"]
    assert "near-term anchored" in js["status"]
    assert any("near-term anchored" in w for w in calibration.warnings)


def test_all_reported_horizons_are_flagged_correctly(calibration):
    """All four requested horizons land in unquoted January 2026."""
    for h in (72, 168, 336, 720):
        det = calibration.expected_spot[f"{h}h_detail"]
        assert det["delivery_month"] == "2026-01"
        assert det["near_term_anchored"] is True
        assert det["directly_constrained_by_a_VEP_quote"] is False
        assert np.isfinite(det["expected_spot_TRY_MWh"])
    assert any("near-term anchored" in w and "reported horizons" in w
               for w in calibration.warnings)


def test_monthly_averages_use_true_delivery_hour_counts(calibration, quotes):
    fit = calibration.fit_table
    assert list(fit["number_of_delivery_hours"]) == [q.n_delivery_hours
                                                     for q in quotes.quotes]
    assert fit["acceptance_passed"].all()


def test_fit_table_schema(calibration):
    for col in ("contract_name", "delivery_start_utc", "delivery_end_utc",
                "number_of_delivery_hours", "market_forward_TRY_MWh",
                "model_average_TRY_MWh", "residual_TRY_MWh",
                "relative_error_pct", "acceptance_passed"):
        assert col in calibration.fit_table.columns


def test_result_json_has_every_required_field(calibration):
    blob = calibration.to_json_dict()
    for key in ("optimizer_success", "calibration_accepted", "valuation_date",
                "quote_source", "model_type", "monthly_RMSE", "monthly_MAE",
                "monthly_MAPE", "maximum_absolute_monthly_error",
                "finite_value_checks", "expected_spot_at_72h",
                "expected_spot_at_168h", "expected_spot_at_336h",
                "expected_spot_at_720h", "parameter_provenance", "warnings",
                "january_calibration_status", "direct_market_constraints",
                "extrapolated_periods"):
        assert key in blob, key
    assert all(blob["finite_value_checks"].values())
    json.dumps(blob)      # must be serialisable


def test_anchor_sensitivity_keeps_quoted_months_exact(quotes, params):
    df = near_term_anchor_sensitivity(
        quotes, params, [2400.0, 2917.78, 3400.0], curve_mode="smooth_constrained")
    assert len(df) == 3
    assert (df["max_abs_monthly_error_TRY_MWh"] < 1e-6).all()
    assert df["expected_spot_72h_TRY_MWh"].is_monotonic_increasing
    assert bool(df.loc[df["january_anchor_TRY_MWh"] == 2917.78,
                       "spot_consistent_at_t0"].iloc[0]) is True


def test_model_mode_registry():
    assert set(MODEL_MODES) == {"legacy_asinh_ou", "forward_centered"}
    assert validate_model_mode("forward_centered") == "forward_centered"
    with pytest.raises(ValueError):
        validate_model_mode("nonexistent_mode")


def test_all_required_output_files_are_written(tmp_path, calibration, params):
    out = write_calibration_outputs(
        calibration, params, tmp_path / "full",
        anchor_sensitivity=pd.DataFrame({
            "january_anchor_TRY_MWh": [2600.0, 2917.78, 3200.0],
            "anchor_vs_spot_pct": [-10.0, 0.0, 10.0],
            "max_abs_monthly_error_TRY_MWh": [0.0, 0.0, 0.0],
            "expected_spot_72h_TRY_MWh": [2600.0, 2917.78, 3200.0],
        }))
    required = {"calibration_result.json", "calibration_audit.md",
                "hourly_forward_curve.csv", "monthly_forward_fit.csv",
                "monthly_forward_fit.png", "legacy_vs_forward_centered.png",
                "calibrated_config.yaml", "parameter_identification.json",
                "model_limitations.md", "near_term_anchor_sensitivity.csv",
                "near_term_anchor_sensitivity.png"}
    assert required.issubset(set(out.files)), required - set(out.files)
    for f in required:
        assert (out.directory / f).exists() and (out.directory / f).stat().st_size > 0
