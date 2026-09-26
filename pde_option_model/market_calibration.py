"""Market calibration of the forward-centered model to EPİAŞ VEP quotes.

What is and is not calibrated
-----------------------------
Six monthly baseload prices identify exactly one thing: the risk-neutral price
LEVEL, month by month.  They say nothing about volatility or about the price of
regime risk.  The calibration therefore produces

    "VEP-forward-curve anchored option prices"

and never "fully market-calibrated option prices".  Parameters are reported in
four disjoint provenance groups (see :func:`parameter_identification`):

    1. calibrated from market data   -- the hourly forward level F(t)
    2. inherited from the historical M2 fit -- kappa, sigma_normal, sigma_stress,
       the TVTP coefficients, the filtered regime distribution
    3. fixed by assumption           -- residual mode, kappa_X choice, regime
       means m_i, the January anchor rule, the discount rate
    4. NOT IDENTIFIED without option prices -- the volatility risk premium, the
       regime-switching premia eta01/eta10, and any regime-specific market price
       of risk lambda_i

Two success flags, deliberately separate
----------------------------------------
``optimizer_success``
    the numerical solve terminated normally.  For the exact constrained curve
    this is a linear KKT solve, so it is essentially always True.

``calibration_accepted``
    the ECONOMIC checks passed.  A converged optimizer with a 2.5 million
    TRY/MWh RMSE is a failed calibration; this flag is what downstream code
    honours.  When it is False the run writes ``calibration_result.json`` with
    the failure recorded, refuses to emit ``calibrated_config.yaml``, and the
    process exits non-zero.

TVTP mode reporting
-------------------
Every result records ``tvtp_mode`` (``rd_lag1_1d`` default, or the
EXPERIMENTAL ``rd_ramp_2d_experimental``) and the parameter / covariate-path
provenance.  In the two-covariate mode the monthly identity is evaluated on
the supplied (z, ramp) path; it holds for any path because the centering
uses the same q sequence, but the path must exist and be embeddable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .calendar_tr import delivery_months_between, to_turkey
from .contracts import EuropeanOption
from .forward_centered import (ForwardCenteredModel, ResidualGridSettings,
                               ResidualSpec, price_forward_centered,
                               simulate_forward_centered)
from .forward_curve import (CURVE_COLUMNS, ForwardCurve, NearTermAnchor,
                            build_forward_curve)
from .generator import (TVTP2Coefficients, TVTPCoefficients, TVTP_MODE_1D,
                        TVTP_MODE_2D, EmbeddabilityError)
from .legacy_moments import (EXPLOSION_WARNING, legacy_expected_spot,
                             legacy_explosion_report, load_legacy_reference)
from .market_data import MarketQuoteSet
from .params_frozen import FrozenM2Parameters

logger = logging.getLogger(__name__)

PRICE_LABEL = "VEP-forward-curve anchored option prices"
FORBIDDEN_LABEL = "Fully market-calibrated option prices"

REPORTING_HORIZONS_HOURS: Tuple[int, ...] = (72, 168, 336, 720)


class CalibrationRejected(RuntimeError):
    """Raised when a calibration result is used despite failing acceptance."""


# ---------------------------------------------------------------------------
@dataclass
class AcceptanceCriteria:
    """Economic acceptance thresholds, separate from optimizer convergence."""

    max_abs_monthly_error_TRY_MWh: float = 0.10     # exact constrained curve
    max_mape_pct: float = 1.0                       # regularized fit
    max_plausible_forward_TRY_MWh: float = 1.0e5
    require_all_finite: bool = True
    require_january_flagged: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_abs_monthly_error_TRY_MWh": self.max_abs_monthly_error_TRY_MWh,
            "max_mape_pct": self.max_mape_pct,
            "max_plausible_forward_TRY_MWh": self.max_plausible_forward_TRY_MWh,
            "require_all_finite": self.require_all_finite,
            "require_january_flagged": self.require_january_flagged,
        }


@dataclass
class AcceptanceCheck:
    name: str
    passed: bool
    detail: str

    def row(self) -> Dict[str, Any]:
        return {"check": self.name, "passed": bool(self.passed), "detail": self.detail}


@dataclass
class CalibrationResult:
    """Complete calibration outcome; see module docstring for the two flags."""

    optimizer_success: bool
    calibration_accepted: bool
    valuation_date: str
    quote_source: str
    model_type: str
    curve: ForwardCurve
    model: Optional[ForwardCenteredModel]
    fit_table: pd.DataFrame
    checks: List[AcceptanceCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    expected_spot: Dict[str, float] = field(default_factory=dict)
    parameter_provenance: Dict[str, Any] = field(default_factory=dict)
    january_status: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    finite_checks: Dict[str, bool] = field(default_factory=dict)
    price_label: str = PRICE_LABEL
    tvtp_mode: str = TVTP_MODE_1D
    tvtp_provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed_checks(self) -> List[str]:
        return [c.name for c in self.checks if not c.passed]

    def require_accepted(self) -> None:
        if not self.calibration_accepted:
            raise CalibrationRejected(
                "calibration_accepted is False (failed checks: "
                f"{self.failed_checks}); refusing to price with a rejected "
                "calibration")

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "optimizer_success": bool(self.optimizer_success),
            "calibration_accepted": bool(self.calibration_accepted),
            "valuation_date": self.valuation_date,
            "quote_source": self.quote_source,
            "model_type": self.model_type,
            "price_label": self.price_label,
            "monthly_RMSE": self.metrics.get("monthly_RMSE"),
            "monthly_MAE": self.metrics.get("monthly_MAE"),
            "monthly_MAPE": self.metrics.get("monthly_MAPE"),
            "maximum_absolute_monthly_error": self.metrics.get(
                "maximum_absolute_monthly_error"),
            "finite_value_checks": self.finite_checks,
            "expected_spot_at_72h": self.expected_spot.get("72h"),
            "expected_spot_at_168h": self.expected_spot.get("168h"),
            "expected_spot_at_336h": self.expected_spot.get("336h"),
            "expected_spot_at_720h": self.expected_spot.get("720h"),
            "expected_spot_detail": self.expected_spot,
            "parameter_provenance": self.parameter_provenance,
            "warnings": self.warnings,
            "january_calibration_status": self.january_status,
            "direct_market_constraints": self.curve.constrained_months,
            "extrapolated_periods": self.curve.extrapolated_months,
            "acceptance_checks": [c.row() for c in self.checks],
            "curve_diagnostics": self.curve.diagnostics,
            "tvtp_mode": self.tvtp_mode,
            "tvtp_provenance": self.tvtp_provenance,
        }


# ---------------------------------------------------------------------------
def tvtp_provenance_block(params: FrozenM2Parameters,
                          tvtp2_params: Optional[Any] = None,
                          covariate_path: Optional[Any] = None,
                          params_file: Optional[str] = None) -> Dict[str, Any]:
    """Selected TVTP mode + parameter / covariate-path provenance (both modes)."""
    if tvtp2_params is None:
        out: Dict[str, Any] = {
            "tvtp_mode": TVTP_MODE_1D,
            "status": "production default (single covariate RD_lag1)",
            "parameters_file": params_file or params.source_file,
            "coefficients": {"alpha01": params.alpha01, "gamma01": params.gamma01,
                             "alpha10": params.alpha10, "gamma10": params.gamma10},
            "covariates": ["RD_lag1"],
            "ramp_covariate": "omitted (single-covariate reduction of M9 TVTP-2)",
            "transition_premium": "ASSUMED zero (q^Q = q^P); not calibrated",
        }
    else:
        out = dict(tvtp2_params.provenance_summary())
    if covariate_path is not None:
        out["covariate_path"] = covariate_path.describe()
    return out


def parameter_identification(params: FrozenM2Parameters,
                             spec: ResidualSpec,
                             anchor: NearTermAnchor,
                             r_annual: float,
                             tvtp2_params: Optional[Any] = None,
                             covariate_path: Optional[Any] = None) -> Dict[str, Any]:
    """The four disjoint provenance groups required by the specification.

    ``tvtp2_params`` (two-covariate mode) replaces the TVTP block by the
    EXPERIMENTAL two-covariate coefficients and their provenance tags.
    """
    blob = _parameter_identification_1d(params, spec, anchor, r_annual)
    blob["tvtp_mode"] = TVTP_MODE_1D if tvtp2_params is None else tvtp2_params.tvtp_mode
    if tvtp2_params is None:
        return blob
    c = tvtp2_params.coefficients
    pv = tvtp2_params.provenance
    blob["tvtp_status"] = tvtp2_params.status
    blob["tvtp_label"] = tvtp2_params.label
    blob["2_inherited_from_historical_M2_fit"]["tvtp_coefficients"] = {
        "tvtp_mode": tvtp2_params.tvtp_mode,
        "status": tvtp2_params.status,
        "label": tvtp2_params.label,
        "alpha01": c.alpha01, "gamma01": c.gamma01, "h01": c.h01,
        "alpha10": c.alpha10, "gamma10": c.gamma10, "h10": c.h10,
        "covariates": list(c.covariates),
        "gamma_source": pv.get("tvtp2.gamma01", ""),
        "h_source": pv.get("tvtp2.h01", ""),
        "alpha_source": pv.get("tvtp2.alpha01", ""),
        "parameters_file": tvtp2_params.source_file,
        "verified_reproduction_of_m9": tvtp2_params.verified_reproduction_of_m9,
        "placeholder": bool(tvtp2_params.placeholders),
        "placeholder_fields": list(tvtp2_params.placeholders),
    }
    g3 = blob["3_fixed_by_assumption"]
    g3["tvtp_ramp_covariate_definition"] = {
        "definition": tvtp2_params.ramp_scaler.definition,
        "status": tvtp2_params.ramp_scaler.status,
        "scaler": tvtp2_params.ramp_scaler.as_dict()}
    g3["tvtp_transition_premium"] = tvtp2_params.transition_premium.get(
        "status", "ASSUMED zero transition premium")
    if covariate_path is not None:
        g3["tvtp_covariate_path"] = covariate_path.describe()
    return blob


def _parameter_identification_1d(params: FrozenM2Parameters,
                                 spec: ResidualSpec,
                                 anchor: NearTermAnchor,
                                 r_annual: float) -> Dict[str, Any]:
    return {
        "price_label": PRICE_LABEL,
        "must_not_be_called": FORBIDDEN_LABEL,
        "1_calibrated_from_market_data": {
            "hourly_forward_curve_F_t": {
                "unit": "TRY/MWh",
                "source": "EPİAŞ VEP monthly baseload quotes",
                "constraint": "mean of F over each delivery month = observed quote",
                "identified": True,
            },
        },
        "2_inherited_from_historical_M2_fit": {
            "kappa_per_hour": {"value": params.kappa_per_hour,
                               "source": "AR(1) phi of the historical fit"},
            "sigma_y_normal": {"value": params.sigma_normal,
                               "unit": "per sqrt(hour), y units",
                               "source": ("M9 sigma1 (low-vol state), "
                                          "regime-label swapped to yaml convention")},
            "sigma_y_stress": {"value": params.sigma_stress,
                               "unit": "per sqrt(hour), y units",
                               "source": ("M9 sigma0 (high-vol state), "
                                          "regime-label swapped to yaml convention")},
            "scale_P": {"value": params.scale_P, "unit": "TRY/MWh",
                        "source": ("preprocessing metadata (training median absolute "
                                   "price); source files pde_export.json / "
                                   "prepared_meta.json NOT present in current inputs, "
                                   "value trusted on faith")},
            "tvtp_coefficients": {
                "alpha01": params.alpha01, "gamma01": params.gamma01,
                "alpha10": params.alpha10, "gamma10": params.gamma10,
                "gamma_source": ("M9 transition_coefficients.csv (RD_lag1 covariate "
                                 "only; RD_Ramp_1h_lag1 excluded — omitted-variable "
                                 "risk documented in model_limitations.md), "
                                 "regime-label swapped to yaml convention"),
                "alpha_source": ("DERIVED, not estimated: occupancy/duration-"
                                 "constrained root-finding on the M9 gamma "
                                 "coefficients (see docs/tvtp_derivation_methodology.md); "
                                 "NOT an MLE estimate, no standard error attached"),
                "placeholder": bool(params.placeholders),
                "placeholder_fields": params.placeholders,
            },
            "pi_filtered": {"value": params.pi_filtered.tolist(),
                            "source": "filtered regime probabilities at valuation"},
        },
        "3_fixed_by_assumption": {
            "residual_mode": spec.mode,
            "residual_kappa_per_hour": spec.kappa_per_hour,
            "residual_regime_means_m_i": spec.regime_means.tolist(),
            "residual_x0_mode": spec.x0_mode,
            "sigma_transfer_rule": ("delta method: sigma^X_i(t) = sigma^y_i * "
                                    "sqrt(F(t)^2 + scale_P^2)"),
            "january_anchor_mode": anchor.mode,
            "january_anchor_level_TRY_MWh": anchor.level_TRY_MWh,
            "discount_rate_r_annual": r_annual,
        },
        "4_not_identified_without_option_prices": {
            "volatility_risk_premium": None,
            "sigma_risk_premium": None,
            "eta01_transition_premium": None,
            "eta10_transition_premium": None,
            "regime_market_price_of_risk_lambda_i": None,
            "reason": ("Forward prices constrain only E^Q[P_t], i.e. the first "
                       "moment. Volatility and regime-transition premia enter "
                       "the pricing measure through higher moments and the "
                       "generator, and are not identified by any number of "
                       "forward quotes. Option premia are required."),
            "how_to_identify_later": (
                "With observed option premia, hold F(t) fixed (it is already "
                "matched) and calibrate sigma multipliers plus eta01/eta10 to "
                "the implied-volatility surface: the level of implied vol pins "
                "the sigma premium, its term structure and skew pin the "
                "transition premia."),
        },
    }


# ---------------------------------------------------------------------------
def _model_monthly_averages(model: ForwardCenteredModel,
                            quotes: MarketQuoteSet) -> Dict[str, float]:
    """Baseload average of E^Q[P_h] over each quoted month, from UTC hours.

    Uses the model's own expected-spot map (not the curve directly), so a
    broken centering would show up as a monthly residual rather than be hidden.
    """
    out: Dict[str, float] = {}
    for q in quotes.quotes:
        hrs = q.delivery.hours_utc()
        h = np.asarray((hrs - model.valuation_utc).total_seconds(), dtype=float) / 3600.0
        out[q.label] = float(np.mean(model.expected_spot(h)))
    return out


def run_market_calibration(
    quotes: MarketQuoteSet,
    params: FrozenM2Parameters,
    curve_mode: str = "smooth_constrained",
    anchor: Optional[NearTermAnchor] = None,
    residual_mode: str = "additive",
    residual_kappa_per_hour: Optional[float] = None,
    residual_regime_means: Sequence[float] = (0.0, 0.0),
    residual_x0_mode: str = "zero",
    smoothness_weight: float = 1.0,
    level_weight: float = 1.0e-4,
    criteria: Optional[AcceptanceCriteria] = None,
    r_annual: float = 0.40,
    curve_solver_tolerance_TRY_MWh: float = 1.0e3,
    tvtp2_params: Optional[Any] = None,
    covariate_path: Optional[Any] = None,
) -> CalibrationResult:
    """Build the curve, centre the residual, and run every acceptance check.

    ``curve_solver_tolerance_TRY_MWh`` is a *numerical* guard only: it catches a
    linear solve that has completely failed.  The ECONOMIC judgement lives
    entirely in :class:`AcceptanceCriteria`, so a curve that solves but misses
    the quotes is reported as ``calibration_accepted = False`` rather than
    raising -- exactly the case the specification asks to be distinguished from
    ``optimizer_success``.

    ``tvtp2_params`` selects the EXPERIMENTAL two-covariate TVTP; it then
    requires ``covariate_path`` (a (z, ramp) path covering the curve horizon).
    A non-embeddable path (p01 + p10 >= 1 anywhere) rejects the mode.
    """
    criteria = criteria or AcceptanceCriteria()
    anchor = anchor or NearTermAnchor()
    warnings: List[str] = list(quotes.validate(strict_interior=True))
    checks: List[AcceptanceCheck] = []
    optimizer_success = False
    curve: Optional[ForwardCurve] = None
    model: Optional[ForwardCenteredModel] = None

    # ---- 1. forward curve (the only genuinely calibrated object) ---------
    try:
        curve = build_forward_curve(
            quotes, mode=curve_mode, anchor=anchor,
            spot_price_TRY_MWh=params.spot_price_TRY_MWh,
            smoothness_weight=smoothness_weight, level_weight=level_weight,
            constraint_tolerance_TRY_MWh=curve_solver_tolerance_TRY_MWh)
        optimizer_success = True
        checks.append(AcceptanceCheck(
            "forward_curve_solved", True,
            f"{curve_mode}: {len(curve.values)} hourly nodes"))
    except Exception as exc:
        checks.append(AcceptanceCheck("forward_curve_solved", False, str(exc)))
        raise

    # ---- 2. residual model ------------------------------------------------
    spec = ResidualSpec.from_frozen(
        params, mode=residual_mode,                      # type: ignore[arg-type]
        kappa_per_hour=residual_kappa_per_hour,
        regime_means=residual_regime_means,
        x0_mode=residual_x0_mode)                        # type: ignore[arg-type]
    if tvtp2_params is None:
        tvtp_coef: Any = TVTPCoefficients(params.alpha01, params.gamma01,
                                          params.alpha10, params.gamma10)
    else:
        if covariate_path is None:
            raise ValueError("the two-covariate TVTP calibration needs a (z, ramp) "
                             "covariate_path covering the curve horizon")
        tvtp_coef = tvtp2_params.coefficients
        audit = covariate_path.embeddability(tvtp_coef)
        if not audit["embeddable"]:
            raise EmbeddabilityError(
                f"covariate path '{covariate_path.name}' has {audit['n_s_ge_1']} hour(s) "
                f"with p01 + p10 >= 1 (max s = {audit['max_s']:.4f} at "
                f"{audit['argmax_label']}); the two-covariate continuous-time mode is "
                "REJECTED for this path", audit)
    model = ForwardCenteredModel(
        curve=curve, spec=spec,
        tvtp=tvtp_coef,
        pi_filtered=params.pi_filtered, valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh,
        covariate_lag_hours=params.covariate_lag_hours,
        covariate_path=covariate_path if tvtp2_params is not None else None,
        expected_ramp_scaler=(tvtp2_params.ramp_scaler if tvtp2_params is not None
                              else None))
    if tvtp2_params is not None:
        warnings.append(
            "EXPERIMENTAL two-covariate TVTP (rd_ramp_2d_experimental): "
            f"{tvtp2_params.label}; the RD_Ramp_1h_lag1 definition is reconstructed, "
            "not verified against M9 -- these results do not replace the accepted "
            "single-covariate results")

    if params.has_placeholders:
        warnings.append(
            "TVTP coefficients are PLACEHOLDERS (fields: "
            f"{params.placeholders}); they do not affect the forward-curve "
            "calibration or its acceptance, but option prices computed with "
            "them are not identified from the historical artefacts")

    # ---- 3. monthly fit from UTC delivery hours ---------------------------
    model_avgs = _model_monthly_averages(model, quotes)
    rows = []
    for q in quotes.quotes:
        ma = model_avgs[q.label]
        resid = ma - q.price_TRY_MWh
        rows.append({
            "contract_name": q.contract_name,
            "delivery_start_utc": q.delivery_start_utc.isoformat(),
            "delivery_end_utc": q.delivery_end_utc.isoformat(),
            "number_of_delivery_hours": int(q.n_delivery_hours),
            "market_forward_TRY_MWh": float(q.price_TRY_MWh),
            "model_average_TRY_MWh": float(ma),
            "residual_TRY_MWh": float(resid),
            "relative_error_pct": float(100.0 * resid / q.price_TRY_MWh),
            "acceptance_passed": bool(abs(resid) <= criteria.max_abs_monthly_error_TRY_MWh),
        })
    fit = pd.DataFrame(rows)
    res = fit["residual_TRY_MWh"].to_numpy(dtype=float)
    rel = fit["relative_error_pct"].to_numpy(dtype=float)
    metrics = {
        "monthly_RMSE": float(np.sqrt(np.mean(res ** 2))),
        "monthly_MAE": float(np.mean(np.abs(res))),
        "monthly_MAPE": float(np.mean(np.abs(rel))),
        "maximum_absolute_monthly_error": float(np.max(np.abs(res))),
    }

    checks.append(AcceptanceCheck(
        "monthly_delivery_average_matches_quote",
        metrics["maximum_absolute_monthly_error"] <= criteria.max_abs_monthly_error_TRY_MWh,
        f"max abs monthly error {metrics['maximum_absolute_monthly_error']:.3e} "
        f"TRY/MWh (limit {criteria.max_abs_monthly_error_TRY_MWh})"))
    checks.append(AcceptanceCheck(
        "monthly_MAPE_within_tolerance",
        metrics["monthly_MAPE"] <= criteria.max_mape_pct,
        f"MAPE {metrics['monthly_MAPE']:.3e}% (limit {criteria.max_mape_pct}%)"))

    # ---- 4. finiteness and plausibility -----------------------------------
    cv = curve.values.to_numpy(dtype=float)
    finite_checks = {
        "hourly_forward_curve_finite": bool(np.all(np.isfinite(cv))),
        "monthly_model_averages_finite": bool(np.all(np.isfinite(
            fit["model_average_TRY_MWh"].to_numpy(dtype=float)))),
        "monthly_residuals_finite": bool(np.all(np.isfinite(res))),
    }
    hz = np.array(REPORTING_HORIZONS_HOURS, dtype=float)
    es = model.expected_spot(hz)
    finite_checks["expected_spot_finite"] = bool(np.all(np.isfinite(es)))
    checks.append(AcceptanceCheck(
        "all_values_finite", all(finite_checks.values()),
        f"{sum(finite_checks.values())}/{len(finite_checks)} finiteness checks passed"))

    biggest = float(np.max(np.abs(np.concatenate([cv, es]))))
    checks.append(AcceptanceCheck(
        "no_implausible_forward_magnitude",
        biggest <= criteria.max_plausible_forward_TRY_MWh,
        f"largest |value| {biggest:.6g} TRY/MWh "
        f"(limit {criteria.max_plausible_forward_TRY_MWh:.0g})"))

    # ---- 5. UTC delivery-hour accounting ----------------------------------
    hour_ok = all(int(r["number_of_delivery_hours"]) ==
                  q.delivery.n_hours for r, q in zip(rows, quotes.quotes))
    checks.append(AcceptanceCheck(
        "monthly_average_uses_true_utc_delivery_hours", hour_ok,
        "delivery-hour counts: " + ", ".join(
            f"{q.contract_name}={q.n_delivery_hours}" for q in quotes.quotes)))

    # ---- 6. January flagged as extrapolation ------------------------------
    missing = quotes.missing_months()
    flagged = set(curve.extrapolated_months)
    jan_ok = (not criteria.require_january_flagged) or all(m in flagged for m in missing)
    checks.append(AcceptanceCheck(
        "unquoted_months_flagged_as_extrapolation", jan_ok,
        f"unquoted months {missing}; flagged extrapolated {sorted(flagged)}"))

    january_status = {
        "months_without_observed_quote": missing,
        "flagged_extrapolated": sorted(flagged),
        "anchor_mode": anchor.mode,
        "anchor_level_TRY_MWh": anchor.level_TRY_MWh,
        "anchor_pins_spot": anchor.pins_spot,
        "spot_used_TRY_MWh": params.spot_price_TRY_MWh,
        "status": ("near-term anchored, not directly constrained by an observed "
                   "January VEP quote" if missing else
                   "every month in the horizon carries an observed quote"),
        "how_to_remove": ("add an EBM0126 row to inputs/market/"
                          "vep_monthly_quotes.csv; it becomes a hard monthly "
                          "constraint with no code change"),
    }
    if missing:
        warnings.append(
            f"months {missing} have no observed VEP quote: near-term anchored, "
            "not directly constrained by an observed January VEP quote")

    # ---- 7. expected spot at the reporting horizons -----------------------
    expected_spot: Dict[str, Any] = {}
    for h, v in zip(REPORTING_HORIZONS_HOURS, es):
        ts = params.valuation_utc + pd.Timedelta(hours=int(h))
        loc = to_turkey(ts)
        lab = f"{loc.year:04d}-{loc.month:02d}"
        is_extrap = lab in flagged
        expected_spot[f"{h}h"] = float(v)
        expected_spot[f"{h}h_detail"] = {
            "expected_spot_TRY_MWh": float(v),
            "timestamp_utc": ts.isoformat(),
            "timestamp_turkey": loc.isoformat(),
            "delivery_month": lab,
            "directly_constrained_by_a_VEP_quote": (not is_extrap) and quotes.has(
                loc.year, loc.month),
            "near_term_anchored": bool(is_extrap),
        }
    n_anchored = sum(1 for h in REPORTING_HORIZONS_HOURS
                     if expected_spot[f"{h}h_detail"]["near_term_anchored"])
    if n_anchored:
        warnings.append(
            f"{n_anchored} of {len(REPORTING_HORIZONS_HOURS)} reported horizons "
            f"({', '.join(str(h) + 'h' for h in REPORTING_HORIZONS_HOURS if expected_spot[f'{h}h_detail']['near_term_anchored'])}) "
            "fall in a month WITHOUT an observed VEP quote and are near-term "
            "anchored, not directly constrained by market data")

    # ---- 8. legacy contrast ----------------------------------------------
    warnings.append(EXPLOSION_WARNING)

    accepted = all(c.passed for c in checks)
    result = CalibrationResult(
        optimizer_success=optimizer_success,
        calibration_accepted=bool(accepted),
        valuation_date=params.valuation_utc.isoformat(),
        quote_source=quotes.quote_source,
        model_type="forward_centered",
        curve=curve, model=model, fit_table=fit, checks=checks,
        warnings=warnings, expected_spot=expected_spot,
        parameter_provenance=parameter_identification(
            params, spec, anchor, r_annual, tvtp2_params,
            covariate_path if tvtp2_params is not None else None),
        january_status=january_status, metrics=metrics, finite_checks=finite_checks,
        tvtp_mode=model.tvtp_mode,
        tvtp_provenance=tvtp_provenance_block(
            params, tvtp2_params, covariate_path if tvtp2_params is not None else None),
    )
    if tvtp2_params is not None:
        result.tvtp_provenance["covariate_path_embeddability"] = audit
    logger.info("calibration: optimizer_success=%s calibration_accepted=%s "
                "(max monthly err %.3e TRY/MWh, MAPE %.3e%%)",
                result.optimizer_success, result.calibration_accepted,
                metrics["maximum_absolute_monthly_error"], metrics["monthly_MAPE"])
    if not accepted:
        logger.error("calibration REJECTED; failed checks: %s", result.failed_checks)
    return result


# ---------------------------------------------------------------------------
def near_term_anchor_sensitivity(
    quotes: MarketQuoteSet,
    params: FrozenM2Parameters,
    anchor_levels_TRY_MWh: Sequence[float],
    curve_mode: str = "smooth_constrained",
    option: Optional[EuropeanOption] = None,
    grid_settings: Optional[ResidualGridSettings] = None,
    z_lagged_fn: Optional[Any] = None,
    tvtp2_params: Optional[Any] = None,
    covariate_path: Optional[Any] = None,
    option_covariate_path: Optional[Any] = None,
) -> pd.DataFrame:
    """Sensitivity of near-term results to the anchor level, under the
    production ``spot_to_next_linear`` mode.

    Each row re-solves the curve with the SAME shape as production (a linear
    ramp from the anchor value at t_0 to the first quoted monthly level at
    t_M) but with the anchor value swept over the supplied levels.  This is
    the counterfactual "what if the spot were different by X%": the shape of
    the near-term window is held fixed and only its level moves.  The row
    with ``anchor == params.spot_price_TRY_MWh`` is the only spot-consistent
    case.

    Rationale (see ``outputs/market_calibration_final/archive/near_term_anchor_review/``):
    a prior version used ``explicit_level`` (flat) which (i) collapsed
    ``expected_spot_{72,168,336}h`` to a single number for every swept level,
    hiding the term structure of near-term ES, and (ii) reported
    ``spot_consistent_at_t0=False`` for every non-trivial row, misleadingly
    implying the model was mispricing the spot for every counterfactual.  The
    ramp-based sensitivity preserves the shape actually used in production.

    ``z_lagged_fn`` (optional callable ``t -> z_lagged(t)``) is forwarded to
    the option pricer so the sensitivity uses the SAME climatology path as
    the production ``run_pde.py price`` command; without it the pricer falls
    back to a constant (z=0) exogenous path, which drifts by 1-2% from the
    production benchmark.  ``cmd_calibrate_market`` in ``run_pde.py`` builds
    the production z path once and passes it here.

    ``tvtp2_params`` runs the same sweep under the EXPERIMENTAL two-covariate
    TVTP: ``covariate_path`` (covering the reporting horizons) feeds the
    expected-spot rows and ``option_covariate_path`` (built for the option's
    master grid) the option value.  Every row records ``tvtp_mode``.
    """
    rows: List[Dict[str, Any]] = []
    for lev in anchor_levels_TRY_MWh:
        anchor = NearTermAnchor(mode="spot_to_next_linear")
        curve = build_forward_curve(quotes, mode=curve_mode, anchor=anchor,
                                    spot_price_TRY_MWh=float(lev))
        model = ForwardCenteredModel(
            curve=curve, spec=ResidualSpec.from_frozen(params),
            tvtp=(TVTPCoefficients(params.alpha01, params.gamma01,
                                   params.alpha10, params.gamma10)
                  if tvtp2_params is None else tvtp2_params.coefficients),
            pi_filtered=params.pi_filtered, valuation_utc=params.valuation_utc,
            spot_price_TRY_MWh=float(lev),
            allow_spot_mismatch=True,
            expected_ramp_scaler=(None if tvtp2_params is None
                                  else tvtp2_params.ramp_scaler))
        # E[P_t] = F(t) for any covariate path; the 2D model uses the given path
        hz_arr = np.array(REPORTING_HORIZONS_HOURS, dtype=float)
        es = (model.expected_spot(hz_arr) if tvtp2_params is None
              else model.expected_spot(hz_arr, covariate_path=covariate_path))
        row: Dict[str, Any] = {
            "january_anchor_TRY_MWh": float(lev),
            "anchor_vs_spot_pct": float(100.0 * (lev / params.spot_price_TRY_MWh - 1.0)),
            "max_abs_monthly_error_TRY_MWh": curve.max_abs_monthly_error(),
            "spot_consistent_at_t0": bool(
                abs(float(lev) - params.spot_price_TRY_MWh) < 1e-6),
            "anchor_mode": "spot_to_next_linear",
        }
        for h, v in zip(REPORTING_HORIZONS_HOURS, es):
            row[f"expected_spot_{h}h_TRY_MWh"] = float(v)
        if option is not None:
            pr = (price_forward_centered(model, option, grid_settings,
                                         z_lagged_fn=z_lagged_fn)
                  if tvtp2_params is None else
                  price_forward_centered(model, option, grid_settings,
                                         covariate_path=option_covariate_path))
            row["option_type"] = option.option_type
            row["strike_TRY_MWh"] = float(option.strike)
            row["option_value_TRY_MWh"] = float(pr.value)
        row["tvtp_mode"] = model.tvtp_mode
        rows.append(row)
    df = pd.DataFrame(rows)
    if option is not None and len(df) > 1:
        base = df["option_value_TRY_MWh"].iloc[len(df) // 2]
        df["option_value_pct_vs_mid"] = 100.0 * (df["option_value_TRY_MWh"] / base - 1.0)
    return df


# ---------------------------------------------------------------------------
def calibrated_config(result: CalibrationResult, params: FrozenM2Parameters,
                      curve_path: str, r_annual: float) -> Dict[str, Any]:
    """The YAML config emitted ONLY when calibration_accepted is True."""
    result.require_accepted()
    spec = result.model.spec                                # type: ignore[union-attr]
    return {
        "_generated_by": "run_pde.py calibrate-market",
        "_price_label": PRICE_LABEL,
        "_warning": (f"This is NOT '{FORBIDDEN_LABEL}'. Only the price LEVEL is "
                     "anchored to the market; volatility and regime-risk premia "
                     "remain unidentified."),
        "model": {
            "mode": "forward_centered",
            "residual_mode": spec.mode,
            "residual_kappa_per_hour": float(spec.kappa_per_hour),
            "residual_regime_means": spec.regime_means.tolist(),
            "residual_x0_mode": spec.x0_mode,
        },
        "market": {
            "valuation_utc": result.valuation_date,
            "quote_source": result.quote_source,
            "curve_file": curve_path,
            "curve_mode": result.curve.mode,
            "january_anchor_mode": result.curve.anchor.mode,
            "january_anchor_level_TRY_MWh": result.curve.anchor.level_TRY_MWh,
            "direct_market_constraints": result.curve.constrained_months,
            "extrapolated_periods": result.curve.extrapolated_months,
        },
        "historical_parameters": params.summary(),
        "tvtp": {
            "mode": result.tvtp_mode,
            "status": result.tvtp_provenance.get("status"),
            "parameters_file": result.tvtp_provenance.get("parameters_file"),
            "label": result.tvtp_provenance.get("label"),
        },
        "contract_defaults": {"r_annual": float(r_annual)},
        "acceptance": {
            "optimizer_success": bool(result.optimizer_success),
            "calibration_accepted": bool(result.calibration_accepted),
            "maximum_absolute_monthly_error_TRY_MWh":
                result.metrics["maximum_absolute_monthly_error"],
            "monthly_MAPE_pct": result.metrics["monthly_MAPE"],
        },
    }


def load_calibration_result_json(path: str | Path) -> Dict[str, Any]:
    """Read back calibration_result.json and refuse rejected calibrations."""
    p = Path(path)
    if p.is_dir():
        p = p / "calibration_result.json"
    if not p.exists():
        raise FileNotFoundError(
            f"calibration_result.json not found at {p}; run calibrate-market first")
    with open(p, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    if not bool(blob.get("calibration_accepted", False)):
        raise CalibrationRejected(
            f"{p} reports calibration_accepted=false (failed checks: "
            f"{[c['check'] for c in blob.get('acceptance_checks', []) if not c['passed']]}); "
            "refusing to price against a rejected calibration")
    return blob
