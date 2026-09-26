"""Frozen historical M2 parameter set, decoupled from the artefact bundle.

The market-calibration stack must not require the full historical estimation
bundle (parquet time series, parameter CSVs, ...) to run: the forward level now
comes from the market, and only the *residual dynamics* are inherited from the
historical M2 fit.  This module holds those inherited numbers in one small,
explicit YAML file with a provenance tag per field.

Provenance tags
---------------
``estimated``    directly estimated in the historical M2 TVTP run
``transformed``  deterministic function of estimated values (formula recorded)
``assumed``      chosen here, not estimated -- must be reported as an assumption
``placeholder``  NOT available in the supplied bundle; a stand-in value is used
                 and every consumer is warned.  Replace with
                 ``run_pde.py freeze-params`` once the artefacts are present.

Units
-----
``sigma_y_*``       per sqrt(hour), in the transformed variable y = asinh(P/scale_P)
``kappa_per_hour``  per hour
``scale_P``         TRY/MWh
prices              TRY/MWh
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PLACEHOLDER_TAG = "placeholder"

# Provenance-text markers that flag a field as a placeholder even when the
# structured `tvtp.placeholder` flag is False.  Matched case-insensitively as
# whole-word substrings.  Kept small and specific to reduce false positives:
# "assumed"/"trusted on faith" are legitimate provenance qualifiers for values
# that are known but unverifiable, whereas the words below all imply the value
# itself is a stand-in.
_PLACEHOLDER_KEYWORDS: tuple[str, ...] = (
    "placeholder",
    "stand-in",
    "standin",
    "stand in",
    "not included",
    "not present in the bundle",
    "fallback",
)


def _provenance_indicates_placeholder(text: str) -> bool:
    """True if the provenance string contains any placeholder-signalling keyword."""
    if not text:
        return False
    lo = str(text).lower()
    return any(k in lo for k in _PLACEHOLDER_KEYWORDS)


class FrozenParameterError(ValueError):
    """Raised on a malformed or internally inconsistent frozen parameter file."""


@dataclass
class FrozenM2Parameters:
    """Historical two-regime TVTP parameters needed by the residual model."""

    scale_P: float
    phi: float
    kappa_per_hour: float
    sigma_y: np.ndarray                  # (2,) [normal, stress], per sqrt(hour)
    alpha01: float
    gamma01: float
    alpha10: float
    gamma10: float
    pi_filtered: np.ndarray              # (2,) [normal, stress] at valuation
    spot_price_TRY_MWh: float
    valuation_utc: pd.Timestamp
    covariate_lag_hours: float = 1.0
    stationary_pi_stress: Optional[float] = None
    legacy_theta_effective: Optional[float] = None
    m9_stationary_pi: Optional[np.ndarray] = None    # optional alternative for --pi-override
    provenance: Dict[str, str] = field(default_factory=dict)
    placeholders: List[str] = field(default_factory=list)
    source_file: Optional[str] = None

    def __post_init__(self) -> None:
        self.sigma_y = np.asarray(self.sigma_y, dtype=float)
        self.pi_filtered = np.asarray(self.pi_filtered, dtype=float)
        if self.sigma_y.shape != (2,):
            raise FrozenParameterError("sigma_y must have shape (2,)")
        if np.any(self.sigma_y <= 0):
            raise FrozenParameterError("regime volatilities must be positive")
        if self.sigma_y[1] <= self.sigma_y[0]:
            raise FrozenParameterError(
                "regime 1 must be the stress (high-volatility) regime: "
                f"sigma_stress={self.sigma_y[1]} <= sigma_normal={self.sigma_y[0]}")
        if self.pi_filtered.shape != (2,):
            raise FrozenParameterError("pi_filtered must have shape (2,)")
        if np.any(self.pi_filtered < 0) or abs(self.pi_filtered.sum() - 1.0) > 1e-8:
            raise FrozenParameterError("pi_filtered must be a probability vector")
        if self.m9_stationary_pi is not None:
            self.m9_stationary_pi = np.asarray(self.m9_stationary_pi, dtype=float)
            if (self.m9_stationary_pi.shape != (2,)
                    or np.any(self.m9_stationary_pi < 0)
                    or abs(self.m9_stationary_pi.sum() - 1.0) > 1e-6):
                raise FrozenParameterError(
                    "m9_stationary_pi must be a two-vector on the simplex")
        if not 0.0 < self.phi < 1.0:
            raise FrozenParameterError("phi must lie in (0, 1)")
        if self.kappa_per_hour <= 0:
            raise FrozenParameterError("kappa must be positive")
        if self.scale_P <= 0 or self.spot_price_TRY_MWh <= 0:
            raise FrozenParameterError("scale_P and spot must be positive")
        if self.valuation_utc.tzinfo is None:
            raise FrozenParameterError("valuation_utc must be tz-aware")
        implied = -np.log(self.phi)
        if abs(implied - self.kappa_per_hour) > 5e-5 * max(implied, 1e-12):
            logger.warning(
                "kappa (%.8g) differs from -ln(phi) (%.8g) by more than 5e-3%%; "
                "using the supplied kappa", self.kappa_per_hour, implied)
        if self.placeholders:
            logger.warning(
                "frozen parameter set contains %d PLACEHOLDER field(s): %s -- "
                "option prices depending on them are NOT identified from the "
                "supplied artefacts", len(self.placeholders), self.placeholders)

    # -- derived -----------------------------------------------------------
    @property
    def half_life_hours(self) -> float:
        return float(np.log(2.0) / self.kappa_per_hour)

    @property
    def sigma_normal(self) -> float:
        return float(self.sigma_y[0])

    @property
    def sigma_stress(self) -> float:
        return float(self.sigma_y[1])

    def mixture_variance_rate(self, pi: Optional[np.ndarray] = None) -> float:
        """Occupancy-weighted variance rate of y, per hour."""
        w = self.pi_filtered if pi is None else np.asarray(pi, dtype=float)
        return float(w @ (self.sigma_y ** 2))

    @property
    def has_placeholders(self) -> bool:
        return bool(self.placeholders)

    def summary(self) -> Dict[str, Any]:
        return {
            "scale_P_TRY_MWh": self.scale_P,
            "phi": self.phi,
            "kappa_per_hour": self.kappa_per_hour,
            "half_life_hours": self.half_life_hours,
            "sigma_y_normal_per_sqrt_hour": self.sigma_normal,
            "sigma_y_stress_per_sqrt_hour": self.sigma_stress,
            "tvtp_alpha01": self.alpha01, "tvtp_gamma01": self.gamma01,
            "tvtp_alpha10": self.alpha10, "tvtp_gamma10": self.gamma10,
            "pi_filtered_normal": float(self.pi_filtered[0]),
            "pi_filtered_stress": float(self.pi_filtered[1]),
            "spot_price_TRY_MWh": self.spot_price_TRY_MWh,
            "valuation_utc": self.valuation_utc.isoformat(),
            "placeholder_fields": list(self.placeholders),
        }


# ---------------------------------------------------------------------------
def load_frozen_parameters(path: str | Path) -> FrozenM2Parameters:
    """Load the frozen M2 parameter YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"frozen parameter file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        blob = yaml.safe_load(fh)
    if not isinstance(blob, dict):
        raise FrozenParameterError(f"{p.name}: expected a YAML mapping")

    prov: Dict[str, str] = dict(blob.get("provenance", {}) or {})
    # Two placeholder signals, taken as OR:
    #   (1) provenance string starts with the "placeholder" tag OR contains any
    #       of the keywords in _PLACEHOLDER_KEYWORDS.  Catches stand-in fields
    #       whose author forgot to also flip the structured flag.
    #   (2) the structured tvtp.placeholder boolean, which flags all four
    #       TVTP coefficients at once.
    placeholders: list[str] = [
        k for k, v in prov.items()
        if str(v).startswith(PLACEHOLDER_TAG) or _provenance_indicates_placeholder(str(v))
    ]

    def need(key: str) -> Any:
        if key not in blob or blob[key] is None:
            raise FrozenParameterError(f"{p.name}: missing required key {key!r}")
        return blob[key]

    ts = pd.Timestamp(str(need("valuation_utc")))
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    tvtp = blob.get("tvtp", {}) or {}
    for k in ("alpha01", "gamma01", "alpha10", "gamma10"):
        if k not in tvtp:
            raise FrozenParameterError(f"{p.name}: missing tvtp.{k}")
    if bool(tvtp.get("placeholder", False)):
        for k in ("alpha01", "gamma01", "alpha10", "gamma10"):
            tag = f"tvtp.{k}"
            if tag not in placeholders:
                placeholders.append(tag)
                prov.setdefault(tag, f"{PLACEHOLDER_TAG}: TVTP coefficient not "
                                     "present in the supplied bundle")

    params = FrozenM2Parameters(
        scale_P=float(need("scale_P")),
        phi=float(need("phi")),
        kappa_per_hour=float(need("kappa_per_hour")),
        sigma_y=np.array([float(need("sigma_y_normal")), float(need("sigma_y_stress"))]),
        alpha01=float(tvtp["alpha01"]), gamma01=float(tvtp["gamma01"]),
        alpha10=float(tvtp["alpha10"]), gamma10=float(tvtp["gamma10"]),
        pi_filtered=np.asarray(need("pi_filtered"), dtype=float),
        spot_price_TRY_MWh=float(need("spot_price_TRY_MWh")),
        valuation_utc=ts,
        covariate_lag_hours=float(blob.get("covariate_lag_hours", 1.0)),
        stationary_pi_stress=(None if blob.get("stationary_pi_stress") is None
                              else float(blob["stationary_pi_stress"])),
        legacy_theta_effective=(None if blob.get("legacy_theta_effective") is None
                                else float(blob["legacy_theta_effective"])),
        m9_stationary_pi=(None if blob.get("m9_stationary_pi") is None
                          else np.asarray(blob["m9_stationary_pi"], dtype=float)),
        provenance=prov, placeholders=sorted(set(placeholders)), source_file=str(p),
    )
    logger.info("frozen M2 parameters loaded from %s (%d placeholder fields)",
                p.name, len(params.placeholders))
    return params


def freeze_from_artifacts(input_root: str | Path, out_path: str | Path,
                          model_name: str = "M2") -> Path:
    """Regenerate the frozen YAML from a full historical artefact bundle.

    Requires the estimation artefacts (parameter_estimates.csv,
    transition_coefficients.csv, pde_timeseries.parquet, ...).  When they are
    present every ``placeholder`` tag is replaced by ``estimated``.
    """
    from .data_loader import load_artifacts
    from .markov_adapter import build_markov_inputs

    raw = load_artifacts(input_root)
    mi = build_markov_inputs(raw, model_name=model_name)
    stress_occ: Optional[float] = None
    try:
        col = ("smooth_p_state0" if mi.labels_flipped_in_shipped_series
               else "smooth_p_state1")
        stress_occ = float(raw.timeseries[col].mean())
    except Exception:                                        # pragma: no cover
        logger.warning("could not compute stationary stress occupancy")

    blob = {
        "description": (f"Frozen historical {model_name} parameters extracted from a "
                        "full artefact bundle by run_pde.py freeze-params."),
        "model": f"{model_name}_tvtp (Gaussian TVTP, 2 regimes)",
        "scale_P": float(mi.transform.scale_P),
        "phi": float(mi.phi),
        "kappa_per_hour": float(mi.ou.kappa_per_hour),
        "half_life_hours": float(mi.ou.half_life_hours),
        "sigma_y_normal": float(mi.ou.sigma_ou[0]),
        "sigma_y_stress": float(mi.ou.sigma_ou[1]),
        "tvtp": {"alpha01": float(mi.tvtp.alpha01), "gamma01": float(mi.tvtp.gamma01),
                 "alpha10": float(mi.tvtp.alpha10), "gamma10": float(mi.tvtp.gamma10),
                 "covariate": str(mi.tvtp.covariate), "placeholder": False},
        "pi_filtered": [float(mi.pi_filtered[0]), float(mi.pi_filtered[1])],
        "spot_price_TRY_MWh": float(mi.price0),
        "valuation_utc": mi.valuation_utc.isoformat(),
        "covariate_lag_hours": 1.0,
        "stationary_pi_stress": stress_occ,
        "legacy_theta_effective": None,
        "provenance": {k: f"estimated: {v}" for k, v in mi.provenance.items()},
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(blob, fh, sort_keys=False, allow_unicode=True)
    logger.info("frozen parameters written to %s", out)
    return out


# ---------------------------------------------------------------------------
# Two-covariate TVTP parameter set (EXPERIMENTAL, separate YAML)
# ---------------------------------------------------------------------------
@dataclass
class FrozenTVTP2Parameters:
    """Frozen two-covariate TVTP set (RD_lag1 + reconstructed RD_Ramp_1h_lag1).

    Loaded from its OWN file (default ``inputs/historical/tvtp2_frozen_parameters.yaml``);
    the single-covariate ``m2_frozen_parameters.yaml`` is never modified.  Only the
    transition law comes from here: kappa, sigma, pi, spot and the valuation time
    stay those of the base (single-covariate) frozen parameter file.
    """

    coefficients: Any                    # generator.TVTP2Coefficients
    ramp_scaler: Any                     # tvtp2.RampScaler
    lag_hours: int
    tvtp_mode: str
    status: str
    label: str
    parameter_set_name: str
    verified_reproduction_of_m9: bool
    history_file: str
    history_sha256: str
    base_parameters_file: str
    base_parameters_sha256: str
    base_valuation_utc: pd.Timestamp
    derivation: Dict[str, Any] = field(default_factory=dict)
    m9_source: Dict[str, Any] = field(default_factory=dict)
    transition_premium: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)
    placeholders: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    base_parameters_match: Optional[bool] = None

    @property
    def is_experimental(self) -> bool:
        return not self.verified_reproduction_of_m9

    def provenance_summary(self) -> Dict[str, Any]:
        c = self.coefficients
        return {
            "tvtp_mode": self.tvtp_mode,
            "status": self.status,
            "label": self.label,
            "parameter_set_name": self.parameter_set_name,
            "verified_reproduction_of_m9": self.verified_reproduction_of_m9,
            "parameters_file": self.source_file,
            "coefficients": {"alpha01": c.alpha01, "gamma01": c.gamma01, "h01": c.h01,
                             "alpha10": c.alpha10, "gamma10": c.gamma10, "h10": c.h10,
                             "covariates": list(c.covariates)},
            "ramp_scaler": self.ramp_scaler.as_dict(),
            "lag_hours": self.lag_hours,
            "history_file": self.history_file,
            "history_sha256": self.history_sha256,
            "base_parameters_file": self.base_parameters_file,
            "base_parameters_sha256": self.base_parameters_sha256,
            "base_parameters_match": self.base_parameters_match,
            "derivation_sample": {k: self.derivation.get("sample", {}).get(k) for k in (
                "first_valid_transition_utc", "last_valid_transition_utc",
                "n_valid_transitions", "n_dropped_sample_start",
                "n_dropped_missing_observation")},
            "transition_premium": self.transition_premium,
            "provenance": dict(self.provenance),
        }


def load_tvtp2_parameters(path: str | Path,
                          base_params: Optional[FrozenM2Parameters] = None,
                          base_params_path: Optional[str | Path] = None
                          ) -> FrozenTVTP2Parameters:
    """Load and validate the two-covariate TVTP YAML.

    Refuses: a mode other than ``rd_ramp_2d_experimental``; a reconstructed ramp
    labelled as verified; a non-zero (i.e. pretend-calibrated) transition
    premium; scaler windows or derivation samples that end after the valuation
    time of the base parameters (look-ahead leakage).
    """
    from .generator import TVTP2Coefficients, TVTP_MODE_2D
    from .tvtp2 import EXPERIMENTAL_LABEL, STATUS_EXPERIMENTAL, RampScaler, sha256_file

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"two-covariate TVTP parameter file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        blob = yaml.safe_load(fh)
    if not isinstance(blob, dict):
        raise FrozenParameterError(f"{p.name}: expected a YAML mapping")

    def need(d: Dict[str, Any], key: str, where: str) -> Any:
        if not isinstance(d, dict) or key not in d or d[key] is None:
            raise FrozenParameterError(f"{p.name}: missing required key {where}{key!r}")
        return d[key]

    mode = str(need(blob, "tvtp_mode", ""))
    if mode != TVTP_MODE_2D:
        raise FrozenParameterError(f"{p.name}: tvtp_mode must be {TVTP_MODE_2D!r}, got {mode!r}")
    tv = need(blob, "tvtp2", "")
    coef = TVTP2Coefficients(
        alpha01=float(need(tv, "alpha01", "tvtp2.")), gamma01=float(need(tv, "gamma01", "tvtp2.")),
        h01=float(need(tv, "h01", "tvtp2.")), alpha10=float(need(tv, "alpha10", "tvtp2.")),
        gamma10=float(need(tv, "gamma10", "tvtp2.")), h10=float(need(tv, "h10", "tvtp2.")),
        covariates=tuple(tv.get("covariates", ("RD_lag1", "RD_Ramp_1h_lag1"))),
        name=str(blob.get("parameter_set_name", "tvtp2")))
    cov = need(blob, "covariates", "")
    ramp = need(cov, "ramp", "covariates.")
    scaler = RampScaler.from_dict(need(ramp, "scaler", "covariates.ramp."))
    status = str(need(blob, "status", ""))
    verified = bool(blob.get("verified_reproduction_of_m9", False))
    ramp_status = str(ramp.get("status", ""))
    reconstructed = any(w in ramp_status.upper() for w in ("RECONSTRUCT", "ASSUMED"))
    if reconstructed and (verified or status != STATUS_EXPERIMENTAL):
        raise FrozenParameterError(
            f"{p.name}: the ramp covariate is reconstructed/assumed, so the set must be "
            f"status={STATUS_EXPERIMENTAL!r} with verified_reproduction_of_m9: false")
    label = str(need(blob, "label", ""))
    if reconstructed and label != EXPERIMENTAL_LABEL:
        raise FrozenParameterError(f"{p.name}: experimental sets must carry the label "
                                   f"{EXPERIMENTAL_LABEL!r}")
    prem = blob.get("transition_premium", {}) or {}
    if float(prem.get("eta01", 0.0)) != 0.0 or float(prem.get("eta10", 0.0)) != 0.0:
        raise FrozenParameterError(
            f"{p.name}: non-zero transition premia are not supported; eta01 = eta10 = 0 is "
            "the documented zero-premium ASSUMPTION and nothing identifies another value")
    base = need(blob, "base_parameters", "")
    bval = pd.Timestamp(str(need(base, "valuation_utc", "base_parameters.")))
    bval = bval.tz_localize("UTC") if bval.tzinfo is None else bval.tz_convert("UTC")
    der = blob.get("derivation", {}) or {}
    ends = [("covariates.ramp.scaler.window_end_utc", scaler.window_end_utc)]
    last = (der.get("sample", {}) or {}).get("last_valid_transition_utc")
    if last:
        ends.append(("derivation.sample.last_valid_transition_utc", pd.Timestamp(str(last))))
    for nm, ts in ends:
        if ts > bval:
            raise FrozenParameterError(f"{p.name}: {nm} = {ts} is after the valuation time "
                                       f"{bval} (look-ahead leakage)")
    if base_params is not None and base_params.valuation_utc != bval:
        raise FrozenParameterError(
            f"{p.name}: derived for valuation {bval}, but the base parameters value at "
            f"{base_params.valuation_utc}")
    prov: Dict[str, str] = {str(k): str(v) for k, v in (blob.get("provenance", {}) or {}).items()}
    placeholders = [k for k, v in prov.items()
                    if str(v).startswith(PLACEHOLDER_TAG) or _provenance_indicates_placeholder(v)]
    base_file = str(base.get("file", ""))
    base_sha = str(base.get("sha256", ""))
    match: Optional[bool] = None
    if base_params_path is not None and Path(base_params_path).exists() and base_sha:
        match = sha256_file(base_params_path) == base_sha
        if not match:
            logger.warning("%s was derived against %s with sha256 %s..., but the base "
                           "parameter file in use has a different hash; kappa/sigma/pi come "
                           "from the file in use", p.name, base_file, base_sha[:12])
    out = FrozenTVTP2Parameters(
        coefficients=coef, ramp_scaler=scaler, lag_hours=int(cov.get("lag_hours", 1)),
        tvtp_mode=mode, status=status, label=label,
        parameter_set_name=str(blob.get("parameter_set_name", "")),
        verified_reproduction_of_m9=verified,
        history_file=str(cov.get("history_file", "")),
        history_sha256=str(cov.get("history_sha256", "")),
        base_parameters_file=base_file, base_parameters_sha256=base_sha,
        base_valuation_utc=bval, derivation=der,
        m9_source=blob.get("m9_source", {}) or {},
        transition_premium=prem, provenance=prov, placeholders=placeholders,
        source_file=str(p), base_parameters_match=match)
    if placeholders:
        logger.warning("two-covariate TVTP set contains PLACEHOLDER field(s): %s", placeholders)
    logger.info("two-covariate TVTP parameters loaded from %s (%s)", p.name, status)
    return out
