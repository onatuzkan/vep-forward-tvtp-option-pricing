"""Adapter from the uploaded Markov artifacts to PDE-ready inputs.

The uploaded artifacts mix TWO estimation runs (see docs/data_inventory.md):

* TRY run  -- the run this valuation targets.  Its M2_tvtp_TVTP-1 emission
  parameters live in parameter_estimates.csv, its TVTP gammas in
  transition_coefficients.csv, its alphas ONLY in
  metadata/model_parameters_and_ou_mapping.json, and its M0-based export
  (scale_P, seasonal betas, regime price sensitivities) in pde_export.json.
* USD run (markov_usd_final, model M9_student_t_tvtp_TVTP-2) -- this run
  generated the SHIPPED filtered/smoothed/transition series inside
  pde_timeseries.parquet.  Verified empirically: the shipped occupancy
  (0.6746 / 0.3254) equals the USD-run M9 occupancy, and regressing
  logit(p01), logit(p10) on lagged standardized RD and RD-ramp recovers the
  M9 TVTP-2 coefficients to 4-5 significant digits.  Crucially the USD run's
  REGIME LABELS ARE FLIPPED relative to the M2 convention: shipped state 0 is
  the high-volatility state (prob-weighted std of hourly dy: 0.206 vs 0.011).

Consequences implemented here:

1. All PRICING dynamics (emission + TVTP + generator) are rebuilt from the
   authoritative TRY/M2 parameters -- the shipped q series is used only for
   validation and historical diagnostics, with its provenance flagged.
2. The valuation-date regime distribution defaults to the shipped filtered
   probabilities AFTER automatic orientation alignment (high-vol shipped
   state -> M2 regime 1 "stress").  This remains a cross-model PROXY (flagged
   as assumption A1); a manual override is available in the config.
3. The RD standardizer (train mean/std) is not present in any metadata file;
   it is reconstructed exactly on the TRY training window
   (timestamps <= prepared_meta.train_end, reproducing the documented 61361
   training rows) from the shipped raw series, with the ddof convention
   recorded as an assumption.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .data_loader import RawArtifacts
from .generator import TVTPCoefficients, probs_to_generator
from .transformations import (SEASONAL_COLUMNS, OUMapping, PriceTransform,
                              SeasonalModel, ar_to_ou)

logger = logging.getLogger(__name__)

REGIME_NAMES = ("normal", "stress")   # M2 / TRY convention: 1 = high-vol


@dataclass
class RDScaler:
    mean: float
    std: float
    ddof: int
    n_train: int
    definition: str = "RD_WS = Demand_MWh - Wind_MWh - Solar_MWh"

    def standardize(self, rd: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
        return (rd - self.mean) / self.std


@dataclass
class MarkovInputs:
    model_name: str
    transform: PriceTransform
    phi: float
    intercepts: np.ndarray            # c_i, mapping-A interpretation
    sigma_eps: np.ndarray
    ou: OUMapping
    tvtp: TVTPCoefficients
    seasonal: SeasonalModel
    rho: np.ndarray                   # regime price sensitivities (proxy, M0 export)
    rd_scaler: RDScaler
    dt_hours: float
    valuation_utc: pd.Timestamp
    y0: float
    price0: float
    pi_filtered: np.ndarray           # aligned to (normal, stress)
    pi_source: str
    labels_flipped_in_shipped_series: bool
    z_history: pd.Series              # TRY-standardized RD_WS, full sample
    shipped_generator: Optional[pd.DataFrame]
    provenance: dict[str, str] = field(default_factory=dict)
    ambiguities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
def _detect_shipped_orientation(ts: pd.DataFrame, transform: PriceTransform
                                ) -> tuple[bool, dict[str, float]]:
    """True if the shipped state labels are FLIPPED vs the M2 convention.

    M2 convention: regime 1 = high volatility.  We compute the smoothed-
    probability-weighted standard deviation of hourly dy for each shipped
    state; if shipped state 0 is the noisier one, labels are flipped.
    """
    y = np.arcsinh(ts["PTF_TRY_MWh"].to_numpy() / transform.scale_P)
    dy = np.diff(y)
    stats: dict[str, float] = {}
    for s in (0, 1):
        w = ts[f"smooth_p_state{s}"].to_numpy()[1:]
        stats[f"vol_state{s}"] = float(np.sqrt((w * dy**2).sum() / w.sum()))
    flipped = stats["vol_state0"] > stats["vol_state1"]
    return flipped, stats


def _resolve_rd_scaler(raw: RawArtifacts, ddof: int) -> tuple[RDScaler, str]:
    ts = raw.timeseries
    rd = ts["Demand_MWh"] - ts["Wind_MWh"] - ts["Solar_MWh"]
    train_end = pd.Timestamp(raw.prepared_meta["train_end"])
    mask = ts.index <= train_end
    n_train = int(mask.sum())
    expected = int(raw.prepared_meta.get("split_counts", {}).get("train", -1))
    note = (f"reconstructed on TRY training window (<= {train_end}); "
            f"{n_train} rows vs prepared_meta train count {expected}")
    if expected > 0 and n_train != expected:
        logger.warning("TRY train-row reconstruction mismatch: %s", note)
    scaler = RDScaler(
        mean=float(rd[mask].mean()),
        std=float(rd[mask].std(ddof=ddof)),
        ddof=ddof,
        n_train=n_train,
    )
    return scaler, note


def build_markov_inputs(
    raw: RawArtifacts,
    model_name: str = "M2",
    ar_mapping: str = "A_intercept",
    seasonal_mode: str = "mean",
    include_rd_in_theta: bool = True,
    rd_ddof: int = 1,
    pi_override: Optional[tuple[float, float]] = None,
    valuation_utc: Optional[str] = None,
    price0_override: Optional[float] = None,
    holiday_fraction: float = 0.043,
) -> MarkovInputs:
    prov: dict[str, str] = {}
    amb: list[str] = []

    # ---------------------------------------------------------------- scale_P
    scale_export = float(raw.pde_export["price_transform"]["scale_P"])
    scale_meta = float(raw.prepared_meta["scale_P"])
    if abs(scale_export - scale_meta) > 1e-9:
        raise ValueError("scale_P disagrees between pde_export and prepared_meta")
    if raw.pde_export["price_transform"].get("type") != "asinh":
        raise ValueError("price transform type is not asinh")
    transform = PriceTransform(scale_P=scale_export)
    prov["scale_P"] = ("pde_export.json & prepared_meta.json (282.48, training "
                       "median absolute price); inverse P = scale_P*sinh(y) "
                       "per report.md section 3")

    # ------------------------------------------------------ emission (M2 row)
    pe = raw.parameter_estimates
    row = pe.loc[pe["model"] == model_name]
    if row.empty:
        raise ValueError(f"model {model_name!r} not found in parameter_estimates.csv "
                         f"(available: {pe['model'].tolist()})")
    row = row.iloc[0]
    if bool(row.get("ambiguous_labels", False)):
        amb.append(f"parameter_estimates.csv flags ambiguous regime labels for {model_name}")
    phi = float(row["phi"])
    intercepts = np.array([float(row["mu0"]), float(row["mu1"])])
    sigma_eps = np.array([float(row["sigma0"]), float(row["sigma1"])])
    if not sigma_eps[1] > sigma_eps[0]:
        amb.append("sigma1 <= sigma0 in the selected row; regime-1-as-stress "
                   "labelling would be violated")
    prov["phi, c_i, sigma_eps_i"] = (
        f"parameter_estimates.csv, row model={model_name} (TRY run; directly estimated)")

    ou = ar_to_ou(phi, intercepts, sigma_eps, dt_hours=1.0, mapping=ar_mapping)  # type: ignore[arg-type]
    prov["kappa, theta_base, sigma_OU"] = (
        f"transformed from (phi, c_i, sigma_eps_i) via exact AR->OU, mapping "
        f"{ar_mapping}; cross-checked against metadata/model_parameters_and_ou_mapping.json")
    amb.append("AR intercept vs long-run-mean parameterization decided as mapping A "
               "(mu_i are intercepts) from magnitude evidence and the bundle's own "
               "ou_conversion output; the estimation source code was not uploaded (A4)")

    # --------------------------------------------------- TVTP alphas + gammas
    tc = raw.transition_coefficients
    trow = tc.loc[tc["model"] == model_name]
    if trow.empty:
        raise ValueError(f"no transition coefficients for model {model_name!r}")
    if len(trow) > 1:
        amb.append(f"multiple TVTP covariates listed for {model_name}; using the first")
    trow = trow.iloc[0]
    pm = raw.ou_mapping.get("physical_measure_parameters", {})
    alpha01, alpha10 = float(pm["alpha01"]), float(pm["alpha10"])
    gamma01, gamma10 = float(trow["gamma01"]), float(trow["gamma10"])
    for nm, csv_v, json_v in (("gamma01", gamma01, float(pm.get("gamma01", gamma01))),
                              ("gamma10", gamma10, float(pm.get("gamma10", gamma10)))):
        if abs(csv_v - json_v) > 1e-5:
            amb.append(f"{nm} differs between CSV ({csv_v}) and ou_mapping json ({json_v}); "
                       "CSV used as authoritative")
    tvtp = TVTPCoefficients(alpha01=alpha01, gamma01=gamma01,
                            alpha10=alpha10, gamma10=gamma10,
                            covariate=str(trow["covariate"]))
    prov["gamma01, gamma10"] = "transition_coefficients.csv (TRY run, M2; directly estimated)"
    prov["alpha01, alpha10"] = ("metadata/model_parameters_and_ou_mapping.json "
                                "physical_measure_parameters (rounded to 6 dp; the only "
                                "uploaded source of the intercepts)")
    amb.append("alpha01/alpha10 exist only at 6-decimal precision in the bundle "
               "metadata; full-precision values were not exported (A5)")

    # -------------------------------------------- seasonal betas + rho (M0!)
    betas = np.asarray(raw.pde_export["seasonal_beta"], dtype=float)
    rho = np.asarray(raw.pde_export["regime_price_sensitivities"], dtype=float)
    exp_mu = np.asarray(raw.pde_export["regime_mu"], dtype=float)
    export_is_m0 = bool(np.allclose(exp_mu,
                                    pe.loc[pe.model == "M0", ["mu0", "mu1"]].iloc[0].values,
                                    atol=1e-12))
    seasonal = SeasonalModel(betas=betas, mode=seasonal_mode,  # type: ignore[arg-type]
                             holiday_fraction=holiday_fraction)
    prov["seasonal_beta (16)"] = (
        "pde_export.json; the export's regime_mu match the M0 row exactly, so these "
        "betas are the M0 estimates used as a PROXY for M2 (M2 betas were not exported)")
    prov["rho (regime_price_sensitivities)"] = (
        "pde_export.json (M0 export); interpreted as the regime-switching loading on "
        "standardized contemporaneous RD_WS in the emission mean")
    amb.append("pde_export.json is an M0 (constant-transition) export -- its "
               f"seasonal betas and price sensitivities are M0 values "
               f"(export_is_m0={export_is_m0}) used as a proxy for M2 (A2/A3)")
    amb.append("the exact construction of the 16 D_* seasonal columns could not be "
               "reproduced from the uploads; seasonal_mode='mean' uses only the "
               "analytic average of the dummies (A2)")

    # ---------------------------------------------------------- RD standardizer
    rd_scaler, scaler_note = _resolve_rd_scaler(raw, ddof=rd_ddof)
    prov["RD standardizer (mean, std)"] = scaler_note + f"; ddof={rd_ddof} (assumed)"
    amb.append("RD_lag1 train mean/std are absent from covariate_scaling.json; "
               "reconstructed from the shipped series on the documented TRY training "
               "window; ddof convention assumed = 1 (A6)")

    ts = raw.timeseries
    rd = ts["Demand_MWh"] - ts["Wind_MWh"] - ts["Solar_MWh"]
    z_history = pd.Series(rd_scaler.standardize(rd), index=ts.index, name="z_RD_WS")

    # ------------------------------------------------- valuation state and pi
    val_ts = (pd.Timestamp(valuation_utc) if valuation_utc is not None
              else ts.index[-1])
    if val_ts.tzinfo is None:
        val_ts = val_ts.tz_localize("UTC")
    if val_ts not in ts.index:
        loc = ts.index.searchsorted(val_ts, side="right") - 1
        if loc < 0:
            raise ValueError("valuation date precedes the data sample")
        logger.warning("valuation %s not on the hourly grid; using last row at %s",
                       val_ts, ts.index[loc])
        val_ts = ts.index[loc]
    price0 = float(price0_override if price0_override is not None
                   else ts.loc[val_ts, "PTF_TRY_MWh"])
    y0 = float(transform.y_from_price(price0))
    prov["valuation state (P0, y0)"] = (
        f"pde_timeseries.parquet row at {val_ts.isoformat()}"
        + (" (price overridden by config)" if price0_override is not None else ""))

    flipped, volstats = _detect_shipped_orientation(ts, transform)
    p0s = float(ts.loc[val_ts, "filter_p_state0"])
    p1s = float(ts.loc[val_ts, "filter_p_state1"])
    if abs(p0s + p1s - 1.0) > 1e-8:
        raise ValueError("shipped filtered probabilities do not sum to one")
    if pi_override is not None:
        pi = np.asarray(pi_override, dtype=float)
        if pi.shape != (2,) or abs(pi.sum() - 1.0) > 1e-8 or np.any(pi < 0):
            raise ValueError("pi_override must be two non-negative values summing to 1")
        pi_source = "manual override from configuration"
    else:
        pi = np.array([p1s, p0s]) if flipped else np.array([p0s, p1s])
        pi_source = (
            "shipped filtered probabilities at the valuation date, orientation-ALIGNED "
            f"(shipped state 0 vol {volstats['vol_state0']:.3f} > state 1 vol "
            f"{volstats['vol_state1']:.3f} -> shipped labels flipped vs M2 convention); "
            "PROXY: the shipped filter was generated by the USD-run M9 model (A1)")
        if flipped:
            logger.warning(
                "shipped filtered-probability labels are FLIPPED vs the M2 convention; "
                "aligned pi(normal, stress) = (%.4f, %.4f) at %s",
                pi[0], pi[1], val_ts)
    prov["pi_filtered at valuation"] = pi_source
    amb.append("the shipped filtered/smoothed/transition series were generated by the "
               "USD-run model M9_student_t_tvtp_TVTP-2 with flipped regime labels, "
               "not by the TRY M2 model; the aligned filtered distribution is used as "
               "a proxy for M2's filter (A1)")

    shipped_gen = None
    cols = {"transition_p01", "transition_p10",
            "transition_q01_per_hour", "transition_q10_per_hour"}
    if cols.issubset(ts.columns):
        shipped_gen = ts[sorted(cols)].copy()
        prov["shipped q01/q10 per hour"] = (
            "pde_timeseries.parquet q*_per_hour columns; verified to equal the exact "
            "2x2 matrix-log of the shipped p columns to machine precision; PROVENANCE "
            "is the USD-run M9 TVTP-2 model -> used for validation/diagnostics only, "
            "never for M2 pricing")

    inputs = MarkovInputs(
        model_name=str(row["model"]),
        transform=transform,
        phi=phi,
        intercepts=intercepts,
        sigma_eps=sigma_eps,
        ou=ou,
        tvtp=tvtp,
        seasonal=seasonal,
        rho=rho,
        rd_scaler=rd_scaler,
        dt_hours=1.0,
        valuation_utc=val_ts,
        y0=y0,
        price0=price0,
        pi_filtered=pi,
        pi_source=pi_source,
        labels_flipped_in_shipped_series=flipped,
        z_history=z_history,
        shipped_generator=shipped_gen,
        provenance=prov,
        ambiguities=amb,
    )
    _cross_check_against_bundle(inputs, raw)
    return inputs


def _cross_check_against_bundle(inputs: MarkovInputs, raw: RawArtifacts) -> None:
    """Verify our transformed parameters against the bundle's own ou_conversion."""
    oc = raw.ou_mapping.get("ou_conversion", {})
    checks = {
        "kappa_per_hour": (inputs.ou.kappa_per_hour, oc.get("kappa_per_hour")),
        "sigma0_continuous": (inputs.ou.sigma_ou[0], oc.get("sigma0_continuous")),
        "sigma1_continuous": (inputs.ou.sigma_ou[1], oc.get("sigma1_continuous")),
        "theta0(mapA)": (inputs.ou.theta_base[0],
                         oc.get("mapping_A_mu_is_intercept", {}).get("theta0")),
        "theta1(mapA)": (inputs.ou.theta_base[1],
                         oc.get("mapping_A_mu_is_intercept", {}).get("theta1")),
    }
    for name, (mine, theirs) in checks.items():
        if theirs is None:
            continue
        # the bundle json was produced from 7-8 significant-digit inputs
        if not np.isclose(mine, float(theirs), rtol=5e-4, atol=1e-9):
            logger.warning("OU cross-check '%s': package %.9g vs bundle %.9g",
                           name, mine, float(theirs))
        else:
            logger.debug("OU cross-check '%s' OK (%.9g)", name, mine)


def theta_summary(inputs: MarkovInputs) -> dict[str, Any]:
    """Convenience numbers for reports."""
    gbar = inputs.seasonal.mean_value()
    th = inputs.ou.theta_base + gbar / inputs.ou.one_minus_phi
    return {
        "kappa_per_hour": inputs.ou.kappa_per_hour,
        "half_life_hours": inputs.ou.half_life_hours,
        "theta_base": inputs.ou.theta_base.tolist(),
        "seasonal_mean_g": gbar,
        "theta_with_mean_seasonal": th.tolist(),
        "implied_longrun_price_TRY": inputs.transform.price_from_y(th).tolist(),
        "sigma_ou_per_sqrt_hour": inputs.ou.sigma_ou.tolist(),
    }
