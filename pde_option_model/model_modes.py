"""Model-mode registry and the calibration output writer.

Two modes are supported and kept strictly apart.

``legacy_asinh_ou``
    The original model, P = scale_P * sinh(y) with a Gaussian regime-switching
    OU in y.  PRESERVED unchanged and still fully runnable, but restricted to
    benchmarking and short-horizon diagnostics: its long-horizon expected price
    inherits the factor exp(v(t)/2) and is flagged wherever it appears.

``forward_centered``
    The market pricing model and the DEFAULT for market work.  Prices live
    around the EPİAŞ VEP hourly forward curve; regimes describe the residual
    only, so the forward level and the residual regime volatility are separated.

Nothing here mutates legacy outputs: every artefact is written under a new
directory supplied by the caller.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .forward_centered import ForwardCenteredModel
from .legacy_moments import (EXPLOSION_WARNING, legacy_expected_spot,
                             legacy_explosion_report, load_legacy_reference)
from .market_calibration import (PRICE_LABEL, REPORTING_HORIZONS_HOURS,
                                 CalibrationResult, calibrated_config)
from .params_frozen import FrozenM2Parameters

logger = logging.getLogger(__name__)

MODEL_MODES = ("legacy_asinh_ou", "forward_centered")
DEFAULT_MARKET_MODE = "forward_centered"

MODE_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "legacy_asinh_ou": {
        "role": "benchmark and short-horizon diagnostics only",
        "state_variable": "y = asinh(P / scale_P)",
        "price_map": "P = scale_P * sinh(y)",
        "long_horizon_flag": EXPLOSION_WARNING,
        "market_anchored": False,
    },
    "forward_centered": {
        "role": "default market pricing model",
        "state_variable": "x = residual around the market forward curve (TRY/MWh)",
        "price_map": "P_t = F(t) + X_t - mu_X(t)",
        "long_horizon_flag": ("finite moments by construction; E^Q[P_t] = F(t) "
                              "for every t"),
        "market_anchored": True,
    },
}


def validate_model_mode(mode: str) -> str:
    if mode not in MODEL_MODES:
        raise ValueError(f"unknown model mode {mode!r}; choose from {MODEL_MODES}")
    return mode


def _ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_monthly_fit(result: CalibrationResult, outpath: Path) -> None:
    fit = result.fit_table
    curve = result.curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8),
                             gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(curve.values.index, curve.values.to_numpy(), lw=0.7, color="#1f77b4",
            label=f"hourly forward F(t) [{curve.mode}]")
    for _, r in fit.iterrows():
        s = pd.Timestamp(r["delivery_start_utc"])
        e = pd.Timestamp(r["delivery_end_utc"])
        ax.hlines(r["market_forward_TRY_MWh"], s, e, color="#d62728", lw=2.4,
                  zorder=5)
        ax.text(s + (e - s) / 2, r["market_forward_TRY_MWh"],
                r["contract_name"], ha="center", va="bottom", fontsize=7,
                color="#d62728")
    ax.hlines([], [], [], color="#d62728", lw=2.4, label="observed VEP monthly quote")
    anch = curve.frame["near_term_anchor_flag"].to_numpy(dtype=bool)
    if anch.any():
        ax.axvspan(curve.values.index[0], curve.values.index[anch.sum() - 1],
                   color="orange", alpha=0.16,
                   label="near-term anchored (no VEP quote)")
    ax.set_ylabel("TRY/MWh")
    ax.set_title("Forward-centered model: hourly curve vs monthly VEP quotes")
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    x = np.arange(len(fit))
    ax.bar(x, fit["residual_TRY_MWh"].to_numpy(), color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(fit["contract_name"], rotation=45, fontsize=8)
    ax.set_ylabel("model average − quote (TRY/MWh)")
    ax.axhline(0, color="k", lw=0.7)
    mx = float(np.abs(fit["residual_TRY_MWh"].to_numpy()).max())
    ax.set_title(f"Monthly delivery-average residuals\nmax |error| = {mx:.2e} TRY/MWh")
    fig.suptitle(PRICE_LABEL, fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_legacy_vs_forward_centered(result: CalibrationResult,
                                    params: FrozenM2Parameters,
                                    legacy_reference_path: Optional[Path],
                                    outpath: Path) -> pd.DataFrame:
    model: ForwardCenteredModel = result.model            # type: ignore[assignment]
    # Reach past the first quoted delivery months so the VEP anchors are visible.
    # The legacy analytic curve turns negative once m(t) crosses zero (~1780 h
    # with the fitted theta), so the window stops before that.
    horizon = float(min(1750.0, (model.curve.end_utc -
                                 model.valuation_utc).total_seconds() / 3600.0))
    h = np.concatenate([np.arange(1.0, 72.0, 1.0),
                        np.arange(72.0, horizon + 1.0, 6.0)])
    fc = model.expected_spot(h)

    theta = params.legacy_theta_effective
    pis = params.stationary_pi_stress
    legacy = None
    if theta is not None and pis is not None:
        y0 = float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P))
        with np.errstate(over="ignore"):
            legacy = legacy_expected_spot(h, params.scale_P, y0, theta,
                                          params.kappa_per_hour, params.sigma_y, pis)

    ref = None
    if legacy_reference_path is not None and Path(legacy_reference_path).exists():
        ref = load_legacy_reference(legacy_reference_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, logy in ((axes[0], True), (axes[1], False)):
        if legacy is not None:
            vis = np.where(legacy > 0, legacy, np.nan)
            ax.plot(h, vis, color="#d62728", lw=1.6,
                    label="legacy_asinh_ou  E[P_t] (analytic sinh-Gaussian)")
        if ref is not None:
            ax.plot(ref["horizon_hours"], ref["expected_spot_TRY_MWh"], "o",
                    color="#8b0000", ms=7, label="legacy: reported model output")
        ax.plot(h, fc, color="#1f77b4", lw=2.0,
                label="forward_centered  E[P_t] = F(t)")
        ax.axhline(params.spot_price_TRY_MWh, color="gray", ls=":", lw=1,
                   label=f"spot {params.spot_price_TRY_MWh:.0f}")
        for _, r in result.fit_table.iterrows():
            s = pd.Timestamp(r["delivery_start_utc"])
            e = pd.Timestamp(r["delivery_end_utc"])
            hs = (s - model.valuation_utc).total_seconds() / 3600.0
            he = (e - model.valuation_utc).total_seconds() / 3600.0
            if hs <= h[-1]:
                ax.hlines(r["market_forward_TRY_MWh"], max(hs, 0), min(he, h[-1]),
                          color="#2ca02c", lw=3.0, zorder=6)
                ax.text(max(hs, 0) + 0.5 * (min(he, h[-1]) - max(hs, 0)),
                        r["market_forward_TRY_MWh"], r["contract_name"],
                        ha="center", va="bottom", fontsize=7, color="#2ca02c",
                        zorder=7)
        ax.hlines([], [], [], color="#2ca02c", lw=2.6, label="observed VEP quotes")
        if logy:
            ax.set_yscale("log")
            ax.set_title("log scale — the legacy explosion")
        else:
            qmax = float(result.fit_table["market_forward_TRY_MWh"].max())
            ax.set_ylim(0, 1.25 * max(float(np.max(fc)), qmax,
                                      params.spot_price_TRY_MWh))
            ax.set_title("linear scale — market-consistent range "
                         "(legacy leaves the axis immediately)")
        ax.set_xlabel("horizon (hours from 2025-12-31)")
        ax.set_ylabel("E[P_t]  (TRY/MWh)")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Legacy sinh-Gaussian vs forward-centered expected spot", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows: List[Dict[str, Any]] = []
    for hh in REPORTING_HORIZONS_HOURS:
        j = int(np.argmin(np.abs(h - hh)))
        rows.append({
            "horizon_hours": hh,
            "forward_centered_TRY_MWh": float(fc[j]),
            "legacy_analytic_TRY_MWh": (None if legacy is None else float(legacy[j])),
            "legacy_reported_TRY_MWh": (
                None if ref is None else
                float(ref.loc[ref["horizon_hours"] == hh, "expected_spot_TRY_MWh"].iloc[0])
                if (ref["horizon_hours"] == hh).any() else None),
        })
    return pd.DataFrame(rows)


def plot_anchor_sensitivity(df: pd.DataFrame, spot: float, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for h in REPORTING_HORIZONS_HOURS:
        col = f"expected_spot_{h}h_TRY_MWh"
        if col in df:
            ax.plot(df["january_anchor_TRY_MWh"], df[col], marker="o", ms=4,
                    label=f"E[P] at {h}h")
    ax.axvline(spot, color="gray", ls=":", label=f"spot {spot:.0f}")
    ax.set_xlabel("January 2026 anchor level (TRY/MWh)")
    ax.set_ylabel("expected spot (TRY/MWh)")
    ax.set_title("Near-term anchor → expected spot")
    ax.legend(fontsize=8)

    ax = axes[1]
    if "option_value_TRY_MWh" in df:
        ax.plot(df["january_anchor_TRY_MWh"], df["option_value_TRY_MWh"],
                marker="s", color="#d62728")
        ax.set_ylabel("option value (TRY/MWh)")
        ttl = "Near-term anchor → 72h option value"
        if "strike_TRY_MWh" in df:
            ttl += f"\n({df['option_type'].iloc[0]}, K={df['strike_TRY_MWh'].iloc[0]:.0f})"
        ax.set_title(ttl)
    else:
        ax.plot(df["january_anchor_TRY_MWh"], df["max_abs_monthly_error_TRY_MWh"],
                marker="s")
        ax.set_ylabel("max abs monthly error (TRY/MWh)")
        ax.set_title("Quoted months stay matched for every anchor")
    ax.axvline(spot, color="gray", ls=":")
    ax.set_xlabel("January 2026 anchor level (TRY/MWh)")
    fig.suptitle("January 2026 is an ASSUMPTION, not a market constraint", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
@dataclass
class CalibrationOutputs:
    directory: Path
    files: List[str]
    accepted: bool


def write_calibration_outputs(
    result: CalibrationResult,
    params: FrozenM2Parameters,
    outdir: str | Path,
    anchor_sensitivity: Optional[pd.DataFrame] = None,
    legacy_reference_path: Optional[str | Path] = None,
    r_annual: float = 0.40,
    extra_notes: Optional[Dict[str, Any]] = None,
) -> CalibrationOutputs:
    """Write every required artefact.

    ``calibrated_config.yaml`` is emitted ONLY when the calibration was
    accepted; a rejected run still writes the full audit trail so the failure
    is inspectable.
    """
    d = _ensure(Path(outdir))
    files: List[str] = []

    def rec(name: str) -> Path:
        files.append(name)
        return d / name

    with open(rec("calibration_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2, ensure_ascii=False)

    result.curve.frame.to_csv(rec("hourly_forward_curve.csv"), index=False)
    result.fit_table.to_csv(rec("monthly_forward_fit.csv"), index=False)

    with open(rec("parameter_identification.json"), "w", encoding="utf-8") as fh:
        json.dump(result.parameter_provenance, fh, indent=2, ensure_ascii=False)

    plot_monthly_fit(result, rec("monthly_forward_fit.png"))
    cmp_table = plot_legacy_vs_forward_centered(
        result, params,
        Path(legacy_reference_path) if legacy_reference_path else None,
        rec("legacy_vs_forward_centered.png"))
    cmp_table.to_csv(rec("legacy_vs_forward_centered.csv"), index=False)

    if anchor_sensitivity is not None:
        anchor_sensitivity.to_csv(rec("near_term_anchor_sensitivity.csv"), index=False)
        plot_anchor_sensitivity(anchor_sensitivity, params.spot_price_TRY_MWh,
                                rec("near_term_anchor_sensitivity.png"))

    with open(rec("calibration_audit.md"), "w", encoding="utf-8") as fh:
        fh.write(_audit_markdown(result, params, cmp_table, anchor_sensitivity,
                                 extra_notes or {}))
    with open(rec("model_limitations.md"), "w", encoding="utf-8") as fh:
        fh.write(_limitations_markdown(result, params))

    if result.calibration_accepted:
        cfg = calibrated_config(result, params, "hourly_forward_curve.csv", r_annual)
        with open(rec("calibrated_config.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    else:
        logger.error("calibration REJECTED -> calibrated_config.yaml NOT written")
        with open(rec("CALIBRATION_REJECTED.txt"), "w", encoding="utf-8") as fh:
            fh.write("calibration_accepted = false\n\nFailed checks:\n" +
                     "\n".join(f"  - {n}" for n in result.failed_checks) +
                     "\n\nNo calibrated_config.yaml was produced. The price "
                     "command will refuse this directory.\n")

    logger.info("wrote %d files to %s", len(files), d)
    return CalibrationOutputs(directory=d, files=files,
                              accepted=result.calibration_accepted)


def _audit_markdown(result: CalibrationResult, params: FrozenM2Parameters,
                    cmp_table: pd.DataFrame,
                    anchor_sens: Optional[pd.DataFrame],
                    extra: Dict[str, Any]) -> str:
    m = result.metrics
    lines = [
        "# Calibration audit — forward-centered model",
        "",
        f"**Result label:** {PRICE_LABEL}",
        "",
        "> This is *not* “Fully market-calibrated option prices”. Only the price "
        "level is anchored to the VEP curve; the volatility and regime-transition "
        "risk premia remain unidentified because no option premia were supplied.",
        "",
        "## 1. Outcome",
        "",
        "| field | value |",
        "|---|---|",
        f"| `optimizer_success` | `{result.optimizer_success}` |",
        f"| `calibration_accepted` | `{result.calibration_accepted}` |",
        f"| valuation date | {result.valuation_date} |",
        f"| quote source | {result.quote_source} |",
        f"| model type | {result.model_type} |",
        f"| curve mode | {result.curve.mode} |",
        f"| monthly RMSE | {m['monthly_RMSE']:.6e} TRY/MWh |",
        f"| monthly MAE | {m['monthly_MAE']:.6e} TRY/MWh |",
        f"| monthly MAPE | {m['monthly_MAPE']:.6e} % |",
        f"| max abs monthly error | {m['maximum_absolute_monthly_error']:.6e} TRY/MWh |",
        "",
        "These two flags are deliberately independent: a converged optimizer with "
        "an economically absurd fit is a **failed** calibration.",
        "",
        "## 2. Acceptance checks",
        "",
        "| check | passed | detail |",
        "|---|---|---|",
    ]
    for c in result.checks:
        lines.append(f"| `{c.name}` | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")

    lines += ["", "## 3. Monthly delivery-average fit", "",
              "Averages are formed over the true UTC delivery hours of each "
              "Turkish local delivery month.", "",
              result.fit_table.to_markdown(index=False), ""]

    lines += ["## 4. Expected spot at the reporting horizons", "",
              "| horizon | E[P_t] (TRY/MWh) | delivery month | directly constrained by a VEP quote? |",
              "|---|---|---|---|"]
    for h in REPORTING_HORIZONS_HOURS:
        det = result.expected_spot[f"{h}h_detail"]
        lines.append(f"| {h} h | {det['expected_spot_TRY_MWh']:.2f} | "
                     f"{det['delivery_month']} | "
                     f"{'yes' if det['directly_constrained_by_a_VEP_quote'] else '**no — near-term anchored**'} |")

    lines += ["", "## 5. Legacy vs forward-centered", "",
              cmp_table.to_markdown(index=False), "",
              "The legacy column is the analytic sinh-Gaussian moment "
              "`E[P] = scale_P · exp(v/2) · sinh(m)`; the reported column is the "
              "output actually observed from the legacy run.", ""]

    if anchor_sens is not None:
        lines += ["## 6. January anchor sensitivity", "",
                  anchor_sens.to_markdown(index=False), "",
                  "Every row reproduces the six quoted months exactly: the January "
                  "assumption moves only the unconstrained near-term window.", ""]

    lines += ["## 7. Warnings", ""]
    lines += [f"- {w}" for w in result.warnings] or ["- none"]
    if extra:
        lines += ["", "## 8. Run notes", ""]
        lines += [f"- **{k}**: {v}" for k, v in extra.items()]
    lines += _anchor_selection_section()
    return "\n".join(lines) + "\n"


def _anchor_selection_section() -> list[str]:
    """§9 documenting the accepted near-term anchor mode.

    Content preserved across regeneration of calibration_audit.md; supporting
    evidence lives permanently under
    ``outputs/market_calibration_final/archive/near_term_anchor_review/``.
    """
    return [
        "",
        "## 9. Near-term anchor mode selection",
        "",
        "The four `NearTermAnchor` modes defined in "
        "`pde_option_model/forward_curve.py` were compared on the same six-quote "
        "VEP strip; `spot_to_next_linear` is the production choice. Supporting "
        "evidence lives in `archive/near_term_anchor_review/` (three full "
        "alternative calibrations, jump audits, and the OLD flat-method "
        "sensitivity that §6 supersedes).",
        "",
        "### Mode comparison",
        "",
        "| Mode | F(t₀) | mean F over Jan | F(first Feb hour) | Jan→Feb jump | K=3000 call, 72h |",
        "|---|---:|---:|---:|---:|---:|",
        "| `spot_flat` | 2917.78 | 2917.71 | 2909.99 | −0.53 | (reference: archive) |",
        "| **`spot_to_next_linear`** (production) | **2917.78** | **2909.40** | **2902.02** | **+0.05** | see §6 |",
        "| `flat_next_month` | 2900.99 | 2901.00 | 2901.94 | +0.07 | (reference: archive) |",
        "| `explicit_level` | user-supplied `L` | `L` (pre-smoothing) | ≈`L` | depends on `L` | depends on `L` |",
        "",
        "All three concrete alternatives pass the acceptance thresholds (monthly "
        "RMSE ∼ 3e-12 TRY/MWh, MAPE ∼ 1e-13 %); the differences live entirely in "
        "the unquoted January window.",
        "",
        "### Hourly-jump audit (|ΔF| > 5 TRY/MWh across the full horizon)",
        "",
        "Aggregate counts are identical across the three modes because the "
        "anchor rule only touches January and every mode shares the same "
        "Feb-onwards KKT solution:",
        "",
        "| metric | value |",
        "|---|---:|",
        "| total jumps > 5 TRY/MWh | 113 |",
        "| at a true month boundary | 3 |",
        "| within one day of a month boundary (smoother ramp) | 110 |",
        "| inside the January (unquoted, anchor-driven) window | **0** |",
        "| max |ΔF| inside January | 0.05 TRY/MWh |",
        "",
        "The three boundary jumps (Feb→Mar −12.3, May→Jun −9.6, Jun→Jul +47.0) "
        "reflect the underlying monthly quote steps; the intra-month jumps are "
        "the smoother's ramp-in/-out of those boundaries. No anomalous jump "
        "appears in a region unrelated to a boundary.",
        "",
        "### Why `spot_to_next_linear` was selected",
        "",
        "1. **Spot consistency at t = 0.** `F(t₀) = spot` holds exactly, so the "
        "model satisfies `E^Q[P₀] = spot`. `flat_next_month` violates this by "
        "17 TRY/MWh and triggers a spot-mismatch warning.",
        "2. **Smooth handover to the constrained region.** The Jan→Feb boundary "
        "jump is +0.05 TRY/MWh versus −0.53 under `spot_flat` (whose implicit "
        "January baseload of 2917.71 TRY/MWh is 17 TRY above the earliest "
        "quoted month, unsupported by market data).",
        "3. **Term structure of near-term expected spot.** Under the ramp the "
        "four reporting horizons receive distinct levels (72 h → 2916.16 down "
        "to 720 h → 2901.51); `explicit_level` collapses them to one number.",
        "4. **Interpretable counterfactual.** §6's sensitivity now sweeps the "
        "anchor level under the production shape, so it is a valid \"what if "
        "the near-term level were X\" analysis rather than an artefact of a "
        "different anchor rule.",
        "",
        "### Sensitivity table alignment with the main benchmark",
        "",
        "As of the M9 integration follow-up, `near_term_anchor_sensitivity` "
        "receives the same climatology `z(t-1)` path that `run_pde.py price` "
        "builds from `inputs/historical/rd_standardized.csv` (train_end "
        "controlled by `scenario.train_end_utc` in the config). The base "
        "row (0% anchor shift) of §6 therefore reproduces the main 72h "
        "benchmark to solver precision (677.23 TRY/MWh) instead of the "
        "constant-z fallback that used to drift by ~1-2%. This makes the "
        "sensitivity table's absolute levels directly comparable to the "
        "`price` command output, not just the relative % vs base column.",
        "",
        "### Weakness acknowledged",
        "",
        "The linear-ramp choice is arbitrary — nothing in the market data says "
        "January should interpolate LINEARLY between spot and the Feb baseload. "
        "If EPİAŞ ever publishes an EBM0126 quote, January flips from "
        "anchor-derived to hard-constrained (see "
        "`january_calibration_status.how_to_remove` in "
        "`calibration_result.json`).",
    ]


def _limitations_markdown(result: CalibrationResult,
                          params: FrozenM2Parameters) -> str:
    model: ForwardCenteredModel = result.model            # type: ignore[assignment]
    hz = np.array([72.0, 168.0, 336.0, 720.0])
    mean, var, p_stress, fwd = model.moments_at(hz)
    sd = np.sqrt(var)
    jan = result.january_status
    lines = [
        "# Model limitations and assumption inventory",
        "",
        f"Label of every price produced here: **{PRICE_LABEL}**.",
        "",
        "## What is genuinely constrained by the market",
        "",
        "| quantity | status |",
        "|---|---|",
        f"| hourly forward level F(t) inside {', '.join(result.curve.constrained_months)} | "
        "**constrained** — monthly averages reproduce the VEP quotes exactly |",
        f"| price level in {', '.join(result.curve.extrapolated_months) or '(none)'} | "
        "**assumed** — near-term anchored, no observed quote |",
        "| residual volatility (σ_normal, σ_stress) | **inherited** from the historical M2 fit, not market-implied |",
        "| regime transition dynamics (TVTP) | **inherited** from the historical fit |",
        "| volatility risk premium | **not identified** — needs option premia |",
        "| regime-transition premia η01, η10 | **not identified** — needs option premia |",
        "| market price of risk λ_i | **not identified** — needs option premia |",
        "| discount rate | **assumed** flat annual rate (0.40); sensitivity swept in `discount_rate_sensitivity.md` — <2 % impact across `r_annual ∈ [0.15, 0.60]` at maturities ≤ 336 h |",
        "",
        "## Results that rest on an assumption, not on data",
        "",
        f"1. **January 2026 (`{jan['anchor_mode']}` anchor).** {jan['status']}. "
        "Every horizon below falls inside it, so all four reported expected "
        "spots are anchored rather than market-constrained:",
        "",
    ]
    for h in REPORTING_HORIZONS_HOURS:
        det = result.expected_spot[f"{h}h_detail"]
        tag = ("near-term anchored" if det["near_term_anchored"]
               else "constrained by a VEP quote")
        lines.append(f"   - {h} h → {det['delivery_month']} — {tag}")
    lines += [
        "",
        "2. **Option prices depend on inherited volatility.** The forward "
        "calibration is exactly invariant to σ (the centering ODE has no σ term), "
        "so a wrong σ cannot break the monthly fit — but it moves every option "
        "value one-for-one. Option prices are therefore *level-anchored, "
        "volatility-assumed*.",
        "",
        "3. **Residual dispersion inherits a near-unit-root κ.** With "
        f"κ = {params.kappa_per_hour:.3e}/h (half-life {params.half_life_hours:.0f} h) "
        "the residual standard deviation keeps growing over the horizon:",
        "",
        "| horizon | F(t) (TRY/MWh) | residual sd (TRY/MWh) | sd / F |",
        "|---|---|---|---|",
    ]
    for h, f, s in zip(hz, fwd, sd):
        lines.append(f"| {int(h)} h | {f:.1f} | {s:.1f} | {s / f:.2f} |")
    lines += [
        "",
        "   In the **additive** residual mode this admits negative simulated "
        "prices at long horizons — economically wrong for PTF, which is floored "
        "at zero. The additive mode is appropriate at day-ahead to few-week "
        "horizons; use `residual_mode: multiplicative` for month-scale work, "
        "where prices stay positive by construction. Either way the monthly "
        "forward fit is unaffected.",
        "",
        f"4. **The regime split is historical, not market-implied.** Regimes here "
        "describe the *residual* around the market curve, not the price level: "
        "regime 0 is the calm state in which the spot tracks the forward closely, "
        f"regime 1 the stress state with ~{params.sigma_stress / params.sigma_normal:.0f}× "
        "the residual volatility. Nothing in the VEP data identifies how the "
        "market prices that stress risk.",
        "",
    ]
    if params.has_placeholders:
        lines += [
            f"5. **Placeholder parameters in use:** `{'`, `'.join(params.placeholders)}`. "
            "The supplied bundle did not contain the fitted TVTP coefficients. "
            "They do not enter the forward calibration or its acceptance "
            "criteria, but they do shape option prices. Regenerate with "
            "`run_pde.py freeze-params --input-root inputs`.",
            "",
        ]
    lines += [
        "## Legacy model",
        "",
        f"> {EXPLOSION_WARNING}",
        "",
        "`legacy_asinh_ou` remains available and unchanged for short-horizon "
        "benchmarking, and its long-horizon output is flagged everywhere it is "
        "produced.",
        "",
        "## Parameter-provenance caveats (post-M9-integration)",
        "",
        "Everything below is a property of `inputs/historical/m2_frozen_parameters.yaml` "
        "after the M9-fit integration.  These are not calibration failures — they "
        "are unavoidable consequences of what the uploaded calibration bundle "
        "contained (and did not contain). Full derivation trail in "
        "`docs/tvtp_derivation_methodology.md`.",
        "",
        "### (a) `tvtp.alpha01`, `tvtp.alpha10` are DERIVED, not estimated",
        "",
        "The uploaded M9 bundle exports the two `gamma` slopes for `RD_lag1` but "
        "no intercepts. The current yaml intercepts "
        f"(`alpha01 = {params.alpha01:.4f}`, `alpha10 = {params.alpha10:.4f}`) were "
        "back-solved by root-finding on the reported occupancy / mean-duration "
        "diagnostics of the same M9 run:",
        "",
        "* `E_z[σ(alpha01 + gamma01·z)] = 1 / mean_duration_normal`",
        "* `E_z[σ(alpha10 + gamma10·z)] = 1 / mean_duration_stress`",
        "",
        "with `z` taken from the full historical `rd_standardized.csv`. "
        "Consequence: no MLE standard error attached; a cross-check simulation "
        "reproduces the target 67% stress occupancy to within 4.5%.",
        "",
        "### (b) `RD_Ramp_1h_lag1` covariate is dropped (omitted-variable risk)",
        "",
        "The M9 fit (`M9_student_t_tvtp_TVTP-2`) uses two transition covariates: "
        "`RD_lag1` and `RD_Ramp_1h_lag1`. The current PDE code hard-codes a "
        "single-covariate TVTP (`generator.py`; `covariate = 'RD_lag1'`) and "
        "only the `RD_lag1` gamma is copied into the yaml. Reported average "
        "marginal effects: `ame_p01` for `RD_lag1` = +0.0086, for "
        "`RD_Ramp_1h_lag1` = +0.0451 (~5× larger). The derived alphas absorb "
        "the mean effect of the missing ramp term, biasing the intercepts by an "
        "unknown but non-zero amount.",
        "",
        "### (c) `scale_P = 282.48` is unverifiable in the current tree",
        "",
        f"The yaml pins `scale_P = {params.scale_P}` TRY/MWh with provenance "
        "\"training median absolute price\". The source files that were meant to "
        "carry this value (`pde_export.json`, `prepared_meta.json`) are NOT "
        "present in the current `inputs/` tree, and the raw historical PTF "
        "series needed to re-derive it is also absent. `scale_P` is treated as "
        "an inherited constant, trusted on faith from external metadata no "
        "longer reachable.",
        "",
        "### (d) M9 regime labels were swapped to the yaml convention",
        "",
        "M9's raw fit puts the high-volatility state at index 0 (occupancy "
        "67.5%, duration 7.56 h). The yaml convention "
        "(`params_frozen.FrozenM2Parameters.__post_init__`) requires "
        "`sigma_y[1] > sigma_y[0]`, i.e. index 0 = normal, index 1 = stress. "
        "All four TVTP coefficients were therefore cross-swapped: "
        "yaml `sigma_y_normal` ← raw M9 `sigma1`, yaml `sigma_y_stress` ← raw "
        "M9 `sigma0`, yaml `gamma01` (normal→stress) ← raw M9 `gamma10`, yaml "
        "`gamma10` (stress→normal) ← raw M9 `gamma01`. Signs of the RD "
        "sensitivities are economically consistent (higher RD reduces "
        "normal→stress and raises stress→normal). Cross-check simulation "
        "reproduces the reported occupancy diagnostics to within 5%.",
        "",
        "### (e) `pi_filtered` is M2-sourced, mixed with M9 dynamics",
        "",
        f"`pi_filtered = [{params.pi_filtered[0]:.4f}, {params.pi_filtered[1]:.4f}]` "
        "comes from the shipped M2 filter output at the valuation hour; "
        "sigma/gamma/alpha values are all from the M9 fit. No M9-specific "
        "valuation-time filtered probability is available "
        "(`pde_timeseries.parquet` is not in the current inputs and the "
        "shipped ZIP handoff did not include it). Empirically measured "
        "impact of this mixed-source choice on the K=3000 European call: "
        "<2 % on 72 h+ maturities (the TVTP chain converges to its "
        "stationary distribution within ~10 mean durations), 5-30 % on "
        "<24 h maturities where the initial condition still dominates. "
        "`run_pde.py price --pi-override stationary` swaps in the M9 "
        "long-run occupancy (`m9_stationary_pi` in the frozen-parameter "
        "yaml) for sensitivity analysis; see "
        "`docs/tvtp_derivation_methodology.md` for the full derivation.",
        "",
        "### (f) Risk-neutral drift adjustment (Q1) is wired but UNCALIBRATED",
        "",
        "The `run_pde.py price` command accepts `--risk-premium-a0` and "
        "`--risk-premium-a1` (TRY/MWh per hour, regime-0 and regime-1 "
        "respectively), which apply a Q1 drift shift `a_i` on the residual "
        "SDE.  The shift is threaded symmetrically into the moment ODE and "
        "the pricing PDE / MC simulator, so `E^Q[P_t] = F(t)` is preserved "
        "exactly (guarded by `tests/test_forward_centered.py::"
        "test_q1_drift_shift_preserves_centering`).  However, **no electricity "
        "option market data exists to estimate a real market price of risk**, "
        "so the flags are UNCALIBRATED sensitivity scenarios only.  Default "
        "`(0, 0)` reproduces every prior benchmark bit-for-bit -- the "
        "physical-measure intensities are used as-is, which is the "
        "zero-risk-premium assumption.  See "
        "`docs/risk_neutral_methodology.md` for the mathematical proof that "
        "the drift channel leaves the forward-curve identity intact and only "
        "moves higher moments (variance -> option value).",
        "",
        "### (g) Within-regime phi vs deseasonalized single-regime AR fit — open consistency question",
        "",
        "M9's regime-conditional phi (0.999996) implies a within-regime OU "
        "half-life of ~19.25 years -- three orders of magnitude longer than the "
        "~8.84-hour half-life computed on the deseasonalized single-regime "
        "series (`metadata/model_parameters_and_ou_mapping.json`, "
        "`discovered_parameter_files`).  This divergence is plausible in "
        "principle for a Markov-switching AR(1) (high within-regime persistence "
        "combined with frequent regime transitions can still produce "
        "fast-appearing marginal dynamics), but has NOT been independently "
        "verified against the original M9 fitting process.  Treated as an open "
        "consistency question, not a resolved one; flagged for follow-up with "
        "the original model author if possible.",
        "",
    ]
    return "\n".join(lines) + "\n"
