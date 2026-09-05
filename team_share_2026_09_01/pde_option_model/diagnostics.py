"""Economic diagnostics: figures and tables for the two-regime PDE model.

Everything here is *descriptive* output built on top of the pricing stack.
All monetary results carry the uncalibrated-price label from
:mod:`pde_option_model.risk_neutral` unless a calibrated
:class:`MeasureAdjustment` is supplied.

Outputs (PNG + CSV) are written under a caller-supplied directory, default
``outputs/example_run``.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import EuropeanOption
from .generator import probs_to_generator
from .markov_adapter import REGIME_NAMES, MarkovInputs
from .pricing import GridSettings, build_dynamics, price_contract
from .risk_neutral import MeasureAdjustment, baseline_q1
from .scenarios import ScenarioPath
from .validation import simulate_paths

log = logging.getLogger(__name__)

_REGIME_COLORS = ("#1f77b4", "#d62728")  # normal, stress


def _ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1. Transition probabilities / generator intensities as functions of z
# ---------------------------------------------------------------------------

def plot_transition_structure(inputs: MarkovInputs, outdir: Path) -> pd.DataFrame:
    """p01/p10 and q01/q10 (and expected regime durations) vs covariate z."""
    z = np.linspace(-3.0, 3.0, 241)
    p01, p10 = inputs.tvtp.probabilities(z)
    gen = probs_to_generator(p01, p10, inputs.dt_hours)
    q01, q10 = gen.q01, gen.q10
    dur0 = np.where(q01 > 0, 1.0 / q01, np.inf)
    dur1 = np.where(q10 > 0, 1.0 / q10, np.inf)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(z, p01, color=_REGIME_COLORS[0], label="p01 (normal→stress)")
    ax.plot(z, p10, color=_REGIME_COLORS[1], label="p10 (stress→normal)")
    ax.set_xlabel("z (standardised RD_WS, lag 1h)")
    ax.set_ylabel("hourly transition probability")
    ax.set_title("TVTP probabilities vs covariate")
    ax.legend()
    ax = axes[1]
    ax.plot(z, q01, color=_REGIME_COLORS[0], label="q01")
    ax.plot(z, q10, color=_REGIME_COLORS[1], label="q10")
    ax.set_xlabel("z")
    ax.set_ylabel("intensity per hour (matrix-log)")
    ax.set_title("Generator intensities vs covariate")
    ax.legend()
    ax = axes[2]
    ax.plot(z, dur0, color=_REGIME_COLORS[0], label="E[duration | normal] = 1/q01")
    ax.plot(z, dur1, color=_REGIME_COLORS[1], label="E[duration | stress] = 1/q10")
    ax.set_yscale("log")
    ax.set_xlabel("z")
    ax.set_ylabel("hours (log scale)")
    ax.set_title("Expected regime durations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "transition_structure.png", dpi=150)
    plt.close(fig)

    df = pd.DataFrame(
        {"z": z, "p01": p01, "p10": p10, "q01": q01, "q10": q10,
         "E_dur_normal_h": dur0, "E_dur_stress_h": dur1}
    )
    df.to_csv(outdir / "transition_structure.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 2. Historical regime probabilities vs price
# ---------------------------------------------------------------------------

def plot_history(inputs: MarkovInputs, raw_ts: pd.DataFrame, outdir: Path,
                 last_days: int = 365) -> None:
    """Price and shipped filtered stress-probability overlay (orientation-fixed)."""
    ts = raw_ts.tail(last_days * 24)
    stress_col = "filter_p_state0" if inputs.labels_flipped_in_shipped_series else "filter_p_state1"
    fig, ax1 = plt.subplots(figsize=(13, 4.2))
    ax1.plot(ts.index, ts["PTF_TRY_MWh"], color="black", lw=0.6, label="PTF (TRY/MWh)")
    ax1.set_ylabel("PTF TRY/MWh")
    ax2 = ax1.twinx()
    ax2.fill_between(ts.index, ts[stress_col], color=_REGIME_COLORS[1], alpha=0.30,
                     label="P(stress) filtered")
    ax2.set_ylabel("filtered P(stress)")
    ax2.set_ylim(0, 1)
    ax1.set_title(
        f"PTF and filtered stress probability — last {last_days} days "
        "(shipped series relabelled to normal/stress orientation)"
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "history_price_stress_prob.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Value profiles V_i(y) and probability-weighted value
# ---------------------------------------------------------------------------

def plot_value_profiles(result, inputs: MarkovInputs, outdir: Path,
                        tag: str = "base") -> pd.DataFrame:
    """V_normal(y), V_stress(y), weighted V(y) with spot marker, in price space."""
    grid = result.solve.grid
    y = grid.y
    P = inputs.transform.price_from_y(y)
    Vgrid = result.solve.V  # (2, Ny) full profiles; result.V_regime is at y0
    Vw = result.pi[0] * Vgrid[0] + result.pi[1] * Vgrid[1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, x, xlabel in ((axes[0], y, "y = asinh(P/s)"), (axes[1], P, "P (TRY/MWh)")):
        for i, name in enumerate(REGIME_NAMES):
            ax.plot(x, Vgrid[i], color=_REGIME_COLORS[i], label=f"V_{name}")
        ax.plot(x, Vw, color="green", ls="--",
                label=f"π-weighted (π_stress={result.pi[1]:.3f})")
        x0 = result.y0 if xlabel.startswith("y") else result.price0
        ax.axvline(x0, color="gray", ls=":", lw=1, label="valuation state")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("option value (TRY/MWh)")
        ax.legend(fontsize=8)
    if np.isfinite(result.price0):
        axes[1].set_xlim(0, max(4 * result.price0, 2 * result.contract.strike))
    fig.suptitle(
        f"{result.contract.option_type} K={result.contract.strike:.0f} — scenario '{tag}' — "
        f"{result.price_label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"value_profiles_{tag}.png", dpi=150)
    plt.close(fig)

    df = pd.DataFrame({"y": y, "P": P,
                       "V_normal": Vgrid[0],
                       "V_stress": Vgrid[1],
                       "V_weighted": Vw})
    df.to_csv(outdir / f"value_profiles_{tag}.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 4. Scenario comparison
# ---------------------------------------------------------------------------

def scenario_table(inputs: MarkovInputs, contract: EuropeanOption,
                   paths: Sequence[ScenarioPath],
                   adjustment: Optional[MeasureAdjustment] = None,
                   gs: Optional[GridSettings] = None,
                   outdir: Optional[Path] = None) -> pd.DataFrame:
    """Price the contract under each deterministic RD scenario."""
    rows = []
    results = {}
    for sp in paths:
        res = price_contract(inputs, contract, sp, adjustment=adjustment,
                             grid_settings=gs)
        results[sp.name] = res
        rows.append({
            "scenario": sp.name,
            "value": res.value,
            "V_at_y0_normal": float(res.V_regime[0]),
            "V_at_y0_stress": float(res.V_regime[1]),
            "pi_normal": res.pi[0], "pi_stress": res.pi[1],
            "max_peclet": res.solve.max_peclet,
            "upwinded_fraction": res.solve.upwinded_fraction,
        })
    df = pd.DataFrame(rows)
    df.attrs["price_label"] = next(iter(results.values())).price_label
    if outdir is not None:
        df.to_csv(outdir / "scenario_comparison.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(df["scenario"], df["value"], color="#4c72b0")
        ax.set_ylabel("option value (TRY/MWh)")
        ax.set_title(f"{contract.option_type} K={contract.strike:.0f}, "
                     f"τ={contract.tau_hours:.0f}h — deterministic RD scenarios")
        for i, v in enumerate(df["value"]):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / "scenario_comparison.png", dpi=150)
        plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# 5. Sensitivity tables (drift shift a, switching premia eta, sigma multiplier)
# ---------------------------------------------------------------------------

def sensitivity_tables(inputs: MarkovInputs, contract: EuropeanOption,
                       scenario: ScenarioPath,
                       drift_shifts: Sequence[float] = (-0.01, -0.005, 0.0, 0.005),
                       etas: Sequence[float] = (-0.5, 0.0, 0.5),
                       sigma1_mults: Sequence[float] = (0.9, 1.0, 1.1),
                       gs: Optional[GridSettings] = None,
                       outdir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """Value sensitivities to the main risk-premium / robustness knobs."""
    out: dict[str, pd.DataFrame] = {}

    rows = []
    for a in drift_shifts:
        adj = baseline_q1(drift_shift_per_hour=(a, a))
        res = price_contract(inputs, contract, scenario, adjustment=adj, grid_settings=gs)
        rows.append({"a_per_hour": a, "value": res.value})
    out["drift_shift"] = pd.DataFrame(rows)

    rows = []
    for e in etas:
        adj = MeasureAdjustment(spec="Q2", eta=np.array([e, -e]))
        res = price_contract(inputs, contract, scenario, adjustment=adj, grid_settings=gs)
        rows.append({"eta01": e, "eta10": -e, "value": res.value})
    out["eta"] = pd.DataFrame(rows)

    rows = []
    for m in sigma1_mults:
        dyn = build_dynamics(inputs, sigma_multipliers=(1.0, m))
        res = price_contract(inputs, contract, scenario, dynamics=dyn, grid_settings=gs)
        rows.append({"sigma_stress_multiplier": m, "value": res.value})
    out["sigma_stress"] = pd.DataFrame(rows)

    if outdir is not None:
        for k, df in out.items():
            df.to_csv(outdir / f"sensitivity_{k}.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        specs = [("drift_shift", "a_per_hour", "drift shift a (per hour)"),
                 ("eta", "eta01", "eta01 (= -eta10), Q2"),
                 ("sigma_stress", "sigma_stress_multiplier", "σ_stress multiplier")]
        for ax, (k, xcol, xlabel) in zip(axes, specs):
            ax.plot(out[k][xcol], out[k]["value"], marker="o")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("value (TRY/MWh)")
        fig.suptitle("Sensitivities — model-implied scenario prices", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / "sensitivities.png", dpi=150)
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 6. Model-implied forward curve (MC horizon prices, P vs adjusted measure)
# ---------------------------------------------------------------------------

def forward_curve(inputs: MarkovInputs, contract: EuropeanOption,
                  scenario: ScenarioPath,
                  horizons_hours: Sequence[float] = (6, 12, 24, 48, 72, 120, 168, 336, 720),
                  adjustment: Optional[MeasureAdjustment] = None,
                  n_paths: int = 20000,
                  outdir: Optional[Path] = None) -> pd.DataFrame:
    """E[P_T] over horizons via the exact-OU Monte Carlo engine.

    Long horizons (>= 168h) are dominated by the near-unit-root θ level and are
    flagged accordingly — treat them as descriptive, not tradeable.
    """
    horizon = max(horizons_hours)
    c = dataclasses.replace(
        contract, maturity_utc=contract.valuation_utc + pd.Timedelta(hours=horizon))
    sim = simulate_paths(inputs, c, scenario, adjustment=adjustment,
                         n_paths=n_paths, return_horizon_prices=list(horizons_hours))
    rows = []
    for h in horizons_hours:
        p = sim["horizon_prices"][float(h)]
        rows.append({"horizon_h": h, "F_model": float(np.mean(p)),
                     "se": float(np.std(p, ddof=1) / np.sqrt(len(p))),
                     "flag": "extrapolative" if h >= 168 else ""})
    df = pd.DataFrame(rows)
    df.attrs["note"] = ("Model-implied forward curve under the current measure; "
                        "NOT calibrated to EPİAŞ/VIOP forwards. Horizons >= 168h "
                        "extrapolate a near-unit-root long-run mean.")
    if outdir is not None:
        df.to_csv(outdir / "forward_curve.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.errorbar(df["horizon_h"], df["F_model"], yerr=1.96 * df["se"],
                    marker="o", capsize=3, label="E[P_T] model")
        ax.axhline(inputs.price0, color="gray", ls=":", label=f"spot {inputs.price0:.0f}")
        ax.axvspan(168, max(horizons_hours), color="orange", alpha=0.12,
                   label="extrapolative (≥168h)")
        ax.set_xlabel("horizon (hours)")
        ax.set_ylabel("TRY/MWh")
        ax.set_title("Model-implied forward curve (uncalibrated)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "forward_curve.png", dpi=150)
        plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all(inputs: MarkovInputs, raw_ts: pd.DataFrame, contract: EuropeanOption,
            scenario_paths: Sequence[ScenarioPath],
            adjustment: Optional[MeasureAdjustment] = None,
            gs: Optional[GridSettings] = None,
            outdir: str | Path = "outputs/example_run",
            sensitivity_cfg: Optional[dict] = None,
            forward_horizons: Sequence[float] = (6, 12, 24, 48, 72, 120, 168, 336, 720),
            ) -> dict[str, pd.DataFrame]:
    """Produce the full diagnostics pack. Returns the key tables."""
    outdir = _ensure_dir(Path(outdir))
    tables: dict[str, pd.DataFrame] = {}

    log.info("diagnostics: transition structure")
    tables["transition_structure"] = plot_transition_structure(inputs, outdir)

    log.info("diagnostics: history overlay")
    plot_history(inputs, raw_ts, outdir)

    log.info("diagnostics: scenario comparison")
    tables["scenarios"] = scenario_table(inputs, contract, scenario_paths,
                                         adjustment=adjustment, gs=gs, outdir=outdir)

    base = next((s for s in scenario_paths if s.name == "base"), scenario_paths[0])
    log.info("diagnostics: value profiles (base scenario)")
    res = price_contract(inputs, contract, base, adjustment=adjustment, grid_settings=gs)
    tables["value_profiles"] = plot_value_profiles(res, inputs, outdir, tag=base.name)

    log.info("diagnostics: sensitivities")
    scfg = sensitivity_cfg or {}
    tables.update(sensitivity_tables(
        inputs, contract, base,
        drift_shifts=scfg.get("drift_shift_per_hour", (-0.01, -0.005, 0.0, 0.005)),
        etas=scfg.get("eta_symmetric", (-0.5, 0.0, 0.5)),
        sigma1_mults=scfg.get("sigma1_multiplier", (0.9, 1.0, 1.1)),
        gs=gs, outdir=outdir))

    log.info("diagnostics: forward curve (MC)")
    tables["forward_curve"] = forward_curve(inputs, contract, base,
                                            horizons_hours=forward_horizons,
                                            adjustment=adjustment, outdir=outdir)
    return tables
