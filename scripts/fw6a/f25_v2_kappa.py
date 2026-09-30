"""FW6a -- reproduction of the F2.5 pooled-vs-M9 comparison under the v2 kappa.

Context
-------
``outputs/market_calibration_final/model_comparison_pooled_vs_M9.md`` (F2.5)
prices a K = 3000 European call at 24 / 72 / 168 / 336 h under three residual
specifications:

    pooled_M0_kappa   pooled single volatility + M0's own fitted kappa
    pooled_M9_kappa   pooled single volatility + M9's kappa
    M9_prod           the production two-regime TVTP model

Every M9-kappa-bearing number in that markdown was produced with
``kappa = 4.108274e-06 /h`` (phi = 0.999995891734), the pre-v2 value.  The
production frozen-parameter file now carries ``kappa = 0.078394 /h``
(phi = 0.9246), so the published F2.5 tables are parameter-inconsistent with
the rest of the paper.  This module re-runs the identical comparison under the
production kappa and also reproduces the pre-v2 era, so the two can be
tabulated side by side.

What is held fixed relative to F2.5
-----------------------------------
* Same forward curve (smooth_constrained, spot_to_next_linear January anchor).
* Same ``pi_filtered = (0.931977, 0.068023)`` from the shipped M2 filter.
* Same climatology ``z(t-1)`` covariate path as ``run_pde.py price``: the
  deterministic path is built on a 0.25 h master grid and interpolated onto
  the solver grid.  No constant-transition (z = 0) fallback is used and no
  synthetic covariate is fabricated (FW12b rule).
* Same strike (3000 TRY/MWh), maturities, option type and r_annual (0.40).
* Same pooled-sigma construction: M0's two fitted volatilities collapsed with
  M9's stationary occupancy, then split by +/- 0.01 % so that the
  ``sigma_y[1] > sigma_y[0]`` invariant of ``FrozenM2Parameters`` still holds
  while the two regimes remain numerically indistinguishable.

What changes
------------
* ``kappa`` of the two M9-kappa variants: 4.108274e-06 -> 0.078394 /h.
* The residual state grid is the production one, ``n_space_nodes = 1201``.

``pooled_M0_kappa`` is kappa-invariant by construction (it carries M0's own
kappa), so it doubles as a control: its four rows must be identical in the two
eras.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption                 # noqa: E402
from pde_option_model.forward_centered import (                       # noqa: E402
    ForwardCenteredModel, ResidualGridSettings, ResidualSpec,
    price_forward_centered)
from pde_option_model.forward_curve import (                          # noqa: E402
    NearTermAnchor, build_forward_curve)
from pde_option_model.generator import TVTPCoefficients               # noqa: E402
from pde_option_model.grid import TimeGrid                            # noqa: E402
from pde_option_model.market_data import load_quotes                  # noqa: E402
from pde_option_model.params_frozen import (                          # noqa: E402
    FrozenM2Parameters, load_frozen_parameters)
from pde_option_model.scenarios import ScenarioBuilder, ScenarioSpec  # noqa: E402

# --- production inputs (identical to config/forward_centered_config.yaml) ---
PARAMS_YAML = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
QUOTES_CSV = REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv"
Z_HISTORY_CSV = REPO_ROOT / "inputs" / "historical" / "rd_standardized.csv"
M0_ESTIMATES_CSV = (REPO_ROOT / "inputs" / "historical" / "archive"
                    / "calibration_bundle" / "parameter_estimates.csv")

TRAIN_END_UTC = pd.Timestamp("2022-12-31 20:00:00+00:00")
COVARIATE_LAG_HOURS = 1.0
SCENARIO_MODE = "climatology"
CURVE_MODE = "smooth_constrained"
JANUARY_ANCHOR_MODE = "spot_to_next_linear"
SMOOTHNESS_WEIGHT = 1.0
LEVEL_WEIGHT = 1.0e-4

STRIKE_TRY_MWH = 3000.0
OPTION_TYPE = "call"
R_ANNUAL = 0.40
MATURITIES_HOURS: Tuple[int, ...] = (24, 72, 168, 336)
N_SPACE_NODES = 1201
N_STD = 6.0

# F2.5's tie-break on the sigma ordering invariant: the pooled volatility is
# split by +/- 1e-4 relative so that regime 1 stays the nominal stress state.
POOLED_SIGMA_SPLIT = 1.0e-4

# The published F2.5 markdown was produced with this kappa.
PRE_V2_KAPPA_PER_HOUR = 4.108274e-06

VARIANT_LABELS: Dict[str, str] = {
    "pooled_M0_kappa": "pooled_M0_kappa (pooled sigma, M0 kappa)",
    "pooled_M9_kappa": "pooled_M9_kappa (pooled sigma, M9 kappa)",
    "M9_prod": "M9_prod (2-regime TVTP, production)",
}

# Accepted output trees -- never written by this work package.
PROTECTED_OUTPUT_DIRS = (
    "outputs/market_calibration_final",
    "outputs/forward_centered_diagnostics",
    "outputs/scenario_sweep",
    "outputs/tvtp2_experimental",
)

DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "f25_v2_kappa"


class FW6aError(RuntimeError):
    """Raised when the FW6a run is misconfigured."""


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_m0_estimates(path: Path = M0_ESTIMATES_CSV) -> Dict[str, float]:
    """M0's fitted (sigma0, sigma1, phi) from the shipped calibration bundle.

    These are raw-bundle values: index 0 is M0's low-volatility state and
    index 1 its high-volatility state (``swapped_labels`` is False for M0),
    which already matches the yaml convention used for the pooled collapse.
    """
    row = pd.read_csv(path).set_index("model").loc["M0"]
    sigma0, sigma1 = float(row["sigma0"]), float(row["sigma1"])
    phi = float(row["phi"])
    if not 0.0 < phi < 1.0:
        raise FW6aError(f"{path}: M0 phi={phi} is outside (0, 1)")
    if sigma1 <= sigma0:
        raise FW6aError(f"{path}: expected M0 sigma1 > sigma0, "
                        f"got {sigma1} <= {sigma0}")
    return {"sigma0": sigma0, "sigma1": sigma1, "phi": phi,
            "kappa_per_hour": -math.log(phi)}


def pooled_sigma(m0: Dict[str, float], stationary_pi: Sequence[float]) -> float:
    """Occupancy-weighted RMS of M0's two volatilities.

    M0's own long-run occupancy is not exported by the handoff bundle, so F2.5
    uses M9's stationary occupancy as the proxy.  Caveat (a) of the F2.5
    markdown records that substitution; this module reproduces it unchanged.
    """
    w = np.asarray(stationary_pi, dtype=float)
    if w.shape != (2,) or np.any(w < 0) or abs(w.sum() - 1.0) > 1e-6:
        raise FW6aError("stationary_pi must be a two-vector on the simplex")
    return float(math.sqrt(w[0] * m0["sigma0"] ** 2 + w[1] * m0["sigma1"] ** 2))


def pooled_sigma_pair(sigma: float,
                      split: float = POOLED_SIGMA_SPLIT) -> np.ndarray:
    """(sigma_normal, sigma_stress) for the one-regime collapse."""
    return np.array([sigma * (1.0 - split), sigma * (1.0 + split)], dtype=float)


def effective_sigma(params: FrozenM2Parameters) -> float:
    """M9's own occupancy-weighted volatility, the counterpart of `pooled_sigma`.

    F2.5 pools M0's two sigmas but leaves M9_prod on M9's, so the two sides of
    the comparison never shared a volatility level.  This is the number that
    makes the mismatch visible: the ratio to the pooled sigma is what the
    residual-sd ratio at expiry converges to once the variance saturates.
    """
    if params.m9_stationary_pi is None:
        raise FW6aError(f"{params.source_file}: m9_stationary_pi is required")
    return float(math.sqrt(
        params.mixture_variance_rate(params.m9_stationary_pi)))


def load_z_history(path: Path = Z_HISTORY_CSV) -> pd.Series:
    """Hourly standardised RD history, exactly as `run_pde.py price` reads it."""
    hist = pd.read_csv(path)
    if not {"datetime", "z"} <= set(hist.columns):
        raise FW6aError(f"{path}: expected columns 'datetime' and 'z'")
    idx = pd.to_datetime(hist["datetime"], utc=True)
    return pd.Series(hist["z"].to_numpy(float), index=idx).sort_index()


# ---------------------------------------------------------------------------
# model assembly
# ---------------------------------------------------------------------------
def variant_parameters(base: FrozenM2Parameters, m0: Dict[str, float],
                       variant: str,
                       m9_kappa_per_hour: float) -> FrozenM2Parameters:
    """The frozen-parameter set behind one comparison variant.

    Going through ``FrozenM2Parameters`` rather than patching ``ResidualSpec``
    directly keeps the ``sigma_y[1] > sigma_y[0]`` invariant in force, which is
    what forced F2.5's +/- 0.01 % split in the first place.
    """
    phi_m9 = float(math.exp(-float(m9_kappa_per_hour)))
    if variant == "M9_prod":
        return replace(base, phi=phi_m9,
                       kappa_per_hour=float(m9_kappa_per_hour),
                       sigma_y=base.sigma_y.copy())
    if base.m9_stationary_pi is None:
        raise FW6aError(f"{base.source_file}: m9_stationary_pi is required "
                        "for the pooled collapse")
    sig = pooled_sigma_pair(pooled_sigma(m0, base.m9_stationary_pi))
    if variant == "pooled_M0_kappa":
        return replace(base, phi=m0["phi"],
                       kappa_per_hour=m0["kappa_per_hour"], sigma_y=sig)
    if variant == "pooled_M9_kappa":
        return replace(base, phi=phi_m9,
                       kappa_per_hour=float(m9_kappa_per_hour), sigma_y=sig)
    raise FW6aError(f"unknown variant {variant!r}")


def build_model(params: FrozenM2Parameters, quotes) -> ForwardCenteredModel:
    """Production forward-centered model for one parameter set.

    The TVTP law, the filtered belief and the forward curve are the production
    ones in every variant; only ``kappa`` and ``sigma_y`` differ between them.
    """
    curve = build_forward_curve(
        quotes, mode=CURVE_MODE,
        anchor=NearTermAnchor(mode=JANUARY_ANCHOR_MODE, level_TRY_MWh=None),
        spot_price_TRY_MWh=params.spot_price_TRY_MWh,
        smoothness_weight=SMOOTHNESS_WEIGHT, level_weight=LEVEL_WEIGHT)
    return ForwardCenteredModel(
        curve=curve, spec=ResidualSpec.from_frozen(params),
        tvtp=TVTPCoefficients(params.alpha01, params.gamma01,
                              params.alpha10, params.gamma10),
        pi_filtered=params.pi_filtered, valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh)


# ---------------------------------------------------------------------------
# climatology covariate path (identical to `run_pde.py price`)
# ---------------------------------------------------------------------------
def climatology_z_lagged_fn(contract: EuropeanOption,
                            grid_settings: ResidualGridSettings,
                            z_history: pd.Series):
    """``t -> z(t - lag)`` on the production climatology path.

    ``run_pde.py price`` builds the deterministic path on a master grid of
    ``max(n_solver_steps, ceil(tau / 0.25))`` steps and linearly interpolates
    it onto whatever grid the solver uses.  The same two-step construction is
    reproduced here so the prices are bit-comparable with the shipped CLI runs;
    ``tests/test_fw6a_f25_v2_kappa.py`` pins that equality.
    """
    tau = float(contract.tau_hours)
    n_master = max(grid_settings.n_steps(tau), int(np.ceil(tau / 0.25)))
    master = TimeGrid(contract.valuation_utc, contract.maturity_utc, n_master)
    builder = ScenarioBuilder(z_history=z_history, train_end=TRAIN_END_UTC,
                              covariate_lag_hours=COVARIATE_LAG_HOURS)
    path = builder.build(ScenarioSpec(name="pricing", mode=SCENARIO_MODE),
                         master)

    def _fn(t: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(t, dtype=float),
                         path.times_hours, path.z_lagged)

    return _fn, path


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def run_comparison(m9_kappa_per_hour: float,
                   era: str,
                   maturities_hours: Sequence[int] = MATURITIES_HOURS,
                   strike: float = STRIKE_TRY_MWH,
                   n_space_nodes: int = N_SPACE_NODES,
                   params_yaml: Path = PARAMS_YAML) -> pd.DataFrame:
    """Price all three variants at every maturity under one M9 kappa."""
    base = load_frozen_parameters(params_yaml)
    quotes = load_quotes(QUOTES_CSV)
    z_history = load_z_history()
    m0 = load_m0_estimates()
    gs = ResidualGridSettings(n_space_nodes=int(n_space_nodes),
                              n_time_steps=None, n_std=N_STD)

    rows: List[Dict[str, object]] = []
    for hours in maturities_hours:
        contract = EuropeanOption(
            option_type=OPTION_TYPE, strike=float(strike),
            valuation_utc=base.valuation_utc,
            maturity_utc=base.valuation_utc + pd.Timedelta(hours=int(hours)),
            r_annual=R_ANNUAL)
        z_fn, path = climatology_z_lagged_fn(contract, gs, z_history)
        for variant, label in VARIANT_LABELS.items():
            params = variant_parameters(base, m0, variant, m9_kappa_per_hour)
            model = build_model(params, quotes)
            res = price_forward_centered(model, contract, gs, z_lagged_fn=z_fn)
            rows.append({
                "era": era,
                "maturity_h": int(hours),
                "variant": variant,
                "model": label,
                "kappa_per_hour": float(params.kappa_per_hour),
                "half_life_hours": round(float(params.half_life_hours), 6),
                "sigma_y_normal": float(params.sigma_y[0]),
                "sigma_y_stress": float(params.sigma_y[1]),
                "F_T_TRY_MWh": round(float(res.forward_at_expiry), 4),
                # F2.5 caveat (d): the centering identity is invariant to the
                # residual sigma AND to kappa; carried here so the re-run can
                # be checked rather than asserted.
                "E_spot_T_TRY_MWh": round(float(res.expected_spot_at_expiry), 4),
                "centering_abs_err_TRY_MWh": abs(
                    float(res.expected_spot_at_expiry)
                    - float(res.forward_at_expiry)),
                "residual_sd_T_TRY_MWh": round(float(res.residual_std_at_expiry), 4),
                "V_regime0_TRY_MWh": round(float(res.V_regime[0]), 4),
                "V_regime1_TRY_MWh": round(float(res.V_regime[1]), 4),
                "call_TRY_MWh": round(float(res.value), 4),
                "n_space_nodes": int(n_space_nodes),
                "n_time_steps": int(res.diagnostics["n_time_steps"]),
                "z_lag_min": round(float(path.z_lagged.min()), 6),
                "z_lag_max": round(float(path.z_lagged.max()), 6),
            })
    return pd.DataFrame(rows)


def side_by_side(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """One row per (maturity, variant) with the pre-v2 and v2 numbers."""
    keys = ["maturity_h", "variant"]
    cols = ["kappa_per_hour", "F_T_TRY_MWh", "residual_sd_T_TRY_MWh",
            "call_TRY_MWh"]
    merged = old[keys + cols].merge(new[keys + cols], on=keys,
                                    suffixes=("_old", "_new"))
    merged["d_call"] = (merged["call_TRY_MWh_new"]
                        - merged["call_TRY_MWh_old"]).round(4)
    merged["pct_call"] = (100.0 * merged["d_call"]
                          / merged["call_TRY_MWh_old"]).round(4)
    merged["d_sd"] = (merged["residual_sd_T_TRY_MWh_new"]
                      - merged["residual_sd_T_TRY_MWh_old"]).round(4)
    merged["pct_sd"] = (100.0 * merged["d_sd"]
                        / merged["residual_sd_T_TRY_MWh_old"]).round(4)
    order = {v: i for i, v in enumerate(VARIANT_LABELS)}
    return (merged.assign(_o=merged["variant"].map(order))
            .sort_values(["maturity_h", "_o"]).drop(columns="_o")
            .reset_index(drop=True))


def gap_table(frame: pd.DataFrame) -> pd.DataFrame:
    """M9_prod vs each pooled baseline, per maturity -- the F2.5 headline."""
    call = frame.pivot(index="maturity_h", columns="variant",
                       values="call_TRY_MWh")
    sd = frame.pivot(index="maturity_h", columns="variant",
                     values="residual_sd_T_TRY_MWh")
    out = pd.DataFrame(index=call.index)
    for base in ("pooled_M0_kappa", "pooled_M9_kappa"):
        out[f"call_{base}"] = call[base].round(4)
    out["call_M9_prod"] = call["M9_prod"].round(4)
    for base in ("pooled_M0_kappa", "pooled_M9_kappa"):
        out[f"delta_vs_{base}"] = (call["M9_prod"] - call[base]).round(4)
        out[f"pct_vs_{base}"] = (100.0 * (call["M9_prod"] - call[base])
                                 / call[base]).round(4)
        out[f"sd_ratio_vs_{base}"] = (sd["M9_prod"] / sd[base]).round(6)
    # The kappa channel on its own: same pooled sigma, M0's kappa vs M9's.
    # This is what F2.5's interpretation point 4 quantifies.
    out["kappa_effect_pct"] = (
        100.0 * (call["pooled_M9_kappa"] / call["pooled_M0_kappa"] - 1.0)
    ).round(4)
    return out.reset_index()


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def assert_writable(outdir: Path) -> Path:
    """Refuse to write into any accepted output tree."""
    res = Path(outdir).resolve()
    for prot in PROTECTED_OUTPUT_DIRS:
        p = (REPO_ROOT / prot).resolve()
        if res == p or p in res.parents:
            raise FW6aError(
                "refusing to write FW6a outputs into the accepted directory "
                f"{prot}; choose an --outdir outside {PROTECTED_OUTPUT_DIRS}")
    return Path(outdir)


def _md_table(frame: pd.DataFrame, floatfmt: str = "%.4f") -> str:
    def cell(v: object) -> str:
        if isinstance(v, float):
            return "nan" if math.isnan(v) else floatfmt % v
        return str(v)
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---:" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([head, rule, *body])


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def _manifest(m9_kappa_per_hour: float, base: FrozenM2Parameters,
              m0: Dict[str, float], strike: float, maturities: Sequence[int],
              n_space_nodes: int) -> Dict[str, object]:
    return {
        "work_package": "FW6a",
        "purpose": ("F2.5 pooled-vs-M9 comparison re-run under the v2 kappa of "
                    "the production frozen-parameter file"),
        "source_report":
            "outputs/market_calibration_final/model_comparison_pooled_vs_M9.md",
        "source_csv":
            "outputs/market_calibration_final/model_comparison_pooled_vs_M9.csv",
        "inputs": {
            "frozen_parameters": _rel(PARAMS_YAML),
            "quotes": _rel(QUOTES_CSV),
            "z_history": _rel(Z_HISTORY_CSV),
            "m0_estimates": _rel(M0_ESTIMATES_CSV),
        },
        "settings": {
            "strike_TRY_MWh": float(strike),
            "option_type": OPTION_TYPE,
            "r_annual": R_ANNUAL,
            "maturities_hours": [int(h) for h in maturities],
            "n_space_nodes": int(n_space_nodes),
            "n_std": N_STD,
            "n_time_steps": "null -> max(96, 2 per hour)",
            "curve_mode": CURVE_MODE,
            "january_anchor_mode": JANUARY_ANCHOR_MODE,
            "scenario_mode": SCENARIO_MODE,
            "covariate_lag_hours": COVARIATE_LAG_HOURS,
            "train_end_utc": TRAIN_END_UTC.isoformat(),
            "pi_filtered": [float(x) for x in base.pi_filtered],
        },
        "kappa_per_hour": {
            "pre_v2_M9": PRE_V2_KAPPA_PER_HOUR,
            "v2_M9": float(m9_kappa_per_hour),
            "M0": m0["kappa_per_hour"],
            "ratio_v2_over_pre_v2":
                float(m9_kappa_per_hour) / PRE_V2_KAPPA_PER_HOUR,
        },
        "pooled_baseline": {
            "M0_sigma0": m0["sigma0"],
            "M0_sigma1": m0["sigma1"],
            "M0_phi": m0["phi"],
            "m9_stationary_pi": [float(x) for x in base.m9_stationary_pi],
            "pooled_sigma_y": pooled_sigma(m0, base.m9_stationary_pi),
            "sigma_split_relative": POOLED_SIGMA_SPLIT,
        },
        # The two sides of F2.5 never shared a volatility level: the pooled
        # baseline is built from M0's sigmas, M9_prod runs on M9's.  The ratio
        # below is what the flat gap actually measures.
        "volatility_level_mismatch": {
            "pooled_sigma_y_from_M0": pooled_sigma(m0, base.m9_stationary_pi),
            "effective_sigma_y_of_M9": effective_sigma(base),
            "ratio_M9_over_pooled": (effective_sigma(base)
                                     / pooled_sigma(m0, base.m9_stationary_pi)),
            "M9_sigma_y": [float(x) for x in base.sigma_y],
        },
    }


_MD_TEMPLATE = """# FW6a -- F2.5 pooled-vs-M9 comparison under the v2 kappa

`outputs/market_calibration_final/model_comparison_pooled_vs_M9.md` (F2.5)
reports every M9-kappa-bearing number at `kappa = {pre_kappa:.6e} /h`
(`phi = 0.999995891734`), the pre-v2 value.  The production frozen-parameter
file now carries `kappa = {new_kappa:g} /h` (`phi = 0.9246`), a factor of
{kappa_ratio:,.0f} faster.  The tables below are the same comparison at the
production kappa, on the production climatology `z(t-1)` path and the
production {nodes}-node residual grid.  Everything else -- forward curve,
`pi_filtered`, TVTP coefficients, strike, maturities, `r_annual`, pooled-sigma
construction -- is as in F2.5.

Pooled baseline: `sigma_pooled = sqrt(pi_normal * M0.sigma0^2 + pi_stress *
M0.sigma1^2) = {pooled:.6f}` with M9's stationary occupancy
`({pi0:.6f}, {pi1:.6f})`, split by +/- {split_pct:.2f} % so that
`sigma_y[1] > sigma_y[0]` still holds.  Note that the same weights applied to
M9's own sigmas give `{effective_m9:.6f}`, a factor {sigma_ratio:.4f} below the
pooled value: the two sides of F2.5 never shared a volatility level, which is
what the Interpretation section below turns on.

## Results at the v2 kappa -- `K = {strike:.0f}` call, PDE only

{table_new}

## Old (pre-v2 kappa) vs new (v2 kappa), side by side

`call_*` and `residual_sd_*` in TRY/MWh; `pct_call` is the change from the
published F2.5 value.

{table_sbs}

`pooled_M0_kappa` carries M0's own kappa and is therefore unchanged between the
two eras; its rows are the control that isolates the kappa edit.

## How the pooled-vs-M9 gap moves

Pre-v2 (reproduces the published F2.5 numbers):

{table_gap_old}

v2 kappa:

{table_gap_new}

## Interpretation

{interpretation}

Files: `f25_v2_kappa.csv` (new), `f25_pre_v2_reproduction.csv` (old,
regenerated at `{pre_kappa:.6e}`), `f25_old_vs_new.csv`, `f25_gap_v2.csv`,
`f25_gap_pre_v2.csv`, `run_manifest.json`.  The accepted F2.5 artefacts under
`outputs/market_calibration_final/` are not modified.
"""


def _interpretation(gap_old: pd.DataFrame, gap_new: pd.DataFrame,
                    new: pd.DataFrame, m9_kappa_per_hour: float,
                    m0_kappa_per_hour: float, pooled: float,
                    effective_m9: float, m0: Dict[str, float],
                    sigma_m9: np.ndarray) -> str:
    """How the F2.5 conclusions move once the production kappa is used."""
    def span(frame: pd.DataFrame, col: str) -> str:
        first, last = float(frame[col].iloc[0]), float(frame[col].iloc[-1])
        return (f"{first:.2f} % at {int(frame['maturity_h'].iloc[0])} h -> "
                f"{last:.2f} % at {int(frame['maturity_h'].iloc[-1])} h")

    half_life = math.log(2.0) / m9_kappa_per_hour
    t_short = float(new["maturity_h"].min())
    saturation = 100.0 * (1.0 - math.exp(-2.0 * m9_kappa_per_hour * t_short))
    prod_calls = new.loc[new["variant"] == "M9_prod", "call_TRY_MWh"]
    flatness = 100.0 * (prod_calls.max() - prod_calls.min()) / prod_calls.mean()
    k_old, k_new = gap_old["kappa_effect_pct"], gap_new["kappa_effect_pct"]
    ptp_old = float(np.ptp(gap_old["pct_vs_pooled_M9_kappa"]))
    ptp_new = float(np.ptp(gap_new["pct_vs_pooled_M9_kappa"]))
    sd_ratio = gap_new["sd_ratio_vs_pooled_M9_kappa"]

    return "\n".join([
        "* **The gap is now flat in maturity, and it is a volatility-level "
        f"gap.**  Against `pooled_M9_kappa` it was {span(gap_old, 'pct_vs_pooled_M9_kappa')} "
        f"before the refit, spanning {ptp_old:.2f} pp; at the production kappa "
        f"it is {span(gap_new, 'pct_vs_pooled_M9_kappa')}, spanning "
        f"{ptp_new:.2f} pp.  With a half-life of {half_life:.2f} h the residual "
        f"variance is already {saturation:.1f} % of its stationary value at "
        f"{t_short:.0f} h, so every maturity prices essentially the same "
        "stationary mixture variance and the M9_prod call is flat in maturity "
        f"(spread {flatness:.1f} % of its mean).",
        "",
        "* **F2.5's claim to isolate the value of regime-conditioning is "
        "withdrawn.**  The two sides of the comparison do not share a "
        "volatility level.  The pooled baseline takes its sigma from M0's "
        f"fitted pair ({m0['sigma0']:.6f}, {m0['sigma1']:.6f}) and lands at "
        f"{pooled:.6f}; M9_prod runs on M9's pair "
        f"({sigma_m9[0]:.6f}, {sigma_m9[1]:.6f}), whose occupancy-weighted "
        f"effective volatility is {effective_m9:.6f} -- a ratio of "
        f"{effective_m9 / pooled:.4f}.  The realised residual-sd ratio at "
        f"expiry is {sd_ratio.min():.4f}-{sd_ratio.max():.4f}, i.e. the whole "
        "flat gap is that sigma ratio.  What the comparison measures is the "
        "disagreement between the M0 and M9 fits about the volatility level, "
        "not the value of holding a filtered regime belief.  Under the pre-v2 "
        "kappa the confound was masked by the term structure; once the "
        "variance saturates inside the shortest maturity the gap collapses "
        "onto the sigma ratio and the confound is visible.",
        "",
        "* **Where the regime-mixture effect is actually measured.**  FW9 "
        "round f prices a single-regime OU at the *same* stationary variance "
        "(`kappa_sensitivity_isovariance_v2.md`): 183.4 TRY/MWh against the "
        "production 166.75 at the 72 h ATM call, so the two-regime mixture is "
        "worth about -9 % at equal variance.  Same sign as the number here, "
        "an order of magnitude smaller: the mixture channel is the thin-tail "
        "correction, and the remaining ~44 pp of the ~-53 % is the M0-vs-M9 "
        "volatility level.  Support for time-varying transitions rests on "
        "FW9's likelihood-ratio test "
        "(`lr_test_TVTP_vs_constant.csv`: LR = 1178.66, df = 2), not on this "
        "comparison.",
        "",
        "* **`pooled_M0_kappa` folds in a third channel.**  Its kappa is M0's "
        f"own, now about {m9_kappa_per_hour / m0_kappa_per_hour:.0f}x slower "
        f"than production rather than "
        f"{m0_kappa_per_hour / PRE_V2_KAPPA_PER_HOUR:.0f}x faster, so its "
        "residual variance keeps growing with maturity.  The "
        f"{span(gap_new, 'pct_vs_pooled_M0_kappa')} column therefore compares "
        "mean-reversion rates on top of the volatility level, and is not a "
        "regime-conditioning measurement either.",
        "",
        "* **F2.5's fourth point inverts.**  It reports the kappa swap alone "
        "(`pooled_M0_kappa` -> `pooled_M9_kappa`, same pooled sigma) as "
        f"raising the call by {k_old.min():.0f}-{k_old.max():.0f} %, an order "
        "of magnitude below what it attributes to regime-conditioning.  At the "
        f"production kappa the same swap moves it by {k_new.max():.0f} to "
        f"{k_new.min():.0f} %, i.e. the mean-reversion timescale is the larger "
        "of the two channels, not the secondary one.",
        "",
        "* **The accepted artefact pair is internally inconsistent.**  The v2 "
        "refit commit regenerated "
        "`outputs/market_calibration_final/model_comparison_pooled_vs_M9.csv` "
        "but not the `.md` beside it, so the published F2.5 tables are the "
        "pre-v2 numbers while its own data source is the v2 ones.  Both files "
        "lie in an accepted output tree and are left unchanged; the "
        "documentation update should mark F2.5 as superseded by this "
        "directory.",
    ])


def write_outputs(outdir: Path, old: pd.DataFrame, new: pd.DataFrame,
                  m9_kappa_per_hour: float, strike: float = STRIKE_TRY_MWH,
                  params_yaml: Path = PARAMS_YAML) -> List[Path]:
    """Write the CSV pack, the manifest and the English results note."""
    outdir = assert_writable(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sbs = side_by_side(old, new)
    gap_old, gap_new = gap_table(old), gap_table(new)
    base = load_frozen_parameters(params_yaml)
    m0 = load_m0_estimates()

    written: List[Path] = []

    def _csv(name: str, frame: pd.DataFrame) -> None:
        p = outdir / name
        frame.to_csv(p, index=False, lineterminator="\n")
        written.append(p)

    _csv("f25_v2_kappa.csv", new)
    _csv("f25_pre_v2_reproduction.csv", old)
    _csv("f25_old_vs_new.csv", sbs)
    _csv("f25_gap_v2.csv", gap_new)
    _csv("f25_gap_pre_v2.csv", gap_old)

    manifest = _manifest(m9_kappa_per_hour, base, m0, float(strike),
                         sorted(int(h) for h in new["maturity_h"].unique()),
                         int(new["n_space_nodes"].iloc[0]))
    p = outdir / "run_manifest.json"
    p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written.append(p)

    gap_cols = ["maturity_h", "call_pooled_M0_kappa", "call_pooled_M9_kappa",
                "call_M9_prod", "pct_vs_pooled_M0_kappa",
                "pct_vs_pooled_M9_kappa"]
    md = _MD_TEMPLATE.format(
        pre_kappa=PRE_V2_KAPPA_PER_HOUR,
        new_kappa=m9_kappa_per_hour,
        kappa_ratio=m9_kappa_per_hour / PRE_V2_KAPPA_PER_HOUR,
        nodes=int(new["n_space_nodes"].iloc[0]),
        pooled=pooled_sigma(m0, base.m9_stationary_pi),
        effective_m9=effective_sigma(base),
        sigma_ratio=(effective_sigma(base)
                     / pooled_sigma(m0, base.m9_stationary_pi)),
        pi0=float(base.m9_stationary_pi[0]),
        pi1=float(base.m9_stationary_pi[1]),
        split_pct=100.0 * POOLED_SIGMA_SPLIT,
        strike=float(strike),
        table_new=_md_table(new[["maturity_h", "variant", "kappa_per_hour",
                                 "F_T_TRY_MWh", "residual_sd_T_TRY_MWh",
                                 "call_TRY_MWh"]], "%.6g"),
        table_sbs=_md_table(sbs[["maturity_h", "variant", "call_TRY_MWh_old",
                                 "call_TRY_MWh_new", "d_call", "pct_call",
                                 "residual_sd_T_TRY_MWh_old",
                                 "residual_sd_T_TRY_MWh_new", "pct_sd"]]),
        table_gap_old=_md_table(gap_old[gap_cols]),
        table_gap_new=_md_table(gap_new[gap_cols]),
        interpretation=_interpretation(
            gap_old, gap_new, new, m9_kappa_per_hour, m0["kappa_per_hour"],
            pooled_sigma(m0, base.m9_stationary_pi),
            effective_sigma(base), m0, base.sigma_y),
    )
    p = outdir / "f25_v2_kappa.md"
    p.write_text(md, encoding="utf-8")
    written.append(p)
    return written


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--strike", type=float, default=STRIKE_TRY_MWH)
    ap.add_argument("--maturities",
                    default=",".join(str(h) for h in MATURITIES_HOURS),
                    help="comma-separated maturities in hours")
    ap.add_argument("--n-space-nodes", type=int, default=N_SPACE_NODES)
    args = ap.parse_args(argv)

    maturities = tuple(int(h) for h in str(args.maturities).split(",")
                       if h.strip())
    v2_kappa = float(load_frozen_parameters(PARAMS_YAML).kappa_per_hour)

    new = run_comparison(v2_kappa, era="v2", maturities_hours=maturities,
                         strike=args.strike, n_space_nodes=args.n_space_nodes)
    old = run_comparison(PRE_V2_KAPPA_PER_HOUR, era="pre_v2",
                         maturities_hours=maturities, strike=args.strike,
                         n_space_nodes=args.n_space_nodes)

    for p in write_outputs(Path(args.outdir), old, new, v2_kappa,
                           strike=float(args.strike)):
        print(f"  wrote {_rel(p)}")
    print(f"FW6a -- F2.5 re-run at kappa = {v2_kappa:g} /h "
          f"(was {PRE_V2_KAPPA_PER_HOUR:.6e} /h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
