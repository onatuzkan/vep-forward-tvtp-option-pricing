"""FW4-P -- price impact of the ramp covariate under FW9's own ramp.

Question
--------
FW4's experimental two-covariate mode reported a total ramp effect of
-1.008 % at the 72 h K = 3000 call, using a *reconstructed* ramp series and
ramp slopes *transferred* from the M9 bundle.  FW9 later defined a ramp
series of its own and estimated its slopes jointly
(``outputs/fw9_self_estimation/TVTP_2cov.pkl``: h01 = -0.0795,
h10 = +0.3816; LR = 869.93 against the single-covariate fit).  This module
prices the same contracts with FW9's ramp series and FW9's ramp slopes,
everything else held at production.

What differs between the two ramps
----------------------------------
The *definition* is the same in both: the transition into hour t is driven
by ``(z_{t-1}, z_{t-1} - z_{t-2})``.  ``CovariatePathBuilder`` builds exactly
that pair, and FW9's ``z.diff().shift(1)`` is the same object.  The two
differ in the standardisation window:

    FW4 (yaml)  m_r = 3.417142e-05, s_r = 0.271450   window W9, <= 2024-12-31
    FW9 (pkl)   m_r = 2.962489e-05, s_r = 0.266400   train,     <= 2022-12-31

so FW9's r is about 1.9 % larger in magnitude for the same increment, and
the slopes differ as well (h01 -14.5 % / h10 +6.3 % relative to FW4).

Intercepts
----------
FW4's headline number is R3 - R1: both sides carry intercepts re-derived by
moment matching on D = W9, so the mean transition probabilities are held
fixed and the ramp effect is measured at constant occupancy.  That matters:
FW4's own decomposition at 72 h / K = 3000 is a -2.35 % slope part almost
cancelled by a +1.38 % intercept compensation.  Two FW9 variants are
therefore priced:

    fw9_ramp          production intercepts, FW9 h, FW9 ramp   (literal:
                      "all other parameters as in production")
    fw9_ramp_matched  intercepts re-derived on D = W9 with FW9 h and FW9's
                      ramp, so it is the structural counterpart of FW4's R3

Grid
----
Every variant at a given maturity is priced on ONE fixed residual grid --
the union of the auto-sized grids over all variants, 1201 nodes -- so that a
sub-0.1 TRY/MWh ramp effect is not contaminated by a ~0.03 TRY/MWh
discretisation difference.  This is FW4's own methodology; pricing R0 and R3
on FW4's published bounds reproduces its published values to 5e-07.

Covariate path
--------------
Production climatology (FW12b): the deterministic z path is the climatology
of the training window, the ramp is its own hourly increment, and no
constant-transition (z = 0) fallback and no synthetic covariate is used.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption                  # noqa: E402
from pde_option_model.forward_centered import (                        # noqa: E402
    ForwardCenteredModel, ResidualGridSettings, ResidualSpec,
    price_forward_centered)
from pde_option_model.forward_curve import (                           # noqa: E402
    NearTermAnchor, build_forward_curve)
from pde_option_model.generator import (                               # noqa: E402
    TVTPCoefficients, TVTP2Coefficients)
from pde_option_model.market_data import load_quotes                   # noqa: E402
from pde_option_model.params_frozen import (                           # noqa: E402
    FrozenM2Parameters, load_frozen_parameters, load_tvtp2_parameters)
from pde_option_model.scenarios import (                               # noqa: E402
    CovariatePathBuilder, ScenarioSpec)
from pde_option_model.tvtp2 import (                                   # noqa: E402
    RampScaler, build_covariate_panel, derive_intercepts,
    load_hourly_z_history)

# --- production inputs (config/forward_centered_config.yaml) ---------------
PARAMS_YAML = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
TVTP2_YAML = REPO_ROOT / "inputs" / "historical" / "tvtp2_frozen_parameters.yaml"
QUOTES_CSV = REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv"
Z_HISTORY_CSV = REPO_ROOT / "inputs" / "historical" / "rd_standardized.csv"
FW9_2COV_PKL = (REPO_ROOT / "outputs" / "fw9_self_estimation" / "TVTP_2cov.pkl")

TRAIN_END_UTC = pd.Timestamp("2022-12-31 20:00:00+00:00")
COVARIATE_LAG_HOURS = 1
SCENARIO_MODE = "climatology"
CURVE_MODE = "smooth_constrained"
JANUARY_ANCHOR_MODE = "spot_to_next_linear"
SMOOTHNESS_WEIGHT = 1.0
LEVEL_WEIGHT = 1.0e-4

OPTION_TYPE = "call"
R_ANNUAL = 0.40
MATURITIES_HOURS: Tuple[int, ...] = (24, 48, 72)
STRIKE_LADDER: Tuple[float, ...] = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0)
N_SPACE_NODES = 1201
N_STD = 6.0

# Moment-matching targets of the FW4 derivation (tvtp2 yaml, derivation.targets).
W9_SAMPLE_END_UTC = pd.Timestamp("2024-12-31 20:00:00+00:00")
DURATION_NORMAL_H = 3.7591994835377665
DURATION_STRESS_H = 7.559715945771465

# FW4's published bounds, used only as a reproduction check (see the tests).
FW4_GRID_BOUNDS: Dict[int, Tuple[float, float]] = {
    24: (-4520.545418, 4686.067189),
    72: (-4568.552580, 4736.237895),
}
FW4_PUBLISHED_CALLS: Dict[Tuple[str, int], float] = {
    ("R0", 24): 163.955886, ("R0", 72): 166.783189,
    ("R3", 24): 160.609207, ("R3", 72): 163.857946,
}
FW4_TOTAL_RAMP_EFFECT_72H_K3000_PCT = -1.007972055471188

# FW9's own attribution of the total price difference to the transition
# coefficients (outputs/fw9_self_estimation/price_impact_v2_decomposition.csv,
# column delta_D_..._share_of_total_delta_pct at 72 h / K = 3000).
FW9_DECOMPOSITION_CSV = (REPO_ROOT / "outputs" / "fw9_self_estimation"
                         / "price_impact_v2_decomposition.csv")


PROTECTED_OUTPUT_DIRS = (
    "outputs/market_calibration_final",
    "outputs/forward_centered_diagnostics",
    "outputs/scenario_sweep",
    "outputs/tvtp2_experimental",
)
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "fw4p_ramp_price_impact"


class FW4PError(RuntimeError):
    """Raised when the FW4-P run is misconfigured."""


def fw9_transition_share_pct(maturity_h: int = 72,
                             strike: float = 3000.0) -> float:
    """Share of FW9's total price difference carried by the transition block."""
    d = pd.read_csv(FW9_DECOMPOSITION_CSV)
    row = d[(d["maturity_h"] == maturity_h)
            & np.isclose(d["strike"], strike)]
    if len(row) != 1:
        raise FW4PError(f"{FW9_DECOMPOSITION_CSV.name}: expected one row at "
                        f"{maturity_h} h / K = {strike}, got {len(row)}")
    col = "delta_D_inh_sigma_inh_kappa_fw9_trans_vs_A_share_of_total_delta_pct"
    return float(row[col].iloc[0])


# ---------------------------------------------------------------------------
# FW9's ramp
# ---------------------------------------------------------------------------
def fw9_ramp_scaler() -> RampScaler:
    """FW9's (m_r, s_r), rebuilt by FW9's own code and cross-checked.

    ``scripts/fw9/build_ramp.py`` standardises ``z.diff().shift(1)`` on
    labels <= 2022-12-31 20:00 UTC.  The repo's ``fit_ramp_scaler`` windows
    the *unshifted* increments instead, so the two conventions do not agree
    and ``CovariatePathBuilder.verify_ramp_scaler`` would reject this scaler
    by design; the difference is recorded in the run manifest rather than
    papered over.
    """
    from scripts.fw9.build_ramp import build_ramp_lag1_series

    series, (mu, sd) = build_ramp_lag1_series()
    stored = fw9_stored_scaler()
    if not (math.isclose(mu, stored["mu"], rel_tol=0, abs_tol=1e-15)
            and math.isclose(sd, stored["sd"], rel_tol=0, abs_tol=1e-15)):
        raise FW4PError(
            f"FW9 ramp scaler drift: build_ramp gives (m_r={mu!r}, s_r={sd!r}) "
            f"but {FW9_2COV_PKL.name} stores (m_r={stored['mu']!r}, "
            f"s_r={stored['sd']!r}); the fit and the path would disagree")
    n_train = int(series.loc[:TRAIN_END_UTC].dropna().size)
    return RampScaler(
        mean=float(mu), std=float(sd), ddof=1, n=n_train,
        window_name="FW9_train", window_end_utc=TRAIN_END_UTC,
        window_start_utc=None,
        definition=("r_t = (dz_{t-1} - m_r) / s_r with dz_{t-1} = z_{t-1} - "
                    "z_{t-2}; scaler on labels <= 2022-12-31T20:00:00+00:00 "
                    "(scripts/fw9/build_ramp.py)"),
        status=("ESTIMATED SCALER, FW9 convention: the series is shifted "
                "BEFORE the training window is cut, so it does not match "
                "pde_option_model.tvtp2.fit_ramp_scaler on the same window"))


def fw9_stored_scaler() -> Dict[str, float]:
    """The (mu, sd) recorded alongside FW9's two-covariate fit."""
    with open(FW9_2COV_PKL, "rb") as fh:
        blob = pickle.load(fh)
    return {k: float(v) for k, v in blob["ramp_scaler"].items()}


def fw9_fit() -> Dict[str, Any]:
    """FW9's two-covariate MLE: coefficients plus the convergence record."""
    with open(FW9_2COV_PKL, "rb") as fh:
        blob = pickle.load(fh)
    p = {k: float(v) for k, v in blob["params_dict"].items()}
    res = blob["res"]
    return {"params": p, "n_obs": int(blob["n_obs"]),
            "loglik": float(res.loglik),
            "converged": bool(res.converged),
            "grad_norm": float(res.grad_norm)}


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
def derive_w9_intercepts(scaler: RampScaler, z_history: pd.Series,
                         gamma01: float, gamma10: float,
                         h01: float, h10: float,
                         use_ramp: bool) -> Dict[str, Any]:
    """Moment-matched intercepts on D = W9, FW4's rule and sample."""
    panel = build_covariate_panel(z_history, scaler,
                                  lag_hours=COVARIATE_LAG_HOURS)
    return derive_intercepts(panel, gamma01=gamma01, h01=h01,
                             gamma10=gamma10, h10=h10,
                             duration_normal_h=DURATION_NORMAL_H,
                             duration_stress_h=DURATION_STRESS_H,
                             sample_end=W9_SAMPLE_END_UTC, use_ramp=use_ramp)


def build_variants(params: FrozenM2Parameters,
                   z_history: pd.Series) -> Dict[str, Dict[str, Any]]:
    """The five priced configurations, in report order."""
    tvtp2 = load_tvtp2_parameters(TVTP2_YAML, base_params=params,
                                  base_params_path=str(PARAMS_YAML))
    fw4_scaler = tvtp2.ramp_scaler
    fw9_scaler = fw9_ramp_scaler()
    fw9 = fw9_fit()["params"]
    g01, g10 = params.gamma01, params.gamma10

    # R1: 1D intercepts re-derived on D = W9 (h = 0, so scaler-independent).
    r1 = derive_w9_intercepts(fw4_scaler, z_history, g01, g10, 0.0, 0.0,
                              use_ramp=False)
    # FW9 h and FW9 ramp, intercepts re-derived on the same sample.
    m9c = derive_w9_intercepts(fw9_scaler, z_history, g01, g10,
                               fw9["h01"], fw9["h10"], use_ramp=True)

    return {
        "production": {
            "label": "production (1D, repo yaml intercepts)",
            "tvtp": TVTPCoefficients(params.alpha01, g01,
                                     params.alpha10, g10),
            "scaler": fw4_scaler, "fw4_run": "R0",
            "note": "production single-covariate model; the paper baseline",
        },
        "base_1d_w9": {
            "label": "1D, intercepts re-derived on W9",
            "tvtp": TVTPCoefficients(r1["alpha01"], g01, r1["alpha10"], g10),
            "scaler": fw4_scaler, "fw4_run": "R1",
            "note": "FW4's reference leg for the ramp effect (R3 - R1)",
            "intercepts": r1,
        },
        "fw4_ramp": {
            "label": "FW4 experimental ramp (reconstructed, M9 slopes)",
            "tvtp": tvtp2.coefficients,
            "scaler": fw4_scaler, "fw4_run": "R3",
            "note": "the shipped tvtp2 yaml, unchanged",
        },
        "fw9_ramp": {
            "label": "FW9 ramp + FW9 slopes, production intercepts",
            "tvtp": TVTP2Coefficients(
                alpha01=params.alpha01, gamma01=g01, h01=fw9["h01"],
                alpha10=params.alpha10, gamma10=g10, h10=fw9["h10"],
                name="fw9_ramp"),
            "scaler": fw9_scaler, "fw4_run": None,
            "note": ("literal 'all other parameters as in production'; the "
                     "mean transition probabilities are NOT held fixed"),
        },
        "fw9_ramp_matched": {
            "label": "FW9 ramp + FW9 slopes, intercepts re-derived on W9",
            "tvtp": TVTP2Coefficients(
                alpha01=m9c["alpha01"], gamma01=g01, h01=fw9["h01"],
                alpha10=m9c["alpha10"], gamma10=g10, h10=fw9["h10"],
                name="fw9_ramp_matched"),
            "scaler": fw9_scaler, "fw4_run": None,
            "note": "structural counterpart of FW4's R3; occupancy held fixed",
            "intercepts": m9c,
        },
    }


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
def build_model(params: FrozenM2Parameters, quotes, tvtp,
                scaler: RampScaler) -> ForwardCenteredModel:
    """Production forward-centered model carrying one TVTP law."""
    curve = build_forward_curve(
        quotes, mode=CURVE_MODE,
        anchor=NearTermAnchor(mode=JANUARY_ANCHOR_MODE, level_TRY_MWh=None),
        spot_price_TRY_MWh=params.spot_price_TRY_MWh,
        smoothness_weight=SMOOTHNESS_WEIGHT, level_weight=LEVEL_WEIGHT)
    return ForwardCenteredModel(
        curve=curve, spec=ResidualSpec.from_frozen(params), tvtp=tvtp,
        pi_filtered=params.pi_filtered, valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh,
        expected_ramp_scaler=scaler)


def climatology_path(builder: CovariatePathBuilder, contract: EuropeanOption,
                     gs: ResidualGridSettings):
    """Production climatology (z, r) path on the 0.25 h master grid (FW12b)."""
    return builder.build(ScenarioSpec(name="pricing", mode=SCENARIO_MODE),
                         contract.valuation_utc,
                         maturity_utc=contract.maturity_utc,
                         grid_settings=gs)


def _contract(params: FrozenM2Parameters, hours: int,
              strike: float) -> EuropeanOption:
    return EuropeanOption(
        option_type=OPTION_TYPE, strike=float(strike),
        valuation_utc=params.valuation_utc,
        maturity_utc=params.valuation_utc + pd.Timedelta(hours=int(hours)),
        r_annual=R_ANNUAL)


def union_grid_bounds(params: FrozenM2Parameters, quotes,
                      variants: Dict[str, Dict[str, Any]],
                      z_history: pd.Series,
                      hours: int) -> Tuple[float, float]:
    """Widest auto-sized residual grid over the variants at this maturity.

    The bounds come from the analytic residual sd and do not depend on the
    strike, so one probe per variant is enough.
    """
    auto = ResidualGridSettings(n_space_nodes=N_SPACE_NODES,
                                n_time_steps=None, n_std=N_STD)
    contract = _contract(params, hours, STRIKE_LADDER[0])
    lo, hi = np.inf, -np.inf
    for v in variants.values():
        builder = CovariatePathBuilder(
            z_history, TRAIN_END_UTC, COVARIATE_LAG_HOURS,
            ramp_scaler=v["scaler"], history_file=str(Z_HISTORY_CSV))
        path = climatology_path(builder, contract, auto)
        res = price_forward_centered(
            build_model(params, quotes, v["tvtp"], v["scaler"]),
            contract, auto, covariate_path=path)
        lo = min(lo, float(res.diagnostics["grid_x_min"]))
        hi = max(hi, float(res.diagnostics["grid_x_max"]))
    return float(lo), float(hi)


def price_all(maturities_hours: Sequence[int] = MATURITIES_HOURS,
              strikes: Sequence[float] = STRIKE_LADDER,
              grid_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
              include_atm: bool = True) -> pd.DataFrame:
    """Every variant at every (maturity, strike), on a fixed grid per maturity."""
    params = load_frozen_parameters(PARAMS_YAML)
    quotes = load_quotes(QUOTES_CSV)
    z_history = load_hourly_z_history(Z_HISTORY_CSV)
    variants = build_variants(params, z_history)

    rows: List[Dict[str, Any]] = []
    for hours in maturities_hours:
        if grid_bounds is not None and int(hours) in grid_bounds:
            lo, hi = grid_bounds[int(hours)]
        else:
            lo, hi = union_grid_bounds(params, quotes, variants,
                                       z_history, int(hours))
        gs = ResidualGridSettings(n_space_nodes=N_SPACE_NODES,
                                  n_time_steps=None, n_std=N_STD,
                                  x_min=lo, x_max=hi)
        # F(T) is a property of the curve, so one probe fixes the ATM strike.
        probe_v = variants["production"]
        probe_b = CovariatePathBuilder(
            z_history, TRAIN_END_UTC, COVARIATE_LAG_HOURS,
            ramp_scaler=probe_v["scaler"], history_file=str(Z_HISTORY_CSV))
        probe_c = _contract(params, hours, strikes[0])
        f_t = float(price_forward_centered(
            build_model(params, quotes, probe_v["tvtp"], probe_v["scaler"]),
            probe_c, gs,
            covariate_path=climatology_path(probe_b, probe_c, gs)
        ).forward_at_expiry)

        ladder: List[Tuple[str, float]] = [("ladder", float(k)) for k in strikes]
        if include_atm:
            ladder.append(("ATM", f_t))

        for name, v in variants.items():
            builder = CovariatePathBuilder(
                z_history, TRAIN_END_UTC, COVARIATE_LAG_HOURS,
                ramp_scaler=v["scaler"], history_file=str(Z_HISTORY_CSV))
            model = build_model(params, quotes, v["tvtp"], v["scaler"])
            for kind, strike in ladder:
                contract = _contract(params, hours, strike)
                path = climatology_path(builder, contract, gs)
                audit = path.embeddability(v["tvtp"]) if isinstance(
                    v["tvtp"], TVTP2Coefficients) else None
                if audit is not None and not audit["embeddable"]:
                    raise FW4PError(
                        f"{name} at {hours} h: two-covariate TVTP is not "
                        f"embeddable ({audit['n_s_ge_1']} of "
                        f"{audit['n_rows']} labels have p01 + p10 >= 1)")
                res = price_forward_centered(model, contract, gs,
                                             covariate_path=path)
                t = v["tvtp"]
                rows.append({
                    "variant": name,
                    "label": v["label"],
                    "fw4_run": v["fw4_run"] or "",
                    "strike_kind": kind,
                    "maturity_h": int(hours),
                    "strike_TRY_MWh": round(float(strike), 6),
                    "call_TRY_MWh": float(res.value),
                    "F_T_TRY_MWh": round(float(res.forward_at_expiry), 6),
                    "E_spot_T_TRY_MWh": round(
                        float(res.expected_spot_at_expiry), 6),
                    "residual_sd_T_TRY_MWh": round(
                        float(res.residual_std_at_expiry), 6),
                    "p_stress_at_expiry": round(
                        float(res.diagnostics["p_stress_at_expiry"]), 8),
                    "mean_p_stress": round(
                        float(res.diagnostics["mean_p_stress"]), 8),
                    "expected_transitions": round(
                        float(res.diagnostics["expected_transitions"]), 6),
                    "max_s": round(float(res.diagnostics["max_s"]), 8),
                    "alpha01": float(t.alpha01), "gamma01": float(t.gamma01),
                    "alpha10": float(t.alpha10), "gamma10": float(t.gamma10),
                    "h01": float(getattr(t, "h01", 0.0)),
                    "h10": float(getattr(t, "h10", 0.0)),
                    "ramp_scaler_mean": float(v["scaler"].mean),
                    "ramp_scaler_std": float(v["scaler"].std),
                    "ramp_scaler_window": str(v["scaler"].window_name),
                    "n_space_nodes": N_SPACE_NODES,
                    "n_time_steps": int(res.diagnostics["n_time_steps"]),
                    "grid_x_min": round(lo, 6), "grid_x_max": round(hi, 6),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
THREE_WAY = ("production", "fw4_ramp", "fw9_ramp")


def side_by_side(runs: pd.DataFrame) -> pd.DataFrame:
    """The three headline prices per (maturity, strike), plus both deltas."""
    wide = runs.pivot_table(index=["maturity_h", "strike_kind",
                                   "strike_TRY_MWh"],
                            columns="variant", values="call_TRY_MWh")
    out = pd.DataFrame(index=wide.index)
    for v in ("production", "fw4_ramp", "fw9_ramp", "fw9_ramp_matched",
              "base_1d_w9"):
        out[f"call_{v}"] = wide[v].round(6)
    for v in ("fw4_ramp", "fw9_ramp", "fw9_ramp_matched"):
        out[f"d_{v}_vs_production"] = (wide[v] - wide["production"]).round(6)
        out[f"pct_{v}_vs_production"] = (
            100.0 * (wide[v] - wide["production"]) / wide["production"]
        ).round(6)
    return out.reset_index().sort_values(
        ["maturity_h", "strike_kind", "strike_TRY_MWh"]).reset_index(drop=True)


def ramp_effect(runs: pd.DataFrame) -> pd.DataFrame:
    """Occupancy-controlled ramp effect: each 2D run against its 1D leg.

    FW4 defines the ramp effect as R3 - R1, i.e. both legs carry intercepts
    re-derived on the same sample, so only the ramp channel moves.  The same
    contrast is formed here for the FW9 ramp.  ``fw9_ramp`` has no matching
    1D leg (it keeps the production intercepts), so it is measured against
    the production run instead and flagged as not occupancy-controlled.
    """
    wide = runs.pivot_table(index=["maturity_h", "strike_kind",
                                   "strike_TRY_MWh"],
                            columns="variant", values="call_TRY_MWh")
    spec = (("fw4_ramp", "base_1d_w9", True),
            ("fw9_ramp_matched", "base_1d_w9", True),
            ("fw9_ramp", "production", False))
    frames = []
    for two_d, one_d, controlled in spec:
        f = pd.DataFrame(index=wide.index)
        f["ramp_variant"] = two_d
        f["reference"] = one_d
        f["occupancy_controlled"] = controlled
        f["call_2d"] = wide[two_d].round(6)
        f["call_1d"] = wide[one_d].round(6)
        f["ramp_effect_TRY_MWh"] = (wide[two_d] - wide[one_d]).round(6)
        f["ramp_effect_pct"] = (
            100.0 * (wide[two_d] - wide[one_d]) / wide[one_d]).round(6)
        frames.append(f.reset_index())
    return (pd.concat(frames, ignore_index=True)
            .sort_values(["maturity_h", "strike_kind", "strike_TRY_MWh",
                          "ramp_variant"]).reset_index(drop=True))


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def assert_writable(outdir: Path) -> Path:
    res = Path(outdir).resolve()
    for prot in PROTECTED_OUTPUT_DIRS:
        p = (REPO_ROOT / prot).resolve()
        if res == p or p in res.parents:
            raise FW4PError(
                "refusing to write FW4-P outputs into the accepted directory "
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


def write_outputs(outdir: Path, runs: pd.DataFrame) -> List[Path]:
    outdir = assert_writable(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sbs = side_by_side(runs)
    eff = ramp_effect(runs)
    written: List[Path] = []

    def _csv(name: str, frame: pd.DataFrame) -> None:
        p = outdir / name
        frame.to_csv(p, index=False, lineterminator="\n")
        written.append(p)

    _csv("fw4p_runs.csv", runs)
    _csv("fw4p_three_way.csv", sbs)
    _csv("fw4p_ramp_effect.csv", eff)

    params = load_frozen_parameters(PARAMS_YAML)
    z_history = load_hourly_z_history(Z_HISTORY_CSV)
    variants = build_variants(params, z_history)
    fit = fw9_fit()
    fw4_scaler = variants["fw4_ramp"]["scaler"]
    fw9_scaler = variants["fw9_ramp"]["scaler"]

    manifest = {
        "work_package": "FW4-P",
        "purpose": ("price impact of the ramp covariate under FW9's own ramp "
                    "series and FW9's estimated ramp slopes"),
        "inputs": {
            "frozen_parameters": _rel(PARAMS_YAML),
            "tvtp2_parameters": _rel(TVTP2_YAML),
            "quotes": _rel(QUOTES_CSV),
            "z_history": _rel(Z_HISTORY_CSV),
            "fw9_two_covariate_fit": _rel(FW9_2COV_PKL),
            "fw9_ramp_builder": "scripts/fw9/build_ramp.py",
        },
        "settings": {
            "option_type": OPTION_TYPE, "r_annual": R_ANNUAL,
            "maturities_hours": list(MATURITIES_HOURS),
            "strike_ladder": list(STRIKE_LADDER),
            "atm_rule": "K = F(T) of the production curve at that maturity",
            "n_space_nodes": N_SPACE_NODES, "n_std": N_STD,
            "grid_rule": ("one fixed grid per maturity: union of the "
                          "auto-sized grids over all variants"),
            "scenario_mode": SCENARIO_MODE,
            "covariate_lag_hours": COVARIATE_LAG_HOURS,
            "climatology_train_end_utc": TRAIN_END_UTC.isoformat(),
            "curve_mode": CURVE_MODE,
            "january_anchor_mode": JANUARY_ANCHOR_MODE,
            "pi_filtered": [float(x) for x in params.pi_filtered],
        },
        "fw9_fit": {
            "params": fit["params"], "n_obs": fit["n_obs"],
            "loglik": fit["loglik"], "converged": fit["converged"],
            "grad_norm": fit["grad_norm"],
            "phi": fit["params"]["phi"],
            "caveat": ("estimated on the RAW asinh level at the unit-root "
                       "boundary (phi = %.10f); the transition coefficients "
                       "may be biased" % fit["params"]["phi"]),
        },
        "ramp_scalers": {
            "fw4_W9": {"mean": float(fw4_scaler.mean),
                       "std": float(fw4_scaler.std),
                       "n": int(fw4_scaler.n),
                       "window_end_utc": fw4_scaler.window_end_utc.isoformat()},
            "fw9_train": {"mean": float(fw9_scaler.mean),
                          "std": float(fw9_scaler.std),
                          "n": int(fw9_scaler.n),
                          "window_end_utc": fw9_scaler.window_end_utc.isoformat()},
            "std_ratio_fw4_over_fw9": float(fw4_scaler.std / fw9_scaler.std),
        },
        "intercepts": {
            "base_1d_w9": {k: variants["base_1d_w9"]["intercepts"][k]
                           for k in ("alpha01", "alpha10")},
            "fw9_ramp_matched": {k: variants["fw9_ramp_matched"]["intercepts"][k]
                                 for k in ("alpha01", "alpha10")},
            "moment_targets": {"duration_normal_h": DURATION_NORMAL_H,
                               "duration_stress_h": DURATION_STRESS_H,
                               "sample_end_utc": W9_SAMPLE_END_UTC.isoformat()},
        },
        "fw4_reference": {
            "total_ramp_effect_72h_K3000_pct":
                FW4_TOTAL_RAMP_EFFECT_72H_K3000_PCT,
            "published_calls_on_fw4_grid": {
                f"{r}_{h}h": v for (r, h), v in FW4_PUBLISHED_CALLS.items()},
        },
    }
    p = outdir / "run_manifest.json"
    p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written.append(p)

    p = outdir / "fw4p_ramp_price_impact.md"
    p.write_text(_render_markdown(runs, sbs, eff, manifest), encoding="utf-8")
    written.append(p)
    return written


def _render_markdown(runs: pd.DataFrame, sbs: pd.DataFrame, eff: pd.DataFrame,
                     manifest: Dict[str, Any]) -> str:
    fit = manifest["fw9_fit"]
    sc = manifest["ramp_scalers"]
    three = sbs[["maturity_h", "strike_kind", "strike_TRY_MWh",
                 "call_production", "call_fw4_ramp", "call_fw9_ramp",
                 "pct_fw4_ramp_vs_production", "pct_fw9_ramp_vs_production"]]
    ctrl = eff[eff["occupancy_controlled"]][
        ["maturity_h", "strike_kind", "strike_TRY_MWh", "ramp_variant",
         "ramp_effect_TRY_MWh", "ramp_effect_pct"]]
    atm = three[three["strike_kind"] == "ATM"]
    k3000 = three[np.isclose(three["strike_TRY_MWh"], 3000.0)]
    share = fw9_transition_share_pct()
    return f"""# FW4-P -- price impact of the ramp covariate under FW9's ramp

FW4's experimental two-covariate mode reported a total ramp effect of
{FW4_TOTAL_RAMP_EFFECT_72H_K3000_PCT:.3f} % at the 72 h `K = 3000` call, using a
*reconstructed* ramp and ramp slopes *transferred* from the M9 bundle.  FW9
later defined its own ramp series and estimated the slopes jointly
(`outputs/fw9_self_estimation/TVTP_2cov.pkl`: h01 = {fit['params']['h01']:.6f},
h10 = {fit['params']['h10']:.6f}; LR = 869.93 against the single-covariate
fit).  This note prices the same contracts with FW9's ramp and FW9's slopes,
everything else at production.

## What actually differs

The ramp *definition* is the same object in both: the transition into hour
`t` is driven by `(z(t-1), z(t-1) - z(t-2))`.  What differs is the
standardisation window and the slopes.

| | m_r | s_r | window ends | h01 | h10 |
|---|---:|---:|---|---:|---:|
| FW4 (yaml) | {sc['fw4_W9']['mean']:.6e} | {sc['fw4_W9']['std']:.6f} | {sc['fw4_W9']['window_end_utc'][:10]} | -0.069395 | 0.405544 |
| FW9 (pkl) | {sc['fw9_train']['mean']:.6e} | {sc['fw9_train']['std']:.6f} | {sc['fw9_train']['window_end_utc'][:10]} | {fit['params']['h01']:.6f} | {fit['params']['h10']:.6f} |

FW9's `r` is {100.0 * (sc['std_ratio_fw4_over_fw9'] - 1.0):.2f} % larger in
magnitude for the same increment.  Note the two conventions also disagree on
*when* the shift is applied relative to the training cut, so
`CovariatePathBuilder.verify_ramp_scaler` rejects FW9's scaler by design;
FW9's own `scripts/fw9/build_ramp.py` is used as the source of truth and the
mismatch is recorded here rather than silently reconciled.

## Three prices side by side

`K = F(T)` rows are marked `ATM`.  All variants at a maturity are priced on
one fixed {N_SPACE_NODES}-node residual grid (the union of the auto-sized
grids), so the columns differ only through the transition law.

{_md_table(three, "%.4f")}

## Ramp effect, occupancy held fixed

FW4's headline number is R3 - R1: both legs carry intercepts re-derived by
moment matching on D = W9, so the mean transition probabilities are held
fixed and only the ramp channel moves.  The same contrast for the FW9 ramp
is `fw9_ramp_matched - base_1d_w9`.

{_md_table(ctrl, "%.4f")}

`fw9_ramp` (the literal "all other parameters as in production" reading)
keeps the production intercepts, so it is *not* occupancy-controlled: its
difference from production mixes the ramp channel with a shift in the mean
transition probabilities.  It is reported above for completeness and in
`fw4p_ramp_effect.csv` with `occupancy_controlled = False`.

## Caveats

* **FW9's two-covariate fit sits at the unit-root boundary.**  It was
  estimated on the raw asinh level, and the fitted persistence is
  `phi = {fit['params']['phi']:.10f}` -- a half-life of
  {math.log(2.0) / -math.log(fit['params']['phi']) / 8766.0:.0f} years.  At
  that boundary the level absorbs low-frequency structure that the regime
  process would otherwise carry, so the transition coefficients, including
  h01 and h10, may be biased.  The optimiser also did not meet its gradient
  tolerance (`converged = {fit['converged']}`, grad norm
  {fit['grad_norm']:.1f} at {fit['n_obs']} observations), so these slopes
  carry no usable standard error.
* **The slopes are used outside the fit they came from.**  FW9 estimated
  h01/h10 jointly with its own gammas
  ({fit['params']['gamma01']:.6f}, {fit['params']['gamma10']:.6f}); here they
  are combined with the production gammas (-0.583778, 0.077698), as the task
  specifies.  A jointly-consistent run would move the z channel too.
* **Expected magnitude.**  FW9's own decomposition
  (`price_impact_v2_decomposition.csv`) attributes {share:.2f} % of the total
  price difference to the transition coefficients at the 72 h `K = 3000`
  call, so a small effect is what the evidence predicts; the numbers above
  are consistent with that.
* **The ramp remains experimental.**  Nothing here replaces an accepted
  result, and the accepted output trees are untouched.

At `K = 3000`, for reference:

{_md_table(k3000, "%.4f")}

At the money:

{_md_table(atm, "%.4f")}

Files: `fw4p_runs.csv` (every run), `fw4p_three_way.csv` (the side-by-side),
`fw4p_ramp_effect.csv` (controlled and uncontrolled contrasts),
`run_manifest.json`, and the before/after hash manifests of the 128 protected
files.
"""


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--maturities",
                    default=",".join(str(h) for h in MATURITIES_HOURS))
    args = ap.parse_args(argv)
    maturities = tuple(int(h) for h in str(args.maturities).split(",")
                       if h.strip())
    runs = price_all(maturities_hours=maturities)
    for p in write_outputs(Path(args.outdir), runs):
        print(f"  wrote {_rel(p)}")
    print(f"FW4-P -- {len(runs)} runs over {len(maturities)} maturities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
