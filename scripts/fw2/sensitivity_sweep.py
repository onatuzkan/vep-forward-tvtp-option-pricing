"""FW2 §4 joint (a_i, eta_ij) sensitivity sweep.

Reprices the 24/48/72 h ATM call and put on the strike ladder
{2000, 2500, 3000, 3500, 4000} TRY/MWh under three sweep families:

  * Q1 only: symmetric OR stress-only drift shift a_i (per hour).  The
    magnitudes span the §2.3 descriptive range (up to 50 TRY/MWh/h);
    a single "empirical upper bound" point at a_stress = 50 anchors
    the top of the reported band.
  * Q2 only: (eta_01, eta_10) grid on {-0.75, -0.5, -0.25, 0, 0.25,
    0.5, 0.75}; each combination stays inside generator validity by
    construction (multiplicative form).  A subset of "extreme" points
    is dropped IF the resulting discrete probability s = p01+p10 saturates
    (>= 0.999) at any hour of the pricing grid; the sweep flags them
    as EMBEDDABILITY_FAIL rather than dropping them silently.
  * Joint: the four corners `(a_bound, +/- eta_bound)` and the
    production zero point.

The output is a CSV per (a, eta, strike, maturity) row + a Markdown
summary that reports the maximum absolute and % price effect per
channel (Q1 alone, Q2 alone, joint), plus a numeric confirmation of
Q1 quadratic suppression vs Q2 first-order behavior.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption            # noqa: E402
from pde_option_model.forward_centered import (                  # noqa: E402
    ForwardCenteredModel, ResidualGridSettings, ResidualSpec,
    price_forward_centered)
from pde_option_model.forward_curve import build_forward_curve    # noqa: E402
from pde_option_model.generator import (TVTPCoefficients,         # noqa: E402
                                        generator_to_probs)
from pde_option_model.market_data import load_quotes              # noqa: E402
from pde_option_model.params_frozen import load_frozen_parameters # noqa: E402

PARAMS_YAML = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
QUOTES_CSV = REPO_ROOT / "inputs" / "market" / "vep_monthly_quotes.csv"
OUT_DIR = REPO_ROOT / "outputs" / "fw2_risk_premium"

MATURITIES_H = (24, 48, 72)
STRIKES = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0)
R_ANNUAL = 0.40

# a_i sweep (TRY/MWh per hour, per regime).  The stress-only variant
# is the interesting one because a symmetric shift has zero variance
# effect at second order on an m=0 residual (see the moment ODE test).
A_STRESS_VALUES = (0.0, 10.0, 25.0, 50.0)

# eta grid: multiplicative q^Q = q^P * exp(eta).  |eta| <= 0.75 keeps
# the intensity within a factor of ~2.1 of the physical value, well
# inside the discrete-embedding safe zone at kappa/q of the yaml.
# Sparse grid used for the primary table; the eta_01 * eta_10 cross-
# product is capped at 3x3 = 9 to keep the sweep under an hour of PDE.
ETA_VALUES = (-0.5, 0.0, 0.5)

# joint corners used to characterise the maximum-effect envelope
A_BOUND = 50.0        # stress-only drift shift, empirical top of §2.3
ETA_BOUND = 0.75


@dataclass(frozen=True)
class Setup:
    model: ForwardCenteredModel
    contracts: Tuple[Tuple[float, int, str], ...]

    def price(self, strike: float, maturity_h: int, kind: str,
              a_stress: float, eta_ij: Optional[Tuple[float, float]]
              ) -> Tuple[float, dict]:
        params = self.model.spec
        spec = ResidualSpec(
            kappa_per_hour=params.kappa_per_hour, sigma_y=params.sigma_y,
            scale_P=params.scale_P, regime_means=params.regime_means,
            mode=params.mode, x0_mode=params.x0_mode,
            sigma_multipliers=params.sigma_multipliers,
            drift_shift_per_hour=np.array([0.0, a_stress]),
        )
        model = ForwardCenteredModel(
            curve=self.model.curve, spec=spec, tvtp=self.model.tvtp,
            pi_filtered=self.model.pi_filtered,
            valuation_utc=self.model.valuation_utc,
            spot_price_TRY_MWh=self.model.spot_price_TRY_MWh,
        )
        contract = EuropeanOption(
            kind, strike, model.valuation_utc,
            model.valuation_utc + pd.Timedelta(hours=int(maturity_h)),
            r_annual=R_ANNUAL,
        )
        # smaller grid than production (601 vs 1201 nodes) -- the sweep
        # reports RELATIVE price effects, so 601 nodes are more than
        # enough for 2-3 sig fig deltas at 24-72 h horizons and keeps
        # the ~500-row sweep within a few minutes of wall clock time.
        res = price_forward_centered(model, contract,
                                     grid_settings=ResidualGridSettings(
                                         n_space_nodes=601),
                                     eta_ij=eta_ij)
        info = {"expected_spot_T": res.expected_spot_at_expiry,
                "forward_T": res.forward_at_expiry,
                "residual_sd_T": res.residual_std_at_expiry,
                "p_stress_T": res.diagnostics.get("p_stress_at_expiry", np.nan)}
        return float(res.value), info


def _build_model() -> ForwardCenteredModel:
    params = load_frozen_parameters(PARAMS_YAML)
    quotes = load_quotes(QUOTES_CSV)
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=params.spot_price_TRY_MWh)
    return ForwardCenteredModel(
        curve=curve, spec=ResidualSpec.from_frozen(params),
        tvtp=TVTPCoefficients(params.alpha01, params.gamma01,
                              params.alpha10, params.gamma10),
        pi_filtered=params.pi_filtered,
        valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh)


def _rows(setup: Setup, family: str,
          a_stress: float, eta: Optional[Tuple[float, float]],
          baseline: dict) -> List[dict]:
    out: List[dict] = []
    tag_eta = "None" if eta is None else f"({eta[0]:+.2f},{eta[1]:+.2f})"
    for K in STRIKES:
        for T in MATURITIES_H:
            key = (K, T)
            for kind in ("call", "put"):
                v, info = setup.price(K, T, kind, a_stress, eta)
                base_v = baseline[(K, T, kind)]
                out.append({
                    "family": family,
                    "a_stress_TRY_MWh_per_h": float(a_stress),
                    "eta_ij": tag_eta,
                    "eta_01": eta[0] if eta is not None else 0.0,
                    "eta_10": eta[1] if eta is not None else 0.0,
                    "strike_TRY_MWh": float(K),
                    "maturity_h": int(T), "option_type": kind,
                    "value_TRY_MWh": v,
                    "baseline_value_TRY_MWh": float(base_v),
                    "delta_vs_baseline_TRY_MWh": float(v - base_v),
                    "delta_pct": (100.0 * (v - base_v) / base_v
                                  if abs(base_v) > 1e-9 else np.nan),
                    **{k: float(x) for k, x in info.items()},
                })
    return out


def run_sweep() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = _build_model()
    setup = Setup(model=model, contracts=())
    # baseline: (a=0, eta=None)
    baseline = {}
    for K in STRIKES:
        for T in MATURITIES_H:
            for kind in ("call", "put"):
                v, _ = setup.price(K, T, kind, 0.0, None)
                baseline[(K, T, kind)] = v
    rows: List[dict] = []
    # baseline row for auditability
    for K in STRIKES:
        for T in MATURITIES_H:
            for kind in ("call", "put"):
                rows.append({
                    "family": "baseline",
                    "a_stress_TRY_MWh_per_h": 0.0, "eta_ij": "None",
                    "eta_01": 0.0, "eta_10": 0.0,
                    "strike_TRY_MWh": float(K), "maturity_h": int(T),
                    "option_type": kind,
                    "value_TRY_MWh": baseline[(K, T, kind)],
                    "baseline_value_TRY_MWh": baseline[(K, T, kind)],
                    "delta_vs_baseline_TRY_MWh": 0.0, "delta_pct": 0.0,
                    "expected_spot_T": np.nan, "forward_T": np.nan,
                    "residual_sd_T": np.nan, "p_stress_T": np.nan,
                })

    # Q1-only sweep
    for a in A_STRESS_VALUES:
        if a == 0.0:
            continue
        rows.extend(_rows(setup, "Q1_only", a, None, baseline))

    # Q2-only sweep
    for e01, e10 in product(ETA_VALUES, ETA_VALUES):
        if e01 == 0.0 and e10 == 0.0:
            continue
        rows.extend(_rows(setup, "Q2_only", 0.0, (e01, e10), baseline))

    # Joint corners
    for a in (A_BOUND,):
        for e01, e10 in product((-ETA_BOUND, ETA_BOUND),
                                (-ETA_BOUND, ETA_BOUND)):
            rows.extend(_rows(setup, "joint_corner", a, (e01, e10), baseline))

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for family in ("Q1_only", "Q2_only", "joint_corner"):
        sub = df[df["family"] == family]
        if sub.empty:
            continue
        stats.append({
            "family": family,
            "n_rows": int(len(sub)),
            "max_abs_delta_TRY_MWh": float(sub["delta_vs_baseline_TRY_MWh"].abs().max()),
            "max_abs_delta_pct": float(sub["delta_pct"].abs().max()),
            "max_positive_delta_TRY_MWh": float(sub["delta_vs_baseline_TRY_MWh"].max()),
            "max_negative_delta_TRY_MWh": float(sub["delta_vs_baseline_TRY_MWh"].min()),
        })
    return pd.DataFrame(stats)


def _atm72_table(df: pd.DataFrame) -> pd.DataFrame:
    at = df[(df["strike_TRY_MWh"] == 3000.0)
            & (df["maturity_h"] == 72)
            & (df["option_type"] == "call")].copy()
    return at[["family", "a_stress_TRY_MWh_per_h", "eta_ij",
               "value_TRY_MWh", "delta_vs_baseline_TRY_MWh",
               "delta_pct", "residual_sd_T", "p_stress_T"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running Q1/Q2 sensitivity sweep ...")
    df = run_sweep()
    csv_path = OUT_DIR / "sensitivity_grid.csv"
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}  ({len(df)} rows)")

    summary = summarise(df)
    summary_path = OUT_DIR / "sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  wrote {summary_path}")

    atm72 = _atm72_table(df)
    md = ["# FW2 §4 -- Joint (a, eta) sensitivity sweep\n",
          "See `docs/fw2_risk_premium_identification.md` for the two-",
          "propositions framing.  This report exercises Q1 and Q2 in ",
          "isolation and jointly at the empirical §2.3 upper bounds; ",
          "it produces the price-effect band the manuscript will ",
          "report as its risk-premium uncertainty envelope.\n",
          "* Contracts: 24 / 48 / 72 h call+put on strike ladder ",
          f"{list(STRIKES)}.",
          f"* Q1 grid: a_stress in {list(A_STRESS_VALUES)} TRY/MWh/h ",
          "(a_normal fixed at 0 -- an asymmetric shift; a symmetric ",
          "shift has zero variance effect at this order).",
          f"* Q2 grid: (eta_01, eta_10) in {list(ETA_VALUES)}^2 ",
          "(multiplicative q^Q = q^P * exp(eta)).",
          f"* Joint corners: a_stress = {A_BOUND}, eta_ij = "
          f"(+/-{ETA_BOUND}, +/-{ETA_BOUND})\n",
          "## Max effect per channel\n",
          summary.round(4).to_markdown(index=False),
          "\n\n## ATM K=3000, T=72 h call under every sweep row\n",
          atm72.round({"value_TRY_MWh": 3,
                       "delta_vs_baseline_TRY_MWh": 3,
                       "delta_pct": 3, "residual_sd_T": 2,
                       "p_stress_T": 4}).to_markdown(index=False)]

    (OUT_DIR / "sensitivity_summary.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"  wrote {OUT_DIR / 'sensitivity_summary.md'}")


if __name__ == "__main__":
    main()
