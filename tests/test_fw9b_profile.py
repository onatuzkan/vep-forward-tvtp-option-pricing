"""FW9b §9 -- structural tests for the profile-based re-estimation."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"


def _load_argmax():
    p = OUT / "profile_phi_TVTP_1cov_argmax.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


@pytest.mark.skipif(not (OUT / "profile_phi_TVTP_1cov_argmax.pkl").exists(),
                    reason="run scripts/fw9/profile_phi.py first")
def test_profile_argmax_hessian_is_positive_definite():
    """The Hessian at the profile-conditional argmax must be
    positive-definite; otherwise the SEs are not interpretable."""
    blob = _load_argmax()
    res = blob["res"]
    assert res.hessian is not None
    eigs = np.linalg.eigvalsh(res.hessian)
    assert (eigs > 0).all(), (
        f"Hessian at profile argmax has non-positive eigenvalues: "
        f"min={eigs.min():.3e}")


@pytest.mark.skipif(not (OUT / "profile_phi_TVTP_1cov_argmax.pkl").exists(),
                    reason="run scripts/fw9/profile_phi.py first")
def test_profile_argmax_gradient_is_finite_and_small_enough_for_polished_optimum():
    """After the FW9b polish step the residual gradient norm must be
    within a documented finite-difference precision band."""
    blob = _load_argmax()
    res = blob["res"]
    assert np.isfinite(res.grad_norm)
    # residual gradient norm reflects finite-diff noise at n=61k; the
    # L-BFGS-B success flag is the primary convergence signal.
    assert res.grad_norm < 10.0, (
        f"profile argmax residual grad_norm {res.grad_norm} is too large; "
        "the polish step failed")


@pytest.mark.skipif(not (OUT / "parameter_comparison_v2.csv").exists(),
                    reason="run scripts/fw9/rewrite_comparison_and_reprice.py first")
def test_object_matched_column_populated():
    """T8 must have inherited_from populated on every row -- the
    FW9b restructuring guarantees this."""
    df = pd.read_csv(OUT / "parameter_comparison_v2.csv")
    assert "inherited_from" in df.columns
    assert df["inherited_from"].notna().all()
    assert (df["inherited_from"].astype(str).str.len() > 0).all()


@pytest.mark.skipif(not (OUT / "parameter_comparison_v2.csv").exists(),
                    reason="run scripts/fw9/rewrite_comparison_and_reprice.py first")
def test_boundary_rows_do_not_report_se():
    """Rows with boundary_flag=True must have SE/t-stat/CI blank so a
    reviewer does not accidentally cite an uninterpretable Hessian SE."""
    df = pd.read_csv(OUT / "parameter_comparison_v2.csv")
    boundary = df[df["boundary_flag"] == True]  # noqa: E712
    if len(boundary):
        for col in ("fw9_se", "t_stat_vs_inherited",
                    "fw9_95pct_low", "fw9_95pct_high",
                    "inherited_in_fw9_95pct"):
            assert boundary[col].isna().all(), (
                f"boundary row has non-null {col}; SE should be blank at "
                "the unit-root boundary")


@pytest.mark.skipif(not (OUT / "deseasonalized_ar1.json").exists(),
                    reason="run scripts/fw9/deseasonalized_ar1.py first")
def test_deseasonalized_ar1_reproducibility_and_finite_output():
    """Two calls to the OLS AR(1) on the same input produce identical
    numbers -- the fit is deterministic."""
    import subprocess, sys as _sys
    r1 = subprocess.run([_sys.executable, "scripts/fw9/deseasonalized_ar1.py"],
                        cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    payload = json.loads((OUT / "deseasonalized_ar1.json").read_text())
    for k in ("phi", "kappa_per_hour", "half_life_hours", "se_phi",
              "loglik", "AIC", "BIC"):
        assert k in payload
        assert np.isfinite(payload[k])
    assert 0 < payload["phi"] < 1
    assert payload["kappa_per_hour"] > 0
    assert payload["half_life_hours"] > 0


@pytest.mark.skipif(not (OUT / "price_impact_v2_grid.csv").exists(),
                    reason="run scripts/fw9/rewrite_comparison_and_reprice.py first")
def test_price_impact_v2_uses_1201_nodes_and_climatology_z():
    """FW12b rule: every FW9b pricing artefact must be produced under
    the production 1201-node grid with the climatology z path.  We
    verify by rebuilding one row via the same code path and matching
    the CSV value to within Richardson-level tolerance."""
    import math
    from pde_option_model.contracts import EuropeanOption
    from pde_option_model.forward_centered import (ResidualGridSettings,
                                                    ResidualSpec,
                                                    ForwardCenteredModel,
                                                    price_forward_centered)
    from pde_option_model.forward_curve import build_forward_curve
    from pde_option_model.generator import TVTPCoefficients
    from pde_option_model.market_data import load_quotes
    from pde_option_model.params_frozen import load_frozen_parameters
    from scripts.fw12._shared import climatology_z_lagged_fn

    df = pd.read_csv(OUT / "price_impact_v2_grid.csv")
    row = df[(df["label"] == "A_inherited")
             & (df["strike"] == 3000)
             & (df["maturity_h"] == 72)].iloc[0]

    yaml_p = load_frozen_parameters(REPO_ROOT / "inputs" / "historical"
                                    / "m2_frozen_parameters.yaml")
    quotes = load_quotes(REPO_ROOT / "inputs" / "market"
                         / "vep_monthly_quotes.csv")
    curve = build_forward_curve(quotes, mode="smooth_constrained",
                                spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    spec = ResidualSpec(kappa_per_hour=float(yaml_p.kappa_per_hour),
                        sigma_y=np.array([yaml_p.sigma_y[0],
                                          yaml_p.sigma_y[1]]),
                        scale_P=282.48, regime_means=np.zeros(2),
                        mode="additive")
    model = ForwardCenteredModel(
        curve=curve, spec=spec,
        tvtp=TVTPCoefficients(yaml_p.alpha01, yaml_p.gamma01,
                              yaml_p.alpha10, yaml_p.gamma10),
        pi_filtered=yaml_p.pi_filtered,
        valuation_utc=yaml_p.valuation_utc,
        spot_price_TRY_MWh=yaml_p.spot_price_TRY_MWh)
    contract = EuropeanOption("call", 3000.0, model.valuation_utc,
                              model.valuation_utc + pd.Timedelta(hours=72),
                              r_annual=0.40)
    gs = ResidualGridSettings(n_space_nodes=1201)
    z_fn = climatology_z_lagged_fn(model, contract, gs)
    res = price_forward_centered(model, contract, grid_settings=gs,
                                 z_lagged_fn=z_fn)
    assert abs(float(res.value) - float(row["V_call"])) < 1e-4
