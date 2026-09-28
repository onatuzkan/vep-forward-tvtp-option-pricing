"""FW9c §7 -- tests for the kappa bracket, pi horizon, deflated fit,
and the relabelled decomposition."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"


@pytest.mark.skipif(not (OUT / "kappa_bracket.csv").exists(),
                    reason="run scripts/fw9/kappa_bracket.py first")
def test_kappa_bracket_dof_ladder_is_monotone_and_expected():
    """S0..S5 must respect the design: S0 = 0 seasonal dof;
    S1 = 23 (hour dummies with intercept-absorbed reference);
    S2 = S1 + 6 (dow); S3 = S2 + 11 (moy); S4 = S3 + 4 (2 harmonics * 2);
    S5 = 168 * 6/7 - 1 = 167 + 11 + 4 hour x dow flavour."""
    df = pd.read_csv(OUT / "kappa_bracket.csv")
    dof = df.set_index("spec")["seasonal_dof"].to_dict()
    assert int(dof["S0"]) == 0
    assert int(dof["S1"]) == 23
    assert int(dof["S2"]) == 29             # 23 + 6
    assert int(dof["S3"]) == 40             # 29 + 11
    assert int(dof["S4"]) == 44             # 40 + 4
    assert int(dof["S5"]) > int(dof["S4"])  # 168 + moy + fourier flavour


@pytest.mark.skipif(not (OUT / "kappa_bracket.csv").exists(),
                    reason="run scripts/fw9/kappa_bracket.py first")
def test_kappa_bracket_reproduces_finite_values():
    df = pd.read_csv(OUT / "kappa_bracket.csv")
    # Every ladder row must produce a positive finite kappa
    for _, r in df.iterrows():
        if r["spec"] == "YAML":
            continue
        assert 0 < float(r["ar1_phi"]) < 1
        assert float(r["ar1_kappa_per_hour"]) > 0
        assert np.isfinite(float(r["atm_call_T72"]))
        assert float(r["atm_call_T72"]) > 0


@pytest.mark.skipif(not (OUT / "pi_filtered_horizon.csv").exists(),
                    reason="run scripts/fw9/pi_filtered_horizon.py first")
def test_pi_horizon_delta_shrinks_toward_zero():
    """Regime memory half-life ~1.4 h implies the initial-distribution
    effect must shrink with horizon; by T = 24 h the absolute price
    delta between yaml pi and FW9 terminal pi is <= 0.1 TRY/MWh, and
    by T = 72 h it is <= 0.001 TRY/MWh (floating-point floor)."""
    df = pd.read_csv(OUT / "pi_filtered_horizon.csv")
    d = df.set_index("T_hours")["delta_TRY"].to_dict()
    assert abs(d[24]) < 0.1
    assert abs(d[48]) < 0.01
    assert abs(d[72]) < 0.001


@pytest.mark.skipif(not (OUT / "price_impact_v2_decomposition.csv").exists(),
                    reason="run scripts/fw9/rewrite_comparison_and_reprice.py first")
def test_decomposition_share_column_sums_to_one_hundred_for_E():
    """The E (full FW9) row's share-of-total-delta % must be 100 % by
    construction; the three isolated variants sum to less than 100 %
    because of the interaction term."""
    df = pd.read_csv(OUT / "price_impact_v2_decomposition.csv")
    for _, row in df.iterrows():
        e_share = row["delta_E_fw9_consistent_vs_A_share_of_total_delta_pct"]
        assert abs(e_share - 100.0) < 1e-6, (
            f"K={row['strike']}, T={row['maturity_h']}: E share "
            f"{e_share} != 100")


@pytest.mark.skipif(not (OUT / "price_impact_v2_decomposition.csv").exists(),
                    reason="run scripts/fw9/rewrite_comparison_and_reprice.py first")
def test_decomposition_has_both_percent_columns():
    df = pd.read_csv(OUT / "price_impact_v2_decomposition.csv")
    for variant in ("B_inh_kappa_inh_trans_fw9_sigma",
                    "C_inh_sigma_inh_trans_fw9_kappa",
                    "D_inh_sigma_inh_kappa_fw9_trans",
                    "E_fw9_consistent"):
        for suffix in ("_price_change_pct", "_share_of_total_delta_pct"):
            col = f"delta_{variant}_vs_A{suffix}"
            assert col in df.columns, f"missing column {col}"


def test_pi_horizon_script_uses_climatology_path():
    """The pi_filtered_horizon script must construct the climatology
    z path via scripts/fw12/_shared.py (FW12b rule)."""
    src = (REPO_ROOT / "scripts" / "fw9" / "pi_filtered_horizon.py").read_text()
    assert "climatology_z_lagged_fn" in src
    assert "allow_constant_transition_scenario" not in src
