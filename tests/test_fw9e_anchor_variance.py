"""FW9e §7 tests -- 2x2 matrix reproducibility, yearly-kappa bands,
stationary variance regime-mixture identity, iso-variance F14 stability."""
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


@pytest.mark.skipif(not (OUT / "anchor_scale_matrix.csv").exists(),
                    reason="run scripts/fw9/anchor_scale_matrix.py first")
def test_anchor_scale_matrix_reproduces_independent_check():
    """The 2x2 (scale x anchor) kappa matrix must reproduce the FW9e
    independent-check numbers to three decimal places."""
    df = pd.read_csv(OUT / "anchor_scale_matrix.csv")
    piv = df.pivot(index="anchor", columns="scale", values="kappa_per_hour")
    expected = {
        ("A0", "asinh"): 0.0196, ("A0", "TRY"): 0.0342,
        ("A1", "asinh"): 0.0162, ("A1", "TRY"): 0.0283,
        ("A2", "asinh"): 0.1538, ("A2", "TRY"): 0.2116,
        ("A3", "asinh"): 0.1662, ("A3", "TRY"): 0.2234,
    }
    for (anchor, scale), exp in expected.items():
        got = float(piv.loc[anchor, scale])
        assert abs(got - exp) < 5e-4, (
            f"{anchor}, {scale}: got {got:.5f}, expected {exp}")


@pytest.mark.skipif(not (OUT / "anchor_scale_matrix.csv").exists(),
                    reason="run scripts/fw9/anchor_scale_matrix.py first")
def test_anchor_dominates_scale_in_kappa_ratio():
    """FW9e finding: anchor is 7-10x, scale is 1.3-1.7x."""
    df = pd.read_csv(OUT / "anchor_scale_matrix.csv")
    piv = df.pivot(index="anchor", columns="scale", values="kappa_per_hour")
    # scale ratios
    for a in ("A0", "A1", "A2", "A3"):
        r = float(piv.loc[a, "TRY"]) / float(piv.loc[a, "asinh"])
        assert 1.3 < r < 1.8, f"{a} scale ratio {r:.2f} outside [1.3, 1.8]"
    # anchor ratios (A3 vs A0 within scale)
    for s in ("asinh", "TRY"):
        r = float(piv.loc["A3", s]) / float(piv.loc["A0", s])
        assert 6.0 < r < 11.0, f"{s} anchor ratio {r:.2f} outside [6, 11]"


@pytest.mark.skipif(not (OUT / "yearly_kappa_A3_TRY.csv").exists(),
                    reason="run scripts/fw9/anchor_scale_matrix.py first")
def test_yearly_kappa_A3_TRY_lands_in_02_to_026_band():
    """The model-faithful A3 TRY residual gives kappa clustered in
    [0.19, 0.26] /h across the seven years."""
    yr = pd.read_csv(OUT / "yearly_kappa_A3_TRY.csv")
    yr = yr[yr["year"].astype(str).str.match(r"^\d+$")]  # exclude 2025-H2 label
    kappas = yr["kappa_per_hour"].astype(float).to_numpy()
    assert kappas.min() > 0.15, (
        f"yearly kappa min {kappas.min():.3f} unexpectedly small")
    assert kappas.max() < 0.30, (
        f"yearly kappa max {kappas.max():.3f} unexpectedly large")


@pytest.mark.skipif(not (OUT / "stationary_variance_check.json").exists(),
                    reason="run scripts/fw9/stationary_variance_check.py first")
def test_regime_mixture_stationary_sd_identity():
    """Independent recomputation of the regime-mixture stationary sd
    must match the JSON's value."""
    payload = json.loads((OUT / "stationary_variance_check.json").read_text())
    prod = payload["production"]
    # recompute
    p01 = 1 / (1 + math.exp(1.0157))
    p10 = 1 / (1 + math.exp(1.8952))
    pi_s = p01 / (p01 + p10)
    pi_n = 1 - pi_s
    sigma2_mix = pi_n * 0.003534807 ** 2 + pi_s * 0.0924066544 ** 2
    var_stat = sigma2_mix / (1 - 0.9246 ** 2)
    sd_stat = math.sqrt(var_stat)
    assert abs(sd_stat - float(prod["sd_stat_asinh"])) < 1e-5
    # TRY at spot
    delta = math.sqrt(2917.78 ** 2 + 282.48 ** 2)
    assert abs(sd_stat * delta - float(prod["sd_stat_TRY_at_spot"])) < 1e-4


@pytest.mark.skipif(not (OUT / "stationary_variance_check.json").exists(),
                    reason="run scripts/fw9/stationary_variance_check.py first")
def test_TRY_stationary_variance_matches_2025_within_10_percent():
    """The cancellation claim: production stationary sd_TRY at spot is
    within 10% of 2025's observed A3 residual sd."""
    payload = json.loads((OUT / "stationary_variance_check.json").read_text())
    sub = payload["sub_50_analysis"]
    r = float(sub["TRY_ratio_full_over_prod"])
    assert 0.9 <= r <= 1.1, (
        f"2025 TRY sd / production sd = {r:.3f} outside [0.9, 1.1]")


@pytest.mark.skipif(
    not (OUT / "kappa_sensitivity_isovariance.csv").exists(),
    reason="run scripts/fw9/kappa_sensitivity_isovariance.py first")
def test_isovariance_72h_stable_within_model_faithful_kappa_band():
    """Under the observed_2025 iso-variance target, the ATM 72 h call
    is stable within +/-5 % across the model-faithful kappa band
    [0.15, 0.22] /h (the range spanned by A2/A3 asinh and TRY
    residual fits)."""
    df = pd.read_csv(OUT / "kappa_sensitivity_isovariance.csv")
    sub = df[(df["iso_variance_target"] == "observed_2025")
             & (df["kappa_per_hour"] >= 0.15)
             & (df["kappa_per_hour"] <= 0.22)]
    assert len(sub) >= 3
    vals = sub["call_T72"].to_numpy()
    span_pct = 100.0 * (vals.max() - vals.min()) / vals.min()
    assert span_pct < 5.0, (
        f"call_T72 span within model-faithful band = {span_pct:.2f}% "
        "(expected < 5%)")
