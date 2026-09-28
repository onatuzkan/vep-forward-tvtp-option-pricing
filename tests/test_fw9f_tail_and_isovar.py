"""FW9f section 6 tests -- retargeted F14 v2 sandwich, regime-matched
2022-2025 fit interior convergence, tail-quantile reproducibility."""
from __future__ import annotations

import math
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


@pytest.mark.skipif(not (OUT / "kappa_sensitivity_isovariance_v2.csv").exists(),
                    reason="run scripts/fw9/kappa_sensitivity_isovariance_v2.py first")
def test_f14v2_price_consistent_targets_sandwich_production():
    """The three price-consistent iso-variance targets (production,
    observed_2025_TRY, observed_2025H2_TRY) must sandwich the
    production 72 h ATM call value at every shared kappa, and the
    ancillary asinh_scale target must sit above all three."""
    df = pd.read_csv(OUT / "kappa_sensitivity_isovariance_v2.csv")
    for kappa in sorted(df["kappa_per_hour"].unique()):
        row_prod = df[(df["iso_variance_target"] == "production")
                      & (df["kappa_per_hour"] == kappa)]
        row_obs = df[(df["iso_variance_target"] == "observed_2025_TRY")
                     & (df["kappa_per_hour"] == kappa)]
        row_h2 = df[(df["iso_variance_target"] == "observed_2025H2_TRY")
                    & (df["kappa_per_hour"] == kappa)]
        row_asinh = df[(df["iso_variance_target"] == "asinh_scale_2025")
                       & (df["kappa_per_hour"] == kappa)]
        assert not row_prod.empty and not row_obs.empty
        assert not row_h2.empty and not row_asinh.empty
        p = float(row_prod["call_T72"].iloc[0])
        o = float(row_obs["call_T72"].iloc[0])
        h = float(row_h2["call_T72"].iloc[0])
        a = float(row_asinh["call_T72"].iloc[0])
        # sd_TRY(observed_2025_TRY) = 624 > sd_TRY(production) = 582.5 >
        # sd_TRY(observed_2025H2_TRY) = 556, and ATM scales roughly
        # linearly in sd_TRY, so the observed_2025_TRY curve must sit
        # above production which must sit above observed_2025H2_TRY.
        assert o >= p - 1e-9, (
            f"kappa={kappa}: observed_2025_TRY {o:.3f} below "
            f"production {p:.3f}")
        assert p >= h - 1e-9, (
            f"kappa={kappa}: production {p:.3f} below "
            f"observed_2025H2_TRY {h:.3f}")
        # The asinh-scale target (sd_TRY 1306) is roughly 2.24x the
        # observed sd_TRY so its ATM must sit clearly above the three
        # price-consistent curves at every kappa.
        assert a > o + 50.0, (
            f"kappa={kappa}: asinh_scale ATM {a:.1f} did not clear "
            f"observed_2025_TRY ATM {o:.1f} by >= 50")


@pytest.mark.skipif(not (OUT / "TVTP_1cov_A3_2022_2025.pkl").exists(),
                    reason="run scripts/fw9/consistent_msar_fit_2022_2025.py first")
def test_regime_matched_2022_2025_fit_interior_and_ordered():
    """The FW9f regime-matched 2022-2025 A3 fit must be interior
    (0 < sigma_normal < sigma_stress, phi in (0, 1)), have a finite
    log-likelihood and a stationary distribution that concentrates
    (not blows up) even if the optimiser flagged converged=False."""
    with open(OUT / "TVTP_1cov_A3_2022_2025.pkl", "rb") as f:
        res = pickle.load(f)
    p = res.params
    assert 0.0 < p.sigma_normal < p.sigma_stress, (
        f"regime labels violated: sigma_normal={p.sigma_normal}, "
        f"sigma_stress={p.sigma_stress}")
    assert 0.0 < p.phi < 1.0, f"phi={p.phi} out of (0, 1)"
    assert math.isfinite(res.loglik), "log-likelihood is not finite"
    # Stationary within-regime variances sigma^2 / (1 - phi^2) must be
    # finite and monotone in sigma.
    var_n = p.sigma_normal ** 2 / (1.0 - p.phi ** 2)
    var_s = p.sigma_stress ** 2 / (1.0 - p.phi ** 2)
    assert math.isfinite(var_n) and math.isfinite(var_s)
    assert var_s > var_n, "stress within-regime variance must exceed normal"
    # SEs from the Hessian must be finite for all 9 parameters.
    assert np.all(np.isfinite(res.se_hessian)), "Hessian SEs contain non-finite"


@pytest.mark.skipif(not (OUT / "tail_validation.csv").exists(),
                    reason="run scripts/fw9/tail_validation.py first")
def test_tail_validation_orderings_and_reproducibility():
    """Section 4 tail-validation numeric ordering + reproducibility
    of the KS ranking (production < FW9e_full_A3 < FW9f_regime_matched)
    that the FW9f report cites."""
    df = pd.read_csv(OUT / "tail_validation.csv").set_index("source")
    ks = pd.read_csv(OUT / "tail_validation_ks.csv").set_index("source")
    # All three model sources must be present alongside the observed
    for src in ("observed_2025", "production", "FW9e_full_A3",
                "FW9f_regime_matched"):
        assert src in df.index, f"missing source {src} in tail_validation.csv"
    # Quantile monotonicity within every row
    quants = ["q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]
    for src, row in df.iterrows():
        vals = [row[q] for q in quants]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)), (
            f"non-monotone quantiles for {src}: {vals}")
    # KS ordering that the FW9f report cites: production is the
    # closest to observed, then FW9e_full_A3, then FW9f_regime_matched
    ks_prod = float(ks.loc["production", "KS_statistic"])
    ks_fw9e = float(ks.loc["FW9e_full_A3", "KS_statistic"])
    ks_fw9f = float(ks.loc["FW9f_regime_matched", "KS_statistic"])
    assert ks_prod < ks_fw9e, (
        f"production KS {ks_prod:.4f} was expected to beat FW9e "
        f"KS {ks_fw9e:.4f}")
    assert ks_fw9e < ks_fw9f, (
        f"FW9e KS {ks_fw9e:.4f} was expected to beat FW9f "
        f"KS {ks_fw9f:.4f}")
    # Standard deviations must obey the same ordering
    sd_obs = float(df.loc["observed_2025", "std"])
    sd_prod = float(df.loc["production", "std"])
    sd_fw9e = float(df.loc["FW9e_full_A3", "std"])
    sd_fw9f = float(df.loc["FW9f_regime_matched", "std"])
    assert sd_prod < sd_fw9e < sd_fw9f, (
        f"stationary sd ordering broken: prod={sd_prod:.1f}, "
        f"FW9e={sd_fw9e:.1f}, FW9f={sd_fw9f:.1f}")
    # Production stationary sd must sit within 25% of observed sd
    assert abs(sd_prod - sd_obs) / sd_obs < 0.25, (
        f"production stationary sd {sd_prod:.1f} deviates from "
        f"observed {sd_obs:.1f} by more than 25%")
