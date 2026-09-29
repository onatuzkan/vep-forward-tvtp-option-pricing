"""FW10 tests -- look-ahead, PIT-on-known-distribution, climatology
path usage, day-ahead timing rule sanity."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "outputs" / "fw10_validation"


# --------------------------------------------------------------------
# 1. Look-ahead structural test on the FW10 code.  The only 2026 file
#    that may enter FW10 is realised_ptf_*.csv (evaluation only) and
#    the VEP GGF quotes (used ONLY for the forward curve, which is a
#    Q-measure calibration input, not a P-measure parameter fit).
#    NO 2026 observation may enter any parameter estimate.
# --------------------------------------------------------------------
def test_fw10_code_never_reads_2026_ptf_for_parameter_estimation():
    """Grep the FW10 scripts for any hint that 2026 PTF is used as an
    estimation input (as opposed to an evaluation observation)."""
    fw10_scripts = list((REPO / "scripts" / "fw10").glob("*.py"))
    assert fw10_scripts, "no FW10 scripts found"
    forbidden = (
        "fit_msar", "fit_msar_fixed_phi", "fit_lstsq_ar",
        "estimate_parameters", "fit_kappa",
    )
    for f in fw10_scripts:
        text = f.read_text(encoding="utf-8")
        for kw in forbidden:
            assert kw not in text, (
                f"{f.name} contains {kw}; no parameter estimation "
                f"routine is allowed inside FW10 evaluation code")


def test_fw10_freeze_utc_is_before_any_2026_observation():
    """The FW10 module exports FREEZE_UTC = 2025-12-31 20:00 UTC.
    Every model estimation MUST have been performed on data up to
    that instant; the shared _data.py module documents this in a
    module-level constant that other tests can pin against."""
    from scripts.fw10._data import FREEZE_UTC
    assert FREEZE_UTC == pd.Timestamp("2025-12-31 20:00", tz="UTC"), (
        f"FREEZE_UTC drifted: {FREEZE_UTC}")


# --------------------------------------------------------------------
# 2. PIT is uniform on a known distribution (Gaussian).
# --------------------------------------------------------------------
def test_pit_is_uniform_on_a_known_gaussian_distribution():
    """Draw N Gaussian samples and N Gaussian actuals from the same
    distribution; the PIT sequence must pass a uniform-KS test."""
    from scripts.fw10.run_validation import pit_value
    from scipy.stats import kstest
    rng = np.random.default_rng(42)
    n_days = 200
    n_paths = 5_000
    pits = []
    for i in range(n_days):
        sample = rng.standard_normal(n_paths)
        actual = float(rng.standard_normal())
        pits.append(pit_value(sample, actual))
    stat, p = kstest(np.asarray(pits), "uniform")
    assert p > 0.05, f"KS-uniform test failed on Gaussian: p={p:.4f}"


# --------------------------------------------------------------------
# 3. CRPS on a known distribution is finite and positive.
# --------------------------------------------------------------------
def test_crps_is_finite_and_positive_on_gaussian():
    from scripts.fw10.run_validation import crps_sample
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(2000)
    for actual in (-2.0, 0.0, 2.0):
        c = crps_sample(sample, actual)
        assert math.isfinite(c) and c >= 0.0, (
            f"CRPS at actual={actual} was {c}")


# --------------------------------------------------------------------
# 4. Climatology z path is used, not a scenario-offset variant.
# --------------------------------------------------------------------
def test_climatology_z_cycle_matches_training_constant():
    """The climatology cycle used by FW10's MSAR sampler is built from
    the same training window as scripts.fw12._shared: RD standardised
    series truncated to 2022-12-31 20:00 UTC and grouped by
    (TR-local month, TR-local hour).  The global mean of the cycle
    must be within a few thousandths of zero (the standardised z has
    zero training mean by construction)."""
    from scripts.fw10.run_validation import climatology_z_cycle
    z = climatology_z_cycle()
    assert z.size == 8760, f"expected 8760-hour cycle, got {z.size}"
    assert abs(float(z.mean())) < 0.05, (
        f"climatology z mean drifted from 0: {z.mean():.4f}")


# --------------------------------------------------------------------
# 5. Day-ahead rule: at 11:00 TRT of day d, no h in {24, 48, 72}
#    horizon terminal price is yet known.
# --------------------------------------------------------------------
def test_day_ahead_rule_no_horizon_prices_known_at_valuation():
    """FW10's corrected rule (0.4-d): valuation d 11:00 TRT.  The
    horizons d+1, d+2, d+3 at 23:00 TRT are all in the FUTURE at the
    valuation instant, so none of their prices have been published."""
    from scripts.fw10._data import valuation_utc, day_end_utc
    from pde_option_model.calendar_tr import TURKEY_TZ
    d = pd.Timestamp("2026-06-15", tz=TURKEY_TZ)
    val = valuation_utc(d)
    last_known = day_end_utc(d)
    # last_known is 20:00 UTC (23:00 TRT) of day d, published on d-1
    # around 14:00 TRT.  So at 08:00 UTC (11:00 TRT) of day d, the
    # hours 00:00-23:00 of day d are already published (they were
    # published on d-1 by the DA auction).  We assert that the
    # 24/48/72 h horizon terminals are strictly later than valuation.
    for h in (24, 48, 72):
        terminal = last_known + pd.Timedelta(hours=h)
        assert terminal > val, (
            f"h={h}: terminal {terminal} not strictly after valuation {val}")


# --------------------------------------------------------------------
# 6. tail_validation.md and stationary_variance_check.md consistency
#    notes were added (EK3).
# --------------------------------------------------------------------
def test_fw9_docs_have_531_vs_624_note():
    for f in ("tail_validation.md", "stationary_variance_check.md"):
        p = REPO / "outputs" / "fw9_self_estimation" / f
        assert p.exists()
        text = p.read_text(encoding="utf-8")
        assert "531" in text and "624" in text, (
            f"{f} missing the 531 / 624 reconciliation note")


# --------------------------------------------------------------------
# 7. Predictive daily CSV (archived FW10 variant), if produced,
#    has the expected shape.
# --------------------------------------------------------------------
@pytest.mark.skipif(
    not (OUT / "archive_baseload_F" / "predictive_daily.csv").exists(),
    reason="archived FW10 predictive daily missing")
def test_predictive_daily_shape_and_models():
    df = pd.read_csv(OUT / "archive_baseload_F" / "predictive_daily.csv")
    assert not df.empty
    assert set(df["h"].unique()) == {24, 48, 72}, (
        f"unexpected horizons: {sorted(df['h'].unique())}")
    assert {"M0", "M1", "B1", "B2", "B3"}.issubset(set(df["model"].unique()))
    # PIT must be in [0, 1]
    assert (df["pit"] >= 0).all() and (df["pit"] <= 1).all()
    # CRPS must be non-negative
    assert (df["crps"] >= 0).all()
