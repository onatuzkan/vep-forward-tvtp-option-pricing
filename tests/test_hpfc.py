"""Tests for the recency-weighted HPFC -- REAL EPİAŞ PTF data only.

Requires inputs/historical/ptf_raw/ptf_2019..2025.csv (downloaded with
scripts/data/download_epias.py); data-dependent tests are skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pde_option_model.hpfc import (apply_shape_to_curve, day_type, fit_shape,
                                   recency_weights, select_half_life, turkish_holidays)
from pde_option_model.premium import load_epias_ptf_dir

TR = "Europe/Istanbul"


@pytest.fixture(scope="module")
def ptf(repo_root):
    d = repo_root / "inputs/historical/ptf_raw"
    if not d.exists() or not list(d.glob("ptf_*.csv")):
        pytest.skip("ptf_raw not downloaded (scripts/data/download_epias.py)")
    return load_epias_ptf_dir(d)


def _cut(date):
    return pd.Timestamp(date, tz=TR).tz_convert("UTC")


def _ratio(s):
    loc = s.index.tz_convert(TR).tz_localize(None)
    return (s / s.groupby(loc.to_period("M")).transform("mean")).to_numpy()


# --------------------------------------------------------------------------
# calendar / weights (no data needed)
# --------------------------------------------------------------------------
def test_turkish_holidays_2026():
    h = turkish_holidays([2026])
    assert len(h) == 14                               # 7 fixed + 3 Ramazan + 4 Kurban
    assert pd.Timestamp("2026-03-20") in h and pd.Timestamp("2026-05-30") in h


def test_day_type_classification():
    idx = pd.DatetimeIndex(["2026-01-01 10:00", "2026-01-03 10:00",
                            "2026-01-04 10:00", "2026-01-05 10:00"])
    np.testing.assert_array_equal(day_type(idx), [2, 1, 2, 0])


def test_recency_weights_halve_every_half_life():
    cut = pd.Timestamp("2026-01-01")
    idx = pd.DatetimeIndex([cut - pd.Timedelta(days=365.25 * k) for k in (0, 0.5, 1.0)])
    np.testing.assert_allclose(recency_weights(idx, cut, 0.5), [1.0, 0.5, 0.25], rtol=1e-12)
    np.testing.assert_allclose(recency_weights(idx, cut, None), [1.0, 1.0, 1.0])


# --------------------------------------------------------------------------
# shape model on EPİAŞ PTF 2019-2025
# --------------------------------------------------------------------------
def test_shape_is_normalised_within_each_month(ptf):
    m = fit_shape(ptf, _cut("2025-01-01"), 0.5, 2)
    idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h", tz=TR).tz_convert("UTC")
    s = pd.Series(m.shape(idx), index=idx.tz_convert(TR).tz_localize(None))
    np.testing.assert_allclose(s.groupby(s.index.to_period("M")).mean().to_numpy(), 1.0,
                               atol=1e-12)


def test_look_ahead_guard(ptf):
    cut = _cut("2025-01-01")
    a = fit_shape(ptf, cut, 0.5, 2)
    corrupted = ptf.copy()
    corrupted[corrupted.index >= cut] *= 10.0
    b = fit_shape(corrupted, cut, 0.5, 2)
    np.testing.assert_array_equal(a.coef, b.coef)


def test_recency_weighting_beats_equal_weights_out_of_sample_2025(ptf):
    cut = _cut("2025-01-01")
    test = ptf[(ptf.index >= cut) & (ptf.index < _cut("2026-01-01"))]
    r = _ratio(test)
    err = {H: np.sqrt(np.mean((r - fit_shape(ptf, cut, H, 2).shape(test.index)) ** 2))
           for H in (0.5, None)}
    assert err[0.5] < 0.95 * err[None]                # measured: 0.199 vs 0.228


def test_select_half_life_on_real_data(ptf):
    res, best = select_half_life(ptf, test_years=(2025,), half_lives=(0.5, None),
                                 harmonics=(2,))
    assert best["half_life_years"] == 0.5
    assert best["mean_ratio_rmse"] < best["equal_weight_ratio_rmse"]


def test_real_profile_has_solar_dip_in_summer(ptf):
    m = fit_shape(ptf, _cut("2026-01-01"), 0.5, 2)
    prof = m.profile_table(2026)
    jul = prof[(prof.month == 7) & (prof.day_type == 0)].set_index("hour")["shape"]
    assert jul.loc[12] < 0.9 < 1.1 < jul.loc[20]      # midday trough, evening peak


# --------------------------------------------------------------------------
# HPFC on the accepted VEP curve
# --------------------------------------------------------------------------
def test_hpfc_preserves_every_monthly_average(ptf, repo_root):
    curve = pd.read_csv(repo_root / "outputs/market_calibration_final/hourly_forward_curve.csv")
    m = fit_shape(ptf, pd.Timestamp("2025-12-31 21:00", tz="UTC"), 0.5, 2)
    out = apply_shape_to_curve(curve, m)
    ym = pd.to_datetime(out["time_utc"], utc=True).dt.tz_convert(TR).dt.strftime("%Y-%m")
    g = out.groupby(ym)[["hourly_forward_TRY_MWh", "hpfc_TRY_MWh"]].mean()
    assert (g.iloc[:, 0] - g.iloc[:, 1]).abs().max() < 1e-8
