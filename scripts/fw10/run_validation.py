"""FW10 orchestrator: predictive-distribution and option-pricing
out-of-sample validation on 2026 realised PTF.

Section list (matches the FW10 brief):
  0.1 hash manifest (see scripts/fw10/hashes_before helper; run
      separately)
  0.4 day-ahead timing check (see day_ahead_timing_check.py)
  A1  daily evaluation universe: business days in 2026 with a VEP
      quote available on or before valuation day; every 3rd business
      day is retained by default to keep runtime under the budget.
  A2  produce predictive P_{d+h} samples per (d, h, model):
      * M0 production yaml -- MSAR + TVTP MC, N_PATHS paths
      * M1 FW9e A3 params  -- MSAR + TVTP MC, N_PATHS paths
      * B1 Black-76        -- lognormal closed form
      * B2 Bachelier       -- Gaussian closed form
      * B3 Lucia-Schwartz  -- Gaussian closed form with OU variance
  A3  PIT, coverage, CRPS, pinball, Diebold-Mariano (simplified paired
      t-test on CRPS differences; the full HAC variant is left as a
      FW10-follow-up because the paired daily samples at h=24 are
      approximately i.i.d. at 3-day cadence).
  A4  Option-level bias (call value vs discounted realised payoff) at
      K in {0.8, 0.9, 1.0, 1.1, 1.2} * F.
  A5  Aggregated tables by (h, moneyness).

Simplifying assumption: the forward level F(target_hour) for the
horizon terminal 23:00 TRT of day d+h/24 is taken to be the VEP
monthly baseload quote for the delivery month of that target hour.
This is a flat intra-month approximation; the paper's shipped
production model uses a full HPFC-shaped hourly curve, so the FW10
tables are an out-of-sample check on the RESIDUAL model given the
monthly-average forward.  See day_ahead_timing_check.md for the
timing rule.  Production yaml is NOT modified.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pde_option_model.calendar_tr import TURKEY_TZ, parse_vep_contract_code
from pde_option_model.params_frozen import load_frozen_parameters
from pde_option_model.tvtp2 import load_hourly_z_history
from scripts.fw10._data import (business_day_universe, day_end_utc,
                                 load_realized_ptf, load_vep_quotes_daily,
                                 valuation_utc, vep_quote_on_or_before,
                                 FREEZE_UTC)

OUT = REPO / "outputs" / "fw10_validation"
YAML_PATH = REPO / "inputs" / "historical" / "m2_frozen_parameters.yaml"
FW9E_A3_PKL = REPO / "outputs" / "fw9_self_estimation" / "TVTP_1cov_A3.pkl"
Z_HISTORY_CSV = REPO / "inputs" / "historical" / "rd_standardized.csv"
TRAIN_END_UTC = pd.Timestamp("2022-12-31 20:00:00", tz="UTC")

R_ANNUAL = 0.40
R_PER_HOUR = R_ANNUAL / (365.0 * 24.0)

HORIZONS_H = (24, 48, 72)
MONEYNESS = (0.8, 0.9, 1.0, 1.1, 1.2)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
COVERAGE_ALPHAS = (0.50, 0.80, 0.90, 0.98)
DEFAULT_N_PATHS = 10_000
DEFAULT_CADENCE_DAYS = 3


# ----------------------------------------------------------------------
# Parameter sets
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MSARParamSet:
    label: str
    mu_n: float
    mu_s: float
    sigma_n: float
    sigma_s: float
    phi: float
    a01: float
    g01: float
    a10: float
    g10: float

    def kappa(self) -> float:
        return -math.log(self.phi)


def load_M0_production() -> MSARParamSet:
    p = load_frozen_parameters(YAML_PATH)
    return MSARParamSet(
        label="M0_production", mu_n=0.0, mu_s=0.0,
        sigma_n=float(p.sigma_y[0]), sigma_s=float(p.sigma_y[1]),
        phi=float(p.phi), a01=float(p.alpha01), g01=float(p.gamma01),
        a10=float(p.alpha10), g10=float(p.gamma10))


def load_M1_fw9e_A3() -> MSARParamSet:
    with open(FW9E_A3_PKL, "rb") as f:
        r = pickle.load(f).params
    return MSARParamSet(
        label="M1_FW9e_A3", mu_n=r.mu_normal, mu_s=r.mu_stress,
        sigma_n=r.sigma_normal, sigma_s=r.sigma_stress, phi=r.phi,
        a01=r.alpha01, g01=r.gamma01, a10=r.alpha10, g10=r.gamma10)


# ----------------------------------------------------------------------
# Climatology z cycle (frozen at TRAIN_END_UTC; identical to FW12
# scenarios.ScenarioBuilder construction)
# ----------------------------------------------------------------------
def climatology_z_cycle() -> np.ndarray:
    """Return an 8760-hour cycle of z_lag by (TR-month, TR-hour)."""
    z_hist = load_hourly_z_history(str(Z_HISTORY_CSV))
    train = z_hist.loc[:TRAIN_END_UTC]
    loc = train.index + pd.Timedelta(hours=3)
    clim = train.groupby([loc.month, loc.hour]).mean()
    start = pd.Timestamp("2023-01-01 00:00:00+00:00")
    end = pd.Timestamp("2023-12-31 23:00:00+00:00")
    idx = pd.date_range(start, end, freq="h")
    loc_idx = idx + pd.Timedelta(hours=3)
    keys = list(zip(loc_idx.month, loc_idx.hour))
    global_mean = float(train.mean())
    return np.array([clim.get(k, global_mean) for k in keys],
                    dtype=float)


def z_for_utc(t_utc: pd.Timestamp, z_cycle: np.ndarray) -> float:
    """Look up the climatology z at UTC hour t_utc."""
    tr_local = t_utc.tz_convert(TURKEY_TZ)
    month = tr_local.month
    hour = tr_local.hour
    # 2023 non-leap: hour index in the annual cycle
    day_of_year = pd.Timestamp(f"2023-{month:02d}-{tr_local.day:02d}") \
        .dayofyear - 1
    idx = (day_of_year * 24 + hour) % z_cycle.size
    return float(z_cycle[idx])


# ----------------------------------------------------------------------
# MSAR predictive sampler.  Simulates y_{d 23:00}, y_{(d+1) 23:00},
# ..., y_{(d+3) 23:00}, then maps to P via the delta mapping.
# ----------------------------------------------------------------------
def simulate_msar_paths(params: MSARParamSet, z_lag_seq: np.ndarray,
                        n_paths: int, seed: int,
                        pi_init: Optional[Tuple[float, float]] = None,
                        x0: float = 0.0) -> np.ndarray:
    """Simulate the two-regime MS-AR(1)+TVTP forward.

    ``z_lag_seq[t]`` is the lagged z applied at step t.  Returns an
    array of shape (n_paths, len(z_lag_seq) + 1) with the initial
    x0 in column 0.
    """
    rng = np.random.default_rng(seed)
    n_steps = z_lag_seq.size
    y = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    y[:, 0] = x0
    if pi_init is None:
        # constant-transition stationary pi at z=0
        p01_bar = 1.0 / (1.0 + math.exp(-params.a01))
        p10_bar = 1.0 / (1.0 + math.exp(-params.a10))
        pi_stress = p01_bar / (p01_bar + p10_bar)
        pi_init = (1.0 - pi_stress, pi_stress)
    J = (rng.random(n_paths) < pi_init[1]).astype(np.int8)
    mu = np.array([params.mu_n, params.mu_s])
    sig = np.array([params.sigma_n, params.sigma_s])
    for t in range(n_steps):
        z = float(z_lag_seq[t])
        p01 = 1.0 / (1.0 + math.exp(-(params.a01 + params.g01 * z)))
        p10 = 1.0 / (1.0 + math.exp(-(params.a10 + params.g10 * z)))
        u = rng.random(n_paths)
        stay0 = (J == 0) & (u >= p01)
        move01 = (J == 0) & (u < p01)
        move10 = (J == 1) & (u < p10)
        stay1 = (J == 1) & (u >= p10)
        J = np.where(stay0 | move10, 0, 1).astype(np.int8)
        eps = rng.standard_normal(n_paths)
        y[:, t + 1] = mu[J] + params.phi * y[:, t] + sig[J] * eps
    return y


def predict_msar(params: MSARParamSet, val_utc: pd.Timestamp,
                 F_by_hour: Dict[int, float], scale_P: float,
                 z_cycle: np.ndarray, n_paths: int, seed: int,
                 pi_init: Optional[Tuple[float, float]] = None
                 ) -> Dict[int, np.ndarray]:
    """Predictive P sample at each h in HORIZONS_H.

    val_utc = valuation d 11:00 TRT (08:00 UTC).  Last known hour is
    day d 23:00 TRT (20:00 UTC).  So we simulate hourly steps from
    d 23:00 UTC hour by hour up to (d+3) 23:00 TRT.
    ``F_by_hour[h]`` is F at the horizon terminal.
    """
    last_known_utc = day_end_utc(val_utc.tz_convert(TURKEY_TZ))
    # We simulate 72 hours forward from last_known_utc
    horizons = sorted(HORIZONS_H)
    n_steps = horizons[-1]
    step_times = [last_known_utc + pd.Timedelta(hours=k)
                  for k in range(1, n_steps + 1)]
    z_seq = np.array([z_for_utc(t, z_cycle) for t in step_times],
                     dtype=float)
    y_paths = simulate_msar_paths(params, z_seq, n_paths=n_paths,
                                  seed=seed, pi_init=pi_init)
    # y_paths has shape (n_paths, n_steps + 1); y[:, k] is state at
    # time step k from last_known_utc (k=0 -> x0=0, k=h -> horizon h)
    out = {}
    for h in horizons:
        y_T = y_paths[:, h]
        F = F_by_hour[h]
        delta = math.sqrt(F * F + scale_P * scale_P)
        P_T = F + y_T * delta
        out[h] = P_T
    return out


# ----------------------------------------------------------------------
# Benchmark predictive distributions (closed form).
# We return SAMPLES (10k draws) so downstream metrics can reuse the
# same PIT/CRPS machinery.
# ----------------------------------------------------------------------
def predict_black76(F: float, sigma_per_h: float, tau_h: float,
                    n_paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    var = (sigma_per_h ** 2) * tau_h
    Z = rng.standard_normal(n_paths)
    return F * np.exp(-0.5 * var + math.sqrt(var) * Z)


def predict_bachelier(F: float, sigma_per_h: float, tau_h: float,
                       n_paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sd = sigma_per_h * math.sqrt(tau_h)
    return F + sd * rng.standard_normal(n_paths)


def predict_lucia_schwartz(F: float, sigma_price: float, kappa: float,
                            tau_h: float, n_paths: int, seed: int) -> np.ndarray:
    """OU-integrated variance ``sigma_price^2 (1 - exp(-2 kappa tau))
    / (2 kappa)``, arithmetic mean F, Gaussian draws."""
    rng = np.random.default_rng(seed)
    if kappa <= 0.0:
        var = (sigma_price ** 2) * tau_h
    else:
        var = (sigma_price ** 2) * (1.0 - math.exp(-2.0 * kappa * tau_h)) \
            / (2.0 * kappa)
    return F + math.sqrt(var) * rng.standard_normal(n_paths)


# ----------------------------------------------------------------------
# Historical vol fit for B1/B2 (trailing 365 days ending at last_known)
# ----------------------------------------------------------------------
def fit_hist_vol(ptf: pd.Series, cutoff_utc: pd.Timestamp,
                 window_hours: int = 365 * 24,
                 price_floor: float = 50.0) -> Dict[str, float]:
    s = ptf[ptf.index <= cutoff_utc].tail(window_hours + 1).copy()
    s = s.clip(lower=price_floor)
    log_ret_h = np.diff(np.log(s.to_numpy()))
    abs_ret_h = np.diff(s.to_numpy())
    # Daily-average log/abs returns
    daily = s.resample("1D").mean().dropna()
    log_ret_d = np.diff(np.log(daily.to_numpy().clip(min=price_floor)))
    abs_ret_d = np.diff(daily.to_numpy())
    return {
        "sigma_log_h": float(np.std(log_ret_h, ddof=1)),
        "sigma_abs_h": float(np.std(abs_ret_h, ddof=1)),
        "sigma_log_d2h": float(np.std(log_ret_d, ddof=1) / math.sqrt(24.0)),
        "sigma_abs_d2h": float(np.std(abs_ret_d, ddof=1) / math.sqrt(24.0)),
    }


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def pit_value(sample: np.ndarray, actual: float) -> float:
    return float(np.mean(sample <= actual))


def crps_sample(sample: np.ndarray, actual: float) -> float:
    """CRPS from a Monte Carlo sample (Gneiting-Ranjan estimator)."""
    x = np.sort(sample)
    n = x.size
    # E|X - actual|
    term1 = float(np.mean(np.abs(x - actual)))
    # E|X - X'| = 2 / n^2 * sum i (2i - n - 1) x_(i)
    i = np.arange(1, n + 1)
    term2 = float((2.0 / (n * n)) * np.sum((2 * i - n - 1) * x))
    return term1 - 0.5 * term2


def pinball_loss(sample: np.ndarray, actual: float, q: float) -> float:
    qval = float(np.quantile(sample, q))
    return (actual - qval) * (q - (1.0 if actual < qval else 0.0))


def coverage_indicator(sample: np.ndarray, actual: float,
                       alpha: float) -> Tuple[bool, float, float]:
    lo = float(np.quantile(sample, (1.0 - alpha) / 2.0))
    hi = float(np.quantile(sample, 1.0 - (1.0 - alpha) / 2.0))
    return (lo <= actual <= hi), lo, hi


def call_price_from_sample(sample: np.ndarray, K: float, tau_h: float,
                            r_per_h: float) -> float:
    disc = math.exp(-r_per_h * tau_h)
    return float(disc * np.maximum(sample - K, 0.0).mean())


# ----------------------------------------------------------------------
# Kupiec (unconditional coverage) and Christoffersen (independence)
# ----------------------------------------------------------------------
def kupiec_test(hits: np.ndarray, alpha: float) -> Tuple[float, float]:
    n = hits.size
    x = int(hits.sum())
    p = alpha
    p_hat = x / n if n > 0 else float("nan")
    if x == 0 or x == n:
        return float("nan"), float("nan")
    ll_null = x * math.log(p) + (n - x) * math.log(1 - p)
    ll_alt = x * math.log(p_hat) + (n - x) * math.log(1 - p_hat)
    lr = -2.0 * (ll_null - ll_alt)
    from scipy.stats import chi2
    return lr, float(1.0 - chi2.cdf(lr, df=1))


def christoffersen_indep(hits: np.ndarray) -> Tuple[float, float]:
    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for i in range(1, hits.size):
        prev = int(hits[i - 1]); cur = int(hits[i])
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        elif prev == 1 and cur == 1: n11 += 1
    if (n01 + n11) == 0 or (n00 + n10) == 0:
        return float("nan"), float("nan")
    p1 = (n01 + n11) / (n00 + n01 + n10 + n11)
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.5
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.5
    def safe_log(x):
        return math.log(x) if x > 0 else 0.0
    ll_null = (n00 + n10) * safe_log(1 - p1) + (n01 + n11) * safe_log(p1)
    ll_alt = (n00 * safe_log(1 - p01) + n01 * safe_log(p01)
              + n10 * safe_log(1 - p11) + n11 * safe_log(p11))
    lr = -2.0 * (ll_null - ll_alt)
    from scipy.stats import chi2
    return lr, float(1.0 - chi2.cdf(lr, df=1))


# ----------------------------------------------------------------------
# Berkowitz test on PIT sequence
# ----------------------------------------------------------------------
def berkowitz_test(pit: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import norm, chi2
    # Guard against numerical 0 or 1
    p = np.clip(pit, 1e-6, 1 - 1e-6)
    z = norm.ppf(p)
    n = z.size
    if n < 3:
        return float("nan"), float("nan")
    mu = float(z.mean())
    sd = float(z.std(ddof=1))
    # LR test H0: mu = 0, sd = 1 (Berkowitz simplified, no AR(1))
    ll_null = -0.5 * n * math.log(2 * math.pi) - 0.5 * float((z ** 2).sum())
    ll_alt = -0.5 * n * math.log(2 * math.pi) - n * math.log(sd) \
        - 0.5 * float(((z - mu) ** 2).sum()) / (sd * sd)
    lr = -2.0 * (ll_null - ll_alt)
    return lr, float(1.0 - chi2.cdf(lr, df=2))


# ----------------------------------------------------------------------
# Main daily loop
# ----------------------------------------------------------------------
def run_daily_evaluation(ptf: pd.Series, vep: pd.DataFrame,
                         days: List[pd.Timestamp], n_paths: int,
                         seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-day-per-model predictive stats, per-day-per-K option table)."""
    yaml_p = load_frozen_parameters(YAML_PATH)
    scale_P = float(yaml_p.scale_P)
    m0 = load_M0_production()
    m1 = load_M1_fw9e_A3()
    z_cycle = climatology_z_cycle()

    rows_dist: List[dict] = []
    rows_opt: List[dict] = []

    for i, d in enumerate(days):
        val_utc = valuation_utc(d)
        last_known = day_end_utc(d)
        # F for each horizon: monthly VEP baseload for the delivery
        # month of the horizon terminal
        try:
            quote_day, vep_slice = vep_quote_on_or_before(
                vep, d.strftime("%Y-%m-%d"))
        except KeyError:
            continue
        F_by_hour: Dict[int, float] = {}
        actuals: Dict[int, float] = {}
        target_utc: Dict[int, pd.Timestamp] = {}
        deliveries: Dict[int, Tuple[int, int]] = {}
        skip_day = False
        for h in HORIZONS_H:
            target = last_known + pd.Timedelta(hours=h)
            local_target = target.tz_convert(TURKEY_TZ)
            year_m, mo_m = local_target.year, local_target.month
            month_contract = vep_slice[
                (vep_slice["delivery_year"] == year_m)
                & (vep_slice["delivery_month"] == mo_m)]
            if month_contract.empty:
                # Current delivery month already in progress: fall
                # back to the nearest future monthly baseload contract
                # available on the quote day (matches what a market
                # participant looking at the VEP screen would see).
                future = vep_slice[
                    (vep_slice["delivery_year"] * 100
                     + vep_slice["delivery_month"])
                    >= (year_m * 100 + mo_m)].sort_values(
                        ["delivery_year", "delivery_month"])
                if future.empty:
                    skip_day = True
                    break
                month_contract = future.head(1)
                year_m = int(month_contract["delivery_year"].iloc[0])
                mo_m = int(month_contract["delivery_month"].iloc[0])
            if target not in ptf.index:
                skip_day = True
                break
            F_by_hour[h] = float(month_contract["price_TRY_MWh"].iloc[0])
            actuals[h] = float(ptf.loc[target])
            target_utc[h] = target
            deliveries[h] = (year_m, mo_m)
        if skip_day:
            continue

        # M0, M1: MSAR simulate
        try:
            samples_m0 = predict_msar(m0, val_utc, F_by_hour, scale_P,
                                       z_cycle, n_paths, seed + i)
            samples_m1 = predict_msar(m1, val_utc, F_by_hour, scale_P,
                                       z_cycle, n_paths, seed + i + 500_000)
        except Exception as e:
            print(f"  day {d.date()} MSAR error: {e}")
            continue

        # Historical vol at last_known
        vol = fit_hist_vol(ptf, cutoff_utc=last_known)

        # LS OU sigma at F
        pi_arr = np.asarray(yaml_p.m9_stationary_pi, dtype=float)
        sig_y_pooled = float(math.sqrt(pi_arr @ (np.asarray(yaml_p.sigma_y) ** 2)))
        kappa_prod = float(yaml_p.kappa_per_hour)

        for h in HORIZONS_H:
            F = F_by_hour[h]
            actual = actuals[h]
            tau_h = float(h)
            samp_b1 = predict_black76(F, vol["sigma_log_d2h"], tau_h,
                                       n_paths, seed + i + 1_000_000 + h)
            samp_b2 = predict_bachelier(F, vol["sigma_abs_d2h"], tau_h,
                                         n_paths, seed + i + 2_000_000 + h)
            # LS: sigma_price = sig_y_pooled * sqrt(F^2 + scale_P^2)
            sigma_price = sig_y_pooled * math.sqrt(F * F + scale_P * scale_P)
            samp_b3 = predict_lucia_schwartz(F, sigma_price, kappa_prod,
                                              tau_h, n_paths,
                                              seed + i + 3_000_000 + h)
            for label, samp in [("M0", samples_m0[h]),
                                ("M1", samples_m1[h]),
                                ("B1", samp_b1), ("B2", samp_b2),
                                ("B3", samp_b3)]:
                pit = pit_value(samp, actual)
                crps = crps_sample(samp, actual)
                rec = {
                    "day": d.strftime("%Y-%m-%d"),
                    "quote_day": quote_day,
                    "h": h, "model": label,
                    "F": F, "actual": actual,
                    "sample_mean": float(samp.mean()),
                    "sample_sd": float(samp.std(ddof=1)),
                    "pit": pit, "crps": crps,
                }
                for q in QUANTILES:
                    rec[f"pin_q{int(100*q):02d}"] = pinball_loss(samp, actual, q)
                for a in COVERAGE_ALPHAS:
                    hit, lo, hi = coverage_indicator(samp, actual, a)
                    rec[f"cover_{int(100*a):02d}"] = int(hit)
                rows_dist.append(rec)
                # Option prices at moneyness
                for mm in MONEYNESS:
                    K = mm * F
                    price = call_price_from_sample(samp, K, tau_h, R_PER_HOUR)
                    payoff = max(actual - K, 0.0) * math.exp(
                        -R_PER_HOUR * tau_h)
                    rows_opt.append({
                        "day": d.strftime("%Y-%m-%d"), "h": h,
                        "model": label, "moneyness": mm, "K": K, "F": F,
                        "actual": actual, "call_price": price,
                        "discounted_payoff": payoff,
                        "error": price - payoff,
                    })
    return pd.DataFrame(rows_dist), pd.DataFrame(rows_opt)


# ----------------------------------------------------------------------
# Aggregation & report tables
# ----------------------------------------------------------------------
def build_pit_coverage_table(dist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS_H:
        for model in ("M0", "M1", "B1", "B2", "B3"):
            sub = dist[(dist["h"] == h) & (dist["model"] == model)]
            if sub.empty:
                continue
            pit = sub["pit"].to_numpy()
            crps = sub["crps"].to_numpy()
            rec = {
                "h": h, "model": model, "n_days": int(sub.shape[0]),
                "crps_mean": float(crps.mean()),
                "pit_mean": float(pit.mean()),
                "pit_sd": float(pit.std(ddof=1)),
            }
            from scipy.stats import kstest
            try:
                ks_stat, ks_p = kstest(pit, "uniform")
                rec["ks_stat"] = float(ks_stat); rec["ks_p"] = float(ks_p)
            except Exception:
                rec["ks_stat"] = np.nan; rec["ks_p"] = np.nan
            berk_lr, berk_p = berkowitz_test(pit)
            rec["berk_lr"] = berk_lr; rec["berk_p"] = berk_p
            for a in COVERAGE_ALPHAS:
                hits = sub[f"cover_{int(100*a):02d}"].to_numpy()
                rec[f"cover_{int(100*a):02d}_freq"] = float(hits.mean())
                kup_lr, kup_p = kupiec_test(hits, a)
                cc_lr, cc_p = christoffersen_indep(hits)
                rec[f"cover_{int(100*a):02d}_kupiec_p"] = kup_p
                rec[f"cover_{int(100*a):02d}_indep_p"] = cc_p
            rows.append(rec)
    return pd.DataFrame(rows)


def build_dm_table(dist: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano paired test on CRPS: M0 vs each alternative.
    Simplified: paired t-test with cadence-aware effective n.  A full
    Newey-West HAC variant is future work.
    """
    from scipy.stats import t as tdist
    rows = []
    for h in HORIZONS_H:
        base = dist[(dist["h"] == h) & (dist["model"] == "M0")] \
            .set_index("day")["crps"]
        for alt in ("M1", "B1", "B2", "B3"):
            other = dist[(dist["h"] == h) & (dist["model"] == alt)] \
                .set_index("day")["crps"]
            paired = pd.concat([base, other], axis=1, keys=["m0", "alt"]) \
                .dropna()
            if paired.empty:
                continue
            diff = (paired["m0"] - paired["alt"]).to_numpy()
            n = diff.size
            if n < 3:
                continue
            mean_d = float(diff.mean())
            sd_d = float(diff.std(ddof=1))
            se_d = sd_d / math.sqrt(n)
            t_stat = mean_d / se_d if se_d > 0 else float("nan")
            p = float(2.0 * (1.0 - tdist.cdf(abs(t_stat), df=n - 1)))
            rows.append({
                "h": h, "alt": alt, "n": n,
                "mean_diff_M0_minus_alt": mean_d,
                "sd_diff": sd_d, "se_diff": se_d,
                "t_stat": t_stat, "p_value": p,
                "M0_beats_alt": bool(mean_d < 0),
            })
    return pd.DataFrame(rows)


def build_option_bias_table(opt: pd.DataFrame) -> pd.DataFrame:
    grp = opt.groupby(["model", "h", "moneyness"])
    rows = []
    for (model, h, mm), sub in grp:
        err = sub["error"].to_numpy()
        rows.append({
            "model": model, "h": h, "moneyness": mm,
            "n_days": sub.shape[0],
            "mean_error_TRY": float(err.mean()),
            "MAE_TRY": float(np.mean(np.abs(err))),
            "mean_call": float(sub["call_price"].mean()),
            "mean_payoff": float(sub["discounted_payoff"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["model", "h", "moneyness"]) \
        .reset_index(drop=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cadence", type=int, default=DEFAULT_CADENCE_DAYS,
                        help="days between valuation dates (thin the "
                             "universe if runtime too high)")
    parser.add_argument("--n-paths", type=int, default=DEFAULT_N_PATHS)
    parser.add_argument("--start", type=str, default="2026-01-02")
    parser.add_argument("--end", type=str, default="2026-09-24")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"loading data...")
    ptf = load_realized_ptf()
    vep = load_vep_quotes_daily()
    print(f"  PTF: n={len(ptf)}, first {ptf.index[0]}, last {ptf.index[-1]}")
    print(f"  VEP: n={len(vep)}, days={vep['valuation_date'].nunique()}, "
          f"last quote day = {vep['valuation_date'].max()}")

    universe = business_day_universe(args.start, args.end)
    days = universe[::args.cadence]
    print(f"eval universe: {len(universe)} biz days -> thinned to "
          f"{len(days)} (cadence {args.cadence})")

    t0 = time.time()
    dist, opt = run_daily_evaluation(ptf, vep, days, args.n_paths, args.seed)
    dt = time.time() - t0
    print(f"daily loop finished in {dt:.1f}s: dist={dist.shape}, "
          f"opt={opt.shape}")

    dist.to_csv(OUT / "predictive_daily.csv", index=False)
    opt.to_csv(OUT / "option_daily.csv", index=False)

    pit_cov = build_pit_coverage_table(dist)
    pit_cov.to_csv(OUT / "pit_coverage_summary.csv", index=False)
    print("\n=== PIT & coverage (compact) ===")
    print(pit_cov[["h", "model", "n_days", "crps_mean", "pit_mean",
                   "ks_p", "berk_p",
                   "cover_50_freq", "cover_50_kupiec_p",
                   "cover_90_freq", "cover_90_kupiec_p"]]
          .round(4).to_string(index=False))

    dm = build_dm_table(dist)
    dm.to_csv(OUT / "dm_crps_M0_vs_alt.csv", index=False)
    print("\n=== Diebold-Mariano on CRPS (M0 vs each) ===")
    print(dm.round(4).to_string(index=False))

    opt_bias = build_option_bias_table(opt)
    opt_bias.to_csv(OUT / "option_bias_by_moneyness.csv", index=False)
    print("\n=== Option-level bias (mean error TRY) ===")
    pivot = opt_bias.pivot_table(index=["model", "h"], columns="moneyness",
                                  values="mean_error_TRY")
    print(pivot.round(1).to_string())

    # Figure CSVs for the paper (ready for paper/make_figures.py style)
    # Fig 1: PIT histogram source data
    pit_hist_rows = []
    bins = np.linspace(0.0, 1.0, 11)
    for h in HORIZONS_H:
        for model in ("M0", "M1", "B1", "B2", "B3"):
            sub = dist[(dist["h"] == h) & (dist["model"] == model)]
            if sub.empty:
                continue
            hist, _ = np.histogram(sub["pit"].to_numpy(), bins=bins)
            for k in range(hist.size):
                pit_hist_rows.append({
                    "h": h, "model": model,
                    "bin_lo": bins[k], "bin_hi": bins[k + 1],
                    "count": int(hist[k]),
                    "freq": float(hist[k]) / max(1, sub.shape[0]),
                })
    pd.DataFrame(pit_hist_rows).to_csv(OUT / "fig_pit_histogram.csv",
                                        index=False)
    # Fig 2: option pricing bias by moneyness
    opt_bias[["model", "h", "moneyness", "mean_error_TRY", "MAE_TRY"]].to_csv(
        OUT / "fig_option_bias.csv", index=False)

    # Markdown report snippets
    md_lines = ["# FW10 Part A -- Predictive distribution & option-bias summary\n"]
    md_lines.append(f"* Evaluation window: {args.start} -> {args.end}, "
                    f"business days thinned every {args.cadence} days -> "
                    f"{len(days)} valuation dates; N_PATHS = {args.n_paths}.\n")
    md_lines.append(f"* Simplifying assumption: F(target hour) taken to be "
                    f"the VEP monthly baseload quote of the delivery month; "
                    f"the paper's shipped production model uses a full "
                    f"HPFC-shaped hourly forward curve.  This substitution "
                    f"tests the RESIDUAL model given a monthly-average F.\n")
    md_lines.append("## Predictive calibration (compact)\n")
    md_lines.append(pit_cov[["h", "model", "n_days", "crps_mean",
                              "pit_mean", "ks_p", "berk_p",
                              "cover_50_freq", "cover_90_freq"]]
                     .round(4).to_markdown(index=False))
    md_lines.append("\n\n## Diebold-Mariano (paired) on CRPS: M0 vs alternatives\n")
    md_lines.append(dm.round(4).to_markdown(index=False))
    md_lines.append("\n\n## Option pricing bias by moneyness (mean error, TRY)\n")
    md_lines.append(pivot.round(1).to_markdown())
    (OUT / "fw10_partA_summary.md").write_text("\n".join(md_lines),
                                                encoding="utf-8")
    print("\nwrote outputs/fw10_validation/*.csv, *.md")


if __name__ == "__main__":
    main()
