"""FW9f section 4 -- Fast tail and quantile validation.

Compare the observed 2025 A3 TRY residual distribution against the
model-implied stationary distribution from three parameter sets:
  (a) production yaml (M9)
  (b) FW9e full-window A3 consistent fit
  (c) FW9f regime-matched 2022-2025 A3 fit

Method (FW9f rewrite): instead of Monte-Carlo path bundles at a fixed
72 h horizon, simulate ONE long stationary path from the residual
process at the model's own time step and read the stationary
distribution off that path.  Concretely:

  * z(t) is drawn deterministically from the training climatology
    (month-of-year, hour-of-day in Europe/Istanbul), cycled hourly
    over the entire simulation.
  * TVTP transition probabilities per step come from the sigmoid
    p_{ij}(t) = Lambda(alpha_{ij} + gamma_{ij} * z_lag(t)).
  * Regime-conditional AR(1): y_t = mu_{J_t} + phi * y_{t-1}
    + sigma_{J_t} * epsilon_t on the asinh scale.
  * Map y_t to a TRY residual via the delta at spot F:
    r_TRY = y * sqrt(F**2 + scale_P**2).
  * Discard the first 10 000 hours (burn-in) and keep at least
    2 000 000 hours.

Compare with the observed 2025 A3 TRY residual (r = P - monthly_mean
- how_mean).  Report:
  * quantiles 1, 5, 10, 25, 50, 75, 90, 95, 99
  * exceedance frequencies at plus/minus 250, 500, 750, 1000, 1500 TRY
  * Kolmogorov-Smirnov distance vs observed

Note: the observed distribution carries the price cap 4500 and floor 0
that the model does not enforce.  This asymmetric truncation is
flagged in the report; the interior (25--75) and centre-tail
(5--95) statistics are the most comparable.
"""
from __future__ import annotations

import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.params_frozen import load_frozen_parameters
from pde_option_model.tvtp2 import load_hourly_z_history
from scripts.fw9._data import build_fit_frame

OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"
YAML_PATH = REPO_ROOT / "inputs" / "historical" / "m2_frozen_parameters.yaml"
Z_HISTORY_CSV = REPO_ROOT / "inputs" / "historical" / "rd_standardized.csv"
TRAIN_END_UTC = pd.Timestamp("2022-12-31 20:00:00+00:00")
YAML_SCALE_P = 282.48
SPOT = 2917.78
DELTA = float(math.sqrt(SPOT * SPOT + YAML_SCALE_P * YAML_SCALE_P))  # ~2931.42

N_HOURS = 2_000_000
BURN_IN = 10_000
SEED = 20260928
TIME_BUDGET_S = 15 * 60


def _obs_2025_TRY_A3() -> np.ndarray:
    """Observed 2025 A3 TRY residual, matching build_fit_frame + A3."""
    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df[df["ts_utc"] >= pd.Timestamp("2025-01-01", tz="UTC")].reset_index(drop=True)
    ts_local = df["ts_utc"].dt.tz_convert("Europe/Istanbul")
    df["hour"] = ts_local.dt.hour
    df["dow"] = ts_local.dt.dayofweek
    df["how"] = df["dow"] * 24 + df["hour"]
    df["month_ts"] = ts_local.dt.to_period("M").dt.to_timestamp()
    p = df["ptf_TRY_MWh"].to_numpy()
    m_mean = df.groupby("month_ts")["ptf_TRY_MWh"].transform("mean").to_numpy()
    r1 = p - m_mean
    df["_r1"] = r1
    how_mean = df.groupby("how")["_r1"].transform("mean").to_numpy()
    return r1 - how_mean


def _climatology_z_hourly_cycle() -> np.ndarray:
    """Return an 8760-hour cycle of z_lag values keyed by (month, TR-hour).

    Uses the same construction as scenarios.ScenarioBuilder: training
    window = z_history up to TRAIN_END_UTC, group by (TR-local month,
    TR-local hour), take the mean.  The output is a full annual cycle
    starting at UTC 2024-01-01 00:00 (2024 is a leap year, but for
    stationary sampling we truncate to 8760 non-leap hours -- z is a
    periodic function of (month, hour), not date).
    """
    z_hist = load_hourly_z_history(str(Z_HISTORY_CSV))
    train = z_hist.loc[:TRAIN_END_UTC]
    loc = train.index + pd.Timedelta(hours=3)
    clim = train.groupby([loc.month, loc.hour]).mean()  # 12 x 24 = 288 keys
    # Build a 365 * 24 = 8760 hour cycle: iterate over each UTC hour of a
    # non-leap year, look up (TR-local month, TR-local hour).
    start = pd.Timestamp("2023-01-01 00:00:00+00:00")  # non-leap year
    end = pd.Timestamp("2023-12-31 23:00:00+00:00")
    idx = pd.date_range(start, end, freq="h")
    loc_idx = idx + pd.Timedelta(hours=3)
    keys = list(zip(loc_idx.month, loc_idx.hour))
    return np.array([clim.get(k, float(train.mean())) for k in keys],
                    dtype=float)


def _stationary_pi_no_z(a01: float, a10: float) -> np.ndarray:
    """Constant-transition stationary pi (used only for warm start)."""
    p01 = 1.0 / (1.0 + math.exp(-a01))
    p10 = 1.0 / (1.0 + math.exp(-a10))
    pi1 = p01 / (p01 + p10)
    return np.array([1.0 - pi1, pi1])


def _simulate_stationary(mu_n: float, mu_s: float,
                         sigma_n: float, sigma_s: float, phi: float,
                         a01: float, g01: float, a10: float, g10: float,
                         n_hours: int, burn_in: int,
                         z_cycle: np.ndarray, seed: int,
                         label: str) -> np.ndarray:
    """Long single-path stationary simulation of the MS-AR(1) + TVTP.

    Returns the burn-in-trimmed asinh residual sample (dimensionless).
    """
    rng = np.random.default_rng(seed)
    n_total = int(n_hours + burn_in)
    y = np.empty(n_total, dtype=np.float64)
    J = np.empty(n_total, dtype=np.int8)

    # Warm start
    pi0 = _stationary_pi_no_z(a01, a10)
    J[0] = int(rng.random() < pi0[1])
    y[0] = 0.0  # unconditional mean is 0 under mixture; burn-in absorbs the transient

    # Pre-draw all Gaussians and Uniforms in blocks to stay fast
    BLOCK = 500_000
    mu = np.array([mu_n, mu_s], dtype=np.float64)
    sig = np.array([sigma_n, sigma_s], dtype=np.float64)

    cycle_len = int(z_cycle.size)

    t0 = time.time()
    t = 1
    while t < n_total:
        end = min(t + BLOCK, n_total)
        m = end - t
        eps = rng.standard_normal(m)
        u = rng.random(m)
        for k in range(m):
            idx = (t + k - 1) % cycle_len
            z = z_cycle[idx]
            prev_J = J[t + k - 1]
            if prev_J == 0:
                p_switch = 1.0 / (1.0 + math.exp(-(a01 + g01 * z)))
                new_J = 1 if u[k] < p_switch else 0
            else:
                p_switch = 1.0 / (1.0 + math.exp(-(a10 + g10 * z)))
                new_J = 0 if u[k] < p_switch else 1
            J[t + k] = new_J
            y[t + k] = mu[new_J] + phi * y[t + k - 1] + sig[new_J] * eps[k]
        t = end
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"  {label}: time budget exceeded at t={t}, stopping early")
            break

    dt = time.time() - t0
    y_out = y[burn_in:t]
    j_out = J[burn_in:t]
    print(f"  {label}: simulated {y_out.size} hours in {dt:.1f}s, "
          f"y mean={y_out.mean():.4f}, sd={y_out.std():.4f}, "
          f"stress occ={j_out.mean():.3f}")
    return y_out


def _quantile_table(sample: np.ndarray) -> dict:
    quantiles = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    return {f"q{int(100 * q):02d}": float(np.quantile(sample, q))
            for q in quantiles}


def _exceedance_table(sample: np.ndarray) -> dict:
    thresholds_pos = (250, 500, 750, 1000, 1500)
    out = {}
    for th in thresholds_pos:
        out[f"P_gt_{th}"] = float((sample > th).mean())
        out[f"P_lt_-{th}"] = float((sample < -th).mean())
    return out


def _param_row(mu_n, mu_s, sig_n, sig_s, phi, a01, g01, a10, g10) -> dict:
    return {"mu_n": mu_n, "mu_s": mu_s,
            "sigma_n": sig_n, "sigma_s": sig_s,
            "phi": phi, "a01": a01, "g01": g01,
            "a10": a10, "g10": g10}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obs = _obs_2025_TRY_A3()
    print(f"Observed 2025 A3 TRY residual: n={obs.size}, "
          f"mean={obs.mean():.2f}, std={obs.std():.2f}")
    print(f"Building climatology z hourly cycle...")
    z_cycle = _climatology_z_hourly_cycle()
    print(f"  z_cycle: n={z_cycle.size}, mean={z_cycle.mean():.4f}, "
          f"sd={z_cycle.std():.4f}")

    # (a) production yaml
    yaml_p = load_frozen_parameters(YAML_PATH)
    prod = _param_row(mu_n=0.0, mu_s=0.0,
                      sig_n=float(yaml_p.sigma_y[0]),
                      sig_s=float(yaml_p.sigma_y[1]),
                      phi=float(yaml_p.phi),
                      a01=float(yaml_p.alpha01), g01=float(yaml_p.gamma01),
                      a10=float(yaml_p.alpha10), g10=float(yaml_p.gamma10))

    # (b) FW9e full-window A3
    with open(OUT / "TVTP_1cov_A3.pkl", "rb") as f:
        r_b = pickle.load(f).params
    fw9e = _param_row(mu_n=r_b.mu_normal, mu_s=r_b.mu_stress,
                      sig_n=r_b.sigma_normal, sig_s=r_b.sigma_stress,
                      phi=r_b.phi,
                      a01=r_b.alpha01, g01=r_b.gamma01,
                      a10=r_b.alpha10, g10=r_b.gamma10)

    # (c) FW9f regime-matched 2022-2025
    pkl_c = OUT / "TVTP_1cov_A3_2022_2025.pkl"
    if not pkl_c.exists():
        raise SystemExit("run scripts/fw9/consistent_msar_fit_2022_2025.py first")
    with open(pkl_c, "rb") as f:
        r_c = pickle.load(f).params
    fw9f = _param_row(mu_n=r_c.mu_normal, mu_s=r_c.mu_stress,
                      sig_n=r_c.sigma_normal, sig_s=r_c.sigma_stress,
                      phi=r_c.phi,
                      a01=r_c.alpha01, g01=r_c.gamma01,
                      a10=r_c.alpha10, g10=r_c.gamma10)

    print("\nSimulating (a) production...")
    y_a = _simulate_stationary(**prod, n_hours=N_HOURS, burn_in=BURN_IN,
                               z_cycle=z_cycle, seed=SEED, label="production")
    print("\nSimulating (b) FW9e full-window A3 fit...")
    y_b = _simulate_stationary(**fw9e, n_hours=N_HOURS, burn_in=BURN_IN,
                               z_cycle=z_cycle, seed=SEED + 1,
                               label="FW9e_full_A3")
    print("\nSimulating (c) FW9f regime-matched 2022-2025 A3 fit...")
    y_c = _simulate_stationary(**fw9f, n_hours=N_HOURS, burn_in=BURN_IN,
                               z_cycle=z_cycle, seed=SEED + 2,
                               label="FW9f_regime_matched")

    # Delta-map to TRY at spot F
    r_a = y_a * DELTA
    r_b_TRY = y_b * DELTA
    r_c_TRY = y_c * DELTA

    rows = []
    for name, s in [("observed_2025", obs), ("production", r_a),
                    ("FW9e_full_A3", r_b_TRY),
                    ("FW9f_regime_matched", r_c_TRY)]:
        rec = {"source": name, "n": s.size, "mean": float(s.mean()),
               "std": float(s.std())}
        rec.update(_quantile_table(s))
        rec.update(_exceedance_table(s))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tail_validation.csv", index=False)

    ks_rows = []
    for name, s in [("production", r_a), ("FW9e_full_A3", r_b_TRY),
                    ("FW9f_regime_matched", r_c_TRY)]:
        ks = stats.ks_2samp(obs, s)
        q05_obs, q95_obs, q99_obs = (float(np.quantile(obs, q))
                                      for q in (0.05, 0.95, 0.99))
        q01_obs = float(np.quantile(obs, 0.01))
        q05_mod, q95_mod, q99_mod = (float(np.quantile(s, q))
                                      for q in (0.05, 0.95, 0.99))
        q01_mod = float(np.quantile(s, 0.01))
        ks_rows.append({
            "source": name,
            "KS_statistic": float(ks.statistic),
            "KS_pvalue": float(ks.pvalue),
            "q01_obs": q01_obs, "q01_model": q01_mod,
            "q01_delta_TRY": q01_mod - q01_obs,
            "q05_obs": q05_obs, "q05_model": q05_mod,
            "q05_delta_TRY": q05_mod - q05_obs,
            "q95_obs": q95_obs, "q95_model": q95_mod,
            "q95_delta_TRY": q95_mod - q95_obs,
            "q99_obs": q99_obs, "q99_model": q99_mod,
            "q99_delta_TRY": q99_mod - q99_obs,
        })
    ks_df = pd.DataFrame(ks_rows)
    ks_df.to_csv(OUT / "tail_validation_ks.csv", index=False)

    cap_notes = {
        "n_obs": int(obs.size),
        "obs_min": float(obs.min()),
        "obs_max": float(obs.max()),
        "n_extreme_positive_gt_1500": int((obs > 1500).sum()),
        "n_extreme_negative_lt_-1500": int((obs < -1500).sum()),
    }

    print("\n=== Distribution comparison ===")
    print(df.round(3).T.to_string())
    print("\n=== KS + tail deltas ===")
    print(ks_df.round(4).T.to_string())
    print("\n=== Observed cap/floor exposure ===")
    for k, v in cap_notes.items():
        print(f"  {k}: {v}")

    md = [
        "# FW9f section 4 -- Fast tail and quantile validation\n",
        f"Observed 2025 A3 TRY residual: n = {obs.size}, "
        f"mean = {obs.mean():.2f}, std = {obs.std():.2f}\n",
        f"Model-implied stationary residual: single long-path simulation "
        f"of the MS-AR(1) plus TVTP process at 1 h step; each path "
        f"n = {N_HOURS} hours after {BURN_IN} h burn-in; climatology z_lag "
        f"cycled hourly; asinh residual mapped to TRY via "
        f"delta = sqrt(F^2 + s_P^2) = {DELTA:.2f} at F = {SPOT} TRY/MWh.\n",
        "## Quantile / exceedance summary\n",
        df.round(3).to_markdown(index=False),
        "\n\n## KS distance + tail quantile deltas vs observed\n",
        ks_df.round(4).to_markdown(index=False),
        "\n\n## Observed cap/floor exposure\n",
        f"* observed 2025 min residual: **{cap_notes['obs_min']:.1f} TRY**",
        f"* observed 2025 max residual: **{cap_notes['obs_max']:.1f} TRY**",
        f"* observed hours with residual > +1500 TRY: "
        f"{cap_notes['n_extreme_positive_gt_1500']}",
        f"* observed hours with residual < -1500 TRY: "
        f"{cap_notes['n_extreme_negative_lt_-1500']}",
        "\n\nThe observed distribution is truncated on the negative side "
        "at approximately -F(monthly_mean) (price floor 0) and on the "
        "positive side at approximately 4500 - F(monthly_mean) (price "
        "cap 4500 TRY/MWh until 2026-04-04, 5000 after).  The model has "
        "no cap or floor, so mismatches at the extreme (1 pct, 99 pct) "
        "tails reflect both the model residual dispersion and this "
        "asymmetric truncation.  The 25-75 interquartile range, the "
        "5-95 centre-tail band and the KS statistic are the most "
        "comparable summaries.\n",
    ]
    (OUT / "tail_validation.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote tail_validation.csv, tail_validation_ks.csv and "
          "tail_validation.md")


if __name__ == "__main__":
    main()
