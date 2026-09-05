"""Runtime validation: generator and probability checks on the actual data,
PDE limiting-case tests, convergence, Crank-Nicolson vs implicit Euler, and a
Monte Carlo cross-check under exactly the same pricing-measure dynamics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .contracts import EuropeanOption
from .dynamics import RegimeDynamics
from .generator import (expm_reproduction_error, generator_to_probs,
                        probs_to_generator, stationary_distribution)
from .grid import TimeGrid
from .markov_adapter import MarkovInputs
from .pricing import (GridSettings, PricingResult, build_dynamics, build_grid,
                      price_contract)
from .risk_neutral import MeasureAdjustment, baseline_q1
from .scenarios import ScenarioPath
from .solver import SolverSettings

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def row(self) -> dict:
        return {"check": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(CheckResult(name, bool(passed), detail))
        (logger.info if passed else logger.error)("[%s] %s: %s",
                                                  "PASS" if passed else "FAIL",
                                                  name, detail)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.row() for c in self.checks])


# ---------------------------------------------------------------------------
# 1. transition-generator and probability checks on the actual artifacts
# ---------------------------------------------------------------------------
def run_generator_checks(inputs: MarkovInputs, raw_ts: pd.DataFrame,
                         report: ValidationReport) -> None:
    ts = raw_ts
    # shipped rows
    tol = 1e-10
    report.add("p00+p01=1 (shipped)",
               float(np.abs(ts.transition_p00 + ts.transition_p01 - 1).max()) < tol,
               f"max abs dev {float(np.abs(ts.transition_p00 + ts.transition_p01 - 1).max()):.2e}")
    report.add("p10+p11=1 (shipped)",
               float(np.abs(ts.transition_p10 + ts.transition_p11 - 1).max()) < tol,
               f"max abs dev {float(np.abs(ts.transition_p10 + ts.transition_p11 - 1).max()):.2e}")
    q01s, q10s = ts.transition_q01_per_hour.values, ts.transition_q10_per_hour.values
    report.add("q01,q10 >= 0 (shipped)", bool((q01s.min() >= 0) and (q10s.min() >= 0)),
               f"min q01 {q01s.min():.4f}, min q10 {q10s.min():.4f}")
    rows0 = np.abs(ts.transition_q00_per_hour + ts.transition_q01_per_hour).max()
    rows1 = np.abs(ts.transition_q11_per_hour + ts.transition_q10_per_hour).max()
    report.add("generator rows sum to 0 (shipped)",
               float(max(rows0, rows1)) < tol, f"max abs {float(max(rows0, rows1)):.2e}")
    gen = probs_to_generator(ts.transition_p01.values, ts.transition_p10.values)
    dev = max(np.abs(gen.q01 - q01s).max(), np.abs(gen.q10 - q10s).max())
    report.add("shipped q == matrix-log(shipped p)", dev < 1e-10,
               f"max abs dev {dev:.2e} (confirms conversion formula)")
    err = expm_reproduction_error(ts.transition_p01.values, ts.transition_p10.values)
    report.add("expm(Q dt) reproduces P (shipped)", err < 1e-10, f"max abs {err:.2e}")

    # recomputed M2 TVTP path on the historical covariate
    z = inputs.z_history.shift(1).dropna().values
    p01, p10 = inputs.tvtp.probabilities(z)
    genM2 = probs_to_generator(p01, p10)
    errM2 = expm_reproduction_error(p01, p10)
    s = p01 + p10
    report.add("M2 TVTP embeddable on history",
               genM2.n_clipped == 0 and s.max() < 1.0,
               f"s=p01+p10 in [{s.min():.3f}, {s.max():.3f}], clipped rows {genM2.n_clipped}")
    report.add("expm(Q dt) reproduces P (M2 recomputed)", errM2 < 1e-10,
               f"max abs {errM2:.2e}")


def run_probability_checks(inputs: MarkovInputs, raw_ts: pd.DataFrame,
                           smoothed: Optional[pd.DataFrame],
                           report: ValidationReport) -> None:
    f = np.abs(raw_ts.filter_p_state0 + raw_ts.filter_p_state1 - 1).max()
    report.add("filtered probabilities sum to 1", float(f) < 1e-10,
               f"max abs dev {float(f):.2e}")
    if smoothed is not None:
        s = np.abs(smoothed.p_state0 + smoothed.p_state1 - 1).max()
        report.add("smoothed probabilities sum to 1", float(s) < 1e-10,
                   f"max abs dev {float(s):.2e}")
    pi = inputs.pi_filtered
    row = raw_ts.loc[inputs.valuation_utc]
    shipped = np.array([row.filter_p_state0, row.filter_p_state1])
    expected = shipped[::-1] if inputs.labels_flipped_in_shipped_series else shipped
    ok = np.allclose(pi, expected, atol=1e-12) and abs(pi.sum() - 1) < 1e-10
    report.add("latest filtered vector correctly used (orientation-aligned)",
               ok, f"pi(normal,stress)={np.round(pi, 6).tolist()} from shipped "
                   f"{np.round(shipped, 6).tolist()} "
                   f"(flipped={inputs.labels_flipped_in_shipped_series})")


# ---------------------------------------------------------------------------
# 2. PDE limiting cases
# ---------------------------------------------------------------------------
def _flat_scenario(tau_hours: float, z: float = 0.0) -> ScenarioPath:
    t = np.linspace(0.0, tau_hours, max(int(tau_hours) + 1, 3))
    return ScenarioPath("validation_flat", t, np.full(t.size, z), np.full(t.size, z))


def run_pde_limit_checks(
    inputs: MarkovInputs,
    contract: EuropeanOption,
    report: ValidationReport,
    gs: Optional[GridSettings] = None,
) -> dict[str, PricingResult]:
    gs = gs or GridSettings(n_space_nodes=401)
    scen = _flat_scenario(contract.tau_hours, 0.0)
    dyn = build_dynamics(inputs)
    out: dict[str, PricingResult] = {}

    base = price_contract(inputs, contract, scen, grid_settings=gs, dynamics=dyn)
    out["base"] = base

    # (1) q = 0 -> two decoupled PDEs
    import copy
    adj0 = MeasureAdjustment(spec="Q2", eta=np.array([-60.0, -60.0]))  # q * e^-60 ~ 0
    dec = price_contract(inputs, contract, scen, adjustment=adj0,
                         grid_settings=gs, dynamics=dyn)
    ind = []
    for i in range(2):
        # replicate regime i in both slots, then solve with q ~ 0
        inp_i = copy.copy(inputs)
        inp_i.ou = type(inputs.ou)(
            kappa_per_hour=inputs.ou.kappa_per_hour,
            theta_base=np.array([inputs.ou.theta_base[i]] * 2),
            sigma_ou=np.array([inputs.ou.sigma_ou[i]] * 2),
            one_minus_phi=inputs.ou.one_minus_phi,
            dt_hours=inputs.ou.dt_hours,
            mapping=inputs.ou.mapping)
        inp_i.rho = np.array([inputs.rho[i]] * 2)
        r_i = price_contract(inp_i, contract, scen, adjustment=adj0,
                             grid_settings=gs, dynamics=build_dynamics(inp_i),
                             grid=dec.solve.grid)  # SAME grid: isolates coupling
        ind.append(r_i.V_regime[0])
    dev = float(np.max(np.abs(dec.V_regime - np.array(ind))))
    tolv = 1e-6 * max(1.0, abs(base.value))
    report.add("q=0 decouples into independent PDEs", dev < tolv,
               f"max |coupled(q~0) - independent| = {dev:.3e} (tol {tolv:.1e})")

    # (2) identical regimes -> V0 == V1
    inp_id = copy.copy(inputs)
    inp_id.ou = type(inputs.ou)(
        kappa_per_hour=inputs.ou.kappa_per_hour,
        theta_base=np.array([inputs.ou.theta_base.mean()] * 2),
        sigma_ou=np.array([inputs.ou.sigma_ou[1]] * 2),
        one_minus_phi=inputs.ou.one_minus_phi,
        dt_hours=inputs.ou.dt_hours, mapping=inputs.ou.mapping)
    inp_id.rho = np.array([inputs.rho.mean()] * 2)
    dyn_id = build_dynamics(inp_id)
    rid = price_contract(inp_id, contract, scen, grid_settings=gs, dynamics=dyn_id)
    dev2 = abs(rid.V_regime[0] - rid.V_regime[1])
    report.add("identical regimes give V0 == V1",
               dev2 < 1e-8 * max(1.0, abs(rid.value)),
               f"|V0 - V1| = {dev2:.3e}")

    # (3) very fast switching of similar regimes -> averaged-regime solution
    inp_sim = copy.copy(inputs)
    sig_sim = np.array([0.9, 1.1]) * inputs.ou.sigma_ou[1]
    th_sim = inputs.ou.theta_base.mean() + np.array([-0.05, 0.05])
    inp_sim.ou = type(inputs.ou)(
        kappa_per_hour=inputs.ou.kappa_per_hour, theta_base=th_sim,
        sigma_ou=sig_sim, one_minus_phi=inputs.ou.one_minus_phi,
        dt_hours=inputs.ou.dt_hours, mapping=inputs.ou.mapping)
    inp_sim.rho = np.zeros(2)
    dyn_sim = build_dynamics(inp_sim, include_rd_in_theta=False)
    adj_fast = MeasureAdjustment(spec="Q2", eta=np.array([4.0, 4.0]))
    fast = price_contract(inp_sim, contract, scen, adjustment=adj_fast,
                          grid_settings=gs, dynamics=dyn_sim)
    p01f, p10f = inputs.tvtp.probabilities(np.array([0.0]))
    g = probs_to_generator(p01f, p10f)
    pi_st = stationary_distribution(float(g.q01[0]) * np.e**4, float(g.q10[0]) * np.e**4)
    inp_avg = copy.copy(inp_sim)
    inp_avg.ou = type(inputs.ou)(
        kappa_per_hour=inputs.ou.kappa_per_hour,
        theta_base=np.array([float(pi_st @ th_sim)] * 2),
        sigma_ou=np.array([float(np.sqrt(pi_st @ sig_sim**2))] * 2),
        one_minus_phi=inputs.ou.one_minus_phi,
        dt_hours=inputs.ou.dt_hours, mapping=inputs.ou.mapping)
    dyn_avg = build_dynamics(inp_avg, include_rd_in_theta=False)
    avg = price_contract(inp_avg, contract, scen, grid_settings=gs, dynamics=dyn_avg)
    rel = abs(fast.value - avg.value) / max(abs(avg.value), 1e-8)
    report.add("fast switching approaches averaged-regime solution", rel < 0.05,
               f"fast={fast.value:.4f} vs averaged={avg.value:.4f} (rel dev {rel:.2%})")

    # (4)-(7) sign and strike monotonicity
    strikes = np.array([0.7, 1.0, 1.3]) * contract.strike
    calls, puts = [], []
    for K in strikes:
        c = EuropeanOption("call", K, contract.valuation_utc,
                           contract.maturity_utc, contract.r_annual)
        p = EuropeanOption("put", K, contract.valuation_utc,
                           contract.maturity_utc, contract.r_annual)
        calls.append(price_contract(inputs, c, scen, grid_settings=gs,
                                    dynamics=dyn).value)
        puts.append(price_contract(inputs, p, scen, grid_settings=gs,
                                   dynamics=dyn).value)
    calls, puts = np.array(calls), np.array(puts)
    report.add("call prices non-negative", bool((calls >= -1e-8).all()),
               f"min {calls.min():.4f}")
    report.add("put prices non-negative", bool((puts >= -1e-8).all()),
               f"min {puts.min():.4f}")
    report.add("call decreasing in strike", bool(np.all(np.diff(calls) < 1e-8)),
               f"values {np.round(calls, 3).tolist()} at K={np.round(strikes, 0).tolist()}")
    report.add("put increasing in strike", bool(np.all(np.diff(puts) > -1e-8)),
               f"values {np.round(puts, 3).tolist()}")
    out["calls_by_strike"] = calls
    out["puts_by_strike"] = puts
    return out


# ---------------------------------------------------------------------------
# 3. convergence and scheme comparison
# ---------------------------------------------------------------------------
def run_convergence_check(
    inputs: MarkovInputs, contract: EuropeanOption, report: ValidationReport,
    nodes: Sequence[int] = (141, 281, 561, 1121),
    steps_per_hour: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> pd.DataFrame:
    scen = _flat_scenario(contract.tau_hours, 0.0)
    dyn = build_dynamics(inputs)
    vals = []
    for n, sph in zip(nodes, steps_per_hour):
        gs = GridSettings(n_space_nodes=n,
                          n_time_steps=max(48, int(sph * contract.tau_hours)))
        v = price_contract(inputs, contract, scen, grid_settings=gs,
                           dynamics=dyn).value
        vals.append({"n_nodes": n, "n_steps": gs.n_time_steps, "value": v})
    df = pd.DataFrame(vals)
    diffs = np.abs(np.diff(df.value.values))
    shrinking = bool(diffs[-1] < diffs[0])
    order = np.log2(diffs[:-1] / diffs[1:]) if len(diffs) >= 2 else np.array([])
    report.add("grid refinement converges",
               shrinking and diffs[-1] < 5e-3 * abs(df.value.iloc[-1]),
               f"successive |dV| {np.round(diffs, 5).tolist()}, "
               f"implied order {np.round(order, 2).tolist()}")

    gs = GridSettings(n_space_nodes=561,
                      n_time_steps=max(96, int(2 * contract.tau_hours)))
    cn = price_contract(inputs, contract, scen, grid_settings=gs, dynamics=dyn,
                        solver_settings=SolverSettings(theta_scheme=0.5)).value
    ie = price_contract(inputs, contract, scen, grid_settings=gs, dynamics=dyn,
                        solver_settings=SolverSettings(theta_scheme=1.0,
                                                       rannacher_steps=0)).value
    rel = abs(cn - ie) / max(abs(cn), 1e-12)
    report.add("Crank-Nicolson agrees with implicit Euler", rel < 0.01,
               f"CN {cn:.4f} vs IE {ie:.4f} (rel dev {rel:.3%})")
    return df


# ---------------------------------------------------------------------------
# 4. Monte Carlo under identical pricing-measure dynamics
# ---------------------------------------------------------------------------
def simulate_paths(
    inputs: MarkovInputs,
    contract: EuropeanOption,
    scenario: ScenarioPath,
    adjustment: Optional[MeasureAdjustment] = None,
    dynamics: Optional[RegimeDynamics] = None,
    n_paths: int = 40_000,
    dt_hours: float = 0.5,
    seed: int = 20260805,
    initial_regime: Optional[int] = None,
    return_horizon_prices: Optional[Sequence[float]] = None,
) -> dict:
    """Exact-in-distribution simulation of the coupled regime/OU dynamics.

    Per step: (i) the regime switches with the exact 2-state transition
    probability implied by the frozen generator over dt; (ii) conditional on
    the regime, Y is advanced with the exact OU transition using the local
    theta_i(t).  This matches the PDE dynamics up to the piecewise freezing of
    coefficients over dt.
    """
    rng = np.random.default_rng(seed)
    adjustment = adjustment or baseline_q1()
    dynamics = dynamics or build_dynamics(inputs)
    tau = contract.tau_hours
    n_steps = int(np.ceil(tau / dt_hours))
    dt = tau / n_steps
    t_nodes = np.linspace(0.0, tau, n_steps + 1)

    z_now = np.interp(t_nodes, scenario.times_hours, scenario.z)
    z_lag = np.interp(t_nodes, scenario.times_hours, scenario.z_lagged)
    theta = dynamics.theta_path(inputs.valuation_utc, t_nodes, z_now,
                                adjustment=None) + adjustment.theta_shift[:, None]
    p01h, p10h = inputs.tvtp.probabilities(z_lag)
    gen = probs_to_generator(p01h, p10h, dt_hours=inputs.dt_hours)
    q01, q10 = adjustment.adjust_generator(gen.q01, gen.q10)

    sig = dynamics.sigma()
    kap = inputs.ou.kappa_per_hour
    e1 = np.exp(-kap * dt)
    std = sig * np.sqrt((1.0 - np.exp(-2.0 * kap * dt)) / (2.0 * kap))
    drift_extra = (adjustment.drift_shift_per_hour
                   - adjustment.market_price_of_risk * sig)

    y = np.full(n_paths, inputs.y0)
    if initial_regime is None:
        reg = (rng.random(n_paths) < inputs.pi_filtered[1]).astype(np.int8)
    else:
        reg = np.full(n_paths, initial_regime, dtype=np.int8)

    horizon_records: dict[float, np.ndarray] = {}
    want = sorted(return_horizon_prices or [])
    for k in range(n_steps):
        p01k, p10k = generator_to_probs(np.array([q01[k]]), np.array([q10[k]]),
                                        dt_hours=dt)
        u = rng.random(n_paths)
        switch0 = (reg == 0) & (u < p01k[0])
        switch1 = (reg == 1) & (u < p10k[0])
        reg = np.where(switch0, 1, np.where(switch1, 0, reg)).astype(np.int8)

        th_mid = 0.5 * (theta[:, k] + theta[:, k + 1])
        th_p = th_mid[reg] + drift_extra[reg] / kap
        y = th_p + (y - th_p) * e1 + std[reg] * rng.standard_normal(n_paths)
        t_now = t_nodes[k + 1]
        for hzn in want:
            key = float(hzn)
            if key not in horizon_records and t_now >= key - 0.5 * dt:
                horizon_records[key] = np.asarray(
                    inputs.transform.price_from_y(y)).copy()

    prices_T = np.asarray(inputs.transform.price_from_y(y))
    pay = contract.payoff_from_price(prices_T)
    disc = np.exp(-contract.r_per_hour * tau)
    mc_value = disc * pay.mean()
    mc_se = disc * pay.std(ddof=1) / np.sqrt(n_paths)
    return {
        "value": float(mc_value),
        "std_error": float(mc_se),
        "mean_price_T": float(prices_T.mean()),
        "n_paths": n_paths,
        "dt_hours": dt,
        "horizon_prices": horizon_records,          # per-path price arrays
        "horizon_means": {h: float(v.mean()) for h, v in horizon_records.items()},
    }


def run_mc_check(
    inputs: MarkovInputs, contract: EuropeanOption, report: ValidationReport,
    n_paths: int = 40_000, gs: Optional[GridSettings] = None,
) -> dict:
    scen = _flat_scenario(contract.tau_hours, 0.0)
    dyn = build_dynamics(inputs)
    gs = gs or GridSettings(n_space_nodes=561)
    pde = price_contract(inputs, contract, scen, grid_settings=gs, dynamics=dyn)
    mc = simulate_paths(inputs, contract, scen, dynamics=dyn, n_paths=n_paths)
    z = abs(pde.value - mc["value"]) / max(mc["std_error"], 1e-12)
    report.add("PDE matches Monte Carlo (same Q dynamics)", z < 3.5,
               f"PDE {pde.value:.4f} vs MC {mc['value']:.4f} "
               f"+/- {mc['std_error']:.4f} (|z| = {z:.2f})")
    for i, nm in enumerate(("normal", "stress")):
        mc_i = simulate_paths(inputs, contract, scen, dynamics=dyn,
                              n_paths=n_paths // 2, initial_regime=i, seed=7 + i)
        z_i = abs(pde.V_regime[i] - mc_i["value"]) / max(mc_i["std_error"], 1e-12)
        report.add(f"PDE matches MC conditional on regime {i} ({nm})", z_i < 3.5,
                   f"PDE {pde.V_regime[i]:.4f} vs MC {mc_i['value']:.4f} "
                   f"+/- {mc_i['std_error']:.4f} (|z| = {z_i:.2f})")
    return {"pde": pde, "mc": mc}


# ---------------------------------------------------------------------------
def run_full_validation(
    inputs: MarkovInputs, raw_ts: pd.DataFrame, smoothed: Optional[pd.DataFrame],
    contract: EuropeanOption,
) -> tuple[ValidationReport, dict]:
    report = ValidationReport()
    extras: dict = {}
    run_generator_checks(inputs, raw_ts, report)
    run_probability_checks(inputs, raw_ts, smoothed, report)
    extras["limits"] = run_pde_limit_checks(inputs, contract, report)
    extras["convergence"] = run_convergence_check(inputs, contract, report)
    extras["mc"] = run_mc_check(inputs, contract, report)
    return report, extras
