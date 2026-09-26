"""1D vs EXPERIMENTAL 2D TVTP comparison: ramp-effect ladder and sensitivities.

    python scripts/tvtp2/compare_tvtp_1d_2d.py                 # full run (~15 min, 2 cores)
    python scripts/tvtp2/compare_tvtp_1d_2d.py --quick         # 72 h, K = 3000 only

Holds F(t), kappa, sigma, pi0, discount rate, contracts, the residual space
grid (union x-range per maturity), time grid, ODE sub-step and the Monte Carlo
seed / paths / step fixed; only the transition law (and, in the sensitivity
block, exactly one other input) changes.  Writes to
outputs/tvtp2_experimental/comparison/ (never to accepted output folders).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_pde as R                                                     # noqa: E402
from pde_option_model import tvtp2_compare as C                         # noqa: E402
from pde_option_model.params_frozen import load_tvtp2_parameters       # noqa: E402
from pde_option_model.tvtp2 import (EXPERIMENTAL_LABEL,                # noqa: E402
                                    load_hourly_z_history, sha256_file)

BUNDLE = "inputs/historical/archive/calibration_bundle"
HISTORY = "inputs/historical/rd_standardized.csv"
TVTP2_YAML = "inputs/historical/tvtp2_frozen_parameters.yaml"
BASE_YAML = "inputs/historical/m2_frozen_parameters.yaml"
_W: dict = {}


def _setup(sens: bool, maturities, gs_override=None):
    import os
    os.chdir(REPO_ROOT)
    cfg = R._load_config(None)                       # production config, unchanged
    ns = argparse.Namespace(quotes=None, params=None)
    quotes, params, _, ppath = R._load_market_inputs(ns, cfg)
    tvtp2 = load_tvtp2_parameters(TVTP2_YAML, base_params=params, base_params_path=ppath)
    base = R._build_model(quotes, params, cfg, R._build_anchor(cfg, None, None),
                          "smooth_constrained")
    gs = gs_override or R._grid_settings(cfg)
    r_annual = float(R._get(cfg, "contract.r_annual", 0.40))
    z = load_hourly_z_history(HISTORY)
    train_end = pd.Timestamp(R._get(cfg, "scenario.train_end_utc", "2022-12-31 20:00:00+00:00"))
    ctx = C.build_context(base, params, tvtp2, z, BUNDLE, train_end, HISTORY,
                          include_sensitivities=sens)
    paths = C.build_paths(ctx, params, maturities, gs)
    return cfg, quotes, params, tvtp2, base, gs, r_annual, ctx, paths


def _init(sens, maturities, grids):
    cfg, quotes, params, tvtp2, base, gs, r, ctx, paths = _setup(sens, maturities)
    _W.update(params=params, base=base, gs=gs, r=r, ctx=ctx, paths=paths, grids=grids,
              runs={x.name: x for x in ctx["runs"]})


def _task(task):
    kind = task[0]
    run = _W["runs"][task[1]]
    T = float(task[2])
    path = _W["paths"][(run.path_key, T)]
    if kind == "pde":
        _, _, _, K, with_put, nts = task
        return ("pde", C.price_one(run, _W["base"], _W["params"], path, T, K, _W["r"],
                                   _W["gs"], _W["grids"][T], with_put=with_put,
                                   n_time_steps=nts), nts)
    _, _, _, strikes, n_paths, dt, seed = task
    return ("mc", C.mc_one(run, _W["base"], _W["params"], path, T, strikes, _W["r"],
                           n_paths, dt, seed), dt)


def _pct(a, b):
    return 100.0 * (a - b) / b if b not in (0.0,) and np.isfinite(b) and abs(b) > 1e-12 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="outputs/tvtp2_experimental/comparison")
    ap.add_argument("--quick", action="store_true", help="72 h and K = 3000 only")
    ap.add_argument("--no-sensitivities", action="store_true")
    ap.add_argument("--no-mc", action="store_true")
    ap.add_argument("--mc-paths", type=int, default=40_000)
    ap.add_argument("--mc-dt", type=float, default=0.05)
    ap.add_argument("--mc-seed", type=int, default=20260808)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)

    out = (REPO_ROOT / args.outdir)
    for prot in R.PROTECTED_OUTPUT_DIRS:
        pr = (REPO_ROOT / prot).resolve()
        if out.resolve() == pr or pr in out.resolve().parents:
            print(f"ERROR: refusing to write into {prot}", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)
    mats = (72.0,) if args.quick else tuple(float(x) for x in C.MATURITIES_H)
    strikes = (3000.0,) if args.quick else C.STRIKES
    sstrikes = (3000.0,) if args.quick else C.SENS_STRIKES
    sens = not args.no_sensitivities
    t0 = time.time()
    cfg, quotes, params, tvtp2, base, gs, r_annual, ctx, paths = _setup(sens, mats)
    grids = C.common_grids(ctx, params, paths, mats, sorted(set(strikes) | set(sstrikes)),
                           gs, r_annual)
    print(f"context ready in {time.time() - t0:.1f}s; {len(ctx['runs'])} runs; grids: "
          + ", ".join(f"{int(T)}h [{g[0]:.0f}, {g[1]:.0f}]" for T, g in grids.items()),
          flush=True)

    tasks = []
    for run in ctx["runs"]:
        ladder = run.family == "ladder"
        for T in mats:
            for K in (strikes if ladder else sstrikes):
                tasks.append(("pde", run.name, T, K, ladder, None))
            if ladder and not args.no_mc:
                tasks.append(("mc", run.name, T, tuple(strikes), args.mc_paths, args.mc_dt,
                              args.mc_seed))
    conv_T = 72.0
    if not args.quick and conv_T in mats:
        base_steps = gs.n_steps(conv_T)
        for name in ("R1", "R3"):
            for nts in (2 * base_steps, 5 * base_steps, 10 * base_steps):
                tasks.append(("pde", name, conv_T, 3000.0, False, nts))
            if not args.no_mc:
                for dt in (0.25, 0.01):
                    tasks.append(("mc", name, conv_T, tuple(strikes), args.mc_paths, dt,
                                  args.mc_seed))
    print(f"{len(tasks)} tasks on {args.workers} worker(s)", flush=True)
    # long tasks first for load balance
    tasks.sort(key=lambda x: -(x[2] * (10 if x[0] == "mc" else 1)))
    results = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                      initargs=(sens, mats, grids)) as pool:
        for i, res in enumerate(pool.imap_unordered(_task, tasks), 1):
            results.append(res)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done ({time.time() - t0:.0f}s)", flush=True)

    pde_rows = [r[1] for r in results if r[0] == "pde" and r[2] is None]
    conv_rows = [dict(r[1], n_time_steps_override=r[2]) for r in results
                 if r[0] == "pde" and r[2] is not None]
    conv_rows += [dict(x, n_time_steps_override=None) for x in pde_rows
                  if x["run"] in ("R1", "R3") and x["maturity_h"] == 72.0
                  and x["strike"] == 3000.0 and conv_rows]
    mc_rows = [x for r in results if r[0] == "mc" and abs(r[2] - args.mc_dt) < 1e-12 for x in r[1]]
    mc_conv = [x for r in results if r[0] == "mc" and abs(r[2] - args.mc_dt) >= 1e-12 for x in r[1]]

    pde = pd.DataFrame(pde_rows).sort_values(["family", "run", "maturity_h", "strike"])
    ladder = pde[pde["family"] == "ladder"].copy()
    if mc_rows:
        mc = pd.DataFrame(mc_rows)
        ladder = ladder.merge(mc, on=["run", "maturity_h", "strike"], how="left")
        ladder["call_z"] = (ladder["call_mc"] - ladder["call_pde"]) / ladder["call_mc_se"]
        ladder["put_z"] = (ladder["put_mc"] - ladder["put_pde"]) / ladder["put_mc_se"]
        ladder["call_mc_minus_pde_pct"] = 100 * (ladder["call_mc"] / ladder["call_pde"] - 1)
        ladder["call_z_mean_matched"] = ((ladder["call_mc_mean_matched"] - ladder["call_pde"])
                                         / ladder["call_mc_se"])
        ladder["put_z_mean_matched"] = ((ladder["put_mc_mean_matched"] - ladder["put_pde"])
                                        / ladder["put_mc_se"])
    ladder = ladder.sort_values(["run", "maturity_h", "strike"])
    ladder.to_csv(out / "comparison_runs.csv", index=False)

    tcols = ["run", "maturity_h", "forward_T", "residual_sd_T", "p_stress_at_expiry",
             "mean_p_stress", "mean_q01_per_hour", "mean_q10_per_hour", "mean_p01_one_hour",
             "mean_p10_one_hour", "min_s", "max_s", "n_nodes_s_ge_0.95", "expected_transitions"]
    extra = [c for c in ("mc_p_stress_T", "mc_p_stress_T_se", "ode_p_stress_T_mc_grid",
                         "mc_mean_residual_T", "mc_mean_residual_T_se", "ode_mean_residual_T",
                         "mc_residual_sd_T", "ode_residual_sd_T_mc_grid") if c in ladder.columns]
    tstats = ladder[tcols + extra].drop_duplicates(["run", "maturity_h"]).sort_values(
        ["maturity_h", "run"])
    tstats.to_csv(out / "comparison_transition_stats.csv", index=False)

    piv = ladder.set_index(["run", "maturity_h", "strike"])
    drows = []
    for (a, b, label) in (("R1", "R0", "sample effect (R1-R0)"),
                          ("R2", "R1", "identity check (R2-R1), must be 0"),
                          ("R3", "R1", "TOTAL RAMP EFFECT (R3-R1)"),
                          ("R4", "R1", "slope part (R4-R1)"),
                          ("R3", "R4", "intercept compensation (R3-R4)")):
        for T in mats:
            for K in strikes:
                x, y = piv.loc[(a, T, K)], piv.loc[(b, T, K)]
                drows.append({"comparison": label, "maturity_h": T, "strike": K,
                              "d_call_TRY_MWh": x["call_pde"] - y["call_pde"],
                              "d_call_pct": _pct(x["call_pde"], y["call_pde"]),
                              "d_put_TRY_MWh": x["put_pde"] - y["put_pde"],
                              "d_put_pct": _pct(x["put_pde"], y["put_pde"]),
                              "d_p_stress_T": x["p_stress_at_expiry"] - y["p_stress_at_expiry"],
                              "d_mean_p_stress": x["mean_p_stress"] - y["mean_p_stress"],
                              "d_residual_sd_T": x["residual_sd_T"] - y["residual_sd_T"],
                              "d_expected_transitions": x["expected_transitions"] - y["expected_transitions"]})
    diffs = pd.DataFrame(drows)
    diffs.to_csv(out / "comparison_differences.csv", index=False)

    sens_df = pd.DataFrame()
    if sens:
        sp = pde[pde["family"] != "ladder"].copy()
        base_eff = {(T, K): piv.loc[("R3", T, K)]["call_pde"] - piv.loc[("R1", T, K)]["call_pde"]
                    for T in mats for K in sstrikes}
        srows = []
        for fam, g in sp.groupby("family"):
            gi = g.set_index(["run", "maturity_h", "strike"])
            names = sorted(g["run"].unique())
            n1 = [n for n in names if n.endswith("_1D")]
            n2 = [n for n in names if n.endswith("_2D")][0]
            for T in mats:
                for K in sstrikes:
                    v2 = gi.loc[(n2, T, K)]
                    v1 = gi.loc[(n1[0], T, K)] if n1 else piv.loc[("R1", T, K)]
                    srows.append({
                        "sensitivity": fam, "run_1d": n1[0] if n1 else "R1 (unchanged)",
                        "run_2d": n2, "maturity_h": T, "strike": K,
                        "call_1d": v1["call_pde"], "call_2d": v2["call_pde"],
                        "ramp_effect_TRY_MWh": v2["call_pde"] - v1["call_pde"],
                        "ramp_effect_pct": _pct(v2["call_pde"], v1["call_pde"]),
                        "base_ramp_effect_TRY_MWh": base_eff[(T, K)],
                        "shift_of_1d_vs_R1": v1["call_pde"] - piv.loc[("R1", T, K)]["call_pde"],
                        "shift_of_2d_vs_R3": v2["call_pde"] - piv.loc[("R3", T, K)]["call_pde"],
                        "p_stress_T_1d": v1["p_stress_at_expiry"], "p_stress_T_2d": v2["p_stress_at_expiry"],
                        "residual_sd_T_1d": v1["residual_sd_T"], "residual_sd_T_2d": v2["residual_sd_T"],
                        "max_s_2d": v2["max_s"]})
        sens_df = pd.DataFrame(srows)
        sens_df.to_csv(out / "sensitivity_ramp_effect.csv", index=False)
        sp.to_csv(out / "sensitivity_runs.csv", index=False)

    conv = pd.DataFrame(conv_rows)
    if len(conv):
        conv = conv[["run", "maturity_h", "strike", "n_time_steps", "call_pde",
                     "residual_sd_T", "p_stress_at_expiry"]].sort_values(["run", "n_time_steps"])
        conv.to_csv(out / "convergence_pde_time_grid.csv", index=False)
    mcc = pd.DataFrame(mc_conv + [x for x in mc_rows if x["run"] in ("R1", "R3")
                                  and x["maturity_h"] == 72.0])
    if len(mcc):
        mcc.to_csv(out / "convergence_mc_dt.csv", index=False)

    # ---- manifest ------------------------------------------------------------
    runs_meta = [{"name": r.name, "family": r.family, "tvtp": "2D" if r.is_2d else "1D",
                  "path": r.path_key, "description": r.description,
                  "coefficients": r.coefficients_dict(),
                  "ramp_scaler": (r.ramp_scaler.as_dict() if r.ramp_scaler is not None else None)}
                 for r in ctx["runs"]]
    manifest = {
        "label": EXPERIMENTAL_LABEL, "status": tvtp2.status,
        "held_fixed": {
            "forward_curve_sha256": C.curve_hash(base), "curve_mode": base.curve.mode,
            "kappa_per_hour": base.spec.kappa_per_hour, "sigma_y": base.spec.sigma_y.tolist(),
            "scale_P": base.spec.scale_P, "regime_means": base.spec.regime_means.tolist(),
            "drift_shift_a": base.spec.drift_shift_per_hour.tolist(), "x0": base.x0,
            "pi0": base.pi_filtered.tolist(), "r_annual": r_annual,
            "valuation_utc": params.valuation_utc.isoformat(),
            "space_grid_per_maturity": {str(int(T)): {"x_min": g[0], "x_max": g[1],
                                                      "n_nodes": gs.n_space_nodes}
                                        for T, g in grids.items()},
            "time_steps_per_maturity": {str(int(T)): gs.n_steps(T) for T in mats},
            "ode_max_substep_hours": 0.25,
            "mc": {"n_paths": args.mc_paths, "dt_hours": args.mc_dt, "seed": args.mc_seed,
                   "common_random_numbers_across_runs": True},
            "transition_premium": "ASSUMED zero (q^Q = q^P)"},
        "files": {"hash_convention": "sha256 of the content, CRLF normalized to LF",
                  "tvtp2_yaml": TVTP2_YAML, "tvtp2_yaml_sha256": sha256_file(TVTP2_YAML),
                  "base_yaml": BASE_YAML, "base_yaml_sha256": sha256_file(BASE_YAML),
                  "history": HISTORY, "history_sha256": sha256_file(HISTORY)},
        "path_variants": {k: {"description": v.description, "mode": v.spec.mode,
                              "offset": v.spec.offset, "initial_hours": v.spec.initial_hours,
                              "lag_hours": v.builder.lag}
                          for k, v in ctx["variants"].items()},
        "runs": runs_meta,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "elapsed_seconds": time.time() - t0,
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    (out / "comparison_report.md").write_text(
        render_report(ladder, tstats, diffs, sens_df, conv, mcc, manifest, mats, strikes),
        encoding="utf-8")
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return 0


def _t(df: pd.DataFrame, cols, fmt=None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(f.format(v) if f and isinstance(v, (int, float, np.floating)) and np.isfinite(v)
                         else ("—" if isinstance(v, float) and not np.isfinite(v) else str(v)))
        body.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(body) + "\n"


def render_report(ladder, tstats, diffs, sens_df, conv, mcc, man, mats, strikes) -> str:
    hf = man["held_fixed"]
    L = ["# 1D vs 2D TVTP comparison (EXPERIMENTAL)", "",
         f"**Label:** {man['label']} (`{man['status']}`).", "",
         "> Prices are conditional on the stated covariate path. The ramp covariate is a "
         "reconstruction; these numbers are NOT a reproduction of M9 and do NOT replace the "
         "accepted single-covariate results (production R0 = paper Table 3 configuration).", "",
         "## Held fixed across every run", "",
         f"- F(t): one curve object, sha256 `{hf['forward_curve_sha256'][:16]}…` ({hf['curve_mode']})",
         f"- kappa = {hf['kappa_per_hour']}/h, sigma_y = {hf['sigma_y']}, scale_P = {hf['scale_P']}, "
         f"m = {hf['regime_means']}, a = {hf['drift_shift_a']}, x0 = {hf['x0']}",
         f"- pi0 = {hf['pi0']}, r = {hf['r_annual']}, valuation {hf['valuation_utc']}",
         "- space grid per maturity (union over all runs and strikes, "
         f"{list(hf['space_grid_per_maturity'].values())[0]['n_nodes']} nodes): "
         + ", ".join(f"{k} h [{v['x_min']:.1f}, {v['x_max']:.1f}]"
                     for k, v in hf["space_grid_per_maturity"].items()),
         "- time steps: " + ", ".join(f"{k} h → {v}" for k, v in hf["time_steps_per_maturity"].items())
         + f"; ODE sub-step ≤ {hf['ode_max_substep_hours']} h",
         f"- Monte Carlo: {hf['mc']['n_paths']} paths, dt = {hf['mc']['dt_hours']} h, seed "
         f"{hf['mc']['seed']} (common random numbers across runs)",
         f"- transition premium: {hf['transition_premium']}", "",
         "## Runs", "", "| run | TVTP | path | α01 | α10 | h01 | h10 | description |",
         "|---|---|---|---|---|---|---|---|"]
    for r in man["runs"]:
        c = r["coefficients"]
        L.append(f"| {r['name']} | {r['tvtp']} | {r['path']} | {c['alpha01']:.6f} | {c['alpha10']:.6f} | "
                 f"{c['h01']:.6f} | {c['h10']:.6f} | {r['description']} |")
    L += ["", "## Ladder: PDE call / put, residual sd, stress probability", ""]
    cols = ["run", "maturity_h", "strike", "call_pde", "put_pde", "parity_error",
            "residual_sd_T", "p_stress_at_expiry", "mean_p_stress"]
    fmt = {"call_pde": "{:.4f}", "put_pde": "{:.4f}", "parity_error": "{:.1e}",
           "residual_sd_T": "{:.2f}", "p_stress_at_expiry": "{:.4f}", "mean_p_stress": "{:.4f}",
           "maturity_h": "{:.0f}", "strike": "{:.0f}"}
    L.append(_t(ladder, cols, fmt))
    if "call_mc" in ladder.columns:
        L += ["## PDE vs Monte Carlo on the same covariate path", "",
              "MC standard errors are reported; z = (MC − PDE)/SE. The paper documents a "
              "systematic MC shortfall of 0.6–3.1 % from time discretization of the regime path; "
              "the 1D and 2D columns show whether the ramp changes it.", ""]
        cols = ["run", "maturity_h", "strike", "call_pde", "call_mc", "call_mc_se", "call_z",
                "call_mc_minus_pde_pct", "put_pde", "put_mc", "put_mc_se", "put_z",
                "call_z_mean_matched", "put_z_mean_matched"]
        fmt2 = dict(fmt, call_mc="{:.4f}", call_mc_se="{:.4f}", call_z="{:+.2f}",
                    call_mc_minus_pde_pct="{:+.2f}", put_mc="{:.4f}", put_mc_se="{:.4f}",
                    put_z="{:+.2f}", call_z_mean_matched="{:+.2f}", put_z_mean_matched="{:+.2f}")
        L.append(_t(ladder, cols, fmt2))
        L += ["`*_z_mean_matched`: the same paths shifted so that the sample mean of X_T equals "
              "the ODE mu_X(T) (first-moment control variate; z uses the raw SE, so it is "
              "approximate). A raw z far from 0 with a small mean-matched z means the gap is the "
              "Monte Carlo error of the MEAN, shared by all strikes of one run (common random "
              "numbers), not a PDE bias.", ""]
    L += ["## Transition statistics over [0, T] (PDE time grid)", ""]
    cols = [c for c in ["run", "maturity_h", "p_stress_at_expiry", "mean_p_stress",
                        "mean_q01_per_hour", "mean_q10_per_hour", "mean_p01_one_hour",
                        "mean_p10_one_hour", "min_s", "max_s", "n_nodes_s_ge_0.95",
                        "expected_transitions", "mc_p_stress_T", "mc_p_stress_T_se",
                        "ode_p_stress_T_mc_grid", "mc_mean_residual_T", "mc_mean_residual_T_se",
                        "ode_mean_residual_T"] if c in tstats.columns]
    fmt3 = {c: "{:.4f}" for c in cols if c not in ("run",)}
    fmt3.update({"maturity_h": "{:.0f}", "n_nodes_s_ge_0.95": "{:.0f}",
                 "expected_transitions": "{:.2f}", "mc_mean_residual_T": "{:.2f}",
                 "mc_mean_residual_T_se": "{:.2f}", "ode_mean_residual_T": "{:.2f}"})
    L.append(_t(tstats, cols, fmt3))
    L += ["## Differences (the ramp effect kept separate)", ""]
    cols = ["comparison", "maturity_h", "strike", "d_call_TRY_MWh", "d_call_pct",
            "d_put_TRY_MWh", "d_put_pct", "d_p_stress_T", "d_mean_p_stress", "d_residual_sd_T"]
    fmt4 = {"maturity_h": "{:.0f}", "strike": "{:.0f}", "d_call_TRY_MWh": "{:+.4f}",
            "d_call_pct": "{:+.3f}", "d_put_TRY_MWh": "{:+.4f}", "d_put_pct": "{:+.3f}",
            "d_p_stress_T": "{:+.4f}", "d_mean_p_stress": "{:+.4f}", "d_residual_sd_T": "{:+.3f}"}
    L.append(_t(diffs, cols, fmt4))
    if len(sens_df):
        L += ["## One-change sensitivities (calls): ramp effect 2D − 1D under each change", ""]
        cols = ["sensitivity", "maturity_h", "strike", "call_1d", "call_2d", "ramp_effect_TRY_MWh",
                "ramp_effect_pct", "base_ramp_effect_TRY_MWh", "shift_of_1d_vs_R1",
                "shift_of_2d_vs_R3", "max_s_2d"]
        fmt5 = {"maturity_h": "{:.0f}", "strike": "{:.0f}", "call_1d": "{:.4f}", "call_2d": "{:.4f}",
                "ramp_effect_TRY_MWh": "{:+.4f}", "ramp_effect_pct": "{:+.3f}",
                "base_ramp_effect_TRY_MWh": "{:+.4f}", "shift_of_1d_vs_R1": "{:+.4f}",
                "shift_of_2d_vs_R3": "{:+.4f}", "max_s_2d": "{:.4f}"}
        L.append(_t(sens_df, cols, fmt5))
    if len(conv):
        L += ["## Time-grid convergence of the PDE (72 h, K = 3000)", "",
              _t(conv, ["run", "n_time_steps", "call_pde", "residual_sd_T", "p_stress_at_expiry"],
                 {"n_time_steps": "{:.0f}", "call_pde": "{:.4f}", "residual_sd_T": "{:.3f}",
                  "p_stress_at_expiry": "{:.4f}"})]
    if len(mcc):
        k3 = mcc[mcc["strike"] == 3000.0].sort_values(["run", "mc_dt_hours"])
        L += ["## Monte Carlo step convergence (72 h, K = 3000)", "",
              _t(k3, ["run", "mc_dt_hours", "call_mc", "call_mc_se", "mc_p_stress_T",
                      "ode_p_stress_T_mc_grid"],
                 {"mc_dt_hours": "{:.2f}", "call_mc": "{:.4f}", "call_mc_se": "{:.4f}",
                  "mc_p_stress_T": "{:.4f}", "ode_p_stress_T_mc_grid": "{:.4f}"})]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
