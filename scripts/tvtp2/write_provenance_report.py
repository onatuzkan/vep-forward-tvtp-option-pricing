"""Compose the provenance report (köken raporu) of the EXPERIMENTAL 2D TVTP.

    python scripts/tvtp2/write_provenance_report.py

Reads only generated artefacts (frozen 2D YAML, derivation audit, comparison
outputs) and the accepted single-covariate files (hashed, never written), and
writes outputs/tvtp2_experimental/PROVENANCE_REPORT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.tvtp2 import sha256_file                      # noqa: E402

OUT = REPO_ROOT / "outputs" / "tvtp2_experimental"
YAML2 = REPO_ROOT / "inputs" / "historical" / "tvtp2_frozen_parameters.yaml"
AUDIT = OUT / "derivation" / "tvtp2_derivation_audit.json"
CMP = OUT / "comparison"
PROTECTED = ["inputs/historical/m2_frozen_parameters.yaml",
             "inputs/historical/rd_standardized.csv",
             "inputs/historical/archive/calibration_bundle/transition_coefficients.csv",
             "outputs/market_calibration_final/calibration_result.json",
             "outputs/market_calibration_final/calibrated_config.yaml",
             "outputs/market_calibration_final/parameter_identification.json",
             "outputs/market_calibration_final/model_limitations.md",
             "outputs/scenario_sweep/rd_scenario_sweep.csv",
             "paper/figures/pde_mc.json"]

EVIDENCE = [
    ("RD_lag1 slopes γ01, γ10", "M9 transition_coefficients.csv, regime-label swapped; production-yaml values (10 dp)", "D (file)"),
    ("Ramp slopes h01, h10", "M9 transition_coefficients.csv row RD_Ramp_1h_lag1, cross-mapped (h01 ← raw gamma10, h10 ← raw gamma01), exact doubles", "D (file)"),
    ("Joint estimation of both slopes", "AME/slope identical for both covariates in each equation (|Δ| ~ 1e-15)", "Ç (strong inference)"),
    ("Ramp definition (direction, lag, RD vs z)", "not found anywhere; reconstructed as dz_t = z_t − z_(t−1h) on the complete UTC grid, lag 1 h", "Y (reconstruction) — original B"),
    ("Ramp standardization (window, ddof)", "not found; train-only mean/std on W9 (≤ 2024-12-31 20:00 UTC), ddof = 1", "Y (reconstruction) — original B"),
    ("Ramp scale", "E[1/p10] = 8.582 with the standardized ramp vs M9 8.585; raw dz gives 7.68", "Ç (consistency, not identity)"),
    ("Calendar-time lag convention", "rd_lag1_standardized.csv carries z(02:00) at 2016-03-27 03:00 (calendar lag, not row order)", "Ç (for RD_lag1; supports, does not prove, the ramp convention)"),
    ("z standardization", "TRY window ≤ 2022-12-31 20:00 UTC (61 361 rows), reconstructed in markov_adapter, ddof = 1 assumed", "Y"),
    ("M9 training window W9", "first n_train = 78 905 rows (n_total = 87 665 = history rows)", "Ç"),
    ("Intercepts α01, α10", "moment matching to M9 mean durations on D = W9 (paired x_(t−1)); no standard error", "derived"),
    ("Duration targets 3.759199 / 7.559716 h", "run_summary diagnostics (value D, sample B)", "D / B"),
    ("Transition premium η01 = η10 = 0", "assumption; forward quotes cannot identify it", "assumed"),
]


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _embeddability_section():
    """p01 + p10 >= 1 audit (count, share, max) of the standard scenario paths."""
    from pde_option_model.params_frozen import load_frozen_parameters, load_tvtp2_parameters
    from pde_option_model.scenarios import CovariatePathBuilder, ScenarioSpec
    from pde_option_model.tvtp2 import load_hourly_z_history
    base = load_frozen_parameters(REPO_ROOT / "inputs/historical/m2_frozen_parameters.yaml")
    tv = load_tvtp2_parameters(YAML2, base_params=base)
    z = load_hourly_z_history(REPO_ROOT / "inputs/historical/rd_standardized.csv")
    import pandas as _pd
    te = _pd.Timestamp("2022-12-31 20:00:00+00:00")
    rows = []
    specs = [ScenarioSpec(f"clim{o:+.1f}", "climatology", o) for o in
             (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)]
    specs += [ScenarioSpec("constant 0", "constant", 0.0),
              ScenarioSpec("clim, observed initial hours", "climatology", 0.0,
                           initial_hours="observed"),
              ScenarioSpec("clim-4.5 (outside the data)", "climatology", -4.5)]
    for lag in (1.0, 0.0, 2.0):
        b = CovariatePathBuilder(z, te, lag, tv.ramp_scaler)
        for sp in (specs if lag == 1.0 else specs[4:5]):
            path = b.build(sp, base.valuation_utc, horizon_hours=720.0)
            r = path.embeddability(tv.coefficients)
            rows.append((sp.name + ("" if lag == 1.0 else f", lag {lag:g} h"), r))
    L = ["## 5b. Scenario-path embeddability audit (720 h, hour labels used by the solver)", "",
         "| path | hours | s >= 1 | share | max s | at | s >= 0.95 |", "|---|---|---|---|---|---|---|"]
    for nm, r in rows:
        L.append(f"| {nm} | {r['n_rows']} | {r['n_s_ge_1']} | {r['share_s_ge_1']:.4f} | "
                 f"{r['max_s']:.4f} | {r['argmax_label']} | {r['n_s_ge_0.95']} |")
    L += ["", "Any path with s >= 1 is REJECTED for pricing (no clipping); the `clim-4.5` "
          "row illustrates the rule with a scenario outside the observed z support.", ""]
    return L


def main() -> int:
    blob = yaml.safe_load(YAML2.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    man = json.loads((CMP / "run_manifest.json").read_text(encoding="utf-8")) \
        if (CMP / "run_manifest.json").exists() else None
    c = blob["tvtp2"]
    sc = blob["covariates"]["ramp"]["scaler"]
    smp = blob["derivation"]["sample"]
    L = ["# Provenance report — two-covariate TVTP (EXPERIMENTAL)", "",
         f"**Label on every output:** {blob['label']}  ",
         f"**Status:** `{blob['status']}`, `verified_reproduction_of_m9: {str(blob['verified_reproduction_of_m9']).lower()}`  ",
         f"**Mode:** `{blob['tvtp_mode']}` (default mode stays `rd_lag1_1d`)", "",
         "> The original construction of `RD_Ramp_1h_lag1` was not found. The ramp used here is a "
         "reconstruction, so this mode is experimental. It does not reproduce M9 and does not "
         "change any accepted paper or production result.", "",
         "## 1. Frozen parameter set", "",
         f"File `inputs/historical/tvtp2_frozen_parameters.yaml` (sha256 `{sha256_file(YAML2)}`).", "",
         "| parameter | value | provenance |", "|---|---|---|"]
    for k in ("alpha01", "gamma01", "h01", "alpha10", "gamma10", "h10"):
        L.append(f"| {k} | {c[k]!r} | {blob['provenance'].get('tvtp2.' + k, '')} |")
    L += ["", "## 2. Ramp covariate and sample", "",
          f"- definition: {blob['covariates']['ramp']['definition']}",
          f"- status: {blob['covariates']['ramp']['status']}",
          f"- scaler: window {sc['window_name']} (labels ≤ {sc['window_end_utc']}), first/last increment "
          f"{sc['first_increment_utc']} / {sc['last_increment_utc']}, n = {sc['n']}, "
          f"m_r = {sc['mean']!r}, s_r = {sc['std']!r}, ddof = {sc['ddof']}",
          f"- intercept sample D: {smp['first_valid_transition_utc']} → {smp['last_valid_transition_utc']}, "
          f"{smp['n_valid_transitions']} transitions; dropped {smp['n_dropped_sample_start']} "
          f"(sample start: {', '.join(smp['dropped_sample_start_labels'])}) and "
          f"{smp['n_dropped_missing_observation']} (gap: {', '.join(smp['dropped_missing_observation_labels'])})",
          f"- history file `{blob['covariates']['history_file']}` sha256 `{blob['covariates']['history_sha256']}`",
          f"- base parameter file `{blob['base_parameters']['file']}` sha256 `{blob['base_parameters']['sha256']}`",
          "", "## 3. Evidence levels", "",
          "D = verified in a file, Ç = inference from data, Y = reconstruction, B = unknown.", "",
          "| object | basis | level |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c_} |" for a, b, c_ in EVIDENCE]
    vm = audit["validation_metrics"]
    L += ["", "## 4. Derivation error measures (from the audit)", "",
          f"- root residuals: p01 {blob['derivation']['roots']['p01']['residual']:.1e}, "
          f"p10 {blob['derivation']['roots']['p10']['residual']:.1e} (Brent, xtol = rtol = 1e-12)",
          f"- forward-recursion stress occupancy {vm['forward_recursion_occupancy_stress']['mean_stress_probability']:.4f} "
          f"vs M9 {vm['forward_recursion_occupancy_stress']['reported_m9']:.6f} "
          f"({100 * vm['forward_recursion_occupancy_stress']['relative_error']:+.1f}%, not targeted)",
          f"- AME ratios {vm['ame_ratio']['p01']:.4f} / {vm['ame_ratio']['p10']:.4f} vs M9 "
          f"{vm['ame_ratio']['m9_p01']:.4f} / {vm['ame_ratio']['m9_p10']:.4f}",
          f"- E[1/p] {vm['mean_inverse_p']['p01']:.3f} / {vm['mean_inverse_p']['p10']:.3f} vs M9 "
          f"{vm['mean_inverse_p']['m9_mean_expected_duration_normal']:.3f} / "
          f"{vm['mean_inverse_p']['m9_mean_expected_duration_stress']:.3f}",
          f"- p01 duration target incompatible with M9's AME: bound "
          f"{vm['p01_duration_target_consistency']['upper_bound_E_p_1mp']:.5f} < "
          f"{vm['p01_duration_target_consistency']['m9_ame_ratio']:.5f} (also in 1D production)",
          f"- historical embeddability: s ≥ 1 in {vm['embeddability_historical_full_sample']['n_s_ge_1']} of "
          f"{vm['embeddability_historical_full_sample']['n_rows']} hours, max s "
          f"{vm['embeddability_historical_full_sample']['max_s']:.4f} at "
          f"{vm['embeddability_historical_full_sample']['argmax_label']}", ""]
    if man:
        hf = man["held_fixed"]
        L += ["## 5. Comparison run (what was held fixed)", "",
              f"- forward curve sha256 `{hf['forward_curve_sha256']}`; kappa {hf['kappa_per_hour']}; "
              f"sigma_y {hf['sigma_y']}; pi0 {hf['pi0']}; r {hf['r_annual']}",
              f"- grids: {json.dumps(hf['space_grid_per_maturity'])}",
              f"- MC: {json.dumps(hf['mc'])}",
              f"- runs: {', '.join(r['name'] for r in man['runs'])}", ""]
        d = pd.read_csv(CMP / "comparison_differences.csv")
        tot = d[d["comparison"].str.startswith("TOTAL")]
        L += ["Total ramp effect R3 − R1 (calls, TRY/MWh):", "",
              "| maturity h | " + " | ".join(f"K={int(k)}" for k in sorted(tot["strike"].unique())) + " |",
              "|---|" + "---|" * tot["strike"].nunique()]
        for T, g in tot.groupby("maturity_h"):
            L.append(f"| {int(T)} | " + " | ".join(f"{v:+.3f} ({p:+.2f}%)" for v, p in
                                                  zip(g.sort_values('strike')['d_call_TRY_MWh'],
                                                      g.sort_values('strike')['d_call_pct'])) + " |")
        L.append("")
    L += _embeddability_section()
    L += ["## 6. Accepted artefacts (read-only; hashes at report time)", "",
          "sha256 of the content with CRLF normalized to LF (identical on Windows "
          "autocrlf and Linux checkouts).", "",
          "| file | sha256 |", "|---|---|"]
    for f in PROTECTED:
        p = REPO_ROOT / f
        L.append(f"| `{f}` | `{sha256_file(p) if p.exists() else 'missing'}` |")
    L += ["", "## 7. Remaining data-provenance limitations", "",
          "1. The ramp construction (direction, lag, RD vs z, window, ddof) is unverified; the transferred "
          "slopes may be applied to a different variable than M9 used.",
          "2. The z standardization is a reconstruction on the TRY window; M9 may have standardized on its "
          "own (USD-run) window — sensitivity `z_scale_W9` quantifies it.",
          "3. The M9 intercepts are not exported; derived intercepts depend on the duration targets, whose "
          "sample is unknown, and the p01 target conflicts with M9's own AME ratio.",
          "4. M9 is a USD-price fit (`markov_usd_final`); its slopes are applied to a TRY pricing model.",
          "5. `pi_filtered` comes from the M2 shipped filter, not from a 2D filter at the valuation hour.",
          "6. The shipped p series (`pde_timeseries.parquet`) and `covariate_scaling.json` are absent, so the "
          "decisive regression test of the definition (logit p on candidate covariates) cannot be run.",
          "7. Prices are conditional on a deterministic covariate path; with random covariates the state "
          "space would grow and Jensen gaps would appear.",
          "", "Resolving 1, 2, 3 and 6 requires the original estimation output "
          "(`res-markov/outputs/markov_usd_final`)."]
    (OUT / "PROVENANCE_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'PROVENANCE_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
