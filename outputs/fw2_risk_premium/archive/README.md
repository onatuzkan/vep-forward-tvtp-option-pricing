# Archive -- FW2 outputs computed under the silent z = 0 fallback

## Why these are here

The four files in this folder were produced by
`scripts/fw2/sensitivity_sweep.py` BEFORE FW12b (2026-09-27) fixed
the scenario-plumbing bug diagnosed in
`outputs/fw12_convergence/settings_diff_FW2_vs_production.md`.  The
original sweep called `price_forward_centered` without a climatology
`z_lagged_fn` and silently hit `z(t-1) = 0` for every t; this priced
the constant-transition limit of the TVTP model, not the shipped
production model.

## What the numbers in these files describe

* Baseline ATM K = 3000, T = 72 h call **= 179.65 TRY/MWh** at the
  constant-transition limit (z = 0), NOT the production value 166.75.
* Q1 (a_stress) percent effects: max 3.9 % on the wrong base.
* Q2 (eta_ij) percent effects: max +/- 27 % on the wrong base.
* Joint corner: max -35 % on the wrong base.
* Production-setting recommendation copy in
  `production_setting_recommendation.silent_z_zero.md` cites these
  wrong-base numbers.

## Not valid for the manuscript

`outputs/fw2_risk_premium/sensitivity_grid.csv`,
`sensitivity_summary.csv`, `sensitivity_summary.md`, and
`production_setting_recommendation.md` at the parent level are the
FW12b-regenerated replacements computed with the production
climatology z path at 1201 spatial nodes.  The manuscript's Section
5.5 / Section 7 uncertainty envelope should cite the REPLACEMENT
files, not the ones in this archive.

## Kept for the audit trail

These files are not deleted so that any reader can trace how the
old numbers arose and what changed.  See also
`outputs/fw12_convergence/README.md` (FW12) and this branch's
FW12b task report.
