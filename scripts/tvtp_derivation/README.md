# TVTP parameter derivation

`derive_tvtp_parameters.py` back-solves the yaml-convention `alpha01` and
`alpha10` intercepts of the two-regime TVTP model from the M9 fit's reported
occupancy / mean-duration diagnostics, and applies the raw-M9 → yaml regime
label swap.  It is the reference script cited in the methodology appendix
`docs/tvtp_derivation_methodology.md`.

Inputs used:
* `inputs/historical/rd_standardized.csv` (87 665 hourly `z` values)
* Numerical M9 diagnostics from
  `inputs/historical/archive/calibration_bundle/parameter_estimates.csv`,
  `.../transition_coefficients.csv`, and
  `.../metadata/model_parameters_and_ou_mapping.json` (raw M9 handoff bundle).

Outputs (written under `outputs/market_calibration_final/archive/tvtp_integration_draft/`
when the script was last run):
* `derived_tvtp_parameters.yaml` — the draft yaml block copied into
  `inputs/historical/m2_frozen_parameters.yaml` at commit `59955ee`.
* `derivation_summary.json` — machine-readable derivation trail.

Re-run this script only if the M9 diagnostics or the historical `z` series
change; otherwise the yaml is the source of truth.
