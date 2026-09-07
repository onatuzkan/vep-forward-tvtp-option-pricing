# Calibration bundle — raw M9 handoff

Verbatim contents of the original `calibration_files_FINAL.zip` (deleted from
the project root during the M9 integration follow-up commit). Kept here so the
raw handoff can be re-inspected without the zip file:

| file | size | purpose |
|---|---:|---|
| `parameter_estimates.csv` | 489 B | M0/M8/M9 sigma_eps, phi, loglik per model |
| `transition_coefficients.csv` | 242 B | M9 RD_lag1 and RD_Ramp_1h_lag1 gamma coefficients |
| `metadata/model_parameters_and_ou_mapping.json` | 6792 B | Fallback M2 physical measure, ou_conversion mappings, and the embedded run_summary block |

The values here are the raw M9 numbers (state 0 = high-vol, state 1 = low-vol)
BEFORE the yaml regime-label swap. The yaml-convention (index 0 = normal,
index 1 = stress) versions live in `inputs/historical/m2_frozen_parameters.yaml`;
the derivation of the missing `alpha01`/`alpha10` intercepts is documented in
`docs/tvtp_derivation_methodology.md`.

Do not overwrite these files. If a fresh calibration bundle arrives, place it
in a new subfolder (e.g. `inputs/historical/archive/calibration_bundle_v2/`)
so the audit trail stays intact.
