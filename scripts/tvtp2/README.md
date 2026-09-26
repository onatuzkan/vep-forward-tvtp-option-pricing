# Two-covariate TVTP scripts (EXPERIMENTAL, FW4)

| script | purpose | output |
|---|---|---|
| `../tvtp_derivation/derive_tvtp2_parameters.py` | ramp reconstruction, train-only scaler, intercept roots, audit; `--write` freezes the 2D YAML | `inputs/historical/tvtp2_frozen_parameters.yaml`, `outputs/tvtp2_experimental/derivation/` |
| `compare_tvtp_1d_2d.py` | controlled 1D vs 2D comparison: ladder R0–R4, one-change sensitivities, PDE vs Monte Carlo with standard errors, time-grid and MC-step convergence | `outputs/tvtp2_experimental/comparison/` |
| `write_provenance_report.py` | provenance report from the frozen YAML, the audit and the comparison | `outputs/tvtp2_experimental/PROVENANCE_REPORT.md` |

Order: derivation → comparison → provenance report. None of them writes into
`outputs/market_calibration_final`, `outputs/forward_centered_diagnostics`,
`outputs/scenario_sweep` or `inputs/historical/m2_frozen_parameters.yaml`.
Everything is labelled *"M9-transferred slopes + reconstructed ramp + derived
intercepts, zero transition premium"*; see `docs/tvtp2_methodology.md`.
