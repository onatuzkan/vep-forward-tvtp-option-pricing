# FW9b §3 -- Object-matched parameter comparison (T8)

* `inherited_from` says which OBJECT the inherited value is
  the natural comparison for; the FW9 estimate is compared
  against the OBJECT-MATCHED inherited value.
* Boundary-flagged rows (phi_MS_AR) do not carry SE/t/CI --
  the Hessian-based estimator is not interpretable at the
  unit-root boundary; see FW9b §1 profile analysis.

| par | inherited from | inherited | FW9 est | SE | t | 95 % low | 95 % high | in FW9 95 %? | comment |
|---|---|---:|---:|---:|---:|---:|---:|:-:|---|
| sigma_normal | M9 CSV row (parameter_estimates.csv, sigma1 low-vol) | 0.00353481 | 0.00705029 | 6.24163e-05 | 56.3 | 0.00692795 | 0.00717262 | **NO** |  |
| sigma_stress | M9 CSV row (parameter_estimates.csv, sigma0 high-vol) | 0.0924067 | 0.241516 | 0.000882232 | 169 | 0.239787 | 0.243245 | **NO** |  |
| phi_MS_AR | M9 CSV row (parameter_estimates.csv, phi) | 0.999996 | 0.99999 | -- | -- | -- | -- | -- | Both fits estimated on the unit-root boundary; Hessian-based SE not interpretable. |
| phi_deseasonalized | yaml (v2 kappa refit; deseasonalized single-regime AR(1)) | 0.9246 | 0.983425 | 0.000731531 | 80.4 | 0.981991 | 0.984859 | **NO** | FW9 pipeline: hour-of-week + month-of-year additive seasonal dummies + OLS AR(1); the M9 original deseasonalisation pipeline is not shipped with the repo -- see preprocessing_audit.md. |
| mu_normal | M9 CSV row (mu1); yaml explicitly zeros mu_i (centering identity absorbs them) | -0.000678331 | -0.00058867 | 5.43042e-05 | 1.65 | -0.000695106 | -0.000482234 | yes |  |
| mu_stress | M9 CSV row (mu0); yaml explicitly zeros mu_i (centering identity absorbs them) | -0.00426608 | 0.000433363 | 0.00122047 | 3.85 | -0.00195876 | 0.00282549 | **NO** |  |
| alpha01 | yaml DERIVED (occupancy/duration root-finding); no MLE counterpart in the bundle | -1.01567 | -0.678121 | 0.0186415 | 18.1 | -0.714658 | -0.641584 | **NO** |  |
| alpha10 | yaml DERIVED (occupancy/duration root-finding); no MLE counterpart in the bundle | -1.89523 | -1.6511 | 0.0170424 | 14.3 | -1.68451 | -1.6177 | **NO** |  |
| gamma01 | M9 transition_coefficients.csv (yaml swap of raw gamma10, normal->stress direction) | -0.583778 | -0.44359 | 0.0158935 | 8.82 | -0.474742 | -0.412439 | **NO** |  |
| gamma10 | M9 transition_coefficients.csv (yaml swap of raw gamma01, stress->normal direction) | 0.0776984 | 0.175878 | 0.0153451 | 6.4 | 0.145801 | 0.205954 | **NO** |  |
