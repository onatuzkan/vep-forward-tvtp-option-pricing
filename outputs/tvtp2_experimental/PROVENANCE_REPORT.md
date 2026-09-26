# Provenance report — two-covariate TVTP (EXPERIMENTAL)

**Label on every output:** M9-transferred slopes + reconstructed ramp + derived intercepts, zero transition premium  
**Status:** `experimental_reconstructed`, `verified_reproduction_of_m9: false`  
**Mode:** `rd_ramp_2d_experimental` (default mode stays `rd_lag1_1d`)

> The original construction of `RD_Ramp_1h_lag1` was not found. The ramp used here is a reconstruction, so this mode is experimental. It does not reproduce M9 and does not change any accepted paper or production result.

## 1. Frozen parameter set

File `inputs/historical/tvtp2_frozen_parameters.yaml` (sha256 `64f7df73fc28558692d4347907323088ac1b24f4d53f988a9b354c644cd39717`).

| parameter | value | provenance |
|---|---|---|
| alpha01 | -1.0490993394164196 | derived: brentq root of mean_D logistic(a + gamma01 z_(t-1) + h01 r_(t-1)) = 1/d_n on D = W9 (paired x_(t-1)); moment matching, NOT an MLE estimate, no standard error |
| gamma01 | -0.5837780825 | estimated (transferred): M9 raw gamma10 of RD_lag1, regime-label swapped; value exactly as in the production yaml (10 dp) |
| h01 | -0.06939531482057137 | estimated (transferred): M9 raw gamma10 of RD_Ramp_1h_lag1 (full precision), regime-label swapped; applied to a RECONSTRUCTED ramp covariate |
| alpha10 | -1.955839031679882 | derived: brentq root of mean_D logistic(a + gamma10 z_(t-1) + h10 r_(t-1)) = 1/d_s on D = W9; moment matching, NOT an MLE estimate |
| gamma10 | 0.0776983936 | estimated (transferred): M9 raw gamma01 of RD_lag1, regime-label swapped; value exactly as in the production yaml (10 dp) |
| h10 | 0.4055443473030015 | estimated (transferred): M9 raw gamma01 of RD_Ramp_1h_lag1 (full precision), regime-label swapped; applied to a RECONSTRUCTED ramp covariate |

## 2. Ramp covariate and sample

- definition: r_t = (dz_t - m_r) / s_r with dz_t = z_t - z_(t-1h) computed on the complete hourly UTC grid (dz_t undefined unless both labels t and t-1h are observed); transition S_(t-1) -> S_t uses x_(t-1) = (z_(t-1h), r_(t-1h))
- status: ASSUMED/RECONSTRUCTED: the original M9 construction of RD_Ramp_1h_lag1 (difference direction, lag, raw RD vs z, standardization window, ddof) was not found in the repository, its git history, the SSRN manuscript or the calibration bundle; z.diff() on the complete hourly grid is a reconstruction and M9 is NOT claimed to be reproduced
- scaler: window W9 (labels ≤ 2024-12-31T20:00:00+00:00), first/last increment 2016-01-01T02:00:00+00:00 / 2024-12-31T20:00:00+00:00, n = 78903, m_r = 3.417142268311894e-05, s_r = 0.271450116132395, ddof = 1
- intercept sample D: 2016-01-01T03:00:00+00:00 → 2024-12-31T20:00:00+00:00, 78902 transitions; dropped 2 (sample start: 2016-01-01T01:00:00+00:00, 2016-01-01T02:00:00+00:00) and 4 (gap: 2016-03-27T01:00:00+00:00, 2016-03-27T02:00:00+00:00, 2016-03-27T03:00:00+00:00, 2016-03-27T04:00:00+00:00)
- history file `inputs/historical/rd_standardized.csv` sha256 `bea51c98c84ea4a6fe62111ae8993410aee8a430434d9601b0c292cbb58c6e04`
- base parameter file `inputs/historical/m2_frozen_parameters.yaml` sha256 `b80fa69d1130e4d7ab05d9830b53deb2d59920381b8c57708ac4e22cc9b35ab2`

## 3. Evidence levels

D = verified in a file, Ç = inference from data, Y = reconstruction, B = unknown.

| object | basis | level |
|---|---|---|
| RD_lag1 slopes γ01, γ10 | M9 transition_coefficients.csv, regime-label swapped; production-yaml values (10 dp) | D (file) |
| Ramp slopes h01, h10 | M9 transition_coefficients.csv row RD_Ramp_1h_lag1, cross-mapped (h01 ← raw gamma10, h10 ← raw gamma01), exact doubles | D (file) |
| Joint estimation of both slopes | AME/slope identical for both covariates in each equation (|Δ| ~ 1e-15) | Ç (strong inference) |
| Ramp definition (direction, lag, RD vs z) | not found anywhere; reconstructed as dz_t = z_t − z_(t−1h) on the complete UTC grid, lag 1 h | Y (reconstruction) — original B |
| Ramp standardization (window, ddof) | not found; train-only mean/std on W9 (≤ 2024-12-31 20:00 UTC), ddof = 1 | Y (reconstruction) — original B |
| Ramp scale | E[1/p10] = 8.582 with the standardized ramp vs M9 8.585; raw dz gives 7.68 | Ç (consistency, not identity) |
| Calendar-time lag convention | rd_lag1_standardized.csv carries z(02:00) at 2016-03-27 03:00 (calendar lag, not row order) | Ç (for RD_lag1; supports, does not prove, the ramp convention) |
| z standardization | TRY window ≤ 2022-12-31 20:00 UTC (61 361 rows), reconstructed in markov_adapter, ddof = 1 assumed | Y |
| M9 training window W9 | first n_train = 78 905 rows (n_total = 87 665 = history rows) | Ç |
| Intercepts α01, α10 | moment matching to M9 mean durations on D = W9 (paired x_(t−1)); no standard error | derived |
| Duration targets 3.759199 / 7.559716 h | run_summary diagnostics (value D, sample B) | D / B |
| Transition premium η01 = η10 = 0 | assumption; forward quotes cannot identify it | assumed |

## 4. Derivation error measures (from the audit)

- root residuals: p01 -1.3e-15, p10 0.0e+00 (Brent, xtol = rtol = 1e-12)
- forward-recursion stress occupancy 0.6352 vs M9 0.674578 (-5.8%, not targeted)
- AME ratios 0.1823 / 0.1114 vs M9 0.1963 / 0.1112
- E[1/p] 4.690 / 8.582 vs M9 4.135 / 8.585
- p01 duration target incompatible with M9's AME: bound 0.19525 < 0.19628 (also in 1D production)
- historical embeddability: s ≥ 1 in 0 of 87662 hours, max s 0.8408 at 2020-05-25T06:00:00+00:00

## 5. Comparison run (what was held fixed)

- forward curve sha256 `860f8decba68defed812e783d2e5028445145c6373715f6f0af0d9f1cfe0cde7`; kappa 0.078394; sigma_y [0.003534807, 0.0924066544]; pi0 [0.931977, 0.068023]; r 0.4
- grids: {"24": {"x_min": -4520.545417558372, "x_max": 4686.067189371408, "n_nodes": 1201}, "72": {"x_min": -4568.552579594436, "x_max": 4736.237895033397, "n_nodes": 1201}, "168": {"x_min": -4563.731592998991, "x_max": 4735.743995688066, "n_nodes": 1201}, "336": {"x_min": -4555.247628921624, "x_max": 4734.83243429374, "n_nodes": 1201}}
- MC: {"n_paths": 40000, "dt_hours": 0.05, "seed": 20260808, "common_random_numbers_across_runs": true}
- runs: R0, R1, R2, R3, R4, S_rampWT_2D, S_alphaFull_1D, S_alphaFull_2D, S_zW9_1D, S_zW9_2D, S_lag0_1D, S_lag0_2D, S_lag2_1D, S_lag2_2D, S_constant0_1D, S_constant0_2D, S_offm2_1D, S_offm2_2D, S_offp2_1D, S_offp2_2D, S_observed_1D, S_observed_2D

Total ramp effect R3 − R1 (calls, TRY/MWh):

| maturity h | K=2000 | K=2500 | K=3000 | K=3500 | K=4000 |
|---|---|---|---|---|---|
| 24 | -0.216 (-0.02%) | -1.090 (-0.23%) | -2.075 (-1.28%) | -0.667 (-1.84%) | -0.119 (-2.16%) |
| 72 | -0.167 (-0.02%) | -0.892 (-0.19%) | -1.668 (-1.01%) | -0.535 (-1.42%) | -0.089 (-1.53%) |
| 168 | -0.167 (-0.02%) | -0.892 (-0.19%) | -1.657 (-1.01%) | -0.527 (-1.43%) | -0.088 (-1.53%) |
| 336 | -0.167 (-0.02%) | -0.892 (-0.19%) | -1.637 (-1.02%) | -0.515 (-1.43%) | -0.085 (-1.53%) |

## 5b. Scenario-path embeddability audit (720 h, hour labels used by the solver)

| path | hours | s >= 1 | share | max s | at | s >= 0.95 |
|---|---|---|---|---|---|---|
| clim-2.0 | 721 | 0 | 0.0000 | 0.7897 | 2026-01-01T02:00:00+00:00 | 0 |
| clim-1.5 | 721 | 0 | 0.0000 | 0.7273 | 2026-01-01T02:00:00+00:00 | 0 |
| clim-1.0 | 721 | 0 | 0.0000 | 0.6603 | 2026-01-01T02:00:00+00:00 | 0 |
| clim-0.5 | 721 | 0 | 0.0000 | 0.5914 | 2026-01-01T02:00:00+00:00 | 0 |
| clim+0.0 | 721 | 0 | 0.0000 | 0.5237 | 2026-01-01T02:00:00+00:00 | 0 |
| clim+0.5 | 721 | 0 | 0.0000 | 0.4599 | 2026-01-01T02:00:00+00:00 | 0 |
| clim+1.0 | 721 | 0 | 0.0000 | 0.4193 | 2026-01-01T05:00:00+00:00 | 0 |
| clim+1.5 | 721 | 0 | 0.0000 | 0.3936 | 2026-01-01T05:00:00+00:00 | 0 |
| clim+2.0 | 721 | 0 | 0.0000 | 0.3747 | 2026-01-01T05:00:00+00:00 | 0 |
| constant 0 | 721 | 0 | 0.0000 | 0.3833 | 2025-12-31T19:00:00+00:00 | 0 |
| clim, observed initial hours | 721 | 0 | 0.0000 | 0.5237 | 2026-01-01T02:00:00+00:00 | 0 |
| clim-4.5 (outside the data) | 721 | 60 | 0.0832 | 1.0076 | 2026-01-01T05:00:00+00:00 | 240 |
| clim+0.0, lag 0 h | 721 | 0 | 0.0000 | 0.5237 | 2026-01-01T02:00:00+00:00 | 0 |
| clim+0.0, lag 2 h | 721 | 0 | 0.0000 | 0.5237 | 2026-01-01T02:00:00+00:00 | 0 |

Any path with s >= 1 is REJECTED for pricing (no clipping); the `clim-4.5` row illustrates the rule with a scenario outside the observed z support.

## 6. Accepted artefacts (read-only; hashes at report time)

sha256 of the content with CRLF normalized to LF (identical on Windows autocrlf and Linux checkouts).

| file | sha256 |
|---|---|
| `inputs/historical/m2_frozen_parameters.yaml` | `b80fa69d1130e4d7ab05d9830b53deb2d59920381b8c57708ac4e22cc9b35ab2` |
| `inputs/historical/rd_standardized.csv` | `bea51c98c84ea4a6fe62111ae8993410aee8a430434d9601b0c292cbb58c6e04` |
| `inputs/historical/archive/calibration_bundle/transition_coefficients.csv` | `930481e0a6009004e537d6ead505e2406bf2c0b34a60e68378238efc13b79536` |
| `outputs/market_calibration_final/calibration_result.json` | `881ca31195c9977dcb45d135bc2bb12c0f71d42c0d4d81b07d206a98c3fa5a68` |
| `outputs/market_calibration_final/calibrated_config.yaml` | `7ca61ac158d55ee10471f8f2bee611ee77ee177d42bf38264dcf1a9dfaa1f5d9` |
| `outputs/market_calibration_final/parameter_identification.json` | `25c6642e657cfb9675716810f9a11a1b5a07a993a43e24368acb4050fba06f68` |
| `outputs/market_calibration_final/model_limitations.md` | `c8ececcc6749185f5a5f86b74c0e3af2fa499b4b6fb006e31b53fecc1b4939bc` |
| `outputs/scenario_sweep/rd_scenario_sweep.csv` | `337ee81f0eb0ac439bb7327a087284d0471aaf9fefefb4aaadb79c95fc069311` |
| `paper/figures/pde_mc.json` | `ed3d28aff56d02e07aefcf2bf633b2070d60b3bc3458f502cca09e0e55962150` |

## 7. Remaining data-provenance limitations

1. The ramp construction (direction, lag, RD vs z, window, ddof) is unverified; the transferred slopes may be applied to a different variable than M9 used.
2. The z standardization is a reconstruction on the TRY window; M9 may have standardized on its own (USD-run) window — sensitivity `z_scale_W9` quantifies it.
3. The M9 intercepts are not exported; derived intercepts depend on the duration targets, whose sample is unknown, and the p01 target conflicts with M9's own AME ratio.
4. M9 is a USD-price fit (`markov_usd_final`); its slopes are applied to a TRY pricing model.
5. `pi_filtered` comes from the M2 shipped filter, not from a 2D filter at the valuation hour.
6. The shipped p series (`pde_timeseries.parquet`) and `covariate_scaling.json` are absent, so the decisive regression test of the definition (logit p on candidate covariates) cannot be run.
7. Prices are conditional on a deterministic covariate path; with random covariates the state space would grow and Jensen gaps would appear.

Resolving 1, 2, 3 and 6 requires the original estimation output (`res-markov/outputs/markov_usd_final`).
