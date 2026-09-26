# 2D TVTP derivation audit (RD_lag1 + RD_Ramp_1h_lag1)

**Status:** `experimental_reconstructed` — M9-transferred slopes + reconstructed ramp + derived intercepts, zero transition premium.

> EXPERIMENTAL. The ramp covariate definition is a reconstruction; nothing here is a reproduction of M9 and the accepted paper results are unchanged.

## 1. Search for the original ramp definition

| source | result |
|---|---|
| repository working tree (code, docs, configs, outputs) | slope values and the covariate NAME only; no construction code |
| git history of the repository (all 27 commits, deleted files, team_share_2026_09_01.zip) | no ramp construction, no covariate_scaling.json, no prepared_meta.json |
| SSRN manuscript (ssrn-7472067, 43 pages) | states the ramp term is omitted in production and that joint vs separate estimation could not be determined; no definition |
| calibration bundle (transition_coefficients.csv, parameter_estimates.csv, metadata json) | slopes and AMEs of both covariates; no definition, no scaler |
| markov_adapter.py docstring | claims a logit(p) regression on 'lagged standardized RD and RD-ramp' recovers M9 to 4-5 digits; the code and pde_timeseries.parquet are absent, so the claim is not reproducible here |
| original estimation output directory (MarkovProject/res-markov/outputs/markov_usd_final) | not accessible from this repository |

## 2. Label mapping (raw M9 → production, one atomic swap)

| production | raw M9 | value |
|---|---|---|
| sigma_y_normal (0) | sigma1 | 0.0035348070 |
| sigma_y_stress (1) | sigma0 | 0.0924066544 |
| gamma01 (normal->stress, RD_lag1) | gamma10 RD_lag1 | -0.5837780825 |
| gamma10 (stress->normal, RD_lag1) | gamma01 RD_lag1 | 0.0776983936 |
| h01 (normal->stress, ramp) | gamma10 RD_Ramp_1h_lag1 | -0.0693953148 |
| h10 (stress->normal, ramp) | gamma01 RD_Ramp_1h_lag1 | 0.4055443473 |
| d_n mean normal duration (h) | mean_duration_state1 | 3.7591994835 |
| d_s mean stress duration (h) | mean_duration_state0 | 7.5597159458 |
| pi_stress occupancy | occupancy[0] | 0.6745780174 |

- AME and slope share their sign in every equation: **yes**
- AME/slope, p01 equation: 0.196280747683 (RD) vs 0.196280747683 (ramp), |diff| = 1.4e-15
- AME/slope, p10 equation: 0.111176975919 (RD) vs 0.111176975919 (ramp), |diff| = 9.7e-16
- Joint-estimation evidence: **yes** — AME/slope = E[Lambda'(eta)] is a property of ONE fitted p path; equal ratios for RD_lag1 and RD_Ramp_1h_lag1 (to ~1e-15) show both slopes come from the same TVTP-2 equation (joint estimation)
- RD slopes used are the production-yaml values (10 dp); rounding vs CSV: 4.1e-11, 3.0e-11

## 3. Hourly grid, windows and scalers

- History: 2016-01-01T01:00:00+00:00 → 2025-12-31T20:00:00+00:00, 87,665 rows on 87,668 grid labels; missing: 2016-03-27T00:00:00+00:00, 2016-03-27T01:00:00+00:00, 2016-03-27T02:00:00+00:00
- W9 = first n_train = 78905 rows of the history (M9 run_summary); n_total = 87665 equals the history row count

| window | end (UTC) | z rows | z mean | z sd | dz n | m_r | s_r |
|---|---|---|---|---|---|---|---|
| W_T | 2022-12-31T20:00:00+00:00 | 61,361 | 0.000029 | 0.999965 | 61,359 | 3.516e-05 | 0.266393 |
| W9 | 2024-12-31T20:00:00+00:00 | 78,905 | 0.092361 | 1.052863 | 78,903 | 3.417e-05 | 0.271450 |
| all | 2025-12-31T20:00:00+00:00 | 87,665 | 0.150536 | 1.078409 | 87,663 | 3.178e-05 | 0.274525 |

Ramp scaler used: **W9**. Ramp values are invariant to an affine re-scaling of z (max |Δr| = 2.6e-15).

## 4. Transition sample D and intercepts

- D: 2016-01-01T03:00:00+00:00 → 2024-12-31T20:00:00+00:00, 78,902 transitions of 78,908 grid labels
- dropped (sample start, rule: the first 2 hour(s) of the sample have no z_(t-1h) / z_(t-2h) pair, so r_(t-1) is undefined: dropped, never filled): 2 → 2016-01-01T01:00:00+00:00, 2016-01-01T02:00:00+00:00
- dropped (missing observation, rule: a transition is dropped when z_(t-1h) or z_(t-2h) is missing on the complete hourly grid; no interpolation, no zero ramp): 4 → 2016-03-27T01:00:00+00:00, 2016-03-27T02:00:00+00:00, 2016-03-27T03:00:00+00:00, 2016-03-27T04:00:00+00:00
- kept although the own label has no z (its x_(t-1) is observed): 2016-03-27T00:00:00+00:00

| root | alpha | target mean p | achieved | residual | iterations |
|---|---|---|---|---|---|
| p01_normal_to_stress | -1.0490993394 | 0.26601408 | 0.26601408 | -1.3e-15 | 13 |
| p10_stress_to_normal | -1.9558390317 | 0.13228010 | 0.13228010 | 0.0e+00 | 13 |

## 5. Variants (derived, not re-estimated)

| variant | α01 | α10 | n | occupancy (fwd rec.) | AME p01 | AME p10 | E[1/p01] | E[1/p10] | max s |
|---|---|---|---|---|---|---|---|---|---|
| 1D production (repo yaml; all rows, contemporaneous z) | -1.015666 | -1.895229 | 87,665 | 0.6500 | 0.1850 | 0.1143 | 4.539 | 7.629 | 0.8538 |
| 1D, D = W9 (x_(t-1) sample)  [R1] | -1.045554 | -1.890591 | 78,902 | 0.6445 | 0.1827 | 0.1147 | 4.646 | 7.598 | 0.8486 |
| 2D, D = W9, ramp scaler W9  [R3, primary] | -1.049099 | -1.955839 | 78,902 | 0.6352 | 0.1823 | 0.1114 | 4.690 | 8.582 | 0.8408 |
| 2D, D = W9, ramp scaler W_T | -1.049188 | -1.958235 | 78,902 | 0.6351 | 0.1823 | 0.1113 | 4.691 | 8.620 | 0.8407 |
| 2D, D = W9, ramp scaler all | -1.049047 | -1.954446 | 78,902 | 0.6352 | 0.1823 | 0.1115 | 4.689 | 8.560 | 0.8409 |
| 2D, D = full sample, ramp scaler W9 | -1.019247 | -1.961902 | 87,662 | 0.6339 | 0.1818 | 0.1114 | 4.746 | 8.609 | 0.8460 |
| 1D, D = full sample (x_(t-1) sample) | -1.015635 | -1.895233 | 87,662 | 0.6434 | 0.1823 | 0.1147 | 4.701 | 7.600 | 0.8538 |
| 2D, z re-standardized on W9  [R3'] | -1.094972 | -1.948266 | 78,902 | 0.6378 | 0.1834 | 0.1115 | 4.590 | 8.576 | 0.8224 |
| 1D, z re-standardized on W9  [R1'] | -1.091534 | -1.883173 | 78,902 | 0.6465 | 0.1838 | 0.1147 | 4.549 | 7.594 | 0.8303 |

## 6. Validation metrics of the primary set (reported, never targeted)

| metric | this set | M9 / target | comment |
|---|---|---|---|
| ergodic ratio occupancy | 0.66788 | 0.674578 | (1/d_n)/(1/d_n + 1/d_s) uses only the two targets; NOT a validation |
| forward-recursion occupancy | 0.6352 | 0.674578 | rel. error -5.8% — expected gap, not forced |
| AME ratio p01 | 0.1823 | 0.196281 | |
| AME ratio p10 | 0.1114 | 0.111177 | |
| E[1/p01] | 4.690 | 4.134850 | auxiliary |
| E[1/p10] | 8.582 | 8.585121 | auxiliary |

**p01 target conflict.** Upper bound pbar(1-pbar) = 0.19525 < M9 AME ratio 0.19628 → compatible: **no**; M9's own mean p01 must be ≥ 0.26823. Present in the 1D production set as well; the ramp does not resolve it.

- Embeddability, full historical sample: n = 87,662, s ≥ 1: 0 (0.000%), s ≥ 0.95: 0, max s = 0.8408 at 2020-05-25T06:00:00+00:00, 99.99% quantile 0.8030
- Embeddability, sample D: n = 78,902, s ≥ 1: 0 (0.000%), s ≥ 0.95: 0, max s = 0.8408 at 2020-05-25T06:00:00+00:00, 99.99% quantile 0.8032

## 7. Ramp scale and timing evidence

| construction | n | α10 | E[1/p10] | E[1/p01] |
|---|---|---|---|---|
| standardized ramp x 0, lag 1h | 78,902 | -1.890591 | 7.598 | 4.646 |
| standardized ramp x 1, lag 1h | 78,902 | -1.955839 | 8.582 | 4.690 |
| standardized ramp x 1.1, lag 1h | 78,902 | -1.968904 | 8.790 | 4.696 |
| raw dz (unstandardized, sd 0.2715), lag 1h | 78,902 | -1.896013 | 7.679 | 4.655 |
| standardized ramp, no lag (r_t) | 78,903 | -1.949465 | 8.483 | 4.631 |
| standardized ramp, lag 2h (r_(t-2)) | 78,900 | -1.960041 | 8.649 | 4.731 |
| standardized ramp, sign flipped | 78,902 | -1.943015 | 8.629 | 4.630 |

M9 mean expected stress duration: 8.585121. E[1/p10] discriminates the ramp SCALE (standardized ~8.58 vs raw ~7.68, M9 8.585) but not its direction or lag (all timing variants land within 8.48-8.65): scale is supported, direction/lag are unverified.

## 8. `rd_lag1_standardized.csv` evidence

- 87,663 / 87,665 rows equal the calendar lag of z; 2 rows need an hour that rd_standardized.csv does not contain.
- Rows contradicting a row-order lag: 2016-03-27T03:00:00+00:00: file -1.531325 vs row-order -1.200441.
- RD_lag1 in the shipped file is a CALENDAR one-hour lag: at 2016-03-27 03:00 UTC it carries z(02:00), an hour absent from rd_standardized.csv, not the row-order value z(2016-03-26 23:00). The upstream lag was therefore built on a calendar-consistent RD series. This supports (does not prove) a calendar-time ramp construction; the ramp itself is still unverified.
- Recoverable extra z values: {'2016-01-01T00:00:00+00:00': -1.8903582032166115, '2016-03-27T02:00:00+00:00': -1.531324537672116} (not used in the primary set; effect on α if used: |Δα01| = 3.6e-05, |Δα10| = 1.1e-05).

## 9. Decisions

- **ramp_scaler_window**: W9 (h slopes come from M9, whose training window W9 is inferred; the p10 moment check mildly favours it)
- **alpha_sample**: W9 (durations are M9 diagnostics); full sample reported as a sensitivity
- **p01_target_conflict**: documented limitation, target kept (same as 1D production)
- **time_alignment**: production l(tau) = t_v + floor(tau) - 1h kept; +/-1h reported as a separate sensitivity
- **non_embeddable_rows**: continuous-time mode REJECTS s >= 1; no silent clipping; discrete-time switching mode not implemented
