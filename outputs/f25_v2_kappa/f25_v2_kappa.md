# FW6a -- F2.5 pooled-vs-M9 comparison under the v2 kappa

`outputs/market_calibration_final/model_comparison_pooled_vs_M9.md` (F2.5)
reports every M9-kappa-bearing number at `kappa = 4.108274e-06 /h`
(`phi = 0.999995891734`), the pre-v2 value.  The production frozen-parameter
file now carries `kappa = 0.078394 /h` (`phi = 0.9246`), a factor of
19,082 faster.  The tables below are the same comparison at the
production kappa, on the production climatology `z(t-1)` path and the
production 1201-node residual grid.  Everything else -- forward curve,
`pi_filtered`, TVTP coefficients, strike, maturities, `r_annual`, pooled-sigma
construction -- is as in F2.5.

Pooled baseline: `sigma_pooled = sqrt(pi_normal * M0.sigma0^2 + pi_stress *
M0.sigma1^2) = 0.134952` with M9's stationary occupancy
`(0.325422, 0.674578)`, split by +/- 0.01 % so that
`sigma_y[1] > sigma_y[0]` still holds.  Note that the same weights applied to
M9's own sigmas give `0.075923`, a factor 0.5626 below the
pooled value: the two sides of F2.5 never shared a volatility level, which is
what the Interpretation section below turns on.

## Results at the v2 kappa -- `K = 3000` call, PDE only

| maturity_h | variant | kappa_per_hour | F_T_TRY_MWh | residual_sd_T_TRY_MWh | call_TRY_MWh |
|---:|---:|---:|---:|---:|---:|
| 24 | pooled_M0_kappa | 0.000819733 | 2917.24 | 1919.01 | 724.091 |
| 24 | pooled_M9_kappa | 0.078394 | 2917.24 | 987.302 | 353.487 |
| 24 | M9_prod | 0.078394 | 2917.24 | 524.08 | 163.926 |
| 72 | pooled_M0_kappa | 0.000819733 | 2916.16 | 3259.31 | 1254.65 |
| 72 | pooled_M9_kappa | 0.078394 | 2916.16 | 998.598 | 356.691 |
| 72 | M9_prod | 0.078394 | 2916.16 | 532.257 | 166.748 |
| 168 | pooled_M0_kappa | 0.000819733 | 2913.99 | 4790.89 | 1854.3 |
| 168 | pooled_M9_kappa | 0.078394 | 2913.99 | 997.871 | 353.844 |
| 168 | M9_prod | 0.078394 | 2913.99 | 531.87 | 164.949 |
| 336 | pooled_M0_kappa | 0.000819733 | 2910.21 | 6349.83 | 2450.7 |
| 336 | pooled_M9_kappa | 0.078394 | 2910.21 | 996.586 | 348.905 |
| 336 | M9_prod | 0.078394 | 2910.21 | 531.186 | 161.845 |

## Old (pre-v2 kappa) vs new (v2 kappa), side by side

`call_*` and `residual_sd_*` in TRY/MWh; `pct_call` is the change from the
published F2.5 value.

| maturity_h | variant | call_TRY_MWh_old | call_TRY_MWh_new | d_call | pct_call | residual_sd_T_TRY_MWh_old | residual_sd_T_TRY_MWh_new | pct_sd |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | pooled_M0_kappa | 724.0911 | 724.0911 | 0.0000 | 0.0000 | 1919.0053 | 1919.0053 | 0.0000 |
| 24 | pooled_M9_kappa | 731.5807 | 353.4871 | -378.0936 | -51.6817 | 1937.8183 | 987.3023 | -49.0508 |
| 24 | M9_prod | 368.1691 | 163.9261 | -204.2430 | -55.4753 | 1038.1586 | 524.0800 | -49.5183 |
| 72 | pooled_M0_kappa | 1254.6469 | 1254.6469 | 0.0000 | 0.0000 | 3259.3072 | 3259.3072 | 0.0000 |
| 72 | pooled_M9_kappa | 1292.8724 | 356.6911 | -936.1813 | -72.4110 | 3355.4713 | 998.5985 | -70.2397 |
| 72 | M9_prod | 687.0410 | 166.7477 | -520.2933 | -75.7296 | 1838.2959 | 532.2570 | -71.0462 |
| 168 | pooled_M0_kappa | 1854.3029 | 1854.3029 | 0.0000 | 0.0000 | 4790.8876 | 4790.8876 | 0.0000 |
| 168 | pooled_M9_kappa | 1985.6464 | 353.8444 | -1631.8020 | -82.1799 | 5122.6816 | 997.8708 | -80.5205 |
| 168 | M9_prod | 1073.9036 | 164.9494 | -908.9542 | -84.6402 | 2823.7732 | 531.8703 | -81.1646 |
| 336 | pooled_M0_kappa | 2450.6964 | 2450.6964 | 0.0000 | 0.0000 | 6349.8263 | 6349.8263 | 0.0000 |
| 336 | pooled_M9_kappa | 2799.3853 | 348.9053 | -2450.4800 | -87.5364 | 7237.4115 | 996.5864 | -86.2301 |
| 336 | M9_prod | 1525.7826 | 161.8446 | -1363.9380 | -89.3927 | 3998.6075 | 531.1857 | -86.7157 |

`pooled_M0_kappa` carries M0's own kappa and is therefore unchanged between the
two eras; its rows are the control that isolates the kappa edit.

## How the pooled-vs-M9 gap moves

Pre-v2 (reproduces the published F2.5 numbers):

| maturity_h | call_pooled_M0_kappa | call_pooled_M9_kappa | call_M9_prod | pct_vs_pooled_M0_kappa | pct_vs_pooled_M9_kappa |
|---:|---:|---:|---:|---:|---:|
| 24 | 724.0911 | 731.5807 | 368.1691 | -49.1543 | -49.6748 |
| 72 | 1254.6469 | 1292.8724 | 687.0410 | -45.2403 | -46.8593 |
| 168 | 1854.3029 | 1985.6464 | 1073.9036 | -42.0859 | -45.9167 |
| 336 | 2450.6964 | 2799.3853 | 1525.7826 | -37.7409 | -45.4958 |

v2 kappa:

| maturity_h | call_pooled_M0_kappa | call_pooled_M9_kappa | call_M9_prod | pct_vs_pooled_M0_kappa | pct_vs_pooled_M9_kappa |
|---:|---:|---:|---:|---:|---:|
| 24 | 724.0911 | 353.4871 | 163.9261 | -77.3611 | -53.6260 |
| 72 | 1254.6469 | 356.6911 | 166.7477 | -86.7096 | -53.2515 |
| 168 | 1854.3029 | 353.8444 | 164.9494 | -91.1045 | -53.3836 |
| 336 | 2450.6964 | 348.9053 | 161.8446 | -93.3960 | -53.6136 |

## Interpretation

* **The gap is now flat in maturity, and it is a volatility-level gap.**  Against `pooled_M9_kappa` it was -49.67 % at 24 h -> -45.50 % at 336 h before the refit, spanning 4.18 pp; at the production kappa it is -53.63 % at 24 h -> -53.61 % at 336 h, spanning 0.37 pp.  With a half-life of 8.84 h the residual variance is already 97.7 % of its stationary value at 24 h, so every maturity prices essentially the same stationary mixture variance and the M9_prod call is flat in maturity (spread 3.0 % of its mean).

* **F2.5's claim to isolate the value of regime-conditioning is withdrawn.**  The two sides of the comparison do not share a volatility level.  The pooled baseline takes its sigma from M0's fitted pair (0.006123, 0.164255) and lands at 0.134952; M9_prod runs on M9's pair (0.003535, 0.092407), whose occupancy-weighted effective volatility is 0.075923 -- a ratio of 0.5626.  The realised residual-sd ratio at expiry is 0.5308-0.5330, i.e. the whole flat gap is that sigma ratio.  What the comparison measures is the disagreement between the M0 and M9 fits about the volatility level, not the value of holding a filtered regime belief.  Under the pre-v2 kappa the confound was masked by the term structure; once the variance saturates inside the shortest maturity the gap collapses onto the sigma ratio and the confound is visible.

* **Where the regime-mixture effect is actually measured.**  FW9 round f prices a single-regime OU at the *same* stationary variance (`kappa_sensitivity_isovariance_v2.md`): 183.4 TRY/MWh against the production 166.75 at the 72 h ATM call, so the two-regime mixture is worth about -9 % at equal variance.  Same sign as the number here, an order of magnitude smaller: the mixture channel is the thin-tail correction, and the remaining ~44 pp of the ~-53 % is the M0-vs-M9 volatility level.  Support for time-varying transitions rests on FW9's likelihood-ratio test (`lr_test_TVTP_vs_constant.csv`: LR = 1178.66, df = 2), not on this comparison.

* **`pooled_M0_kappa` folds in a third channel.**  Its kappa is M0's own, now about 96x slower than production rather than 200x faster, so its residual variance keeps growing with maturity.  The -77.36 % at 24 h -> -93.40 % at 336 h column therefore compares mean-reversion rates on top of the volatility level, and is not a regime-conditioning measurement either.

* **F2.5's fourth point inverts.**  It reports the kappa swap alone (`pooled_M0_kappa` -> `pooled_M9_kappa`, same pooled sigma) as raising the call by 1-14 %, an order of magnitude below what it attributes to regime-conditioning.  At the production kappa the same swap moves it by -51 to -86 %, i.e. the mean-reversion timescale is the larger of the two channels, not the secondary one.

* **The accepted artefact pair is internally inconsistent.**  The v2 refit commit regenerated `outputs/market_calibration_final/model_comparison_pooled_vs_M9.csv` but not the `.md` beside it, so the published F2.5 tables are the pre-v2 numbers while its own data source is the v2 ones.  Both files lie in an accepted output tree and are left unchanged; the documentation update should mark F2.5 as superseded by this directory.

Files: `f25_v2_kappa.csv` (new), `f25_pre_v2_reproduction.csv` (old,
regenerated at `4.108274e-06`), `f25_old_vs_new.csv`, `f25_gap_v2.csv`,
`f25_gap_pre_v2.csv`, `run_manifest.json`.  The accepted F2.5 artefacts under
`outputs/market_calibration_final/` are not modified.
