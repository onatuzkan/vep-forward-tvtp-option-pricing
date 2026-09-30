# Model limitations and assumption inventory

Label of every price produced here: **VEP-forward-curve anchored option prices**.

## What is genuinely constrained by the market

| quantity | status |
|---|---|
| hourly forward level F(t) inside 2026-02, 2026-03, 2026-04, 2026-05, 2026-06, 2026-07 | constrained; monthly averages reproduce the VEP quotes exactly |
| price level in 2025-12, 2026-01 | assumed; near-term anchored, no observed quote |
| residual volatility (sigma_normal, sigma_stress) | inherited from the historical M2 fit, not market-implied |
| regime transition dynamics (TVTP) | inherited from the historical fit |
| volatility risk premium | not identified; needs option premia |
| regime-transition premia eta01, eta10 | not identified; needs option premia (FW2 traces the identifiability envelope only) |
| market price of risk lambda_i | not identified; needs option premia |
| discount rate | assumed flat annual rate (0.40); the sensitivity swept in `discount_rate_sensitivity.md` gives under 2 % impact across `r_annual in [0.15, 0.60]` at maturities up to 336 h |

## Results that rest on an assumption, not on data

1. **January 2026 (`spot_to_next_linear` anchor).** near-term anchored, not directly constrained by an observed January VEP quote. Every horizon below falls inside it, so all four reported expected spots are anchored rather than market-constrained:

   - 72 h to 2026-01: near-term anchored
   - 168 h to 2026-01: near-term anchored
   - 336 h to 2026-01: near-term anchored
   - 720 h to 2026-01: near-term anchored

2. **Option prices depend on inherited volatility.** The forward calibration is exactly invariant to sigma (the centering ODE has no sigma term), so a wrong sigma cannot break the monthly fit but it moves every option value one-for-one. Option prices are therefore level-anchored and volatility-assumed.

3. **Residual dispersion under the v2 kappa.** With kappa = 7.8394e-02 /h (half-life 8.84 h) the residual standard deviation reaches its stationary value within roughly 24 h and stays flat thereafter:

| horizon | F(t) (TRY/MWh) | residual sd (TRY/MWh) | sd / F |
|---|---|---|---|
| 72 h | 2916.16 | 532.3 | 0.183 |
| 168 h | 2913.99 | 532.3 | 0.183 |
| 336 h | 2910.21 | 532.3 | 0.183 |
| 720 h | 2901.51 | 532.3 | 0.183 |

   In the **additive** residual mode this can admit small negative simulated prices in the tail; the Monte Carlo probability P(P_T < 0) collapses from 0.055 under the pre-v2 kappa to 0.000 under the shipped v2 kappa. The additive mode is appropriate at day-ahead and few-week horizons; use `residual_mode: multiplicative` for month-scale work, where prices stay positive by construction. Either way the monthly forward fit is unaffected.

4. **The regime split is historical, not market-implied.** Regimes here describe the residual around the market curve, not the price level: regime 0 is the calm state in which the spot tracks the forward closely, regime 1 the stress state. Nothing in the VEP data identifies how the market prices that stress risk.

## Legacy model

> legacy_asinh_ou: E[P_t] = scale_P * exp(v(t)/2) * sinh(m(t)) grows exponentially in the variance v(t) = sigma^2 (1 - e^{-2 kappa t}) / (2 kappa). With the fitted near-unit-root kappa and the stress-regime sigma this makes long-horizon expected prices economically meaningless. Use the legacy mode for short-horizon benchmarking and diagnostics only.

`legacy_asinh_ou` remains available and unchanged for short-horizon benchmarking, and its long-horizon output is flagged everywhere it is produced.

## Parameter-provenance caveats (post-M9-integration)

Everything below is a property of `inputs/historical/m2_frozen_parameters.yaml` after the M9-fit integration and the v2-kappa refit. These are not calibration failures; they are consequences of what the uploaded calibration bundle contained (and did not contain). Full derivation trail in `docs/tvtp_derivation_methodology.md`.

### (a) `tvtp.alpha01`, `tvtp.alpha10` are DERIVED, not estimated

The uploaded M9 bundle exports the two `gamma` slopes for `RD_lag1` but no intercepts. The current yaml intercepts (`alpha01 = -1.0157`, `alpha10 = -1.8952`) were back-solved by root-finding on the reported occupancy and mean-duration diagnostics of the same M9 run:

* `E_z[sigma(alpha01 + gamma01 z)] = 1 / mean_duration_normal`
* `E_z[sigma(alpha10 + gamma10 z)] = 1 / mean_duration_stress`

with `z` taken from the full historical `rd_standardized.csv`.

FW9c ran an independent profile MLE at `phi = 0.99999` (fixed) with a positive-definite Hessian and obtained `alpha01 = -0.6781 +/- 0.019` and `alpha10 = -1.6511 +/- 0.017`. The yaml DERIVED values sit roughly 14 to 18 standard errors outside these confidence intervals, so the occupancy and duration root-finding pipeline does not reproduce the direct MLE alphas even at the correct conditional profile. However, the FW9c price-impact decomposition shows that this discrepancy moves the 72 h call at K = 3000 by only +3.75 TRY (roughly 2.2 % of the price and 0.4 % of the total FW9-vs-production gap), so the shipped alpha choice is not the dominant driver of the production-vs-independent-MLE gap. Source: `outputs/fw9_self_estimation/parameter_comparison_v2.csv`, `outputs/fw9_self_estimation/price_impact_v2_decomposition.csv`.

### (b) `RD_Ramp_1h_lag1` covariate is dropped in the default TVTP-1 mode

The M9 fit (`M9_student_t_tvtp_TVTP-2`) uses two transition covariates: `RD_lag1` and `RD_Ramp_1h_lag1`. The current PDE code hard-codes a single-covariate TVTP (`generator.py`; `covariate = 'RD_lag1'`) in the default mode, and only the `RD_lag1` gamma is copied into the yaml. Reported average marginal effects from M9: `ame_p01` for `RD_lag1` = +0.0086, for `RD_Ramp_1h_lag1` = +0.0451 (roughly five times larger). The derived alphas absorb the mean effect of the missing ramp term, biasing the intercepts by an unknown but non-zero amount.

FW9 jointly re-estimated a two-covariate TVTP (`outputs/fw9_self_estimation/TVTP_2cov.pkl`) on the raw asinh(PTF) level. The fit is at the unit-root boundary (`phi = 0.9999987156`, half-life about 62 years) and did not meet the gradient tolerance (`converged = False`, gradient norm 96.6 at n = 61 368), so the ramp slopes carry no interpretable standard error. The two-covariate log-likelihood exceeds the one-covariate value by LR = 869.93 (df = 2); because the fit did not converge and sits at the phi boundary, the statistic is indicative rather than a formal test. FW4-P then priced the FW9 ramp channel with occupancy held fixed: at 72 h K = 3000 the ramp effect is -1.042 % of the option value (compared with -1.010 % from the original FW4 reconstructed ramp). The ramp channel is therefore small (about 1 % of the option value near the money) and both the estimated slopes and the reconstructed ramp are labelled experimental. See `outputs/fw4p_ramp_price_impact/` and `docs/tvtp2_methodology.md`.

### (c) `scale_P = 282.48` is unverifiable in the current tree

The yaml pins `scale_P = 282.48` TRY/MWh with provenance "training median absolute price". The source files that were meant to carry this value (`pde_export.json`, `prepared_meta.json`) are NOT present in the current `inputs/` tree, and the raw historical PTF series needed to re-derive it directly against the M9 window is also absent. FW9 applied the training-median-absolute-price definition on the 2019-2025 window and finds 1 399.99 TRY/MWh, and on the 2019-2020 subwindow (before the 2021-2024 TRY depreciation) 302.02 TRY/MWh, within 7 % of the yaml value 282.48. FW9 concludes the yaml `scale_P` is consistent with an early-window TRY median but the exact reference window is not shipped. The FW9 decisive test against the alternative "USD-fit" hypothesis (bundle metadata `target` field is `"asinh(PTF_TRY_MWh)"`, and the TVTP fit on `asinh(USD_PTF/282.48)` does not land on the yaml sigmas) rules out that possibility. Source: `outputs/fw9_self_estimation/scale_P.json`, `preprocessing_audit.md`.

### (d) M9 regime labels were swapped to the yaml convention

M9's raw fit puts the high-volatility state at index 0 (occupancy 67.5 %, duration 7.56 h). The yaml convention (`params_frozen.FrozenM2Parameters.__post_init__`) requires `sigma_y[1] > sigma_y[0]`, i.e. index 0 = normal and index 1 = stress. All four TVTP coefficients were therefore cross-swapped: yaml `sigma_y_normal` from raw M9 `sigma1`, yaml `sigma_y_stress` from raw M9 `sigma0`, yaml `gamma01` (normal to stress) from raw M9 `gamma10`, yaml `gamma10` (stress to normal) from raw M9 `gamma01`. Signs of the RD sensitivities are economically consistent (higher RD reduces normal to stress and raises stress to normal). Cross-check simulation reproduces the reported occupancy diagnostics to within 5 %.

### (e) `pi_filtered` is M2-sourced, mixed with M9 dynamics

`pi_filtered = [0.9320, 0.0680]` comes from the shipped M2 filter output at the valuation hour; sigma, gamma and alpha values are all from the M9 fit. No M9-specific valuation-time filtered probability is available (`pde_timeseries.parquet` is not in the current inputs and the shipped ZIP handoff did not include it). FW9c ran the Hamilton filter on the FW9 data at the profile parameters and obtained a terminal filtered distribution `[0.983, 0.017]` at the valuation instant, roughly a 5 pp shift, both dominated by the normal regime. FW9c reprices the K = 3000 call under both distributions at T in {1, 2, 6, 12, 24, 48, 72} h: the effect shrinks with horizon (20.8 % at T = 1 h, 1.6 % at T = 6 h, 0.06 % at T = 24 h, essentially zero at T = 72 h) because the regime memory half-life at climatology z = 0 is about 1.37 h. For the shipped reporting horizons (T >= 24 h) the `pi_filtered` source uncertainty is a small fraction of one basis point. `run_pde.py price --pi-override stationary` swaps in the M9 long-run occupancy (`m9_stationary_pi` in the frozen-parameter yaml) for sensitivity analysis. Source: `outputs/fw9_self_estimation/pi_filtered_horizon.csv`.

### (f) Risk-premium channels (Q1 drift shift and Q2 transition shift) are wired but UNCALIBRATED

The `run_pde.py price` command accepts `--risk-premium-a0` and `--risk-premium-a1` (TRY/MWh per hour, regime-0 and regime-1 respectively), which apply a Q1 drift shift `a_i` on the residual SDE. FW2 added the Q2 transition-intensity shift `q_ij^Q = q_ij^P * exp(eta_ij)` end-to-end via `price_forward_centered(..., eta_ij=...)`. Both are threaded symmetrically into the moment ODE and the pricing PDE and MC simulator, so `E^Q[P_t] = F(t)` is preserved exactly for every real `(a, eta)` (multiplicative form guarantees generator validity; guarded by `tests/test_forward_centered.py::test_q1_drift_shift_preserves_centering` and the FW2 Q2 wiring tests).

No electricity option market exists to estimate a central premium, so both channels stay UNCALIBRATED sensitivity scenarios. Default `(a, eta) = (0, 0)` reproduces every prior benchmark bit-for-bit. FW2 §2 derives an ex-post empirical envelope from the 7 tracked VEP snapshots plus hourly realised PTF, look-ahead-guarded to 2025-12-31 20:00 UTC. FW2 §4 sensitivity sweep (480 rows) at the empirical upper bounds: at the 72 h call with K = 3000 the Q1 drift channel alone moves the value by up to +5.7 % at `a_stress = 50`, the Q2 transition channel alone by +21 % to -30 % at `|eta| = 0.5`, and the joint corner at `(a_stress = 50, eta = (+0.75, -0.75))` reaches +29 %. The Q1 effect is O(a^2), the Q2 effect O(eta) in the terminal variance, so Q2 is a priori materially the more powerful channel. See `docs/fw2_risk_premium_identification.md`, `docs/risk_neutral_methodology.md`, and `outputs/fw2_risk_premium/`.

### (g) Within-regime phi versus deseasonalized single-regime AR fit: resolved by the v2 kappa refit

M9's regime-conditional phi (0.999996) implies a within-regime OU half-life of about 19.25 years on raw asinh(PTF), dominated by TRY-inflation-era trend. The deseasonalized single-regime AR(1) on the same series delivers about 8.84 h. Theoretical and numerical analysis in `outputs/market_calibration_final/half_life_reconciliation.md` shows the two numbers describe different variables. The v2 kappa refit reconciled the yaml `phi` and `kappa_per_hour` to the deseasonalized single-regime value appropriate for the (P - F) residual (`phi = 0.9246`, `kappa_per_hour = 0.078394`, half-life 8.84 h). Pre-refit yaml is archived in `inputs/historical/archive/m2_frozen_parameters.PRE_V2_KAPPA_REFIT.yaml`.

### (h) `scale_P` was fit on a 9-year window straddling severe TRY depreciation

`scale_P = 282.48` is the training-window median absolute PTF over roughly 9 years (2016 to 2024/2025, `n_train = 78 905` hours). This window spans a period of extreme TRY depreciation and large nominal PTF inflation. A single 9-year median absolute price may not represent the same price regime as the current valuation point (spot 2917.78 TRY/MWh, end-2025), so the `asinh(PTF / scale_P)` normalisation may be mis-scaled. FW9 quantified this by running the full TVTP profile fit on the CPI-deflated series (`scripts/fw9/estimate_deflated.py`; base 2025-12, deflated `scale_P = 3092.05`): real-terms sigmas are 22 to 33 % smaller than nominal (`sigma_normal` 0.0055 vs 0.0070, `sigma_stress` 0.162 vs 0.242), so inflation contributes about 30 % of the observed volatility. Even after deflation the FW9 sigmas remain well above the yaml values. A fit on the 2019-2021 sub-window closes 47 % of the sigma_stress gap but widens the sigma_normal gap by 43 %, so the absence of 2016-2018 data does not explain the remainder, which stays unexplained.

## Findings from the FW9 to FW12 audit rounds

### (i) Kappa and sigma are individually mis-specified, but their stationary variance holds until 2026

FW9e re-estimated the residual process on the model-faithful A3 residual (price minus the calendar-month mean minus the within-month hour-of-week shape). Single-regime AR(1) fits give kappa between 0.19 and 0.26 /h in every year from 2019 to 2025, and a consistent two-regime TVTP fit on the full window has an interior phi of 0.859 (kappa 0.152 /h); the production value of 0.078 /h is reached in none of them. The production innovation scale is also narrower than the data (222 against 359 TRY/MWh per hour in 2025). The two errors offset in the stationary variance, which is what governs the 24 to 72 h option values: the production parameters imply a stationary residual sd of 582.5 TRY/MWh at the valuation forward, against a realised 2025 value of 531 (hour-of-week shape estimated on 2025 alone) or 624 (shape pooled over 2019-2025). In a long-path simulation the Kolmogorov-Smirnov distance to the realised 2025 residual is 0.087 for the production set, against 0.125 for the FW9e full-window fit and 0.159 for the FW9f 2022-2025 fit. No set is accepted by the test, and the realised series is truncated by the price floor and cap, which the model does not contain.

FW11 repeats the check at seven valuation dates with a trailing twelve-month window and the shape estimated on that window. The ratio of realised to production-implied dispersion is 0.93 to 1.13 (mean 1.02) at the six dates from 2023-06-30 to 2025-12-31, 1.26 at 2022-12-31, 1.65 for the twelve months to 2026-09-27 and 1.91 for 2026 alone. The pooled shape gives ratios 9 to 18 % higher at every date. Hours below 50 TRY/MWh rise from 0.4-1.0 % before 2026 to 5.0-6.7 % in 2026, but excluding them leaves the 2026 ratios essentially unchanged, so the wider 2026 dispersion is not a zero-price effect.

**Consequence for option values.** With the stationary variance held at the production target, the 72 h call at K = 3000 stays within 169 to 176 TRY/MWh across the model-faithful kappa range 0.15 to 0.22 /h and within 162 to 187 across 0.01 to 0.30 /h, while the 1 h call moves from 7 to 98. Kappa matters at short horizons and not at the reported ones: values below about 12 h are unreliable, and the 24 to 72 h values reported in the paper are not materially affected. Source: `outputs/fw9_self_estimation/stationary_variance_check.md`, `tail_validation.md`, `kappa_sensitivity_isovariance_v2.md`, `outputs/fw11_variance_stability/variance_stability.md`, `outputs/fw11_variance_stability/FW11_report_TR.md`.

The earlier FW9c/FW9d framing ("yaml kappa is wrong, outside all three brackets") has been WITHDRAWN in favour of the FW9f finding above.

### (j) Constant scale mapping breaks in 2026

The residual model applies `asinh` with a constant `scale_P = 282.48` and the delta-method mapping `sigma_price = sigma_y * sqrt(F^2 + scale_P^2)`, which forces the implied sd to scale roughly linearly with the price level. With the twelve-month shape, realised `sd / L` is 0.187 to 0.225 from mid-2023 to end-2025, bracketing the production value of 0.199; it is 0.252 in 2022, 0.330 for the twelve months to 2026-09-27 and 0.384 for 2026 alone. The 2026 price level lies within the 2023-2025 range, so the 2026 gap is a rise in relative volatility, not a level effect. Source: `outputs/fw11_variance_stability/variance_stability.md`.

### (k) Out-of-sample 2026 evidence (FW10b)

The 60-day evaluation window (2026-01-05 to 2026-09-24, N_PATHS = 10 000, forecasts under the FW10b corrected day-ahead timing rule and the production HPFC-shaped forward curve) shows most of the forecast error is driven by the forward curve, not by the residual dynamics. The pure-residual sd across all 2026 hours January-September is 723.1 TRY/MWh; the model residual sd runs 387 to 518 TRY/MWh over the 6 to 72 h horizons, so the observed pure residual is wider by a factor of 1.33 to 1.55. The forward-error component has a mean of +385 to +794 TRY/MWh, positive at every horizon, and a standard deviation of 697 to 843 TRY/MWh, the same order as the residual itself; the forward curve carries all of the bias and roughly half of the error variance. Reading `var(z)` from `bias_dispersion_summary.csv` therefore mixes forward-curve error with residual dispersion. Source: `outputs/fw10_validation/forward_residual_decomposition.md`.

Delta-hedge effectiveness against the aggregate VEP monthly quote is zero across every (model, horizon, moneyness) cell of the FW10b evaluation panel: on 2022-2026 the monthly VEP GGF quote changes on 8.5 % of contract-day pairs (584 / 6 893), on 2026 delivery-year contracts on 1.0 % (14 / 1 354), and on quotation dates in 2026 on 0.6 % (6 / 964). Within the 60-day evaluation window the count of within-day quote changes is exactly zero for every cell, so the hedge P and L is identically zero by construction. This is a property of the market data, not of the model: the daily reference price of the monthly VEP contracts rarely changes. Source: `outputs/fw10_validation/vep_quote_staleness.csv`, `outputs/fw10_validation/FW10_report_TR.md`.

### (l) Valuation timing (day-ahead publication audit)

Valuation instant is 2025-12-31 20:00 UTC (23:00 TRT). The Turkish day-ahead auction clears the next day's hourly prices at approximately 14:00 TRT of the current day, so at the valuation instant the 24 h target (2026-01-01) is already published while the 48 h and 72 h targets are not. FW10b enforces a strict day-ahead rule for the 60-day daily evaluation: valuation at 11:00 TRT of day d, last known PTF hour d 23:00 TRT, forward curve built from the last VEP GGF publication strictly before d 11:00 TRT. Under the corrected rule the shipped 2025-12-31 valuation would be rebuilt from the 2025-12-30 VEP quote day, but the 2025-12-30 and 2025-12-31 VEP GGF quotes are bit-identical (the reference price did not change), so the shipped 15-cell (K, T) grid moves by 0 TRY (0 %). No production number changes. Source: `outputs/fw10_validation/day_ahead_timing_check.md`, `outputs/fw10_validation/shipped_vs_corrected_reprice.csv`.

### (m) Regime structure: TVTP support and the withdrawn F2.5 claim

The LR statistic for TVTP against a constant-transition benchmark on the transformed price level asinh(P / 282.48) is 1178.66 with df = 2 (`lr_test_TVTP_vs_constant.csv`); on the model-faithful A3 residual it is 1851 over 2019-2025 and 1692.56 over 2022-2025 (log-likelihood gain 846.28, `FW9f_report_TR.md` §3), so time-varying transitions are supported statistically.

The equal-stationary-variance regime-mixture price effect comes from FW9f: at 72 h and K = 3000 the production two-regime call is 166.75 TRY, against 183.4 TRY for a single-regime OU with the same stationary variance, so the mixture lowers the call by about 9 %. This is the effect of the regime mixture at equal variance, separated from the volatility-level channel.

The earlier F2.5 comparison in `outputs/market_calibration_final/model_comparison_pooled_vs_M9.md` conflates the two channels and, after FW6a rebuilt it under the v2 kappa, its "M9 vs pooled_M9_kappa" contrast collapses to a flat -53 % across maturities (versus the pre-v2 4.18 pp spread). FW6a therefore WITHDRAWS the F2.5 interpretation that regime conditioning explains the term-structure spread; the correct read is that most of the F2.5 gap is a sigma-level difference between M0 and M9 estimates rather than a value of regime conditioning. The F2.5 files themselves are left untouched inside the accepted output tree; the withdrawal is recorded here and in `docs/PROJECT_STATUS_AND_FUTURE_WORK.md`. Source: `outputs/f25_v2_kappa/FW6a_report_TR.md`, `outputs/f25_v2_kappa/f25_v2_kappa.csv`.

### (n) Ramp covariate re-estimation (FW4-P)

Under FW4-P, holding stationary occupancy fixed, the FW9 ramp (own series and own slopes) moves the 72 h K = 3000 call by -1.042 % (compared with -1.010 % from the FW4 reconstructed ramp). If instead the production intercepts are kept and only the ramp channel is turned on, the effect grows to -2.18 % because the ramp then also shifts the stationary stress occupancy from 0.6008 to 0.6364, mixing the ramp channel with an occupancy shift. The occupancy-controlled number is the correct read of the ramp channel and is roughly 1 % near the money. The joint two-covariate MLE that produced these slopes is at the unit-root boundary (`phi = 0.9999987`) and did not meet the gradient tolerance (gradient norm 96.6 at n = 61 368), so the slopes carry no interpretable standard error and the ramp remains experimental. Source: `outputs/fw4p_ramp_price_impact/fw4p_ramp_effect.csv`, `outputs/fw4p_ramp_price_impact/FW4P_report_TR.md`.

### (o) Risk-premium identifiability

FW2 §2 makes the identifiability status explicit: the ex-post premium panel has `n_obs = 5-6` per horizon (7 tracked VEP snapshots against realised hourly PTF), pooled mean about 625 TRY/MWh, block-bootstrap SEs 280-530 TRY/MWh (of the same order as the means), and 95th percentile of `|premium|` about 2000 TRY/MWh (dominated by the 2026 Q2 realised collapse). Converting the pooled mean to an `|a_i|` bound via `|a| = kappa * L` at yaml v2 `kappa = 0.078394 /h` yields about 49 TRY/MWh/h; the empirical envelope is a mathematical range, not an identified estimate. `eta_ij` has no matching empirical anchor: the bound `|eta_ij| <= 1` used in the FW2 §4 sweep is generator-validity only. No electricity option market exists for this delivery point. Source: `outputs/fw2_risk_premium/literature_anchor_and_a_bound.md`, `docs/fw2_risk_premium_identification.md`.

### (p) Numerical convergence

At the shipped production grid `ResidualGridSettings(n_space_nodes = 1201, n_time_steps = 2 per hour, n_std = 6)` the 72 h call at K = 3000 is 166.7477 TRY. Richardson extrapolation from (n = 2401, 4801) with observed order p about 1.66 gives V_star = 166.686 TRY, so the discretisation error at production is about 0.062 TRY/MWh (0.037 % relative). Time-step convergence is under the spatial error at 2401 nodes. Boundary sensitivity is `|Delta V / V| <= 0.024 %` across `n_std in {4, 5, 6, 7.5, 9}`. Observed spatial order is 1.21 to 1.39 rather than the theoretical 2 for Crank-Nicolson, which reflects the payoff kink at K, the far-field boundary treatment for OTM contracts, and the interaction with the O(k^2) time component; §2 of the FW12 report separates the two. The discretisation error is two orders of magnitude below the FW2 risk-premium envelope, so the production grid is kept. Source: `outputs/fw12_convergence/spatial_convergence.md`, `outputs/fw12_convergence/grid_recommendation.md`.
