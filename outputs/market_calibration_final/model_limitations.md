# Model limitations and assumption inventory

Label of every price produced here: **VEP-forward-curve anchored option prices**.

## What is genuinely constrained by the market

| quantity | status |
|---|---|
| hourly forward level F(t) inside 2026-02, 2026-03, 2026-04, 2026-05, 2026-06, 2026-07 | **constrained** — monthly averages reproduce the VEP quotes exactly |
| price level in 2025-12, 2026-01 | **assumed** — near-term anchored, no observed quote |
| residual volatility (σ_normal, σ_stress) | **inherited** from the historical M2 fit, not market-implied |
| regime transition dynamics (TVTP) | **inherited** from the historical fit |
| volatility risk premium | **not identified** — needs option premia |
| regime-transition premia η01, η10 | **not identified** — needs option premia |
| market price of risk λ_i | **not identified** — needs option premia |
| discount rate | **assumed** flat annual rate |

## Results that rest on an assumption, not on data

1. **January 2026 (`spot_to_next_linear` anchor).** near-term anchored, not directly constrained by an observed January VEP quote. Every horizon below falls inside it, so all four reported expected spots are anchored rather than market-constrained:

   - 72 h → 2026-01 — near-term anchored
   - 168 h → 2026-01 — near-term anchored
   - 336 h → 2026-01 — near-term anchored
   - 720 h → 2026-01 — near-term anchored

2. **Option prices depend on inherited volatility.** The forward calibration is exactly invariant to σ (the centering ODE has no σ term), so a wrong σ cannot break the monthly fit — but it moves every option value one-for-one. Option prices are therefore *level-anchored, volatility-assumed*.

3. **Residual dispersion inherits a near-unit-root κ.** With κ = 3.852e-04/h (half-life 1799 h) the residual standard deviation keeps growing over the horizon:

| horizon | F(t) (TRY/MWh) | residual sd (TRY/MWh) | sd / F |
|---|---|---|---|
| 72 h | 2916.2 | 1834.0 | 0.63 |
| 168 h | 2914.0 | 2769.8 | 0.95 |
| 336 h | 2910.2 | 3803.8 | 1.31 |
| 720 h | 2901.5 | 5195.7 | 1.79 |

   In the **additive** residual mode this admits negative simulated prices at long horizons — economically wrong for PTF, which is floored at zero. The additive mode is appropriate at day-ahead to few-week horizons; use `residual_mode: multiplicative` for month-scale work, where prices stay positive by construction. Either way the monthly forward fit is unaffected.

4. **The regime split is historical, not market-implied.** Regimes here describe the *residual* around the market curve, not the price level: regime 0 is the calm state in which the spot tracks the forward closely, regime 1 the stress state with ~26× the residual volatility. Nothing in the VEP data identifies how the market prices that stress risk.

## Legacy model

> legacy_asinh_ou: E[P_t] = scale_P * exp(v(t)/2) * sinh(m(t)) grows exponentially in the variance v(t) = sigma^2 (1 - e^{-2 kappa t}) / (2 kappa). With the fitted near-unit-root kappa and the stress-regime sigma this makes long-horizon expected prices economically meaningless. Use the legacy mode for short-horizon benchmarking and diagnostics only.

`legacy_asinh_ou` remains available and unchanged for short-horizon benchmarking, and its long-horizon output is flagged everywhere it is produced.

## Parameter-provenance caveats (post-M9-integration)

Everything below is a property of `inputs/historical/m2_frozen_parameters.yaml` after the M9-fit integration.  These are not calibration failures — they are unavoidable consequences of what the uploaded calibration bundle contained (and did not contain). Full derivation trail in `docs/tvtp_derivation_methodology.md`.

### (a) `tvtp.alpha01`, `tvtp.alpha10` are DERIVED, not estimated

The uploaded M9 bundle exports the two `gamma` slopes for `RD_lag1` but no intercepts. The current yaml intercepts (`alpha01 = -1.0157`, `alpha10 = -1.8952`) were back-solved by root-finding on the reported occupancy / mean-duration diagnostics of the same M9 run:

* `E_z[σ(alpha01 + gamma01·z)] = 1 / mean_duration_normal`
* `E_z[σ(alpha10 + gamma10·z)] = 1 / mean_duration_stress`

with `z` taken from the full historical `rd_standardized.csv`. Consequence: no MLE standard error attached; a cross-check simulation reproduces the target 67% stress occupancy to within 4.5%.

### (b) `RD_Ramp_1h_lag1` covariate is dropped (omitted-variable risk)

The M9 fit (`M9_student_t_tvtp_TVTP-2`) uses two transition covariates: `RD_lag1` and `RD_Ramp_1h_lag1`. The current PDE code hard-codes a single-covariate TVTP (`generator.py`; `covariate = 'RD_lag1'`) and only the `RD_lag1` gamma is copied into the yaml. Reported average marginal effects: `ame_p01` for `RD_lag1` = +0.0086, for `RD_Ramp_1h_lag1` = +0.0451 (~5× larger). The derived alphas absorb the mean effect of the missing ramp term, biasing the intercepts by an unknown but non-zero amount.

### (c) `scale_P = 282.48` is unverifiable in the current tree

The yaml pins `scale_P = 282.48` TRY/MWh with provenance "training median absolute price". The source files that were meant to carry this value (`pde_export.json`, `prepared_meta.json`) are NOT present in the current `inputs/` tree, and the raw historical PTF series needed to re-derive it is also absent. `scale_P` is treated as an inherited constant, trusted on faith from external metadata no longer reachable.

### (d) M9 regime labels were swapped to the yaml convention

M9's raw fit puts the high-volatility state at index 0 (occupancy 67.5%, duration 7.56 h). The yaml convention (`params_frozen.FrozenM2Parameters.__post_init__`) requires `sigma_y[1] > sigma_y[0]`, i.e. index 0 = normal, index 1 = stress. All four TVTP coefficients were therefore cross-swapped: yaml `sigma_y_normal` ← raw M9 `sigma1`, yaml `sigma_y_stress` ← raw M9 `sigma0`, yaml `gamma01` (normal→stress) ← raw M9 `gamma10`, yaml `gamma10` (stress→normal) ← raw M9 `gamma01`. Signs of the RD sensitivities are economically consistent (higher RD reduces normal→stress and raises stress→normal). Cross-check simulation reproduces the reported occupancy diagnostics to within 5%.

### (e) `pi_filtered` is M2-sourced, mixed with M9 dynamics

`pi_filtered = [0.9320, 0.0680]` comes from the shipped M2 filter output at the valuation hour; sigma/gamma/alpha values are all from the M9 fit. No M9-specific valuation-time filtered probability is available (`pde_timeseries.parquet` is not in the current inputs and the shipped ZIP handoff did not include it). Empirically measured impact of this mixed-source choice on the K=3000 European call: <2 % on 72 h+ maturities (the TVTP chain converges to its stationary distribution within ~10 mean durations), 5-30 % on <24 h maturities where the initial condition still dominates. `run_pde.py price --pi-override stationary` swaps in the M9 long-run occupancy (`m9_stationary_pi` in the frozen-parameter yaml) for sensitivity analysis; see `docs/tvtp_derivation_methodology.md` for the full derivation.

### (f) Risk-neutral drift adjustment (Q1) is wired but UNCALIBRATED

The `run_pde.py price` command accepts `--risk-premium-a0` and `--risk-premium-a1` (TRY/MWh per hour, regime-0 and regime-1 respectively), which apply a Q1 drift shift `a_i` on the residual SDE.  The shift is threaded symmetrically into the moment ODE and the pricing PDE / MC simulator, so `E^Q[P_t] = F(t)` is preserved exactly (guarded by `tests/test_forward_centered.py::test_q1_drift_shift_preserves_centering`).  However, **no electricity option market data exists to estimate a real market price of risk**, so the flags are UNCALIBRATED sensitivity scenarios only.  Default `(0, 0)` reproduces every prior benchmark bit-for-bit -- the physical-measure intensities are used as-is, which is the zero-risk-premium assumption.  See `docs/risk_neutral_methodology.md` for the mathematical proof that the drift channel leaves the forward-curve identity intact and only moves higher moments (variance -> option value).

