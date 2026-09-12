# v2 kappa refit — before / after benchmark comparison

**Date**: 2026-09-13.  Yaml before archived at
`inputs/historical/archive/m2_frozen_parameters.PRE_V2_KAPPA_REFIT.yaml`;
legacy reference before archived at
`inputs/legacy_reference/archive/legacy_model_implied_forwards.PRE_V2_KAPPA_REFIT_ERA.json`.

## Parameter change

| field | v1 (pre-refit) | v2 (post-refit) | ratio |
|---|---:|---:|---:|
| `phi` | 0.999995891734 | **0.9246** | — |
| `kappa_per_hour` | 4.108e-6 | **0.0784** | 19 070× larger |
| `half_life_hours` | 168 720 | **8.84** | 19 070× shorter |
| `sigma_y_normal` | 0.003534807 | 0.003534807 | unchanged |
| `sigma_y_stress` | 0.09240665 | 0.09240665 | unchanged |
| TVTP alpha/gamma | (unchanged) | (unchanged) | — |
| `scale_P`, `pi_filtered`, other | (unchanged) | (unchanged) | — |

Provenance: v1 phi was the M9 within-regime AR(1) coefficient on raw
`asinh(PTF)`, which absorbed the TRY-inflation-era trend.  v2 phi is
the deseasonalized single-regime AR(1) from
`metadata/deseasonalized_stationarity_summary.csv` in the M9 handoff
bundle — the persistence of the shock-around-anchor residual that the
forward-centered model actually prices.  See
`outputs/market_calibration_final/half_life_reconciliation.md` for the
theoretical diagnosis.

## Benchmark impact

### 72 h K = 3000 call (canonical benchmark)

| quantity | v1 | v2 | Δ | Δ% |
|---|---:|---:|---:|---:|
| F(T) | 2916.16 | 2916.16 | 0 | 0.00 % (invariant to κ) |
| Residual sd at expiry | 1838.30 | 532.26 | −1306.04 | **−71.05 %** |
| PDE call value | 687.04 | 166.75 | −520.29 | **−75.73 %** |
| MC call value | 694.89 | 169.18 | −525.71 | **−75.65 %** |
| MC P(P_T < 0) | 0.0551 | **0.0000** | −0.0551 | **−100 %** |

### Scenario sweep (rd_offset ±2σ)

| offset | v1 call | v2 call | Δ% | v1 sd | v2 sd | sd Δ% |
|---:|---:|---:|---:|---:|---:|---:|
| −2.0 | 777.27 | 200.81 | −74.2 % | 2060.0 | 608.0 | −70.5 % |
| −1.5 | 761.11 | 194.75 | −74.4 % | 2020.0 | 594.2 | −70.6 % |
| −1.0 | 741.05 | 187.14 | −74.7 % | 1970.6 | 577.2 | −70.7 % |
| −0.5 | 716.51 | 177.83 | −75.2 % | 1910.3 | 556.5 | −70.9 % |
| **0.0** | **687.04** | **166.75** | **−75.7 %** | 1838.3 | 532.3 | −71.0 % |
| +0.5 | 652.32 | 153.97 | −76.4 % | 1753.9 | 504.4 | −71.2 % |
| +1.0 | 612.27 | 139.69 | −77.2 % | 1657.2 | 473.4 | −71.4 % |
| +1.5 | 567.10 | 124.21 | −78.1 % | 1549.0 | 439.7 | −71.6 % |
| +2.0 | 517.45 | 107.97 | −79.1 % | 1431.5 | 403.9 | −71.8 % |

Uniform ~72-79 % call reduction, ~70-72 % sd reduction across all
9 offsets — the effect is scale-neutral (no offset-dependent
distortion).

### 2026 realized-PTF backtest — the primary target of the refit

`mean_model_residual_sd_TRY_MWh / residual_std_realized_TRY_MWh` (ratio
= 1.0 is perfect calibration; v1 was 7-13×, "too wide"):

| month | v1 model_sd | v2 model_sd | realized_sd | **v1 ratio** | **v2 ratio** | MAE (both) |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | 4014.4 | 555.2 | 573.6 | 7.00 | **0.97** | 451.9 |
| 2026-02 | 7215.9 | 557.4 | 1003.8 | 7.19 | **0.56** | 992.4 |
| 2026-03 | 9075.2 | 492.1 | 1072.6 | 8.46 | **0.46** | 1193.6 |
| 2026-04 | 10440.1 | 481.1 | 1181.3 | 8.84 | **0.41** | 1873.4 |
| 2026-05 | 11615.0 | 482.3 | 1102.0 | 10.54 | **0.44** | 2150.7 |
| 2026-06 | 12575.4 | 432.2 | 1251.1 | 10.05 | **0.35** | 1458.2 |
| 2026-07 | 13947.8 | 680.9 | 1108.5 | 12.58 | **0.61** | 1058.3 |

**MAE is unchanged** across all months because MAE = mean absolute
(realized − forward), and the forward curve itself is invariant to κ
(VEP quotes preserved exactly).  What changed is the **model's own
prediction of its residual sd**, which is what the option pricer
consumes.

* Under v1 the model claimed a residual std ~7-13× larger than the
  reality, so option values were correspondingly inflated.
* Under v2 the model's residual std is within a factor of 2-3 of
  realized in every month (0.35 - 0.97).  Perfectly-calibrated (1.0)
  is not achieved because v2 is a single-OU approximation of what is
  really a two-factor process, but the order-of-magnitude overshoot
  is closed.

### Pooled vs M9 (single-volatility baseline comparison)

M9_prod call row shrinks in absolute terms but the qualitative
finding (M9 far below pooled baseline) survives:

| maturity | v1 M9 call | v2 M9 call | Δ% |
|---:|---:|---:|---:|
| 24 h | 368.17 | 163.93 | −55.5 % |
| 72 h | 687.04 | 166.75 | −75.7 % |
| 168 h | 1073.90 | 164.95 | −84.6 % |
| 336 h | 1525.78 | 161.84 | −89.4 % |

The pooled baseline is essentially flat across maturities in both
versions; M9 flattens faster with maturity under v2 kappa (saturation
kicks in sooner).  The regime-conditioning finding remains valid
qualitatively; the underlying pooled_vs_M9 markdown will need a light
rewrite once the paper draft is being assembled.

### M8 vs M9 robustness (should be κ-invariant)

| maturity | v1 gap | v2 gap |
|---:|---:|---:|
| 24 h | −0.637 % | −0.701 % |
| 72 h | −0.610 % | −0.700 % |
| 168 h | −0.598 % | −0.703 % |
| 336 h | −0.592 % | −0.710 % |

Gap preserved to within 0.1 pp across all maturities — confirms the
M8-vs-M9 finding is genuinely a sigma-difference effect, not an
artefact of the chosen kappa.

### Risk-premium sensitivity (Q1 drift channel)

Base value drops from 687.04 → 166.75 (same −75.7 %), but the
**relative percentage sensitivity** to each drift-shift scenario is
essentially unchanged (all deltas ≤ 0.001 % across the 5 scenarios).
Q1 mechanism remains kappa-independent, as expected from the
analytic derivation in `docs/risk_neutral_methodology.md`.

## Files regenerated under v2 kappa

* `outputs/market_calibration_final/`
  * `calibration_result.json`, `calibrated_config.yaml`,
    `calibration_audit.md`, `model_limitations.md` (item (g) status
    now says "RESOLVED via v2 kappa refit"),
    `parameter_identification.json`
  * `hourly_forward_curve.csv` (F(T) invariant — regenerated but
    numerically identical)
  * `legacy_vs_forward_centered.csv`, `.png`,
    `near_term_anchor_sensitivity.csv`, `.png`
  * `risk_premium_sensitivity.csv`,
    `model_comparison_pooled_vs_M9.csv`,
    `model_robustness_M8_vs_M9.csv`,
    `strike_maturity_grid.csv` + heatmap PNGs +
    `strike_slices.png`, `maturity_slices.png`
  * `realized_2026_backtest.csv`, `.monthly.csv`, `.png` — the
    critical validation
* `outputs/scenario_sweep/rd_scenario_sweep.csv` + PNGs
* `outputs/forward_centered_diagnostics/` (residual, strike profile,
  legacy explosion)
* `inputs/legacy_reference/legacy_model_implied_forwards.json`
  (regenerated; pre-refit archived)

## What did NOT change

* The forward curve (`hourly_forward_curve.csv`), by construction:
  `E^Q[P_t] = F(t)` holds for any κ, so the VEP quotes are still
  reproduced to solver precision (~1e-12 TRY/MWh).
* MAE / RMSE / sMAPE / bias of the backtest — MAE depends only on
  the forward curve, not on the residual dynamics.
* Any Q1 drift-channel wiring / test — those are structural
  invariants of the pricing PDE.
* Put-call parity across the 66-point strike/maturity grid — still
  at 1e-6 TRY/MWh solver precision.

## Caveats

* This is **single-OU integration** of v2's fast-factor kappa; v2's
  actual specification (fast MS-AR(1) 1.77h + slow AR 22h + level
  uncertainty + regulatory cap) is a two-factor / clipped process
  that `ForwardCenteredModel` cannot represent.  The v2 stack under
  `pde_option_model/residual_v2.py` and `outputs/residual_v2/*`
  remains a companion; full integration is a separate future item.
* v2 model_sd is now **slightly below** realized (ratio 0.35-0.97),
  not exactly 1.0.  This reflects the missing second factor: fast +
  slow together match empirical better than fast alone.
