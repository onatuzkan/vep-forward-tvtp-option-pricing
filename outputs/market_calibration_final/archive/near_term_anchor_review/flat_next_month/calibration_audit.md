# Calibration audit — forward-centered model

**Result label:** VEP-forward-curve anchored option prices

> This is *not* “Fully market-calibrated option prices”. Only the price level is anchored to the VEP curve; the volatility and regime-transition risk premia remain unidentified because no option premia were supplied.

## 1. Outcome

| field | value |
|---|---|
| `optimizer_success` | `True` |
| `calibration_accepted` | `True` |
| valuation date | 2025-12-31T20:00:00+00:00 |
| quote source | EPIAS_VEP_daily_reference_prices |
| model type | forward_centered |
| curve mode | smooth_constrained |
| monthly RMSE | 1.982198e-12 TRY/MWh |
| monthly MAE | 1.818989e-12 TRY/MWh |
| monthly MAPE | 6.867875e-14 % |
| max abs monthly error | 3.183231e-12 TRY/MWh |

These two flags are deliberately independent: a converged optimizer with an economically absurd fit is a **failed** calibration.

## 2. Acceptance checks

| check | passed | detail |
|---|---|---|
| `forward_curve_solved` | PASS | smooth_constrained: 5089 hourly nodes |
| `monthly_delivery_average_matches_quote` | PASS | max abs monthly error 3.183e-12 TRY/MWh (limit 0.1) |
| `monthly_MAPE_within_tolerance` | PASS | MAPE 6.868e-14% (limit 1.0%) |
| `all_values_finite` | PASS | 4/4 finiteness checks passed |
| `no_implausible_forward_magnitude` | PASS | largest |value| 3608.92 TRY/MWh (limit 1e+05) |
| `monthly_average_uses_true_utc_delivery_hours` | PASS | delivery-hour counts: EBM0226=672, EBM0326=744, EBM0426=720, EBM0526=744, EBM0626=720, EBM0726=744 |
| `unquoted_months_flagged_as_extrapolation` | PASS | unquoted months ['2026-01']; flagged extrapolated ['2025-12', '2026-01'] |

## 3. Monthly delivery-average fit

Averages are formed over the true UTC delivery hours of each Turkish local delivery month.

| contract_name   | delivery_start_utc        | delivery_end_utc          |   number_of_delivery_hours |   market_forward_TRY_MWh |   model_average_TRY_MWh |   residual_TRY_MWh |   relative_error_pct | acceptance_passed   |
|:----------------|:--------------------------|:--------------------------|---------------------------:|-------------------------:|------------------------:|-------------------:|---------------------:|:--------------------|
| EBM0226         | 2026-01-31T21:00:00+00:00 | 2026-02-28T21:00:00+00:00 |                        672 |                  2900.99 |                 2900.99 |        9.09495e-13 |          3.13512e-14 | True                |
| EBM0326         | 2026-02-28T21:00:00+00:00 | 2026-03-31T21:00:00+00:00 |                        744 |                  2556.31 |                 2556.31 |        1.81899e-12 |          7.11568e-14 | True                |
| EBM0426         | 2026-03-31T21:00:00+00:00 | 2026-04-30T21:00:00+00:00 |                        720 |                  2500.66 |                 2500.66 |        2.27374e-12 |          9.09255e-14 | True                |
| EBM0526         | 2026-04-30T21:00:00+00:00 | 2026-05-31T21:00:00+00:00 |                        744 |                  2506.25 |                 2506.25 |       -3.18323e-12 |         -1.27012e-13 | True                |
| EBM0626         | 2026-05-31T21:00:00+00:00 | 2026-06-30T21:00:00+00:00 |                        720 |                  2245.33 |                 2245.33 |       -9.09495e-13 |         -4.05061e-14 | True                |
| EBM0726         | 2026-06-30T21:00:00+00:00 | 2026-07-31T21:00:00+00:00 |                        744 |                  3558.19 |                 3558.19 |        1.81899e-12 |          5.11212e-14 | True                |

## 4. Expected spot at the reporting horizons

| horizon | E[P_t] (TRY/MWh) | delivery month | directly constrained by a VEP quote? |
|---|---|---|---|
| 72 h | 2900.99 | 2026-01 | **no — near-term anchored** |
| 168 h | 2900.99 | 2026-01 | **no — near-term anchored** |
| 336 h | 2900.99 | 2026-01 | **no — near-term anchored** |
| 720 h | 2900.96 | 2026-01 | **no — near-term anchored** |

## 5. Legacy vs forward-centered

|   horizon_hours |   forward_centered_TRY_MWh |   legacy_analytic_TRY_MWh |   legacy_reported_TRY_MWh |
|----------------:|---------------------------:|--------------------------:|--------------------------:|
|              72 |                    2900.99 |                   5004.48 |                      5075 |
|             168 |                    2900.99 |                   9749.46 |                      9946 |
|             336 |                    2900.99 |                  27432.4  |                     27642 |
|             720 |                    2900.96 |                 169063    |                    167837 |

The legacy column is the analytic sinh-Gaussian moment `E[P] = scale_P · exp(v/2) · sinh(m)`; the reported column is the output actually observed from the legacy run.

## 7. Warnings

- valuation falls inside delivery month(s) ['2025-12']; only the hours from valuation onward are priced and no monthly baseload average is formed for them
- delivery months inside the pricing horizon with NO observed quote: ['2026-01'] -> near-term anchored, not directly constrained by an observed VEP quote
- months ['2026-01'] have no observed VEP quote: near-term anchored, not directly constrained by an observed January VEP quote
- 4 of 4 reported horizons (72h, 168h, 336h, 720h) fall in a month WITHOUT an observed VEP quote and are near-term anchored, not directly constrained by market data
- legacy_asinh_ou: E[P_t] = scale_P * exp(v(t)/2) * sinh(m(t)) grows exponentially in the variance v(t) = sigma^2 (1 - e^{-2 kappa t}) / (2 kappa). With the fitted near-unit-root kappa and the stress-regime sigma this makes long-horizon expected prices economically meaningless. Use the legacy mode for short-horizon benchmarking and diagnostics only.

## 8. Run notes

- **quotes_file**: inputs/market/vep_monthly_quotes.csv
- **frozen_parameters_file**: inputs/historical/m2_frozen_parameters.yaml
- **curve_mode**: smooth_constrained
- **january_anchor_mode**: flat_next_month
