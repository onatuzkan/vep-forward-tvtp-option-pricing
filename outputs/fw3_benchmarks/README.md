# FW3 benchmark model comparison

Closed-form benchmark option prices (Black-76, Bachelier, Lucia-Schwartz 2002) evaluated on the SAME contracts, F(T) and discount factor as the accepted PDE model.  See `docs/fw3_benchmark_methodology.md` for the full methodology.

## Setup

* Valuation: 2025-12-31T20:00:00+00:00
* Discount: `r_annual = 0.40`  (`r_per_hour = 4.566210e-05`)
* Contracts: 66 rows from `outputs/market_calibration_final/strike_maturity_grid.csv`
* Real repo data only: PTF history from `inputs/historical/ptf_raw/`; realized PTF from `inputs/market/realized_ptf_2026.csv`.  No synthetic values.

## Historical volatility inputs (no look-ahead)

* Window: last 8760 hours (~365 days) ending 2025-12-31 20:00:00+00:00; source `inputs/historical/ptf_raw/ptf_2019..2025.csv`
* Price floor: 50 TRY/MWh (guards log-returns against near-zero clearings)

**Two sampling frequencies computed side by side:**
* HOURLY log-return sample: 8655 obs -> `sigma_log_hourly = 0.3060/sqrt(h)` (annualised ~28.64). Dominated by the intraday demand cycle; NOT the primary input for a multi-day option.
* DAILY (calendar-day mean) log-return sample: 364 obs -> `sigma_log_daily = 0.1718/sqrt(d)`, rescaled to per-sqrt(hour) via `/sqrt(24)` = 0.0351. Annualised = 3.28. PRIMARY input for B1 (Black-76).
* DAILY abs-return -> `sigma_abs_daily = 382.47 TRY/MWh/sqrt(d)`, per-sqrt(hour) = 78.07. PRIMARY input for B2 (Bachelier).

## Lucia-Schwartz single-factor OU inputs (from the yaml, NOT re-fitted)

* `kappa_per_hour = 0.078394` (half-life 8.84 h) -- v2 reconciled kappa in the accepted yaml
* Pooled `sigma_y = 0.075923` (M9 stationary occupancy weights (0.3254, 0.6746))
* Mapped to price space at each contract's F(T) via the same delta-method transfer `sigma_price = sigma_y * sqrt(F^2 + scale_P^2)` used by `ResidualSpec.sigma_price`

## Grid comparison summary (ATM K=3000)

|   maturity_h |     F_T |   model_call |   B1_black76_call_hist |   B2_bachelier_call_hist |   B3_lucia_schwartz_call |   B1_vs_model_call_pct |   B2_vs_model_call_pct |   B3_vs_model_call_pct |
|-------------:|--------:|-------------:|-----------------------:|-------------------------:|-------------------------:|-----------------------:|-----------------------:|-----------------------:|
|           24 | 2917.24 |       163.93 |                 163.68 |                   114.63 |                   182.45 |                  -0.15 |                 -30.07 |                  11.3  |
|           48 | 2916.7  |       167.12 |                 245.82 |                   176.3  |                   184.5  |                  47.1  |                   5.5  |                  10.4  |
|           72 | 2916.16 |       166.75 |                 308.58 |                   223.74 |                   184.07 |                  85.06 |                  34.18 |                  10.39 |
|          168 | 2913.99 |       164.95 |                 486.06 |                   359.38 |                   182.16 |                 194.67 |                 117.87 |                  10.44 |
|          336 | 2910.21 |       161.84 |                 690.37 |                   519.11 |                   178.86 |                 326.56 |                 220.75 |                  10.51 |
|          720 | 2901.51 |       154.91 |                 986.78 |                   761.94 |                   171.47 |                 537.01 |                 391.87 |                  10.69 |


Positive `%` = benchmark above model.  Deep-OTM and long-dated rows can appear extreme in % terms because the model call value is small in the denominator; see `grid_comparison.csv` for absolute levels.

## ATM implied vol term structure (extracted from the MODEL call, per maturity)

|   maturity_h |     F_T |   K_ATM |   iv_black76_annual_ATM |   iv_bachelier_annual_ATM_TRY_MWh |
|-------------:|--------:|--------:|------------------------:|----------------------------------:|
|           24 | 2917.24 |    3000 |                  3.2869 |                           9712.12 |
|           48 | 2916.7  |    3000 |                  2.3665 |                           6991.48 |
|           72 | 2916.16 |    3000 |                  1.9329 |                           5709.89 |
|          168 | 2913.99 |    3000 |                  1.265  |                           3735.5  |
|          336 | 2910.21 |    3000 |                  0.894  |                           2638.34 |
|          720 | 2901.51 |    3000 |                  0.61   |                           1797.5  |


Both columns are annualised (multiplied by sqrt(8760)) so readers can compare against literature vols directly.

## Implied Black-76 vol smile (annualised, per maturity)

|   strike |   24.0 |   48.0 |   72.0 |   168.0 |   336.0 |   720.0 |
|---------:|-------:|-------:|-------:|--------:|--------:|--------:|
|     2000 | 4.3344 | 3.1009 | 2.5323 |  1.657  |  1.1707 |  0.7982 |
|     2200 | 4.0585 | 2.9064 | 2.3735 |  1.553  |  1.0972 |  0.748  |
|     2400 | 3.809  | 2.7314 | 2.2306 |  1.4595 |  1.0311 |  0.7028 |
|     2600 | 3.5884 | 2.5779 | 2.1054 |  1.3776 |  0.9732 |  0.6634 |
|     2800 | 3.4073 | 2.4525 | 2.003  |  1.3107 |  0.9261 |  0.6315 |
|     3000 | 3.2869 | 2.3665 | 1.9329 |  1.265  |  0.894  |  0.61   |
|     3200 | 3.2241 | 2.3172 | 1.8926 |  1.2388 |  0.8757 |  0.5978 |
|     3400 | 3.1888 | 2.2875 | 1.8683 |  1.223  |  0.8646 |  0.5903 |
|     3600 | 3.1634 | 2.266  | 1.8507 |  1.2115 |  0.8565 |  0.5848 |
|     3800 | 3.14   | 2.2469 | 1.8351 |  1.2012 |  0.8492 |  0.5799 |
|     4000 | 3.1155 | 2.2277 | 1.8193 |  1.1909 |  0.8419 |  0.5748 |


The columns are maturities in hours.  If the model were pure Black-76, every row/column combination would show the same number; departures encode the skew induced by the regime-switching mixture (see methodology).

## Put-call parity on every benchmark

* max |C - P - e^-rT(F-K)| across all 66 rows: B1 = 4.55e-13, B2 = 2.27e-13, B3 = 2.27e-13


## Realized 2026 discounted-payoff backtest

Each contract has EXACTLY ONE realized draw of P_T; the per-contract error is single-path noise.  Only the cross-contract aggregate is interpretable.  66 contracts matched (contracts whose maturity hour fell outside the realized CSV window are dropped).

| model             |   n_contracts |   mean_bias |   std_error_of_bias |    MAE |   RMSE |
|:------------------|--------------:|------------:|--------------------:|-------:|-------:|
| Model_PDE         |            66 |      102.03 |               25.27 | 151.37 | 227.85 |
| B1_Black76        |            66 |      361.28 |               40.96 | 369.19 | 489.47 |
| B2_Bachelier      |            66 |      259.91 |               35.56 | 272.47 | 386.95 |
| B3_Lucia_Schwartz |            66 |      109.43 |               25.37 | 156.65 | 231.94 |


Interpretation: with a single realized draw per contract, `mean_bias` is dominated by the systematic over-forecast of the VEP forward curve documented in `realized_2026_backtest.md` (F2.9).  MAE ranking across the four pricers is more informative than absolute values; small differences within a single std_error should not be over-interpreted.
