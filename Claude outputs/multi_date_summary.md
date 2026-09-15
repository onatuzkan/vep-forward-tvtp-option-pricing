# Multi-date forward-curve backtest

**This study evaluates the forward curve only. No option is priced anywhere in it.**

The TVTP, kappa and sigma parameters of the model were estimated on data running to the end of 2025. Applying them at a 2022 or 2023 valuation date would be look-ahead bias, and any option price produced that way would be contaminated. The forward-curve construction uses only the VEP quotations published on the valuation date and the spot PTF of that hour; it contains no historical parameter, so replaying it at an earlier date is leak-free. That is the whole reason this backtest exists in this restricted form, and it is the reason its conclusions say nothing directly about option-price accuracy.

* valuation dates attempted: 1
* valuation dates with a usable VEP strip: 1
* delivery horizon per date: 1..7 months ahead
* curve mode: `smooth_constrained`, anchor rule: `spot_to_next_linear`, smoothness 1.0, level 0.0001
* usable (valuation date x delivery month) observations: 7

## Equivalence with the production pipeline

Rebuilding the 2025-12-31 curve with this script and comparing it hour by hour with the published `outputs/market_calibration_final/hourly_forward_curve.csv` gives a maximum absolute difference of **5.055e-07 TRY/MWh** over 5089 hours. The curve built here is the production curve.

## 1. Coverage: which dates carried a strip, and did the anchor fire?

| valuation date | spot TRY/MWh | quoted months in horizon | nearest month quoted? | anchored months | months dropped at an interior gap | max abs monthly constraint error |
|---|---|---|---|---|---|---|
| 2025-12-31 | 2917.78 | 6 (2026-02, 2026-03, 2026-04, 2026-05, 2026-06, 2026-07) | **no — anchor fired** | 2026-01 | — | 9.09e-12 |

The near-term anchor rule actually fired at **1 of 1** valuation dates, producing 1 anchored (valuation date x delivery month) observations against 6 quote-constrained ones.

## 2. Distribution of the bias (realized − forward)

| statistic | TRY/MWh |
|---|---|
| observations | 7 |
| mean | -1,018.85 |
| median | -935.99 |
| standard deviation | 605.27 |
| minimum | -1,915.35 |
| 25th percentile | -1,292.39 |
| 75th percentile | -840.69 |
| maximum | -14.47 |

The single-date figure this study exists to contextualise — the **-1,019.04 TRY/MWh** mean bias of the 2025-12-31 valuation over 2026 — sits at the **28.6th percentile** of the pooled bias distribution: 2 of 7 individual (date x month) biases are more negative than it.

That published figure is the *hour-weighted* mean of the seven 2026 monthly biases. Recomputed here from the same curve it is -1,019.04 TRY/MWh hour-weighted and -1,018.85 TRY/MWh unweighted; the tables below use the unweighted convention throughout.

Compared like with like — against the 1 per-valuation-date mean biases rather than individual months — it sits at the **0.0th percentile**, in a date-mean distribution running from -1,018.85 to -1,018.85 TRY/MWh (median -1,018.85).

Mean bias by valuation date:

| valuation date | n | mean bias | median bias | hour-weighted mean |
|---|---|---|---|---|
| 2025-12-31 | 7 | -1,018.85 | -935.99 | -1,019.04 |

## 3. Is the bias systematic?

Pooled one-sample t-test against zero: mean -1,018.85 TRY/MWh, t = -4.454, p = 0.004313 on n = 7.

Aggregating to one observation per valuation date first: mean of date means -1,018.85 TRY/MWh, t = nan, p = nan on n = 1 dates.

**Read both p-values with suspicion.** The pooled observations are NOT independent: the same delivery month is seen from several valuation dates, and the months of one valuation date share a single market state. Both p-values are therefore anti-conservative and must not be read as calibrated significance levels. The date-level aggregate removes the within-date dependence only, not the overlap between dates. 0 delivery months appear from more than one valuation date. The share of negative biases is 100.0 %.

## 4. Error where the anchor rule fired

| statistic | anchored months, relative bias % | anchored months, bias TRY/MWh |
|---|---|---|
| observations | 1 | 1 |
| mean | -0.50 | -14.47 |
| median | -0.50 | -14.47 |
| standard deviation | nan | nan |
| minimum | -0.50 | -14.47 |
| 25th percentile | -0.50 | -14.47 |
| 75th percentile | -0.50 | -14.47 |
| maximum | -0.50 | -14.47 |

The manuscript quotes a ±20 % band for the near-term anchor. Empirically **100.0 %** of the anchored months fall inside ±20 %, and the realised spread runs from -0.5 % to -0.5 %. Treat this as the empirical width of that band, on 1 observations.

For comparison, quote-constrained months over the same sample have mean relative bias -45.58 % (median -40.69 %, sd 20.49 %).

## 5. Error by delivery horizon

|   horizon_months |   n |   mean_bias_TRY_MWh |   median_bias_TRY_MWh |   std_bias_TRY_MWh |   min_bias_TRY_MWh |   q25_bias_TRY_MWh |   q75_bias_TRY_MWh |   max_bias_TRY_MWh |   MAE_TRY_MWh |   sMAPE_pct |
|-----------------:|----:|--------------------:|----------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|--------------:|------------:|
|                1 |   1 |             -14.473 |               -14.473 |                nan |            -14.473 |            -14.473 |            -14.473 |            -14.473 |        14.473 |      0.4987 |
|                2 |   1 |            -822.795 |              -822.795 |                nan |           -822.795 |           -822.795 |           -822.795 |           -822.795 |       822.795 |     33.0494 |
|                3 |   1 |            -935.986 |              -935.986 |                nan |           -935.986 |           -935.986 |           -935.986 |           -935.986 |       935.986 |     44.8201 |
|                4 |   1 |           -1579.6   |             -1579.6   |                nan |          -1579.6   |          -1579.6   |          -1579.6   |          -1579.6   |      1579.6   |     92.3281 |
|                5 |   1 |           -1915.35  |             -1915.35  |                nan |          -1915.35  |          -1915.35  |          -1915.35  |          -1915.35  |      1915.35  |    123.685  |
|                6 |   1 |           -1005.17  |             -1005.17  |                nan |          -1005.17  |          -1005.17  |          -1005.17  |          -1005.17  |      1005.17  |     57.6773 |
|                7 |   1 |            -858.577 |              -858.577 |                nan |           -858.577 |           -858.577 |           -858.577 |           -858.577 |       858.577 |     27.4402 |

## 6. Honesty check

* **No option was priced in this study.** Only the forward curve was evaluated. The reason is stated at the top: the residual-process parameters are fitted to data through 2025 and would be look-ahead bias at every earlier valuation date.
* `run_pde.py calibrate-market` was not invoked per date. It reads the valuation timestamp and the spot from the frozen 2025-12-31 parameter file and, by default, prices a 72 h option for its sensitivity table. The curve itself is built by the same `build_forward_curve` call with the same configuration, and section 'Equivalence with the production pipeline' above measures the difference against the published curve.
* Realised PTF comes from the local EPİAŞ archive only (`inputs/historical/ptf_raw/`, `inputs/market/realized_ptf_2026.csv`). Delivery months with less than 98% hourly coverage are reported with an empty bias rather than a partial average.
* Every metric here is computed on **monthly baseload averages**: one forward number and one realised number per delivery month. The sMAPE column is therefore not comparable with the hourly sMAPE in `realized_2026_backtest.md`, which averages the error hour by hour and is much larger because individual hours swing far more than a monthly mean.
* Every requested valuation date produced a usable strip.
* The t-tests in section 3 assume independent observations, which these are not. They are reported because they were asked for, with the dependence stated rather than corrected.

