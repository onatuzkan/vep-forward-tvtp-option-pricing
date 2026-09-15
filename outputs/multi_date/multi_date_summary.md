# Multi-date forward-curve backtest

**This study evaluates the forward curve only. No option is priced anywhere in it.**

The TVTP, kappa and sigma parameters of the model were estimated on data running to the end of 2025. Applying them at a 2022 or 2023 valuation date would be look-ahead bias, and any option price produced that way would be contaminated. The forward-curve construction uses only the VEP quotations published on the valuation date and the spot PTF of that hour; it contains no historical parameter, so replaying it at an earlier date is leak-free. That is the whole reason this backtest exists in this restricted form, and it is the reason its conclusions say nothing directly about option-price accuracy.

* valuation dates attempted: 7
* valuation dates with a usable VEP strip: 7
* delivery horizon per date: 1..7 months ahead
* curve mode: `smooth_constrained`, anchor rule: `spot_to_next_linear`, smoothness 1.0, level 0.0001
* usable (valuation date x delivery month) observations: 49

## 0. Is the downloaded series the series the manuscript uses?

The production quote file `inputs/market/vep_monthly_quotes.csv` (source `EPIAS_VEP_daily_reference_price`, the 2025-12-31 strip the single-date result is built on) was compared contract by contract with the downloaded EPİAŞ `vep-ggf` (VEP Günlük Gösterge Fiyatı) rows of 2025-12-31:

| contract | production file | downloaded GGF | abs diff | matching decimals |
|---|---|---|---|---|
| EBM0226 | 2900.99 | 2900.99 | 0.00e+00 | 12 |
| EBM0326 | 2556.31 | 2556.31 | 0.00e+00 | 12 |
| EBM0426 | 2500.66 | 2500.66 | 0.00e+00 | 12 |
| EBM0526 | 2506.25 | 2506.25 | 0.00e+00 | 12 |
| EBM0626 | 2245.33 | 2245.33 | 0.00e+00 | 12 |
| EBM0726 | 3558.19 | 3558.19 | 0.00e+00 | 12 |

Maximum absolute difference 0.00e+00 TRY/MWh against a stop tolerance of 1.0 TRY/MWh. The two names label the same EPİAŞ series; the multi-date analysis is comparable with the single-date result. (`scripts/backtest/build_vep_quote_sets.py` exits with code 2 and writes nothing if this check fails.)

## Equivalence with the production pipeline

Rebuilding the 2025-12-31 curve with this script and comparing it hour by hour with the published `outputs/market_calibration_final/hourly_forward_curve.csv` gives a maximum absolute difference of **4.547e-13 TRY/MWh** over 5089 hours. The curve built here is the production curve.

## 1. Coverage: which dates carried a strip, and did the anchor fire?

| valuation date | weekday | GGF quotation date (staleness) | contracts in file | spot TRY/MWh | quoted months in horizon | nearest month quoted? | anchored months | months dropped at an interior gap | max abs monthly constraint error |
|---|---|---|---|---|---|---|---|---|---|
| 2022-12-31 | Sat | 2022-12-30 (1 d) | 6 | 3475.27 | 6 (2023-02, 2023-03, 2023-04, 2023-05, 2023-06, 2023-07) | **no — anchor fired** | 2023-01 | — | 4.09e-12 |
| 2023-06-30 | Fri | 2023-06-26 (4 d) | 6 | 1932.00 | 6 (2023-08, 2023-09, 2023-10, 2023-11, 2023-12, 2024-01) | **no — anchor fired** | 2023-07 | — | 2.73e-12 |
| 2023-12-31 | Sun | 2023-12-29 (2 d) | 6 | 1345.15 | 6 (2024-02, 2024-03, 2024-04, 2024-05, 2024-06, 2024-07) | **no — anchor fired** | 2024-01 | — | 2.73e-12 |
| 2024-06-30 | Sun | 2024-06-28 (2 d) | 6 | 2424.99 | 6 (2024-08, 2024-09, 2024-10, 2024-11, 2024-12, 2025-01) | **no — anchor fired** | 2024-07 | — | 4.55e-12 |
| 2024-12-31 | Tue | 2024-12-31 (0 d) | 6 | 1984.00 | 6 (2025-02, 2025-03, 2025-04, 2025-05, 2025-06, 2025-07) | **no — anchor fired** | 2025-01 | — | 5.46e-12 |
| 2025-06-30 | Mon | 2025-06-30 (0 d) | 6 | 3155.00 | 6 (2025-08, 2025-09, 2025-10, 2025-11, 2025-12, 2026-01) | **no — anchor fired** | 2025-07 | — | 2.73e-12 |
| 2025-12-31 | Wed | 2025-12-31 (0 d) | 6 | 2917.78 | 6 (2026-02, 2026-03, 2026-04, 2026-05, 2026-06, 2026-07) | **no — anchor fired** | 2026-01 | — | 3.64e-12 |

GGF is published on business days only. 4 of the 7 valuation dates fall on a weekend or public holiday; for them the quote set is the last GGF published *before* the valuation date (2022-12-31 ← 2022-12-30, 2023-06-30 ← 2023-06-26, 2023-12-31 ← 2023-12-29, 2024-06-30 ← 2024-06-28). That is the information a market participant held at the end of the valuation day; no later publication is used. The spot is still the 23:00 PTF of the valuation day itself. Under a strict same-day rule these 4 dates would have been dropped and the study would rest on 3 dates.

Every quoted month is a hard equality constraint of the KKT system; the largest monthly reproduction error over all 7 curves is 5.46e-12 TRY/MWh — all at solver precision.

Minimum strip length for inclusion: 3 quoted months. No date was excluded on that rule.

The near-term anchor rule actually fired at **7 of 7** valuation dates, producing 7 anchored (valuation date x delivery month) observations against 42 quote-constrained ones.

## 2. Distribution of the bias (realized − forward)

| statistic | TRY/MWh |
|---|---|
| observations | 49 |
| mean | -603.40 |
| median | -353.70 |
| standard deviation | 774.74 |
| minimum | -2,105.63 |
| 25th percentile | -1,299.88 |
| 75th percentile | 144.40 |
| maximum | 489.39 |

The single-date figure this study exists to contextualise — the **-1,019.04 TRY/MWh** mean bias of the 2025-12-31 valuation over 2026 — sits at the **32.7th percentile** of the pooled bias distribution: 16 of 49 individual (date x month) biases are more negative than it.

That published figure is the *hour-weighted* mean of the seven 2026 monthly biases. Recomputed here from the same curve it is -1,019.04 TRY/MWh hour-weighted and -1,018.85 TRY/MWh unweighted; the tables below use the unweighted convention throughout.

Compared like with like — against the 7 per-valuation-date mean biases rather than individual months — it sits at the **28.6th percentile**, in a date-mean distribution running from -1,464.63 to 289.76 TRY/MWh (median -756.89).

Mean bias by valuation date:

| valuation date | n | mean bias TRY/MWh | median bias | hour-weighted mean | mean relative bias % |
|---|---|---|---|---|---|
| 2022-12-31 | 7 | -1,464.63 | -1,545.59 | -1,463.92 | -39.8 |
| 2023-06-30 | 7 | -1,250.01 | -1,299.88 | -1,249.93 | -36.5 |
| 2023-12-31 | 7 | -756.89 | -813.27 | -750.70 | -24.5 |
| 2024-06-30 | 7 | -174.47 | -214.48 | -173.78 | -6.2 |
| 2024-12-31 | 7 | 151.28 | 179.60 | 151.11 | +7.2 |
| 2025-06-30 | 7 | 289.76 | 307.18 | 290.08 | +11.4 |
| 2025-12-31 | 7 | -1,018.85 | -935.99 | -1,019.04 | -39.1 |

Relative view (bias / forward, %), because the TRY price level is not constant across the sample: pooled mean -18.2 %, median -13.8 %, sd 25.0 %, quartiles -40.1 % / +6.2 %, range -76.4 % to +24.9 %. The 2025-12-31 date mean is -39.1 %, which is the **28.6th percentile** of the 7 date means (2 dates more negative) and the 26.5th percentile of the pooled monthly relative biases.

**Verdict on the manuscript narrative.** 2026 is **not an outlier**: 2 of the other 6 valuation dates were more negative in TRY/MWh and 2 in relative terms, and the 2026 date mean lies inside the interquartile range of the date means. It is a large negative bias of a kind the VEP strip has produced before (2022-12-31, 2023-06-30), not a one-off. The sign of the date mean is not stable either: the biases run from -1,465 to 290 TRY/MWh; realised came in *above* the strip at 2 valuation dates (2024-12-31, 2025-06-30) and below it at 5 (2022-12-31, 2023-06-30, 2023-12-31, 2024-06-30, 2025-12-31). The VEP monthly strip has therefore been a **biased and unstable** predictor of the realised monthly baseload over 2023-26, not a consistently over-priced one.

## 3. Is the bias systematic?

| sample | valuation dates | n obs | mean bias TRY/MWh | sd | t | p | share negative |
|---|---|---|---|---|---|---|---|
| all dates, pooled months | 7 | 49 | -603.40 | 774.74 | -5.452 | 1.703e-06 | 71.4 % |
| all dates, one mean per date | 7 | 7 | -603.40 | 696.09 | -2.293 | 0.06165 | — |
| non-overlapping subset from 2022-12-31, pooled months | 4 | 28 | -772.27 | 788.15 | -5.185 | 1.854e-05 | 78.6 % |
| non-overlapping subset from 2022-12-31, one mean per date | 4 | 4 | -772.27 | 681.50 | -2.266 | 0.1083 | — |
| non-overlapping subset from 2023-06-30, pooled months | 3 | 21 | -378.24 | 713.30 | -2.430 | 0.02464 | 61.9 % |
| non-overlapping subset from 2023-06-30, one mean per date | 3 | 3 | -378.24 | 789.85 | -0.829 | 0.4941 | — |

* subset from 2022-12-31: 2022-12-31 → 2023-01..2023-07, 2023-12-31 → 2024-01..2024-07, 2024-12-31 → 2025-01..2025-07, 2025-12-31 → 2026-01..2026-07
* subset from 2023-06-30: 2023-06-30 → 2023-07..2024-01, 2024-06-30 → 2024-07..2025-01, 2025-06-30 → 2025-07..2026-01

**Read every p-value with suspicion.** The pooled observations are NOT independent: the same delivery month is seen from several valuation dates, and the months of one valuation date share a single market state. Both p-values are therefore anti-conservative and must not be read as calibrated significance levels. The date-level aggregate removes the within-date dependence only, not the overlap between dates. 6 delivery months appear from more than one valuation date in the full sample. The non-overlapping subsets remove the shared-delivery-month overlap and the date-level rows remove the within-date dependence, but consecutive delivery months of one curve still share a single market state, and the price level of one half-year is not independent of the next; none of these tests is a calibrated significance level.

## 4. Error where the anchor rule fired

| statistic | anchored months, relative bias % | anchored months, bias TRY/MWh |
|---|---|---|
| observations | 7 | 7 |
| mean | -3.51 | -104.20 |
| median | -3.27 | -87.59 |
| standard deviation | 11.74 | 284.84 |
| minimum | -21.15 | -530.43 |
| 25th percentile | -9.44 | -266.68 |
| 75th percentile | 1.26 | 36.05 |
| maximum | 16.21 | 349.86 |

The manuscript quotes a ±20 % band for the near-term anchor. Empirically **85.7 %** of the anchored months fall inside ±20 %, the realised spread runs from -21.2 % to 16.2 %, the interquartile range is 10.7 percentage points, the standard deviation is 11.7 % and the RMS relative error is 11.4 %, on 7 observations.

**Verdict:** a ±20 % band read as a uniform prior has standard deviation 11.5 %; the empirical standard deviation is 11.7 %, so in *spread* the empirical distribution is **comparable to** the band (difference 0.2 percentage points on 7 observations). The empirical distribution is **not narrower** than the band: 1 of 7 anchored months fall outside ±20 % (widest 21.2 %), so the band does not contain the tail. Its centre is -3.5 % (median -3.3 %): the ramp over-prices the anchored month on average, and a band symmetric around zero ignores that shift. Read the ±20 % figure as an approximately right *scale* for the anchor uncertainty, not as a containment bound.

For comparison, quote-constrained months over the same sample have mean relative bias -20.67 % (median -25.57 %, sd 25.89 %).

## 5. Error by delivery horizon

|   horizon_months |   n |   mean_bias_TRY_MWh |   median_bias_TRY_MWh |   std_bias_TRY_MWh |   min_bias_TRY_MWh |   q25_bias_TRY_MWh |   q75_bias_TRY_MWh |   max_bias_TRY_MWh |   MAE_TRY_MWh |   sMAPE_pct |
|-----------------:|----:|--------------------:|----------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|--------------:|------------:|
|                1 |   7 |            -104.201 |              -87.5946 |            284.835 |           -530.426 |           -266.679 |            36.0544 |            349.86  |       228.899 |      9.3235 |
|                2 |   7 |            -537.476 |             -822.795  |            593.087 |          -1171.7   |           -947.809 |          -104.651  |            337.089 |       675.044 |     24.6533 |
|                3 |   7 |            -694.48  |             -621.67   |            703.946 |          -1827.63  |          -1076.89  |          -282.728  |            307.182 |       782.246 |     30.2465 |
|                4 |   7 |            -896.977 |            -1407.56   |           1074.41  |          -2105.63  |          -1697.9   |            70.3816 |            489.385 |      1120.84  |     45.9022 |
|                5 |   7 |            -860.788 |            -1299.88   |            874.364 |          -1915.35  |          -1457.15  |           -50.6418 |            205.298 |       970.759 |     43.619  |
|                6 |   7 |            -751.681 |             -813.271  |            803.952 |          -1852.54  |          -1284.85  |          -219.104  |            411.962 |       869.385 |     35.7223 |
|                7 |   7 |            -378.196 |              177.316  |            860.265 |          -1488.59  |          -1157.25  |           296.142  |            386.121 |       708.402 |     25.7895 |

Expectation: error grows with horizon. Observed MAE by horizon (1 → 7): 1: 229, 2: 675, 3: 782, 4: 1,121, 5: 971, 6: 869, 7: 708 TRY/MWh. MAE is **not** monotone in the horizon; Spearman rank correlation between horizon and absolute error is +0.253 (p = 0.08, same dependence caveat as section 3), and between horizon and sMAPE +0.237 (p = 0.101).

Two things to read from that profile. First, horizon 1 is *always* the anchored month in this sample (the front contract is delisted before month end), so the horizon-1 column is the spot-to-next ramp, not a VEP quote; its MAE of 229 TRY/MWh is the smallest of all horizons. Second, the error rises from horizon 1 to a peak at horizon 4 (1,121 TRY/MWh) and then falls back toward horizon 7 (708 TRY/MWh). The decline at the long end should not be read as a forecasting virtue of the far contract: with 7 horizons and 7 valuation dates six months apart, horizon 7 is always the same two calendar months (July for the December dates, January for the June dates), so the long-end column is as much a calendar-month effect as a maturity effect. The expectation 'error grows with horizon' holds up to horizon 4 and not beyond, and the rank correlation is weak.

## 6. Honesty check

* **No option was priced in this study.** Only the forward curve was evaluated. The reason is stated at the top: the residual-process parameters are fitted to data through 2025 and would be look-ahead bias at every earlier valuation date.
* `run_pde.py calibrate-market` was not invoked per date. It reads the valuation timestamp and the spot from the frozen 2025-12-31 parameter file and, by default, prices a 72 h option for its sensitivity table. The curve itself is built by the same `build_forward_curve` call with the same configuration, and section 'Equivalence with the production pipeline' above measures the difference against the published curve.
* Realised PTF comes from the local EPİAŞ archive only (`inputs/historical/ptf_raw/`, `inputs/market/realized_ptf_2026.csv`). Delivery months with less than 98% hourly coverage are reported with an empty bias rather than a partial average.
* Every metric here is computed on **monthly baseload averages**: one forward number and one realised number per delivery month. The sMAPE column is therefore not comparable with the hourly sMAPE in `realized_2026_backtest.md`, which averages the error hour by hour and is much larger because individual hours swing far more than a monthly mean.
* Every requested valuation date produced a usable strip.
* The t-tests in section 3 assume independent observations, which these are not: the same delivery month is seen from more than one valuation date and consecutive delivery months of one curve share a market state. They are reported because they were asked for, with the dependence stated rather than corrected; the non-overlapping subsets are shown beside the pooled result so the reader can see how much of the pooled t-statistic is overlap.
* Contract counts per date are in section 1; every retained date carries at least 3 quoted months. No date was dropped for too few quotes.
* 4 valuation dates are non-business days; their quote set is the last GGF published before the date (1–4 days stale), never after it. This is the only substitution made anywhere in the study and it is flagged per row in `multi_date_backtest.csv` (`ggf_quotation_date`, `ggf_staleness_days`). The stalest case (2023-06-30, GGF of 2023-06-26, spot of a non-business night) is also an anchored month outside ±20 % (-21.2 %); the reader should weigh that observation accordingly rather than have it removed.
* The downloaded EPİAŞ `vep-ggf` series reproduces the production quote file on 2025-12-31 for all 6 contracts to 0.0e+00 TRY/MWh (section 0); `EPIAS_VEP_daily_reference_price` in the schema and `VEP Günlük Gösterge Fiyatı` in the API are the same series.

