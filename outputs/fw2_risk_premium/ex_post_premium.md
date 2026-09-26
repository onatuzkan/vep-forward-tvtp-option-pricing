# Ex-post forward premium panel (FW2 §2.1)

**Descriptive**, not an estimator.  Small sample: seven semi-annual VEP snapshots, `n_obs` per horizon in the table below.  With this sample size the sample mean is NOT a consistent estimator of the risk premium; the numbers are a sanity-bounded plausible range for the FW2 sensitivity sweep.

* Look-ahead cut-off: 2025-12-31T20:00:00+00:00 (realised months beyond this cut-off are dropped).
* Panel rows: 35 (valuation x delivery-month pairs).
* Overlapping horizons -> monthly block bootstrap SEs (block_length = 3, n_boot = 2000).

## Ex-post premium by horizon

|   horizon_months |   n_obs |   mean_premium_TRY_MWh |   median_premium_TRY_MWh |   std_premium_TRY_MWh |   block_bootstrap_SE_TRY_MWh |   mean_premium_pct_of_forward |   median_premium_pct_of_forward |
|-----------------:|--------:|-----------------------:|-------------------------:|----------------------:|-----------------------------:|------------------------------:|--------------------------------:|
|                2 |       6 |                 489.92 |                   593.17 |                634.91 |                       280.5  |                         14.1  |                           19.54 |
|                3 |       6 |                 654.23 |                   486.32 |                762.26 |                       297.16 |                         19.25 |                           17.98 |
|                4 |       6 |                 783.21 |                   780.46 |               1129.82 |                       527.71 |                         18.76 |                           25.27 |
|                5 |       6 |                 685.03 |                   790.38 |                811.13 |                       386.17 |                         19.64 |                           24.42 |
|                6 |       6 |                 709.43 |                   560.91 |                872.13 |                       372.35 |                         20.82 |                           19.58 |
|                7 |       5 |                 434.98 |                  -177.32 |                949.9  |                       382.6  |                         10.91 |                           -7.61 |


**Sign convention.**  Positive = forward quoted ABOVE realised, i.e. the buyer PAID a positive risk premium.  Negative = forward under-called realised, i.e. the seller collected a premium.


## Descriptive band for FW2 sensitivity sweep

* Sample mean (all horizons pooled): 631.6 TRY/MWh
* 95th percentile of |premium|: 1835.1 TRY/MWh
* These are the bounds used in FW2 §2.3 to translate into per-hour drift-shift a_i.
