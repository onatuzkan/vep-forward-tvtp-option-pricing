# FW9f section 4 -- Fast tail and quantile validation

**Note on the 2025 sd_TRY = 531 figure and the difference from
`stationary_variance_check.md` (624).**  The 531 value used here is
the observed A3 TRY residual computed with an hour-of-week
climatological shape estimated **from 2025 alone**;
`stationary_variance_check.md` reports 624 with the hour-of-week
shape pooled over **2019-2025**.  Both are legitimate constructions
of the observed sd; production 582.5 sits between them (9.7 pct
above 531, 6.7 pct below 624).  The choice of shape window is what
moves the observed sd, not a change in the residual definition.

Observed 2025 A3 TRY residual: n = 8757, mean = -0.00, std = 531.00

Model-implied stationary residual: single long-path simulation of the MS-AR(1) plus TVTP process at 1 h step; each path n = 2000000 hours after 10000 h burn-in; climatology z_lag cycled hourly; asinh residual mapped to TRY via delta = sqrt(F^2 + s_P^2) = 2931.42 at F = 2917.78 TRY/MWh.

## Quantile / exceedance summary

| source              |       n |    mean |      std |      q01 |       q05 |       q10 |      q25 |    q50 |     q75 |      q90 |      q95 |     q99 |   P_gt_250 |   P_lt_-250 |   P_gt_500 |   P_lt_-500 |   P_gt_750 |   P_lt_-750 |   P_gt_1000 |   P_lt_-1000 |   P_gt_1500 |   P_lt_-1500 |
|:--------------------|--------:|--------:|---------:|---------:|----------:|----------:|---------:|-------:|--------:|---------:|---------:|--------:|-----------:|------------:|-----------:|------------:|-----------:|------------:|------------:|-------------:|------------:|-------------:|
| observed_2025       |    8757 |  -0     |  531.004 | -1714.81 | -1003.62  |  -655.425 | -230.752 | 52.303 | 334.174 |  580.493 |  730.057 | 1105.18 |      0.318 |       0.24  |      0.137 |       0.138 |      0.046 |       0.083 |       0.015 |        0.05  |       0.004 |        0.016 |
| production          | 2000000 |  -1.165 |  581.177 | -1402.63 |  -962.549 |  -735.87  | -373.104 |  0.637 | 369.418 |  731.158 |  958.95  | 1406.05 |      0.322 |       0.323 |      0.184 |       0.186 |      0.095 |       0.096 |       0.044 |        0.044 |       0.007 |        0.007 |
| FW9e_full_A3        | 2000000 | -14.075 |  984.017 | -2958.52 | -1724.39  | -1157.2   | -477.711 | 37.798 | 512.209 | 1033.03  | 1472.99  | 2559.51 |      0.38  |       0.346 |      0.255 |       0.242 |      0.164 |       0.17  |       0.106 |        0.122 |       0.048 |        0.066 |
| FW9f_regime_matched | 2000000 | -50.016 | 1176.38  | -3542.3  | -2141.47  | -1466.3   | -606.905 | 30.404 | 595.258 | 1207.54  | 1720.5   | 2942.22 |      0.395 |       0.374 |      0.286 |       0.282 |      0.199 |       0.213 |       0.136 |        0.162 |       0.067 |        0.097 |


## KS distance + tail quantile deltas vs observed

| source              |   KS_statistic |   KS_pvalue |   q01_obs |   q01_model |   q01_delta_TRY |   q05_obs |   q05_model |   q05_delta_TRY |   q95_obs |   q95_model |   q95_delta_TRY |   q99_obs |   q99_model |   q99_delta_TRY |
|:--------------------|---------------:|------------:|----------:|------------:|----------------:|----------:|------------:|----------------:|----------:|------------:|----------------:|----------:|------------:|----------------:|
| production          |         0.0869 |           0 |   -1714.8 |    -1402.63 |         312.175 |  -1003.62 |    -962.549 |         41.0755 |   730.057 |      958.95 |         228.893 |   1105.18 |     1406.05 |         300.876 |
| FW9e_full_A3        |         0.1246 |           0 |   -1714.8 |    -2958.52 |       -1243.72  |  -1003.62 |   -1724.39  |       -720.77   |   730.057 |     1472.99 |         742.936 |   1105.18 |     2559.51 |        1454.34  |
| FW9f_regime_matched |         0.1587 |           0 |   -1714.8 |    -3542.3  |       -1827.49  |  -1003.62 |   -2141.47  |      -1137.85   |   730.057 |     1720.5  |         990.445 |   1105.18 |     2942.22 |        1837.04  |


## Observed cap/floor exposure

* observed 2025 min residual: **-2703.5 TRY**
* observed 2025 max residual: **2098.5 TRY**
* observed hours with residual > +1500 TRY: 31
* observed hours with residual < -1500 TRY: 142


The observed distribution is truncated on the negative side at approximately -F(monthly_mean) (price floor 0) and on the positive side at approximately 4500 - F(monthly_mean) (price cap 4500 TRY/MWh until 2026-04-04, 5000 after).  The model has no cap or floor, so mismatches at the extreme (1 pct, 99 pct) tails reflect both the model residual dispersion and this asymmetric truncation.  The 25-75 interquartile range, the 5-95 centre-tail band and the KS statistic are the most comparable summaries.
