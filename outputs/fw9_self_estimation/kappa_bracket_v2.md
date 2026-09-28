# FW9d §1 + §3 -- kappa bracket rebuilt across residual definitions

Three residual definitions, six seasonal specifications (S0..S5) each; total 18 (residual, spec) pairs.  This makes the FW9b vs FW9c gap explicit: the same seasonal ladder gives very different kappa depending on the residual.

## Residual definitions

* **R1_asinh_raw**: y = asinh(P/282.48).  No anchor subtraction.   Matches FW9b §2 (the raw-asinh single-regime AR that gave 0.017/h).
* **R2_TRY_minus_monthly**: P - mean(P over calendar month).  TRY-space, monthly anchor.  Matches FW9c §1 (the 0.21/h fit).
* **R3_asinh_minus_how_clim**: asinh(P/282.48) minus hour-of-week climatology mean over the training window.  The 'shock-around-anchor' residual on the same asinh scale the shipped forward-centered model uses -- the FW9d preferred definition for reproducing the yaml phi.

## Full ladder

| residual_definition     | spec   |   seasonal_dof |   seasonal_R2 |     phi |   se_phi |   kappa_per_hour |   half_life_hours |   sigma_eps |   n_obs |
|:------------------------|:-------|---------------:|--------------:|--------:|---------:|-----------------:|------------------:|------------:|--------:|
| R1_asinh_raw            | S0     |              0 |       -0      | 0.98059 |  0.00079 |           0.0196 |            35.367 |      0.1921 |   61367 |
| R1_asinh_raw            | S1     |             23 |        0.0144 | 0.98314 |  0.00074 |           0.017  |            40.758 |      0.1778 |   61367 |
| R1_asinh_raw            | S2     |             29 |        0.0239 | 0.9826  |  0.00075 |           0.0175 |            39.496 |      0.1797 |   61367 |
| R1_asinh_raw            | S3     |             40 |        0.051  | 0.98209 |  0.00076 |           0.0181 |            38.362 |      0.1797 |   61367 |
| R1_asinh_raw            | S4     |             44 |        0.0519 | 0.98208 |  0.00076 |           0.0181 |            38.327 |      0.1797 |   61367 |
| R1_asinh_raw            | S5     |            182 |        0.0628 | 0.98341 |  0.00073 |           0.0167 |            41.433 |      0.172  |   61367 |
| R2_TRY_minus_monthly    | S0     |              0 |        0      | 0.80932 |  0.00237 |           0.2116 |             3.276 |    300.193  |   61367 |
| R2_TRY_minus_monthly    | S1     |             23 |        0.1362 | 0.81468 |  0.00234 |           0.205  |             3.382 |    275.461  |   61367 |
| R2_TRY_minus_monthly    | S2     |             29 |        0.1839 | 0.80191 |  0.00241 |           0.2208 |             3.14  |    275.852  |   61367 |
| R2_TRY_minus_monthly    | S3     |             40 |        0.1839 | 0.8019  |  0.00241 |           0.2208 |             3.14  |    275.852  |   61367 |
| R2_TRY_minus_monthly    | S4     |             44 |        0.1875 | 0.80103 |  0.00242 |           0.2219 |             3.124 |    275.778  |   61367 |
| R2_TRY_minus_monthly    | S5     |            182 |        0.2452 | 0.79881 |  0.00243 |           0.2246 |             3.086 |    267.118  |   61367 |
| R3_asinh_minus_how_clim | S0     |              0 |        0      | 0.9839  |  0.00072 |           0.0162 |            42.713 |      0.172  |   61367 |
| R3_asinh_minus_how_clim | S1     |             23 |        0      | 0.9839  |  0.00072 |           0.0162 |            42.713 |      0.172  |   61367 |
| R3_asinh_minus_how_clim | S2     |             29 |        0      | 0.9839  |  0.00072 |           0.0162 |            42.713 |      0.172  |   61367 |
| R3_asinh_minus_how_clim | S3     |             40 |        0.0281 | 0.98342 |  0.00073 |           0.0167 |            41.471 |      0.172  |   61367 |
| R3_asinh_minus_how_clim | S4     |             44 |        0.029  | 0.98341 |  0.00073 |           0.0167 |            41.433 |      0.172  |   61367 |
| R3_asinh_minus_how_clim | S5     |            182 |        0.029  | 0.98341 |  0.00073 |           0.0167 |            41.433 |      0.172  |   61367 |


## kappa range per residual definition (with yaml placement)

| residual_definition     |   kappa_min |   kappa_max |   yaml_kappa | yaml_inside   |
|:------------------------|------------:|------------:|-------------:|:--------------|
| R1_asinh_raw            |      0.0167 |      0.0196 |       0.0784 | False         |
| R2_TRY_minus_monthly    |      0.205  |      0.2246 |       0.0784 | False         |
| R3_asinh_minus_how_clim |      0.0162 |      0.0167 |       0.0784 | False         |


## S2/S3 identity on R2 (FW9c anomaly explanation)

For R2_TRY_minus_monthly the within-month sum of the residual is zero by construction (P - monthly-mean subtracts the monthly mean).  Month-of-year dummies lie in the null space of the residual, so adding them from S2 to S3 does not change the fit.  This is not a bug in the ladder; it is a specification consequence of the R2 anchor choice.  R1 and R3, which do NOT subtract a monthly mean, DO show a non-degenerate S2 -> S3 step.

## FW9d picked definition and why

R3 is the natural analog of the shipped model's residual in the same variable and on the same scale as the yaml phi: the yaml v2 kappa refit comment attributes phi=0.9246 to a 'shock-around-anchor' single-regime AR(1) on the residual the forward-centered model actually prices.  R1 has no anchor (so persistence carries the seasonal cycle); R2 anchors on a monthly mean (so moy is degenerate).  R3 anchors on hour-of-week climatology, which is the intraday+weekly cycle that both the forward curve and the M9 deseasonalisation would remove first.
