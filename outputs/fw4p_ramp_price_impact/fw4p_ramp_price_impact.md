# FW4-P -- price impact of the ramp covariate under FW9's ramp

FW4's experimental two-covariate mode reported a total ramp effect of
-1.008 % at the 72 h `K = 3000` call, using a
*reconstructed* ramp and ramp slopes *transferred* from the M9 bundle.  FW9
later defined its own ramp series and estimated the slopes jointly
(`outputs/fw9_self_estimation/TVTP_2cov.pkl`: h01 = -0.079462,
h10 = 0.381573; LR = 869.93 against the single-covariate
fit).  This note prices the same contracts with FW9's ramp and FW9's slopes,
everything else at production.

## What actually differs

The ramp *definition* is the same object in both: the transition into hour
`t` is driven by `(z(t-1), z(t-1) - z(t-2))`.  What differs is the
standardisation window and the slopes.

| | m_r | s_r | window ends | h01 | h10 |
|---|---:|---:|---|---:|---:|
| FW4 (yaml) | 3.417142e-05 | 0.271450 | 2024-12-31 | -0.069395 | 0.405544 |
| FW9 (pkl) | 2.962489e-05 | 0.266400 | 2022-12-31 | -0.079462 | 0.381573 |

FW9's `r` is 1.90 % larger in
magnitude for the same increment.  Note the two conventions also disagree on
*when* the shift is applied relative to the training cut, so
`CovariatePathBuilder.verify_ramp_scaler` rejects FW9's scaler by design;
FW9's own `scripts/fw9/build_ramp.py` is used as the source of truth and the
mismatch is recorded here rather than silently reconciled.

## Three prices side by side

`K = F(T)` rows are marked `ATM`.  All variants at a maturity are priced on
one fixed 1201-node residual grid (the union of the auto-sized
grids), so the columns differ only through the transition law.

| maturity_h | strike_kind | strike_TRY_MWh | call_production | call_fw4_ramp | call_fw9_ramp | pct_fw4_ramp_vs_production | pct_fw9_ramp_vs_production |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | ATM | 2917.2391 | 202.2946 | 198.8448 | 198.2308 | -1.7053 | -2.0088 |
| 24 | ladder | 2000.0000 | 927.1080 | 926.7123 | 926.4840 | -0.0427 | -0.0673 |
| 24 | ladder | 2500.0000 | 479.5332 | 477.6751 | 477.1044 | -0.3875 | -0.5065 |
| 24 | ladder | 3000.0000 | 163.9351 | 160.5858 | 159.9664 | -2.0430 | -2.4209 |
| 24 | ladder | 3500.0000 | 36.6351 | 35.4582 | 34.9922 | -3.2124 | -4.4845 |
| 24 | ladder | 4000.0000 | 5.5892 | 5.3731 | 5.2318 | -3.8659 | -6.3954 |
| 48 | ATM | 2916.6982 | 205.7749 | 202.7569 | 202.0447 | -1.4667 | -1.8128 |
| 48 | ladder | 2000.0000 | 926.2230 | 925.8647 | 925.6158 | -0.0387 | -0.0656 |
| 48 | ladder | 2500.0000 | 480.8387 | 479.1587 | 478.5392 | -0.3494 | -0.4782 |
| 48 | ladder | 3000.0000 | 167.1250 | 164.1871 | 163.4758 | -1.7579 | -2.1835 |
| 48 | ladder | 3500.0000 | 38.1080 | 37.0426 | 36.5404 | -2.7957 | -4.1135 |
| 48 | ladder | 4000.0000 | 5.9534 | 5.7587 | 5.6035 | -3.2704 | -5.8772 |
| 72 | ATM | 2916.1573 | 205.5968 | 202.5882 | 201.8747 | -1.4634 | -1.8104 |
| 72 | ladder | 2000.0000 | 924.6992 | 924.3411 | 924.0917 | -0.0387 | -0.0657 |
| 72 | ladder | 2500.0000 | 479.9070 | 478.2286 | 477.6085 | -0.3497 | -0.4789 |
| 72 | ladder | 3000.0000 | 166.7567 | 163.8289 | 163.1163 | -1.7558 | -2.1831 |
| 72 | ladder | 3500.0000 | 38.0145 | 36.9531 | 36.4512 | -2.7921 | -4.1124 |
| 72 | ladder | 4000.0000 | 5.9370 | 5.7431 | 5.5882 | -3.2652 | -5.8757 |

## Ramp effect, occupancy held fixed

FW4's headline number is R3 - R1: both legs carry intercepts re-derived by
moment matching on D = W9, so the mean transition probabilities are held
fixed and only the ramp channel moves.  The same contrast for the FW9 ramp
is `fw9_ramp_matched - base_1d_w9`.

| maturity_h | strike_kind | strike_TRY_MWh | ramp_variant | ramp_effect_TRY_MWh | ramp_effect_pct |
|---:|---:|---:|---:|---:|---:|
| 24 | ATM | 2917.2391 | fw4_ramp | -2.1436 | -1.0665 |
| 24 | ATM | 2917.2391 | fw9_ramp_matched | -2.1761 | -1.0827 |
| 24 | ladder | 2000.0000 | fw4_ramp | -0.2157 | -0.0233 |
| 24 | ladder | 2000.0000 | fw9_ramp_matched | -0.2294 | -0.0248 |
| 24 | ladder | 2500.0000 | fw4_ramp | -1.0892 | -0.2275 |
| 24 | ladder | 2500.0000 | fw9_ramp_matched | -1.1226 | -0.2345 |
| 24 | ladder | 3000.0000 | fw4_ramp | -2.0770 | -1.2769 |
| 24 | ladder | 3000.0000 | fw9_ramp_matched | -2.1097 | -1.2970 |
| 24 | ladder | 3500.0000 | fw4_ramp | -0.6654 | -1.8419 |
| 24 | ladder | 3500.0000 | fw9_ramp_matched | -0.6938 | -1.9206 |
| 24 | ladder | 4000.0000 | fw4_ramp | -0.1182 | -2.1528 |
| 24 | ladder | 4000.0000 | fw9_ramp_matched | -0.1265 | -2.3029 |
| 48 | ATM | 2916.6982 | fw4_ramp | -1.7295 | -0.8458 |
| 48 | ATM | 2916.6982 | fw9_ramp_matched | -1.7832 | -0.8720 |
| 48 | ladder | 2000.0000 | fw4_ramp | -0.1663 | -0.0180 |
| 48 | ladder | 2000.0000 | fw9_ramp_matched | -0.1844 | -0.0199 |
| 48 | ladder | 2500.0000 | fw4_ramp | -0.8937 | -0.1862 |
| 48 | ladder | 2500.0000 | fw9_ramp_matched | -0.9397 | -0.1957 |
| 48 | ladder | 3000.0000 | fw4_ramp | -1.6785 | -1.0120 |
| 48 | ladder | 3000.0000 | fw9_ramp_matched | -1.7318 | -1.0441 |
| 48 | ladder | 3500.0000 | fw4_ramp | -0.5362 | -1.4269 |
| 48 | ladder | 3500.0000 | fw9_ramp_matched | -0.5736 | -1.5264 |
| 48 | ladder | 4000.0000 | fw4_ramp | -0.0894 | -1.5292 |
| 48 | ladder | 4000.0000 | fw9_ramp_matched | -0.1003 | -1.7159 |
| 72 | ATM | 2916.1573 | fw4_ramp | -1.7221 | -0.8429 |
| 72 | ATM | 2916.1573 | fw9_ramp_matched | -1.7762 | -0.8693 |
| 72 | ladder | 2000.0000 | fw4_ramp | -0.1660 | -0.0180 |
| 72 | ladder | 2000.0000 | fw9_ramp_matched | -0.1841 | -0.0199 |
| 72 | ladder | 2500.0000 | fw4_ramp | -0.8919 | -0.1862 |
| 72 | ladder | 2500.0000 | fw9_ramp_matched | -0.9381 | -0.1958 |
| 72 | ladder | 3000.0000 | fw4_ramp | -1.6708 | -1.0095 |
| 72 | ladder | 3000.0000 | fw9_ramp_matched | -1.7245 | -1.0420 |
| 72 | ladder | 3500.0000 | fw4_ramp | -0.5334 | -1.4229 |
| 72 | ladder | 3500.0000 | fw9_ramp_matched | -0.5708 | -1.5228 |
| 72 | ladder | 4000.0000 | fw4_ramp | -0.0888 | -1.5232 |
| 72 | ladder | 4000.0000 | fw9_ramp_matched | -0.0998 | -1.7107 |

`fw9_ramp` (the literal "all other parameters as in production" reading)
keeps the production intercepts, so it is *not* occupancy-controlled: its
difference from production mixes the ramp channel with a shift in the mean
transition probabilities.  It is reported above for completeness and in
`fw4p_ramp_effect.csv` with `occupancy_controlled = False`.

## Caveats

* **FW9's two-covariate fit sits at the unit-root boundary.**  It was
  estimated on the raw asinh level, and the fitted persistence is
  `phi = 0.9999987156` -- a half-life of
  62 years.  At
  that boundary the level absorbs low-frequency structure that the regime
  process would otherwise carry, so the transition coefficients, including
  h01 and h10, may be biased.  The optimiser also did not meet its gradient
  tolerance (`converged = False`, grad norm
  96.6 at 61368 observations), so these slopes
  carry no usable standard error.
* **The slopes are used outside the fit they came from.**  FW9 estimated
  h01/h10 jointly with its own gammas
  (-0.423428, 0.116406); here they
  are combined with the production gammas (-0.583778, 0.077698), as the task
  specifies.  A jointly-consistent run would move the z channel too.
* **Expected magnitude.**  FW9's own decomposition
  (`price_impact_v2_decomposition.csv`) attributes 0.39 % of the total
  price difference to the transition coefficients at the 72 h `K = 3000`
  call, so a small effect is what the evidence predicts; the numbers above
  are consistent with that.
* **The ramp remains experimental.**  Nothing here replaces an accepted
  result, and the accepted output trees are untouched.

At `K = 3000`, for reference:

| maturity_h | strike_kind | strike_TRY_MWh | call_production | call_fw4_ramp | call_fw9_ramp | pct_fw4_ramp_vs_production | pct_fw9_ramp_vs_production |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | ladder | 3000.0000 | 163.9351 | 160.5858 | 159.9664 | -2.0430 | -2.4209 |
| 48 | ladder | 3000.0000 | 167.1250 | 164.1871 | 163.4758 | -1.7579 | -2.1835 |
| 72 | ladder | 3000.0000 | 166.7567 | 163.8289 | 163.1163 | -1.7558 | -2.1831 |

At the money:

| maturity_h | strike_kind | strike_TRY_MWh | call_production | call_fw4_ramp | call_fw9_ramp | pct_fw4_ramp_vs_production | pct_fw9_ramp_vs_production |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | ATM | 2917.2391 | 202.2946 | 198.8448 | 198.2308 | -1.7053 | -2.0088 |
| 48 | ATM | 2916.6982 | 205.7749 | 202.7569 | 202.0447 | -1.4667 | -1.8128 |
| 72 | ATM | 2916.1573 | 205.5968 | 202.5882 | 201.8747 | -1.4634 | -1.8104 |

Files: `fw4p_runs.csv` (every run), `fw4p_three_way.csv` (the side-by-side),
`fw4p_ramp_effect.csv` (controlled and uncontrolled contrasts),
`run_manifest.json`, and the before/after hash manifests of the 128 protected
files.
