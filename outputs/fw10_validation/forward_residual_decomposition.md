# FW10b supplement -- forward error versus residual dispersion

The predictive mean equals the forward F(T), so var(z) in
bias_dispersion_summary.csv mixes forward-curve error with residual
dispersion. Using the realised 2026 calendar-month mean and the
realised month-specific hour-of-day shape as a reference R (for
diagnosis only), the error is split into F - R and actual - R.

2026 hourly pure-residual sd, all hours January-September: 723.1 TRY/MWh.

## Production model (M0)

| h | model sd | forward error mean | forward error sd | pure residual sd | ratio |
|---:|---:|---:|---:|---:|---:|
| 6 | 387 | +463 | 767 | 600 | 1.55 |
| 12 | 400 | +794 | 843 | 533 | 1.33 |
| 24 | 497 | +385 | 697 | 755 | 1.52 |
| 48 | 509 | +427 | 716 | 771 | 1.51 |
| 72 | 518 | +484 | 717 | 698 | 1.35 |

## Reading

* The forward curve overshoots the realised level and shape at every
  horizon, and its day-to-day error is of the same size as the
  residual itself. The forward curve, not the residual model, is the
  source of all of the bias and of roughly half of the error variance.
* The pure residual is wider than the production model predicts by a
  factor of about 1.3 to 1.6 in standard deviation. The realised
  residual dispersion rose from 2025 (531-624 TRY/MWh, FW9) to 2026
  (about 720 TRY/MWh), so part of the gap is drift across years.
* The var(z) values of 4 to 11 in bias_dispersion_summary.csv
  therefore overstate the residual under-dispersion; they should not be
  read as a residual-model result on their own.
* The sign of the forward bias at h = 6 in the FW10b report (-260) is a
  single-day value; the mean over the 60 evaluation days is positive.

## Monthly VEP reference price staleness

| subset | contract-day pairs | nonzero changes | share | median distinct prices per contract |
|---|---:|---:|---:|---:|
| all contracts 2022-2026 | 6893 | 584 | 8.5% | 2 |
| delivery year 2026 | 1354 | 14 | 1.0% | 2 |
| quotation dates in 2026 | 964 | 6 | 0.6% | 2 |
