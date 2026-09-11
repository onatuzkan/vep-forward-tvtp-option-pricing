# Realized 2026 backtest — VEP-anchored forward curve vs realized PTF

Compares the accepted hourly forward curve produced by
`run_pde.py calibrate-market` (`outputs/market_calibration_final/hourly_forward_curve.csv`,
5089 hourly nodes covering 2025-12-31 20:00 UTC → 2026-07-31 20:00 UTC)
against the realized EPİAŞ PTF `inputs/market/realized_ptf_2026.csv` (semicolon-
separated, Turkish-locale numbers, Turkish local timestamps, converted to
UTC via `Europe/Istanbul`).

## Data check

* Realized file: 5856 hourly rows, 31.12.2025 00:00 → 31.08.2026 23:00 Turkish
  local (= 2025-12-30 21:00 → 2026-08-31 20:00 UTC).  No duplicated
  timestamps; every consecutive gap is exactly 1 h.
* Forward curve: 5089 hourly rows, 2025-12-31 20:00 → 2026-07-31 20:00 UTC.
* Merged inner-join on UTC hour: **5089** rows (the forward curve's horizon
  is the binding constraint; the extra August realized rows are dropped).

## Monthly summary (VEP-anchored forward vs realized PTF)

Columns:
* `n_hours_realized_<10` = number of hours where the realized PTF was below
  10 TRY/MWh (near-zero-price events; the Turkish market saw occasional
  zero-price hours from renewable oversupply in 2026-Q2).
* `MAE`, `RMSE` in TRY/MWh; `sMAPE` = symmetric MAPE = `200·|A−F|/(|A|+|F|)`,
  bounded in [0, 200] and robust to near-zero realized values.
  `MAPE_capped` uses `max(|realized|, 10)` as denominator; `median_APE_capped`
  is the median of the same, robust to outliers.
* `signed_pct_vs_forward` = mean of `100·(realized − forward) / forward`
  (negative = realized was below forward).
* `realized_std` = std of hourly realized PTF within the month.
* `residual_std_realized` = std of hourly `(realized − forward)` within the
  month (empirical residual dispersion).
* `mean_model_residual_sd` = time-average of the model's ANALYTIC residual
  std over the hours in the month.  Grows monotonically with hour-from-
  valuation because the reconciled near-unit-root kappa lets residual
  variance accumulate linearly.
* `model_over_realized_ratio` = `mean_model_residual_sd / residual_std_realized`.

| month | n_hours | n<10 | mean_realized | mean_forward | mean_bias | MAE | RMSE | sMAPE % | median_APE_capped % | signed_% vs fwd | realized_std | residual_std_realized | mean_model_residual_sd | model/realized ratio | anchor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-01 | 744 | 0 | 2894.92 | 2909.39 | **−14.47** | 451.93 | 573.38 | **16.76** | 14.19 | −0.50 | 573.25 | 573.58 | 4014.39 | 7.00 | **anchor** |
| 2026-02 | 672 | 22 | 2078.20 | 2900.99 | −822.79 | 992.40 | 1297.34 | 49.98 | 38.95 | −28.37 | 1004.68 | 1003.79 | 7215.95 | 7.19 | quoted |
| 2026-03 | 744 | 23 | 1620.32 | 2556.31 | −935.99 | 1193.58 | 1423.03 | 71.52 | 59.68 | −36.60 | 1071.50 | 1072.61 | 9075.17 | 8.46 | quoted |
| 2026-04 | 720 | 98 | 921.06 | 2500.66 | −1579.60 | 1873.41 | 1972.00 | 126.25 | 588.94 | −63.17 | 1181.31 | 1181.34 | 10440.06 | 8.84 | quoted |
| 2026-05 | 744 | **222** | **590.90** | 2506.25 | **−1915.35** | 2150.69 | 2209.40 | **155.21** | 1221.63 | **−76.39** | 1100.49 | 1102.04 | 11615.05 | 10.54 | quoted |
| 2026-06 | 720 | 40 | 1240.16 | 2245.33 | −1005.17 | 1458.19 | 1604.18 | 101.27 | 179.69 | −44.91 | 1258.41 | 1251.08 | 12575.41 | 10.05 | quoted |
| 2026-07 | 744 | 0 | 2699.61 | 3558.19 | −858.58 | 1058.33 | 1401.56 | 40.93 | 20.79 | −24.08 | 1104.87 | 1108.54 | 13947.76 | 12.58 | quoted |

### Overall (all 5089 merged hours)

| metric | value |
|---|---:|
| MAE | **1312.39 TRY/MWh** |
| RMSE | 1575.43 TRY/MWh |
| sMAPE | 80.39 % |
| mean bias (realized − forward) | **−1019.04 TRY/MWh** |
| mean signed % vs forward | **−39.16 %** |

## Interpretation — three separable findings

### Independent cross-check (data authenticity)

The realized monthly means reported here (Feb 2078, Mar 1620, Jul 2700
TRY/MWh) match independently published EPİAŞ-sourced monthly averages
reported in Turkish energy-sector press — e.g., Selenka Enerji's Feb
2026 PTF report citing 2.078 TL/MWh; HESİAD's Mar 2026 report citing
1.620 TL/MWh; sector press citing Jul 2026 at 2.699,61 TL/MWh after a
117.68 % MoM rise.  This independently confirms the backtest data is
**not a unit / parsing artifact**, and that the Feb-Jun price collapse
was a **real, widely-reported market event** attributed to an
exceptionally strong 2026 hydrological year combined with record
renewable output displacing gas generation — not anticipated by the
Dec-2025 VEP forward curve.

### 1. The near-term anchor (January) beat the VEP-quoted months on realised performance

Jan 2026 is the one month **not** anchored to any observed VEP quote — its
744 hourly forwards come entirely from the `spot_to_next_linear` rule
(ramp from `spot = 2917.78` down to the first quoted month's average
`F_Feb = 2900.99`).  On realised data, Jan came in with:

* **mean bias −14 TRY/MWh** (a rounding-level miss)
* sMAPE **16.76 %** — the lowest of any month in the horizon
* zero hours below 10 TRY/MWh

Every VEP-quoted month (Feb-Jul) missed by hundreds of TRY/MWh in the
same direction (realized far below quoted): sMAPE 41-155 %, mean bias
−820 to −1915 TRY/MWh, signed −24 % to −76 % vs the quoted level.  In
other words, **the analyst-supplied anchor rule did better on this
sample than the market's own observed forwards did.**  This is not a
statement about the anchor rule's superiority — it is a statement about
the 2026 EPİAŞ VEP market having systematically over-forecasted realized
prices, particularly across Q2.

### 2. The market surprise is real; the calibration is not "wrong"

The forward curve produced by `calibrate-market` reproduces the six VEP
monthly quotes to solver precision (max abs monthly error ≈ 3.64e-12
TRY/MWh in the accepted run).  Any monthly-level bias observed here
propagates 1:1 from the VEP inputs, not from the model.  What the table
documents is that the *Turkish day-ahead market delivered materially
lower prices than the VEP forwards implied* over Feb-Jul 2026 — a
market-outcome finding, not a model-fit failure.

Direction: consistently NEGATIVE bias (realized below forward) in every
month, growing through spring (Feb −823 → Apr −1580 → May −1915) and
partially recovering in Jun-Jul.  The month with the largest miss is
**May 2026**, which combines the largest absolute bias (−1915 TRY/MWh)
with the highest count of near-zero-price hours (222 / 744 = 30 % of
May hours below 10 TRY/MWh).  This pattern is consistent with a Q2
solar / wind oversupply regime driving repeated near-zero clearings — a
real market phenomenon that no six-month baseload forward strip could
have anticipated at end-2025.  The report abstains from attribution
beyond what the data itself supports (no counter-factual "why" claim;
only "what").

### 3. The model over-predicts residual volatility by ~7-13×

The `mean_model_residual_sd` column is the time-average of the analytic
residual std the model expected to see, from `model.moments_at(h)`, over
each month's hours.  It grows monotonically with horizon (4014 TRY/MWh
in Jan → 13 948 TRY/MWh in Jul) because the reconciled near-unit-root
kappa (~4.11e-6/h) lets the residual variance accumulate almost linearly
in time.  The `residual_std_realized` column is the empirical std of
hourly `(realized − forward)` within each month — it is roughly
CONSTANT across months (573-1258 TRY/MWh), because in the real market
the residual is bounded by structural features (price caps, near-zero
floors, daily / weekly cycles) that a pure OU cannot capture.

Ratio `model / realized` grows from 7× at short horizons to 12-13× at
long horizons.  This means **the M9 sigmas, which come from an
in-sample fit on hourly asinh(P/scale_P), map to a residual dispersion
much wider than what the 2026 out-of-sample data actually produced**.
For option pricing this is a conservative bias: option values under the
current yaml will be systematically higher than what a variance-matched
recalibration would produce.  A calibration follow-up that shrinks the
sigmas (or introduces a price-floor / clip mechanism) is a natural
next step; a candidate for the "future work" list.

## What this says about the "VEP-forward-anchored" design

**Structural claim** — the calibration methodology *is doing what it
says on the tin*.  It reproduces the six VEP quotes to solver
precision, produces a smooth hourly curve with an economically-
motivated near-term anchor, and preserves the `E^Q[P_t] = F(t)`
identity everywhere.  The 2026 realized-data backtest exposes a
different question — *how well did the underlying VEP forwards
themselves anticipate realized prices?* — and on that question the
answer for 2026 Feb-Jul is "poorly".

**Practical claim** — for downstream option users, the VEP curve is
the *price-level* input, not a model output.  If VEP forwards move
away from realized prices (as here), option values will inherit the
level miss 1:1.  The `--pi-override` and `--risk-premium-a0/a1` knobs
are level-neutral; they only reshape the option payoff around F(t),
they do not correct the level itself.  A hedger who uses these outputs
as if they were forecast prices — rather than as pricing-measure
consistency inputs — will inherit the market's forecast bias.

This is arguably the **single most important limitation for any reader
of this model**: the forward-anchoring design is only as good as the
forward market's own predictive accuracy.  In a market subject to
large hydrological / renewable supply shocks, VEP forwards can miss
realized outcomes by 40-155 % over a 6-month horizon.  This is **not a
flaw in the option-pricing methodology** — it is a structural property
of ANY forward-anchored pricing approach, and should be stated
prominently in the paper's limitations / discussion section, not
buried.

Recorded as future-work items to make this explicit in the paper and
in the model_limitations documentation.

## Files

* `outputs/market_calibration_final/realized_2026_backtest.csv` — hourly
  detail (5089 rows): forward, realized, error, sMAPE, capped-MAPE,
  signed %, model residual sd, near-term-anchor flag.
* `outputs/market_calibration_final/realized_2026_backtest.monthly.csv` —
  the summary table above.
* `outputs/market_calibration_final/realized_2026_backtest.png` — dual
  bar chart (monthly sMAPE and MAE, colour-coded by anchor vs
  VEP-quoted).

No production code was touched; the analysis is inline in a script
(reproducible: load realized CSV → merge on UTC → group by
`delivery_month`).  Full test suite (172 passing) is unaffected.
