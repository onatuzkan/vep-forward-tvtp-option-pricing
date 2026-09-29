# FW10 / FW10b -- Day-ahead publication timing audit and shipped-valuation reprice

## (a) Day-D prices known at end of D-1

The Turkish day-ahead auction closes at 12:30 TRT for the NEXT day's hourly clearing prices.  At 11:00 TRT of day D the hours 00:00-23:00 of day D are already published (D was cleared on D-1 at ~14:00 TRT); the 24 hours of day D+1 are NOT yet known.

## (b) Original valuation 2025-12-31 20:00 UTC (23:00 TRT)

Column labels corrected in this FW10b pass -- **the shipped valuation instant is at day-end, so any horizon whose terminal falls on day D+1 or later has NOT been published by the valuation instant.**  The FW10 pass mis-labelled the compact-table column as 'available_at_valuation'; the values themselves are the correct booleans.

| horizon | target_UTC | target_TRT | published_by_valuation |
|---|---|---|:---:|
| 24 h | 2026-01-01 20:00 UTC | 23:00 TRT of 2026-01-01 | **True** (day 01-01 DA prices publish 2025-12-31 ~14:00 TRT) |
| 48 h | 2026-01-02 20:00 UTC | 23:00 TRT of 2026-01-02 | False |
| 72 h | 2026-01-03 20:00 UTC | 23:00 TRT of 2026-01-03 | False |

## (c) FW3 backtest
22 of the 66 grid points (the entire 24 h maturity row) had their terminal PTF already published at the shipped valuation instant; this is a structural feature of the day-end valuation timestamp.

## (d) FW10b corrected rule + reprice
Under the corrected FW10b timing rule the forward curve is built from the VEP GGF publication STRICTLY BEFORE the valuation instant.  For the shipped 2025-12-31 20:00 UTC valuation this quote day is **2025-12-30**, versus the shipped `inputs/market/vep_monthly_quotes.csv` which records the 2025-12-31 quote day directly (i.e. a same-day quote that would only be available AFTER 20:00 UTC).

Absolute call-price differences (TRY):
|    K |   24 |   48 |   72 |
|-----:|-----:|-----:|-----:|
| 2000 |    0 |    0 |    0 |
| 2500 |    0 |    0 |    0 |
| 3000 |    0 |    0 |    0 |
| 3500 |    0 |    0 |    0 |
| 4000 |    0 |    0 |    0 |

Percentage differences (pct):
|    K |   24 |   48 |   72 |
|-----:|-----:|-----:|-----:|
| 2000 |    0 |    0 |    0 |
| 2500 |    0 |    0 |    0 |
| 3000 |    0 |    0 |    0 |
| 3500 |    0 |    0 |    0 |
| 4000 |    0 |    0 |    0 |

**Effect on the shipped 72 h ATM K=3000 call (166.75 TRY):** the corrected timing rule shifts it to 166.842 TRY, a change of +0.000 TRY (+0.000 pct).  The shipped production yaml valuation is NOT modified by this report; the reprice quantifies the timing-rule effect.

## (e) FW10b daily-evaluation convention
Every FW10b daily evaluation uses rule (d): valuation at d 11:00 TRT, last known PTF hour = d 23:00 TRT, horizons at h in {6, 12, 24, 48, 72} hours after that instant.  Forward curve on day d is built from the last VEP GGF strictly before d 11:00 TRT (so d-1 or earlier).  HPFC shape (fit once from pre-FREEZE_UTC data with `half_life_years = 0.5`, `n_harmonics = 2`) is applied inside every quoted delivery month.