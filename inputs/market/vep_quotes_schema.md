# Market quote schema (`inputs/market/`)

## What a VEP monthly quote *is*

An EPİAŞ VEP monthly contract (`EBM<MM><YY>`) is a **baseload average**, not a
point-maturity forward:

```
F_m = (1 / N_m) * sum over all delivery hours h in month m of E^Q[P_h]
```

`N_m` is the true number of Turkish local delivery hours = `days_in_month * 24`
(Türkiye is fixed UTC+3, no daylight saving, so there are no 23/25-hour days).

| contract | delivery month | N_m |
|---|---|---|
| EBM0226 | 2026-02 | 672 (2026 is **not** a leap year) |
| EBM0326 | 2026-03 | 744 |
| EBM0426 | 2026-04 | 720 |
| EBM0526 | 2026-05 | 744 |
| EBM0626 | 2026-06 | 720 |
| EBM0726 | 2026-07 | 744 |

`MonthlyBaseloadQuote.maturity_utc` and `.as_point_forward()` deliberately raise
`MonthlyQuoteMisuseError`.

## CSV columns (`vep_monthly_quotes.csv`)

| column | type | notes |
|---|---|---|
| `contract_name` | str | `EBM0226`; validated against delivery year/month |
| `delivery_year` | int | Turkish local delivery year |
| `delivery_month` | int | 1–12 |
| `price_TRY_MWh` | float | baseload price, TRY/MWh |
| `quote_type` | str | must be `monthly_baseload` |
| `source` | str | free text provenance |
| `valuation_date` | str | one value for the whole file; bare dates = Turkish day end (23:00 TRT = 20:00 UTC) |
| `weight` | float | optional least-squares weight (default 1.0) |
| `spot_price_TRY_MWh` | float | optional; only the first non-empty value is used |

## Adding January 2026 or day-ahead data later

No code change is needed. Add a row

```csv
EBM0126,2026,1,<price>,monthly_baseload,EPIAS_VEP,2025-12-31,1.0,
```

and January stops being an extrapolation: it becomes a hard monthly-average
constraint, `extrapolation_flag` / `near_term_anchor_flag` flip to `false` in
`hourly_forward_curve.csv`, and it appears in `monthly_forward_fit.csv` with its
own acceptance row.

`vep_2025-12-31.json` additionally reserves a `future_extension_schema` block for
day-ahead baseload quotes and for option premia (the latter are what would be
needed to identify the volatility and transition risk premia).
