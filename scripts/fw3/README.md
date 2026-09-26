# scripts/fw3

FW3 benchmark model comparison (Faz 5 Category A).  Full methodology in
`docs/fw3_benchmark_methodology.md`.

## Run

```bash
python scripts/fw3/compare_benchmarks.py
```

Writes to `outputs/fw3_benchmarks/`:

* `grid_comparison.csv` -- 66-row (K, T) grid: model call/put, benchmark
  call/put with daily-derived historical vol AND hourly naive vol,
  model-implied Black-76 / Bachelier vol, put-call parity errors on
  every benchmark.
* `implied_vol_surface_black76.csv` -- Black-76 IV smile pivoted by
  (strike, maturity), annualised.
* `implied_vol_atm_term_structure.csv` -- ATM (K = 3000) IV per maturity.
* `realized_backtest.csv` and `realized_backtest_summary.csv` -- realized
  discounted-payoff error for Model, B1, B2, B3 on the 66 contracts.
* `paper_table.csv` and `paper_table.md` -- compact table of
  representative (K, T) points ready for the manuscript (EK2).
* `README.md` -- narrative report.
* `historical_vol.json` -- inputs.

## Tests

```bash
python -m pytest tests/test_fw3_benchmarks.py -q
```

603 tests: put-call parity on B1/B2/B3, sigma -> 0 discounted intrinsic,
implied-vol round-trip, B3 -> Bachelier equivalence, model IV
round-trip on all 66 grid rows, and hash-invariance guards on the
accepted frozen artefacts.

## Data provenance

* Model prices, F(T), put values: `outputs/market_calibration_final/
  strike_maturity_grid.csv`.
* Historical PTF (for the volatility inputs): `inputs/historical/
  ptf_raw/ptf_2019..2025.csv` (real EPIAS hourly, 61,368 rows).  No
  look-ahead past 2025-12-31 20:00 UTC.
* Realized backtest spot: `inputs/market/realized_ptf_2026.csv`.
* Lucia-Schwartz sigma and kappa: pooled from
  `inputs/historical/m2_frozen_parameters.yaml` (M9 stationary
  occupancy weights, yaml v2 kappa).  Not re-fitted.

No synthetic data anywhere.  No accepted-output file is modified.
