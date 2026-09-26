# FW3 -- Benchmark model comparison (methodology)

Faz 5 Category A item.  Adds referee-proof answers to two questions the
manuscript needs to answer explicitly:

1. **"Why not just Black-76?"**  -- by pricing the exact same F2.8 contracts
   under the three closed-form benchmarks that the electricity-option
   literature uses as standard baselines, and by extracting the
   model-implied Black-76 / Bachelier volatility surface so the mismatch
   is visible directly.
2. **"Which pieces of the model actually matter?"**  -- by including
   Lucia & Schwartz (2002) as a single-factor arithmetic-OU baseline that
   isolates the regime-switching contribution from the arithmetic-OU
   contribution the two share.

Nothing in FW3 changes the accepted PDE model or its outputs.  Every
benchmark price is computed from real repo data: `strike_maturity_grid.csv`
supplies F(T) and the model prices, `inputs/historical/ptf_raw/`
supplies the historical price series used for the volatility inputs, and
`inputs/market/realized_ptf_2026.csv` supplies the out-of-sample realized
spot for the discounted-payoff backtest.  No synthetic values.

## What the production model prices

`pde_option_model.contracts.EuropeanOption` is a European call/put on the
single-hour spot PTF `P_T` (TRY/MWh), valued at
`t0 = 2025-12-31 20:00 UTC`, with the terminal payoffs

* call: `max(P_T - K, 0)`
* put:  `max(K - P_T, 0)`

and discount factor `exp(-r * tau)` with `r_per_hour = r_annual / 8760`
(ACT/365 convention, `contracts.HOURS_PER_YEAR`).  The forward level at
maturity `F(T) = E^Q[P_T]` is the exact identity delivered by the
forward-centered PDE (`forward_centered.expected_spot` returns
`F(T)` up to solver precision because the residual centering
`mu_X(t) = E^Q[X_t]` is subtracted at every hour).

All three benchmarks below take exactly these primitives (`F(T)`, `K`,
`tau_hours`, `r_per_hour`) and differ only in how they model the
distribution of `P_T`.

## Benchmarks

### B1 -- Black (1976): lognormal on the forward

`P_T = F(T) * exp(sigma * W_tau - 0.5 sigma^2 tau)` under Q.  Closed
form:

    C = exp(-r tau) [F N(d1) - K N(d2)]
    d1 = [ln(F/K) + 0.5 sigma^2 tau] / (sigma sqrt(tau))
    d2 = d1 - sigma sqrt(tau)

Sigma is the log-return volatility per `sqrt(hour)` (per-hour convention
throughout FW3; annualise by `* sqrt(8760)` for reading).  Excludes the
possibility of `P_T <= 0`, which the Turkish market repeatedly showed in
2026-Q2 (222 sub-10 TRY/MWh hours in May alone) -- part of the reason
Black-76 alone is not a defensible benchmark for a market with daily
price caps and a soft floor at zero.

### B2 -- Bachelier: arithmetic-normal on the forward

`P_T ~ N(F(T), sigma^2 tau)`, closed form:

    C = exp(-r tau) [(F - K) N(d) + sigma sqrt(tau) phi(d)]
    d = (F - K) / (sigma sqrt(tau))

Sigma is TRY/MWh per `sqrt(hour)`.  Handles `F <= K` and `P_T <= 0`
gracefully, which is why it is the natural baseline for electricity: the
distribution has no positivity constraint, matching the market's
observed sub-10 TRY/MWh clearings.  The **cap** at 3400/4500 TRY/MWh is
NOT enforced by Bachelier (nor by the production model itself) -- the
cap is a truncation that neither closed form encodes; documented as a
model limitation for both.

### B3 -- Lucia & Schwartz (2002): one-factor arithmetic-OU

`P_t = F(t) + X_t`, `dX_t = -kappa X_t dt + sigma dW_t`, `X_0 = 0`.  The
terminal law is `P_T ~ N(F(T), Var_T)` with

    Var_T = sigma^2 (1 - exp(-2 kappa T)) / (2 kappa)

so B3 is Bachelier with an equivalent flat vol `sigma_B = sqrt(Var_T / T)`
(the implementation calls Bachelier internally; the equivalence is a
unit test).  This is the "regime-switching closed" reference: it retains
the accepted OU rate and the accepted regime volatilities (pooled by the
M9 stationary occupancy), but removes the two-regime chain, so the gap
between B3 and the model isolates the **regime-conditioning + TVTP**
contribution of the production spec.

`sigma` and `kappa` come from `inputs/historical/m2_frozen_parameters.yaml`
(the v2 kappa refit).  **They are not re-fitted for FW3.**  The single-
regime collapse is realised by pooling the two `sigma_y` values with the
M9 stationary occupancy `pi = (0.325, 0.675)`:

    pooled_sigma_y = sqrt(pi_normal * sigma_y_normal^2
                          + pi_stress * sigma_y_stress^2)
                    = sqrt(0.325 * 0.0035348^2 + 0.675 * 0.0924067^2)
                    = 0.075923 per sqrt(hour) (y-space)

Mapped to price space at each F(T) via the same delta-method transfer
`ResidualSpec.sigma_price` uses (`sigma_price = sigma_y * sqrt(F^2 +
scale_P^2)`) so the OU dispersion has the same level-dependence as the
production model.  `kappa_per_hour = 0.078394` (half-life 8.84 h) is the
v2 reconciled value in the yaml.

## Volatility inputs -- two views reported side by side

### (i) Historical (no look-ahead past 2025-12-31 20:00 UTC)

Source: `inputs/historical/ptf_raw/ptf_2019..2025.csv` (real EPIAS hourly
PTF, 61,368 rows), Turkey local timestamps.  Window: **last 8760 hours
(~365 days)** ending at the valuation instant.  Price floor 50 TRY/MWh
guards the log-return sample against occasional near-zero clearings
(147 zeros and 539 sub-10 TRY/MWh hours over 2019-2025).

Two sampling frequencies are computed:

* **Hourly** returns: `sigma_log_hourly ~ 0.30 per sqrt(h)` (annualised
  ~28.6).  Dominated by the intraday demand cycle (a peak-hour vs pre-
  dawn move is signal, not innovation), so hourly log-vol is inflated.
  Reported for transparency only; NOT the primary benchmark input.
* **Daily** returns on the calendar-day mean price:
  `sigma_log_daily ~ 0.17 per sqrt(d)` (annualised ~3.28), rescaled to
  per-`sqrt(hour)` via `/ sqrt(24)`.  Strips the intraday cycle; this
  is the PRIMARY input for Black-76 and Bachelier.  It matches the
  sampling frequency at which the Black-76 lognormal assumption is
  approximately defensible (daily returns are much closer to serially
  uncorrelated than hourly returns are).

### (ii) Model-implied (Brent root-finding on each closed form)

For every one of the 66 (K, T) points we take the accepted PDE call
price and invert Black-76 to get an implied lognormal vol, and Bachelier
to get an implied arithmetic vol.  Two consequences:

* The **implied-vol surface** encodes the smile / skew visible in the
  production model.  A pure Black-76 model would show a flat surface;
  the observed `iv_black76` term structure (ATM ~3.29 at 24 h -> 0.61
  at 720 h annualised) and the mild negative-skew smile at each
  maturity are the answer to "why not just Black-76" -- the model
  price cannot be summarised by a single lognormal vol.
* Round-trip `price -> iv -> price` matches to 1e-9 by construction
  (asserted in `tests/test_fw3_benchmarks.py`).  The surface is a
  presentation of the model prices, not an independent estimate.

## Comparisons produced

`scripts/fw3/compare_benchmarks.py` produces four artefacts under
`outputs/fw3_benchmarks/`:

1. **`grid_comparison.csv`** (66 rows) -- F(T), K, T, model call/put,
   benchmark call/put with the historical vol AND with the hourly
   naive vol, implied vols from the model call, put-call parity
   errors on every benchmark.  Absolute and % differences against
   the model call.
2. **`implied_vol_surface_black76.csv`** and
   **`implied_vol_atm_term_structure.csv`** -- the smile / term
   structure explained above.
3. **`realized_backtest.csv`** and **`realized_backtest_summary.csv`**
   -- for every F2.8 contract, the realized `P_T` from
   `realized_ptf_2026.csv`, the realized discounted payoff, and the
   error of each pricer.  Only aggregate mean/MAE/RMSE are
   interpretable (each contract is a single-path draw).
4. **`paper_table.csv`** and **`paper_table.md`** -- EK2 compact table
   of representative (K, T) points ready to drop into the manuscript.

## Put-call parity as a common correctness check

`parity_error(C, P, F, K, tau, r_per_hour) = C - P - exp(-r tau) (F - K)`.
The three closed-form benchmarks return zero analytically; the accepted
PDE model returns zero to `1e-6 TRY/MWh` at the shipped resolution.
FW3 asserts `< 1e-10` on every benchmark case in
`tests/test_fw3_benchmarks.py`.

## EK1 -- relationship to the existing pooled-vs-M9 comparison

The manuscript already carries a "pooled single-volatility baseline vs
M9" comparison in
`outputs/market_calibration_final/model_comparison_pooled_vs_M9.md`
(shown as F2.5).  B3 (Lucia-Schwartz) is conceptually closest to that
baseline -- both collapse the two-regime residual to a single-regime
OU with a pooled sigma.  The two comparisons ISOLATE DIFFERENT things,
and this must be stated in the paper:

| aspect | F2.5 pooled baseline | B3 Lucia-Schwartz (FW3) |
|---|---|---|
| pooled sigma source | M0's two sigmas (0.0061, 0.164) | M9's two sigmas (0.0035, 0.0924) |
| pooling weights | M9 stationary occupancy (0.325, 0.675) | same (M9 stationary) |
| kappa | two variants: M0's 8.2e-4/h AND M9's 4.11e-6/h (pre-v2) | yaml v2 kappa 0.0784/h |
| implementation | 2-regime pipeline with equal sigmas | closed-form Bachelier w/ effective OU vol |
| what the gap vs M9 measures | (regime-switching) + (M0 vs M9 sigma) + (kappa) mixed | pure regime-conditioning under M9's OWN sigmas and the CURRENT yaml kappa |
| headline gap vs model | -37 to -49 % across maturities | +10.4 to +11.3 % across maturities |

**The two comparisons answer different questions.** F2.5 asks "how much
of the M9 price is attributable to using M9's parameters + regime
switching vs an unrelated M0 pool"; the -45 % gap conflates the
sigma-source change and the regime-conditioning change, and it uses a
kappa that pre-dates the v2 refit.  B3 asks "at the current yaml
parameters, how much does the two-regime chain add on top of a single-
factor OU with the same long-run variance rate"; the +10 % gap is a
clean regime-conditioning measurement.

**Recommendation** (my judgment; kararı sen ver):

1. **Keep F2.5 in the paper.**  It stays as a historical / robustness
   claim: even under a *different* model's sigmas the qualitative
   picture (M9 systematically cheaper at short horizons, converging at
   long horizons) survives.  Do NOT re-quote its -45 % as a
   "regime-switching value" number -- that would over-claim, exactly as
   the report itself already flags in its caveats (b) and (c).
2. **Add B3 as the FW3 benchmark comparison.**  Frame it as the
   referee-answer to "OK, so how much does the regime chain add over a
   single-factor OU at the CURRENT yaml parameters?"  The answer is
   +10-11 % across maturities (single-regime OU prices ATM K=3000 calls
   ~10 % above the model).  This is the cleaner number for the
   manuscript's "why regime-switching" argument.
3. If space forces a choice, **replace F2.5 with B3**.  B3 is
   parameter-consistent with the shipped model (uses the yaml v2 kappa
   and M9 sigmas); F2.5 leans on M0's sigmas which nobody in the
   paper uses for anything else.  Section 7.3's headline number would
   change from "regime-conditioning is worth 37-49 %" to
   "regime-conditioning is worth ~10 % at the current parameters",
   which is a more defensible referee-answer.

The comparison-of-comparisons (this table + the two headline numbers)
is enough for a footnote in the paper's discussion section; the full
FW3 vs F2.5 numerical setup lives in this file and in the FW3
outputs.

## EK2 -- paper-ready table

`outputs/fw3_benchmarks/paper_table.md` and `paper_table.csv` carry a
compact 8-row table (six ATM K=3000 maturities plus a deep-OTM
K=4000/168h and a deep-ITM K=2000/168h contrast) with columns Model /
B1 / B2 / B3 and the three % differences.  CSV + Markdown only; LaTeX
conversion is left to the manuscript editor.

## Nothing modified

The FW3 tests (`tests/test_fw3_benchmarks.py`) include a hash-guard on
`inputs/historical/m2_frozen_parameters.yaml`,
`inputs/historical/tvtp2_frozen_parameters.yaml`, and a non-empty-tree
guard on the four accepted output folders.  Both the accepted 296-test
suite and the new FW3 tests must pass together; the recorded hashes
were snapshotted at the start of the FW3 branch (2026-09-26).
