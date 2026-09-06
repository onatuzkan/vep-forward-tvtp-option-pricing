# Model limitations and assumption inventory

Label of every price produced here: **VEP-forward-curve anchored option prices**.

## What is genuinely constrained by the market

| quantity | status |
|---|---|
| hourly forward level F(t) inside 2026-02, 2026-03, 2026-04, 2026-05, 2026-06, 2026-07 | **constrained** — monthly averages reproduce the VEP quotes exactly |
| price level in 2025-12, 2026-01 | **assumed** — near-term anchored, no observed quote |
| residual volatility (σ_normal, σ_stress) | **inherited** from the historical M2 fit, not market-implied |
| regime transition dynamics (TVTP) | **inherited** from the historical fit |
| volatility risk premium | **not identified** — needs option premia |
| regime-transition premia η01, η10 | **not identified** — needs option premia |
| market price of risk λ_i | **not identified** — needs option premia |
| discount rate | **assumed** flat annual rate |

## Results that rest on an assumption, not on data

1. **January 2026 (`spot_flat` anchor).** near-term anchored, not directly constrained by an observed January VEP quote. Every horizon below falls inside it, so all four reported expected spots are anchored rather than market-constrained:

   - 72 h → 2026-01 — near-term anchored
   - 168 h → 2026-01 — near-term anchored
   - 336 h → 2026-01 — near-term anchored
   - 720 h → 2026-01 — near-term anchored

2. **Option prices depend on inherited volatility.** The forward calibration is exactly invariant to σ (the centering ODE has no σ term), so a wrong σ cannot break the monthly fit — but it moves every option value one-for-one. Option prices are therefore *level-anchored, volatility-assumed*.

3. **Residual dispersion inherits a near-unit-root κ.** With κ = 3.852e-04/h (half-life 1799 h) the residual standard deviation keeps growing over the horizon:

| horizon | F(t) (TRY/MWh) | residual sd (TRY/MWh) | sd / F |
|---|---|---|---|
| 72 h | 2917.8 | 3226.9 | 1.11 |
| 168 h | 2917.8 | 4865.0 | 1.67 |
| 336 h | 2917.8 | 6680.7 | 2.29 |
| 720 h | 2918.0 | 9137.0 | 3.13 |

   In the **additive** residual mode this admits negative simulated prices at long horizons — economically wrong for PTF, which is floored at zero. The additive mode is appropriate at day-ahead to few-week horizons; use `residual_mode: multiplicative` for month-scale work, where prices stay positive by construction. Either way the monthly forward fit is unaffected.

4. **The regime split is historical, not market-implied.** Regimes here describe the *residual* around the market curve, not the price level: regime 0 is the calm state in which the spot tracks the forward closely, regime 1 the stress state with ~23× the residual volatility. Nothing in the VEP data identifies how the market prices that stress risk.

## Legacy model

> legacy_asinh_ou: E[P_t] = scale_P * exp(v(t)/2) * sinh(m(t)) grows exponentially in the variance v(t) = sigma^2 (1 - e^{-2 kappa t}) / (2 kappa). With the fitted near-unit-root kappa and the stress-regime sigma this makes long-horizon expected prices economically meaningless. Use the legacy mode for short-horizon benchmarking and diagnostics only.

`legacy_asinh_ou` remains available and unchanged for short-horizon benchmarking, and its long-horizon output is flagged everywhere it is produced.

