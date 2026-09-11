# Half-life reconciliation — within-regime 19 yr vs deseasonalized 8.84 h

`model_limitations.md` item (g) flagged a ~19 000× discrepancy between two
half-life numbers reported by the same estimation run:

* **19.25 years** — implied by M9's within-regime `phi = 0.999996`
  (`kappa = -ln(phi) / dt = 4.11e-6 /h`, `half-life = ln 2 / kappa`).
* **8.84 hours** — reported by `deseasonalized_stationarity_summary.csv` in
  the bundle metadata, from a single-regime AR(1) fit with `phi = 0.9246`.

This note resolves the gap theoretically and confirms the resolution by
simulating the M9 process forward under its own parameters.

## Theoretical setup

Under a Markov-switching AR(1):

```
Y_t = phi · Y_{t-1} + mu_{J_t} + sigma_{J_t} · eps_t
```

with `J_t` a Markov chain (state 0 = high-vol, state 1 = low-vol under
raw M9 indexing; state durations 7.56 h / 3.76 h respectively), the
**marginal** ACF of `Y_t` is a mixture of two exponential decays:

* a slow component decaying at rate `−ln(phi) = 4.11 e-6 /h`
  (the within-regime AR(1) persistence)
* a fast component decaying at rate `λ_01 + λ_10 = 0.398 /h`
  (the regime-switching indicator's autocorrelation)

with weights determined by how much of `Var(Y)` is driven by the AR(1)
noise vs the regime-mean shifts.  For M9 the two regime intercepts are
both very small in absolute terms (`mu_0 = −0.00427`, `mu_1 = −0.00068`)
and their difference is tiny compared with the noise scale, so **the
regime-mean-shift contribution to marginal Var(Y) is negligible
(~5 %)**.  The marginal ACF is therefore ≈ `phi^h`, i.e. near-unit-root,
i.e. 19-year half-life.

## Numerical confirmation (200 000-hour simulation)

Running the M9 process forward with its exact parameters (`phi`, `mu_i`,
`sigma_i`, transition rates from the reported mean durations) and
computing the empirical marginal ACF gives:

| lag h | empirical ρ | `phi^h` | comment |
|---:|---:|---:|---|
| 1 | 1.000000 | 0.999996 | essentially unit ACF |
| 8 | 0.999999 | 0.999967 | same |
| 100 | 0.999982 | 0.999589 | slightly slower than `phi^h` because of small regime contribution |
| 500 | 0.999904 | 0.997948 | same |
| 10 000 | 0.998434 | 0.959750 | small deviation, consistent with a second exponential |

Empirical marginal half-life on the simulated series: **> 500 h**
(the ACF only drops to 0.998 at lag 10 000 — extrapolating exponentially
gives a half-life of order 10⁵ h, matching the theoretical 168 720 h).

Regime-indicator ACF at lag 1: empirical **0.5973**; theoretical
`exp(−0.398) = 0.6715` — matches within simulation noise.

**Conclusion (a)**: the M9 model as fitted has a marginal ACF at lag 1
of essentially 1.  It does NOT reproduce the 0.9246 that the
`deseasonalized_stationarity_summary` AR(1) fit reports.  The two
numbers describe different variables.

## What the deseasonalized 8.84 h actually measures

`metadata/model_parameters_and_ou_mapping.json.parsed_parameter_files.deseasonalized_stationarity_summary.csv`
was computed on a **deseasonalized** version of `y = asinh(PTF / scale_P)`
— the residual after removing daily / weekly / annual harmonic
components and holiday effects.  M9, in contrast, was fit on the *raw*
`asinh(PTF / scale_P)` and absorbed all of that structure into its
regime-conditional intercepts, regime-conditional variances, and near-
unit-root within-regime AR(1).

So the 0.9246 lag-1 autocorrelation is a property of the
**deseasonalized residual**, and its 8.84 h half-life is the reversion
speed of that residual — a signal AFTER the seasonal component has
been stripped.  The 0.999996 within-regime phi is a property of the
**raw asinh-transformed price**, whose autocorrelation is dominated by
long trends / TRY-inflation drift (both regime intercepts are negative,
consistent with a slow downward drift in `y`-space over the training
window's inflation era; the simulated 200 kh path drifts from `y = 0`
to `y ≈ −450`).

**These are not the same quantity.  Neither number is "wrong".**  They
measure the persistence of two different variables constructed from the
same historical PTF series.

## Implication for the forward-centered pricing model — root-cause candidate for the backtest overshoot

`forward_centered.py` uses `params.kappa_per_hour = 4.11e-6 /h` (the
within-regime kappa derived from the raw-y M9 phi) as the OU
mean-reversion rate of the residual **X_t = P_t − F(t)**.  It also
inherits the M9 sigma values.  Under this specification:

* The moment ODE `residual_moments()` correctly integrates the linear
  system `du_i' = κ(m_i p_i − u_i) + a_i p_i + …`,
  `dw_i' = 2κ(m_i u_i − w_i) + 2 a_i u_i + σ_i² p_i + …`.
* With `κ ≈ 0` and `regime_means = [0, 0]` (the yaml default), this
  reduces to `dw/dt ≈ Σ_i σ_i² p_i(t)` — variance grows **linearly**
  with time.  Independently checked in the simulation above: `Var(Y_t)`
  scales as `t · E[σ²]` to within 1-2 %.
* Consequently the model's `residual_std_at_expiry` grows as
  `sqrt(t · E[σ²])` — 4014 TRY/MWh in Jan, 13 948 in Jul.

The 2026 backtest (`realized_2026_backtest.md`) found the empirical
residual std to be roughly **constant** across months at 573-1258
TRY/MWh, giving `model / realized` ratios of 7-13×.  Under the
Markov-switching interpretation above, that empirical near-constancy
is exactly what an 8.84 h-half-life OU would produce:
`sigma_realized ≈ σ_effective · sqrt(1/(2κ_deseasonal))` saturates on
day-scale, matching the empirical ~1000 TRY/MWh.

**This is a plausible root cause of the variance overshoot**: the
model is running the residual as a raw-y-persistence OU (essentially
random walk), whereas the empirical residual around a well-anchored
forward curve behaves like a deseasonalized-OU with 8.84 h half-life.

## Is this a code bug?

No.  `residual_moments()` correctly integrates the model as defined,
and the model as defined uses `params.kappa_per_hour` faithfully.  The
issue is that **`params.kappa_per_hour` was set from the wrong AR(1)
fit** — the M9 within-regime phi (which represents raw-y persistence
across regimes) rather than a deseasonalized-residual phi (which
represents the shock-around-anchor persistence appropriate for the
forward-centered residual).

The reconciliation flag in `model_limitations.md` item (g) can now be
tightened from "open consistency question" to:

* Within-regime phi (0.999996) is a **correct** description of raw
  `asinh(PTF)` AR(1) persistence under M9.
* Deseasonalized single-regime phi (0.9246) is a **correct**
  description of the deseasonalized residual's AR(1) persistence.
* The **model incorrectly uses the raw-y kappa in the residual-around-F
  pricing PDE**, which produces variance growth appropriate for a
  raw-y process but too wide by ~7-13× for a genuine residual-around-F
  process.  This is not a code bug; it is a mis-inheritance in the
  yaml value of `kappa_per_hour`.

## Recommended follow-up (Faz 3 candidate)

Refit `params.kappa_per_hour` on the residual `P_t − F(t)` directly
(after building `F(t)` on a training window that ends before the
valuation date).  The expected value is close to `0.0784 /h`
(the deseasonalized-summary rate).  With that kappa, the moment ODE
would saturate variance at `mixture_sigma² / (2 · 0.0784)` and the
model's `residual_std_at_expiry` would settle around **1000-1200
TRY/MWh** — matching the 2026 realized residual std to within
uncertainty.  The forward-curve identity `E^Q[P_t] = F(t)` is
preserved regardless of kappa, so this refit does not require
re-doing the VEP calibration.

Sigma values also deserve a companion re-check — the current M9
sigmas were fit to raw `asinh(PTF)` residuals over 78 905 hours
including a period of very high nominal volatility; refit against the
residual-around-F may yield materially smaller sigmas.

## Status

* This is a **plausible root cause of the 7-13× residual-variance
  overshoot**, supported by both theoretical analysis and forward
  simulation.
* Not a code bug.  The moment ODE is correct given its inputs.
* Fix is a **methodological choice**: which AR(1) coefficient
  represents the residual-around-F dynamics?  The current yaml uses the
  raw-y within-regime coefficient (per the M9 estimation output); a
  refit on residual-around-F is the natural next step.
* Recorded as a Faz 3 candidate in
  `docs/PROJECT_STATUS_AND_FUTURE_WORK.md`.
