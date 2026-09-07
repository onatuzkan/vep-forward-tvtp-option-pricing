# Pooled single-volatility baseline vs M9 (TVTP) — regime-conditioning value

Prices a `K = 3000` European call at 24 / 72 / 168 / 336 h under the shipped
M9 production model (two regimes + TVTP driven by lagged standardised RD)
against a **regime-agnostic single-volatility baseline** derived by pooling
M0's two fitted volatilities.  Same forward curve, same `pi_filtered`, same
climatology `z(t-1)` path — only the residual dynamics differ.

## What this comparison is (and is not)

**Important scope note.**  M0's own constant-transition-probability
parameters (its `alpha`, `gamma` — i.e. the intercepts and slopes of its
two-state transition logistic) are **not present** in this handoff:
`inputs/historical/archive/calibration_bundle/transition_coefficients.csv`
carries only M9 rows.  Consequently **we did not run M0 as fitted**.  What
we ran instead is a "regime-agnostic pooled baseline": M0's two sigmas
`(sigma0, sigma1)` collapsed to a single effective volatility via
occupancy-weighted averaging, treated as a one-regime AR(1).

The comparison therefore isolates the value of **regime-conditioning
volatility during pricing** (i.e. holding a filtered belief about which
regime the process is currently in, and letting a Markov chain evolve
that belief), against a model that has neither state nor transitions.
It does **not** — see caveat (b) below — isolate the value of
**time-varying transition probabilities specifically**.

## Method

M0 in the bundle is a two-regime **constant-transition** Gaussian AR(1)
fit (spec = `M0_constant`).  Its own long-run occupancy is not exported,
so to pool its two volatilities to a single-regime AR(1) we use M9's
stationary occupancy `(pi_normal, pi_stress) = (0.325, 0.675)` as the
best available proxy for "how much time this data-generating process
spends in each state".

Numerical inputs (from `parameter_estimates.csv`):

* `M0.sigma0 = 0.006123` (yaml index 0 = normal, low-vol)
* `M0.sigma1 = 0.164255` (yaml index 1 = stress, high-vol)
* `M0.phi   = 0.999181`  → kappa = 8.20e-4 /h, half-life 846 h
* pooled sigma = `sqrt(0.325 · 0.006123² + 0.675 · 0.164255²) = 0.13495`

Two pooled-baseline variants are priced so the reader can separate the
regime-conditioning effect from the mean-reversion timescale effect:

| variant | sigma | kappa | notes |
|---|---:|---:|---|
| `pooled_M0_kappa`  | 0.13495 | 8.20e-4 /h (M0's own) | pooled sigma + M0's fitted kappa |
| `pooled_M9_kappa`  | 0.13495 | 3.85e-4 /h (M9's) | pooled sigma + M9's kappa (isolates the regime-conditioning effect from the kappa change) |
| `M9_prod`          | (0.00353, 0.09241) | 3.85e-4 /h | production two-regime TVTP |

The 1-regime collapse is realised inside the existing 2-regime pipeline
by setting `sigma_normal = sigma_stress = pooled` (differing by ±0.01 %
to satisfy the `sigma[1] > sigma[0]` invariant); the PDE solutions in
each regime then coincide and the regime chain becomes vacuous.

## Results — `K = 3000` call, PDE-only

| maturity | `pooled_M0_kappa` | `pooled_M9_kappa` | **`M9_prod`** | Δ(M9 − pooled_M0_κ) | M9 vs pooled_M0_κ % | Δ(M9 − pooled_M9_κ) | M9 vs pooled_M9_κ % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 h  | 724.09  | 728.07  | **369.71**  | −354.38  | **−48.94 %** | −358.35  | −49.22 % |
| 72 h  | 1254.65 | 1274.78 | **685.62**  | −569.03  | **−45.35 %** | −589.16  | −46.22 % |
| 168 h | 1854.31 | 1922.42 | **1052.77** | −801.54  | **−43.23 %** | −869.65  | −45.24 % |
| 336 h | 2450.70 | 2626.63 | **1449.49** | −1001.21 | **−40.85 %** | −1177.14 | −44.82 % |

Residual standard deviation at expiry (TRY/MWh):

| maturity | `pooled_M0_kappa` | `pooled_M9_kappa` | `M9_prod` |
|---:|---:|---:|---:|
| 24 h  | 1919.0 | 1929.0 | 1041.5 |
| 72 h  | 3259.3 | 3310.0 | 1834.0 |
| 168 h | 4790.9 | 4963.0 | 2769.8 |
| 336 h | 6349.8 | 6797.7 | 3803.8 |

## Interpretation

**The M9 model is systematically 41-49 % cheaper than the pooled
single-volatility baseline across the tested maturities, and the gap
SHRINKS with horizon** (48.9 % at 24 h → 40.9 % at 336 h).

The mechanism is a *regime-conditioning* effect, not a
*transition-timing* effect:

1. **Short-horizon advantage from the initial regime belief.**  The
   filtered valuation-time state is `pi_filtered = (0.932, 0.068)` —
   heavily concentrated in the low-vol *normal* regime.  M9's
   instantaneous mixture-variance rate is therefore only
   `0.932 · 0.00353² + 0.068 · 0.0924² ≈ 5.9 · 10⁻⁴`, versus the pooled
   baseline's `0.135² = 1.82 · 10⁻²` — a **31 × lower** variance rate at
   `t = 0`.  This is why the 24 h call under M9 is barely half of the
   baseline's.

2. **Long-horizon convergence.**  As the (TVTP) chain drifts from
   `pi_0 = (0.932, 0.068)` toward the stationary `(0.325, 0.675)`
   (dominant stress state), M9's mixture-variance rate rises to
   `≈ 5.8 · 10⁻³`, closing the gap.  The residual-sd column shows this
   directly: `sd_M9 / sd_pooled` climbs from `0.54` at 24 h to `0.60` at
   336 h.

3. **The term-structure asymmetry is a hallmark of regime-conditioning.**
   A single-regime AR(1) has a flat effective volatility — it cannot
   encode "we are currently in a calm state, so near-dated options should
   be cheap, but longer-dated options must reflect the possibility of
   transitioning into the stress state".  The 8-pp shrinking of the
   percentage gap from 24 h to 336 h is the quantitative signature of
   that asymmetry.

4. **The kappa difference is a secondary effect.**  Comparing
   `pooled_M0_kappa` vs `pooled_M9_kappa` isolates the kappa change
   alone (same pooled sigma).  Slower mean-reversion (M9's kappa)
   modestly raises the call (variance persists longer) by 0.5-7 %.  This
   is roughly one order of magnitude smaller than the regime-conditioning
   effect (41-49 %), so the majority of the observed gap is attributable
   to regime-conditioning itself.

## What this comparison does and does not justify

* **JUSTIFIES: the regime-conditioning ingredient.**  Holding a filtered
  belief `pi_filtered` about the current regime and letting a Markov
  chain evolve that belief materially changes the option value versus a
  pooled single-volatility residual — a first-order effect (~45 %), not
  a rounding-level refinement.

* **DOES NOT JUSTIFY (on this data alone): the TVTP-specific
  ingredient.**  A *constant-transition* two-regime model with the same
  `pi_filtered = (0.932, 0.068)` would also start with the low-vol
  mixture-variance rate and would also drift toward some stationary
  occupancy; its call-price term structure would resemble M9's in
  qualitative shape.  Whether TVTP (transitions that depend on lagged
  standardised RD) delivers a **further** material improvement over
  constant transitions is a *separate* ablation that requires M0's own
  alpha / gamma parameters.  Those parameters are not in this handoff
  bundle, so the ablation is deferred (recorded as a future-work item in
  `docs/PROJECT_STATUS_AND_FUTURE_WORK.md`).

## Caveats

* **(a) Pooled sigma uses M9's stationary occupancy as a proxy.**  The
  M0-specific occupancy is not exported.  If M0's own stationary
  occupancy differs materially, the pooled sigma moves and the numerical
  gap does too; the qualitative picture (baseline too high at short
  horizons, closing gap at long horizons) is unlikely to reverse because
  it is a direct consequence of `pi_filtered` starting in the low-vol
  regime.
* **(b) The comparison target is a single-regime pooled baseline, not
  the fitted M0.**  M0 as fitted is a two-regime constant-transition
  model whose transition coefficients we do not have.  Attributing the
  full 45 % gap to "TVTP" would over-claim; the gap is attributable to
  regime-conditioning + TVTP jointly, and the TVTP-specific portion is
  not isolated by this test.
* **(c) Kappa mismatch.**  `pooled_M0_kappa` uses M0's fitted
  `phi = 0.99918`; `M9_prod` uses M9's `0.999996`.  The `pooled_M9_kappa`
  variant controls for this and shows the effect is overwhelmingly
  regime-structural, not kappa-driven.
* **(d) Forward calibration is unaffected.**  `E^Q[P_t] = F(t)` in every
  variant — the centering identity is invariant to residual sigma.

Data source: `outputs/market_calibration_final/model_comparison_pooled_vs_M9.csv`.
No production code was touched; the pooled-baseline collapse is realised
by pipeline configuration only (equal regime sigmas).  Full test suite
(172 passing) is unchanged by this analysis.
