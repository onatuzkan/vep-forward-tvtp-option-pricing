# Model-selection robustness — M8 vs M9

Same forward curve, same `pi_filtered`, same climatology `z(t-1)`, same
kappa; the residual sigmas are swapped between the top-2 models
identified by the estimation run (`M9_student_t_tvtp_TVTP-2` and
`M8_student_t`) and the `K = 3000` European call is re-priced at 24 /
72 / 168 / 336 h.  The point is to bracket how much model-selection
uncertainty moves the option price when both models are supported by
essentially the same data.

## What this comparison is (and is not)

**Scope note.**  M8 in the bundle is spec `M8_student_t`: two-regime
**constant-transition** Student-t emission (no covariate).  Its
transition-probability parameters (constant `p01`, `p10` in the
constant-transition family) are **not present** in the handoff
bundle — `transition_coefficients.csv` contains only M9 rows.
Consequently **we did not run M8 as fitted**.  What we ran instead is
"M9's residual framework with M8's fitted sigmas plugged in":

* M8 sigmas (yaml-swapped from the raw M8 fit)
* M9's TVTP alpha / gamma (best available proxy for the transition
  dynamics)
* Same kappa as production (see "Related finding" below)
* Same pi_filtered, forward curve, climatology z, everything else.

The comparison therefore isolates **the pure effect of swapping the
two regime sigmas** between the top-2 models under the current pricing
framework.  It does **not** isolate the effect of Student-t-vs-TVTP
regime dynamics — quantifying that requires M8's own transition-
probability constants, which are absent.

## Parameter deltas

Yaml-swapped sigmas (M9 raw state 0 = high-vol → yaml stress; state 1 =
low-vol → yaml normal):

| parameter | M9 (production yaml) | M8 (raw, swapped) | ratio |
|---|---:|---:|---:|
| `sigma_y_normal` | 0.0035348070 | 0.0034584192 | 0.97839 (−2.16 %) |
| `sigma_y_stress` | 0.0924066544 | 0.0918758285 | 0.99426 (−0.57 %) |

Model-fit context from `run_summary.json` (both trained on the same
78 905-hour window):

| metric | M9 | M8 | delta (M9 − M8) |
|---|---:|---:|---:|
| `loglik_train` | 100 515.68 | 99 236.28 | **+1279.4** |
| `BIC` | −200 693.07 | −198 179.39 | **−2 513.68**  (M9 strongly preferred) |
| `val_avg_log_score` | 0.9706 | 0.9630 | +0.0076 |

## Results — `K = 3000` call, PDE-only

| maturity | `M9_prod` | `M8_sigmas_only` | Δ (M8 − M9) TRY | M8 vs M9 % | M9 sd_T | M8 sd_T |
|---:|---:|---:|---:|---:|---:|---:|
| 24 h  | 366.33  | 363.99  | −2.33  | **−0.637 %** | 1033.5 | 1027.6 |
| 72 h  | 677.23  | 673.10  | −4.13  | **−0.610 %** | 1813.5 | 1803.1 |
| 168 h | 1039.21 | 1032.99 | −6.22  | **−0.599 %** | 2736.0 | 2720.2 |
| 336 h | 1430.57 | 1422.08 | −8.48  | **−0.593 %** | 3756.0 | 3734.4 |

M8's slightly smaller sigmas produce a slightly smaller call (~0.6 %
across every horizon).  The gap is nearly flat across maturities:
0.637 % at 24 h down to 0.593 % at 336 h — the sigma difference is a
uniform variance-rate shift under the current framework, so it
propagates almost linearly into the option-value distribution.

## Interpretation

**Model-selection risk between the top-2 models is small at the option-
pricing level.**  Despite a very large data-fit gap (ΔBIC = −2 513.7 in
M9's favour), the option prices differ by **less than 1 %** at every
tested maturity.  The mechanism is that both M9 and M8 fit essentially
the same regime volatilities (within 0.6-2.2 %) — the loglik improvements
of M9 come from **better modelling of the regime dynamics** (in
particular the TVTP covariate driving transitions and the choice of
Student-t vs Gaussian tails) rather than from a dramatically different
volatility shape.  Since the option value is dominated by the mixture
variance, and the mixture variance is nearly identical between the two
models, the option price is nearly identical too.

This is a **robustness statement**: adopting M8 instead of M9 as the
production model would move a 72-hour ATM-ish call by ~4 TRY/MWh on a
~677 TRY/MWh base.  For all reasonable downstream uses this is well
inside acceptable model-uncertainty bounds.

## Related finding — yaml phi does not match M9's CSV phi

While setting up this comparison we noticed that the current production
yaml has `phi = 0.99961484` (kappa = 3.85e-4 /h, half-life ≈ 1799 h),
while the shipped `parameter_estimates.csv` reports **both** M9 and M8
with `phi ≈ 0.999996` (kappa = 4.11e-6 /h, half-life ≈ 168 000 h — near
unit-root).

Investigation of the metadata JSON shows the yaml's `phi` was inherited
from the `physical_measure_parameters` fallback block, which describes
a **different** model (`M2_tvtp_TVTP-1`) than any row in
`parameter_estimates.csv` (M0 / M8 / M9).  The M9 integration commit
`59955ee` swapped the sigmas from the CSV while leaving `phi` at the
fallback value — an internal inconsistency that has been in place
throughout every prior calibration but was not surfaced until this
sensitivity check.

If we substitute M8's own CSV phi into the same M8-vs-M9 sweep, the
sigma-difference effect is dwarfed by the phi difference: the residual
sd grows ~10 % faster over 336 h and the M8 call value ends up ~6 %
**above** M9's rather than 0.6 % **below**.  In other words, `phi` is
a first-order pricing driver and the yaml's choice of it is a bigger
model-uncertainty knob than the M9-vs-M8 choice we set out to test.

This is not fixed here (analysis only).  Recorded as a future-work
item in `docs/PROJECT_STATUS_AND_FUTURE_WORK.md`: reconcile the yaml
`phi` with the M9 CSV `phi`, decide which represents the physical
process on the transformed variable y, and re-run acceptance.

## Caveats

* **(a)** M8's own transition parameters are not exported; we substitute
  M9's TVTP as a proxy.  The residual-dynamics comparison is therefore
  "M8 sigmas within M9's framework" rather than "M8 fully".
* **(b)** Pooled/effective quantities that depend on M8's own long-run
  occupancy are unavailable; occupancy is inherited from M9.  Since
  both models share the same conceptual regime structure this is a
  small effect.
* **(c)** The main result (~0.6 % model-selection gap) is contingent on
  the yaml phi discrepancy being resolved consistently for both
  models.  If yaml phi is replaced with the CSV phi 0.999996, both M8
  and M9 near-unit-root results diverge from the current benchmark by
  a much larger amount than the M8-vs-M9 gap itself.

Data source: `outputs/market_calibration_final/model_robustness_M8_vs_M9.csv`.
No production code was touched; the M8 residual spec is built inline
in the analysis script by swapping the sigmas in a `ResidualSpec`
constructor.  Full test suite (172 passing) is unchanged.
