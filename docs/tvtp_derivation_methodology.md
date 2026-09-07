# M9 → yaml Convention TVTP Derivation

**Draft only.** No production file was modified. Nothing is committed.

Purpose: (a) re-label the M9 fit under the yaml convention (`index0=normal`,
`index1=stress`), and (b) recover the missing `alpha01`, `alpha10` from the
long-run duration/occupancy diagnostics reported alongside the M9 fit.

Artefacts:
* `_derive.py` — the derivation script
* `derived_tvtp_parameters.yaml` — the draft parameter set
* `derivation_summary.json` — machine-readable diagnostic block

---

## Step 0 — Convention verification

**Does the CSV's `gamma01`/`gamma10` refer to raw M9 state indices (state0 = high-vol
"stress", state1 = low-vol "normal") or to some semantic labelling?**

* `pde_option_model/generator.py:6-7` documents the fitting-time convention
  explicitly:
  ```
  p01_t = logistic(alpha01 + gamma01 * z_{t-1})
  p10_t = logistic(alpha10 + gamma10 * z_{t-1})
  ```
  where `p01` is defined as the one-step probability of moving from raw state 0
  to raw state 1. The subscripts refer to the state indices of the **fitted
  model**, not to any downstream re-labelling.
* No schema note in `metadata/model_parameters_and_ou_mapping.json` or in the
  CSV itself contradicts or overrides this. The `parsed_parameter_files` block
  merely lists column names; there is no "how to interpret gamma01" comment
  anywhere in the bundle.
* `pde_option_model/markov_adapter.py:14-17, 91-106` explicitly acknowledges
  that "the USD run's REGIME LABELS ARE FLIPPED relative to the M2 convention:
  shipped state 0 is the high-volatility state". This is the labelling used
  when the M9 fit was carried out; the CSV was written under that labelling.

**Verdict.** Standard convention assumed: `gamma01 = ∂ logit(P(state0→state1)) / ∂z`
with `state0`, `state1` referring to the **raw M9 state indexing** (state 0 =
high-vol, state 1 = low-vol). Any downstream code that intends to use the M9
gammas under the yaml convention (state 0 = normal = low-vol) must swap them.

---

## Step 1 — Regime-label swap

### Yaml convention
* `index 0 = normal (low sigma)`
* `index 1 = stress (high sigma)`
* Enforced by `pde_option_model/params_frozen.py:74-77`: `sigma_y[1] > sigma_y[0]`.

### Mapping table

| Yaml field | Raw M9 source | Numerical value |
|---|---|---:|
| `sigma_y_normal` (index 0) | raw `sigma1` (low-vol state) | **0.003534807** |
| `sigma_y_stress` (index 1) | raw `sigma0` (high-vol state) | **0.092406654** |
| `occupancy_normal` | raw `occupancy_state1` | 0.325422 |
| `occupancy_stress` | raw `occupancy_state0` | 0.674578 |
| `mean_duration_normal_h` | raw `mean_duration_state1` | 3.759199 |
| `mean_duration_stress_h` | raw `mean_duration_state0` | 7.559716 |
| `gamma01` (normal → stress) | raw `gamma10` (state1 → state0) | **−0.583778** |
| `gamma10` (stress → normal) | raw `gamma01` (state0 → state1) | **+0.077698** |

### Invariant check
* `sigma_y_normal = 0.003534807`, `sigma_y_stress = 0.092406654`
* `sigma_y_normal < sigma_y_stress` → **True**
* `FrozenM2Parameters.__post_init__` invariant would pass unchanged.

### Sanity of the labelling
* Stress duration (7.56 h) is longer than normal duration (3.76 h) — this is
  the empirical outcome of the M9 fit and is *unusual* for typical "stress =
  brief spike" reasoning. Under M9's own statistics, the "high-volatility of
  hourly asinh returns" regime is actually the dominant, longer-lived one
  in the deseasonalized state; the label "stress" here is a volatility-class
  label, not a scarcity/spike label. Recorded in the yaml so downstream
  reviewers see the counter-intuitive numbers directly.

---

## Step 2 — Deriving the missing `alpha01`, `alpha10`

The bundle does not export intercepts. Given the yaml-convention gammas above
and the reported duration diagnostics, we back out the alphas as the unique
values that make the **empirical average** transition probability match the
target hourly rate.

### Targets (from the diagnostics of the M9 run)

* Normal → stress: `target_p01 = 1 / mean_duration_normal_h = 1 / 3.7592 = 0.26602`
* Stress → normal: `target_p10 = 1 / mean_duration_stress_h = 1 / 7.5597 = 0.13228`

### Root-finding

`z` = the full historical `inputs/historical/rd_standardized.csv`
(`87 665` hourly observations, `datetime` and `z` columns; the exact series
that drives the TVTP transition covariate in production).

Equations solved separately (gammas fixed, alphas unknown, monotone in α):

```
E_z[ σ(alpha01 + gamma01 * z) ] = target_p01
E_z[ σ(alpha10 + gamma10 * z) ] = target_p10
```

where `σ` is the logistic function. Solved with `scipy.optimize.brentq` on the
bracket `[-20, 20]` (`xtol = rtol = 1e-12`).

### Results

| Parameter | Value | Achieved mean p | Target mean p |
|---|---:|---:|---:|
| `alpha01` | **−1.015666** | 0.266015 | 0.266015 |
| `alpha10` | **−1.895229** | 0.132280 | 0.132280 |

The achieved means match the targets to solver precision (~1e-12).

---

## Cross-check — implied long-run stress occupancy

Two independent checks against the reported `occupancy_stress = 0.6746`:

### (a) Ergodic average-rate approximation

```
p̄01 = E_z[σ(alpha01 + gamma01 z)] = 0.26602
p̄10 = E_z[σ(alpha10 + gamma10 z)] = 0.13228
π̂_stress ≈ p̄01 / (p̄01 + p̄10) = 0.6679
```

**Error: −0.99 %** vs target 0.6746. Well inside the ±5 % consistency band
suggested in the brief.

### (b) Monte-Carlo simulation along the actual z path

Ten replicates of a 2-state Markov chain simulation using `p01(z_t)`,
`p10(z_t)` at every hour of the historical `z` series (seed 20260906):

```
mean fraction of hours in state 1 (stress) = 0.6441  ± 0.0026 (1σ)
```

**Error: −4.52 %** vs target 0.6746. Just inside the ±5 % band. The MC
estimate is systematically lower than the ergodic approximation because it
respects the actual persistence structure of `z` along time; the
average-rate approximation assumes independent draws of `z` per hour and
therefore over-estimates mixing.

### Interpretation
The reported `mean_duration_state0/state1` used as targets are themselves
**diagnostic summaries** of the smoothed state path, not exact moments of the
generating chain. A ~1-5 % discrepancy between the back-solved alphas and the
target occupancy is expected and is well within acceptable tolerance. Both
checks converge on the same qualitative picture: the derived (α, γ) pair
reproduces the reported occupancy to sub-percent accuracy under the ergodic
lens and to sub-5 % accuracy under a full simulation.

---

## Sign / magnitude comparison with the old placeholder

The current production yaml
(`inputs/historical/m2_frozen_parameters.yaml:31-34`) uses stand-in TVTP
coefficients that were chosen "to encode expected regime durations of
~300 h (normal) and ~20 h (stress) at z = 0, giving a stationary stress
share of ~6.3 %". Those targets are **very different** from the M9 numbers,
which imply 3.76 h / 7.56 h durations and 67 % stress occupancy.

| Coefficient | Old placeholder | New (derived M9) | Sign agrees? | \|new\|/\|old\| |
|---|---:|---:|:---:|---:|
| `alpha01` | −0.876728 | **−1.015666** | ✓ | 1.158 |
| `gamma01` | −0.680174 | **−0.583778** | ✓ | 0.858 |
| `alpha10` | −1.347666 | **−1.895229** | ✓ | 1.406 |
| `gamma10` | +0.203006 | **+0.077698** | ✓ | 0.383 |

All four **signs match** the old placeholder:
* `gamma01 < 0`: higher standardized RD reduces the normal→stress transition
  probability (economically: high RD is a stable regime driver).
* `gamma10 > 0`: higher standardized RD raises the stress→normal transition
  probability (mean-reversion signal).

Magnitudes:
* `|gamma01|` drops by 14 %.
* `|gamma10|` drops by **62 %** — a substantial reduction. The M9 fit says the
  stress-clearing signal from residual demand is much weaker than the
  placeholder assumed.
* `|alpha10|` grows by 41 % (more negative), pushing the baseline
  stress→normal rate lower. This is *required* for the reported 7.56 h stress
  duration under the yaml labelling: since the state is more persistent than
  the placeholder assumed, the intercept has to compensate.

The **direction of the two RD signals is unchanged**, but the M9 fit puts
much more of the regime-switching mass into the intercepts (persistence) and
less into the covariate sensitivities (RD-driven modulation).

---

## Downstream implication — sanity-check numbers if adopted

If the derived draft replaces the current yaml values (a separate decision;
NOT done in this task):

| Quantity | Current yaml | M9-derived draft | Ratio |
|---|---:|---:|---:|
| `sigma_y_normal` | 0.0075052 | 0.003534807 | 0.47 |
| `sigma_y_stress` | 0.1729706 | 0.092406654 | 0.53 |
| stress-regime variance rate | 0.02992 | 0.00854 | 0.29 |
| mixture-vol at `pi = (0.93, 0.07)` | 0.04554 | 0.02445 | 0.54 |
| stress-regime variance rate at `pi = (0.33, 0.67)` | 0.02015 | 0.00577 | 0.29 |

The residual OU variance rate would fall by roughly **~2×** under the M9
values at the current filtered `pi = (0.93, 0.07)`, or **~3.5×** if the
occupancy also flips to the M9 stationary distribution `(0.33, 0.67)`. Both
knobs feed into `ForwardCenteredModel.spec.sigma_price(F)` and hence into the
`residual sd at expiry` — the 3160 TRY/MWh figure currently seen in the
production `run_pde.py price` output. Any adoption of the draft should
re-run the full acceptance suite AND spot-check the option value change.

---

## Files in this draft folder

* `_derive.py` — reproducible derivation script (no external dependencies
  beyond scipy)
* `derived_tvtp_parameters.yaml` — the draft parameter set with full
  provenance comments
* `derivation_summary.json` — machine-readable summary of raw inputs, yaml
  outputs, and cross-check numbers
* `DERIVATION_REPORT.md` — this report

Production files unchanged: `inputs/historical/m2_frozen_parameters.yaml`,
`pde_option_model/params_frozen.py`, everything under
`outputs/market_calibration_final/*` outside `archive/`.
No commit; nothing staged.
