# Q1 → forward-centered wiring: design & impact analysis

**Scope.** Read-only analysis; no code was modified.  The purpose of this
document is to specify how `MeasureAdjustment` (Q1 spec) should be
threaded into `ForwardCenteredModel` so that a drift-channel risk premium
becomes an actual CLI knob, without disturbing the forward-curve
calibration.

## 1. `risk_neutral.MeasureAdjustment` — full field / semantics table

`pde_option_model/risk_neutral.py:44-88`.

| field | type | shape | default | meaning |
|---|---|---|---|---|
| `spec` | `Literal["P","Q1","Q2"]` | — | `"Q1"` | which family of adjustments is admitted |
| `theta_shift` | `np.ndarray` | `(2,)` | `[0., 0.]` | δᵢ: regime-i long-run mean shift, θᵢ^Q = θᵢ^P + δᵢ |
| `drift_shift_per_hour` | `np.ndarray` | `(2,)` | `[0., 0.]` | aᵢ: regime-i raw drift shift, bᵢ^Q = bᵢ^P + aᵢ per hour |
| `market_price_of_risk` | `np.ndarray` | `(2,)` | `[0., 0.]` | λᵢ: bᵢ^Q = bᵢ^P − λᵢ σᵢ |
| `eta` | `np.ndarray` | `(2,)` | `[0., 0.]` | (η₀₁, η₁₀): generator premium, Q2 only |
| `calibrated` | `bool` | — | `False` | true only when a market-calibrated adjustment was produced |

Three-way equivalence of the drift channel (the module docstring makes
this explicit at lines 9-18):

```
theta_shift  δᵢ    ↔    drift_shift  aᵢ = κ · δᵢ    ↔    market_price_of_risk  λᵢ = −aᵢ / σᵢ
```

Because κ = −ln(φ) ≈ 3.852e−4 /h is tiny, `drift_shift_per_hour` is the
numerically better-conditioned knob and is the one the docstring
recommends exposing to the forward calibrator.

Q1 leaves the generator untouched (`adjust_generator` returns identity
unless `spec == "Q2"`, line 72-78) — no regime-switching premium in Q1.

## 2. Why `expected_spot()` reduces to F(t), and does Q1 break it?

### The centering identity as it stands today

`pde_option_model/forward_centered.py:416-427`:

```python
def expected_spot(self, query_hours, max_step_hours=1.0):
    q = np.atleast_1d(np.asarray(query_hours, dtype=float))
    mean, var, _, f = self.moments_at(q, max_step_hours)
    if self.spec.mode == "additive":
        return f + mean - mean                       # exact zero residual mean
    return f * np.exp((mean + 0.5 * var) - (mean + 0.5 * var))
```

`residual_summary` (line 429-446) does the same construction with a
named intermediate `cen`:

```python
cen = mom.mean if self.spec.mode == "additive" else mom.mean + 0.5 * mom.variance
expected = f + mom.mean - cen                        # additive
expected = f * np.exp(mom.mean + 0.5 * mom.variance - cen)   # multiplicative
```

The identity `expected == f` is enforced by construction — the same
`mean` computed by the moment ODE is subtracted from itself.  The
verification test at `tests/test_forward_centered.py:27-31`:

```python
def test_expected_spot_equals_the_market_forward(model):
    t = np.arange(0.0, 721.0, 1.0)
    s = model.residual_summary(t)
    dev = np.max(np.abs(s["expected_spot_TRY_MWh"] - s["forward_TRY_MWh"]))
    assert dev < 1e-9
```

passes to solver precision.

### Where does a Q1 drift shift enter the moment ODE?

Under Q1 with `drift_shift_per_hour = a = (a₀, a₁)`, the residual SDE
becomes

    dX_t = [ κ_X ( m_{J_t} − X_t ) + a_{J_t} ] dt + σ^X_{J_t}(t) dW_t.

The current moment ODE (`residual_moments`, line 242-253) already
tracks `u_i = E[X_t · 1{J_t=i}]` and `w_i = E[X_t² · 1{J_t=i}]`; the
drift term enters:

* `u_i` gets an additional `+ a_i · p_i` because
  `d(X 1_{J=i})/dt` picks up `a_i · p_i` from the added drift.
* `w_i` gets an additional `+ 2 a_i · u_i` because
  `d(X² 1_{J=i})/dt` picks up `2 · X · a_i · 1{J=i}`.
* `p_i` **unchanged** — Q1 does not adjust the generator, so regime
  probabilities evolve identically to P.

Now the total residual mean is

    μ_X(t) = u₀(t) + u₁(t) = ∫₀^t [ κ (m_i − u_i/p_i) + a_i ] p_i dt  (summed over i)

which absorbs the drift-shift contribution exactly.  Substituting into
the centering identity:

    E^Q[P_t] = F(t) + E^Q[X_t] − μ_X(t)
             = F(t) + μ_X(t)   − μ_X(t)
             = F(t).                                     ← still exact

**Conclusion.** The E^Q[P_t] = F(t) invariant is **preserved** under any
Q1 drift shift, *provided the moment ODE and the pricing PDE both
absorb the same aᵢ*.  The invariant would only break if we wired the
drift shift into one of the two channels but forgot the other — a
symmetry hazard, not a mathematical obstacle.

### What actually changes under Q1?

* **First moment** `E^Q[P_t]` — unchanged (= F(t)).
* **Variance** `Var^Q[X_t]` — **changes**, because the second-moment
  ODE gets the extra `+ 2 aᵢ · uᵢ` term.  Non-zero aᵢ moves the
  distribution around F(t) without moving its mean.
* **Regime probabilities** `p_i(t)` — unchanged (Q1 leaves the
  generator intact).
* **Option value** — **changes**, through the variance and higher
  moments of the residual PDE solution.

This is precisely the pattern the user wants: the drift premium
functions as a "shape control" for the option-value distribution while
leaving the forward-curve constraint untouched.

## 3. Recommended design (no code yet, structural spec only)

### 3.1 New optional argument on `ForwardCenteredModel`

Add an optional `adjustment: MeasureAdjustment | None = None` field to
`ForwardCenteredModel`.  Default `None` = physical measure = today's
behaviour, so every existing call site is unchanged.

Two conservative rules:

1. **Q1 only in the first cut.**  Reject `spec == "Q2"` with a clear
   error until the generator-adjustment path in `generator_path` is
   also wired (that's the eta channel; independent extension).
2. **Reduce to a single canonical parameterisation internally.**
   Convert (`theta_shift δ`, `drift_shift a`, `market_price_of_risk λ`)
   into an effective `a_eff_i` per regime:
   ```
   a_eff_i = a_i + κ · δ_i − λ_i · σ_y_i · scale_P   (or the delta-method-consistent σ^X)
   ```
   Store `a_eff` internally.  Documentation exposed only for one knob
   (`a` in per-hour TRY/MWh), keeping the API narrow.

### 3.2 Threading through the moment ODE

`residual_moments(spec, times_hours, q01, q10, pi0, x0, sigma_price_path)`
gains an optional `drift_shift_per_hour: np.ndarray | None = None`.
Inside the RK4 `deriv` closure:

```
du = np.array([
    k * (m[0] * p[0] - u[0]) + a_eff[0] * p[0] - q01t * u[0] + q10t * u[1],
    k * (m[1] * p[1] - u[1]) + a_eff[1] * p[1] + q01t * u[0] - q10t * u[1],
])
dw = np.array([
    2 * k * (m[0] * u[0] - w[0]) + 2 * a_eff[0] * u[0] + s2[0] * p[0] - q01t * w[0] + q10t * w[1],
    2 * k * (m[1] * u[1] - w[1]) + 2 * a_eff[1] * u[1] + s2[1] * p[1] + q01t * w[0] - q10t * w[1],
])
```

`ForwardCenteredModel.moments` / `moments_at` / `expected_spot` /
`residual_summary` / `centering` — all thread `self.adjustment.drift_shift_per_hour`
(or the derived `a_eff`) to `residual_moments`.

### 3.3 Threading through the pricing PDE

`price_forward_centered` (line 535+) currently passes the residual
`spec` and the model.  The drift coefficient inside `solve_coupled_pde`
must gain `+ a_eff_i` in the regime-i drift row.  Concretely: whatever
computes `κ_X · (m_i − x)` for the PDE stencil adds `+ a_eff_i` under
the adjustment.

The symmetry is what preserves the centering identity: **the same
`a_eff_i` fed to the moment ODE must be fed to the PDE**.  A tiny
integration test can guard this (see §5).

### 3.4 Multiplicative mode

In multiplicative mode `P_t = F(t) · exp(X_t) / E^Q[exp(X_t)]`, the
centering term is `μ_X + ½·Var^Q[X_t]` (line 434 of forward_centered.py).
Under a Q1 drift shift, both terms move by the same integral so
`E^Q[P_t] = F(t)` still holds.  No extra care needed if the
mean/variance ODE is faithfully adjusted.

## 4. Proposed CLI surface (specification only — no implementation)

Two new flags on the `price` sub-command, mirroring the drift channel:

```
--risk-premium-a0 <float>     Regime-0 (normal) drift shift, TRY/MWh per hour.
                              Enters both the moment ODE and the pricing PDE
                              as a symmetric add to the regime-i drift.
                              Preserves E^Q[P_t] = F(t) exactly; alters
                              variance and hence option values.
                              Default: 0.0 (physical measure, backward-compatible).

--risk-premium-a1 <float>     Regime-1 (stress) drift shift.  Same units,
                              default 0.0.
```

**Unit rationale.**  TRY/MWh per hour is the raw drift-shift unit of
the underlying SDE and requires no delta-method translation.  Small
numbers: for a 72 h horizon a shift of ±0.02 TRY/MWh/h moves the
integrated drift by ±1.4 TRY/MWh, which is enough to move a K=3000
call value by a few TRY/MWh.  The order-of-magnitude reference from
the (since removed) round-trip self-test was `(−0.004, −0.004)` — i.e.
tiny numbers, well-suited to a scalar flag.

**Deferred to a second increment (not in this design):**

* `--risk-premium-lambda0/-1`  the market-price-of-risk parameterisation.
  Requires deciding whether λ multiplies σ^y (y-space) or σ^X (X-space);
  keep out of the first cut.
* `--risk-premium-eta01/-10`  Q2 generator premium.  Requires wiring
  `MeasureAdjustment.adjust_generator` into `ForwardCenteredModel.generator_path`.

**Config-file counterpart.**  A `risk_premium: { a: [0.0, 0.0] }` block
in `config/forward_centered_config.yaml`, read by `run_pde.py cmd_price`
alongside the CLI flags, with CLI overriding config.  Same
default-zero convention.

## 5. Test impact preview

Tests that touch the centering identity or the moment machinery (grep
of `expected_spot`, `centering`, `forward_at`):

| test | file | what it asserts | behaviour under Q1 (a≠0) |
|---|---|---|---|
| `test_expected_spot_equals_the_market_forward` | `test_forward_centered.py:27` | `max |E[P_t] − F(t)| < 1e−9` over 720 h | **PASSES** by construction if the drift shift is symmetric across ODE and PDE; guards symmetry |
| `test_monthly_average_of_expected_spot_equals_the_vep_quote` | `test_forward_centered.py:34` | monthly averages hit VEP quotes to 0.10 TRY/MWh | **PASSES** for the same reason |
| `test_centering_works_with_nonzero_regime_means` | `test_forward_centered.py:44` | `max |E[P_t] − F(t)| < 1e−8` with regime_means=(150, −400) | **PASSES**; a and m are additive in the same ODE row |
| `test_centering_works_with_nonzero_initial_residual` | `test_forward_centered.py:54` | same tolerance under x0_mode="spot_minus_curve" | **PASSES** |
| `test_multiplicative_mode_is_also_centred` | `test_forward_centered.py:73` | same tolerance for multiplicative mode | **PASSES**; the μ + ½Var normalisation adjusts too |
| `test_expected_spot_never_reaches_implausible_magnitudes` | `test_forward_centered.py:99` | finite E[P_t] over 5000 h | **PASSES**; drift shift is bounded |
| `test_monthly_delivery_average_matches_quote` | `test_calibration_acceptance.py`, calibration acceptance | max abs monthly error < 0.10 TRY/MWh | **PASSES**; forward calibration is invariant to residual dynamics |
| `test_anchor_sensitivity_keeps_quoted_months_exact` | `test_calibration_acceptance.py:194` | quoted-month averages exact for every anchor level | **PASSES**; same reason |

**Recommended new test to add alongside the wiring:**

```python
def test_q1_drift_shift_preserves_centering(model_ctor):
    m = model_ctor(adjustment=baseline_q1(drift_shift_per_hour=(0.01, -0.02)))
    s = m.residual_summary(np.arange(0.0, 721.0, 1.0))
    assert np.max(np.abs(s["expected_spot_TRY_MWh"] - s["forward_TRY_MWh"])) < 1e-8
```

This is the guardrail against the symmetry hazard in §3.3: if a
future refactor breaks the ODE↔PDE symmetry, this test fires.

Also worth adding, but strictly optional at first cut:

```python
def test_q1_drift_shift_changes_variance_but_not_mean(model_ctor):
    m0 = model_ctor()
    m1 = model_ctor(adjustment=baseline_q1(drift_shift_per_hour=(0.02, -0.02)))
    s0, s1 = m0.residual_summary(t), m1.residual_summary(t)
    assert np.allclose(s0["expected_spot_TRY_MWh"], s1["expected_spot_TRY_MWh"], atol=1e-8)
    assert np.max(np.abs(s0["residual_std_TRY_MWh"] - s1["residual_std_TRY_MWh"])) > 1e-3
```

## 6. Summary

* Q1's drift channel is **compatible** with the forward-curve
  calibration: the centering identity `E^Q[P_t] = F(t)` is preserved
  exactly, so long as the drift shift is threaded into **both** the
  moment ODE and the pricing PDE.
* The variance ODE (and therefore option values) *does* change with
  aᵢ, which is exactly the desired handle: the drift channel becomes
  a "risk-premium knob for option prices without touching the forward
  strip".
* Recommended API: a single symmetric pair `--risk-premium-a0 / -a1`
  in TRY/MWh per hour, default zero, exposed via `run_pde.py price`
  and optionally through the config file.
* Recommended safety net: one new unit test that runs a non-zero
  drift shift and asserts the centering identity, guarding the
  ODE↔PDE symmetry.
* Q2's generator channel (`eta01/eta10`) and the λ parameterisation
  are deferred to a second increment; they require touching
  `generator_path` and the σ↔λ mapping respectively, which are
  scope-separate from the drift wiring.

No production file was modified while writing this document.
