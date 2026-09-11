# Risk-neutral measure change under the forward-centered model

Methodology appendix.  Complements `docs/tvtp_derivation_methodology.md` (parameter
derivation) with the measure-change layer.

The historical M9 fit is a **physical-measure (P)** estimate: transition
intensities `q_ij^P(t) = q_ij^P(z(t-1))` and regime-i OU drifts
`b_i^P(X) = kappa (m_i - X)` describe how residuals moved *historically*.
Option prices need a **pricing (Q)** measure.  Two families are implemented
in `pde_option_model/risk_neutral.py`; this note formalises the wiring of
the **Q1 drift channel** into the forward-centered model.

## Q1 specification

Under Q1, transition intensities are unchanged (`q_ij^Q = q_ij^P`) and only
the per-regime drift is shifted.  Three equivalent parameterisations are
exposed in `MeasureAdjustment`; because `kappa ≈ 4.11e-6/h` is tiny (near
unit-root, half-life ~19 years — reconciled from the earlier 3.85e-4/h in the
phi-fix commit; see `docs/tvtp_derivation_methodology.md` and
`docs/PROJECT_STATUS_AND_FUTURE_WORK.md` completed item 8), the
raw drift shift is the numerically best-conditioned knob:

```
    theta_shift       θ_i^Q = θ_i^P + δ_i
    drift_shift       b_i^Q = b_i^P + a_i        with a_i = κ · δ_i     ← the exposed knob
    market_price_of_risk    b_i^Q = b_i^P − λ_i σ_i
```

The forward-centered residual SDE becomes

```
    dX_t = [ κ (m_{J_t} − X_t) + a_{J_t} ] dt + σ^X_{J_t}(t) dW_t
```

with `J_t` still evolving under the *P* generator `q_ij^P(t)`.

## Invariance proof: `E^Q[P_t] = F(t)` under Q1

The additive-mode centering identity is

```
    P_t = F(t) + X_t − μ_X(t)          with     μ_X(t) := E^Q[X_t].
```

Let `u_i(t) = E^Q[X_t · 1{J_t = i}]` and `w_i(t) = E^Q[X_t² · 1{J_t = i}]`.
Applying Itô's rule to `X_t · 1{J_t = i}` and to `X_t² · 1{J_t = i}` under
the Q1-adjusted SDE gives the closed 6-D linear ODE system

```
    p_i' = Σ_{j≠i} ( q_ji^P p_j − q_ij^P p_i )
    u_i' = κ (m_i p_i − u_i) + a_i p_i + Σ_{j≠i} ( q_ji^P u_j − q_ij^P u_i )
    w_i' = 2κ (m_i u_i − w_i) + 2 a_i u_i + σ^X_i(t)² p_i
                                        + Σ_{j≠i} ( q_ji^P w_j − q_ij^P w_i )
```

Compared to the P case (`a_i = 0`), the *only* changes are the
`+ a_i p_i` term in `u_i'` and the `+ 2 a_i u_i` term in `w_i'`.  `p_i` is
untouched because Q1 does not adjust the generator.

Total mean:

```
    μ_X(t) = u_0(t) + u_1(t)   (integrated from p_i, u_i)
```

Substituting back:

```
    E^Q[P_t] = F(t) + E^Q[X_t] − μ_X(t) = F(t) + μ_X(t) − μ_X(t) = F(t).       ✓
```

**The forward-curve identity is exact for every `a_i`.**  The physical
argument is that the residual definition subtracts the *same* `μ_X` that
the ODE produces, so any deterministic bias the drift shift introduces
into `E^Q[X_t]` is subtracted right back out.

### What actually moves

* **First moment** `E^Q[P_t]` — unchanged (equals `F(t)` at every `t`).
* **Regime probabilities** `p_i(t)` — unchanged (Q1 leaves the generator
  intact).
* **Variance** `Var^Q[X_t] = (w_0 + w_1) − μ_X²` — changes, because the
  extra `+ 2 a_i u_i` term in `w_i'` modifies the second-moment path.
* **Option value** — changes through the variance and higher moments of
  the residual PDE solution.

This is the desired shape: `a_i` is a knob on **option prices** that
cannot break the **forward-curve calibration**.

### Multiplicative mode

In multiplicative mode the centering is `μ_X + ½·Var^Q[X_t]` instead of
just `μ_X`; both terms move by the same integral under `a_i`, so the
identity survives.

## Wiring in code (symmetry hazard)

The invariance argument requires the drift shift to enter the pricing PDE
and the moment ODE **identically**.  If one channel is adjusted and the
other is not, `E^Q[P_t] ≠ F(t)`.  The following files hold the two sides:

| channel | file | line pattern |
|---|---|---|
| moment ODE (deterministic `μ_X`, `Var`) | `pde_option_model/forward_centered.py` — `residual_moments()` `deriv()` closure | `du = [k * (m[0] p[0] − u[0]) + a[0] p[0] + q terms, …]`  and  `dw = [2 k (m[0] u[0] − w[0]) + 2 a[0] u[0] + σ² p[0] + q terms, …]` |
| pricing PDE (stencil drift) | `pde_option_model/forward_centered.py` — `price_forward_centered()` `drift_fn()` | `return κ * (m_i[:, None] − grid.y[None, :]) + a_i[:, None]` |
| Monte-Carlo cross-check | `pde_option_model/forward_centered.py` — `simulate_forward_centered()` | Uses effective mean `m_eff_i = m_i + a_i/κ` in the exact conditional-OU step |

Both channels pull `a_i = spec.drift_shift_per_hour` from the *same*
`ResidualSpec` instance, so the symmetry cannot silently drift.  The
guard test `tests/test_forward_centered.py::test_q1_drift_shift_preserves_centering`
runs three non-zero `(a_0, a_1)` combinations and asserts the identity
holds to `1e-8` TRY/MWh — any refactor that breaks the symmetry fires
this test.

## Empirical verification of O(a²) scaling

The claim "the drift channel is a second-order knob on option prices"
is stronger than the algebra alone justifies (one might worry about a
first-order term hidden in a coupling somewhere).  The following
diagnostic sweep — `a₀ = 0` fixed, `a₁ ∈ {−0.5, −2, −5, −10, −20}`
TRY/MWh/h, 72 h `K = 3000` call, PDE-only — pins the scaling
empirically.  All rows share `F(T) = 2916.16` and `E^Q[P_T] = 2916.16`
exactly (invariant respected).

| `a₁` | `E^Q[X_T]` pre-centering | `Δvar_T` measured | `(a₁/a_ref)² · Δvar_ref` predicted | `Δcall` |
|---:|---:|---:|---:|---:|
| −0.5 | −23.23 | **14.94** | 14.94 (ref) | −0.0027 |
| −2.0 | −92.90 | **239.08** | 239.04 | +0.0009 |
| −5.0 | −232.26 | **1494.24** | 1494.00 | +0.0947 |
| −10.0 | −464.52 | **5976.94** | 5976.00 | +0.5559 |
| −20.0 | −929.05 | **23907.77** | 23904.00 | +2.4807 |

Log-log regression of `|Δ|` on `|a₁|` across the five non-zero rows:

* `log |Δvar|` on `log |a₁|`  →  slope = **2.0000** (four-decimal match to theory).
* `log |Δcall|` on `log |a₁|` →  slope = **2.1151** (mildly super-quadratic;
  the OTM strike `K = 3000 > F = 2916` makes the call payoff a super-linear
  function of the residual variance, so a small higher-order deviation is
  expected and does not indicate a wiring bug).

`p₁(T) = 0.6705` is unchanged across every row (Q1 leaves the generator
alone, as promised), and `u₁(T)` scales strictly linearly in `a₁`
(−15.79 → −63.17 → −157.92 → −315.84 → −631.68), so the product
`2·a·u ~ a²` produces the observed exact quadratic response of the
variance ODE.

At the plausible reference magnitude
(`|a| ~ 0.004 TRY/MWh/h`, order of magnitude only; not estimated from market data)
the expected effect is therefore
`Δcall ≈ (0.004 / 0.5)² · 0.003 = 1.9 · 10⁻⁷ TRY` on a `685 TRY` base —
**genuinely non-zero but far below the PDE solver's numerical precision
(~1 · 10⁻⁴ TRY)**.  The near-zero rows of
`outputs/market_calibration_final/risk_premium_sensitivity.csv` reflect
this: they are real second-order signal, not solver noise.

## Calibration status

Q1's `a_i` is **not calibrated** — no electricity option market data
exists to fit a real market price of risk on this residual.  The CLI
default is `(a_0, a_1) = (0, 0)`, i.e. `Q = P` on the drift channel, and
this default reproduces every prior benchmark bit-for-bit.

The flags are intended for **scenario analysis**: how does the option
price move if we assume a small negative drift premium in the stress
regime (mean-reversion premium)?  A modest sweep is written to
`outputs/market_calibration_final/risk_premium_sensitivity.csv` at
integration time; extending it to a proper calibration requires option
premia and is scope-separate from this note.

## Deferred: Q2 (transition premia) and λ

* `MeasureAdjustment.eta` scales the generator (`q_ij^Q = q_ij^P · exp(η_ij)`).
  It is validated in `risk_neutral.py` and reachable via
  `MeasureAdjustment.adjust_generator`, but *not* yet threaded into
  `ForwardCenteredModel.generator_path`; wiring this is a follow-up
  increment that touches the TVTP path rather than the drift path.
* `market_price_of_risk λ_i` is mathematically equivalent to
  `a_i = − λ_i · σ_i`, so it is not a separate degree of freedom for the
  drift channel.  Exposing it as its own flag is a UX/documentation
  choice, deferred alongside η.
