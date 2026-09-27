# FW2 -- Risk-neutral premium: what is identified, what is not, what to do

Faz 5 Category A item.  The production setting is (a_0, a_1) = 0 and
eta = 0 (physical measure for the residual dynamics; only the level is
risk-neutral through forward centering).  This document derives WHY
that setting is defensible for the first moment, WHERE the model is not
identified without option premia, and HOW an empirical premium range
enters the moment / pricing machinery.

## Notation and building blocks

* Two-state chain J_t in {0=normal, 1=stress}, TVTP intensities
  q01(t), q10(t) per hour under P.
* Residual OU:  dX_t = [kappa (m_i - X_t) + a_i] dt + sigma^X_i(t) dW_t.
* Deterministic centering: P_t = F(t) + X_t - mu_X(t), with
  mu_X(t) := E^Q[X_t] pulled from the 6-D linear ODE for (p_i, u_i, w_i)
  (see `docs/risk_neutral_methodology.md`, and equations (12)-(15) in
  `pde_option_model/forward_centered.py`).
* Q1 change of measure: physical-drift shift a_i per hour (equivalent to
  a market-price-of-risk lambda_i = -a_i / sigma_i via Girsanov;
  documented in `docs/risk_neutral_methodology.md`).
* Q2 additionally shifts transition intensities:  q_ij^Q = q_ij^P *
  exp(eta_ij).  The multiplicative form guarantees
  q_ij^Q >= 0 and row sums remain zero for every real eta_ij, so
  generator validity is preserved by construction (no clipping needed).

## Proposition 1 -- forward centering absorbs the first moment

For every admissible (a_0, a_1, eta_01, eta_10) the pipeline delivers
E^Q[P_t] = F(t) for all t in [t0, T] by construction, because
mu_X(t) is COMPUTED from the same (a, eta) and the same q^Q that drive
the pricing PDE, then subtracted from the payoff map at every time
slice.  The identity holds not up to a limit but exactly (the moment
ODE and the PDE share q01(t), q10(t), sigma_i(t), a_i and any Q2 shift
by construction; the six-dimensional linear system for (p, u, w) is
solved once and its p_i(t), u_i(t) trajectories are passed to both the
centering subtraction and the PDE's boundary field).

**Consequence for the paper.**  This is the correct place to write:
"the level risk premium is absorbed by the forward curve; the residual
dynamics fit around the observed forwards, not around a physical
expectation."  Insert in the manuscript at Section 5 (Valuation),
immediately after equation (12) (`E^Q[P_t] = F(t)`), as a one-sentence
proposition + one-sentence corollary.  The identity is verified
numerically in every existing centering test
(`test_forward_centered.py::test_expected_spot_equals_forward` and its
tvtp2 counterpart) and continues to hold for every case in FW2's
extended sensitivity sweep -- see §7 below.

## Proposition 2 -- the two channels differ in ORDER on the residual variance

Let Var_Q(P_T) be the terminal-hour residual variance (the quantity that
drives European option value once F(T) and the strike are fixed).

**Q1 drift channel.**  With centering enforced, the payoff variance is a
QUADRATIC form in (a_0, a_1) around zero.  The formal statement (proved
by expanding the coupled ODE for w_i - u_i^2 in a) is

    Var_Q(P_T) = Var_Q(P_T | a=0)
                + (a_i)^T Sigma_a(T; kappa, q, pi) (a_i) + O(||a||^3),

with Sigma_a positive semi-definite in the neighbourhood of a=0 and
containing the same p_i and integrated kappa-decay kernels the moment
ODE emits.  The linear term vanishes because centering already
subtracts E[X_t] -- so the first-order correction cannot survive.  This
is the analytic reason for the log-log slope 2.0000 documented in
`outputs/market_calibration_final/risk_premium_sensitivity.csv` (the
variance uplift channel of `docs/risk_neutral_methodology.md`).
FW12b re-measured this slope under the PRODUCTION climatology z(t-1)
path (the original measurement was under the silent z = 0 fallback);
the observed log-log slope is **2.0000** to four significant digits
in both scenarios, i.e. the quadratic suppression is a property of
the model and does not depend on the z path.  A priori Q1 is HARDLY
IDENTIFIED from options at plausible premium magnitudes: the option-
value gradient at a = 0 is O(||a||^2).

**Q2 transition channel.**  eta_ij enters the occupancy p_i(t)
LINEARLY at first order, and the mixture variance rate depends on p_i
LINEARLY (mixture-variance = pi_normal * sigma_n^2 + pi_stress *
sigma_s^2).  Hence Var_Q(P_T) is LINEAR in eta_ij around zero:

    Var_Q(P_T) = Var_Q(P_T | eta=0)
                + eta_ij * dVar/deta_ij |_{eta=0} + O(||eta||^2),

with the first-order coefficient bounded below by
(sigma_s^2 - sigma_n^2) * dpi_stress/deta * T, which for the shipped
sigmas (0.0035, 0.0924) is a large positive number.  A priori Q2 is
FIRST-ORDER identified from options and materially more powerful
than Q1.  The FW2 sensitivity sweep in §4 verifies this numerically
(FW12b-regenerated version at 1201 nodes with the climatology z
path -- the archived FW2 output at 601 nodes with silent z = 0 gave
similar-order but different numeric magnitudes; see
`outputs/fw2_risk_premium/archive/README.md` for the audit trail).

**Consequence for the paper.**  The two channels are not symmetric.  A
paper making a "we set a=0 for a reason" claim should state Proposition
2 explicitly.  The manuscript's Section 5.4 (or a new Section 5.5) is
the natural home for this: two paragraphs, one per channel, with a
sentence pointing at the numeric evidence in
`outputs/fw2_risk_premium/sensitivity_grid.csv`.

## What is identified from forwards alone

* F(t) itself (VEP monthly quotes + smooth constrained interpolation).
* The residual VOLATILITY structure (sigma_y_normal, sigma_y_stress,
  kappa) up to the invariance E[X_t]=0 by centering -- the yaml carries
  the M9 fits.
* The regime chain P-measure intensities (alpha, gamma), from the M9
  transition-coefficient fit.

## What is NOT identified from forwards alone

* Q1 drift shifts (a_0, a_1) -- the two-dimensional space is
  unconstrained by E^Q[P_t] = F(t) (Prop. 1) and only weakly identified
  from option prices (Prop. 2).
* Q2 transition premia (eta_01, eta_10) -- linear in variance but,
  without any option quote, cannot be pinned.  Q2 IS what the paper
  should call out as "requires option data to identify".
* The DECOMPOSITION between Q1 and Q2: even given a single option price,
  the two channels are jointly not identified (drift-transition
  trade-off).  Two option prices at DIFFERENT maturities partially
  separate them because Q1 is O(||a||^2 T) while Q2 is O(||eta|| T).

## Empirical anchors used in this iteration

Because no Turkish electricity option is quoted publicly, an option-
calibration route is closed.  FW2 substitutes two empirical anchors:

1. **Ex-post forward-vs-realised premium.**  The 7 tracked VEP snapshots
   (2022-12-31 through 2025-12-31, semi-annual) combined with the
   realised monthly-average PTF give a small out-of-sample panel of
   (F_quoted - realised_monthly) values.  Descriptive statistics per
   horizon and per delivery month, WITH standard errors that account
   for overlapping horizons via Newey-West / block bootstrap.  This is
   NOT an estimator of the risk premium; it is a distribution of ex-
   post forward errors, which by finance-theoretic conventions equals
   the risk premium PLUS a mean-zero forecast-error term IF the sample
   is long enough.  The 7-date sample is far too short for that
   equality to hold; the statistics are reported as an upper-bound
   descriptive input, not as a point estimate.

2. **Literature bounds** drawn from `paper/refs.bib`.  Where an author
   reports a per-MWh premium and its horizon, we translate it (via the
   delta-method `sigma^X_i = sigma^y_i * sqrt(F^2 + scale_P^2)`) into
   an hourly a_i.  Where the reference reports only qualitative signs
   the bound is not usable and is flagged as such.

Both anchors bound the plausible a_i range.  Neither pins a value.  The
production yaml therefore stays at a=0 and eta=0; the empirical range
is presented as the manuscript's uncertainty band around the reported
option prices.

## What the FW2 sweep answers

At the empirical a_i and eta_ij bounds from §2, price 24/48/72 h ATM
call and put on the 2000/2500/3000/3500/4000 strike ladder under

  * (a, eta) = (0, 0)                  -- production
  * a = a_bound, eta = 0                -- pure Q1 upper bound
  * a = 0, eta = eta_bound              -- pure Q2 upper bound
  * a = a_bound, eta = eta_bound        -- joint upper bound

Compare the price shift in TRY/MWh and % vs the production run.  The
resulting numbers ARE the "uncertainty band around the option prices"
the manuscript should report.

## Numerics -- generator embedding invariant

Every q_ij used by the moment ODE and the PDE originates from a
discrete transition probability p_ij(dt) mapped through the exact 2x2
matrix logarithm: with s = p01 + p10 in (0, 1) we have
q_ij = -ln(1 - s) * p_ij / s.  This is embedding-safe as long as
s < 1.  The multiplicative Q2 form eta_ij (applied to q_ij, not to
p_ij) commutes with this mapping in the correct direction: we adjust
the CONTINUOUS-TIME rate, then only need to check that the resulting
generator has non-negative off-diagonal entries (trivially satisfied by
q_ij >= 0 * exp(eta_ij) = q_ij^Q >= 0) and zero row sums (holds by
setting q_ii^Q := -q_ij^Q).  There is no p-space embeddability question
to worry about UNLESS an eta is so large that a downstream discrete
recovery of p^Q(dt) via `generator_to_probs` blows past s^Q = 1; that
is a numerical guardrail only, not a modelling constraint, and is
enforced in `probs_to_generator`'s existing `on_nonembeddable=` hook.
