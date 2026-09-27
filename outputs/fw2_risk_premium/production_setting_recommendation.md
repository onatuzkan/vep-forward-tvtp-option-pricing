# FW2 §5 -- Production risk-premium setting: recommendation (FW12b-revised)

## Question

Should the production `(a_0, a_1) = (0, 0)`, `eta = 0` default be changed?

## What the sensitivity envelope actually is

FW12b re-priced FW2's manuscript rows at the production 1201-node
grid with the climatology z path (see
`outputs/fw2_risk_premium/sensitivity_grid.csv` and
`sensitivity_summary.csv`; the ATM K = 3000 / T = 72 h base rises
from the archived silent-z-zero value 179.65 to the correct
production value ~166.75).  The Q2 envelope, in particular, is
LARGER than the archived numbers suggested.

**Important framing.**  The eta_ij grid used in the sweep
(|eta_01|, |eta_10| <= 0.75) is NOT an ampirical band derived from
observed premium magnitudes.  There is no observable in this repo
that pins the sign or size of the Q2 transition premium.  The upper
edge |eta| = 0.75 is a **mathematical envelope** dictated by
generator embeddability: the multiplicative form
`q_ij^Q = q_ij^P * exp(eta_ij)` keeps the intensity within a factor
of ~2.1 of the physical value, which is well inside the
discrete-embedding safe zone at the yaml's kappa/q ratio.  The sweep
thus reports the LARGEST price effect a Q2 shift COULD produce
before the generator stops being invertible into a valid discrete
transition matrix, not the largest effect an estimated Q2 premium
WOULD produce.

That distinction is why the Q2 envelope reaching +/- 46 % of the
production ATM call is not an argument for "the risk premium could
be huge" -- it is an argument for "there is a mathematically
consistent range of Q2 measures within which the model has to
choose, and no observable in this repo pins the choice."

## Three candidate settings

| # | Setting | Rationale | Downside |
|---|---|---|---|
| (i) | `(a_0, a_1) = (0, 0)`, `eta = 0` (STATUS QUO); the FW2 §4 envelope is reported as the pricing uncertainty attributable to the residual-measure choice. | First moment identity `E^Q[P_t] = F(t)` holds by centering.  Both channels sweep the terminal residual variance; without option quotes to pin the choice, reporting the envelope is the honest quantitative statement.  The envelope's outer edge is a mathematical bound (generator embeddability), NOT an empirical band, and this must be stated explicitly. | The paper still says "prices are level-risk-neutral but residual-physical for the higher moments."  A referee may push for a central risk-adjusted value. |
| (ii) | A literature-central prim becomes the production yaml default. | Would give a single reported number. | The literature anchor is INCOMPLETE (FW2 §2.2): magnitudes are not recoverable from the current repo without reading the papers.  A "central" value chosen now would be a fabricated calibration. |
| (iii) | Only one channel calibrated. | Mixes calibration with defaulting. | Same identifiability problem as (ii); the choice of channel is arbitrary at zero option data. |

## Recommendation (unchanged from the archived version, sharpened framing)

**Adopt (i).**

Reasoning after FW12b:

1. **First-moment risk neutrality is already achieved.**  Proposition 1
   of `docs/fw2_risk_premium_identification.md` shows `E^Q[P_t] = F(t)`
   is an algebraic identity of the pipeline for every admissible
   `(a, eta)`.

2. **The Q2 envelope reaching +/- 46 % is not a claim about the risk
   premium; it is a claim about how much the pricing model would move
   IF a Q2 measure were pushed to the edge of generator
   embeddability.**  Without option data to pin `eta`, we do not
   pretend to know where the true measure sits inside that envelope.
   The paper's Section 5.5 should say this explicitly: the reported
   band is a mathematical outer envelope, not an empirical
   confidence interval.

3. **The empirical Q1 band is small.**  The ex-post forward-premium
   panel (FW2 §2.1) translates to `|a_i|` <= 50 TRY/MWh/h via the
   `|a| = kappa * |L|` map; the Q1 sweep at that upper edge moves
   the ATM 72 h call by 5.6 % of the production value.  That IS an
   empirical band and is small enough to report as-is.

## Consequence for the paper

Section 5.5 ("First-moment risk neutrality and the residual measure")
should say, in order:

1. State Proposition 1 (level risk premium is absorbed by the forward
   curve; centering identity).
2. State Proposition 2 (Q1 quadratic, Q2 linear in terminal variance
   around the physical measure; log-log slope 2.0000 to 4 sig figs
   under BOTH z = 0 and climatology z, i.e. it is a property of the
   model not of the scenario).
3. **Q1 uncertainty band = empirical**: 0 to +5.6 % on the ATM
   72 h call as `a_stress` sweeps 0 to 50 TRY/MWh/h; the upper edge
   comes from the FW2 §2.1 ex-post premium panel.
4. **Q2 uncertainty band = mathematical (embeddability) envelope**:
   +/- 46 % on the ATM 72 h call as (eta_01, eta_10) sweep to
   (-/+0.75, +/-0.75); the upper edge is generator embeddability,
   not an empirical anchor.  Absent option data, the paper reports
   the envelope but does NOT claim the true Q2 measure sits anywhere
   in particular inside it.
5. Production yaml stays at `(a, eta) = (0, 0)`.

## Provenance tags

FW12b does not change any yaml.  Any parameter that ships in a
future yaml as a non-zero `(a, eta)` would carry:

* `a_i`: `Calibrated (option quotes)` if option data becomes
  available; otherwise `Assumed (bounded by ex-post
  ex_post_premium.csv panel)`.
* `eta_ij`: `Assumed (bounded by generator embeddability)` --
  intrinsically unobservable without option data.

No candidate yaml is written in this iteration; the frozen yaml
stays untouched.

## Numeric snapshot (ATM K = 3000, T = 72 h; production 1201 nodes)

| kanal | (a_stress, eta_01, eta_10) | V | % vs 166.75 |
|---|---|---:|---:|
| baseline | (0, 0, 0) | 166.75 | 0 |
| Q1 only  | (50, 0, 0) | ~176.15 | +5.6 % |
| Q2 only  | (0, +0.5, -0.5) | ~201.9 | +21.1 % |
| Q2 only  | (0, -0.5, +0.5) | ~116.1 | -30.3 % |
| Q2 only  | (0, +0.75, -0.75) | ~212.7 | +27.6 % |
| Q2 only  | (0, -0.75, +0.75) | ~89.9 | -46.1 % |
| joint    | (50, +0.75, -0.75) | ~215.2 | +29.1 % |
| joint    | (50, -0.75, +0.75) | ~102.2 | -38.7 % |

Full CSV in `outputs/fw2_risk_premium/sensitivity_grid.csv`
(FW12b-regenerated) and cross-verified against
`outputs/fw12_convergence/fw2_at_converged.csv` (FW12 at 2401 nodes).
The archived silent-z-zero version is in
`outputs/fw2_risk_premium/archive/` for the audit trail; its
numbers were computed against a baseline of 179.65 and are not
valid for the manuscript.
