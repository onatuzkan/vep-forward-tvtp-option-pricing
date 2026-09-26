# FW2 §5 -- Production risk-premium setting: recommendation

## Three candidate settings

| # | Setting | Rationale | Downside |
|---|---|---|---|
| (i) | `(a_0, a_1) = (0, 0)`, `eta = 0` (STATUS QUO); the empirical band in §2-§4 is reported as the manuscript's uncertainty envelope. | First moment identity `E^Q[P_t] = F(t)` holds by centering under this setting.  The residual physical / Q1 shifts and Q2 shifts are not identified from forwards alone.  No option quote to pin a central value.  The uncertainty band is the honest quantitative statement. | The paper still says "prices are level-risk-neutral but residual-physical for the higher moments".  A referee may push for a central risk-adjusted value. |
| (ii) | A literature-central prim (e.g. `a_stress = 25 TRY/MWh/h`, `eta = (0.25, -0.25)`) becomes the production yaml default; the status quo becomes a "physical measure" sensitivity. | Would give a single reported number that is neither zero nor at the empirical extreme. | The literature anchor is INCOMPLETE (§2.2): magnitudes are not recoverable from the current repo without reading the papers.  Choosing a central value now would be a fabricated calibration. |
| (iii) | Only one channel calibrated (e.g. `eta_01 = 0.25` fixed, `a = 0`); the other stays zero. | Mixes calibration with defaulting.  A "half calibrated" yaml can hide an unidentified degree of freedom. | Same identifiability problem as (ii); the choice of channel is arbitrary at zero option data. |

## Recommendation

**Adopt (i).**

Reasoning:

1. **First-moment risk neutrality is already achieved.**  Proposition 1
   of `docs/fw2_risk_premium_identification.md` shows that
   `E^Q[P_t] = F(t)` is an algebraic identity of the pipeline for every
   admissible `(a, eta)`.  A referee's "why aren't you risk-neutral"
   objection is answered at the level of the FIRST MOMENT without any
   parameter change.

2. **Higher-moment identification requires option data that does not
   exist.**  Both channels sweep changes in `Var_Q(P_T)` (§4), but the
   sweep envelope is BOUNDED, not pointed.  Picking a central value
   without option quotes would fabricate a calibration; that would be
   less defensible than reporting the envelope explicitly.

3. **The empirical band is small enough to be reported honestly.**  The
   FW2 sensitivity sweep gives the maximum absolute call-value effect
   per channel; both channels are single-digit-percent of the ATM
   72 h call at empirical magnitudes.  A referee ranking the pricing
   errors will see this as one column of an uncertainty budget, not
   as a fatal indeterminacy.

## Consequence for the paper

Add a short (~half-page) section right after Section 5 (or as Section
5.5), titled "First-moment risk neutrality and the residual measure":

* State Proposition 1 (Prop. 1 of the FW2 identification doc): the
  centering identity absorbs the level premium.  Cite `E^Q[P_t] =
  F(t)` (already equation (12) in the manuscript) and the two-
  regime moment ODE (Appendix A).
* State Proposition 2: the two premium channels differ by order on
  the residual variance -- Q1 quadratic, Q2 linear.  Number the two
  channels as (5.5.1) drift and (5.5.2) transition.
* Report the FW2 sensitivity envelope as a table (max absolute and
  max percent effect on the ATM 72 h call, per channel).  Source:
  `outputs/fw2_risk_premium/sensitivity_summary.csv`.
* Close with "we set (a, eta) = 0 as the production default; the
  envelope above is the pricing uncertainty attributable to the
  residual-measure choice."

## Provenance tags for anything new

FW2 does not change any yaml.  Any parameter that ships in a future
yaml as a nonzero (a, eta) would carry the following provenance tags:

* `a_i`: `Calibrated (option quotes)` if option data become available;
  otherwise `Assumed (bounded by ex-post ex_post_premium.csv panel)`.
* `eta_ij`: `Assumed (bounded by generator embeddability)` -- there is
  no observable in the repo that could ever change this to
  `Calibrated` without option data.

No candidate yaml is written in this FW2 iteration; the frozen yaml
stays as `inputs/historical/m2_frozen_parameters.yaml` (v2 kappa
0.078394/h) unchanged.  A candidate yaml would live at
`inputs/historical/candidate/m2_frozen_parameters.WITH_Q2.yaml` with a
provenance block explaining every non-zero entry; this file is NOT
produced here.
