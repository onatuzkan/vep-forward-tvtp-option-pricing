# FW10 archive -- monthly baseload F(target hour) variant (SUPERSEDED)

The Part A and Part B tables in this directory were produced by the
FIRST FW10 pass, which used the VEP monthly baseload quote of the
target-hour delivery month as F(target hour).  That construction is
NOT the production forward curve: the shipped model builds a smooth
constrained hourly curve with a near-term spot anchor
(`spot_to_next_linear`) that dominates F(T) at 24-72 h horizons.

Consequences documented in the FW10b (correction) pass:

* The monthly-baseload F is a coarse block level, so the FW10
  predictive distribution tested a residual-around-a-coarse-mean
  rather than a residual-around-the-production-curve.  This is why
  the PIT means were 0.33-0.44 and the 50 pct coverage was 17-28
  pct.
* The delta P(P_T > K) in Part B assumed the hedge instrument was a
  contract with dF/dF_M = 1 (fully passing monthly moves through to
  the delivery-hour forward).  Under the production curve dF/dF_M
  is well below 1 at 24-72 h; the correct hedge ratio needs the
  dF(target)/dF_M sensitivity of the calibrated curve.

The corrected Part A and Part B outputs are in
`outputs/fw10_validation/` (one level up).  These archived files are
kept for auditability and MUST NOT be cited in the paper.
