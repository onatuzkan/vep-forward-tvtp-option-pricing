# Archive -- superseded FW9 outputs

The four files in this folder were produced by the first pass of FW9
(2026-09-27) and were superseded by FW9b (2026-09-28) for two
independent reasons:

## `price_impact_grid.phi_1_unconverged.csv` + `price_impact_pivot.phi_1_unconverged.csv`

Priced with the FW9 free-phi MLE point estimate, which had
`phi = 1.000000` (unit-root boundary; MLE did not converge).  With
`phi = 1`, `kappa = -ln(phi) = 0`, so the OU degenerates to a pure
random walk and the residual dispersion diverges.  The reported price
deltas (+1 026 % ATM, +23 945 % deep-OTM) were a specification
consequence, NOT a parameter finding.  Replaced by
`price_impact_v2_grid.csv` and `price_impact_v2_decomposition.csv`
which use the FW9b-consistent set: profile sigmas + alphas + gammas at
`phi = 0.99999` and `kappa` from the deseasonalised single-regime AR(1)
fit (§2), matching what F2.12 did in production.

## `parameter_comparison.hybrid_inheritance.csv/.md`

Compared FW9's MS-AR object estimates against the yaml's hybrid values
(`phi = 0.9246` is from the deseasonalised object; sigmas / gammas
from the MS-AR object).  This mixed comparison generated the
"9 of 9 outside CI" wording, which was not like-for-like.  Replaced by
`parameter_comparison_v2.csv/.md` (T8) with an explicit
`inherited_from` column and object-matched comparison per FW9b §3.

None of the files are deleted so the audit trail is intact.
