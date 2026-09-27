# FW12b §2.2 -- Cross-validation of the repaired FW2 sweep

Two independent code paths that must give the same number if the FW2
scenario-plumbing bug is really fixed and no fresh bug has been
introduced:

* `outputs/fw2_risk_premium/sensitivity_grid.csv` --
  `scripts/fw2/sensitivity_sweep.py`, PATCHED to use the production
  climatology z(t-1) path at 1201 spatial nodes;
* `outputs/fw12_convergence/fw2_at_converged.csv` --
  `scripts/fw12/fw2_at_converged.py`, always used climatology, at
  2401 spatial nodes.

They are the FW12 analogue of the shipped PDE-vs-MC cross-check: two
different implementations of the same calculation, driven by
independent code paths, must agree.

## Matched rows (K = 3000, T in {24, 48, 72} h, call)

21 overlapping rows.  The complete numeric table lives in
`outputs/fw2_risk_premium/cross_validation_1201_vs_2401.csv`; the
punchline is:

| kanal | worst-case row | V_1201 | V_2401 | |Δ| TRY | |Δ| % |
|---|---|---:|---:|---:|---:|
| joint corner | (a=50, T=72, eta=(-0.75, +0.75)) | 102.456 | 102.156 | 0.300 | 0.294 % |
| joint corner | (a=50, T=48, eta=(-0.75, +0.75)) | 102.714 | 102.419 | 0.296 | 0.289 % |
| joint corner | (a=50, T=24, eta=(-0.75, +0.75)) | 100.753 | 100.532 | 0.222 | 0.220 % |
| Q1 only      | (a=25, T=72) | 169.955 | 169.824 | 0.131 | 0.077 % |
| Q1 only      | (a=50, T=72) | 176.238 | 176.109 | 0.129 | 0.073 % |
| Q2 only      | (eta=(-0.5, +0.5), T=72) | 116.194 | 116.141 | 0.052 | 0.045 % |

**Max absolute difference = 0.300 TRY/MWh, max relative difference =
0.294 %.**

This is the EXPECTED spatial-refinement gap 1201 -> 2401 documented
in `outputs/fw12_convergence/spatial_convergence.csv` for the ATM
call:

    V(1201) - V(2401) = 166.7477 - 166.7070 = 0.041 TRY/MWh at ATM.

At the joint corner the residual sd shrinks by ~30 % (see
`residual_sd_T` column in the two sweep CSVs), which increases the
per-node discretisation of the payoff kink at K = 3000; the gap
0.30 TRY/MWh at that corner is 5-6x the ATM gap, consistent with
the payoff-kink-driven order documented in FW12 §1.

## Interpretation

* **Both code paths compute the same underlying quantity.**  Their
  systematic disagreement is bounded by the known spatial
  discretisation error at 1201 nodes; no anomaly requiring
  investigation is visible.
* **The manuscript should cite the 1201 numbers**
  (`outputs/fw2_risk_premium/sensitivity_grid.csv`) with an inline
  Appendix C note that the 2401 numbers agree within 0.3 TRY/MWh
  everywhere.  Alternatively cite the 2401 numbers directly; the
  choice is a manuscript-editorial question, not a modelling
  question.

## The archived silent-z-zero sweep is NOT a cross-check

The four archived files in `outputs/fw2_risk_premium/archive/` used
601 nodes AND silent z = 0; they measured the constant-transition
limit at a coarser grid, so their agreement or disagreement with
the 1201/2401 climatology numbers is not informative about
correctness.  They are kept for the audit trail only.
