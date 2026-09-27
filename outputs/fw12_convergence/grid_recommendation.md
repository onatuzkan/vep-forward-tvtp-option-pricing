# FW12 §7 -- Grid-setting recommendation

## Question

Should the production `ResidualGridSettings(n_space_nodes=1201, ...)`
default be changed?

## Evidence (from §1 -- §3)

* Spatial convergence sweep at production climatology z, T=72 h, K=3000
  (see `spatial_convergence.csv`):

  | n_space_nodes | V | Δ vs V(1201) | rel Δ |
  |---:|---:|---:|---:|
  | 301 | 167.0716 | +0.3238 | +0.194 % |
  | 601 | 166.8415 | +0.0939 | +0.056 % |
  | **1201 (production)** | **166.7477** | 0 | 0 |
  | 2401 | 166.7070 | -0.0407 | -0.024 % |
  | 4801 | 166.6940 | -0.0537 | -0.032 % |

  Richardson extrapolation from (2401, 4801) with the observed order
  p ≈ 1.66 gives V_star ≈ 166.686; the production 1201-node value is
  ~0.062 TRY/MWh above the extrapolated limit, i.e. **~0.037 %
  relative error**.  Same pattern at T=24 h and T=48 h.

* Time-step convergence at 2401 nodes (see `time_convergence.csv`):
  values in the [166.687, 166.696] band across 4x refinement of the
  time grid at T=72 h; time-discretisation error is BELOW the
  spatial-discretisation error at the production time-step default.

* Boundary sensitivity (see `boundary_sensitivity.csv`): |ΔV / V|
  <= 0.024 % across n_std in {4, 5, 6, 7.5, 9}.  Well under the
  0.1 % threshold; the far-field spec is fine.

## Interpretation

At the shipped default (1201 nodes, 2 steps/hour, n_std = 6):

* the ATM K=3000 T=72 h call is 166.75, and the truly converged
  value is ~166.69;
* **the discretisation error is on the order of 0.04 % of the value
  (a few hundredths of a TRY/MWh)**, which is well inside the model
  uncertainty band the manuscript already reports for the residual
  measure (FW2 sensitivity envelope ~+/-27 % under Q2) and the
  forward-curve fit precision (calibration residual ~1e-12).

## Recommendation

**Keep the production default `n_space_nodes = 1201`.**  The
discretisation error at this resolution is two orders of magnitude
smaller than any reported physical or measure-choice uncertainty in
the paper.  Doubling to 2401 buys ~0.024 % accuracy at 2x wall-clock
cost per PDE call, which is not warranted for the tables the paper
publishes.

If the manuscript's Appendix C prefers to cite the RICHARDSON
extrapolated value ("converged to ~4 significant digits"), do so
explicitly using the numbers in this report; the tables in Section 7
of the paper stay at the 1201 value, with the 0.04 % Appendix-C
caveat inline.

## Downstream outputs -- what would need re-generation if the grid changed

Not needed under this recommendation, but listed for completeness in
case a future decision goes the other way:

* `outputs/market_calibration_final/strike_maturity_grid.csv` and
  `.md` (66 rows) -- ~4-6 min at 2401 nodes.
* `outputs/market_calibration_final/discount_rate_sensitivity.csv`
  (sweep over r_annual).
* `outputs/market_calibration_final/near_term_anchor_sensitivity.csv`.
* `outputs/market_calibration_final/model_comparison_pooled_vs_M9.csv`
  (already flagged in FW2 §0.3 as pre-v2-kappa; would need TWO
  regenerations: v2 kappa AND 2401 nodes).
* `outputs/market_calibration_final/model_robustness_M8_vs_M9.csv`.
* `outputs/market_calibration_final/risk_premium_sensitivity.csv`.
* `outputs/scenario_sweep/rd_scenario_sweep.csv` (already documented
  as scenario_sweep specialist output).
* `outputs/tvtp2_experimental/comparison/*` if the two-covariate
  runs share the same 1201-node default.
* All figures under `paper/figures/` and PDF re-compile.

**None of the above are regenerated in this FW12 iteration.**  The
recommendation is to KEEP 1201; if the recommendation is overturned,
the list above is the checklist for downstream work.

## FW2 (a, eta) sensitivity envelope at the converged grid

See `fw2_at_converged.csv` for the full re-priced table.  The percent
effects under Q1 and Q2 at 2401 nodes with climatology z are what the
manuscript should cite; FW2's original percent numbers were computed
against a 179.65 baseline that reflected the scenario/plumbing bug
diagnosed in §0.2, not a converged-grid quantity.
