# Forward-Curve Calibration Review

**Scope.** Compare the four `NearTermAnchor` modes for the January-2026 window
against the accepted (`spot_to_next_linear`) calibration in
`outputs/market_calibration_final/`.
Nothing in that directory is touched. TVTP and residual-volatility parameters
are held fixed at the frozen M2 values throughout — this review is purely about
how the level of `F(t)` in the unquoted January window is constructed.

Valuation: 2025-12-31T20:00 UTC (Turkey local 2025-12-31 23:00).
Spot used: **2917.78 TRY/MWh**.  Six monthly VEP quotes for Feb-Jul 2026.

---

## 1. Anchor modes defined in `pde_option_model/forward_curve.py`

Source: `forward_curve.py:64-65, 82-110, 222-263`.

Let `t_0` be the valuation hour, `t_M` the first hour of the first quoted month
(2026-01-31T21:00 UTC), and `P_spot` the observed spot.  All modes fill the
"pre" window `t_0 <= t < t_M` (744 hourly nodes for January) with a rule; the
solver then applies the smoothness+equality KKT system on top, so the actual
delivered curve is a smoothed version of the piecewise baseline.

| Mode | Baseline rule for `t in [t_0, t_M)` | `F(t_0) = P_spot`? |
|---|---|---|
| **`spot_flat`** | `F(t) = P_spot` (flat) | Yes |
| **`spot_to_next_linear`** *(production)* | Linear ramp: `F(t) = P_spot + (t - t_0)/(t_M - t_0) * (F_{Feb} - P_spot)` where `F_{Feb}` is the first quoted monthly level (2900.99) | Yes |
| **`flat_next_month`** | `F(t) = F_{Feb}` (ignores spot) | **No** |
| **`explicit_level`** | `F(t) = L` for a user-supplied constant `L` | Only if `L == P_spot` |

The KKT step (`_solve_smooth`) then adds a hard equality row `e_0^T F = P_spot`
only when the mode pins spot (`spot_flat` / `spot_to_next_linear`,
`forward_curve.py:378-383`); the other two modes explicitly override spot.

---

## 2. Alternative-mode calibration results

All three full calibrations under `outputs/market_calibration_review/<mode>/`
pass the acceptance checks (max abs monthly error ~3e-12 TRY/MWh, MAPE
~1e-13 %).  The differences live entirely in the near-term window.

### Monthly fit — identical numerically
Feb-Jul quotes are reproduced to solver precision (~1e-12) in every mode, since
they enter as hard KKT equalities.  Anchor choice cannot break the fit.

### January-window shape summary

| Mode | F(t_0) | F(mid-Jan) | F(end-Jan) | Jan mean | F(first Feb hour) | Jan→Feb jump |
|---|---:|---:|---:|---:|---:|---:|
| `spot_flat` | 2917.78 | 2917.78 | 2910.53 | **2917.71** | 2909.99 | -0.53 |
| `spot_to_next_linear` | 2917.78 | 2909.40 | 2901.97 | **2909.40** | 2902.02 | +0.05 |
| `flat_next_month` | 2900.99 | 2900.99 | 2901.88 | **2901.00** | 2901.94 | +0.07 |

Note the smoother makes the `spot_flat` window curve DOWN in the last few days
of January (from 2917.78 to 2910.53) — it is not literally flat once the
smoothness weight is applied, because the KKT solver has to reconcile
`F(t_0)=spot` with the Feb-average constraint.  So `spot_flat` implicitly
gives an unnaturally high January baseload (2917.71) with a mild downward
ramp at month-end.

### K=3000 call values (frozen M2 TVTP, no drift/vol premia)

| Maturity | `spot_flat` | `spot_to_next_linear` | `flat_next_month` | spread |
|---:|---:|---:|---:|---:|
| 24 h | 686.35 | 686.02 | 674.18 | 12.17 (1.8%) |
| 72 h | 1237.84 | 1236.69 | 1222.39 | 15.45 (1.2%) |
| 168 h | 1882.54 | 1879.41 | 1863.35 | 19.19 (1.0%) |

Ordering is monotone in `F(t_0)` (2917.78 > 2917.78 > 2900.99) as expected: the
call value moves with the spot input.  The spread narrows in relative terms with
maturity because the forward reverts to the same Feb-onwards constrained curve.

Full per-mode outputs written to `option_value_comparison.csv`.

---

## 3. `near_term_anchor_sensitivity()` — fixed version

**Bug.** The original function (`market_calibration.py:464-510`) sweeps the
January level using `mode="explicit_level"` — a **flat** anchor — regardless of
the anchor mode actually in production.  That yields three misleading
artefacts:

1. **All near-term horizons collapse to a single number.**  In the accepted
   run `expected_spot_72h = expected_spot_168h = expected_spot_336h` for every
   swept level (see the OLD_ES_* columns of
   `near_term_anchor_sensitivity_comparison.csv`).  This is a property of the
   flat anchor, not of the underlying curve.
2. **`spot_consistent_at_t0` is False for every row except the trivial one.**
   The sensitivity says "the near-term level is L", but `F(t_0)` should be
   `spot` — a nonsense state under the production anchor.
3. **The curvature of the anchor (linear ramp back to F_Feb) is invisible.**
   The user sees no term-structure of near-term ES, whereas under
   `spot_to_next_linear` all four reporting horizons receive different levels.

**Fix.** `outputs/market_calibration_review/near_term_anchor_sensitivity_fixed.csv`
sweeps the ASSUMED SPOT (`spot_price_TRY_MWh` argument to
`build_forward_curve`) at the same ±20% grid, keeping
`mode="spot_to_next_linear"`.  This preserves the production shape (a linear
ramp to `F_Feb`) and shifts only the anchor level, which is exactly the
counterfactual the sensitivity is trying to describe.

### Side-by-side comparison (K=3000 call, 72h)

| level shift | OLD anchor (flat) | NEW anchor (ramp) | ΔES_72h | Δcall |
|---:|---:|---:|---:|---:|
| -20% (2334.22) | ES_72h=2334.22, call=727.83 | ES_72h=2388.99, call=761.61 | +54.77 | +33.77 |
| -10% (2626.00) | ES_72h=2626.00, call=975.09 | ES_72h=2652.58, call=992.87 | +26.58 | +17.78 |
|  0% (2917.78) | ES_72h=2917.78, call=1237.84 | ES_72h=2916.16, call=1236.69 | -1.62 | -1.15 |
| +10% (3209.56) | ES_72h=3209.56, call=1512.01 | ES_72h=3179.74, call=1489.88 | -29.82 | -22.13 |
| +20% (3501.34) | ES_72h=3501.34, call=1794.81 | ES_72h=3443.32, call=1750.20 | -58.02 | -44.60 |

Every non-baseline row moves by 20+ TRY/MWh in `ES_72h` — the OLD table over-
states the impact of an anchor shock at short horizons because it applies the
shift as a **step** instead of a ramp.  Symmetrically, the 720h ES in the OLD
table barely moves (2326→2510), whereas in the NEW table it moves by less
still (2882→2920): both agree that far horizons are pinned by the Feb+ VEP
quotes; the disagreement is entirely near-term.

The fixed table also flags `spot_consistent_at_t0=True` at the base row only,
correctly signalling that only the base case is a valid pricing state.

Note: this is a *NEW file* alongside the accepted run's original — the
production `market_calibration.py` code is unchanged.  Whether to make the
fix permanent (i.e. replace `near_term_anchor_sensitivity`) is left as a
follow-up decision, since the OLD function is called by
`run_pde.py cmd_calibrate_market` and rewriting its signature affects the
JSON schema the audit reports on.

---

## 4. Hourly-jump audit (|ΔF| > 5 TRY/MWh)

Full lists per mode: `hourly_jumps_<mode>.csv`.

**Aggregate counts (identical across the three modes):**

| metric | value |
|---|---|
| total jumps > 5 TRY/MWh | 113 |
| at a month boundary (last hour of month → first hour of next) | 3 |
| intra-month | 110 |
| jumps inside the January (unquoted) window | **0** |
| earliest jump | 2026-02-28T04:00 UTC |
| max distance from a month boundary of any intra-month jump | 27 hours |
| median distance from boundary of intra-month jumps | 10.5 h |

**Interpretation.**
* The three modes share the identical Feb-onwards curve (only the January
  window changes with the anchor rule), so their jump patterns coincide.
* The three **boundary** jumps live at the true monthly steps that are largest:
  Feb→Mar (-12.3), May→Jun (-9.6), Jun→Jul (+47.0).  The Feb-Mar and May-Jun
  quote differences are moderate (2901→2556, 2506→2245); Jun→Jul is a big
  1300 TRY/MWh jump in the underlying monthly quotes (2245→3558) which the
  smoother can only partially soften without violating the hard equality
  constraints.
* The 110 intra-month jumps sit within one day (≤27 h) of a month boundary and
  are the smoother's ramp-in / ramp-out of those boundaries.  This is a
  structural consequence of the second-difference roughness penalty balancing
  against exact delivery-average equalities: near a large adjacent-month step,
  many small (5-8 TRY/MWh) hourly increments accumulate over the last day of
  the "high" month to bring the average down without violating the constraint.
* **Zero anomalous intra-month jumps.**  No jump lives in a region unrelated
  to a month boundary.  The January (unquoted, anchor-driven) window has
  `max |dF| = 0.05 TRY/MWh` — perfectly smooth.

**Potential curve-fit concern.** The Jun→Jul boundary sees a 47 TRY/MWh
single-hour step at the transition.  This is not a bug — it is the smoother
accepting a hard cliff at the boundary because the D2 penalty against the
neighbouring flat regions would otherwise require the average to be moved
substantially in each adjacent month.  If a smoother Jun→Jul transition is
desired the smoothness weight (`market.smoothness_weight`) can be raised, or
a bilinear shape profile (`shape_profile`) applied.

---

## 5. Recommendation

**Keep `spot_to_next_linear` as the accepted anchor mode.**

Ranked, with justification:

| Mode | Verdict | Reasoning |
|---|---|---|
| `spot_to_next_linear` | **RECOMMENDED** | (i) Enforces `F(t_0) = spot` exactly, so `E^Q[P_0] = spot` (the arbitrage-free requirement at t=0). (ii) Terminates at the first quoted monthly level, giving a continuous handoff to the market-constrained region (Jan→Feb boundary jump 0.05 TRY/MWh, versus 0.53 for `spot_flat`). (iii) Provides an economically-motivated term structure of near-term expected spot instead of collapsing it to one number. |
| `spot_flat` | Acceptable fallback | Preserves spot consistency at t_0. Downside: an implicit January baseload of 2917.71 TRY/MWh is nearly 17 TRY/MWh (0.6%) above the flat-Feb-adjacent choice; there is no market signal supporting that Jan is a "high" month relative to Feb. Also creates a mild residual ramp at end-of-January driven by the smoother rather than by any assumption the analyst can defend. |
| `flat_next_month` | **NOT recommended** | Breaks `F(t_0) = spot` (2900.99 vs 2917.78, a 17 TRY mismatch flagged by `forward_centered.py` as a warning). Model no longer prices `P_0` to spot; every 24-168h call value shifts by ~15-19 TRY. Suitable only when the analyst genuinely believes the day-ahead spot is stale relative to the forward strip. |
| `explicit_level` | Diagnostic only | Useful for sensitivity analysis over a stated January baseload assumption, but requires a defensible external input. Should not be a production default. |

### Statistical checks (all modes pass)
* Max abs monthly delivery-average error ~ 3e-12 TRY/MWh (limit 0.1).
* MAPE ~ 1e-13 % (limit 1.0%).
* No jump anomalies outside the smoother's boundary-adjacent adjustments.

### Economic argument for `spot_to_next_linear` over the alternatives
1. **No unit-root arbitrage window at t=0.**  Only the two spot-pinning modes
   deliver `E^Q[P_0] = spot`; using `flat_next_month` violates this equality
   and creates a 17 TRY/MWh instantaneous mispricing at valuation, which the
   downstream residual PDE cannot repair.
2. **No implicit "January premium" assumption.**  `spot_flat` implicitly
   asserts a January baseload of 2917.71 TRY/MWh — 0.6% above the earliest
   quoted month — with no market data supporting the claim.  The ramp gives a
   defensible interpolation (2909.40 mean) that treats the missing January
   quote as a linear blend between the two nearest observed states (spot at
   t_0, and F_Feb at t_M).
3. **Term-structure fidelity.**  Under `spot_to_next_linear` the ES map
   (ES_72h = 2916.16, ES_168h = 2913.99, ES_336h = 2910.21, ES_720h = 2901.51)
   inherits the ramp shape, which is what downstream option pricing sees.
   Under `explicit_level` all these numbers collapse to one figure, which is
   a lie about the model's actual dynamics.

The one weakness of `spot_to_next_linear` that a reviewer might raise is that
the linear ramp is arbitrary: nothing in the market says the January-baseload
should interpolate LINEARLY between spot and F_Feb; it could equally well
follow a load-shape profile or a mean-reverting decay towards F_Feb. If a
January quote (EBM0126) is ever published, all of this becomes moot — that
month flips from anchor-derived to hard-constrained (`january_status.how_to_remove`
in `calibration_result.json`).

---

## Files in this review folder

* `spot_flat/`, `spot_to_next_linear/`, `flat_next_month/` — full calibration outputs (10 files each)
* `option_value_comparison.csv` — K=3000 call at 24/72/168h across modes
* `hourly_curve_shape_january.csv` — January hourly curves stacked for plotting
* `hourly_jumps_<mode>.csv` — every |ΔF| > 5 TRY/MWh with boundary flag
* `near_term_anchor_sensitivity_fixed.csv` — sensitivity with production anchor mode
* `near_term_anchor_sensitivity_comparison.csv` — OLD vs NEW side-by-side
* `summary_metrics.json` — machine-readable per-mode metrics
* `_run_review.py` — the analysis script that produced everything above
* `COMPARISON_REPORT.md` — this file

Nothing in `outputs/market_calibration_final/` was modified; nothing is committed.
