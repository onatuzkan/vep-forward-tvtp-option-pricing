# M9 TVTP Integration — Before / After Comparison

Snapshot report generated when the M9-derived TVTP parameters were promoted
from `outputs/tvtp_integration_draft/` (this folder) into
`inputs/historical/m2_frozen_parameters.yaml`.

The old (placeholder) yaml is preserved at
`inputs/historical/archive/m2_frozen_parameters.OLD_PLACEHOLDER.yaml`.
The old scenario-sweep table is preserved at
`outputs/scenario_sweep/rd_scenario_sweep.OLD_PLACEHOLDER.csv`.

---

## 1. Parameter values

| field | OLD placeholder | NEW M9-derived | |new|/|old| |
|---|---:|---:|---:|
| `sigma_y_normal` | 0.0075052 | **0.0035348** | 0.47 |
| `sigma_y_stress` | 0.1729706 | **0.0924067** | 0.53 |
| `tvtp.alpha01` | −0.876728 | **−1.015666** | 1.16 (derived) |
| `tvtp.gamma01` | −0.680174 | **−0.583778** | 0.86 |
| `tvtp.alpha10` | −1.347666 | **−1.895229** | 1.41 (derived) |
| `tvtp.gamma10` | +0.203006 | **+0.077698** | **0.38** |

All four TVTP signs preserved.  Notable: `|gamma10|` collapses to 38% of
placeholder (RD sensitivity of stress→normal is much weaker than the
placeholder assumed); `|sigma_stress|` falls to 53% (stress-regime variance
rate falls to 0.53² ≈ 0.28× the placeholder value).

## 2. 72 h K=3000 call — pricing benchmark

Both calibrations use the identical spot_to_next_linear forward curve;
only the residual dynamics change.

| quantity | OLD placeholder | NEW M9-derived | Δ | Δ% |
|---|---:|---:|---:|---:|
| F(T=72h) | 2916.16 | 2916.16 | 0 | 0.0 % |
| residual sd at expiry | 3160.04 | **1813.52** | −1346.5 | **−42.6 %** |
| PDE call value | 1210.15 | **677.23** | −532.9 | **−44.0 %** |
| MC call value | 1225.64 ± 7.48 | **685.00 ± 4.24** | −540.6 | −44.1 % |
| MC |z| vs PDE | 2.07 | 1.83 | — | (better parity) |
| MC E[P_T] | 2925.13 | 2925.48 | +0.35 | ≈0 |
| MC P(P_T < 0) | 0.1776 | **0.0531** | −0.1245 | **−70.1 %** |

Interpretation
* Residual sd drops by 43 % because sigma_stress is ~half the placeholder.
* The K=3000 call halves in value — this is a real change in the model's
  volatility assumption, not a bug.
* MC-vs-PDE parity improves: z-statistic drops from 2.07 σ to 1.83 σ.
* Negative-price mass under the additive residual (which is a model
  pathology, not a real market feature) also drops from ~18 % to ~5 %.
  Under a multiplicative residual this would be zero by construction.

## 3. Scenario sweep — RD-offset sensitivity (72 h call, K=3000)

Both tables use the same climatology-based `z(t)` path, shifted uniformly by
`rd_offset_sigma` standard deviations.  Old base call ≈ 1210 TRY/MWh;
new base ≈ 677 TRY/MWh.

| offset (σ) | OLD sd | OLD call | OLD % vs base | NEW sd | NEW call | NEW % vs base | ratio call NEW/OLD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −2.0 | 3755 | 1450.56 | **+19.87 %** | 2032 | 766.24 | **+13.14 %** | 0.53 |
| −1.5 | 3651 | 1408.62 | +16.40 % | 1993 | 750.30 | +10.79 % | 0.53 |
| −1.0 | 3519 | 1355.41 | +12.00 % | 1944 | 730.51 | +7.87 % | 0.54 |
| −0.5 | 3356 | 1289.46 | +6.55 % | 1885 | 706.31 | +4.29 % | 0.55 |
| **0.0** | 3160 | **1210.15** | 0 | 1814 | **677.23** | 0 | 0.56 |
| +0.5 | 2933 | 1118.02 | −7.61 % | 1730 | 642.98 | −5.06 % | 0.58 |
| +1.0 | 2681 | 1014.87 | −16.14 % | 1635 | 603.47 | −10.89 % | 0.59 |
| +1.5 | 2410 | 903.73 | −25.32 % | 1528 | 558.90 | −17.47 % | 0.62 |
| +2.0 | 2131 | 788.67 | **−34.83 %** | 1412 | 509.92 | **−24.70 %** | 0.65 |

### RD sensitivity flattening

Full range of the call value across the ±2σ sweep, relative to the base:

| metric | OLD | NEW | flattening |
|---|---:|---:|---:|
| max increase (−2σ) | +19.87 % | +13.14 % | 0.66 |
| max decrease (+2σ) | −34.83 % | −24.70 % | 0.71 |
| peak-to-peak swing | 54.7 pp | 37.8 pp | **0.69** |
| slope (Δpct / Δσ) near 0 | −13.5 pp/σ | −8.8 pp/σ | **0.65** |

The M9-derived yaml **flattens the RD sensitivity by ~30–35 %**.  This is
the direct, quantitatively expected effect of `|gamma10|` collapsing to
0.38× of the placeholder: with weaker RD-driven modulation of the
stress→normal transition, the mixture volatility becomes less responsive
to the RD path, and the option value is a smoother function of the RD
scenario.

Note also the residual-sd side: the OLD sweep spans 2131 → 3755 (76 % range
across ±2σ); the NEW sweep spans 1412 → 2032 (44 % range).  The residual sd
itself is also less RD-responsive, for the same reason.

## 4. Summary

* Direction: option values FALL and RD sensitivity FLATTENS.  Both moves
  are qualitatively expected from the parameter update (smaller sigmas,
  smaller gamma10).
* Numerical magnitudes:
  * ATM call value → 56 % of placeholder (44 % lower).
  * Residual sd at expiry → 57 % of placeholder.
  * RD-sensitivity slope near 0σ → 65 % of placeholder.
* Sign preservation: every TVTP coefficient sign matches the placeholder;
  no economic-direction surprises.
* Test suite: 168 passed + 1 skipped after the update (the skip flags the
  legacy-reference reconstruction test, whose reference file is a
  historical snapshot under the old sigmas and must be regenerated
  separately to re-enable).

The move brings the option-pricing model into consistency with the actual
M9 fit (occupancy 33 % / 67 %, mean durations 3.76 h / 7.56 h) instead of
the placeholder targeting (roughly 94 % / 6 % durations of 300 h / 20 h).
Since option pricing is not identified without option premia, this is a
*first-moment* consistency improvement: the model now uses the same
regime dynamics under which its residual volatilities were estimated.
