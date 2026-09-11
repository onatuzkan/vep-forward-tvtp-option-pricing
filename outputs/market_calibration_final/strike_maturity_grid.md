# Strike × maturity option-value grid

66-point PDE sweep (11 strikes × 6 maturities) plus companion put pricings
(132 PDE solves total).  Same production settings as the accepted
calibration:

* forward curve = `hourly_forward_curve.csv` (accepted VEP-anchored run)
* `pi_filtered = [0.9320, 0.0680]` (M2 shipped filter — default)
* TVTP `z(t-1)` = climatology path (train_end = 2022-12-31 20:00 UTC)
* residual sigmas from `m2_frozen_parameters.yaml` (M9 CSV, yaml-swapped)
* `r_annual = 0.40`; `--risk-premium-a0 = --risk-premium-a1 = 0`
* residual PDE mode = additive
* grid: 1201 space nodes, `n_std = 6`

## Grid definition

* **strikes**: `[2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000]`
  TRY/MWh — 11 uniform points at 200 TRY/MWh spacing.  Chosen to bracket
  the current forward level (F(T) ≈ 2900-2917 TRY/MWh) with ~±35 %
  moneyness on each side.  Config's `diagnostics.strikes` (5 uniform
  points at 500 TRY/MWh spacing) was too coarse for a heat-map; the 11-
  point grid keeps the sweep manageable (132 PDE solves ≈ 15-20 min
  wall-clock).
* **maturities (hours)**: `[24, 48, 72, 168, 336, 720]` — day-ahead
  through one month, log-spaced.
* Total: `11 × 6 = 66 (strike, maturity)` points, priced for both call
  and put (put-call parity gives a free correctness check).

## Correctness check

Put-call parity `C - P = e^{-r τ} (F(T) - K)`:

* Max `|C - P - e^{-r τ}(F − K)|` across all 66 rows = **1.0e-6 TRY/MWh**.

PDE stencil holds parity to solver precision at every point in the grid.

## Forward level per maturity (near-term anchor + curve)

| maturity | F(T) (TRY/MWh) |
|---:|---:|
| 24 h | 2917.24 |
| 48 h | 2916.70 |
| 72 h | 2916.16 |
| 168 h | 2913.99 |
| 336 h | 2910.21 |
| 720 h | 2901.51 |

F(T) drifts very slightly downward with maturity (spot 2917.78 → Feb quote 2900.99
via the linear ramp), so ATM strikes shift left by ~16 TRY/MWh across the
horizon.  The heatmaps overlay this F(T) trajectory as a white dashed line
so the reader can locate the ATM contour at each maturity.

## Selected values (ATM strike K = 3000)

| maturity | call | put | put/call | call / F(T) |
|---:|---:|---:|---:|---:|
| 24 h | 368.17 | 450.84 | 1.22 | 0.126 |
| 48 h | 550.06 | 633.18 | 1.15 | 0.189 |
| 72 h | 687.04 | 770.61 | 1.12 | 0.236 |
| 168 h | 1073.90 | 1159.25 | 1.08 | 0.368 |
| 336 h | 1525.78 | 1614.21 | 1.06 | 0.524 |
| 720 h | 2209.01 | 2304.32 | 1.04 | 0.761 |

Put > call at K = 3000 across every maturity because strike is a hair OTM
of the forward at all horizons (F(T) ≤ 2917 < K = 3000).  The put-call
ratio compresses toward 1 as τ grows and residual variance dominates the
moneyness gap.

## Reading the heatmaps

### `strike_maturity_heatmap_call.png` — raw call value

* **x = maturity** (24 → 720 h, log-spaced by index but linear on
  the axis so the 720 h column is visually wide)
* **y = strike** (2000 → 4000 TRY/MWh, linear)
* **colour** = call value in TRY/MWh (viridis, low = dark, high = light)
* **white dashed line + dots** = F(T) trajectory (ATM contour)

Vertical bands of similar colour: at each fixed strike, value grows
monotonically with maturity — the well-known "value of optionality"
increases with time-to-expiry.  Horizontal bands: at each fixed maturity,
value decays as strike moves further OTM (K ≫ F).

### `strike_maturity_heatmap_moneyness.png` — call / F(T)

* Same axes; colour is `call / F(T)`, ranging from 0.02 (deep-OTM 4000-
  strike, 24 h) to 1.36 (deep-ITM 2000-strike, 720 h).  This normalisation
  strips out the F(T) drift and makes the moneyness / time-value pattern
  directly readable.
* The white ATM contour cuts the colour bar near 0.13-0.76 (small at
  short maturity, large at long).  Points above the contour are OTM
  and always below 1; points below are ITM and rise smoothly toward 1
  as strike drops.

## Patterns visible in the grid

1. **Value of optionality grows with maturity, sub-linearly.**  At
   ATM (K = 3000): 24 h call = 368 TRY, 720 h call = 2209 TRY — a 6×
   growth over 30× time.  Under an OU with roughly Brownian variance
   accumulation (near-unit-root κ) we'd expect option value to grow
   like √τ or `σ√τ`; 6× / √30 ≈ 1.1 — the sub-linear-in-τ scaling is
   the expected signature of a diffusion-driven price process, not a
   drift-driven one.

2. **ITM / OTM asymmetry is time-dependent.**  At short τ = 24 h the
   deep-OTM K = 4000 call is 80 TRY (2 % of F); the deep-ITM K = 2000
   call is 1023 TRY (35 % of F) — most of the ITM value is intrinsic
   `F − K = 917`, plus a small time-value premium of ~106 TRY.  At τ =
   720 h the same strikes are 1765 and 2719 TRY — the ITM call value
   is nearly F, and the OTM call has grown to 61 % of F.  Long
   maturities make the strike almost irrelevant relative to the
   variance channel.

3. **Month-boundary jumps in the underlying forward do NOT show up
   as banding in the grid.**  The most-visible boundary in the
   accepted curve is Jun→Jul (∼47 TRY/MWh jump between adjacent
   hours), but no strike / maturity in this sweep intersects that
   boundary — the largest maturity (720 h) sits comfortably inside
   Jan-Feb.  For a longer-dated grid (>2160 h) one would expect to
   see faint discontinuities where F(T) itself steps at a delivery-
   month boundary.  Not observed here.

4. **Put-call parity is exact everywhere.**  The 1e-6 TRY/MWh ceiling
   on `|C − P − e^{−r τ}(F − K)|` is solver precision (single-round
   floating-point on discount factor arithmetic), not physically
   meaningful.  This is a strong confidence signal for the PDE stencil
   at every point in the grid.

## 2D slice plots (academic alternative to a 3D surface)

Two additional line plots view the same 66-point surface from the two
canonical academic angles — a strike-slice ("smile / skew") view and a
maturity-slice ("term structure") view.  Both use a viridis palette
ordered by the sliced dimension (colourblind-safe), no top / right
spines, faint horizontal grid, and 200 dpi for print.

* **`strike_slices.png`** — call value vs strike, one line per maturity
  (24 / 48 / 72 / 168 / 336 / 720 h).  Per-maturity ATM markers appear
  as short dashed vertical lines at each maturity's F(T).  Because
  F(T) drifts only 16 TRY/MWh across the horizon (2917 → 2901), the six
  ATM lines nearly overlap; a single "F(T) range: 2901-2917"
  annotation is more informative than six separate labels.
* **`maturity_slices.png`** — call value vs maturity (log x-axis
  because maturities span 24 → 720 h), one line per strike (11
  strikes 2000-4000).  The ATM-like K = 3000 line is drawn in black
  and thicker so the reader's eye lands on the reference case first;
  the other 10 strikes fan above and below it, coloured by strike
  level.

### Pattern in `strike_slices.png` — negatively-sloped, no visible smile

At every maturity the call value is a **monotone-decreasing, mildly
convex function of strike**.  Slopes near ATM at 72 h:

| K range | ΔC (TRY / 200 TRY strike step) | ΔC/ΔK |
|---|---:|---:|
| 2600 → 2800 | −109.12 | −0.546 |
| 2800 → 3000 | −100.37 | −0.502 |
| 3000 → 3200 | −91.63 | −0.458 |
| 3200 → 3400 | −83.00 | −0.415 |

The slope softens monotonically as strike moves OTM, which is the
`N(d₁)`-style delta compression a standard diffusion model produces.
**No smile / skew is visible** — expected under this pricing framework
because the model has neither jump risk nor stochastic-vol overlay
(only regime-switching mixture of two Gaussian OU sigmas).  A future
extension that adds skew (e.g. jump-diffusion, or Q2 transition
premia) should be visible directly as a departure from the near-
symmetric convexity seen here.

### Pattern in `maturity_slices.png` — concave, sub-linear-in-τ term structure

Every strike's line is monotone-increasing and **concave** in maturity
(on the log-x axis).  Concavity means the value gained by extending
maturity from 24 h → 48 h is much larger than from 336 h → 720 h — the
signature of `σ · √τ` variance accumulation under a diffusion.  ATM
K = 3000 rises from 368 (24 h) → 687 (72 h) → 1526 (336 h) → 2209
(720 h), a 6× growth over 30× time (i.e. sub-linear).  Deep-OTM
K = 4000 grows steepest in log-log terms (80 → 313 → 1091 → 1765 —
22× growth) because at short τ its value is dominated by the small
tail probability that F(T) + ε > K, which grows fastest as the
residual variance grows.  Deep-ITM K = 2000 grows slowest in relative
terms (1023 → 1275 → 2058 → 2719, 2.7×) because most of its value is
already intrinsic `F − K` at short τ.

## Files

* `outputs/market_calibration_final/strike_maturity_grid.csv` —
  66 rows, columns: strike, maturity_h, F_T, call, put,
  put_call_LHS, put_call_RHS, put_call_parity_error,
  residual_sd_T.
* `outputs/market_calibration_final/strike_maturity_heatmap_call.png`
* `outputs/market_calibration_final/strike_maturity_heatmap_moneyness.png`
* `outputs/market_calibration_final/strike_slices.png` — 2D slice-by-
  maturity view (see "2D slice plots" section above).
* `outputs/market_calibration_final/maturity_slices.png` — 2D
  slice-by-strike view, log x-axis.

No production code was touched.  Full test suite (172 passing)
unaffected.
