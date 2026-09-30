# Project Status and Roadmap

Living document. Single source of truth on the status of every work
package. Numbers cited here come from the tracked artefacts under
`outputs/`; the CLAUDE.md guardrails on the accepted output trees
still apply.

## Model in one paragraph

European option on the expiry-hour Turkish day-ahead price (PTF,
TRY/MWh), valued 2025-12-31 20:00 UTC. Price level pinned to the
EPIAS VEP monthly baseload strip (smooth constrained KKT curve,
exact to 1e-12); residual around that curve follows a two-regime
TVTP Markov-switching OU with parameters inherited from the M9 fit
of the historical hourly PTF series and reconciled to the (P - F)
deseasonalized single-regime persistence by the v2-kappa refit
(`phi = 0.9246`, `kappa_per_hour = 0.078394`, half-life 8.84 h).
Pricing engine is a coupled Crank-Nicolson PDE with a Monte Carlo
cross-check. Every price is labelled *VEP-forward-curve anchored*:
only the first moment is market-identified; volatility and
regime-transition risk premia are not.

## Status by work package

### FW1: kappa refit on (P - F) residuals directly (DONE, promoted to production)

Yaml `phi`, `kappa_per_hour` and `half_life_hours` were reconciled
from the raw asinh(PTF) within-regime persistence (`phi = 0.999996`,
half-life about 19 y) to the deseasonalized single-regime AR(1)
appropriate for the (P - F) residual (`phi = 0.9246`,
`kappa_per_hour = 0.078394`, half-life 8.84 h). Pre-refit yaml
archived to `inputs/historical/archive/m2_frozen_parameters.PRE_V2_KAPPA_REFIT.yaml`.
The 2026-backtest `model_over_realized_ratio` moved from 7-13x
(pre-v2) to 0.35-0.97 (v2). Full v2 residual stack (two-factor
MS-AR(1) plus slow daily AR, level-scale factors, cap and floor)
is NOT integrated into the single-OU yaml; only the kappa was
promoted. Commit `e1140a2`. Outputs under
`outputs/market_calibration_final/` (accepted).

### FW2: risk-premium wiring (Q1 drift, Q2 transition) and identifiability envelope (DONE)

`price_forward_centered(..., eta_ij=...)` and the Q1 CLI flags
`--risk-premium-a0`, `--risk-premium-a1` are wired end-to-end;
`E^Q[P_t] = F(t)` is preserved by construction. Multiplicative Q2
form guarantees generator validity. The ex-post forward-premium
panel uses the 7 tracked VEP snapshots against realised hourly PTF,
look-ahead-guarded to 2025-12-31 20:00 UTC. Panel size is `n_obs
= 5-6` per horizon; pooled mean about 625 TRY/MWh; block-bootstrap
SEs 280-530 TRY/MWh. Identifiability is derived in
`docs/fw2_risk_premium_identification.md`: Q1 is O(a^2) in the
terminal variance, Q2 is O(eta), so Q2 is a priori more powerful.
The FW2 §4 sweep (480 rows) at the empirical upper bounds moves
the 72 h call at K = 3000 by up to +5.7 % under Q1 alone (`a_stress = 50`),
+21 % to -30 % under Q2 alone (`|eta| = 0.5`), and +29 % at the joint corner `(a_stress = 50, eta = (+0.75, -0.75))`. Production `(a, eta) = (0, 0)` retained; the
sweep is the manuscript's risk-premium uncertainty band. Outputs
under `outputs/fw2_risk_premium/`.

### FW3: closed-form benchmarks (Black-76, Bachelier, Lucia-Schwartz) (DONE)

Three benchmarks priced on the SAME F2.8 (K, T) grid (11 strikes
by 6 maturities), the SAME F(T) and the SAME discount factor as the
accepted PDE. Historical vol from real EPIAS PTF 2019-2025 (no
look-ahead past 2025-12-31 20:00 UTC): hourly log-return sample
gives 0.306 per sqrt(h) (annualised 28.64); daily log-return
sample gives 0.0351 per sqrt(h) (annualised 3.28), PRIMARY input
for Black-76; daily abs-return gives 78.07 per sqrt(h), PRIMARY
input for Bachelier. Lucia-Schwartz sigma and kappa come from the
yaml (M9 sigmas pooled by stationary occupancy, v2 kappa), not
re-fitted. Model-implied Black-76 ATM IV drops from 3.29 at 24 h
to 0.61 at 720 h, a mild negative smile at every maturity. Realized
2026 discounted-payoff backtest (66 contracts): Model MAE 151, B3
Lucia-Schwartz 157, B2 Bachelier 272, B1 Black-76 369 TRY/MWh. B3
is within one std_error of the model; B1 and B2 are about two
times worse. See `docs/fw3_benchmark_methodology.md` and
`outputs/fw3_benchmarks/`.

### FW4: two-covariate TVTP (experimental) and FW4-P ramp price impact (DONE, experimental)

FW4 added an experimental `rd_ramp_2d_experimental` mode with
M9-transferred slopes and a reconstructed ramp; the original ramp
definition could not be located in the shipped bundle, so the
build is labelled reconstruction throughout. FW9 then jointly
re-estimated the two-covariate TVTP on raw asinh(PTF) and produced
its own ramp series and slopes (`outputs/fw9_self_estimation/TVTP_2cov.pkl`;
LR = 869.93 df = 2 versus one-covariate). The fit is at the unit
root (`phi = 0.9999987`, half-life about 62 y) and did not meet
gradient tolerance (gradient norm 96.6 at n = 61 368), so the ramp
slopes carry no interpretable standard error. FW4-P priced the FW9
ramp channel with occupancy held fixed: at 72 h K = 3000 the ramp
effect is -1.042 % of the option value (versus -1.010 % from the
FW4 reconstructed ramp). Ramp remains experimental; default TVTP-1
mode is unchanged. Outputs under `outputs/tvtp2_experimental/` and
`outputs/fw4p_ramp_price_impact/`.

### FW5: real-terms scale_P (DONE inside FW9)

The documented scale definition applied to 2019-2025 gives 1 399.99
TRY/MWh against the shipped 282.48, and 3 092.05 on the CPI-deflated
series with base 2025-12. On the deflated series the regime sigmas
fall by 22 to 33 %. The shipped value is consistent with an
early-window median, but its reference window is not shipped. See
items (c) and (h) of `model_limitations.md` and
`outputs/fw9_self_estimation/scale_P.json`.

### F2.5 (predecessor of FW6a): WITHDRAWN

The pooled-vs-M9 comparison in
`outputs/market_calibration_final/model_comparison_pooled_vs_M9.md`
compared a pooled single-sigma OU against the two-regime TVTP with
DIFFERENT sigmas on the two sides, so the reported 38 to 50 % gap and
the term-structure spread conflate a regime-conditioning channel
with a sigma-level channel. Under the v2 kappa the M9-vs-pooled
contrast collapses to a nearly flat -53 % across 24 to 336 h. The effect of the regime mixture, measured at equal stationary variance, is about -9 % on the 72 h call at K = 3000 (FW9 round f). The F2.5 files are left untouched inside the accepted
output tree; the withdrawal is recorded here and in
`outputs/market_calibration_final/model_limitations.md`. See
`outputs/f25_v2_kappa/FW6a_report_TR.md` for the full rebuild.

### FW6a: F2.5 rebuild under the v2 kappa (DONE)

Independent reproduction of the F2.5 comparison under the shipped
v2 kappa. Verified line for line against both eras (pre-v2 markdown
and current CSV). Findings: (i) the term-structure spread the
original F2.5 attributed to regime conditioning is a pre-v2 kappa
artefact; (ii) most of the remaining flat -53 % gap is a sigma
level difference between M0 and M9 estimates, not the value of
regime conditioning; (iii) `E^Q[P_t] = F(t)` holds at machine
precision in every variant. See `outputs/f25_v2_kappa/`.

### FW6b: dropped

Superseded by FW10b, which covers the same out-of-sample validation
territory with a stricter day-ahead timing rule and the production
HPFC-shaped forward curve.

### FW7: TVTP ablation (DONE inside FW9)

Time-varying against constant transitions: LR = 1178.66 (df = 2) on
the transformed price level and LR = 1851 on the model-faithful A3
residual. At equal stationary variance the regime mixture lowers the
72 h call at K = 3000 by about 9 % relative to a single-regime
process (FW9 round f). See `outputs/fw9_self_estimation/`.

### FW8: absorbed into the manuscript

Discussion moved directly into `paper/sections/`; no separate
work-package artefact tree.

### FW9: independent re-estimation and validation of the residual process (DONE)

Full independent MLE of the two-regime TVTP MS-AR(1) with `RD_lag1`
covariate on the model-faithful A3 residual, plus a companion
two-covariate fit and a battery of sensitivities. Headline
findings: (i) yaml `kappa = 0.078` and the innovation scale (222 against 359 TRY/MWh per hour in 2025) sit individually far from the FW9 estimates, but they offset in the stationary residual variance, whose TRY sd is 582.5 versus
observed 2025 A3 residual TRY sd 531-624; KS statistic of yaml
against 2025 is 0.087 versus 0.125 for FW9e A3 full window and
0.159 for FW9f 2022-2025 regime-matched (FW9f §4); (ii) the FW9 profile-MLE alphas sit 14 to 18 standard errors from the derived yaml alphas, but the price impact is +3.75 TRY on the 72 h call at K = 3000 (0.4 % of the total FW9-vs-production gap); (iii)
`pi_filtered` source uncertainty is effectively closed for T >= 24
h; (iv) the FW9c/FW9d "yaml kappa outside bracket" framing is
WITHDRAWN by FW9f; (v) the TVTP LR statistic is 1178.66 (df = 2) on the transformed price level, 1851 on the model-faithful A3 residual over 2019-2025 and 1692.56 over 2022-2025 (FW9f §3). FW4 (estimation), FW5 and FW7 were closed inside FW9. See `outputs/fw9_self_estimation/`,
`FW9f_report_TR.md`, and `proposed_limitations_lines.md`.

### FW10: day-ahead publication timing audit and out-of-sample validation (DONE)

FW10b evaluates the shipped pipeline on 60 valuation days
(2026-01-05 to 2026-09-24, N_PATHS = 10 000) under the corrected
day-ahead timing rule (valuation at 11:00 TRT of day d, last known
PTF hour d 23:00 TRT, forward curve from the last VEP GGF strictly
before d 11:00 TRT, HPFC shape applied). PIT means run 0.33 to
0.44 across all models and horizons, well below 0.5 (VEP forwards
above realised); `var(z)` runs 4 to 11 for M0 across horizons.
Under the FW10b forward-residual decomposition the pure-residual
sd across 2026 is 723.1 TRY/MWh and the model residual is 387 to
518 TRY/MWh, a factor 1.33 to 1.55 too narrow; the forward-curve error mean is +385 to +794 TRY/MWh and its sd 697 to 843 TRY/MWh,
so the forward curve carries the bulk of the bias and about half
of the error variance. Delta-hedge effectiveness against the VEP
monthly quote is zero across every (model, h, K) cell: the VEP reference price rarely changes (14 of 1 354 contract-day pairs change on
2026-delivery contracts), so hedge P and L is identically zero and
the market microstructure, not the model, blocks hedge inference.
Under the corrected timing rule the shipped 2025-12-31 valuation
would be rebuilt from the 2025-12-30 VEP quote day, but the two
quote days are bit-identical, so the shipped (K, T) grid moves by
0 TRY. See `outputs/fw10_validation/`, `FW10_report_TR.md`,
`forward_residual_decomposition.md`, `day_ahead_timing_check.md`,
`vep_quote_staleness.csv`.

### FW11: stability of the stationary variance check across dates (DONE)

Seven FW4 valuation dates plus 2026-09-27 plus a 2026-only
sub-window, each with a 12-month rolling window; no new MLE fit,
only OLS AR(1) and descriptive statistics; look-ahead guarded on
every row. With the shape estimated on the twelve-month window,
realised over production-implied dispersion is 0.93 to 1.13 (mean
1.02) at the six dates from 2023-06-30 to 2025-12-31; the pooled
shape gives ratios 9 to 18 % higher. The 2022 crisis year sits at
1.26, the twelve months to 2026-09-27 at 1.65 and 2026 alone at 1.91,
with `sd/L` of 0.330 and 0.384 against the production 0.199. Sub-50 TRY/MWh share rises from about
1 % to 5-7 % in 2026 but excluding those hours moves the ratios
by less than 0.03. See `outputs/fw11_variance_stability/`,
`variance_stability.md`, `FW11_report_TR.md`.

### FW12: numerical convergence and scenario-plumbing repair (DONE)

Spatial convergence sweep at the production climatology z path,
T = 72 h, K = 3000: values 167.0716, 166.8415, **166.7477 (n =
1201, production)**, 166.7070, 166.6940 at n = 301, 601, 1201,
2401, 4801. Observed order 1.21 to 1.39 (payoff kink and far-field
boundary interaction). Richardson extrapolation from (2401, 4801)
gives V_star = 166.686, so the production 1201-node value carries
about 0.037 % relative error, two orders of magnitude below the
FW2 risk-premium envelope. Boundary sensitivity `|Delta V / V| <=
0.024 %` across `n_std in {4, 5, 6, 7.5, 9}`. FW12b repaired the
scenario-plumbing path so every FW9 and downstream pricing artefact
uses the same climatology z path as `run_pde.py price`, and locked
this into a test. Production grid (1201 nodes, 2 steps/hour, n_std
= 6) is kept. See `outputs/fw12_convergence/`, `grid_recommendation.md`.

## What is left

The engineering side of the project is finished. The remaining work
is manuscript production and outreach:

1. Paper v2. Fold the FW9 to FW12 findings into the manuscript,
   principally (a) add the withdrawal of F2.5 and cite the FW6a
   rebuild, (b) add the FW9 stationary-variance closeness and the
   FW11 stability band across seven dates, (c) add the FW10b PIT
   and forward-residual decomposition and the VEP quote staleness
   finding, (d) add the FW12 convergence table and the 0.04 %
   Richardson-extrapolated error, (e) note the 2026 break in
   `sd / L` and the constant scale-mapping caveat.
2. Advisor read on the v2 draft.
3. SSRN v2 (or arXiv q-fin.PR) once the read comes back.
4. Journal submission.

## Second-paper future work

Items that a follow-up paper (or a follow-up chapter) should address;
none of them are required for the current manuscript.

1. **Price-level-dependent scale or level-sensitive volatility.** The
   FW11 sd/L break in 2026 (0.330 and 0.384 against the production value of 0.199) and the FW9 note on the structural upper bound of
   the asinh + delta mapping at fixed `scale_P` both point at a
   constant-scale limit. Candidates: rolling or shorter-window
   re-estimation of `scale_P`, `scale_P` on a real (inflation-deflated)
   series, or a level-sensitive residual volatility.
2. **Regime-sensitive forward-curve error.** The FW10b PIT and
   forward-residual decomposition attribute most of the 2026
   out-of-sample error to the forward curve, not to the residual.
   A regime-aware forward-curve error model (with a separate
   treatment of solar oversupply hours) is a natural next step.
3. **Risk-premium identification when option data exists.** If a
   liquid Turkish electricity option or a comparable derivative
   ever becomes available, the FW2 wiring can be reversed to
   identify `(a, eta)` from prices rather than exercised as a
   sensitivity envelope.
