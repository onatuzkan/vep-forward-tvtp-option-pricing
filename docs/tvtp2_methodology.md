# Two-covariate TVTP (RD_lag1 + RD_Ramp_1h_lag1) — EXPERIMENTAL (FW4)

> **Status: `experimental_reconstructed`.** Label carried by every output:
> *"M9-transferred slopes + reconstructed ramp + derived intercepts, zero
> transition premium"*. The original construction of the ramp covariate was
> not found, so this mode does **not** reproduce M9 and does **not** replace
> any accepted result (paper tables, `outputs/market_calibration_final/`,
> `outputs/scenario_sweep/`, `inputs/historical/m2_frozen_parameters.yaml`).
> The production default remains the single-covariate mode `rd_lag1_1d`.

Companion artefacts:

| artefact | content |
|---|---|
| `inputs/historical/tvtp2_frozen_parameters.yaml` | frozen 2D parameter set (separate from the 1D YAML) |
| `outputs/tvtp2_experimental/derivation/tvtp2_derivation_audit.{json,md}` | windows, scalers, sample dates, dropped rows, roots, validation metrics, evidence |
| `outputs/tvtp2_experimental/PROVENANCE_REPORT.md` | provenance report (evidence levels, hashes, open limitations) |
| `outputs/tvtp2_experimental/comparison/` | controlled 1D vs 2D comparison (R0–R4, sensitivities, PDE vs MC, convergence) |
| `outputs/tvtp2_experimental/{market_calibration,diagnostics,scenario_sweep}/` | the CLI run in the 2D mode |
| `config/forward_centered_tvtp2_experimental.yaml` | selects the mode (`extends` the production config) |

---

## 1. Scope

In scope: the transition law only. The residual (single OU factor, additive,
kappa, sigma, x0, m, a), the forward curve, the contract, the discount rate and
the initial regime law are those of the production model. Explicitly **not**
used: the two-factor residual, slow factor, level uncertainty and price cap of
`residual_v2.py` (its fast-factor ramp slopes are a different object, fitted
to a different dependent variable, and are never mixed with the M9 transfer).

## 2. Sources and what they establish

Searched for the ramp definition: the working tree, all git history (incl.
the deleted `team_share_2026_09_01.zip`), the SSRN manuscript, the
calibration bundle, the connected Google Drive, and the design note. **No
construction code, scaler file (`covariate_scaling.json`) or shipped p series
(`pde_timeseries.parquet`) exists in any accessible source.**

| object | finding | level |
|---|---|---|
| RD slopes | M9 `transition_coefficients.csv`; production yaml values (10 dp) used | D |
| ramp slopes | same file, row `RD_Ramp_1h_lag1`; exact doubles (CSV parsed round-trip) | D |
| joint estimation | AME/slope equal for both covariates in each equation (\|Δ\| ≈ 1e-15) | Ç (strong) |
| ramp direction / lag / RD vs z | not found → reconstruction | Y (original B) |
| ramp scale | E[1/p10] = 8.582 standardized vs 8.585 in M9 (raw dz: 7.68) | Ç (consistency only) |
| calendar-time lags upstream | `rd_lag1_standardized.csv` carries z(02:00) at 2016-03-27 03:00 — an hour absent from `rd_standardized.csv`; row order would give z(2016-03-26 23:00) | Ç (for RD_lag1) |
| M9 window W9 | first n_train = 78 905 rows = 2016-01-01 01:00 → 2024-12-31 20:00 UTC | Ç |
| intercepts | not exported → derived | derived |

D = in a file, Ç = inference from data, Y = reconstruction, B = unknown.

## 3. Regime labels (one atomic swap)

Raw M9: state 0 = high volatility. Production: 0 = normal (low σ), 1 = stress.
`tvtp2.m9_raw_to_production` swaps σ, both slope pairs, AMEs, durations and
occupancy together and refuses a bundle whose state 0 is not the high-vol one.

| production | raw M9 | value |
|---|---|---|
| γ01 (normal→stress, RD) | gamma10 RD_lag1 | −0.5837780825 |
| γ10 (stress→normal, RD) | gamma01 RD_lag1 | +0.0776983936 |
| h01 (normal→stress, ramp) | gamma10 RD_Ramp_1h_lag1 | −0.06939531482057137 |
| h10 (stress→normal, ramp) | gamma01 RD_Ramp_1h_lag1 | +0.4055443473030015 |
| d_n, d_s | mean_duration_state1, _state0 | 3.759199 h, 7.559716 h |

Checks: every AME has the sign of its slope (Λ′ > 0), and σ_normal < σ_stress
after the swap. The legacy `alpha01/alpha10` of the 1D yaml are **not** used.

## 4. Equations

Hourly UTC label t denotes the delivery hour [t, t+1). G is the complete
hourly grid of the sample; missing hours stay NaN.

    RD_t = Demand_t − Wind_t − Solar_t,        z_t = (RD_t − m_z)/s_z   (TRY window, inherited)
    dz_t = z_t − z_{t−1h}                      (only if both labels exist on G)
    r_t  = (dz_t − m_r)/s_r                    (train-only scaler, ddof = 1)
    x_{t−1} = (z_{t−1h}, r_{t−1h})             (covariates of S_{t−1} → S_t)

    p01_t = Λ(α01 + γ01 z_{t−1} + h01 r_{t−1}),   p10_t = Λ(α10 + γ10 z_{t−1} + h10 r_{t−1})

Because r is built from dz, it is invariant to the (unknown) z scaler:
r = (ΔRD − m_ΔRD)/s_ΔRD (verified: max |Δr| = 2.6e−15 under re-standardizing z).

Generator (unchanged exact matrix log, `probs_to_generator`): with
s = p01 + p10 and λ = −ln(1 − s),

    q01 = λ p01 / s,   q10 = λ p10 / s,   q01 + q10 = λ,   exp(Q · 1h) = Π     (0 < s < 1)

Pricing alignment (identical to production): `l(τ) = floor(t_v + τ − 1h)`,
z(τ) = z_hour[l(τ)], r(τ) = r_hour[l(τ)]; both come from ONE path object.
Residual, moments (6 linear ODEs) and the two coupled 1-D PDEs are unchanged:
deterministic covariate paths only change the scalar functions q01(τ), q10(τ),
so the state space stays (x, S). Prices are conditional on the path.

Measure: q^Q = q^P (η01 = η10 = 0) — the **zero transition premium
assumption**, labelled as such; forward quotes cannot identify η because the
centering makes F independent of η. No Q2 premium is implemented or claimed.

## 5. Historical panel and data rules (`tvtp2.build_covariate_panel`)

* The CSV is mapped onto G (87 665 rows on 87 668 labels; missing:
  2016-03-27 00:00, 01:00, 02:00 UTC). Increments are never taken in row
  order (the row diff at 2016-03-27 03:00 would book a 4-hour move, −0.5116,
  as a one-hour ramp).
* z and r are lagged by the same calendar shift on G.
* Transition sample D: labels whose x_{t−1} is observed. The first two hours
  of the sample (2016-01-01 01:00, 02:00) and four gap transitions
  (2016-03-27 01:00–04:00) are **dropped and counted**; nothing is filled and a
  missing ramp is never zero. 2016-03-27 00:00 is kept (own z missing, x_{t−1}
  observed).
* Train-only standardization: (m_r, s_r) from increments with label ≤ the
  window end; data after the window cannot change m_r, s_r or α (tested by
  scrambling and by truncating the post-window history).

| window | end (UTC) | dz n | m_r | s_r |
|---|---|---|---|---|
| W_T (z scaler window) | 2022-12-31 20:00 | 61 359 | 3.52e−05 | 0.266393 |
| **W9 (primary)** | 2024-12-31 20:00 | 78 903 | 3.42e−05 | **0.271450** |
| all | 2025-12-31 20:00 | 87 663 | 3.18e−05 | 0.274525 |

## 6. Intercepts (moment matching, not estimation)

    mean_D Λ(α01 + γ01 z + h01 r) = 1/d_n,    mean_D Λ(α10 + γ10 z + h10 r) = 1/d_s

Each left side is strictly increasing in α (derivative = mean Λ′ > 0), with
limits −1/d and 1 − 1/d, so a single root exists; Brent on [−20, 20],
xtol = rtol = 1e−12. D = W9 (78 902 transitions).

| set | α01 | α10 | note |
|---|---|---|---|
| 1D production (repo) | −1.015666 | −1.895229 | all rows, contemporaneous z (re-derived to 1e−10) |
| 1D, D = W9 (R1) | −1.045554 | −1.890591 | same D as the 2D primary |
| **2D, D = W9, scaler W9 (R3, frozen)** | **−1.0490993394** | **−1.9558390317** | residuals −1.3e−15, 0 |
| 2D, scaler W_T | −1.049188 | −1.958235 | |
| 2D, D = full sample | −1.019247 | −1.961902 | |
| 2D, z re-standardized on W9 | −1.094972 | −1.948266 | z-scale sensitivity |

Validation metrics (reported, never targeted):

| metric | 2D primary | M9 | comment |
|---|---|---|---|
| forward-recursion stress occupancy | 0.6352 | 0.674578 | −5.8 %; covariate-conditional chain vs smoothed average — a gap is expected |
| (1/d_n)/(1/d_n + 1/d_s) | 0.66788 | 0.674578 | uses the targets only — **not** a validation (the 1D doc's "−0.99 %" is not one either) |
| AME ratio p01 / p10 | 0.1823 / 0.1114 | 0.1963 / 0.1112 | |
| E[1/p01] / E[1/p10] | 4.690 / 8.582 | 4.135 / 8.585 | auxiliary (local approximation) |

**p01 target conflict.** E[p(1−p)] ≤ p̄(1−p̄) = 0.19525 for p̄ = 1/3.759199,
but M9's AME ratio is 0.19628, so M9's own mean p01 is ≥ 0.26823: the duration
target is not the mean of M9's fitted p01 path. The conflict exists in the 1D
production set too; it is documented, not resolved.

**Ramp scale / timing evidence.** Re-deriving α10 for alternative ramps:
×0 → E[1/p10] 7.598, raw dz → 7.679, standardized → 8.582, ×1.1 → 8.790;
no lag 8.483, lag 2 h 8.649, sign flipped 8.629. The moment supports the
standardized scale but cannot identify direction or lag.

## 7. Embeddability decision

Π has eigenvalues 1 and 1 − s. For s = 1 the logarithm does not exist; for
s > 1 the second eigenvalue is negative while every 2-state generator gives
e^{−(q01+q10)} > 0 — no continuous-time chain reproduces Π. Hence:

* the 2D pricing path uses `probs_to_generator(..., on_nonembeddable="raise")`:
  any node with s ≥ 1 raises `EmbeddabilityError` with an audit (count,
  share, max s and its label, first/last violation, count of s ≥ 0.95);
* every scenario path is audited on its hour labels before pricing
  (`TVTPCovariatePath.embeddability`), and the CLI exits with code 2 and
  prints the audit; nothing is clipped;
* the historical 2D path has no violation (max s = 0.8408 at 2020-05-25 06:00
  UTC; 99.99 % quantile 0.803); climatology max s = 0.524 (−2σ: 0.790);
* violations start at a −4.5σ climatology offset (60 of 721 hours, max
  s = 1.0076) — beyond the observed z support (min −3.64) — while the 1D law is
  still embeddable there. At ≤ −4.75σ the 1D production law also reaches
  s ≥ 1; the single-covariate API keeps its historical clip-and-warn default
  (unchanged, outside the accepted ±2σ sweep) — flagged here as a pre-existing
  limitation;
* the discrete-time switching alternative is documented in the design note but
  **not implemented**; there is no automatic fallback;
* ODE accuracy guard: the RK4 sub-step also satisfies h_sub·λ_max ≤ 1 (never
  active on the production paths, so 1D results are bit-identical).

## 8. Scenario paths (`scenarios.CovariatePathBuilder`)

Hour labels [t_v − 2h, T − 1h] are built first; r is the hourly difference of
that path; then the labels are sampled on the 0.25 h master grid exactly as in
the production `_build_tvtp_scenario`, and solvers interpolate between master
nodes as before.

| mode | z at hour labels | ramp |
|---|---|---|
| constant | offset | −m_r/s_r (≈ −1.3e−4): the standardized zero increment, not a filled zero |
| climatology | month × hour mean of z on the TRY window + offset (bit-identical to 1D) | hourly difference of the climatology path (keeps the diurnal ramp; sd 0.91 in January) |
| custom | every required label must be in the CSV (hour-aligned, unique, finite, parsed exactly); missing labels are listed in the error; no interpolation | hourly difference |

Initial hours: default `scenario` (production rule: labels ≤ t_v come from
the scenario). Option `observed` uses the observed z at t_v − 2h, t_v − 1h,
t_v (never offset). Offsets shift z only; the ramp of a shifted path is
unchanged. The frozen ramp scaler is recomputed from the history on its window
whenever a path is built and a mismatch is refused (unit guard); a model also
refuses a path standardized with a different scaler.

Alignment convention (inherited, affects 1D and 2D alike): between master
nodes the covariates are interpolated linearly and the ODE/PDE interpolate q
between their own nodes, so the effective q path depends slightly on the
solver grid. On the 72 h reference contract the ODE stress probability at T is
0.6406 on the 0.5 h PDE grid and 0.6376 on 0.25 h or finer grids (1D: 0.5939
vs 0.5915). See §11 for the price impact.

## 9. Integration

* `generator.py`: `TVTP2Coefficients` (shape / missing-value / unit guards,
  no broadcasting, no default ramp), `embeddability_report`, the `raise`
  policy; `TVTPCoefficients` unchanged.
* `params_frozen.py`: `load_tvtp2_parameters` refuses a reconstructed ramp
  labelled verified, non-zero transition premia, look-ahead windows, and
  checks the base yaml hash.
* `forward_centered.py`: `resolve_covariates` / `generator_path` /
  `moments` / `price_forward_centered` / `simulate_forward_centered` take one
  `covariate_path`; the moment ODE and the PDE share one q array, the MC
  evaluates the same path on its grid. New MC outputs: E[X_T] and
  P(stress at T) with SEs against the ODE; `simulate_residual_at_hours`.
* `market_calibration.py`, `model_modes.py`, `run_pde.py`: `tvtp_mode` and
  `tvtp_provenance` in `calibration_result.json`, `parameter_identification.json`,
  `tvtp_provenance.json`, `calibrated_config.yaml`, the audit and limitations
  markdown, the anchor-sensitivity CSV and the CLI printout; experimental outputs
  are refused in `outputs/market_calibration_final`,
  `outputs/forward_centered_diagnostics` and `outputs/scenario_sweep`.

## 10. Validation (tests use repository data only)

| requirement | tests |
|---|---|
| 1. h = 0 reproduces 1D probabilities and prices | `test_zero_ramp_reproduces_single_covariate_probabilities_bitwise`, `test_zero_ramp_reproduces_the_production_price_bit_for_bit` (166.7477, call and put, identical V_regime), `..._monte_carlo`, comparison R2 = R1 (all 20 contracts, exactly 0) |
| 2. labels, probabilities, row sums, expm | `test_label_swap_*`, `test_two_covariate_probabilities_match_the_closed_form`, `test_generator_rows_sum_to_zero_and_expm_returns_the_one_hour_matrix` (scipy expm, max error < 1e−13), rejection tests |
| 3. lags, train-only scaler, gaps | `test_z_and_ramp_are_lagged_together_by_one_hour`, `test_ramp_is_a_calendar_difference_*`, `test_first_two_hours_*`, `test_dropping_a_real_row_*`, `test_ramp_scaler_and_intercepts_ignore_data_after_the_window`, custom-CSV tests |
| 4. centering, VEP months, parity (non-circular) | MC mean of X_T vs ODE μ_X with m = (150, −400), a = (0.05, −0.03) (\|μ_X\| > 50); simulated February-2026 average of P_h vs the VEP quote with m ≠ 0; parity with m ≠ 0 compares the PDE's implied mean with the ODE — the gap (0.043 → 0.015 → 0.004 TRY/MWh for 144/288/576 steps) is second-order discretization error, strike-independent and equal in size to the 1D model's |
| 5. PDE vs MC on the same 2D path | `test_pde_matches_monte_carlo_on_the_same_2d_path` (40 000 paths, dt 0.05, SE reported); comparison tables |
| 6. controlled 1D vs 2D | `test_controlled_comparison_changes_only_the_transition_law` (same curve / spec / π0 objects, same grid) and the full comparison run |

## 11. Results (comparison run, `outputs/tvtp2_experimental/comparison/`)

Fixed: one curve object, κ = 0.078394/h, σ_y = (0.0035348070, 0.0924066544),
π0 = (0.931977, 0.068023), r = 0.40, x-grid per maturity = union over all 22
runs and strikes (1 201 nodes), production time grid, MC 40 000 paths,
dt = 0.05 h, seed 20260808 (common random numbers). On this common grid R0 at
72 h / 3 000 is 166.7832 (production grid: 166.7477; grid-width effect 0.02 %).

Ramp effect R3 − R1 (calls; puts move by the same TRY amount, parity):

| T (h) | K = 2 000 | 2 500 | 3 000 | 3 500 | 4 000 |
|---|---|---|---|---|---|
| 24 | −0.22 (−0.02 %) | −1.09 (−0.23 %) | −2.07 (−1.28 %) | −0.67 (−1.84 %) | −0.12 (−2.16 %) |
| 72 | −0.17 (−0.02 %) | −0.89 (−0.19 %) | −1.67 (−1.01 %) | −0.53 (−1.42 %) | −0.09 (−1.53 %) |
| 168 | −0.17 | −0.89 | −1.66 (−1.01 %) | −0.53 (−1.43 %) | −0.09 (−1.53 %) |
| 336 | −0.17 | −0.89 | −1.64 (−1.02 %) | −0.51 (−1.43 %) | −0.09 (−1.53 %) |

Channel (72 h): residual sd −3.16 TRY/MWh (529.5 → 526.3), P(stress at T)
+0.047 (0.594 → 0.641), mean P(stress) over [0, T] −0.0095, expected
switches 15.9 → 15.5. Decomposition at 72 h / 3 000: sample effect R1 − R0
−1.26; slope part R4 − R1 −3.90; intercept compensation R3 − R4 +2.23.

Sensitivities (72 h, K = 3 000; ramp effect 2D − 1D): base −1.67; ramp scaler
W_T −1.67; intercepts on the full sample −1.53; z re-standardized on W9
−1.54; alignment lag 0 h −0.20 and lag 2 h −3.15; constant z = 0 +2.04;
climatology −2σ +0.48, +2σ −4.13; observed initial hours −1.67 (only the
24 h contract moves, by −0.04). **The ramp effect is of the same order as the
±1 h alignment uncertainty and changes sign across scenarios.**

Numerics: PDE time-grid convergence at 72 h / 3 000 — R1 165.526 (144
steps) → 165.492 (1 440); R3 163.858 → 163.677. The measured ramp effect is
therefore −1.67 on the default grid and −1.82 on the converged grid; use
`grid.n_time_steps` ≥ 4 per hour for 2D-only work.

PDE vs MC (same path, same seed for all runs): largest |z| over the 20
contracts — 1D runs 2.3 raw / 1.7 after matching the MC mean of X_T to μ_X
(first-moment control variate); 2D primary R3 2.9 raw / 2.1 matched;
diagnostic R4 3.3 raw / 2.3 matched. The average call gap MC − PDE is −0.83 %
(R1) and −0.62 % (R3), so the ramp does not materially change the PDE–MC
relation; the slightly larger 2D extremes are consistent with the coarser
effective q path of the default PDE time grid (see convergence above). In this
run the sign of MC − PDE for calls changes with maturity (72 h below, 168/336 h
mostly above) and puts move opposite to calls — the signature of the Monte
Carlo error of the MEAN shared by all strikes (72 h R0: sample E[X_T] =
−5.54 ± 2.66, whereas the exact value is 0). This bears on the paper's
reading of a "systematic" MC shortfall (its Figure 3 uses exactly this
seed and step); the paper is not changed here. MC P(stress at T) agrees with
the ODE on the MC grid (R3: 0.6368 ± 0.0024 vs 0.6376); only at dt = 0.25 h
is the first-order regime-step bias visible (0.6326).

## 12. Remaining limitations

1. Ramp direction, lag, RD-vs-z construction, window and ddof are unverified.
2. z standardization is a TRY-window reconstruction; M9 may have used its own
   (USD-run) window (sensitivity above).
3. Intercepts are derived from duration targets of unknown sample; the p01
   target conflicts with M9's AME ratio.
4. M9 is a USD-price fit applied to a TRY pricing model; π0 comes from the M2
   filter.
5. The decisive test — regress logit(p) of the shipped M9 series on candidate
   (z_{t−1}, r_{t−1}), which would fix definition, scale and intercepts at
   once — needs `pde_timeseries.parquet`, absent here.
6. Prices are conditional on deterministic covariate paths.
7. No discrete-time mode for s ≥ 1 paths (pricing is refused instead).

## 13. Commands

    python scripts/tvtp_derivation/derive_tvtp2_parameters.py --write     # frozen 2D yaml + audit
    python run_pde.py --config config/forward_centered_tvtp2_experimental.yaml validate
    python run_pde.py --config config/forward_centered_tvtp2_experimental.yaml price --strike 3000 --maturity-hours 72
    python run_pde.py --config config/forward_centered_tvtp2_experimental.yaml calibrate-market
    python run_pde.py --config config/forward_centered_tvtp2_experimental.yaml diagnostics
    python scenario_sweep.py --config config/forward_centered_tvtp2_experimental.yaml
    python scripts/tvtp2/compare_tvtp_1d_2d.py                            # ~16 min on 2 cores
    python scripts/tvtp2/write_provenance_report.py
    python -m pytest tests/test_tvtp2_*.py
