# Project Status and Roadmap

Living document.  This is the **single source of truth** for both the
team and future sessions: check a box (`[ ]` → `[x]`) as each item lands,
and add commit hashes for completed work.  Faz 1 and Faz 2 are done;
Faz 3 (manuscript) is in progress; Faz 4 (advisor outreach) and Faz 5
(post-manuscript / advisor-directed work) are queued.

---

## Faz 1 — Foundational Correctness (COMPLETED)

Baseline pipeline, repo hygiene, and correctness fixes that make the
project reproducible on any developer machine without regressions.

- [x] **F1.1 Handoff baseline** — accepted forward-curve calibration
  (six VEP monthly quotes reproduced to solver precision; near-term
  anchor by `spot_to_next_linear`), 154 tests passing.  Commit
  `e019439`.
- [x] **F1.2 Repo reorganisation** — flat top-level layout, `.gitignore`
  added, source zip removed from tracking, docs consolidated under
  `docs/`.  Commit `aec36eb`.
- [x] **F1.3 Windows UTF-8 encoding fix + near-term anchor sensitivity
  methodology rewrite** — Turkish locale (cp1254) failures on
  `Path.read_text` / `Path.write_text` closed by explicit
  `encoding="utf-8"`; `near_term_anchor_sensitivity()` switched from a
  flat `explicit_level` sweep to the production `spot_to_next_linear`
  ramp so the table matches the anchor mode actually shipped.  Commit
  `72ceea2`.
- [x] **F1.4 Professional repo cleanup** — dead backup files removed
  (`forward_calibration_backup.py`, `forward_centered_before_*.py`),
  dev / derivation scripts moved to `scripts/dev_checks/` and
  `scripts/tvtp_derivation/`, `.gitignore` UTF-16 artefact corrected,
  `README.md` upgraded to professional standard, `requirements.txt`
  and `requirements-dev.txt` pinned to test-verified versions.  Commit
  `07c587a`.

---

## Faz 2 — Option Pricing Engine (COMPLETED)

TVTP integration, risk-neutral wiring, comprehensive sensitivity /
robustness analyses, realized-data backtest.  Everything the paper's
"Model" and "Results" sections need to reference exists on `main`
under `outputs/market_calibration_final/`.

### Core parameter integration

- [x] **F2.1 TVTP M9 integration** —
  `inputs/historical/m2_frozen_parameters.yaml` now carries the M9 fit
  under the yaml regime-label convention (`index 0 = normal`,
  `index 1 = stress`).  Sigmas and gammas are the raw M9 numbers; the
  missing `alpha01` / `alpha10` intercepts are DERIVED by root-finding
  on the reported M9 mean-duration diagnostics
  (`scripts/tvtp_derivation/derive_tvtp_parameters.py`; full write-up
  in `docs/tvtp_derivation_methodology.md`).
  `--pi-override {filtered,stationary}` added to the `price` CLI so the
  M2-sourced `pi_filtered` is switchable to the M9 stationary occupancy
  for short-horizon work.  Placeholder-keyword detector strengthened in
  `params_frozen.py`.  Model_limitations items (a)-(e) auto-generated
  thereafter.  Commit `59955ee`.
- [x] **F2.2 Sigma-dependent diagnostic regeneration + sensitivity
  z-path alignment + source-zip archival** — every artefact that
  depended on the OLD placeholder sigmas rebuilt under the M9 yaml
  (`outputs/forward_centered_diagnostics/`,
  `legacy_vs_forward_centered`, `inputs/legacy_reference/`).
  `near_term_anchor_sensitivity` now uses the same climatology z path
  as `run_pde.py price`, so the sensitivity base row reproduces the
  72 h benchmark (677.23 TRY/MWh) to solver precision.  Raw M9 zip
  archived at `inputs/historical/archive/calibration_bundle/`.  Commit
  `7986b44`.
- [x] **F2.3 `phi` reconciliation with the M9 CSV row** — the yaml
  carried `phi = 0.99961485` (kappa = 3.85e-4 /h, half-life ~1799 h)
  inherited from the metadata `physical_measure_parameters` fallback
  block (a different model, `M2_tvtp_TVTP-1`), while the shipped
  `parameter_estimates.csv` assigns both M9 and M8 the same
  `phi ≈ 0.999996` (kappa 4.11e-6/h, half-life ~19 years — essentially
  a random walk on the transformed variable).  Yaml updated to the M9
  CSV value; `kappa_per_hour` and `half_life_hours` re-derived; pre-fix
  yaml archived to
  `inputs/historical/archive/m2_frozen_parameters.PRE_PHI_FIX.yaml`.
  All sigma-dependent artefacts regenerated; M8-vs-M9 robustness gap
  preserved (~-0.60 % at every maturity), confirming the sigma-
  difference finding is kappa-independent.  Test thresholds updated
  for the new near-unit-root regime.

### Risk-neutral measure

- [x] **F2.4 Risk-neutral Q1 drift channel wired** — `ResidualSpec`
  grows an optional `drift_shift_per_hour`; the term enters the moment
  ODE (`+ a_i p_i` on `u_i'`, `+ 2 a_i u_i` on `w_i'`), the pricing PDE
  stencil (`+ a_i` on the drift row), and the MC simulator (effective
  mean `m_eff_i = m_i + a_i/κ`).  Symmetric wiring preserves
  `E^Q[P_t] = F(t)` exactly (verified by
  `test_q1_drift_shift_preserves_centering`; three parametrised
  `(a_0, a_1)` cases).  New CLI flags `--risk-premium-a0` and
  `--risk-premium-a1` on `price`, default `(0, 0)` = physical measure =
  every prior benchmark reproduced bit-for-bit.  Log-log diagnostic
  confirmed the O(a²) scaling of the variance channel to 4 significant
  digits.  Model_limitations item (f) auto-generated; full methodology
  in `docs/risk_neutral_methodology.md`.  Commit `53adf35`.

### Sensitivities, robustness and comparisons

- [x] **F2.5 Pooled-baseline vs M9 comparison** — 4-maturity call sweep
  showing regime-conditioning changes option prices by 38-49 % vs a
  single-volatility baseline; see
  `outputs/market_calibration_final/model_comparison_pooled_vs_M9.md`.
- [x] **F2.6 M8 vs M9 robustness check** — swapping to the top-2
  alternative model's sigmas moves the 24 / 72 / 168 / 336 h call by
  <1 % despite a large in-sample BIC gap; see
  `outputs/market_calibration_final/model_robustness_M8_vs_M9.md`.
- [x] **F2.7 Discount-rate (r_annual) sensitivity** — sweep at
  `r_annual ∈ {0.15, 0.25, 0.30, 0.40, 0.50, 0.60}` at 24 / 72 / 336 h
  shows <2 % impact on call value across the range; the flat-rate
  assumption is non-critical at target maturities.  See
  `outputs/market_calibration_final/discount_rate_sensitivity.md`.
- [x] **F2.8 Strike × maturity option-value grid** — 66-point PDE
  sweep (11 strikes × 6 maturities) + companion put pricings.  Two
  heatmaps produced (raw call value and moneyness-normalised) with the
  F(T) contour overlaid.  Put-call parity max error 1e-6 TRY/MWh.  See
  `outputs/market_calibration_final/strike_maturity_grid.md`.
- [x] **F2.9 Realized-PTF 2026 backtest** — hourly forward curve
  compared against realised EPİAŞ PTF over 2025-12-31 → 2026-07-31
  (5089 matched hours).  Overall MAE 1312 TRY/MWh, mean bias
  −1019 TRY/MWh; realised prices systematically 39 % below forward.
  Independently cross-checked against Turkish energy-sector press.
  See `outputs/market_calibration_final/realized_2026_backtest.md`.
- [x] **F2.10 Half-life reconciliation** — theoretical + numerical
  (200 kh simulation) analysis showing the "19 yr vs 8.84 h"
  discrepancy is a variable mismatch (raw `asinh(PTF)` vs
  deseasonalised residual), not a bug; identifies the model's
  inheritance of the raw-y kappa as the likely root cause of the
  7-13× variance overshoot found in F2.9.  See
  `outputs/market_calibration_final/half_life_reconciliation.md`.

### Documentation

- [x] **F2.11 Model limitations documentation** — items (a) through
  (h) auto-generated by `_limitations_markdown` on every
  `calibrate-market` run, covering: derived intercepts (a),
  omitted-variable risk (b), unverifiable `scale_P` (c), regime-label
  swap (d), M2-sourced `pi_filtered` (e), uncalibrated Q1 flags (f),
  half-life reconciliation status (g), and the `scale_P` /
  TRY-depreciation window mismatch (h).

---

## Faz 3 — Manuscript Writing (IN PROGRESS)

Goal: assemble a coherent draft manuscript from the Faz 2 analyses.
Every item below points to existing artefacts under `outputs/` or
`docs/`; the writing effort is mostly assembly + academic formatting.

- [ ] **W1 Literature review and positioning** — Deng (2000),
  Huisman & de Jong, Janczura & Weron, Benth et al.,
  Turkish / EPİAŞ-specific literature if any.  Identify the two or
  three sentences of positioning: what does this paper add over
  existing regime-switching electricity-option work?
- [ ] **W2 Manuscript outline** — section-by-section mapping of
  existing analyses to paper sections (Introduction, Model,
  Calibration, Results, Discussion, Limitations, Conclusion,
  Appendices).  One-page skeleton before drafting begins.
- [ ] **W3 Methodology section** — forward calibration (VEP
  interpolation + near-term anchor rule) + TVTP + forward-centering
  identity (`E^Q[P_t] = F(t)`) + Q1 risk-neutral wiring.  Source
  material: `docs/tvtp_derivation_methodology.md`,
  `docs/risk_neutral_methodology.md`, module docstrings.
- [ ] **W4 Results section** — benchmarks (F2.1-F2.4), sensitivities
  (F2.5-F2.7), backtest (F2.9), strike / maturity grid (F2.8) — figures
  need an academic-formatting pass (consistent axes, colourblind-safe
  palette, publication DPI).
- [ ] **W5 Limitations and Future Work section** — start from
  `outputs/market_calibration_final/model_limitations.md` items
  (a)-(h) and this document's Faz 5 items; edit down to the 3-5
  strongest.
- [ ] **W6 Introduction / Abstract / Conclusion, full draft assembly**
  — last piece written; anchors the narrative and sets the
  contribution claims.
- [ ] **W7 Team internal review** — full read-through by the team
  member (Onat) with a formal checklist before external outreach.

---

## Faz 4 — Advisor Outreach

- [ ] **D1 Publish preprint** — arXiv (q-fin.PR) or SSRN, with the
  reproducibility bundle (repo commit hash, `outputs/` frozen snapshot,
  `docs/*.md`).
- [ ] **D2 Targeted academic outreach with draft attached** — short
  list of researchers whose recent work overlaps (electricity option
  pricing, Markov-switching, Turkish market).
- [ ] **D3 If a response is received** — joint revision plan agreed
  in writing before further work; scope, authorship, and timelines
  fixed up front.
- [ ] **D4 If no response by an agreed deadline** — proceed
  independently to Faz 5 based on internal review and preprint
  feedback.

---

## Faz 5 — Future Work (post-manuscript or advisor-directed)

Prioritised by "likely required for a Q1-tier submission" (Category A)
→ "strengthens the case but not strictly required" (Category B) →
"nice to have, time-permitting" (Category C).

### Category A — likely required for Q1-level ambition

- [ ] **FW1 Kappa refit on (P−F) residuals directly** — the root cause
  identified in `half_life_reconciliation.md`.  Addresses the 7-13×
  variance overshoot found in the 2026 backtest (F2.9).  Expected
  refit value near `0.0784 /h` (half-life ~8.84 h).  The forward-curve
  identity `E^Q[P_t] = F(t)` is preserved regardless of kappa, so no
  VEP re-calibration is needed.  Companion sigma re-check on the same
  residual is warranted.

  **Status as of 2026-09-12 (post-pull audit):**  a candidate v2 stack
  is available in `pde_option_model/{hpfc, residual_v2, premium}.py`
  and `outputs/{hpfc, residual_v2}/*` — a merge from a teammate,
  reviewed under `outputs/market_calibration_final/teammate_change_review.md`.
  v2 fits an MS-AR(1) TVTP on the ratio `x = (P − M·S)/L` over
  2023-2025 (26 304 hours) and reports a 2026 out-of-sample coverage
  much closer to nominal than v1 (`cov50 = 34.3 %` overall vs v1's
  ~100 %; residual sd at 336 h ≈ 1 246 TRY/MWh vs v1's 3 803 — the
  ~3× reduction matches this doc's prediction).  v2 does NOT touch
  v1's production `inputs/historical/m2_frozen_parameters.yaml` or
  any `outputs/market_calibration_final/*` artefact; it lives as a
  **companion analysis**.  Promotion of v2 to production is a
  **separate decision** and must include:
    (i) an official-source citation for the residual_v2 price-cap
        schedule (currently [DATA-INFERRED] from realized 2026 PTF;
        see `residual_v2.price_limits` docstring);
    (ii) a v2 yaml artefact analogous to
        `m2_frozen_parameters.yaml` with full provenance;
    (iii) `test_forward_centered.py`-style invariant tests re-run
        against the new residual spec;
    (iv) an updated `model_limitations.md` covering v2's own
        assumptions (level proxy, cap schedule, MS-AR(1) SEs).
  Until those are done, v2 remains a Faz 5 candidate, not
  production.
- [ ] **FW2 Risk-neutral premium beyond zero** — either implement Q2
  (transition-intensity shift `η_ij`; skeleton already in
  `risk_neutral.py`), or adopt a literature-grounded risk-premium
  range for the drift channel to replace the current
  `(a_0, a_1) = (0, 0)` default.  Currently the model is "physical-
  measure priced" which is not defensible for options in general.
- [ ] **FW3 At least one benchmark model comparison** — a simple
  GBM / Black-76 baseline OR a known electricity-option approach from
  the literature (e.g. Deng or Huisman-de Jong), priced on the same
  contracts as F2.8's grid.  Referee-proof against "why not just X?".

### Category B — strengthens the Q1 case, not strictly required

- [ ] **FW4 Reconstruct `RD_Ramp_1h_lag1` as a genuine 2-covariate
  TVTP** — currently the ramp covariate is dropped (documented in
  `model_limitations.md` item (b)).  Adding it closes the omitted-
  variable disclosure and may materially change the TVTP-specific
  contribution documented in
  `model_comparison_pooled_vs_M9.md`.
- [ ] **FW5 Re-estimate `scale_P` on a real / inflation-deflated
  price series** — `model_limitations.md` item (h).
  Methodologically interesting given TRY's high-inflation context;
  could be framed as a broader contribution for emerging-market
  electricity price modelling.  Two alternatives: (i) real (deflated)
  series, (ii) rolling / shorter estimation window.
- [ ] **FW6 Extend the realized-PTF backtest as new months arrive** —
  the current backtest (F2.9) ends 2026-07-31.  As `realized_ptf_2026.csv`
  gets extended, re-running produces a longer out-of-sample record.

### Category C — nice to have, time-permitting

- [ ] **FW7 True TVTP-specific ablation** — if M8's own constant-
  transition coefficients are ever located (they are absent from the
  handoff bundle) or re-estimated on the shipped hourly series, run
  the ablation described in Faz 5's original item (constant-transition
  M8 vs TVTP M9, holding sigmas fixed).  Isolates the TVTP-specific
  contribution the F2.5 comparison cannot separate on its own.
- [ ] **FW8 Note (not implement) further stochastic extensions** —
  jump-diffusion, stochastic-volatility overlays, non-Gaussian
  innovations.  Suitable for the manuscript's "future work" paragraph
  rather than for actual implementation in this project's scope.

---

## Legacy notes retained for audit continuity

*(Items that were listed as future work at some point but have since
been resolved.  Kept for traceability.)*

- ~~**Reconcile yaml `phi` with the M9 CSV `phi`.**~~ Resolved in
  F2.3; the yaml value 0.99961485 has been replaced with the M9 CSV
  value 0.999995891734.  Pre-fix yaml archived.  See
  `docs/tvtp_derivation_methodology.md`.
- ~~**Legacy-explosion figure as motivation.**~~ Recorded as a paper
  writing task (Faz 3 W3 / W4); the number
  `exp(σ² / (4κ)) ≈ 4.66e+225` is the strongest available quantitative
  argument for the "why we chose forward-centering over the legacy
  transform" section and should be plotted from
  `outputs/forward_centered_diagnostics/legacy_explosion_table.csv`.
