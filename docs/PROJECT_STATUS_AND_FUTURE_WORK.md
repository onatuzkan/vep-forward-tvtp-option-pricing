# Project status and future work

Living document.  Add a bullet under **Completed** each time a milestone
lands on `main`; add candidates under **Future work** whenever a promising
extension is scoped but not yet built.

---

## Completed

1. **Handoff baseline** — accepted forward-curve calibration (six VEP monthly
   quotes exactly reproduced, near-term anchor by `spot_to_next_linear`),
   154 tests passing.  Commit `e019439`.

2. **Repo reorganisation** — flat top-level layout, `.gitignore` added,
   source zip removed from tracking, docs consolidated under `docs/`.
   Commit `aec36eb`.

3. **Windows UTF-8 encoding fix + near-term anchor sensitivity methodology
   rewrite** — Turkish locale (cp1254) failures on `Path.read_text` /
   `Path.write_text` closed by explicit `encoding="utf-8"`;
   `near_term_anchor_sensitivity()` switched from a flat `explicit_level`
   sweep to the production `spot_to_next_linear` ramp so the table matches
   the anchor mode actually shipped.  Commit `72ceea2`.

4. **TVTP M9 integration** — `inputs/historical/m2_frozen_parameters.yaml`
   now carries the M9 fit under the yaml regime-label convention
   (`index 0 = normal`, `index 1 = stress`).  Sigmas and gammas are the raw
   M9 numbers; the missing `alpha01` / `alpha10` intercepts are DERIVED by
   root-finding on the reported M9 mean-duration diagnostics
   (`scripts/tvtp_derivation/derive_tvtp_parameters.py`; full write-up in
   `docs/tvtp_derivation_methodology.md`).  `--pi-override
   {filtered,stationary}` added to the `price` CLI so the M2-sourced
   `pi_filtered` is switchable to the M9 stationary occupancy for short-
   horizon work.  Placeholder-keyword detector strengthened in
   `params_frozen.py`.  Model_limitations.md items (a)-(e) auto-generated
   thereafter.  Commit `59955ee`.

5. **Sigma-dependent diagnostic regeneration + sensitivity z-path
   alignment + source-zip archival** — every artefact that depended on
   the OLD placeholder sigmas rebuilt under the M9 yaml
   (`outputs/forward_centered_diagnostics/`, `legacy_vs_forward_centered`,
   `inputs/legacy_reference/`).  `near_term_anchor_sensitivity` now uses
   the same climatology z path as `run_pde.py price`, so the sensitivity
   base row reproduces the 72h benchmark (677.23 TRY/MWh) to solver
   precision.  Raw M9 zip archived at
   `inputs/historical/archive/calibration_bundle/`.  Commit `7986b44`.

6. **Professional repo cleanup** — dead backup files removed
   (`forward_calibration_backup.py`, `forward_centered_before_*.py`),
   dev/derivation scripts moved to `scripts/dev_checks/` and
   `scripts/tvtp_derivation/`, `.gitignore` UTF-16 artefact corrected,
   `README.md` upgraded to professional standard, `requirements.txt`
   and `requirements-dev.txt` pinned to test-verified versions.
   Commit `07c587a`.

7. **Risk-neutral Q1 drift channel wired** — `ResidualSpec` grows an
   optional `drift_shift_per_hour`; the term enters the moment ODE
   (`+ a_i p_i` on `u_i'`, `+ 2 a_i u_i` on `w_i'`), the pricing PDE
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

8. **`phi` reconciliation with the M9 CSV row.**  Surfaced during the
   M8-vs-M9 robustness check: the yaml carried `phi = 0.99961485`
   (kappa = 3.85e-4 /h, half-life ~1799 h) inherited from the metadata
   `physical_measure_parameters` fallback block (a different model,
   `M2_tvtp_TVTP-1`), while the shipped `parameter_estimates.csv`
   assigns both M9 and M8 the same `phi ≈ 0.999996` (kappa 4.11e-6/h,
   half-life ~19 years — essentially a random walk on the transformed
   variable).  Yaml updated to the M9 CSV value; `kappa_per_hour` and
   `half_life_hours` re-derived; pre-fix yaml archived to
   `inputs/historical/archive/m2_frozen_parameters.PRE_PHI_FIX.yaml`.
   Test thresholds updated for the new stationary quantities: the
   normal-regime stationary inflation factor is now O(1) instead of
   ~1.04, the stress-regime factor is astronomical (~1e225), the OU
   half-life invariant expects ~168720 h instead of ~1800 h.  All
   sigma-dependent artefacts (calibration outputs, scenario sweep,
   risk-premium sensitivity, pooled-vs-M9 comparison, robustness
   comparison, diagnostics, legacy reference) regenerated under the
   reconciled kappa; the M8-vs-M9 robustness gap is preserved
   (~-0.60 % at every maturity, up from -0.62 % pre-fix), confirming
   the sigma-difference finding is kappa-independent.

---

## Future work

1. **TVTP-specific ablation.**  The current pooled-baseline-vs-M9
   comparison (`outputs/market_calibration_final/model_comparison_pooled_vs_M9.md`)
   demonstrates that *regime-conditioning* materially changes option
   prices, but cannot separate the contribution of *time-varying*
   transition probabilities from that of a plain two-regime
   *constant-transition* alternative.  Isolating the TVTP-specific
   contribution requires M0's fitted alpha / gamma (constant-transition
   logistic coefficients), which are **not present in this handoff
   bundle** — `inputs/historical/archive/calibration_bundle/transition_coefficients.csv`
   contains only M9 rows.  When M0's alpha / gamma become available
   (either recovered from the estimation team or re-estimated on the
   shipped hourly series), run the same 24 / 72 / 168 / 336 h call
   sweep with (i) constant-transition M0 and (ii) M9, holding sigma
   values and forward curve identical, so the pure "does TVTP add value
   over constant-transition" contribution can be quantified.

2. ~~**Reconcile yaml `phi` with the M9 CSV `phi`.**~~
   **RESOLVED** in the phi-fix commit; see Completed item 8.  Note
   retained here in strikethrough for audit continuity:
   the yaml `phi` was inherited from the `physical_measure_parameters`
   fallback block — which describes a *different* model
   (`M2_tvtp_TVTP-1`) than any row in the CSV.  The sigmas were
   updated to M9's CSV values during the integration commit
   (`59955ee`) while `phi` was silently left at the fallback value.
   This was a **first-order** inconsistency: substituting M9's CSV
   `phi` into the same pipeline moves the 336 h call by ~6 %, which
   is 10× the M8-vs-M9 sigma-difference effect.  Action item: decide
   which `phi` represents the physical dynamics of the transformed
   variable `y = asinh(P / scale_P)` (there may be a delta-method /
   preprocessing reason the fallback block used a different value),
   commit the reconciled value into the yaml with provenance, and
   re-run acceptance + benchmark suite.  A companion inspection of
   whether the acceptance thresholds themselves need updating (they
   were calibrated against the current yaml `phi` regime) should
   accompany the change.

3. **Legacy-explosion figure as motivation for the paper's "why
   forward-centering" section.**  Under the reconciled near-unit-root
   kappa (4.11e-6/h), the legacy sinh-Gaussian model's stress-regime
   stationary inflation factor is `exp(sigma_stress² / (4 kappa)) ≈
   4.66e+225` — up from ~255 under the pre-fix kappa.  This is *not
   a bug*: it is the same `exp(v(t)/2)` pathology of the legacy asinh-
   OU model, just more sharply exposed once the near-unit-root
   dynamics are correctly represented.  For the paper's "why we chose
   forward-centering over the legacy transform" section, the
   astronomical figure is a **far stronger quantitative argument** than
   the pre-fix 255× number: it directly demonstrates that the legacy
   model's long-horizon expected price is not merely "inflated" but
   *catastrophically undefined* at pricing-relevant horizons.  Action
   item: promote this figure (with the derivation
   `exp(sigma²/(4 kappa))`, the sigma_stress = 0.0924 M9 value, and
   the reconciled kappa) into the "Model choice" or "Motivation"
   section of the manuscript, with a plot from
   `outputs/forward_centered_diagnostics/legacy_explosion_table.csv`.

4. **Real / inflation-deflated `scale_P` re-estimation.**  Candidate
   for Faz 3.  The current `scale_P = 282.48` is the training-window
   median absolute PTF fit on a 9-year (2016 - 2024/25) window that
   spans severe TRY depreciation, so it may mis-scale the
   `asinh(PTF / scale_P)` transform relative to the end-2025
   valuation regime (see `model_limitations.md` item (h)).  Two
   alternatives worth exploring: (i) re-estimate on a real
   (inflation-deflated) price series so `scale_P` represents a
   stationary purchasing-power unit; (ii) use a rolling / shorter
   estimation window aligned with the valuation date.  Methodologically
   interesting given TRY's high-inflation context; could be framed as
   a broader methodological contribution for emerging-market
   electricity price modelling generally, not just this dataset.

5. **Residual-around-F kappa refit — Faz 3 candidate, likely root cause
   of the 7-13× variance overshoot found in the 2026 backtest.**
   `outputs/market_calibration_final/half_life_reconciliation.md`
   resolves the ~19 000× "half-life discrepancy" flagged in item (g)
   of `model_limitations.md`.  Both fitted phi values (within-regime
   0.999996 for M9, and deseasonalized-single-regime 0.9246 in the
   summary) describe different variables (raw `asinh(PTF)` vs
   deseasonalized residual) and are individually correct.  However,
   the pricing model currently uses the raw-y within-regime kappa
   (`4.11 e-6 /h`, near random walk) as the OU mean-reversion rate of
   the residual X = P − F, which produces a residual variance that
   grows almost linearly in time — inconsistent with the empirical
   near-saturation of the (P − F) residual std at ~1 000 TRY/MWh
   observed in the 2026 backtest.  Numerical simulation of the M9
   process confirms marginal ACF ≈ `phi^h` (unit-root-like), so this
   is a mis-inheritance in the yaml, not a code bug in the moment ODE.
   Fix: refit `kappa_per_hour` directly on the (P − F) residual over
   a training window (expected value near `0.0784 /h`, half-life
   ~8.84 h); the forward-curve identity `E^Q[P_t] = F(t)` is preserved
   regardless of kappa, so this does not require re-doing the VEP
   calibration.  Companion sigma re-check on the same residual is
   also warranted.
