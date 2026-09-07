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

2. **Reconcile yaml `phi` with the M9 CSV `phi`.**  Surfaced during the
   M8-vs-M9 robustness check
   (`outputs/market_calibration_final/model_robustness_M8_vs_M9.md`):
   the current `inputs/historical/m2_frozen_parameters.yaml` carries
   `phi = 0.99961485` (kappa = 3.85e-4 /h, half-life ≈ 1799 h), while
   the shipped `parameter_estimates.csv` reports **both** M9 and M8
   with `phi ≈ 0.999996` (kappa ≈ 4.11e-6 /h, near unit-root, half-
   life ≈ 168 000 h).  Investigation of the metadata JSON shows the
   yaml `phi` was inherited from the `physical_measure_parameters`
   fallback block — which describes a *different* model
   (`M2_tvtp_TVTP-1`) than any row in the CSV.  The sigmas were
   updated to M9's CSV values during the integration commit
   (`59955ee`) while `phi` was silently left at the fallback value.
   This is a **first-order** inconsistency: substituting M9's CSV
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
