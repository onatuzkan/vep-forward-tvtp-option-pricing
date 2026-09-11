# Kullanım notu (Türkçe)

Bu promptu tek parça olarak ChatGPT'ye yapıştır. Aynı mesaja depoyu (ZIP) ve `RobustnessPaper.pdf` dosyasını ekle. Denetim raporunu (`Denetim_Raporu_VEP_TVTP_ve_Bayesci_Gelistirme.md`) da eklersen model bulguları kendi başına doğrulayabilir.

"Astra 6" modelinin özelliklerini bilmiyorum, bu yüzden prompt modelden bağımsız yazıldı: kod çalıştırabilen (Python sandbox) herhangi bir güçlü model ile çalışır. Model kod çalıştıramıyorsa "EXECUTION MODE" bölümündeki B seçeneği devreye girer.

Prompt İngilizce yazıldı çünkü depo, kod ve dokümanlar İngilizce ve teknik terimler bu dilde daha kesin. Model sana raporlarını Türkçe verecek.

---

# PROMPT — copy everything below this line

## 0. ROLE

You are a senior quantitative researcher and Python engineer working in energy derivatives. Your background covers stochastic calculus, regime-switching models, PDE numerics (Crank–Nicolson on coupled systems), Bayesian computation (MCMC, SMC, importance sampling, convergence diagnostics) and electricity-market microstructure (Turkish day-ahead market: EPİAŞ PTF, VEP forwards, VİOP electricity futures, EPDK price caps).

You are joining a university research team (İTÜ) that has a working, well-tested Python repository. Your job is to:

- (A) fix the model's structural errors, which have been measured against real data,
- (B) add a Bayesian robust-calibration layer adapted from Gupta & Reisinger (2012), *Robust Calibration of Financial Models Using Bayesian Estimators* (PDF attached),
- (C) deliver a publication-grade out-of-sample evaluation.

You deliver **complete, runnable files** (never fragments or "…rest unchanged"), tests for every new feature, and honest reports of what does and does not work.

**Language:** code, docstrings and commit messages in English. At the end of each phase, give me a **Turkish** summary covering what was done, which numbers changed, what is still open, and what you need from me.

---

## 1. PROJECT CONTEXT (read before touching anything)

Repository: `vep-forward-tvtp-option-pricing-main` (attached ZIP). Key facts:

- **Product:** a European option on the *expiry-hour* Turkish day-ahead spot price (PTF, TRY/MWh), valued 2025-12-31 20:00 UTC (= 23:00 TR). Spot at valuation: 2917.78.
- **Price level:** anchored to the EPİAŞ VEP monthly baseload forward strip of 2025-12-31 (EBM0226…EBM0726). The strip is fitted as hard KKT equality constraints by a smoothness-regularised hourly curve (`forward_curve.py`, `curve_mode: smooth_constrained`, `smoothness_weight=1.0`, `level_weight=1e-4`). January 2026 is unquoted and is filled by the `spot_to_next_linear` rule.
- **Residual:** X_t = P_t − F(t) follows a 2-regime TVTP Markov-switching OU (`forward_centered.py`):
  `dX = [κ(m_J − X) + a_J]dt + σ^X_J(t) dW`,
  where `σ^X_i(t) = σ^y_i · sqrt(F(t)² + scale_P²)`, `scale_P = 282.48`, and
  `p01 = logistic(α01 + γ01 z_{t−1})`, `p10 = logistic(α10 + γ10 z_{t−1})`, with z = standardised residual demand (RD = load − wind − solar), lagged 1 h.
  A centering ODE guarantees E[P_t] = F(t).
- **Frozen parameters** (`inputs/historical/m2_frozen_parameters.yaml`):
  - σ^y = (0.0035348, 0.0924067) per √h
  - φ = 0.999995892, i.e. κ = 4.11e-6 /h (half-life ≈ 19 y)
  - γ01 = −0.5838, γ10 = 0.0777, taken from M9
  - α01 = −1.0157, α10 = −1.8952: **derived** from duration diagnostics, not estimated
  - π_filtered = (0.932, 0.068), from a different model (M2)
- **Numerics:** coupled 2-regime Crank–Nicolson PDE (`solver.py`), moment ODE, and a Monte Carlo cross-check. `python run_pde.py validate` passes 15/15; the pytest suite reportedly has 169–172 passing tests. The CLI has the modes `validate | calibrate-market | price | diagnostics | freeze-params`, plus `scenario_sweep.py`.
- **Documentation to read first:** `README.md`, `docs/PROJECT_STATUS_AND_FUTURE_WORK.md` (living roadmap — update it), `docs/tvtp_derivation_methodology.md`, `docs/risk_neutral_methodology.md`, `outputs/market_calibration_final/{model_limitations.md, realized_2026_backtest.md, half_life_reconciliation.md, strike_maturity_grid.md}`, and `pde_option_model/markov_adapter.py` (module docstring).
- **Realized data:** `inputs/market/realized_ptf_2026.csv` holds hourly PTF from 31.12.2025 to 31.08.2026. The file is semicolon-separated, uses Turkish number format ("2.799,98"), local Europe/Istanbul time (UTC+3, no DST), and has TL/USD/EUR columns.

---

## 2. VERIFIED AUDIT FINDINGS (your starting point — re-verify each one yourself first)

An independent audit recomputed the following from the repository's own files. **Step 0 of your work is to reproduce every number below with your own script (`scripts/audit/reproduce_audit.py`) and report any discrepancy.**

**F1 — Price cap violated (critical).**
- Realized PTF is capped at **3400 TRY/MWh through 2026-03** (225 January hours sit exactly at 3400) and at **4500 from 2026-04** (EPDK decision).
- The floor is **0**: May 2026 had 214 hours at 0.
- All grid maturities (24–720 h) fall in January, so every call must satisfy C ≤ e^{−rτ}(3400 − K)⁺.
- In `strike_maturity_grid.csv`, **54 of 66 calls violate this bound**. Example: K=3000, 72 h → model 687.04 vs bound 398.7. K ≥ 3400 calls should be 0 but are priced 313–2023.
- The model assigns P(P<0) ≈ 5.5% at 72 h (MC) and ≈ 37% averaged over the backtest horizon (Gaussian approximation).

**F2 — Residual dispersion 5–10× too wide (critical).**
- Across 5088 hours (Jan–Jul 2026), the realized-hour coverage of the model's central intervals is 99.7% for the 50% band, 99.9% for the 90% band and 100% for the 95% band.
- The ratio model sd / RMSE (RMSE includes the monthly bias) runs 5.3–10 by month. Mean Gaussian CRPS is 2401 TRY/MWh.
- Empirical persistence:

  | Series | lag-1 AR | Half-life |
  |---|---:|---:|
  | Realized P−F | 0.882 | 5.5 h |
  | P−F minus month×hour profile | 0.841 | 4.0 h |
  | Daily-mean residual | 0.782 | 2.8 days |

  ACF(24 h) = 0.37, ACF(168 h) = 0.11. The model instead uses a half-life of 168 720 h.
- Root cause: κ and σ were inherited from an AR fit on *raw hourly asinh prices*, so deterministic intraday swings are accumulated as random-walk shocks.

**F3 — No hourly price-forward shape (high).**
- Decomposition of Var(P−F): about 22% is the monthly level miss (VEP's own forecast error), about **40% is a deterministic month×hour-of-day profile**, and about 38% is stochastic (sd ≈ 740 TRY/MWh).
- The forward curve is smooth within each month, so options expiring at 13:00 and at 20:00 receive the same value.

**F4 — Currency/window mixing (high).**
- M9 comes from `outputs/markov_usd_final`. Its `val_mae_price / val_mae_y ≈ 7.12 / 0.090 ≈ 79` implies a price level around 70–80, which is consistent with **USD/MWh**.
- `markov_adapter.py` states that this USD run is "validation only". F2.1 nevertheless moved pricing to M9, with the TRY `scale_P = 282.48` and a TRY-window z-standardiser (≤ 2022-12-31, 61 361 rows). M9 itself was trained on about 78 905 hours.

**F5 — M9 fails its own diagnostics (medium).** val PIT-KS p = 5.9e-17, Ljung-Box p = 0, ARCH-LM p = 0, generalised residual std = 10.7. The "stress" regime has 67% stationary occupancy, so it is really an intraday-volatility state rather than market stress.

**F6 — The Q1 risk-premium channel is inert (medium).** Because of forward-centering, a drift shift a_i changes prices only at O(a²): a₁ = −0.5 moves the 72 h call by −0.0004%. Any meaningful measure change must work through volatility or regime intensities (Q2: η_ij).

**F7 — Documentation drift (low).** README, PROJECT_STATUS and calibration_audit §6 quote the 72 h benchmark as 677.23, but the current code gives **687.04**.

**What is correct and must be preserved:** the exact VEP fit (error ~1e-12), E[P_t] = F(t), put–call parity (1e-6), PDE vs MC agreement, the regime-label swap logic, the Turkish calendar handling (672 h in Feb 2026, UTC+3), the provenance labelling, and the legacy mode for benchmarking.

---

## 3. OBJECTIVES AND SCOPE

Build **v2 of the pricer** as a *new model mode* (`forward_hpfc_bayes`) alongside the existing `forward_centered` mode, which stays untouched and becomes the "v1 / MAP-baseline". Every v1 benchmark must still reproduce bit-for-bit, and all existing tests must keep passing.

Work in four phases, **in order**. Do not start Phase B until Phase A passes its acceptance criteria. Stop at the end of each phase, report, and wait for my "devam" (continue).

---

## 4. PHASE A — STRUCTURAL CORRECTIONS

### A0. Reproduce the audit
- Write `scripts/audit/reproduce_audit.py`. It must print every number in §2 and write `outputs/audit/audit_reproduction.md`.
- Fix F7 in the docs.

### A1. Regulatory price bounds
- New module `pde_option_model/price_bounds.py`: a dated cap/floor schedule loaded from `inputs/market/price_limits.csv`, with columns `valid_from_local, cap_TRY_MWh, floor_TRY_MWh, source`.
  - Seed rows: `2025-01-01: 3400/0`, `2026-04-01: 4500/0`.
  - Verify the exact 2026 switch date from the realized data (first hour > 3400) and mark it `[DATA-INFERRED]` until I confirm it with an EPDK reference.
- Choose a bounded price representation and justify it in `docs/bounded_price_methodology.md`. Implement **(i)** as the default and **(ii)** as a benchmark:
  - (i) keep the Gaussian residual but price the **capped/floored payoff** h(min(max(P,floor),cap)). The forward identity becomes E[clip(P)] = F, so re-solve the centering by root-finding on a per-hour shift of the residual mean. The monthly VEP averages must then hold for the *clipped* expectation.
  - (ii) a bounded transform, e.g. P = floor + (cap − floor)·Φ(Y) or a logit link, with Y following the regime-switching OU; the centering is solved numerically per hour.
- Add no-arbitrage tests:
  - 0 ≤ C ≤ DF·(cap − K)⁺ and 0 ≤ Put ≤ DF·(K − floor)⁺;
  - calls monotone decreasing and convex in K;
  - put–call parity holds with the clipped forward.

### A2. Hourly price forward curve (HPFC)
- `pde_option_model/hpfc.py`: F(t) = M(month(t)) · S(hour, daytype, month), with Σ_{h∈month} S = N_hours so that the monthly baseload VEP averages are preserved exactly (keep the current KKT smoothness solver for M).
- Estimate S from historical hourly PTF (§7 data request) using normalised profiles, with regularisation across months (a Sobolev-type penalty; see Phase B).
- Include Turkish holidays (`calendar_tr.py`) and weekday/Saturday/Sunday types. Solar-driven midday dips must shift with season.
- Out-of-sample check: on 2026 data, the share of Var(P − F_HPFC) explained by month×hour should drop from about 40% to under 10%.

### A3. Residual dynamics re-estimated on the right variable
- Target variable: e_t = P_t − F_HPFC(t), built with a forward curve that only uses information available before t (rolling historical HPFC, **no look-ahead**).
- Model: **two-factor regime-switching**, e_t = X^fast_t + X^slow_t, where
  - X^fast has κ_f ~ O(0.1–0.3)/h and regime-dependent σ_f,J;
  - X^slow has κ_s ~ O(0.005–0.02)/h and σ_s;
  - optionally, a spike/zero-price jump component or a regime-dependent mean for oversupply hours.
  - Regimes are TVTP with **both** covariates (RD_lag1, RD_Ramp_1h_lag1), fixing the omitted-variable issue.
- Estimate by exact Hamilton filter MLE on a state-space form, using a Kim filter / collapsing approximation for the two-factor + regime case. Estimate **α and γ jointly** (no derived intercepts) and report standard errors from the observed Fisher information.
- Validate on a hold-out: PIT uniformity, Ljung-Box on standardised residuals, coverage at 50/80/90/95%, and CRPS.
- Update the moment ODE, the PDE (now 2D in (x_fast, x_slow) per regime — use ADI/Douglas–Rachford, or reduce by noting that X^slow is Gaussian given the regime path and integrate it analytically) and the MC simulator. Keep the E[·] = F identity under bounds (A1).

### A4. Currency and window consistency
- One currency: TRY nominal, plus an optional real TRY variant deflated with TÜFE/ÜFE.
- One estimation window, with the z-standardiser rebuilt on the same window.
- Retire M9-USD from pricing and keep it only as a comparison row. Resolve `scale_P`, or drop the asinh transform if A1(i) makes it unnecessary.

### Acceptance criteria for Phase A (all must hold)
1. Zero no-arbitrage bound violations on the full 66-point grid and on a denser 21×10 grid.
2. Rolling daily out-of-sample backtest over Jan–Aug 2026 (re-value each day at 23:00 TR using only the information available then):
   - central 50% interval coverage within [40%, 60%];
   - 90% interval coverage within [83%, 95%];
   - sd/RMSE ratio within [0.7, 1.5].
   Report the numbers by month and by hour of day.
3. CRPS at least 50% lower than v1.
4. VEP monthly fit error ≤ 0.10 TRY/MWh.
5. All old tests pass, and at least 25 new tests cover bounds, HPFC, estimation, the PDE/MC cross-check (|z| < 3) and the look-ahead guard.

---

## 5. PHASE B — BAYESIAN ROBUST CALIBRATION (Gupta & Reisinger 2012, adapted)

**Read the paper in full first.** Then write `docs/bayesian_calibration_methodology.md`, which maps every paper element to our setting. At minimum it must cover:

| Paper | Our adaptation |
|---|---|
| θ = log local-vol on spline nodes (§4.1) | θ = (log σ_f,i, log σ_s, log κ_f, log κ_s, α_ij, γ_ij (2 covariates), HPFC shape coefficients on a (hour × month) log-spline grid, spike params, **risk-premia η01, η10, σ-premium ψ**) |
| Gaussian prior with H¹/Sobolev norm, eq. (4)–(5): ‖u‖²_κ = (1−κ)‖u‖²₀ + κ‖∇u‖²₀ | The same norm on the function-valued parts: HPFC shape S(h, m) and optionally σ_f(h, m) as a "local-vol surface" analogue. Weakly informative Gaussian priors on transformed scalars (half-life priors: fast 1–24 h, slow 1–14 days). Literature-based priors on the risk premia, which are **not identified** by data and must remain prior-dominated — show this explicitly. |
| Likelihood from option bid-ask, eq. (6)–(9), truncated at δ | **Two blocks:** (1) *Market block:* each VEP/VİOP forward quote i with tolerance δ_i taken from the bid-ask spread (VİOP) or the day's min–max / weighted-average band (VEP). Use a truncated Gaussian on the basis-point error, replacing today's hard equality. This makes January (unquoted) and intra-month shape uncertain in a controlled way. (2) *Historical block:* the Hamilton-filter log-likelihood of e_t from Phase A, **tempered** by β ∈ (0,1] so it does not swamp the market block. Report the sensitivity to β. |
| Posterior, eq. (10); MAP ≡ Tikhonov, eq. (11) | Show analytically that the current `smooth_constrained` curve is the MAP/Tikhonov limit (δ → 0, point-mass residual parameters). Keep MAP as a benchmark estimator. |
| Metropolis RW with prior-covariance proposal θ' = θ + √(2du)·Bξ, A = BBᵀ; ~23% acceptance; m overdispersed chains; burn-in b; thinning k (§4.2) | Implement exactly this as the baseline sampler (`pde_option_model/bayes/mcmc.py`). Also implement **pCN** (preconditioned Crank–Nicolson), which is dimension-robust for the spline parts, and an **adaptive Metropolis** variant. Defaults: m = 16 chains, overdispersed starts from prior draws, adaptive du tuned during burn-in only. |
| PSRF / Gelman–Rubin, eq. (13) | Compute split-R̂ (Vehtari et al. 2021) and ESS for every scalar parameter **and** for key option prices. Acceptance: R̂ < 1.05 (paper uses 1.1) and bulk-ESS > 400. |
| Bayes price = posterior mean, eq. (15); MAP price; posterior price pdf (§6.1) | For every contract, report the Bayes price, the MAP price, the posterior sd, the 68%/95% credible intervals and the posterior pdf plot. Define a **model-uncertainty bid/ask** as the 2.5%/97.5% posterior quantiles. |
| Robustness tests §6.2–6.4 | Replicate the paper's figures for our case: (a) number/placement of spline knots, (b) prior norm weight κ ∈ {10^−2 … 10^−0.1}, (c) noise δ scaled ×{0.5, 1, 2, 4}, (d) number of calibration instruments (drop VEP months one by one), (e) β tempering. Test whether the paper's main claim holds here, i.e. whether the Bayes price is more stable than MAP. Report honestly if it does not. |
| Recalibration / consistency via importance sampling (§5.3) | `bayes/sequential.py`: each new trading day (new VEP strip + realized PTF), re-weight the existing posterior samples with the new likelihood, track ESS, and resample-move (SMC with MCMC rejuvenation) when ESS < N/2. Show the posterior evolution over Jan–Aug 2026, analogous to the paper's Figure 3. |
| Exotic contracts where MAP fails (§6, barrier / American) | Our analogues: (i) capped hourly call/put, (ii) **monthly baseload and peak Asian (average-price) options**, (iii) a strip of daily options, (iv) a simple swing/take-or-pay option (optional; LSMC). Compare MAP and Bayes prices on these. |

**Computational design (mandatory):**
- Never solve the PDE inside the MCMC likelihood. The likelihood uses the Kalman/Hamilton filter and closed-form forward-curve algebra.
- The PDE and MC are used only to price N_post ∈ [500, 2000] thinned posterior draws, in parallel (`joblib`) and with caching.
- Report wall-clock time. Target: full posterior plus pricing of the grid in under 2 h on a laptop (8 cores).

**Phase B acceptance:**
- R̂ and ESS criteria met.
- The prior-vs-posterior plot shows which parameters the data identify.
- The Bayes vs MAP robustness figures are produced.
- The credible intervals of the Bayes price achieve nominal coverage in a **synthetic-truth experiment**: simulate from known θ*, calibrate, and check that the true price falls within the 95% CI in about 95% of 100 replications, as in paper §6.4.

---

## 6. PHASE C — EVALUATION, BENCHMARKS, MANUSCRIPT ASSETS

1. **Rolling out-of-sample backtest**, Jan–Aug 2026 plus whatever new data I provide, for three models:
   - v1 (MAP, current);
   - v2-MAP;
   - v2-Bayes.

   Metrics:
   - coverage, PIT histogram, CRPS, pinball loss at 5/25/50/75/95%;
   - Diebold–Mariano test on CRPS;
   - **option-payoff backtest:** for daily-rolled 24/72/168 h ATM-forward and ±10% capped calls, compare the average realized discounted payoff with the average model price, with a t-test and a block bootstrap.
2. **Benchmarks** priced on the same contracts: Black-76 on F with a constant historical vol, Lucia–Schwartz (2002) two-factor, and a single-regime version of v2.
3. **Risk-premium discussion:** use F6 to show why the Q1 drift channel is inert under forward-centering. Implement Q2 (η_ij intensity shifts; skeleton in `risk_neutral.py`) and a σ-premium ψ. Present prices as prior-dominated Bayesian bands. Never call them market-calibrated.
4. **Figures** at publication quality (colourblind-safe, 300 dpi, consistent axes), and tables in both CSV and Markdown.
5. **Update** `docs/PROJECT_STATUS_AND_FUTURE_WORK.md`: tick completed items and add new ones. Also update `model_limitations.md` generation (`_limitations_markdown`) and the README.

---

## 7. DATA I WILL PROVIDE (ask if anything is missing — do not invent data)

- Hourly PTF in TRY (and USD) for 2016-01-01 → latest, from the EPİAŞ Şeffaflık Platformu (same CSV format as `realized_ptf_2026.csv`).
- Hourly load, wind and solar (for RD and RD_Ramp) for the same span.
- VEP daily reference prices plus min/max/volume per contract for as many valuation dates as available. VİOP electricity futures settlement and bid/ask if available.
- The EPDK price-cap history.
- TÜFE/ÜFE for the real-price variant.

If a file is missing, build the code with a clear loader interface plus a synthetic-data fixture, mark results `[SYNTHETIC]`, and list precisely what you need. **Never fabricate market numbers.**

---

## 8. ENGINEERING RULES

- Python 3.11+. Pinned deps: numpy, pandas, scipy, matplotlib, PyYAML, pytest. You may add statsmodels, joblib and arviz (for diagnostics only). Keep everything CPU-only with no GPU requirement.
- Put new code under `pde_option_model/` (subpackage `bayes/`). Keep new configs under `config/` (`forward_hpfc_bayes_config.yaml`) with the same provenance tags: `[CALIBRATED] [INHERITED] [ASSUMED] [UNIDENTIFIED] [DATA-INFERRED] [SYNTHETIC]`.
- Every output keeps an honest label. v2 prices are labelled "VEP-anchored, HPFC-shaped, Bayesian model-uncertainty priced", never "market-calibrated".
- Handle Turkish locale explicitly with `encoding="utf-8"` and Turkish number parsing, and treat time zones explicitly (UTC internally, Europe/Istanbul for delivery).
- Seed all randomness and log seeds. Every CLI command must be reproducible end-to-end. Add CLI subcommands `estimate-residual`, `calibrate-bayes`, `price-bayes` and `backtest-rolling`.
- Unit tests plus property tests for no-arbitrage, including a look-ahead-bias test: shuffling future data must not change a past valuation.
- **Deliver complete files.** When you modify a file, output the full new file. After each phase, provide a single ZIP of the updated repository plus a `CHANGELOG.md` entry.

---

## 9. EXECUTION MODE

- **A (preferred) — you can run code:** actually execute the scripts and tests, show real output, and iterate until the acceptance criteria hold. Never report a number you did not compute.
- **B — you cannot run code:** say so explicitly at the start. Write the code plus exact commands for me to run, and give an expected-output checklist. Mark every number you state as `[EXPECTED, NOT RUN]`.

---

## 10. REPORTING FORMAT (end of each phase, in Turkish)

1. **Yapılanlar:** files created and changed, one line each.
2. **Sayılar:** a before → after table of key metrics (coverage, CRPS, bound violations, 72 h K=3000 call, runtime).
3. **Kabul kriterleri:** each criterion marked ✅ or ❌, with its value.
4. **Açık riskler ve varsayımlar:** anything `[ASSUMED]` or `[DATA-INFERRED]`.
5. **Benden istenenler:** data and decisions you need.

Be critical. If the paper's method brings no improvement on some dimension, show that with numbers and say it plainly. If you find an error in my audit (§2), report it with evidence.

Begin with Phase A0.
