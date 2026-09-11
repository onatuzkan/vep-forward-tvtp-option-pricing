# CLAUDE.md — VEP-forward TVTP option pricing (İTÜ research project)

Project memory for Claude Code.  Read this before doing anything.

## Working rules (from the project owner, Koray)

- **No synthetic data, anywhere** — not in code, tests, demos or examples.  Use only
  real EPİAŞ / VEP data.  If you find synthetic data, remove it.  Monte Carlo paths of
  the model itself and closed-form identity checks are allowed (they are not data).
- Work **step by step**: propose one change or command block, run it, show the real
  output, then continue.  Report limitations honestly; never state a number you did
  not compute.
- Deliver **complete files**, never fragments.  Keep v1 (`forward_centered`) untouched:
  every existing benchmark must reproduce bit-for-bit.
- Explanations to Koray in **Turkish**; code, docstrings and commit messages in English.

## Environment (Windows, PowerShell)

- Repo root: `C:\Users\koray\OneDrive\Masaüstü\vep-forward-tvtp-option-pricing-main\vep-forward-tvtp-option-pricing-main`
- Virtual env lives OUTSIDE OneDrive: `C:\Users\koray\venvs\vep`
  - activate: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; C:\Users\koray\venvs\vep\Scripts\Activate.ps1`
- Python 3.11.3; pinned deps in `requirements.txt` (+ `tabulate==0.9.0`, needed by
  `DataFrame.to_markdown`) and `requirements-dev.txt` (pytest 9.1.1).
- Extra tools installed in the venv: `eptr2`, `python-dotenv` (EPİAŞ API).
- `.env` in the repo root holds EPİAŞ credentials (`EPTR_USERNAME`, `EPTR_PASSWORD`).
  **Never print, read aloud, commit or copy it.**
- Checks: `python run_pde.py validate` (15/15) and `python -m pytest -q`
  (172 original + 29 new tests).

## Model in one paragraph

European option on the expiry-hour Turkish day-ahead price (PTF, TRY/MWh), valued
2025-12-31 23:00 TRT.  v1: price level pinned to the EPİAŞ VEP monthly baseload strip
(smooth KKT curve, exact to 1e-12), residual = 2-regime TVTP Markov-switching OU with
parameters inherited from the M9 fit; coupled Crank–Nicolson PDE + MC cross-check.

## Audit findings on v1 (measured against realised 2026 PTF)

1. No price cap/floor: cap 3400 TRY/MWh until 2026-04-04, 4500 after, floor 0;
   54/66 grid calls exceed the cap-implied upper bound (72 h K=3000: 687 vs bound 399).
2. Residual dispersion 5-10x too wide (kappa half-life 19 y inherited from raw asinh
   prices); 50 % band covered 99.7 % of realised hours.
3. No hourly shape (HPFC): ~40 % of residual variance was deterministic month x hour.
4. M9 parameters come from a USD run (`markov_usd_final`) mixed with TRY scale_P.
5. Q1 drift premium is structurally inert under forward-centering (Girsanov).
6. Docs quoted a stale 677.23 benchmark; code gives 687.04.

## v2 work done (all on real data; new modules beside v1)

- `pde_option_model/premium.py` — Q1 as forward risk-premium term structure:
  E^P[P]=F−π, a(t)=π′(t)+κπ(t); look-ahead-guarded realised premium panel;
  Tikhonov (MAP) premium curve.  Stress-premium variance uplift p0p1Δa²/(κ(κ+q)).
  Needs VEP history to estimate π (not yet downloaded).
- `pde_option_model/hpfc.py` — recency-weighted hourly shape, w=2^(−age/H),
  24 h × 3 day types × Fourier(K).  CV 2023-25 chose H=0.5 y, K=2 (ratio RMSE 0.1965
  vs 0.2088 equal weights).  2026 out-of-sample: within-month variance −33.6 %.
  Script: `scripts/hpfc/fit_hpfc.py` → `outputs/hpfc/`.
- `pde_option_model/residual_v2.py` — x=(P−M·S)/L (L = trailing 12-month mean);
  fast MS-AR(1) with TVTP on z and ramp (φ=0.676, half-life 1.8 h, σ=0.068/0.162·L,
  Hessian SEs), slow daily AR (half-life 0.93 d), naive level proxy [ASSUMED],
  prices clipped to [0, cap].  2026 coverage 50 %/90 %: v2 34/63 (January 68/95)
  vs v1 99.7/99.9.  72 h K=3000 call: v1 687 → v2 41.  Zero cap violations.
  Script: `scripts/residual/fit_residual_v2.py` → `outputs/residual_v2/`.
- `scripts/data/download_epias.py` — eptr2 downloader (`ptf`, `list-calls`, `probe`).
  PTF 2019-2025 is in `inputs/historical/ptf_raw/`.

## Next steps (in order)

1. Delete the old synthetic demo output folder `outputs/premium/` if it still exists.
2. Find the VEP API call: `python scripts\data\download_epias.py probe --call vep-price-summaries --start-date 2025-12-29 --end-date 2025-12-31`
   (also try `vep-contract-price-summary`, `vep-transaction-history`).  Check: EBM0226
   on 2025-12-31 should be 2900.99.  Then download VEP history 2022→ and write it in the
   schema of `inputs/market/vep_quotes_schema.md` (one row per valuation date × contract).
3. Estimate the real π(τ) with `scripts/premium/calibrate_premium.py` and replace the
   naive level proxy in residual_v2 with VEP forecast errors; re-check 2026 coverage
   (target: 50 % band 40-60 %, 90 % band 83-95 %).
4. Wire v2 into the PDE/CLI as a new model mode; then the Bayesian layer
   (Gupta & Reisinger 2012) — see `docs/Denetim_Raporu_VEP_TVTP_ve_Bayesci_Gelistirme.md`
   and `docs/ChatGPT_Proje_Promptu.md`.
