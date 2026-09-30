# Energy Markets Options Valuation

PDE-based European option pricer for the Turkish day-ahead market
(EPIAS PTF, TRY/MWh). The price level is anchored to the EPIAS VEP
monthly baseload forward strip; the residual around that curve
follows a two-regime TVTP (time-varying transition probability)
Markov-switching OU process whose parameters are inherited from the
M9 fit of the historical hourly PTF series and reconciled to the
deseasonalized (P - F) persistence by the v2-kappa refit. Every
option value produced here is labelled **VEP-forward-curve anchored
option prices**: only the first moment is market-identified;
volatility and regime-transition risk premia are not.

## Parameter provenance and independent validation

The frozen parameters live in
`inputs/historical/m2_frozen_parameters.yaml`. The M9-fit values for
sigma and gamma are copied in with the regime-label swap
(`sigma_y[1] > sigma_y[0]`) enforced by
`params_frozen.FrozenM2Parameters.__post_init__`. The two
alpha intercepts are derived from the reported M9 mean-duration
diagnostics by root-finding in
`scripts/tvtp_derivation/derive_tvtp_parameters.py`. The v2-kappa
refit reconciled `phi`, `kappa_per_hour` and `half_life_hours` to
the deseasonalized single-regime AR(1) values (`phi = 0.9246`,
`kappa_per_hour = 0.078394`, half-life 8.84 h). Every provenance
caveat is documented in
`outputs/market_calibration_final/model_limitations.md` items (a)
through (h).

FW9 rebuilt the same TVTP MS-AR(1) fit independently on the model-
faithful residual and produced a full battery of validation
outputs (see `outputs/fw9_self_estimation/`). Headline: the yaml kappa and innovation scale are individually far from the FW9 estimates, but they offset in the implied stationary residual variance, which is closer to the observed 2025 outcome than any FW9 alternative (TRY sd 582.5 versus observed 531-624; KS distance
0.087 versus 0.125 for FW9e A3 full window and 0.159 for FW9f
2022-2025 regime-matched). FW11 finds the same match at the six dates from mid-2023 to end-2025; the 2022 crisis year exceeds it and 2026 breaks it. See `outputs/market_calibration_final/model_limitations.md` items (i) and (j).

## Repository layout

```
.
├── README.md                         this file
├── requirements.txt                  runtime deps (numpy, pandas, scipy, matplotlib, PyYAML, tabulate, openpyxl)
├── requirements-dev.txt              adds pytest for the test suite
├── run_pde.py                        CLI: validate | calibrate-market | price | diagnostics | freeze-params
├── scenario_sweep.py                 helper: +/- 2 sigma RD-offset sensitivity of the 72 h call
├── config/                           YAML configs (default forward_centered_config.yaml)
├── docs/                             methodology notes and this project's status document
├── inputs/
│   ├── historical/                   frozen TVTP parameters, z series, ramp series
│   ├── historical/archive/           raw M9 handoff bundle, pre-refit yaml snapshots
│   ├── market/                       EPIAS VEP monthly quotes (CSV + JSON + schema)
│   ├── legacy_reference/             legacy sinh-Gaussian benchmark curve
│   └── macro/                        TUIK consumer price index (CPI deflation)
├── pde_option_model/                 Python package (PDE solver, forward curve, TVTP generator, calibration)
├── tests/                            pytest suite (real count reported at the bottom)
├── paper/                            LaTeX manuscript (elsarticle), figures, compiled PDF
├── scripts/
│   ├── backtest/                     multi-date forward-curve backtest
│   ├── data/                         EPIAS downloader (eptr2 wrapper)
│   ├── dev_checks/                   one-shot dev verification scripts
│   ├── fw2/                          FW2 risk-premium wiring and sensitivity sweep
│   ├── fw3/                          FW3 closed-form benchmark comparison
│   ├── fw4p/                         FW4-P ramp price-impact rebuild
│   ├── fw6a/                         FW6a rebuild of F2.5 under the v2 kappa
│   ├── fw9/                          FW9 independent MLE re-estimation and validation battery
│   ├── fw10/                         FW10 day-ahead timing audit and out-of-sample validation
│   ├── fw11/                         FW11 stationary-variance stability across dates
│   ├── fw12/                         FW12 convergence study and FW12b scenario-plumbing repair
│   ├── hpfc/                         HPFC (hourly price forward curve) fitter
│   ├── premium/                      ex-post forward-premium panel builder
│   ├── residual/                     residual_v2 estimator
│   ├── tvtp_derivation/              alpha-derivation script (referenced by docs/tvtp_derivation_methodology.md)
│   └── tvtp2/                        experimental two-covariate TVTP mode
└── outputs/                          generated artefacts
    ├── market_calibration_final/     accepted calibration + audit + archived alternatives
    ├── forward_centered_diagnostics/ residual diagnostics for the accepted model
    ├── scenario_sweep/               RD-offset sensitivity (single file)
    ├── tvtp2_experimental/           experimental two-covariate TVTP outputs
    ├── multi_date/                   multi-date forward-curve backtest summary
    ├── hpfc/                         HPFC fit outputs
    ├── residual_v2/                  residual_v2 fit outputs
    ├── fw2_risk_premium/             FW2 identifiability envelope and sensitivity sweep
    ├── fw3_benchmarks/               FW3 benchmark grids and IV surfaces
    ├── fw4p_ramp_price_impact/       FW4-P occupancy-controlled ramp effect
    ├── f25_v2_kappa/                 FW6a rebuild of F2.5 under the v2 kappa
    ├── docs_update/                  hash check for the documentation update
    ├── fw9_self_estimation/          FW9 MLE re-estimation, kappa bracket, tail validation
    ├── fw10_validation/              FW10 day-ahead audit and FW10b evaluation panel
    ├── fw11_variance_stability/      FW11 stationary-variance stability table
    └── fw12_convergence/             FW12 spatial and time convergence, grid recommendation
```

## Installation

```
python -m venv .venv
.venv/Scripts/activate                # Windows PowerShell / bash-for-Windows
# source .venv/bin/activate           # Linux / macOS
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The requirements files pin the versions the test suite is verified
against: numpy 1.26, pandas 2.3, scipy 1.16, matplotlib 3.10,
PyYAML 6.0, tabulate 0.9, openpyxl 3.1, pytest 9.1.

## Running

```
# 1. Structural and numerical self-checks (no calibration, about 1 s)
python run_pde.py validate

# 2. Rebuild the accepted forward-curve calibration under the frozen yaml
python run_pde.py calibrate-market --model forward_centered \
    --outdir outputs/market_calibration_final

# 3. Price a 72 h European call at K = 3000 TRY/MWh (PDE + Monte Carlo cross-check)
python run_pde.py price --model forward_centered \
    --curve outputs/market_calibration_final \
    --option-type call --strike 3000 --maturity-hours 72

# 4. RD-scenario sweep (+/- 2 sigma standardized RD offset, 9 rows)
python scenario_sweep.py

# 5. Full test suite
python -m pytest -q
```

### Optional: sensitivity to the regime-probability initial condition

The default `pi_filtered` in the frozen yaml is the M2-shipped
filter `[0.9320, 0.0680]`; a valuation-time M9 filter is not
available. Under `--pi-override stationary` the pricer swaps in the
M9 stationary occupancy `[0.3254, 0.6746]`. Impact is under 2 % on
72 h and longer maturities and up to about 30 % on under-24 h
maturities. See `model_limitations.md` item (e).

```
python run_pde.py price --model forward_centered \
    --curve outputs/market_calibration_final \
    --strike 3000 --maturity-hours 72 --pi-override stationary
```

### Optional: EXPERIMENTAL two-covariate TVTP

The default transition law is the single-covariate `rd_lag1_1d`.
A second, explicitly selected mode adds the residual-demand ramp
covariate `RD_Ramp_1h_lag1` of the M9 TVTP-2 fit. Its definition
could not be found in any available source, so the ramp is a
reconstruction (hourly difference of z on the complete UTC grid,
train-only standardization) and every output is labelled
*M9-transferred slopes + reconstructed ramp + derived intercepts,
zero transition premium*. It does not reproduce M9 exactly and does
not change any accepted result. See
[`docs/tvtp2_methodology.md`](docs/tvtp2_methodology.md),
`outputs/tvtp2_experimental/PROVENANCE_REPORT.md`, and the FW4-P
occupancy-controlled price impact under `outputs/fw4p_ramp_price_impact/`.

```
python run_pde.py --config config/forward_centered_tvtp2_experimental.yaml validate
python run_pde.py --config config/forward_centered_tvtp2_experimental.yaml price --strike 3000 --maturity-hours 72
python scripts/tvtp2/compare_tvtp_1d_2d.py
```

## Tests

The pytest suite is a mix of structural invariants, calibration
regression tests, and long-running numerical checks. The `slow`
marker is registered in `pytest.ini` and is NOT deselected by
default; a plain `python -m pytest -q` runs everything. Test
count is reported by pytest itself at the bottom of the run; the
`FW9f_report_TR.md`, `FW10_report_TR.md`, `FW11_report_TR.md` and
`FW4P_report_TR.md` documents record the count at their respective
work-package cut-offs.

To skip the slow tests locally:

```
python -m pytest -q -m "not slow"
```

`tests/test_fw9b_profile.py::test_deseasonalized_ar1_reproducibility_and_finite_output`
invokes `scripts/fw9/deseasonalized_ar1.py`. The script writes `outputs/fw9_self_estimation/deseasonalized_ar1.json`; the test backs the file up and restores it, so a full run leaves the tracked file unchanged.

## Main results (headline numbers, source in parentheses)

| quantity | value | source |
|---|---|---|
| 72 h call, K = 3000, production grid | 166.75 TRY/MWh | `outputs/fw12_convergence/spatial_convergence.md` |
| Richardson-extrapolated value | 166.686 TRY/MWh | `outputs/fw12_convergence/grid_recommendation.md` |
| production-grid relative error vs Richardson | 0.037 % | `outputs/fw12_convergence/grid_recommendation.md` |
| stationary residual sd (TRY at spot 2917.78) | 582.5 | `outputs/fw9_self_estimation/stationary_variance_check.md` |
| observed 2025 A3 residual sd (2025-only shape / pooled shape) | 531 / 624 TRY | `outputs/fw9_self_estimation/tail_validation.md` |
| KS distance yaml vs 2025 observed | 0.087 | `outputs/fw9_self_estimation/tail_validation.md` |
| KS distance FW9e A3 full-window vs 2025 observed | 0.125 | same |
| KS distance FW9f 2022-2025 regime-matched vs 2025 observed | 0.159 | same |
| TVTP vs constant-transition LR (deseasonalized) | 1178.66 (df = 2) | `outputs/fw9_self_estimation/lr_test_TVTP_vs_constant.csv` |
| TVTP vs constant-transition LR (A3 residual, 2019-2025 / 2022-2025) | 1851 / 1692.56 | `outputs/fw9_self_estimation/README.md`, `FW9f_report_TR.md` §3 |
| FW9 two-covariate LR vs one-covariate (fit not converged) | 869.93 (df = 2) | `outputs/fw4p_ramp_price_impact/FW4P_report_TR.md` §1 |
| ramp channel on 72 h K = 3000 (occupancy-controlled) | -1.042 % | `outputs/fw4p_ramp_price_impact/fw4p_ramp_effect.csv` |
| FW2 sensitivity envelope, 72 h, K = 3000 (Q2 alone at \|eta\| = 0.5) | +21 % to -30 % | `outputs/fw2_risk_premium/sensitivity_summary.md` |
| FW3 realized 2026 MAE, Model / B3 / B2 / B1 | 151 / 157 / 272 / 369 TRY/MWh | `outputs/fw3_benchmarks/README.md` |
| FW10b pure-residual sd, 2026 (Jan-Sep) | 723.1 TRY/MWh | `outputs/fw10_validation/forward_residual_decomposition.md` |
| model residual sd, h = 6..72 (FW10b) | 387..518 TRY/MWh | same |
| VEP GGF quote change frequency (2026 delivery contracts) | 14 / 1354 (1.0 %) | `outputs/fw10_validation/vep_quote_staleness.csv` |
| FW11 realised / production dispersion, 2023-06 to 2025-12 (12-month shape) | 0.93 to 1.13 | `outputs/fw11_variance_stability/variance_stability.md` |
| FW11 sd/L, 2026 only, 12-month shape (production 0.199) | 0.384 | same |

## Reproducibility

Four output trees are treated as "accepted" and are guarded against
accidental changes:

* `outputs/market_calibration_final/`
* `outputs/forward_centered_diagnostics/`
* `outputs/scenario_sweep/`
* `outputs/tvtp2_experimental/`

A 128-file SHA-256 manifest (CRLF-to-LF normalised) of these trees
and the frozen inputs lives at
`outputs/fw12_convergence/hashes_fw12b_after.txt`. Every work
package after FW12 saves an equivalent before-and-after manifest
under its own `outputs/fw*/hash_manifest_*.txt` and reports the
diff in its Turkish report.

Valuation cutoff for the shipped yaml is 2025-12-31 20:00 UTC
(23:00 TRT). No script uses 2026 hourly PTF in any parameter
estimate; FW10b enforces a strict day-ahead publication rule for
its 60-day daily evaluation (valuation at 11:00 TRT of day d, last
known PTF hour d 23:00 TRT, forward curve from the last VEP GGF
publication strictly before d 11:00 TRT). See
`outputs/fw10_validation/day_ahead_timing_check.md` for the audit.

## Data

The market data used in this repository is taken from the **EPIAS
Seffaflik Platformu** (Energy Market Transparency Platform,
https://seffaflik.epias.com.tr/), which publishes Turkish electricity
market data under an EPDK (Enerji Piyasasi Duzenleme Kurumu) Board
Decision. The relevant files under `inputs/market/` and
`inputs/historical/` were re-published here for the sole purpose of
reproducing the accompanying manuscript's calibration, backtest, and
sensitivity results. Their original terms of use continue to apply;
see the "Raw EPIAS data" section of [`LICENSE`](LICENSE) and
https://seffaflik.epias.com.tr/ for the current Terms of Use.

EPIAS is the original publisher of this data and is not responsible
for the accuracy, completeness, or interpretation of the data as
re-published or transformed here. Any errors introduced by the
preparation scripts (`scripts/data/download_epias.py` and the
downstream feature builders) are the authors' responsibility, not
EPIAS's.

EPIAS credentials for the downloader are read from a local `.env`
file at the repo root; the file itself is not committed. The
downloader documents the environment-variable names it looks for;
neither the names nor their values are printed by any of the
scripts.

Users who wish to redistribute or reuse the raw EPIAS-sourced files,
or any derivative thereof, should consult
https://seffaflik.epias.com.tr/ directly.

## License

Four-part licensing; see [`LICENSE`](LICENSE) for the full text:

* **Code** MIT. Everything under `pde_option_model/`, `scripts/`,
  `tests/`, `paper/make_figures.py`, `paper/make_pde_mc.py`,
  `run_pde.py`, `scenario_sweep.py`.
* **Manuscript (`paper/`)** CC BY-NC-ND 4.0, matching the SSRN
  release. Applies to the LaTeX source (`main.tex`, `refs.bib`,
  `sections/*.tex`, `figures/*.pdf`, `figures/pipeline_src.tex`),
  the compiled PDF, and `paper/README.md`. The regeneration scripts
  `paper/make_figures.py` and `paper/make_pde_mc.py` themselves
  fall under the MIT code licence.
* **Documentation and derived data** CC BY 4.0. Everything under
  `docs/`, every `.md` under `outputs/`, every derived CSV / JSON /
  YAML / PNG under `outputs/`, and
  `inputs/historical/m2_frozen_parameters.yaml`. When citing please
  reference the accompanying preprint.
* **Raw EPIAS data** under the EPIAS Seffaflik Platformu Terms of
  Use.

## Manuscript

The preprint accompanying this repository:

* **Title:** *Anchoring Options to an Incomplete Forward Curve: Identification and Out-of-Sample Evidence from Turkish Electricity*
* **Authors:** Onat Uzkan (Ozyegin University) and Koray Omercan Sacli (Istanbul Technical University)
* **Class:** `elsarticle` (Elsevier preprint), pdfLaTeX; 10 body sections and 3 appendices

The manuscript source and compiled PDF live under [`paper/`](paper/).
Compile with pdfLaTeX and BibTeX. Bump the PDF version suffix
(`Uzkan_Sacli_2026_forward_anchored_option_valuation_v<N>.pdf`) on
every material revision.
