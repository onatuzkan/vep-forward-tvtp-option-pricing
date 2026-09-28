# FW9c §6 -- Proposed lines for `outputs/market_calibration_final/model_limitations.md`

The four existing items (a)-(h) that FW9 touches are updated below with
the FW9b/FW9c evidence.  **Do not edit `model_limitations.md` before
your approval.**

## (a) DERIVED alpha01 / alpha10

Proposed line:
> FW9c profile MLE at phi = 0.99999 (fixed) with a positive-definite
> Hessian gives alpha01 = -0.6781 +/- 0.019 and alpha10 = -1.6511 +/-
> 0.017.  The yaml DERIVED values (-1.0157 and -1.8952) sit ~14-18 SEs
> outside these CIs, so the occupancy/duration root-finding pipeline
> does not reproduce the direct MLE alphas even at the correct
> conditional profile.  However, FW9c §5 decomposition shows that this
> discrepancy moves the ATM K=3000 T=72 h call by only +3.75 TRY
> (2.2 % price change, 0.4 % share of the total FW9-vs-production
> gap), so the shipped alpha choice is not the dominant driver of the
> production-vs-independent-MLE gap.
> Source: `outputs/fw9_self_estimation/parameter_comparison_v2.csv`,
> `price_impact_v2_decomposition.csv`.

## (c) unverifiable `scale_P`

Proposed line:
> FW9 applies the "training median absolute price" definition on the
> 2019-2025 window and finds 1 399.99 TRY/MWh; on the 2019-2020
> subwindow (before the 2021-2024 TRY depreciation) it finds
> 302.02 TRY/MWh, which is within 7 % of the yaml value 282.48.  FW9
> concludes the yaml scale_P is consistent with an early-window TRY
> median but the exact reference window is not shipped with the repo
> (see `outputs/fw9_self_estimation/scale_P.json` and
> `preprocessing_audit.md`).  The FW9 decisive test against the
> alternative "USD-fit" hypothesis (§0a) killed that possibility: the
> bundle metadata's `target` field is `"asinh(PTF_TRY_MWh)"`, and the
> decisive TVTP fit on `asinh(USD_PTF/282.48)` does not land on the
> yaml sigmas.

## (e) M2-sourced `pi_filtered`

Proposed line:
> FW9c runs the Hamilton filter on the FW9 data at the profile
> parameters and obtains a terminal filtered distribution `[0.983,
> 0.017]` at the valuation instant, vs the yaml M2-sourced `[0.932,
> 0.068]` -- a ~5 pp shift, both dominated by the normal regime.
> FW9c §2 additionally reprices the ATM K=3000 call under both
> distributions at T in {1, 2, 6, 12, 24, 48, 72} h: the effect
> shrinks with horizon (20.8 % at T=1 h, 1.6 % at T=6 h, 0.06 % at
> T=24 h, essentially zero at T=72 h) because the regime memory
> half-life at climatology z=0 is ~1.37 h.  **For the shipped
> reporting horizons (T >= 24 h) the pi_filtered source uncertainty
> is a fraction of one basis point and the (e) limitation is
> effectively closed.**
> Source: `outputs/fw9_self_estimation/pi_filtered_horizon.csv`.

## (h) scale_P / TRY-depreciation window mismatch

Proposed line:
> Quantified by FW9 §3 and §7 (see (c)).  FW9c additionally runs the
> full TVTP profile fit on the CPI-deflated series (`scripts/fw9/
> estimate_deflated.py`; base 2025-12, deflated `scale_P = 3092.05`);
> real-terms sigmas are 22-33 % smaller than nominal (`sigma_normal`
> 0.0055 vs 0.0070, `sigma_stress` 0.162 vs 0.242), so inflation
> contributes ~30 % of the observed volatility.  Even after deflation
> FW9 sigmas remain 55-75 % larger than the yaml values, so the
> remaining gap is attributable to the training-window difference
> (M9 saw 2016-2018 that FW9 does not).

## (i) FW9e revision -- Mean-reversion + variance error cancellation

Proposed line (supersedes both the FW9c and FW9d drafts of (i)):
> The production yaml and FW9-family estimates disagree individually
> on both `kappa` (yaml 0.0784 /h vs FW9e A3-consistent MLE 0.152 /h;
> ratio 1.94x) and `sigma_y` (yaml mixture sigma 0.0757 vs FW9e A3
> mixture 0.162; ratio 2.14x), but the two errors partially CANCEL
> in the implied stationary residual variance
> `sigma_mix^2 / (1 - phi^2)`.  In TRY terms at the valuation spot
> (delta-method from asinh at F = 2 917.78):
>  * production stationary sd_TRY = **582.5**
>  * FW9e A3 consistent-fit stationary sd_TRY = 929.0
>  * observed 2025 A3-residual sd_TRY = 624.0
>  * production/observed_2025 ratio = **1.07** (within 7 %)
>  * FW9e_A3/observed_2025 ratio = 1.49 (49 % excess)
> **The shipped kappa and sigma are BOTH mis-specified against a
> direct-MLE benchmark, but their PRODUCT (stationary variance)
> lands within 7 % of the 2025 outcome.**  For reporting horizons
> T >= 24 h the option price is dominated by the stationary variance
> (see FW9e §5 iso-variance sweep: within the model-faithful
> kappa band [0.15, 0.22] /h the 72 h ATM call ranges 439-442, a
> 0.6 % band); the 12 h and shorter horizons carry a larger kappa
> sensitivity (call_T24 spans 105-182 in the production-target
> iso-variance sweep, 73 %).  The paper should therefore report a
> kappa-sensitivity figure and cite the 24-72 h numbers with a "one-
> sigma parameter uncertainty on the underlying kappa cluster is
> approximately +/- 3 % of the ATM 72 h call" caveat.  Source:
> `outputs/fw9_self_estimation/stationary_variance_check.csv`,
> `price_impact_A3_consistent.csv`,
> `kappa_sensitivity_isovariance.csv`.

## FW9c/FW9d (i) draft -- SUPERSEDED

The FW9d (i) framing ("kappa OUTSIDE all three brackets") is
mechanically correct on individual parameters but MISLEADS in the
implied-variance picture; see the FW9e revision above.  The
correct paper claim is:
> yaml kappa and sigma are individually far from the FW9-family
> point estimates, but the STATIONARY VARIANCE they imply is close
> to the observed 2025 outcome.

## NEW proposed item -- Mean-reversion parameter reproducibility

Proposed new item (id `(i)` if unused):
> **(i) Mean-reversion parameter not independently reproducible.**  The
> yaml carries `phi = 0.9246, kappa_per_hour = 0.078394,
> half_life = 8.84 h`, sourced from the deseasonalized single-regime
> AR(1) reported in
> `metadata/deseasonalized_stationarity_summary.csv` of the M9 bundle.
> FW9c §1 brackets the (P - F_monthly) residual AR(1) `kappa` across
> six deseasonalization specifications S0-S5 (from no seasonality to
> hour x dow interaction + annual/semi-annual Fourier).  All six give
> `kappa` in the range **[0.205, 0.225] /h** (`half_life` ~3.1-3.4 h);
> the yaml value 0.078 is **~2.7x smaller and outside the bracket**.
> Reproducing the yaml value requires the exact M9 deseasonalization
> pipeline, which is not shipped with the repo.  FW9c §5 decomposition
> shows the kappa disagreement is one of the two dominant contributors
> to the FW9-vs-production ATM 72 h call gap (+143 % isolated price
> effect, 25 % share of the total delta).
> Source: `outputs/fw9_self_estimation/kappa_bracket.csv`,
> `price_impact_v2_decomposition.csv`.

## (i) FW9f revision -- Stationary-variance closeness confirmed by direct simulation

Proposed line (SUPERSEDES the FW9e revision above; do NOT edit `model_limitations.md` before approval):
> FW9f section 4 runs a fast single-long-path stationary simulation of
> the two-regime MS-AR(1) plus TVTP process under three parameter
> sets and compares the delta-mapped TRY residual distribution to the
> observed 2025 A3 residual (`outputs/fw9_self_estimation/
> tail_validation.csv`, `.md`, `.ks.csv`).  The findings:
>  * The **production yaml** delivers the closest match to the 2025
>    outcome across every summary: TRY sd 581 (observed 531,
>    +9 pct), KS statistic 0.087, 5-95 pct centre-tail deltas
>    +41 / +229 TRY, 25-75 pct interquartile range within 33 % of
>    observed.
>  * The **FW9e A3 consistent full-window fit** widens the dispersion
>    to sd 984 (+85 pct vs observed), KS 0.125, 5-95 pct deltas
>    -721 / +743 TRY.
>  * The **FW9f regime-matched 2022-2025 A3 fit** widens further to
>    sd 1176 (+121 pct), KS 0.159, 5-95 pct deltas -1138 / +990 TRY.
>  * Restricting the estimation to 2022-2025 (removing the low-price
>    2019-2021 sub-sample) does NOT tighten the residual dispersion;
>    the two-regime mixture pushes more mass into the stress regime
>    when the sample is high-price-only.
> The correct paper claim on the shipped yaml is therefore:
> **the yaml kappa and sigma_y jointly imply a stationary residual
> distribution that is closer to the 2025 outcome than any of the
> FW9-family direct-MLE alternatives**, once the observed price cap
> at 4500 TRY/MWh and floor at 0 TRY/MWh are taken into account (the
> observed distribution is truncated, the model residual is not).
> The earlier FW9c-FW9d framing ("yaml kappa is wrong, outside all
> three brackets") IS WITHDRAWN and should not appear in the paper.

## FW9c/FW9d (i) draft -- WITHDRAWN

The FW9c "kappa outside bracket [0.205, 0.225]" claim and the FW9d
"kappa outside all three brackets" claim are both WITHDRAWN in
favour of the FW9f revision above.  The bracket is a property of a
different residual specification (raw asinh - single-regime AR(1)
under a chosen deseasonalization) than the one the pricer actually
uses (A3 asinh residual under two-regime MS-AR(1) with TVTP), and
comparing them directly conflates two distinct kappas.  The paper
should state the yaml kappa as-shipped and cite FW9f section 4 as
the empirical justification.

## Summary of closability

| item | before FW9 | after FW9c |
|---|---|---|
| (a) alpha derived | open | measurably rejected but 0.4 % of the gap; document with FW9c numbers, keep open |
| (c) scale_P unverifiable | open | window-explained (2019-2020 median ~302); USD hypothesis killed; still not fully verifiable |
| (e) M2-sourced pi_filtered | open | **effectively closed for T >= 24 h reporting horizons** |
| (h) scale_P depreciation | open | quantified (~30 % from inflation); remaining ~55-75 % from training-window difference |
| (i) mean-reversion reproducibility | (new) | WITHDRAWN by FW9f; yaml stationary-variance match to 2025 outcome is closest of all tested parameter sets |
