# Note on `risk_premium_sensitivity.csv`

Effect sizes at these magnitudes are near/below PDE solver precision
(~1e-4 TRY); see `docs/risk_neutral_methodology.md` for the log-log
verification that this reflects genuine O(a²) scaling, **not** numerical
noise.

The `call_delta_vs_baseline_pct` column at rows 2-3 (`±0.004`) sits at
~1e-5 %, which is 3-4 orders of magnitude smaller than the PDE grid can
resolve.  The theoretical prediction for these magnitudes is
`Δcall ≈ (a / a_ref)² · Δcall_ref` — checked out to four significant
digits on the variance channel across five orders of magnitude in `|a|`
in the diagnostic sweep documented in
`docs/risk_neutral_methodology.md` §"Empirical verification of O(a²)
scaling".

The intended use of this table:

1. **Confirm** that `F(T)` and `E^Q[P_T]` are invariant to `(a_0, a_1)`
   (both columns are identical across rows).  This is the forward-curve
   identity `E^Q[P_t] = F(t)` being upheld under Q1.
2. **Confirm** that the large-magnitude probe row (`a_1 = −0.5`)
   produces a visible-but-still-small effect on `residual_sd_T` and
   `call_value`, demonstrating the drift channel is genuinely wired to
   the variance path.
3. **Read the "plausible" rows as the baseline**: at magnitudes that
   would actually calibrate to option premia, the drift channel moves
   the price by well under 1 TRY — a knob that exists but requires
   sizable premia to shift a 72 h call value.
