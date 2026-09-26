# Literature anchor and a_i drift-shift bound (FW2 §2.2-2.3)

## §2.2 -- Literature bounds from `paper/refs.bib`

The three canonical Q1-style references cited in the manuscript
bibliography are

* Bessembinder & Lemmon (2002), *Equilibrium Pricing and Optimal
  Hedging in Electricity Forward Markets*, J. Finance 57(3), 1347-1382.
* Longstaff & Wang (2004), *Electricity Forward Prices: A
  High-Frequency Empirical Analysis*, J. Finance 59(4), 1877-1900.
* Lucia & Schwartz (2002), *Electricity Prices and Power Derivatives:
  Evidence from the Nordic Power Exchange*, RDR 5(1), 5-50.

The `refs.bib` entries carry the citation metadata but not the
premium magnitudes themselves.  Extracting numeric bounds requires
reading the papers directly; that has NOT been done in this session,
and no other repository file (docs/*, outputs/*) carries the numbers
either.  As per the FW2 rule "yeni kaynak uydurmayacaksın; sadece
repoda zaten atıf verilen kaynaklardan çıkarılabilenleri kullan",
this anchor is reported as **incomplete**:

* the three references ARE in the bibliography;
* the numeric magnitudes are NOT recoverable from the current repo;
* a complete literature-anchored bound requires a follow-up pass in
  which the papers are read and the reported premium ranges (per-
  MWh, per-horizon) are transcribed into a table here.

Two qualitative signs ARE reported in the manuscript's own literature
review (paper/sections/02-market-data.tex and 01-introduction.tex):
peak-hour forwards typically trade AT A POSITIVE PREMIUM to expected
spot; off-peak forwards can trade at a small negative premium.  Both
signs are consistent with the ex-post panel in §2.1 (positive mean
premium across every horizon).  Magnitude bounds are deferred.

## §2.3 -- Translating a premium bound to a_i (TRY/MWh per hour)

### Derivation

Under Q1 (drift shift a_i but no transition shift), the residual OU
`dX = [kappa (m - X) + a] dt + sigma dW` reaches a steady-state
conditional mean `mu_ss = m + a/kappa` in each regime, and the
per-hour drift-shift a_i can be translated to a level bound by

    |mu_ss - m| = |a| / kappa    <=>    |a| = kappa * |level shift|.

Because forward centering subtracts mu_X(t) from the residual for
E^Q[P_t] = F(t), the OBSERVED level of E^Q[P_t] does not change with
a_i; the "level shift" bound is therefore the bound on the DIFFERENCE
between the physical expectation E^P[P_t] and the forward F(t),
i.e. the risk premium itself.

With yaml v2 `kappa = 0.078394 /h` (half-life 8.84 h), a delivery-
month-average premium bound of L TRY/MWh translates as

    |a_i| <= kappa * L = 0.078394 * L.

### Numbers from the ex-post panel (descriptive only)

`outputs/fw2_risk_premium/ex_post_premium_summary.csv` reports (with
`n_obs = 5-6` per horizon):

* mean premium across horizons 2-7 months: ~435-783 TRY/MWh;
* pooled mean across all rows: ~625 TRY/MWh;
* 95th percentile of |premium|: ~2000 TRY/MWh (dominated by the
  2026 Q2 realised collapse; see F2.9 backtest);
* block-bootstrap SEs: 280-530 TRY/MWh -- the same order of
  magnitude as the means.

Converting to a_i bounds:

| bound source | premium L TRY/MWh | |a_i| bound TRY/MWh/h |
|---|---:|---:|
| pooled mean (central, descriptive) | 625 | **48.99** |
| horizon-2 mean (short-dated ceiling) | 490 | 38.41 |
| horizon-7 mean (long-dated floor) | 435 | 34.11 |
| 95th %ile of \|premium\| (envelope) | 2000 | 156.79 |

**These are per-regime BOUNDS, not identified estimates.**  The panel
size (n = 5-6 per horizon, one realised outcome per delivery month)
is far too small for a consistent estimator of the risk premium; the
mean is dominated by the 2026 realised collapse and by the shift in
Turkey's electricity-price regime around 2022-Q1.

### Comparison to the accepted risk-premium sensitivity CSV

`outputs/market_calibration_final/risk_premium_sensitivity.csv`
already tests a_i in {0.004, 0.02, 0.5} TRY/MWh/h.  The empirical
band above sits ABOVE all of these, by 2-3 orders of magnitude.  A
priori this is a very large bound; the reason is that the ex-post
panel is dominated by two structural regime shifts (2022 gas shock,
2026 Q2 renewables collapse) rather than by an equilibrium premium.
The FW2 sensitivity sweep in §4 therefore uses a MORE CONSERVATIVE
range for the primary comparison ({0, 5, 25, 50} TRY/MWh/h per
regime), with a single "empirical upper bound" point at
|a_stress| = 50 included for the sensitivity envelope.

The take-away for the manuscript: **at the empirical-upper-bound
magnitudes, the Q1 drift channel is still a variance-effect knob;
the option-value sensitivity it produces is quadratic in a and
therefore small in the empirical range**.  See §4 for the numeric
verification.

### On Q2 transition premia eta_ij

There is no matching empirical anchor: eta_ij shifts the P-measure
transition intensity into a Q-measure one, and no observable in this
repo separates the Q-measure intensity from the P-measure intensity.
The bound used in §4 is therefore GENERATOR-VALIDITY only: any real
eta is admissible, but the sweep restricts to `|eta_ij| <= 1` so the
resulting q_ij^Q stays within one order of magnitude of q_ij^P (a
conservative range that keeps the discrete-recovery `s = p01 + p10`
well inside the embeddability region).
