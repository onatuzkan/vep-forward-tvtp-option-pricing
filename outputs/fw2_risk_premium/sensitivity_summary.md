# FW2 §4 -- Joint (a, eta) sensitivity sweep

See `docs/fw2_risk_premium_identification.md` for the two-
propositions framing.  This report exercises Q1 and Q2 in 
isolation and jointly at the empirical §2.3 upper bounds; 
it produces the price-effect band the manuscript will 
report as its risk-premium uncertainty envelope.

* Contracts: 24 / 48 / 72 h call+put on strike ladder 
[2000.0, 2500.0, 3000.0, 3500.0, 4000.0].
* Q1 grid: a_stress in [0.0, 10.0, 25.0, 50.0] TRY/MWh/h 
(a_normal fixed at 0 -- an asymmetric shift; a symmetric 
shift has zero variance effect at this order).
* Q2 grid: (eta_01, eta_10) in [-0.5, 0.0, 0.5]^2 
(multiplicative q^Q = q^P * exp(eta)).
* Joint corners: a_stress = 50.0, eta_ij = (+/-0.75, +/-0.75)

## Max effect per channel

| family       |   n_rows |   max_abs_delta_TRY_MWh |   max_abs_delta_pct |   max_positive_delta_TRY_MWh |   max_negative_delta_TRY_MWh |
|:-------------|---------:|------------------------:|--------------------:|-----------------------------:|-----------------------------:|
| Q1_only      |       90 |                  7.1789 |             33.7388 |                       7.1789 |                      -2.2075 |
| Q2_only      |      240 |                 48.347  |             62.851  |                      30.0074 |                     -48.347  |
| joint_corner |      120 |                 63.2584 |             97.7558 |                      40.0191 |                     -63.2584 |


## ATM K=3000, T=72 h call under every sweep row

| family       |   a_stress_TRY_MWh_per_h | eta_ij        |   value_TRY_MWh |   delta_vs_baseline_TRY_MWh |   delta_pct |   residual_sd_T |   p_stress_T |
|:-------------|-------------------------:|:--------------|----------------:|----------------------------:|------------:|----------------:|-------------:|
| baseline     |                        0 | None          |         179.648 |                       0     |       0     |          nan    |     nan      |
| Q1_only      |                       10 | None          |         180.278 |                       0.63  |       0.35  |          560.53 |       0.6705 |
| Q1_only      |                       25 | None          |         181.981 |                       2.333 |       1.299 |          562.78 |       0.6705 |
| Q1_only      |                       50 | None          |         186.694 |                       7.046 |       3.922 |          570.77 |       0.6705 |
| Q2_only      |                        0 | (-0.50,-0.50) |         177.387 |                      -2.261 |      -1.259 |          560.09 |       0.6705 |
| Q2_only      |                        0 | (-0.50,+0.00) |         155.836 |                     -23.812 |     -13.255 |          508.51 |       0.5524 |
| Q2_only      |                        0 | (-0.50,+0.50) |         131.37  |                     -48.278 |     -26.874 |          447.83 |       0.4281 |
| Q2_only      |                        0 | (+0.00,-0.50) |         196.386 |                      16.737 |       9.317 |          600.28 |       0.7704 |
| Q2_only      |                        0 | (+0.00,+0.50) |         158.584 |                     -21.065 |     -11.726 |          508.51 |       0.5524 |
| Q2_only      |                        0 | (+0.50,-0.50) |         209.47  |                      29.822 |      16.6   |          629.33 |       0.8469 |
| Q2_only      |                        0 | (+0.50,+0.00) |         197.518 |                      17.87  |       9.947 |          600.28 |       0.7704 |
| Q2_only      |                        0 | (+0.50,+0.50) |         181.152 |                       1.504 |       0.837 |          560.1  |       0.6705 |
| joint_corner |                       50 | (-0.75,-0.75) |         190.696 |                      11.047 |       6.149 |          579.59 |       0.6705 |
| joint_corner |                       50 | (-0.75,+0.75) |         116.501 |                     -63.147 |     -35.151 |          397.57 |       0.3123 |
| joint_corner |                       50 | (+0.75,-0.75) |         219.321 |                      39.673 |      22.084 |          651.65 |       0.9012 |
| joint_corner |                       50 | (+0.75,+0.75) |         184.988 |                       5.339 |       2.972 |          565.55 |       0.6705 |