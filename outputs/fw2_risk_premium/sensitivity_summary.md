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
| Q1_only      |       90 |                  9.6804 |             47.7109 |                       9.6804 |                      -3.7485 |
| Q2_only      |      240 |                 50.6261 |             75.0786 |                      35.2249 |                     -50.6261 |
| joint_corner |      120 |                 64.4632 |            126.283  |                      48.6892 |                     -64.4632 |


## ATM K=3000, T=72 h call under every sweep row

| family       |   a_stress_TRY_MWh_per_h | eta_ij        |   value_TRY_MWh |   delta_vs_baseline_TRY_MWh |   delta_pct |   residual_sd_T |   p_stress_T |
|:-------------|-------------------------:|:--------------|----------------:|----------------------------:|------------:|----------------:|-------------:|
| baseline     |                        0 | None          |         166.748 |                       0     |       0     |          nan    |     nan      |
| Q1_only      |                       10 | None          |         167.591 |                       0.843 |       0.506 |          532.82 |       0.6008 |
| Q1_only      |                       25 | None          |         169.955 |                       3.207 |       1.924 |          535.76 |       0.6008 |
| Q1_only      |                       50 | None          |         176.237 |                       9.49  |       5.691 |          546.14 |       0.6008 |
| Q2_only      |                        0 | (-0.50,-0.50) |         166.266 |                      -0.482 |      -0.289 |          537.48 |       0.5979 |
| Q2_only      |                        0 | (-0.50,+0.00) |         142.077 |                     -24.67  |     -14.795 |          478.99 |       0.4744 |
| Q2_only      |                        0 | (-0.50,+0.50) |         116.193 |                     -50.554 |     -30.318 |          413.31 |       0.357  |
| Q2_only      |                        0 | (+0.00,-0.50) |         186.456 |                      19.709 |      11.82  |          579.53 |       0.7107 |
| Q2_only      |                        0 | (+0.00,+0.50) |         143.402 |                     -23.346 |     -14.001 |          474.5  |       0.4817 |
| Q2_only      |                        0 | (+0.50,-0.50) |         201.91  |                      35.162 |      21.087 |          612.84 |       0.8059 |
| Q2_only      |                        0 | (+0.50,+0.00) |         187.1   |                      20.352 |      12.205 |          577.16 |       0.718  |
| Q2_only      |                        0 | (+0.50,+0.50) |         167.908 |                       1.16  |       0.696 |          530.08 |       0.6106 |
| joint_corner |                       50 | (-0.75,-0.75) |         183.954 |                      17.206 |      10.318 |          564.01 |       0.6013 |
| joint_corner |                       50 | (-0.75,+0.75) |         102.456 |                     -64.292 |     -38.556 |          363.84 |       0.2533 |
| joint_corner |                       50 | (+0.75,-0.75) |         215.238 |                      48.491 |      29.08  |          641.83 |       0.8742 |
| joint_corner |                       50 | (+0.75,+0.75) |         173.144 |                       6.397 |       3.836 |          537.16 |       0.6158 |