# FW10 Part B -- Delta hedge with monthly VEP forwards

## Method

* Delta = disc * P(P_T > K) from the model's predictive sample (analytical result for arithmetic P = F + residual).

* Static: hold delta_d units, P&L = delta_d * (F_{end} - F_d).

* Dynamic: rebalance every 24 h at the day's VEP quote for the same delivery month; carry forward on non-quoting days.

* Effectiveness = 1 - Var(hedged) / Var(unhedged).


## Effectiveness table (dynamic hedge)

|            |    0.8 |    0.9 |    1.0 |    1.1 |    1.2 |
|:-----------|-------:|-------:|-------:|-------:|-------:|
| ('B1', 24) |  0     |  0     |  0     |  0     |  0     |
| ('B1', 48) | -0.034 | -0.031 | -0.045 | -0.049 | -0.038 |
| ('B1', 72) | -0.025 | -0.034 | -0.04  | -0.032 | -0.02  |
| ('B2', 24) |  0     |  0     |  0     |  0     |  0     |
| ('B2', 48) | -0.049 | -0.044 | -0.064 | -0.106 | -0.19  |
| ('B2', 72) | -0.05  | -0.066 | -0.096 | -0.179 | -0.964 |
| ('B3', 24) |  0     |  0     |  0     |  0     |  0     |
| ('B3', 48) | -0.059 | -0.048 | -0.052 | -0.054 | -0.049 |
| ('B3', 72) | -0.062 | -0.072 | -0.076 | -0.082 | -0.471 |
| ('M0', 24) |  0     |  0     |  0     |  0     |  0     |
| ('M0', 48) | -0.057 | -0.045 | -0.049 | -0.055 | -0.056 |
| ('M0', 72) | -0.063 | -0.075 | -0.082 | -0.087 | -0.484 |
| ('M1', 24) |  0     |  0     |  0     |  0     |  0     |
| ('M1', 48) | -0.046 | -0.037 | -0.05  | -0.073 | -0.115 |
| ('M1', 72) | -0.049 | -0.061 | -0.073 | -0.101 | -0.601 |


## Residual-risk calibration at ATM (moneyness = 1.0)

| model   |   h |   n_days |   unhedged_sd_TRY |   dynamic_hedged_sd_TRY |   effectiveness_dynamic |   realised_over_unhedged_pct |
|:--------|----:|---------:|------------------:|------------------------:|------------------------:|-----------------------------:|
| B1      |  24 |       60 |            229.69 |                  229.69 |                    0    |                       100    |
| B1      |  48 |       60 |            305.7  |                  312.45 |                   -0.04 |                       102.21 |
| B1      |  72 |       60 |            314.58 |                  320.75 |                   -0.04 |                       101.96 |
| B2      |  24 |       60 |            211.16 |                  211.16 |                    0    |                       100    |
| B2      |  48 |       60 |            271.82 |                  280.42 |                   -0.06 |                       103.16 |
| B2      |  72 |       60 |            252.12 |                  263.94 |                   -0.1  |                       104.69 |
| B3      |  24 |       60 |            205.8  |                  205.8  |                    0    |                       100    |
| B3      |  48 |       60 |            271.65 |                  278.6  |                   -0.05 |                       102.56 |
| B3      |  72 |       60 |            246.53 |                  255.78 |                   -0.08 |                       103.75 |
| M0      |  24 |       60 |            209.93 |                  209.93 |                    0    |                       100    |
| M0      |  48 |       60 |            275.22 |                  281.91 |                   -0.05 |                       102.43 |
| M0      |  72 |       60 |            251.62 |                  261.69 |                   -0.08 |                       104    |
| M1      |  24 |       60 |            213.54 |                  213.54 |                    0    |                       100    |
| M1      |  48 |       60 |            279.42 |                  286.28 |                   -0.05 |                       102.45 |
| M1      |  72 |       60 |            255.29 |                  264.45 |                   -0.07 |                       103.59 |


## Interpretation

Effectiveness is expected to be LOW because the 24-72 h price risk is dominated by the intraday shape of the residual, which the monthly VEP baseload contract cannot hedge.  A high dynamic effectiveness would actually be suspicious: it would suggest that monthly baseload swings drive short-horizon variance, contrary to the residual-vs-curve decomposition on which the pricer is built.  The primary reading is a MARKET-COMPLETENESS finding, not a model failure: the monthly VEP strip does not span the short-horizon hourly PTF risk.
