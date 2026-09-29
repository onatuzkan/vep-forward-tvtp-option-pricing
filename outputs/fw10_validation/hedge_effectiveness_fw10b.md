# FW10b Part B -- Delta hedge on the production forward curve

## Method

* Hedge ratio in units of the delivery-month VEP contract = delta_C wrt F(target) times dF(target)/dF_M, both measured on day d.  dF(target)/dF_M is obtained by bumping the delivery-month monthly quote by +10 TRY/MWh and rebuilding the same smooth+HPFC curve; this drops the FW10 pass's implicit assumption that the monthly move passes through 1-for-1 to the delivery hour.

* Dynamic hedge: at each 24 h step the forward curve, F(target) sample and dF(target)/dF_M are all rebuilt on the new day's VEP quotation.

## dF(target)/dF_M averages, by horizon and moneyness (model-averaged)

|   h |    0.8 |    0.9 |    1.0 |    1.1 |    1.2 |
|----:|-------:|-------:|-------:|-------:|-------:|
|   6 | 0.0852 | 0.0852 | 0.0852 | 0.0852 | 0.0852 |
|  12 | 0.1016 | 0.1016 | 0.1016 | 0.1016 | 0.1016 |
|  24 | 0.1754 | 0.1754 | 0.1754 | 0.1754 | 0.1754 |
|  48 | 0.2673 | 0.2673 | 0.2673 | 0.2673 | 0.2673 |
|  72 | 0.3477 | 0.3477 | 0.3477 | 0.3477 | 0.3477 |


## Effectiveness (dynamic rebalance)

|            |   0.8 |   0.9 |   1.0 |   1.1 |   1.2 |
|:-----------|------:|------:|------:|------:|------:|
| ('B1', 6)  |     0 |     0 |     0 |     0 |     0 |
| ('B1', 12) |     0 |     0 |     0 |     0 |     0 |
| ('B1', 24) |     0 |     0 |     0 |     0 |     0 |
| ('B1', 48) |     0 |     0 |     0 |     0 |     0 |
| ('B1', 72) |     0 |     0 |     0 |     0 |     0 |
| ('B2', 6)  |     0 |     0 |     0 |     0 |     0 |
| ('B2', 12) |     0 |     0 |     0 |     0 |     0 |
| ('B2', 24) |     0 |     0 |     0 |     0 |     0 |
| ('B2', 48) |     0 |     0 |     0 |     0 |     0 |
| ('B2', 72) |     0 |     0 |     0 |     0 |     0 |
| ('B3', 6)  |     0 |     0 |     0 |     0 |     0 |
| ('B3', 12) |     0 |     0 |     0 |     0 |     0 |
| ('B3', 24) |     0 |     0 |     0 |     0 |     0 |
| ('B3', 48) |     0 |     0 |     0 |     0 |     0 |
| ('B3', 72) |     0 |     0 |     0 |     0 |     0 |
| ('M0', 6)  |     0 |     0 |     0 |     0 |     0 |
| ('M0', 12) |     0 |     0 |     0 |     0 |     0 |
| ('M0', 24) |     0 |     0 |     0 |     0 |     0 |
| ('M0', 48) |     0 |     0 |     0 |     0 |     0 |
| ('M0', 72) |     0 |     0 |     0 |     0 |     0 |
| ('M1', 6)  |     0 |     0 |     0 |     0 |     0 |
| ('M1', 12) |     0 |     0 |     0 |     0 |     0 |
| ('M1', 24) |     0 |     0 |     0 |     0 |     0 |
| ('M1', 48) |     0 |     0 |     0 |     0 |     0 |
| ('M1', 72) |     0 |     0 |     0 |     0 |     0 |


## Static vs dynamic identity check

|   h |   identical_static_dynamic |
|----:|---------------------------:|
|   6 |                          1 |
|  12 |                          1 |
|  24 |                          1 |
|  48 |                          1 |
|  72 |                          1 |


## Interpretation

The measured `dF(target)/dF_M` at 6-72 h horizons is 0.04-0.22 for the near-term month (the delivery month already in progress or the first future month).  The monthly VEP baseload contract is therefore a very weak intra-day pass-through for hourly PTF at short horizons -- the delivery-hour move is dominated by the spot anchor, not the monthly quote.

**Second finding: VEP GGF settlement quotes are effectively STALE at the daily frequency.**  Across the full 2022-2026 quotation calendar the fraction of daily (contract, day) observations with a nonzero move in the settlement price is about 1 pct (14 of 1354 for the 2026-delivery contracts).  When F_M does not move between valuation day d and the horizon end, the hedge P&L is exactly zero regardless of the hedge ratio, and Var(hedged) = Var(unhedged); effectiveness is definitionally zero.  This sharpens the market-completeness finding: VEP publishes an administratively smoothed daily reference price that carries no exploitable information about the delivery-hour PTF.
