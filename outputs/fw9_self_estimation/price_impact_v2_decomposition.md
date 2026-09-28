# FW9c §4 -- Price-impact decomposition (relabeled)

Two percentage columns per variant, per FW9c §4:
* **price change %** = 100 * delta / A_inherited  (how much the ATM
  call moves in absolute terms)
* **share of total delta %** = 100 * delta / (E - A)  (how much of
  the FW9-vs-production total gap this variant accounts for)

ATM K = 3000, T = 72 h call:

| variant | delta (TRY) | price change % | share of total delta % |
|---|---:|---:|---:|
| B: FW9 sigmas only | +329.80 | +197.78 | +34.06 |
| C: FW9 kappa only | +239.24 | +143.47 | +24.71 |
| D: FW9 transitions only | +3.75 | +2.25 | +0.39 |
| E: full FW9 | +968.37 | +580.74 | +100.00 |

A_inherited = 166.748 TRY/MWh; E_fw9 = 1135.114 TRY/MWh; total delta = 968.367 TRY.

Sigmas and (deseasonalised) kappa each roughly double the ATM call in
isolation; transitions move it by less than 3 TRY (a 2.2% price
change, less than 0.4% of the total FW9-vs-production gap).  The
interaction term (E minus A minus the three isolated deltas) accounts
for the remainder.
