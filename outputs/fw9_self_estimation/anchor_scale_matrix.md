# FW9e §1 -- 2x2 (scale x anchor) kappa matrix

OLS AR(1) on the FW9 window (2019-01-01 -> 2025-12-31 20:00 UTC,
n = 61368) under four anchor definitions x two scales.
Anchor definitions:

* A0: no anchor (raw variable)
* A1: pooled hour-of-week climatology mean subtracted
* A2: within-month mean subtracted
* A3: A2 + intra-month hour-of-week shape (model-faithful)

## kappa /h (2x2 pivot)

| anchor   |    TRY |   asinh |
|:---------|-------:|--------:|
| A0       | 0.0342 |  0.0196 |
| A1       | 0.0283 |  0.0162 |
| A2       | 0.2116 |  0.1538 |
| A3       | 0.2234 |  0.1662 |


## half-life (hours)

| anchor   |    TRY |   asinh |
|:---------|-------:|--------:|
| A0       | 20.296 |  35.367 |
| A1       | 24.468 |  42.713 |
| A2       |  3.276 |   4.506 |
| A3       |  3.102 |   4.171 |


## residual sd (in-scale units)

| anchor   |      TRY |   asinh |
|:---------|---------:|--------:|
| A0       | 1217.83  |  0.9799 |
| A1       | 1191.55  |  0.9627 |
| A2       |  511.075 |  0.3615 |
| A3       |  445.08  |  0.3121 |


## Ratio isolation

* Scale ratio (TRY / asinh) at each anchor:  A0=1.74, A1=1.75, A2=1.38, A3=1.34
* Anchor ratio (A3 / A0) at each scale:  asinh=8.48, TRY=6.54


**The dominant effect is the anchor: subtracting the monthly level pushes kappa 6-10x higher.  The scale change (asinh vs TRY) contributes a much smaller 1.3-1.7x factor.**  This inverts the FW9d attribution.  R1 and R3 both look 'the same' (kappa ~ 0.017/h) because BOTH keep the multi-year 2019-2025 level trend in the residual, so their phi is measuring the persistence of the TRY inflation trend, not the persistence of a price shock around a slow anchor.

* yaml kappa = **0.078394** falls between the A0/A1 (no trend removed) and A2/A3 (trend removed) rows on both scales.