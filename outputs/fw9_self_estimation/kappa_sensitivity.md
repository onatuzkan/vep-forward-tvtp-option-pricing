# FW9d §4 -- ATM call value vs kappa (paper figure)

Sweep of the mean-reversion rate kappa; all other parameters at production values; 1201 spatial nodes; climatology z covariate path.  ATM K=3000 call priced at T = 24, 48, 72 h at each kappa.

**Markers** on the curve show the estimates from FW9b (raw asinh, R1), FW9c (TRY monthly-anchor, R2), FW9d (asinh hour-of-week anchor, R3), and the shipped yaml value 0.078394.  None of the FW9-family estimates land on the yaml value; yaml sits **between** the two clusters (R1/R3 ~ 0.017/h in asinh space, R2 ~ 0.21/h in TRY-space).

|   kappa_per_hour |   half_life_hours |   call_T24 |   call_T48 |   call_T72 | marker           |
|-----------------:|------------------:|-----------:|-----------:|-----------:|:-----------------|
|          0.01    |            69.315 |    324.057 |    433.403 |    489.264 |                  |
|          0.0134  |            51.73  |    310.809 |    402.682 |    443.444 |                  |
|          0.01669 |            41.531 |    298.743 |    376.334 |    406.234 | FW9d_R3_S5       |
|          0.01673 |            41.434 |    298.604 |    376.04  |    405.83  | FW9b_R1_asinh_S5 |
|          0.01795 |            38.606 |    294.295 |    367.002 |    393.512 |                  |
|          0.0196  |            35.366 |    288.659 |    355.473 |    378.111 | FW9b_R1_asinh_S0 |
|          0.02406 |            28.812 |    274.196 |    327.305 |    341.974 |                  |
|          0.03224 |            21.502 |    250.478 |    285.316 |    291.738 |                  |
|          0.04319 |            16.047 |    223.569 |    243.385 |    245.278 |                  |
|          0.05788 |            11.976 |    194.531 |    203.837 |    203.926 |                  |
|          0.07755 |             8.938 |    164.999 |    168.336 |    167.972 |                  |
|          0.07839 |             8.842 |    163.926 |    167.117 |    166.748 | yaml             |
|          0.10392 |             6.67  |    136.879 |    137.614 |    137.215 |                  |
|          0.13924 |             4.978 |    111.784 |    111.725 |    111.365 |                  |
|          0.18658 |             3.715 |     90.529 |     90.344 |     90.026 |                  |
|          0.21156 |             3.276 |     82.589 |     82.409 |     82.108 | FW9c_R2_TRY_S0   |
|          0.22463 |             3.086 |     79.036 |     78.861 |     78.568 | FW9c_R2_TRY_S5   |
|          0.25    |             2.773 |     73.056 |     72.89  |     72.61  |                  |


## Reading the curve

* Between 0.01 and 0.25 /h the ATM 72 h call sweeps roughly 489.3 to 72.6 TRY/MWh -- a ~574 % dynamic range.
* The three FW9 estimation branches produce point estimates at ~0.017 (asinh space), ~0.21 (TRY space), and yaml 0.078 (deseasonalised in the M9 handoff).
* The paper's honest conclusion: kappa is not sharply identified from this dataset without knowing the M9 deseasonalisation pipeline; the option price at any strike / maturity is a KNOWN FUNCTION of kappa (this table), and the paper reports the family of prices rather than a single point.


## Files

* `kappa_sensitivity.csv` -- 3-maturity price grid over kappa.
* `figure_F14_kappa_sensitivity.csv` -- same, with log10(kappa) column added for the paper figure (DejaVu Serif 9.5 pt, W=5.5 in, no top/right spines).
* Markers to overlay: FW9b_R1_asinh_S0/S5 (asinh cluster), yaml (0.078394), FW9c_R2_TRY_S0/S5 (TRY cluster), FW9d_R3_S5 (asinh how-climatology cluster).