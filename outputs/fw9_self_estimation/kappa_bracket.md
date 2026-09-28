# FW9c §1 -- Kappa bracket by seasonal-model richness

Ladder from S0 (no seasonality removed) to S5 (hour x dow + month-of-year + annual/semi-annual Fourier).  For each spec: project residual = P - F(t) on the seasonal design, fit AR(1) on the projection residual (OLS), report kappa and the ATM K=3000 call price with that kappa (all other parameters at production, climatology z, 1201 nodes).

The **YAML** row shows the shipped `kappa_per_hour = 0.078394` for reference.

| spec   |   seasonal_dof |   seasonal_R2 |   ar1_phi |   ar1_se_phi |   ar1_kappa_per_hour |   ar1_half_life_hours |   ar1_sigma_eps |   atm_call_T24 |   atm_call_T48 |   atm_call_T72 |
|:-------|---------------:|--------------:|----------:|-------------:|---------------------:|----------------------:|----------------:|---------------:|---------------:|---------------:|
| S0     |              0 |        0      |  0.809321 |     0.002371 |             0.21156  |                 3.276 |         300.193 |         82.589 |         82.409 |         82.108 |
| S1     |             23 |        0.1362 |  0.814679 |     0.002341 |             0.204962 |                 3.382 |         275.461 |         84.527 |         84.345 |         84.039 |
| S2     |             29 |        0.1839 |  0.801905 |     0.002412 |             0.220765 |                 3.14  |         275.852 |         80.05  |         79.873 |         79.578 |
| S3     |             40 |        0.1839 |  0.801904 |     0.002412 |             0.220767 |                 3.14  |         275.852 |         80.05  |         79.873 |         79.578 |
| S4     |             44 |        0.1875 |  0.801035 |     0.002416 |             0.221851 |                 3.124 |         275.778 |         79.762 |         79.586 |         79.292 |
| S5     |            182 |        0.2452 |  0.798809 |     0.002428 |             0.224634 |                 3.086 |         267.118 |         79.036 |         78.861 |         78.568 |
| YAML   |            nan |      nan      |  0.9246   |   nan        |             0.078394 |                 8.842 |         nan     |        163.926 |        167.117 |        166.748 |

## kappa aralığı (S0..S5): [0.20496, 0.22463]  /  yaml = 0.078394
Yaml kappa aralığın **DIŞINDA**.
