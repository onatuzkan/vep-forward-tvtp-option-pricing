# FW12 §2 -- Time-step convergence

Spatial grid fixed at 2401 nodes; climatology z path; boundary and other settings as production.  Sweep multiplies the base steps/hour by [1, 2, 4, 8, 16], so k=2 reproduces the shipped default and k=16 gives an 8x refinement of the time grid.

|   strike |   maturity_h |   n_space_nodes |   n_time_steps |   steps_per_hour |   V_call |   residual_sd_T |
|---------:|-------------:|----------------:|---------------:|-----------------:|---------:|----------------:|
|     3000 |           24 |            2401 |             96 |          4       |  163.893 |         524.08  |
|     3000 |           24 |            2401 |             96 |          4       |  163.893 |         524.08  |
|     3000 |           24 |            2401 |             96 |          4       |  163.893 |         524.08  |
|     3000 |           24 |            2401 |            192 |          8       |  163.894 |         524.1   |
|     3000 |           24 |            2401 |            384 |         16       |  163.897 |         524.113 |
|     3000 |           48 |            2401 |             96 |          2       |  167.076 |         532.166 |
|     3000 |           48 |            2401 |             96 |          2       |  167.076 |         532.166 |
|     3000 |           48 |            2401 |            192 |          4       |  167.059 |         532.199 |
|     3000 |           48 |            2401 |            384 |          8       |  167.06  |         532.227 |
|     3000 |           48 |            2401 |            768 |         16       |  167.065 |         532.244 |
|     3000 |           72 |            2401 |             96 |          1.33333 |  166.832 |         532.626 |
|     3000 |           72 |            2401 |            144 |          2       |  166.707 |         532.257 |
|     3000 |           72 |            2401 |            288 |          4       |  166.689 |         532.29  |
|     3000 |           72 |            2401 |            576 |          8       |  166.691 |         532.318 |
|     3000 |           72 |            2401 |           1152 |         16       |  166.696 |         532.336 |


## Observed time order per maturity

| T (h) | V_k1 | V_k2 | V_k4 | V_k8 | V_k16 | p_k2->k8 |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 163.89253 | 163.89253 | 163.89253 | 163.89357 | 163.89691 | nan |
| 48 | 167.07620 | 167.07620 | 167.05856 | 167.06047 | 167.06479 | 3.210 |
| 72 | 166.83242 | 166.70702 | 166.68946 | 166.69142 | 166.69577 | 3.167 |


Crank-Nicolson time discretisation is theoretically O(k^2). If the observed order deviates materially from 2, the space error is likely masking the time error (§1 residual dominates).
