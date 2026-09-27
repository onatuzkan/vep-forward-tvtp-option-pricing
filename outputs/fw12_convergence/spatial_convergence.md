# FW12 §1 -- Spatial convergence sweep

Same climatology z(t-1) path as the production `run_pde.py
price` command (train_end 2022-12-31 20:00 UTC, lag 1 h),
same `n_std = 6.0`, same discount `r_annual = 0.40`, and no
measure adjustment.  Only `n_space_nodes` varies.  The
grid halves each step so the observed order `p =
log2(|V_h - V_{h/2}| / |V_{h/2} - V_{h/4}|)` is meaningful.

## Raw values

|   strike |   maturity_h |   n_space_nodes |   n_time_steps |   V_call |   diff_from_prev |   residual_sd_T |   p_stress_T |
|---------:|-------------:|----------------:|---------------:|---------:|-----------------:|----------------:|-------------:|
|     3000 |           24 |             301 |             96 |  164.208 |       nan        |         524.08  |     0.598484 |
|     3000 |           24 |             601 |             96 |  164.004 |        -0.204351 |         524.08  |     0.598484 |
|     3000 |           24 |            1201 |             96 |  163.926 |        -0.077761 |         524.08  |     0.598484 |
|     3000 |           24 |            2401 |             96 |  163.893 |        -0.033541 |         524.08  |     0.598484 |
|     3000 |           24 |            4801 |             96 |  163.883 |        -0.009169 |         524.08  |     0.598484 |
|     3000 |           48 |             301 |             96 |  167.431 |       nan        |         532.166 |     0.600819 |
|     3000 |           48 |             601 |             96 |  167.211 |        -0.220191 |         532.166 |     0.600819 |
|     3000 |           48 |            1201 |             96 |  167.117 |        -0.093786 |         532.166 |     0.600819 |
|     3000 |           48 |            2401 |             96 |  167.076 |        -0.040689 |         532.166 |     0.600819 |
|     3000 |           48 |            4801 |             96 |  167.063 |        -0.01282  |         532.166 |     0.600819 |
|     3000 |           72 |             301 |            144 |  167.072 |       nan        |         532.257 |     0.600819 |
|     3000 |           72 |             601 |            144 |  166.842 |        -0.230055 |         532.257 |     0.600819 |
|     3000 |           72 |            1201 |            144 |  166.748 |        -0.093833 |         532.257 |     0.600819 |
|     3000 |           72 |            2401 |            144 |  166.707 |        -0.040678 |         532.257 |     0.600819 |
|     3000 |           72 |            4801 |            144 |  166.694 |        -0.012984 |         532.257 |     0.600819 |


## Order + Richardson estimate per maturity

| K | T (h) | V_301 | V_601 | V_1201 | V_2401 | V_4801 | p_601->1201 | p_1201->2401 | V_star (Richardson from 2401/4801) | |V_1201 - V_star| | rel err at 1201 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 24 | 164.2082 | 164.0038 | 163.9261 | 163.8925 | 163.8834 | 1.394 | 1.213 | 163.8764 | 0.0497 | 0.0303 % |
| 3000 | 48 | 167.4309 | 167.2107 | 167.1169 | 167.0762 | 167.0634 | 1.231 | 1.205 | 167.0536 | 0.0633 | 0.0379 % |
| 3000 | 72 | 167.0716 | 166.8415 | 166.7477 | 166.7070 | 166.6940 | 1.294 | 1.206 | 166.6841 | 0.0636 | 0.0382 % |


Theoretical order for Crank-Nicolson on a uniform residual grid is O(h^2) in space.  The observed order is reported above; departures from 2.0 typically reflect the gamma-zero boundary contribution (first-order in the far field for OTM payoffs), the mixture of an irregular payoff kink at K, and the Crank-Nicolson scheme's O(k^2) time component intermingling with the spatial one at this solver's default time step.  §2 fixes the spatial grid at the finest available level and varies time steps to separate the two.
