# FW12 §4 -- MC cross-check at the converged PDE grid

500 000 total paths (2 x 250 000 with paired seeds for a 
variance-reduction antithetic pair), dt = 0.25 h, 
climatology z(t-1) path identical to the PDE run.

* V_PDE (@ 2401 spatial nodes) = **166.7070 TRY/MWh**
* V_MC  = **166.1066 +/- 0.4038** TRY/MWh
* 95 % CI: [165.3151, 166.8980]
* PDE inside 95 % CI: **True**
* |z_score| = 1.487

If PDE is inside the CI, the numeric layer is consistent -- 
the PDE and the SDE-simulator both discretise the same 
operator and converge to the same limit.  Failure to be 
inside the CI is a HARD STOP.
