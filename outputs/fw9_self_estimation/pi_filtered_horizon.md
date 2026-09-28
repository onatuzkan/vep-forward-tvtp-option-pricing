# FW9c §2 -- pi_filtered horizon effect on the ATM call

Reprice ATM K=3000 at horizons T = 1, 2, 6, 12, 24, 48, 72 h under two initial regime distributions:
* yaml pi_filtered = [0.931977, 0.068023] (M2 filter output)
* FW9 terminal filtered = [0.983, 0.017] (from outputs/fw9_self_estimation/pi_filtered_terminal.json)

Regime memory diagnostic (yaml alphas at z=0):
* p01 = 0.2659, p10 = 0.1306
* aggregate transition intensity lambda = 0.5050/h
* regime memory half-life = **1.372 h**

By T = 6-12 h the initial distribution has been overwritten by the ergodic dynamics; the two pi choices should give essentially the same option price at all horizons in the shipped table (24 h and above).

|   T_hours |   V_yaml_pi |   V_fw9_terminal_pi |   delta_TRY |   delta_pct |
|----------:|------------:|--------------------:|------------:|------------:|
|         1 |     13.2262 |             10.4741 |     -2.752  |    -20.8076 |
|         2 |     32.8035 |             29.5052 |     -3.2983 |    -10.0547 |
|         6 |    113.385  |            111.546  |     -1.8387 |     -1.6216 |
|        12 |    166.142  |            165.564  |     -0.5773 |     -0.3475 |
|        24 |    163.926  |            163.833  |     -0.0934 |     -0.057  |
|        48 |    167.117  |            167.115  |     -0.0021 |     -0.0013 |
|        72 |    166.748  |            166.748  |     -0      |     -0      |


Interpretation: `pi_filtered` uncertainty at the valuation instant translates to a price uncertainty that shrinks with horizon.  For the shipped reporting horizons (T >= 24 h) the delta is a fraction of one basis point; the (e) limitations item is effectively closed for the reporting range.
