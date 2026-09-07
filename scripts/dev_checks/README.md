# Development check scripts

One-shot verification scripts kept as reference for the pipeline changes they
prompted.  Their functionality is now covered by the production CLI:

| script | purpose | production replacement |
|---|---|---|
| `mc_dt_check.py` | Monte Carlo time-step convergence sweep at `dt = 0.25 / 0.10 / 0.05` h | `run_pde.py price --mc-dt-hours <dt>` |
| `test_tvtp_path.py` | Effect of the deterministic TVTP `z(t-1)` path on option value vs the constant-z fallback | `run_pde.py diagnostics` (residual + strike profile CSV/PNG use the same climatology z path) |

These scripts are not part of the automated test suite.  They are safe to run
against the current `inputs/historical/m2_frozen_parameters.yaml` but do not
gate any acceptance check.  Delete them once the design decisions they
documented (dt = 0.05 h; production climatology z path) are no longer
questioned.
