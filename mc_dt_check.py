import numpy as np
import pandas as pd

from run_pde import (
    _load_config,
    _build_anchor,
    _build_model,
    _grid_settings,
)

from pde_option_model.market_data import load_quotes
from pde_option_model.params_frozen import load_frozen_parameters
from pde_option_model.contracts import EuropeanOption
from pde_option_model.grid import TimeGrid
from pde_option_model.scenarios import ScenarioBuilder, ScenarioSpec
from pde_option_model.forward_centered import price_forward_centered


cfg = _load_config(None)

quotes = load_quotes("inputs/market/vep_monthly_quotes.csv")
params = load_frozen_parameters(
    "inputs/historical/m2_frozen_parameters.yaml"
)

anchor = _build_anchor(cfg, None, None)

model = _build_model(
    quotes,
    params,
    cfg,
    anchor,
    "smooth_constrained",
)

contract = EuropeanOption(
    option_type="call",
    strike=3000.0,
    valuation_utc=params.valuation_utc,
    maturity_utc=params.valuation_utc + pd.Timedelta(hours=72),
    r_annual=0.40,
)

# ---------------------------------------------------------
# Historical contemporaneous RD standardized with the
# exact RD_lag1 training scaler recovered from the M2 fit.
# ---------------------------------------------------------
hist = pd.read_csv(
    "inputs/historical/rd_standardized.csv"
)

hist["datetime"] = pd.to_datetime(
    hist["datetime"],
    utc=True,
)

z_history = pd.Series(
    hist["z"].to_numpy(),
    index=hist["datetime"],
)

gs = _grid_settings(cfg)

tgrid = TimeGrid(
    contract.valuation_utc,
    contract.maturity_utc,
    gs.n_steps(contract.tau_hours),
)

builder = ScenarioBuilder(
    z_history=z_history,
    train_end=pd.Timestamp(
        "2022-12-31 20:00:00+00:00"
    ),
    covariate_lag_hours=1.0,
)

scenario = builder.build(
    ScenarioSpec(
        name="base",
        mode="climatology",
        offset=0.0,
    ),
    tgrid,
)


def z_lagged_fn(t):
    return np.interp(
        t,
        scenario.times_hours,
        scenario.z_lagged,
    )


# Benchmark: old behaviour, z(t)=0
constant = price_forward_centered(
    model,
    contract,
    gs,
)

# New behaviour: climatological time-varying z(t-1)
tvtp = price_forward_centered(
    model,
    contract,
    gs,
    z_lagged_fn=z_lagged_fn,
)


q01, q10 = model.generator_path(
    scenario.times_hours,
    scenario.z_lagged,
)

from pde_option_model.forward_centered import simulate_forward_centered

print()
print("=== MC TIME-STEP CONVERGENCE ===")
print("PDE value =", tvtp.value)
print()

for dt in [0.25, 0.10, 0.05]:
    mc = simulate_forward_centered(
        model,
        contract,
        n_paths=20000,
        dt_hours=dt,
        seed=20260808,
        z_lagged_fn=z_lagged_fn,
    )

    zscore = abs(tvtp.value - mc["value"]) / mc["std_error"]

    print(
        f"dt={dt:>4.2f} h | "
        f"MC={mc['value']:.4f} | "
        f"SE={mc['std_error']:.4f} | "
        f"|z|={zscore:.3f} | "
        f"mean P_T={mc['mean_price_T']:.2f} | "
        f"P(P_T<0)={mc['prob_negative_price']:.4f}"
    )