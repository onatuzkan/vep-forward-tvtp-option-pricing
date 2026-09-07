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

print()
print("=== TVTP PATH TEST ===")
print("z_lagged min/max :", scenario.z_lagged.min(),
      scenario.z_lagged.max())

print("q01 min/max      :", q01.min(), q01.max())
print("q10 min/max      :", q10.min(), q10.max())

print()
print("constant z=0 price :", constant.value)
print("TVTP path price     :", tvtp.value)
print("difference          :", tvtp.value - constant.value)

print()
print("constant residual sd:", constant.residual_std_at_expiry)
print("TVTP residual sd    :", tvtp.residual_std_at_expiry)

print()
print("constant stress P(T):",
      constant.diagnostics["p_stress_at_expiry"])
print("TVTP stress P(T)    :",
      tvtp.diagnostics["p_stress_at_expiry"])