"""Produce PDE vs Monte Carlo verification data across strikes.

Reuses the repository's own CLI plumbing so the numbers are the model's,
not a reimplementation.

Run from anywhere; the script cd's into the repository root (two levels up
from this file) so all the relative paths inside `run_pde` resolve
correctly.
"""
import sys, json, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
sys.path.insert(0, str(REPO_ROOT))
import numpy as np
import pandas as pd
import run_pde as R
from pde_option_model.forward_centered import (
    price_forward_centered, simulate_forward_centered)
from pde_option_model.contracts import EuropeanOption


class A:  # stand-in for argparse namespace
    config = "config/forward_centered_config.yaml"
    curve = "outputs/market_calibration_final"
    params = "inputs/historical/m2_frozen_parameters.yaml"
    quotes = None
    january_anchor_mode = None
    january_anchor_level = None
    curve_mode = None
    pi_override = "filtered"
    scenario_mode = "climatology"
    scenario_history = "inputs/historical/rd_standardized.csv"
    risk_premium_a0 = 0.0
    risk_premium_a1 = 0.0


os.chdir(REPO_ROOT)
args = A()
cfg = R._load_config(args.config)
quotes, params, _, _ = R._load_market_inputs(args, cfg)
anchor = R._build_anchor(cfg, None, None)
curve_mode = R._get(cfg, "market.curve_mode", "smooth_constrained")
model = R._build_model(quotes, params, cfg, anchor, curve_mode)
gs = R._grid_settings(cfg)

rows = []
for K in [2000, 2400, 2800, 3000, 3200, 3600, 4000]:
    val = params.valuation_utc
    opt = EuropeanOption(option_type="call", strike=float(K),
                         valuation_utc=val,
                         maturity_utc=val + pd.Timedelta(hours=72),
                         r_annual=float(R._get(cfg, "contract.r_annual", 0.40)))
    zfn = R._build_tvtp_scenario(opt, gs, cfg, args)
    if isinstance(zfn, tuple): zfn = zfn[1]
    pde = price_forward_centered(model, opt, gs, z_lagged_fn=zfn)
    mc = simulate_forward_centered(model, opt, n_paths=40000, dt_hours=0.05,
                                   z_lagged_fn=zfn)
    rows.append(dict(strike=K, pde=float(pde.value),
                     mc=float(mc["value"]), mc_se=float(mc["std_error"])))
    print(rows[-1], flush=True)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
with open(FIGURES_DIR / "pde_mc.json", "w") as fh:
    json.dump(rows, fh, indent=1)
print("saved")
