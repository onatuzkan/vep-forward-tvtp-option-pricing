"""FW9b §8 -- Hamilton filter terminal filtered distribution.

Runs the Hamilton filter at the FW9 phi-profile-argmax parameters on
the full FW9 estimation window and reports the filtered probability
pi_{T|T} at the valuation instant.  Comparison with:
* yaml pi_filtered = [0.931977, 0.068023] (M2 filter output)
* M9 stationary pi   = [0.325422, 0.674578]

Contributes to a proposed line for model_limitations.md item (e).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.msar_estimation import hamilton_loglik
from scripts.fw9._data import build_fit_frame

YAML_SCALE_P = 282.48
OUT = REPO_ROOT / "outputs" / "fw9_self_estimation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "profile_phi_TVTP_1cov_argmax.pkl"
    with open(p, "rb") as f:
        blob = pickle.load(f)
    res = blob["res"]
    df = build_fit_frame(scale_P=YAML_SCALE_P, currency="TRY")
    y = df["y"].to_numpy()
    z = df["z_lag1"].to_numpy()
    _, filt = hamilton_loglik(res.theta, y, z, return_filtered=True)
    pi_T = filt[-1]
    payload = {
        "fw9_terminal_filtered": {"normal": float(pi_T[0]),
                                  "stress": float(pi_T[1])},
        "yaml_pi_filtered": {"normal": 0.931977, "stress": 0.068023},
        "m9_stationary_pi": {"normal": 0.325422, "stress": 0.674578},
        "terminal_ts_utc": str(df["ts_utc"].iloc[-1]),
        "n_obs": int(len(y)),
    }
    with open(OUT / "pi_filtered_terminal.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
