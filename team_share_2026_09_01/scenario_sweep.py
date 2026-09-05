from pathlib import Path
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd


OFFSETS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

OUTDIR = Path("outputs/scenario_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)


def extract(pattern, text, name):
    match = re.search(pattern, text)
    if match is None:
        raise RuntimeError(f"Could not parse {name} from output")
    return match


rows = []

env = os.environ.copy()
env["PYTHONUTF8"] = "1"

for offset in OFFSETS:
    print(f"Running RD offset = {offset:+.1f} sigma ...")

    cmd = [
        sys.executable,
        "run_pde.py",
        "price",
        "--model", "forward_centered",
        "--curve", "outputs/market_calibration_final",
        "--params", "inputs/historical/m2_frozen_parameters.yaml",
        "--option-type", "call",
        "--strike", "3000",
        "--maturity-hours", "72",
        "--scenario-mode", "climatology",
        "--scenario-history", "inputs/historical/rd_standardized.csv",
        "--scenario-offset", str(offset),
        "--no-mc",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            f"Pricing failed for offset {offset:+.1f}"
        )

    text = result.stdout

    z_match = extract(
        r"TVTP z\(t-1\) range\s*:\s*\[\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\s*\]",
        text,
        "z range",
    )

    f_match = extract(
        r"F\(T\)\s*:\s*([-+]?\d+(?:\.\d+)?)",
        text,
        "F(T)",
    )

    sd_match = extract(
        r"residual sd at expiry\s*:\s*([-+]?\d+(?:\.\d+)?)",
        text,
        "residual sd",
    )

    value_match = extract(
        r"VALUE\s*:\s*([-+]?\d+(?:\.\d+)?)",
        text,
        "option value",
    )

    rows.append(
        {
            "rd_offset_sigma": offset,
            "z_min": float(z_match.group(1)),
            "z_max": float(z_match.group(2)),
            "forward_T_TRY_MWh": float(f_match.group(1)),
            "residual_sd_TRY_MWh": float(sd_match.group(1)),
            "call_value_TRY_MWh": float(value_match.group(1)),
        }
    )


df = pd.DataFrame(rows)

base_value = float(
    df.loc[df["rd_offset_sigma"] == 0.0, "call_value_TRY_MWh"].iloc[0]
)

df["call_change_vs_base_TRY_MWh"] = (
    df["call_value_TRY_MWh"] - base_value
)

df["call_change_vs_base_pct"] = (
    100.0
    * df["call_change_vs_base_TRY_MWh"]
    / base_value
)

csv_path = OUTDIR / "rd_scenario_sweep.csv"
df.to_csv(csv_path, index=False)


plt.figure(figsize=(8, 5))
plt.plot(
    df["rd_offset_sigma"],
    df["call_value_TRY_MWh"],
    marker="o",
)
plt.axvline(0.0, linestyle="--", linewidth=1)
plt.xlabel("Residual-demand scenario offset (standard deviations)")
plt.ylabel("72 h European call value (TRY/MWh)")
plt.title("Option Value Response to Residual-Demand Scenario")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUTDIR / "option_value_vs_rd_offset.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(8, 5))
plt.plot(
    df["rd_offset_sigma"],
    df["residual_sd_TRY_MWh"],
    marker="o",
)
plt.axvline(0.0, linestyle="--", linewidth=1)
plt.xlabel("Residual-demand scenario offset (standard deviations)")
plt.ylabel("Residual standard deviation at expiry (TRY/MWh)")
plt.title("Residual Volatility Response to Residual Demand")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUTDIR / "residual_sd_vs_rd_offset.png",
    dpi=200,
)
plt.close()


print()
print(df.to_string(index=False))
print()
print(f"CSV   : {csv_path}")
print(f"Plot  : {OUTDIR / 'option_value_vs_rd_offset.png'}")
print(f"Plot  : {OUTDIR / 'residual_sd_vs_rd_offset.png'}")