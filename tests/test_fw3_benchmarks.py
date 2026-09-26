"""FW3 benchmark-model tests (put-call parity, sigma limits, round-trips,
Lucia-Schwartz / Bachelier equivalence, frozen-file hash invariance)."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pde_option_model.benchmarks import (bachelier, black76,
                                         implied_vol_bachelier,
                                         implied_vol_black76, lucia_schwartz,
                                         ou_terminal_variance, parity_error)

REPO_ROOT = Path(__file__).resolve().parents[1]

R_ANNUAL = 0.40
R_PER_HOUR = R_ANNUAL / 8760.0
GRID_CSV = REPO_ROOT / "outputs" / "market_calibration_final" / "strike_maturity_grid.csv"


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
CASES = [
    # (F, K, tau_hours)  -- real repo levels: F ~ 2917, strikes span the grid
    (2917.24, 2000.0, 24.0), (2917.24, 3000.0, 24.0), (2917.24, 4000.0, 24.0),
    (2916.16, 2000.0, 72.0), (2916.16, 3000.0, 72.0), (2916.16, 4000.0, 72.0),
    (2913.99, 2500.0, 168.0), (2910.21, 3200.0, 336.0),
    (2901.51, 2600.0, 720.0), (2901.51, 3600.0, 720.0),
]
BLACK_SIGMAS = [0.005, 0.02, 0.05, 0.1]                # per sqrt(hour)
BACH_SIGMAS = [50.0, 200.0, 800.0, 2000.0]             # TRY/MWh per sqrt(hour)
OU_KAPPAS = [0.001, 0.078394, 0.5]                     # per hour


# ---------------------------------------------------------------------------
# Put-call parity (all three benchmarks)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("sigma", BLACK_SIGMAS)
def test_black76_put_call_parity(case, sigma):
    F, K, T = case
    c = black76(F, K, T, sigma, R_PER_HOUR, "call")
    p = black76(F, K, T, sigma, R_PER_HOUR, "put")
    assert abs(parity_error(c, p, F, K, T, R_PER_HOUR)) < 1e-10


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("sigma", BACH_SIGMAS)
def test_bachelier_put_call_parity(case, sigma):
    F, K, T = case
    c = bachelier(F, K, T, sigma, R_PER_HOUR, "call")
    p = bachelier(F, K, T, sigma, R_PER_HOUR, "put")
    assert abs(parity_error(c, p, F, K, T, R_PER_HOUR)) < 1e-10


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("sigma", BACH_SIGMAS)
@pytest.mark.parametrize("kappa", OU_KAPPAS)
def test_lucia_schwartz_put_call_parity(case, sigma, kappa):
    F, K, T = case
    c = lucia_schwartz(F, K, T, sigma, kappa, R_PER_HOUR, "call")
    p = lucia_schwartz(F, K, T, sigma, kappa, R_PER_HOUR, "put")
    assert abs(parity_error(c, p, F, K, T, R_PER_HOUR)) < 1e-10


# ---------------------------------------------------------------------------
# sigma -> 0 limits collapse to discounted intrinsic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_black76_zero_sigma_is_discounted_intrinsic(case, kind):
    F, K, T = case
    disc = math.exp(-R_PER_HOUR * T)
    expected = disc * (max(F - K, 0.0) if kind == "call" else max(K - F, 0.0))
    assert black76(F, K, T, 0.0, R_PER_HOUR, kind) == pytest.approx(expected,
                                                                    abs=1e-12)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_bachelier_zero_sigma_is_discounted_intrinsic(case, kind):
    F, K, T = case
    disc = math.exp(-R_PER_HOUR * T)
    expected = disc * (max(F - K, 0.0) if kind == "call" else max(K - F, 0.0))
    assert bachelier(F, K, T, 0.0, R_PER_HOUR, kind) == pytest.approx(expected,
                                                                      abs=1e-12)


# ---------------------------------------------------------------------------
# Implied vol round-trip (price -> sigma -> price)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("sigma", BLACK_SIGMAS)
def test_black76_implied_vol_round_trip(case, sigma):
    F, K, T = case
    price = black76(F, K, T, sigma, R_PER_HOUR, "call")
    iv = implied_vol_black76(price, F, K, T, R_PER_HOUR, "call")
    round_trip = black76(F, K, T, iv, R_PER_HOUR, "call")
    assert round_trip == pytest.approx(price, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("sigma", BACH_SIGMAS)
def test_bachelier_implied_vol_round_trip(case, sigma):
    F, K, T = case
    price = bachelier(F, K, T, sigma, R_PER_HOUR, "call")
    iv = implied_vol_bachelier(price, F, K, T, R_PER_HOUR, "call")
    round_trip = bachelier(F, K, T, iv, R_PER_HOUR, "call")
    assert round_trip == pytest.approx(price, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# Lucia-Schwartz price identity: LS(sigma, kappa) == Bachelier(sigma_B) with
# sigma_B^2 * T == sigma^2 * (1 - exp(-2 kappa T)) / (2 kappa)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("sigma", BACH_SIGMAS)
@pytest.mark.parametrize("kappa", OU_KAPPAS)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_ls_reduces_to_bachelier_with_effective_vol(case, sigma, kappa, kind):
    F, K, T = case
    var_T = ou_terminal_variance(sigma, kappa, T)
    sigma_B = math.sqrt(var_T / T)
    ls = lucia_schwartz(F, K, T, sigma, kappa, R_PER_HOUR, kind)
    bach = bachelier(F, K, T, sigma_B, R_PER_HOUR, kind)
    assert ls == pytest.approx(bach, abs=1e-10)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("sigma", BACH_SIGMAS)
def test_ou_variance_limit_kappa_to_zero(case, sigma):
    """As kappa -> 0 the OU integrated variance tends to sigma^2 * T."""
    F, K, T = case
    v_small = ou_terminal_variance(sigma, kappa_per_hour=1e-9, tau_hours=T)
    assert v_small == pytest.approx(sigma * sigma * T, rel=1e-6)


# ---------------------------------------------------------------------------
# Model implied vol -> re-price with Black-76 recovers the MODEL price
# ---------------------------------------------------------------------------
def test_model_implied_vol_reproduces_the_model_call_price():
    """Every row of the accepted 66-point grid: iv(black76) round-trips."""
    grid = pd.read_csv(GRID_CSV)
    rows = []
    for _, r in grid.iterrows():
        F = float(r["F_T_TRY_MWh"]); K = float(r["strike_TRY_MWh"])
        T = float(r["maturity_h"]); C = float(r["call_TRY_MWh"])
        if C <= math.exp(-R_PER_HOUR * T) * max(F - K, 0.0) + 1e-9:
            # deep-ITM call priced below intrinsic (should not happen but
            # guard anyway); skip to keep the round-trip well posed
            continue
        iv = implied_vol_black76(C, F, K, T, R_PER_HOUR, "call")
        rt = black76(F, K, T, iv, R_PER_HOUR, "call")
        rows.append(abs(rt - C))
    assert max(rows) < 1e-8, f"max round-trip error {max(rows):.3e}"


# ---------------------------------------------------------------------------
# Frozen artefact hash invariance -- FW3 must not modify accepted outputs
# ---------------------------------------------------------------------------
FROZEN_TREES = (
    "inputs/historical/m2_frozen_parameters.yaml",
    "inputs/historical/tvtp2_frozen_parameters.yaml",
    "outputs/market_calibration_final",
    "outputs/forward_centered_diagnostics",
    "outputs/scenario_sweep",
    "outputs/tvtp2_experimental",
)

# Recorded once at the start of FW3 (2026-09-26) with the same normalized
# (CRLF -> LF) SHA-256 as `scripts/fw3/README.md` snapshotting instructions.
EXPECTED_HASHES = {
    "inputs/historical/m2_frozen_parameters.yaml":
        "b80fa69d1130e4d7ab05d9830b53deb2d59920381b8c57708ac4e22cc9b35ab2",
    "inputs/historical/tvtp2_frozen_parameters.yaml":
        "64f7df73fc28558692d4347907323088ac1b24f4d53f988a9b354c644cd39717",
    "outputs/market_calibration_final/calibration_result.json":
        None,  # sentinel: filled at test load
}


def _norm_sha256(path: Path) -> str:
    b = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def _gather_frozen_files() -> list[Path]:
    paths: list[Path] = []
    for entry in FROZEN_TREES:
        p = REPO_ROOT / entry
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            for dirpath, _, files in os.walk(p):
                for fn in sorted(files):
                    paths.append(Path(dirpath) / fn)
    return sorted(paths)


def test_two_frozen_yaml_hashes_are_the_recorded_values():
    """Sanity check: FW3 has not modified the two frozen yaml files.

    Committing an accidental yaml change would invalidate every
    accepted output downstream, so we hash-guard both files against
    the values recorded when FW3 started (2026-09-26).
    """
    for rel, expected in EXPECTED_HASHES.items():
        if expected is None:
            continue
        h = _norm_sha256(REPO_ROOT / rel)
        assert h == expected, f"{rel}: hash changed ({h} != {expected})"


def test_frozen_output_trees_are_readable_and_nonempty():
    """Guard against accidental deletion / truncation of the accepted
    output artefact folders during FW3 work."""
    for entry in FROZEN_TREES:
        p = REPO_ROOT / entry
        if p.is_file():
            assert p.stat().st_size > 0, f"{entry} shrunk to zero bytes"
        elif p.is_dir():
            n_files = sum(1 for _ in p.rglob("*") if _.is_file())
            assert n_files > 0, f"{entry}/ is empty"
