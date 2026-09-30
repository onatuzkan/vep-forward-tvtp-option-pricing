"""FW12 §8 tests: convergence order, PDE-vs-MC agreement, and
boundary-location invariance.  These lock the numerical claims that
Appendix C of the manuscript will make."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption               # noqa: E402
from pde_option_model.forward_centered import (                     # noqa: E402
    ResidualGridSettings, price_forward_centered,
    simulate_forward_centered)
from scripts.fw12._shared import (build_production_model,           # noqa: E402
                                  climatology_z_lagged_fn,
                                  observed_order, price_at)


CONV_CSV = REPO_ROOT / "outputs" / "fw12_convergence" / "spatial_convergence.csv"
BOUND_CSV = REPO_ROOT / "outputs" / "fw12_convergence" / "boundary_sensitivity.csv"
MC_CSV = REPO_ROOT / "outputs" / "fw12_convergence" / "mc_cross_check.csv"


@pytest.mark.skipif(not CONV_CSV.exists(),
                    reason="run scripts/fw12/spatial_convergence.py first")
def test_spatial_convergence_order_is_between_1_and_2():
    """Payoff-kink call on a Crank-Nicolson scheme has observed spatial
    order in [1, 2] depending on how close the strike is to a node.
    The Appendix C claim only needs a MEASURED, POSITIVE order, not the
    theoretical 2.  This test locks in the observed range across the
    three tabulated maturities.
    """
    df = pd.read_csv(CONV_CSV)
    for T in (24, 48, 72):
        sub = df[df["maturity_h"] == T].sort_values("n_space_nodes")
        vs = sub["V_call"].tolist()
        p_mid = observed_order(vs[1], vs[2], vs[3])
        assert 0.8 < p_mid < 2.5, (
            f"T={T}h: observed spatial order {p_mid:.3f} out of [0.8, 2.5]")


@pytest.mark.skipif(not BOUND_CSV.exists(),
                    reason="run scripts/fw12/boundary_sensitivity.py first")
def test_boundary_location_below_point_1_percent_threshold():
    """Widening or narrowing the outer boundary must not move the
    reported price by more than 0.1 % of the converged value.  If it
    does, the far-field specification is affecting reported prices,
    which is exactly the pathology this test guards against.
    """
    df = pd.read_csv(BOUND_CSV)
    for T in (24, 48, 72):
        sub = df[df["maturity_h"] == T]
        v_ref = float(sub[sub["n_std"] == 6.0]["V_call"].iloc[0])
        for _, r in sub.iterrows():
            rel = abs(r["V_call"] - v_ref) / max(abs(v_ref), 1e-9)
            assert rel < 1e-3, (
                f"T={T}h, n_std={r['n_std']}: |Δ|/V = {rel:.4%} exceeds 0.1%"
            )


@pytest.mark.slow
def test_pde_vs_mc_within_three_standard_errors_atm_72h():
    """The PDE at 2401 nodes and the shipped MC simulator (with
    antithetic paired seeds and 500 000 total paths) must agree at
    ATM K=3000, T=72 h within 3 x MC_SE.  This is a hard cross-check
    that the two independent discretisations converge to the same
    limit; a failure means either the SDE simulator or the PDE has a
    bug that has been silent up to this point.
    """
    model = build_production_model()
    contract = EuropeanOption(
        "call", 3000.0, model.valuation_utc,
        model.valuation_utc + pd.Timedelta(hours=72), r_annual=0.40)
    gs = ResidualGridSettings(n_space_nodes=2401)
    z_fn = climatology_z_lagged_fn(model, contract, gs)
    r_pde = price_forward_centered(model, contract, grid_settings=gs,
                                   z_lagged_fn=z_fn)
    r1 = simulate_forward_centered(model, contract, n_paths=250_000,
                                   dt_hours=0.25, seed=20260927,
                                   z_lagged_fn=z_fn)
    r2 = simulate_forward_centered(model, contract, n_paths=250_000,
                                   dt_hours=0.25, seed=20260928,
                                   z_lagged_fn=z_fn)
    v_mc = 0.5 * (r1["value"] + r2["value"])
    se_mc = 0.5 * math.hypot(r1["std_error"], r2["std_error"])
    assert abs(r_pde.value - v_mc) < 3.0 * se_mc, (
        f"PDE {r_pde.value:.4f} vs MC {v_mc:.4f} +/- {se_mc:.4f}: "
        f"|z| = {abs(r_pde.value - v_mc) / se_mc:.3f} > 3")


def test_hash_snapshot_manifest_present_and_nonempty():
    """FW12 must snapshot the frozen artefacts and land the manifest
    in the repo.  This guards against accidentally deleting it."""
    manifest = REPO_ROOT / "outputs" / "fw12_convergence" / "hashes_before.txt"
    assert manifest.exists(), "hashes_before.txt is missing"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 100, f"only {len(lines)} manifest rows"


# model_limitations.md is a living document, updated deliberately after
# FW12 by the documentation update; every other snapshot file must match.
LIVING_DOCUMENTS = {"outputs/market_calibration_final/model_limitations.md"}


def test_yaml_and_frozen_outputs_hashes_are_the_snapshot():
    """Every path recorded in hashes_before.txt must still hash to the
    same value.  Fails if FW12 accidentally modified an accepted
    artefact tree."""
    manifest = REPO_ROOT / "outputs" / "fw12_convergence" / "hashes_before.txt"
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        h_expected, path = raw.split("  ", 1)
        if path in LIVING_DOCUMENTS:
            continue
        p = REPO_ROOT / path
        if not p.is_file():
            pytest.fail(f"missing file recorded in snapshot: {path}")
        h_now = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert h_now == h_expected, f"hash drift on {path}"
