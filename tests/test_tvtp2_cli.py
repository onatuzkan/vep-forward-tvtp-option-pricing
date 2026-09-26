"""CLI selection and reporting of the TVTP mode; protection of accepted outputs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from .conftest import REPO_ROOT

import run_pde                                                        # noqa: E402

CFG2 = "config/forward_centered_tvtp2_experimental.yaml"
ACCEPTED = REPO_ROOT / "outputs" / "market_calibration_final"


def _tree_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _value(out: str) -> float:
    return float(re.search(r"VALUE\s*:\s*([-+]?\d+\.\d+)", out).group(1))


def test_default_mode_is_the_single_covariate_production_model(capsys):
    assert run_pde.main(["price", "--strike", "3000", "--maturity-hours", "72", "--no-mc"]) == 0
    out = capsys.readouterr().out
    assert "TVTP mode                : rd_lag1_1d" in out
    assert "EXPERIMENTAL" not in out
    assert abs(_value(out) - 166.7477) < 1e-4


def test_two_covariate_mode_is_selected_by_its_own_config(capsys):
    rc = run_pde.main(["--config", CFG2, "price", "--strike", "3000",
                       "--maturity-hours", "72", "--no-mc"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rd_ramp_2d_experimental  ** EXPERIMENTAL **" in out
    assert "M9-transferred slopes + reconstructed ramp + derived intercepts, zero " \
           "transition premium" in out
    assert "TVTP r(t-1) range" in out and "ramp scaler" in out
    assert "not a reproduction of M9" in out
    v = _value(out)
    assert abs(v - 166.7477) > 0.5 and 150.0 < v < 180.0


def test_cli_flag_selects_the_mode_too(capsys):
    rc = run_pde.main(["price", "--tvtp-mode", "rd_ramp_2d_experimental", "--strike", "3000",
                       "--maturity-hours", "24", "--no-mc"])
    assert rc == 0
    assert "rd_ramp_2d_experimental" in capsys.readouterr().out


def test_nonembeddable_scenario_is_rejected_with_an_audit(capsys):
    rc = run_pde.main(["--config", CFG2, "price", "--maturity-hours", "720",
                       "--scenario-offset", "-4.5", "--no-mc"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "REJECTED" in err and "p01 + p10 >= 1" in err and "no clipping" in err


def test_experimental_mode_refuses_accepted_output_directories(capsys):
    before = _tree_hash(ACCEPTED)
    rc = run_pde.main(["--config", CFG2, "calibrate-market", "--outdir", str(ACCEPTED),
                       "--no-sensitivity"])
    assert rc == 2
    assert "refusing to write EXPERIMENTAL" in capsys.readouterr().err
    rc = run_pde.main(["--config", CFG2, "diagnostics", "--outdir",
                       "outputs/forward_centered_diagnostics"])
    assert rc == 2
    assert _tree_hash(ACCEPTED) == before


def test_two_covariate_calibration_reports_mode_and_provenance(tmp_path, capsys):
    out = tmp_path / "cal2d"
    rc = run_pde.main(["--config", CFG2, "calibrate-market", "--outdir", str(out),
                       "--no-sensitivity"])
    assert rc == 0
    blob = json.loads((out / "calibration_result.json").read_text(encoding="utf-8"))
    assert blob["calibration_accepted"] is True
    assert blob["tvtp_mode"] == "rd_ramp_2d_experimental"
    prov = blob["tvtp_provenance"]
    assert prov["status"] == "experimental_reconstructed"
    assert prov["verified_reproduction_of_m9"] is False
    assert prov["ramp_scaler"]["window_end_utc"] == "2024-12-31T20:00:00+00:00"
    assert prov["derivation_sample"]["n_valid_transitions"] == 78902
    assert prov["covariate_path"]["mode"] == "climatology"
    ident = json.loads((out / "parameter_identification.json").read_text(encoding="utf-8"))
    tv = ident["2_inherited_from_historical_M2_fit"]["tvtp_coefficients"]
    assert tv["h01"] == -0.06939531482057137 and tv["h10"] == 0.4055443473030015
    assert "RECONSTRUCTED" in ident["3_fixed_by_assumption"]["tvtp_ramp_covariate_definition"]["status"]
    cfg = yaml.safe_load((out / "calibrated_config.yaml").read_text(encoding="utf-8"))
    assert cfg["tvtp"]["mode"] == "rd_ramp_2d_experimental"
    assert (out / "tvtp_provenance.json").exists()
    assert "EXPERIMENTAL two-covariate TVTP" in (out / "model_limitations.md").read_text(
        encoding="utf-8")


def test_single_covariate_calibration_also_reports_its_mode(tmp_path):
    out = tmp_path / "cal1d"
    assert run_pde.main(["calibrate-market", "--outdir", str(out), "--no-sensitivity"]) == 0
    blob = json.loads((out / "calibration_result.json").read_text(encoding="utf-8"))
    assert blob["tvtp_mode"] == "rd_lag1_1d"
    assert blob["tvtp_provenance"]["covariates"] == ["RD_lag1"]


def test_two_covariate_validate_passes(capsys):
    assert run_pde.main(["--config", CFG2, "validate"]) == 0
    out = capsys.readouterr().out
    for name in ("tvtp2_parameters_are_labelled_experimental",
                 "tvtp2_ramp_scaler_reproduces_from_history",
                 "tvtp2_intercepts_reproduce_from_history",
                 "tvtp2_zero_ramp_reproduces_single_covariate",
                 "tvtp2_historical_path_is_embeddable",
                 "tvtp2_scenario_path_is_embeddable",
                 "tvtp2_residual_expectation_is_centred"):
        assert f"[PASS] {name}" in out
    assert "FAIL" not in out


def test_config_extends_inherits_everything_but_the_tvtp_block():
    base = run_pde._load_config(None)
    exp = run_pde._load_config(CFG2)
    assert exp["tvtp"]["mode"] == "rd_ramp_2d_experimental"
    for key in ("market", "residual", "contract", "grid", "acceptance", "scenario"):
        assert exp[key] == base[key], key
