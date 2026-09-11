"""End-to-end CLI behaviour, including exit codes on rejection."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from .conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
import run_pde                                                       # noqa: E402


def test_validate_exits_zero(capsys):
    assert run_pde.main(["validate"]) == 0
    out = capsys.readouterr().out
    assert "checks passed" in out
    assert "february_2026_has_672_delivery_hours" in out
    assert "FAIL" not in out


def test_calibrate_market_writes_every_required_file(tmp_path, capsys):
    outdir = tmp_path / "cal"
    rc = run_pde.main(["calibrate-market", "--quotes",
                       str(REPO_ROOT / "inputs/market/vep_monthly_quotes.csv"),
                       "--model", "forward_centered",
                       "--outdir", str(outdir), "--no-sensitivity"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "calibration_accepted     : True" in out
    assert "VEP-forward-curve anchored option prices" in out
    for f in ("calibration_result.json", "calibration_audit.md",
              "hourly_forward_curve.csv", "monthly_forward_fit.csv",
              "monthly_forward_fit.png", "legacy_vs_forward_centered.png",
              "calibrated_config.yaml", "parameter_identification.json",
              "model_limitations.md"):
        assert (outdir / f).exists(), f


def test_calibrate_market_rejects_legacy_mode(capsys):
    rc = run_pde.main(["calibrate-market", "--model", "legacy_asinh_ou",
                       "--outdir", "outputs/should_not_exist"])
    assert rc == 2
    assert "benchmark-only" in capsys.readouterr().err


def test_price_runs_and_reports_the_right_label(capsys):
    rc = run_pde.main(["price", "--model", "forward_centered",
                       "--strike", "3000", "--maturity-hours", "72", "--no-mc"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VEP-forward-curve anchored option prices" in out
    assert "EXPIRY-HOUR spot PTF" in out
    assert "NOT a monthly baseload option" in out
    assert "VALUE" in out


def test_price_refuses_a_rejected_calibration(tmp_path, capsys):
    d = tmp_path / "rejected"
    d.mkdir()
    (d / "calibration_result.json").write_text(json.dumps({
        "calibration_accepted": False, "optimizer_success": True,
        "acceptance_checks": [{"check": "monthly_delivery_average_matches_quote",
                               "passed": False, "detail": "RMSE 2.5e6"}]}),
        encoding="utf-8")
    rc = run_pde.main(["price", "--model", "forward_centered", "--curve", str(d),
                       "--no-mc"])
    assert rc == 1
    assert "calibration_accepted=false" in capsys.readouterr().err


def test_price_accepts_a_good_calibration(tmp_path, capsys):
    outdir = tmp_path / "cal_ok"
    assert run_pde.main(["calibrate-market", "--outdir", str(outdir),
                         "--no-sensitivity"]) == 0
    capsys.readouterr()
    rc = run_pde.main(["price", "--model", "forward_centered",
                       "--curve", str(outdir / "hourly_forward_curve.csv"),
                       "--no-mc"])
    assert rc == 0
    assert "using accepted calibration" in capsys.readouterr().out


def test_price_missing_calibration_directory(tmp_path, capsys):
    rc = run_pde.main(["price", "--curve", str(tmp_path / "nothing"), "--no-mc"])
    assert rc == 2


def test_legacy_pricing_reports_the_missing_artefacts(capsys):
    rc = run_pde.main(["price", "--model", "legacy_asinh_ou", "--no-mc"])
    assert rc == 2
    assert "historical artefact bundle" in capsys.readouterr().err


def test_diagnostics_runs(tmp_path, capsys):
    rc = run_pde.main(["diagnostics", "--model", "forward_centered",
                       "--outdir", str(tmp_path / "diag")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "parity error" in out
    for f in ("residual_diagnostics.csv", "residual_diagnostics.png",
              "strike_profile.csv", "strike_profile.png",
              "legacy_explosion_table.csv"):
        assert (tmp_path / "diag" / f).exists(), f


def test_freeze_params_without_artefacts_exits_two(tmp_path, capsys):
    rc = run_pde.main(["freeze-params", "--input-root", str(tmp_path),
                       "--out", str(tmp_path / "out.yaml")])
    assert rc == 2
    assert "artefact bundle" in capsys.readouterr().err


def test_calibrated_config_is_loadable_yaml(tmp_path):
    outdir = tmp_path / "cfg"
    assert run_pde.main(["calibrate-market", "--outdir", str(outdir),
                         "--no-sensitivity"]) == 0
    cfg = yaml.safe_load((outdir / "calibrated_config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["mode"] == "forward_centered"
    assert cfg["acceptance"]["calibration_accepted"] is True
    assert "Fully market-calibrated" in cfg["_warning"]
    assert cfg["_price_label"] == "VEP-forward-curve anchored option prices"


def test_january_anchor_override_from_cli(tmp_path, capsys):
    outdir = tmp_path / "anchor"
    rc = run_pde.main(["calibrate-market", "--outdir", str(outdir),
                       "--january-anchor-mode", "explicit_level",
                       "--january-anchor-level", "3100", "--no-sensitivity"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "3100.00" in out or "3100.0" in out
    blob = json.loads((outdir / "calibration_result.json").read_text(encoding="utf-8"))
    assert blob["january_calibration_status"]["anchor_mode"] == "explicit_level"
    assert blob["january_calibration_status"]["anchor_level_TRY_MWh"] == 3100.0
    assert blob["calibration_accepted"] is True


def test_piecewise_curve_mode_from_cli(tmp_path):
    outdir = tmp_path / "pwc"
    assert run_pde.main(["calibrate-market", "--outdir", str(outdir),
                         "--curve-mode", "piecewise_constant",
                         "--no-sensitivity"]) == 0
    blob = json.loads((outdir / "calibration_result.json").read_text(encoding="utf-8"))
    assert blob["calibration_accepted"] is True
    assert blob["maximum_absolute_monthly_error"] < 0.10
