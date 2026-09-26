"""Derive the EXPERIMENTAL two-covariate TVTP parameter set (FW4).

    python scripts/tvtp_derivation/derive_tvtp2_parameters.py            # dry run: audit only
    python scripts/tvtp_derivation/derive_tvtp2_parameters.py --write    # also write the YAML

Inputs (repository data only):
  * inputs/historical/rd_standardized.csv          hourly z (TRY-window standardization)
  * inputs/historical/rd_lag1_standardized.csv     used only as lag-convention evidence
  * inputs/historical/m2_frozen_parameters.yaml    RD slopes and valuation time (read-only)
  * inputs/historical/archive/calibration_bundle/  raw M9 slopes, AMEs, durations

Outputs:
  * inputs/historical/tvtp2_frozen_parameters.yaml             (with --write)
  * outputs/tvtp2_experimental/derivation/tvtp2_derivation_audit.json / .md

The single-covariate ``m2_frozen_parameters.yaml`` is never written.  An existing
two-covariate YAML with different content is only replaced with ``--force``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model import tvtp2 as T                                   # noqa: E402
from pde_option_model.params_frozen import load_tvtp2_parameters         # noqa: E402

HEADER = """\
# =============================================================================
# EXPERIMENTAL two-covariate TVTP parameters (RD_lag1 + RD_Ramp_1h_lag1)  [FW4]
# =============================================================================
# Label: {label}
#
# * Transition SLOPES are transferred from the M9 TVTP-2 fit (regime-label
#   swapped: raw state 0 = high vol -> production index 1 = stress).
# * The RAMP COVARIATE is RECONSTRUCTED: its original definition, lag and
#   standardization were not found in any accessible source.  Here
#       r_t = (dz_t - m_r)/s_r,  dz_t = z_t - z_(t-1h) on the complete UTC grid,
#   with (m_r, s_r) fitted on the training window recorded below.
# * INTERCEPTS are DERIVED by moment matching to the M9 mean durations; they
#   are not estimates and carry no standard error.
# * Transition premium is ZERO by assumption (q^Q = q^P); not calibrated.
#
# This file does NOT reproduce M9 and does NOT replace any accepted result.
# kappa, sigma, pi, spot and valuation time come from the base file named in
# `base_parameters` (inputs/historical/m2_frozen_parameters.yaml, unchanged).
#
# Regenerate:  python scripts/tvtp_derivation/derive_tvtp2_parameters.py --write
# Audit:       outputs/tvtp2_experimental/derivation/tvtp2_derivation_audit.md
# =============================================================================
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the frozen YAML")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing two-covariate YAML with different content")
    ap.add_argument("--out", default=T.DEFAULT_TVTP2_YAML)
    ap.add_argument("--audit-dir", default=T.DEFAULT_AUDIT_DIR)
    ap.add_argument("--ramp-window", default="W9", choices=["W9", "W_T", "all"])
    ap.add_argument("--sample-window", default="W9", choices=["W9", "W_T", "all"])
    args = ap.parse_args(argv)

    out = (REPO_ROOT / args.out).resolve()
    if out.name == "m2_frozen_parameters.yaml" or out == (REPO_ROOT / T.DEFAULT_BASE_PARAMS).resolve():
        print("ERROR: refusing to write over the single-covariate frozen parameter file",
              file=sys.stderr)
        return 2

    blob, audit = T.run_tvtp2_derivation(REPO_ROOT, ramp_window=args.ramp_window,
                                         sample_window=args.sample_window)
    adir = REPO_ROOT / args.audit_dir
    adir.mkdir(parents=True, exist_ok=True)
    with open(adir / "tvtp2_derivation_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False, default=str)
    (adir / "tvtp2_derivation_audit.md").write_text(T.render_derivation_markdown(audit),
                                                     encoding="utf-8")
    text = HEADER.format(label=T.EXPERIMENTAL_LABEL) + yaml.safe_dump(
        blob, sort_keys=False, allow_unicode=True, width=100)

    c = blob["tvtp2"]
    s = blob["covariates"]["ramp"]["scaler"]
    d = blob["derivation"]["sample"]
    print(f"2D TVTP ({blob['status']}): {blob['label']}")
    print(f"  alpha01 = {c['alpha01']:+.10f}   gamma01 = {c['gamma01']:+.10f}   h01 = {c['h01']:+.17f}")
    print(f"  alpha10 = {c['alpha10']:+.10f}   gamma10 = {c['gamma10']:+.10f}   h10 = {c['h10']:+.17f}")
    print(f"  ramp scaler {s['window_name']}: m_r = {s['mean']:.6e}, s_r = {s['std']:.6f}, "
          f"n = {s['n']}, ddof = {s['ddof']}, labels <= {s['window_end_utc']}")
    print(f"  D: {d['first_valid_transition_utc']} -> {d['last_valid_transition_utc']}, "
          f"n = {d['n_valid_transitions']}, dropped start {d['n_dropped_sample_start']}, "
          f"dropped missing {d['n_dropped_missing_observation']}")
    for k in ("p01", "p10"):
        r = blob["derivation"]["roots"][k]
        print(f"  root {k}: residual {r['residual']:.2e}, achieved {r['achieved_mean_p']:.10f} "
              f"(target {r['target_mean_p']:.10f})")
    print(f"  audit: {adir / 'tvtp2_derivation_audit.md'}")

    if not args.write:
        print("  (dry run: YAML not written; pass --write)")
        return 0
    if out.exists():
        if out.read_text(encoding="utf-8") == text:
            print(f"  {out.relative_to(REPO_ROOT)} already up to date")
            return 0
        if not args.force:
            print(f"ERROR: {out} exists with different content; use --force to replace it",
                  file=sys.stderr)
            return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    load_tvtp2_parameters(out)                     # validate what was written
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
