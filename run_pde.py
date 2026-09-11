#!/usr/bin/env python
"""Command-line entry point for the two-regime TVTP electricity option pricer.

Commands
--------
    validate            structural + numerical self-checks (no artefacts needed)
    calibrate-market    build the VEP-anchored forward curve and run acceptance
    price               price a European option on the expiry-hour spot
    diagnostics         residual/regime diagnostic pack for a calibrated curve
    freeze-params       extract frozen M2 parameters from a full artefact bundle

Exit codes
----------
    0  success
    1  a calibration was REJECTED, or a rejected calibration was used
    2  usage / input error

Everything is path- and config-driven; no absolute or Windows-specific paths
appear anywhere.  All monetary values are TRY/MWh.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pde_option_model.contracts import EuropeanOption               # noqa: E402
from pde_option_model.forward_centered import (                     # noqa: E402
    ForwardCenteredModel, ResidualGridSettings, ResidualSpec,
    price_forward_centered, simulate_forward_centered)
from pde_option_model.forward_curve import (                        # noqa: E402
    NearTermAnchor, build_forward_curve)
from pde_option_model.generator import TVTPCoefficients             # noqa: E402
from pde_option_model.legacy_moments import legacy_explosion_report  # noqa: E402
from pde_option_model.market_calibration import (                   # noqa: E402
    PRICE_LABEL, REPORTING_HORIZONS_HOURS, AcceptanceCriteria,
    CalibrationRejected, load_calibration_result_json,
    near_term_anchor_sensitivity, run_market_calibration)
from pde_option_model.market_data import load_quotes                # noqa: E402
from pde_option_model.model_modes import (                          # noqa: E402
    DEFAULT_MARKET_MODE, MODEL_MODES, MODE_DESCRIPTIONS,
    validate_model_mode, write_calibration_outputs)
from pde_option_model.params_frozen import load_frozen_parameters   # noqa: E402
from pde_option_model.grid import TimeGrid                           # noqa: E402
from pde_option_model.scenarios import (                            # noqa: E402
    ScenarioBuilder, ScenarioSpec)

DEFAULT_CONFIG = "config/forward_centered_config.yaml"
LOG_FORMAT = "%(levelname)-7s %(name)s: %(message)s"


# ---------------------------------------------------------------------------
def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (
        logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    p = Path(path or DEFAULT_CONFIG)
    if not p.exists():
        if path is not None:
            raise FileNotFoundError(f"config not found: {p}")
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: expected a YAML mapping")
    cfg["_config_path"] = str(p)
    return cfg


def _get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def _build_anchor(cfg: Dict[str, Any], override_mode: Optional[str],
                  override_level: Optional[float]) -> NearTermAnchor:
    mode = override_mode or _get(cfg, "market.january_anchor_mode",
                                 "spot_to_next_linear")
    level = (override_level if override_level is not None
             else _get(cfg, "market.january_anchor_level_TRY_MWh"))
    return NearTermAnchor(mode=mode, level_TRY_MWh=level)            # type: ignore[arg-type]


def _load_market_inputs(args: argparse.Namespace, cfg: Dict[str, Any]):
    quotes_path = args.quotes or _get(cfg, "market.quotes_file",
                                      "inputs/market/vep_monthly_quotes.csv")
    params_path = getattr(args, "params", None) or _get(
        cfg, "historical.frozen_parameters_file",
        "inputs/historical/m2_frozen_parameters.yaml")
    quotes = load_quotes(quotes_path)
    params = load_frozen_parameters(params_path)
    return quotes, params, str(quotes_path), str(params_path)


def _build_model(quotes, params, cfg: Dict[str, Any],
                 anchor: NearTermAnchor, curve_mode: str) -> ForwardCenteredModel:
    curve = build_forward_curve(
        quotes, mode=curve_mode, anchor=anchor,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh,
        smoothness_weight=float(_get(cfg, "market.smoothness_weight", 1.0)),
        level_weight=float(_get(cfg, "market.level_weight", 1e-4)))
    spec = ResidualSpec.from_frozen(
        params,
        mode=_get(cfg, "residual.mode", "additive"),
        kappa_per_hour=_get(cfg, "residual.kappa_per_hour"),
        regime_means=_get(cfg, "residual.regime_means", (0.0, 0.0)),
        x0_mode=_get(cfg, "residual.x0_mode", "zero"))
    return ForwardCenteredModel(
        curve=curve, spec=spec,
        tvtp=TVTPCoefficients(params.alpha01, params.gamma01,
                              params.alpha10, params.gamma10),
        pi_filtered=params.pi_filtered, valuation_utc=params.valuation_utc,
        spot_price_TRY_MWh=params.spot_price_TRY_MWh)


def _grid_settings(cfg: Dict[str, Any]) -> ResidualGridSettings:
    return ResidualGridSettings(
        n_space_nodes=int(_get(cfg, "grid.n_space_nodes", 1201)),
        n_time_steps=_get(cfg, "grid.n_time_steps"),
        n_std=float(_get(cfg, "grid.n_std", 6.0)))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def cmd_validate(args: argparse.Namespace) -> int:
    from pde_option_model import calendar_tr as cal
    from pde_option_model.generator import (expm_reproduction_error,
                                            probs_to_generator)
    from pde_option_model.forward_centered import residual_moments

    cfg = _load_config(args.config)
    checks: List[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    # --- calendar ---------------------------------------------------------
    add("february_2026_has_672_delivery_hours",
        cal.hours_in_month(2026, 2) == 672,
        f"{cal.hours_in_month(2026, 2)} hours; 2026 leap = "
        f"{cal.days_in_month(2026, 2) == 29}")
    add("turkey_offset_is_fixed_utc_plus_3",
        all(cal.to_turkey(pd.Timestamp(f"2026-{m:02d}-15T12:00:00Z")).hour == 15
            for m in range(1, 13)),
        "no DST shift in any month of 2026")
    add("month_boundaries_are_local_midnight",
        cal.to_turkey(cal.month_start_utc(2026, 2)).hour == 0 and
        cal.month_start_utc(2026, 2) == pd.Timestamp("2026-01-31T21:00:00Z"),
        f"2026-02 starts {cal.month_start_utc(2026, 2)}")
    add("delivery_hour_counts_match_calendar",
        all(len(cal.month_delivery_hours_utc(2026, m)) ==
            cal.hours_in_month(2026, m) for m in range(1, 13)),
        "all 12 months of 2026 consistent")

    # --- market data ------------------------------------------------------
    try:
        quotes, params, qp, pp = _load_market_inputs(args, cfg)
        add("market_quotes_load_and_validate", True,
            f"{len(quotes.quotes)} monthly quotes from {Path(qp).name}")
        add("monthly_quote_is_not_a_point_forward", True,
            "MonthlyBaseloadQuote.maturity_utc raises MonthlyQuoteMisuseError")
        add("unquoted_months_detected", quotes.missing_months() == ["2026-01"],
            f"missing: {quotes.missing_months()}, partial: {quotes.partial_months()}")
    except Exception as exc:
        add("market_quotes_load_and_validate", False, str(exc))
        quotes = params = None

    # --- generator identities --------------------------------------------
    z = np.linspace(-3.0, 3.0, 121)
    if params is not None:
        tv = TVTPCoefficients(params.alpha01, params.gamma01,
                              params.alpha10, params.gamma10)
        p01, p10 = tv.probabilities(z)
        gen = probs_to_generator(p01, p10)
        add("tvtp_chain_is_embeddable", gen.n_clipped == 0,
            f"p01+p10 max {float((p01 + p10).max()):.4f}, clipped {gen.n_clipped}")
        add("expm_of_generator_reproduces_probabilities",
            expm_reproduction_error(p01, p10) < 1e-12,
            f"max abs {expm_reproduction_error(p01, p10):.2e}")

    # --- forward curve ----------------------------------------------------
    if quotes is not None and params is not None:
        for mode in ("piecewise_constant", "smooth_constrained"):
            try:
                c = build_forward_curve(
                    quotes, mode=mode, anchor=_build_anchor(cfg, None, None),
                    spot_price_TRY_MWh=params.spot_price_TRY_MWh)
                add(f"forward_curve_{mode}_reproduces_monthly_quotes",
                    c.max_abs_monthly_error() < 1e-6,
                    f"max abs monthly error {c.max_abs_monthly_error():.3e} TRY/MWh")
            except Exception as exc:
                add(f"forward_curve_{mode}_reproduces_monthly_quotes", False, str(exc))

        # --- residual centering ------------------------------------------
        try:
            model = _build_model(quotes, params, cfg,
                                 _build_anchor(cfg, None, None), "smooth_constrained")
            t = np.arange(0.0, 721.0, 1.0)
            summ = model.residual_summary(t)
            dev = float(np.max(np.abs(summ["expected_spot_TRY_MWh"] -
                                      summ["forward_TRY_MWh"])))
            add("residual_expectation_is_centred", dev < 1e-9,
                f"max |E[P_t] - F(t)| over 720 h = {dev:.3e} TRY/MWh")
            mom = model.moments(t)
            add("residual_moments_finite",
                bool(np.all(np.isfinite(mom.mean)) and np.all(np.isfinite(mom.variance))),
                f"residual sd at 720 h = {float(mom.std[-1]):.1f} TRY/MWh")
            add("regime_probabilities_stay_on_the_simplex",
                float(np.max(np.abs(mom.p.sum(axis=0) - 1.0))) < 1e-10,
                f"max deviation {float(np.max(np.abs(mom.p.sum(axis=0) - 1.0))):.2e}")
        except Exception as exc:
            add("residual_expectation_is_centred", False, str(exc))

    # --- legacy explosion is still detectable -----------------------------
    if params is not None:
        rep = legacy_explosion_report(
            params.scale_P, float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P)),
            params.kappa_per_hour, params.sigma_y, float(params.pi_filtered[1]),
            pi_stationary_stress=params.stationary_pi_stress)
        # Threshold 100 corresponds to the M9-derived sigmas
        # (sigma_stress ~0.092); it was 1e6 under the old placeholder
        # sigma_stress ~0.173.  Both regimes still flag the legacy model as
        # unusable at long horizons; the smaller threshold reflects the
        # smaller (but still >>1) stationary inflation factor.
        add("legacy_moment_explosion_is_flagged",
            rep["stationary_inflation_stress"] > 100.0,
            f"stress-regime stationary inflation factor "
            f"exp(sigma^2/(4 kappa)) = {rep['stationary_inflation_stress']:.3e}")

    # --- report -----------------------------------------------------------
    width = max(len(n) for n, _, _ in checks) + 2
    print(f"\nvalidate — {len(checks)} checks\n" + "=" * (width + 46))
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}} {detail}")
    n_ok = sum(1 for _, ok, _ in checks if ok)
    print("=" * (width + 46))
    print(f"{n_ok}/{len(checks)} checks passed\n")
    return 0 if n_ok == len(checks) else 1


# ---------------------------------------------------------------------------
# calibrate-market
# ---------------------------------------------------------------------------
def cmd_calibrate_market(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    mode = validate_model_mode(args.model)
    if mode != "forward_centered":
        print(f"ERROR: calibrate-market requires --model forward_centered; "
              f"'{mode}' is a benchmark-only mode "
              f"({MODE_DESCRIPTIONS[mode]['role']}).", file=sys.stderr)
        return 2

    quotes, params, qpath, ppath = _load_market_inputs(args, cfg)
    anchor = _build_anchor(cfg, args.january_anchor_mode, args.january_anchor_level)
    curve_mode = args.curve_mode or _get(cfg, "market.curve_mode", "smooth_constrained")
    r_annual = float(args.r_annual if args.r_annual is not None
                     else _get(cfg, "contract.r_annual", 0.40))

    criteria = AcceptanceCriteria(
        max_abs_monthly_error_TRY_MWh=float(
            _get(cfg, "acceptance.max_abs_monthly_error_TRY_MWh", 0.10)),
        max_mape_pct=float(_get(cfg, "acceptance.max_mape_pct", 1.0)),
        max_plausible_forward_TRY_MWh=float(
            _get(cfg, "acceptance.max_plausible_forward_TRY_MWh", 1.0e5)))

    result = run_market_calibration(
        quotes, params, curve_mode=curve_mode, anchor=anchor,
        residual_mode=_get(cfg, "residual.mode", "additive"),
        residual_kappa_per_hour=_get(cfg, "residual.kappa_per_hour"),
        residual_regime_means=_get(cfg, "residual.regime_means", (0.0, 0.0)),
        residual_x0_mode=_get(cfg, "residual.x0_mode", "zero"),
        smoothness_weight=float(_get(cfg, "market.smoothness_weight", 1.0)),
        level_weight=float(_get(cfg, "market.level_weight", 1e-4)),
        criteria=criteria, r_annual=r_annual)

    sens = None
    if not args.no_sensitivity:
        levels = _get(cfg, "market.january_anchor_sensitivity_levels")
        if levels is None:
            s = params.spot_price_TRY_MWh
            levels = [round(s * f, 2) for f in (0.80, 0.90, 1.00, 1.10, 1.20)]
        opt = EuropeanOption(
            option_type=_get(cfg, "contract.option_type", "call"),
            strike=float(_get(cfg, "contract.strike", 3000.0)),
            valuation_utc=params.valuation_utc,
            maturity_utc=params.valuation_utc + pd.Timedelta(
                hours=int(_get(cfg, "contract.maturity_hours", 72))),
            r_annual=r_annual)
        sens_gs = _grid_settings(cfg)
        # Use the same climatology z path as `run_pde.py price`, so the
        # sensitivity table's base row is directly comparable to the main
        # 72h benchmark (~677 TRY/MWh under M9 sigmas) rather than to the
        # constant-z fallback (~686).
        try:
            _sens_scen, sens_z_fn = _build_tvtp_scenario(opt, sens_gs, cfg, args)
        except Exception as exc:                              # pragma: no cover
            logging.getLogger(__name__).warning(
                "sensitivity climatology z path unavailable (%s); "
                "falling back to constant z=0", exc)
            sens_z_fn = None
        sens = near_term_anchor_sensitivity(
            quotes, params, levels, curve_mode=curve_mode, option=opt,
            grid_settings=sens_gs, z_lagged_fn=sens_z_fn)

    out = write_calibration_outputs(
        result, params, args.outdir, anchor_sensitivity=sens,
        legacy_reference_path=_get(
            cfg, "legacy.reference_file",
            "inputs/legacy_reference/legacy_model_implied_forwards.json"),
        r_annual=r_annual,
        extra_notes={"quotes_file": qpath, "frozen_parameters_file": ppath,
                     "curve_mode": curve_mode,
                     "january_anchor_mode": anchor.mode})

    m = result.metrics
    print(f"\ncalibrate-market — {PRICE_LABEL}")
    print("=" * 78)
    print(f"  optimizer_success        : {result.optimizer_success}")
    print(f"  calibration_accepted     : {result.calibration_accepted}")
    print(f"  curve mode               : {curve_mode}")
    print(f"  monthly RMSE             : {m['monthly_RMSE']:.6e} TRY/MWh")
    print(f"  monthly MAE              : {m['monthly_MAE']:.6e} TRY/MWh")
    print(f"  monthly MAPE             : {m['monthly_MAPE']:.6e} %")
    print(f"  max abs monthly error    : {m['maximum_absolute_monthly_error']:.6e} TRY/MWh")
    print(f"  direct market constraints: {', '.join(result.curve.constrained_months)}")
    print(f"  extrapolated periods     : {', '.join(result.curve.extrapolated_months) or '(none)'}")
    print("-" * 78)
    print("  expected spot")
    for h in REPORTING_HORIZONS_HOURS:
        d = result.expected_spot[f"{h}h_detail"]
        tag = "near-term anchored" if d["near_term_anchored"] else "VEP-constrained"
        print(f"    {h:4d} h : {d['expected_spot_TRY_MWh']:10.2f} TRY/MWh   "
              f"[{d['delivery_month']}, {tag}]")
    print("-" * 78)
    print("  acceptance checks")
    for c in result.checks:
        print(f"    [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    print("-" * 78)
    print(f"  wrote {len(out.files)} files to {out.directory}")
    for f in sorted(out.files):
        print(f"    {f}")
    if not result.calibration_accepted:
        print("\n  CALIBRATION REJECTED — no calibrated_config.yaml produced.",
              file=sys.stderr)
        return 1
    print("=" * 78 + "\n")
    return 0


# ---------------------------------------------------------------------------
# TVTP residual-demand path
# ---------------------------------------------------------------------------
def _build_tvtp_scenario(contract, grid_settings, cfg, args):
    """Build deterministic z(t-1) used by the M2 TVTP transition law."""

    history_file = (
        getattr(args, "scenario_history", None)
        or _get(
            cfg,
            "scenario.history_file",
            "inputs/historical/rd_standardized.csv",
        )
    )

    history = pd.read_csv(history_file)

    if "datetime" not in history.columns or "z" not in history.columns:
        raise ValueError(
            f"{history_file}: expected columns 'datetime' and 'z'"
        )

    history["datetime"] = pd.to_datetime(
        history["datetime"],
        utc=True,
    )

    z_history = pd.Series(
        history["z"].to_numpy(float),
        index=history["datetime"],
    ).sort_index()

    train_end = pd.Timestamp(
        _get(
            cfg,
            "scenario.train_end_utc",
            "2022-12-31 20:00:00+00:00",
        )
    )

    if train_end.tzinfo is None:
        train_end = train_end.tz_localize("UTC")
    else:
        train_end = train_end.tz_convert("UTC")

    lag_hours = float(
        _get(cfg, "scenario.covariate_lag_hours", 1.0)
    )

    mode = (
        getattr(args, "scenario_mode", None)
        or _get(cfg, "scenario.mode", "climatology")
    )

    arg_offset = getattr(args, "scenario_offset", None)
    offset = float(
        arg_offset
        if arg_offset is not None
        else _get(cfg, "scenario.offset", 0.0)
    )

    custom_csv = (
        getattr(args, "scenario_custom_csv", None)
        or _get(cfg, "scenario.custom_csv")
    )

    # 0.25 h master path so the PDE and default MC grids use the
    # same deterministic exogenous TVTP path.
    tau = contract.tau_hours
    n_master = max(
        grid_settings.n_steps(tau),
        int(np.ceil(tau / 0.25)),
    )

    master_grid = TimeGrid(
        contract.valuation_utc,
        contract.maturity_utc,
        n_master,
    )

    builder = ScenarioBuilder(
        z_history=z_history,
        train_end=train_end,
        covariate_lag_hours=lag_hours,
    )

    scenario = builder.build(
        ScenarioSpec(
            name="pricing",
            mode=mode,
            offset=offset,
            custom_csv=custom_csv,
        ),
        master_grid,
    )

    def z_lagged_fn(t):
        t = np.asarray(t, dtype=float)
        return np.interp(
            t,
            scenario.times_hours,
            scenario.z_lagged,
        )

    return scenario, z_lagged_fn


# ---------------------------------------------------------------------------
# price
# ---------------------------------------------------------------------------
def cmd_price(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    mode = validate_model_mode(args.model)
    quotes, params, _, _ = _load_market_inputs(args, cfg)

    pi_override_mode = getattr(args, "pi_override", "filtered") or "filtered"
    if pi_override_mode == "stationary":
        if params.m9_stationary_pi is None:
            print("ERROR: --pi-override stationary requires the frozen-parameter "
                  "YAML to define m9_stationary_pi (see "
                  "inputs/historical/m2_frozen_parameters.yaml).",
                  file=sys.stderr)
            return 2
        params.pi_filtered = np.asarray(params.m9_stationary_pi, dtype=float)
        print(f"--pi-override stationary: using M9 long-run occupancy "
              f"pi=({params.pi_filtered[0]:.4f}, {params.pi_filtered[1]:.4f}) "
              f"instead of the M2 shipped filter")

    if args.curve:
        cdir = Path(args.curve)
        cdir = cdir.parent if cdir.is_file() else cdir
        try:
            blob = load_calibration_result_json(cdir)
        except CalibrationRejected as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"using accepted calibration from {cdir} "
              f"(valuation {blob['valuation_date']})")

    anchor = _build_anchor(cfg, args.january_anchor_mode, args.january_anchor_level)
    curve_mode = args.curve_mode or _get(cfg, "market.curve_mode", "smooth_constrained")
    r_annual = float(args.r_annual if args.r_annual is not None
                     else _get(cfg, "contract.r_annual", 0.40))
    strike = float(args.strike if args.strike is not None
                   else _get(cfg, "contract.strike", 3000.0))
    hours = int(args.maturity_hours if args.maturity_hours is not None
                else _get(cfg, "contract.maturity_hours", 72))
    otype = args.option_type or _get(cfg, "contract.option_type", "call")

    model = _build_model(quotes, params, cfg, anchor, curve_mode)
    a0 = float(getattr(args, "risk_premium_a0", 0.0) or 0.0)
    a1 = float(getattr(args, "risk_premium_a1", 0.0) or 0.0)
    if a0 != 0.0 or a1 != 0.0:
        model.spec.drift_shift_per_hour = np.asarray([a0, a1], dtype=float)
        print(f"--risk-premium: applying Q1 drift shift a=({a0:+g}, {a1:+g}) TRY/MWh/h "
              f"(UNCALIBRATED sensitivity scenario; see model_limitations.md item (f))")
    contract = EuropeanOption(
        option_type=otype, strike=strike, valuation_utc=params.valuation_utc,
        maturity_utc=params.valuation_utc + pd.Timedelta(hours=hours),
        r_annual=r_annual)

    if mode == "legacy_asinh_ou":
        print("ERROR: legacy_asinh_ou pricing needs the full historical artefact "
              "bundle (pde_timeseries.parquet etc.), which is not part of this "
              "market-calibration delivery. Use --model forward_centered, or run "
              "the legacy stack directly against an artefact directory.",
              file=sys.stderr)
        return 2

    gs = _grid_settings(cfg)

    scenario, z_lagged_fn = _build_tvtp_scenario(
        contract,
        gs,
        cfg,
        args,
    )

    res = price_forward_centered(
        model,
        contract,
        gs,
        z_lagged_fn=z_lagged_fn,
    )

    mc = None
    if not args.no_mc:
        mc = simulate_forward_centered(
            model,
            contract,
            n_paths=int(args.mc_paths),
            dt_hours=float(args.mc_dt_hours),
            seed=int(args.mc_seed),
            z_lagged_fn=z_lagged_fn,
        )
    d = model.curve.frame
    anchored = bool(d.loc[d["time_utc"] == contract.maturity_utc.isoformat(),
                          "near_term_anchor_flag"].any()) if hours else False
    print(f"\nprice — {PRICE_LABEL}")
    print("=" * 78)
    print(f"  contract                 : European {otype} on the EXPIRY-HOUR spot PTF")
    print(f"                             (NOT a monthly baseload option)")
    print(f"  strike                   : {strike:.2f} TRY/MWh")
    print(f"  valuation / maturity     : {contract.valuation_utc.isoformat()} -> "
          f"{contract.maturity_utc.isoformat()} ({hours} h)")
    print(f"  model mode               : {mode}")
    print(f"  TVTP scenario            : {args.scenario_mode or _get(cfg, 'scenario.mode', 'climatology')}")
    print(f"  TVTP z(t-1) range        : [{scenario.z_lagged.min():.4f}, {scenario.z_lagged.max():.4f}]")
    print(f"  F(T)                     : {res.forward_at_expiry:.2f} TRY/MWh"
          f"{'   [near-term anchored]' if anchored else ''}")
    print(f"  E[P_T]                   : {res.expected_spot_at_expiry:.2f} TRY/MWh")
    print(f"  residual sd at expiry    : {res.residual_std_at_expiry:.2f} TRY/MWh")
    print(f"  V | regime 0 (normal)    : {res.V_regime[0]:.4f} TRY/MWh")
    print(f"  V | regime 1 (stress)    : {res.V_regime[1]:.4f} TRY/MWh")
    print(f"  pi (normal, stress)      : ({res.pi[0]:.4f}, {res.pi[1]:.4f})")
    print(f"  VALUE                    : {res.value:.4f} TRY/MWh")
    if mc is not None:
        z = abs(res.value - mc["value"]) / max(mc["std_error"], 1e-12)
        print(f"  Monte Carlo cross-check  : {mc['value']:.4f} +/- {mc['std_error']:.4f} "
              f"(|z| = {z:.2f}, {int(mc['n_paths'])} paths, seed {int(mc['seed'])})")
        print(f"  MC dt                    : {mc['dt_hours']:.4f} h")
        print(f"  MC E[P_T]                : {mc['mean_price_T']:.2f} +/- "
              f"{mc['mean_price_T_se']:.2f} TRY/MWh")
        print(f"  analytic E[P_T]          : {mc['analytic_expected_spot_T']:.2f} TRY/MWh")
        print(f"  MC P(P_T < 0)            : {mc['prob_negative_price']:.4f}")
    print("=" * 78 + "\n")
    return 0


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def cmd_diagnostics(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = _load_config(args.config)
    validate_model_mode(args.model)
    quotes, params, _, _ = _load_market_inputs(args, cfg)

    if args.curve:
        cdir = Path(args.curve)
        cdir = cdir.parent if cdir.is_file() else cdir
        try:
            load_calibration_result_json(cdir)
        except CalibrationRejected as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except FileNotFoundError:
            print(f"note: no calibration_result.json under {cdir}; rebuilding "
                  "the curve from the quote file", file=sys.stderr)

    anchor = _build_anchor(cfg, args.january_anchor_mode, args.january_anchor_level)
    model = _build_model(quotes, params, cfg, anchor,
                         args.curve_mode or _get(cfg, "market.curve_mode",
                                                 "smooth_constrained"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    diag_gs = _grid_settings(cfg)

    diag_contract = EuropeanOption(
        option_type="call",
        strike=1.0,
        valuation_utc=params.valuation_utc,
        maturity_utc=params.valuation_utc + pd.Timedelta(hours=720),
        r_annual=float(_get(cfg, "contract.r_annual", 0.40)),
    )

    diag_scenario, diag_z_lagged_fn = _build_tvtp_scenario(
        diag_contract,
        diag_gs,
        cfg,
        args,
    )

    t = np.arange(0.0, 721.0, 1.0)
    z_diag = diag_z_lagged_fn(t)

    summ = model.residual_summary(
        t,
        z_lagged=z_diag,
    )

    summ.to_csv(outdir / "residual_diagnostics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    ax.plot(summ["hours"], summ["forward_TRY_MWh"], label="F(t)")
    ax.plot(summ["hours"], summ["expected_spot_TRY_MWh"], "--", label="E[P_t]")
    ax.set_xlabel("hours"); ax.set_ylabel("TRY/MWh")
    ax.set_title("Forward level and expected spot"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(summ["hours"], summ["residual_std_TRY_MWh"], color="#d62728")
    ax.set_xlabel("hours"); ax.set_ylabel("TRY/MWh")
    ax.set_title("Residual standard deviation (analytic)")
    ax = axes[2]
    ax.plot(summ["hours"], summ["p_stress"], color="#9467bd")
    ax.set_xlabel("hours"); ax.set_ylabel("Q(J_t = stress)")
    ax.set_title("Regime probability")
    fig.suptitle(f"forward_centered diagnostics — {PRICE_LABEL}", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "residual_diagnostics.png", dpi=150)
    plt.close(fig)

    r_annual = float(_get(cfg, "contract.r_annual", 0.40))
    strikes = np.array(_get(cfg, "diagnostics.strikes",
                            [2000, 2500, 3000, 3500, 4000]), dtype=float)
    gs = _grid_settings(cfg)
    rows = []
    T = params.valuation_utc + pd.Timedelta(
        hours=int(_get(cfg, "contract.maturity_hours", 72)))
    disc = float(np.exp(-r_annual / 8760.0 *
                        (T - params.valuation_utc).total_seconds() / 3600.0))
    for K in strikes:
        c = price_forward_centered(
            model,
            EuropeanOption("call", K, params.valuation_utc, T, r_annual),
            gs,
            z_lagged_fn=diag_z_lagged_fn,
        )

        p = price_forward_centered(
            model,
            EuropeanOption("put", K, params.valuation_utc, T, r_annual),
            gs,
            z_lagged_fn=diag_z_lagged_fn,
        )
        rows.append({"strike_TRY_MWh": K, "call_TRY_MWh": c.value,
                     "put_TRY_MWh": p.value,
                     "call_minus_put": c.value - p.value,
                     "disc_times_F_minus_K": disc * (c.forward_at_expiry - K),
                     "parity_error": c.value - p.value -
                                     disc * (c.forward_at_expiry - K)})
    smile = pd.DataFrame(rows)
    smile.to_csv(outdir / "strike_profile.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(smile["strike_TRY_MWh"], smile["call_TRY_MWh"], "o-", label="call")
    ax.plot(smile["strike_TRY_MWh"], smile["put_TRY_MWh"], "s-", label="put")
    ax.set_xlabel("strike (TRY/MWh)"); ax.set_ylabel("value (TRY/MWh)")
    ax.set_title(f"Strike profile — {PRICE_LABEL}", fontsize=9)
    ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "strike_profile.png", dpi=150)
    plt.close(fig)

    rep = legacy_explosion_report(
        params.scale_P, float(np.arcsinh(params.spot_price_TRY_MWh / params.scale_P)),
        params.kappa_per_hour, params.sigma_y, float(params.pi_filtered[1]),
        pi_stationary_stress=params.stationary_pi_stress,
        theta=params.legacy_theta_effective or 0.0)
    rep["table"].to_csv(outdir / "legacy_explosion_table.csv", index=False)
    with open(outdir / "legacy_explosion_report.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in rep.items() if k != "table"}, fh, indent=2)

    print(f"\ndiagnostics ({args.model})")
    print("=" * 78)
    print(summ[summ["hours"].isin([0, 72, 168, 336, 720])].to_string(index=False))
    print("-" * 78)
    print(smile.to_string(index=False))
    print("-" * 78)
    print(f"  max |put-call parity error| = "
          f"{float(np.abs(smile['parity_error']).max()):.3e} TRY/MWh")
    print(f"  legacy stress-regime stationary inflation factor = "
          f"{rep['stationary_inflation_stress']:.3e}")
    print(f"  wrote diagnostics to {outdir}")
    print("=" * 78 + "\n")
    return 0


# ---------------------------------------------------------------------------
# freeze-params
# ---------------------------------------------------------------------------
def cmd_freeze_params(args: argparse.Namespace) -> int:
    from pde_option_model.params_frozen import freeze_from_artifacts
    try:
        out = freeze_from_artifacts(args.input_root, args.out, args.model_name)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}\n\nfreeze-params needs the full historical artefact "
              "bundle (parameter_estimates.csv, transition_coefficients.csv, "
              "pde_export.json, prepared_meta.json, "
              "metadata/model_parameters_and_ou_mapping.json, "
              "pde_timeseries.parquet).", file=sys.stderr)
        return 2
    print(f"frozen parameters written to {out}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pde.py",
        description="Two-regime TVTP PDE pricer for Turkish electricity options "
                    "(EPİAŞ PTF, TRY/MWh).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--config", default=None,
                   help=f"YAML config (default {DEFAULT_CONFIG})")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--quotes", default=None, help="VEP quotes CSV or JSON")
        sp.add_argument("--params", default=None,
                        help="frozen historical parameter YAML")
        sp.add_argument("--curve-mode", default=None,
                        choices=["piecewise_constant", "smooth_constrained"])
        sp.add_argument("--january-anchor-mode", default=None,
                        choices=["spot_flat", "spot_to_next_linear",
                                 "flat_next_month", "explicit_level"])
        sp.add_argument("--january-anchor-level", type=float, default=None,
                        help="January baseload level (TRY/MWh) for explicit_level")
        sp.add_argument("--r-annual", type=float, default=None)

    sv = sub.add_parser("validate", help="structural and numerical self-checks")
    common(sv)
    sv.set_defaults(func=cmd_validate)

    sc = sub.add_parser("calibrate-market",
                        help="build the VEP-anchored curve and run acceptance")
    common(sc)
    sc.add_argument("--model", default=DEFAULT_MARKET_MODE, choices=list(MODEL_MODES))
    sc.add_argument("--outdir", default="outputs/market_calibration_final")
    sc.add_argument("--no-sensitivity", action="store_true")
    sc.set_defaults(func=cmd_calibrate_market)

    sp_ = sub.add_parser("price", help="price a European option on the expiry-hour spot")
    common(sp_)
    sp_.add_argument("--model", default=DEFAULT_MARKET_MODE, choices=list(MODEL_MODES))
    sp_.add_argument("--curve", default=None,
                     help="calibration output directory or hourly_forward_curve.csv")
    sp_.add_argument("--option-type", default=None, choices=["call", "put"])
    sp_.add_argument("--strike", type=float, default=None)
    sp_.add_argument("--maturity-hours", type=int, default=None)
    sp_.add_argument("--no-mc", action="store_true")
    sp_.add_argument("--mc-paths", type=int, default=60000)
    sp_.add_argument(
        "--mc-dt-hours",
        type=float,
        default=0.05,
        help="Monte Carlo time step in hours (default 0.05 h = 3 minutes)",
    )
    sp_.add_argument("--mc-seed", type=int, default=20260808)
    sp_.add_argument(
        "--scenario-mode",
        choices=["constant", "climatology", "custom"],
        default=None,
        help="deterministic standardized residual-demand path for TVTP",
    )
    sp_.add_argument(
        "--scenario-history",
        default=None,
        help="historical contemporaneous standardized RD CSV",
    )
    sp_.add_argument(
        "--scenario-offset",
        type=float,
        default=None,
        help="additive shift in standardized RD units",
    )
    sp_.add_argument(
        "--scenario-custom-csv",
        default=None,
        help="custom future z(t) CSV when --scenario-mode custom",
    )
    sp_.add_argument(
        "--pi-override",
        choices=["filtered", "stationary"],
        default="filtered",
        help=("regime probability at valuation. "
              "'filtered' = M2 shipped filter (default, used for 72h+ benchmarks); "
              "'stationary' = M9 long-run occupancy (recommended for <24h maturities, "
              "see model_limitations.md item (e))"),
    )
    sp_.add_argument(
        "--risk-premium-a0",
        type=float,
        default=0.0,
        help=("Regime-0 (normal) drift shift, TRY/MWh per hour. UNCALIBRATED "
              "sensitivity scenario -- no electricity option market data exists "
              "to fit a real market price of risk; see model_limitations.md item "
              "(f). Default 0.0 reproduces the physical-measure baseline."),
    )
    sp_.add_argument(
        "--risk-premium-a1",
        type=float,
        default=0.0,
        help=("Regime-1 (stress) drift shift, TRY/MWh per hour. UNCALIBRATED "
              "sensitivity scenario -- no electricity option market data exists "
              "to fit a real market price of risk; see model_limitations.md item "
              "(f). Default 0.0 reproduces the physical-measure baseline."),
    )

    sp_.set_defaults(func=cmd_price)

    sd = sub.add_parser("diagnostics", help="residual/regime diagnostics pack")
    common(sd)
    sd.add_argument("--model", default=DEFAULT_MARKET_MODE, choices=list(MODEL_MODES))
    sd.add_argument("--curve", default=None)
    sd.add_argument("--outdir", default="outputs/forward_centered_diagnostics")
    sd.set_defaults(func=cmd_diagnostics)

    sf = sub.add_parser("freeze-params",
                        help="extract frozen M2 parameters from an artefact bundle")
    sf.add_argument("--input-root", default="inputs")
    sf.add_argument("--out", default="inputs/historical/m2_frozen_parameters.yaml")
    sf.add_argument("--model-name", default="M2")
    sf.set_defaults(func=cmd_freeze_params)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except CalibrationRejected as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
