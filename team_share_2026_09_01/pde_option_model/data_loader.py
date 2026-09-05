"""Load and validate the uploaded Markov artifacts.

Expected input layout (configurable in pde_config.yaml; no absolute
Windows-style paths are ever used):

    inputs/
      parameter_estimates.csv           TRY run, rows M0 / M2      [authoritative emission]
      transition_coefficients.csv       TRY run, M2 gammas         [authoritative gammas]
      model_comparison.csv              TRY run model table
      pde_export.json                   TRY run, M0-based export   [scale_P, seasonal betas, rho]
      prepared_meta.json                TRY run preprocessing meta [scale_P, splits, RD definition]
      run_summary.json                  TRY run summary
      pde_timeseries.parquet            master time series + shipped probabilities
      transition_probabilities_and_generator.parquet
      smoothed_probabilities.parquet
      filtered_probabilities.parquet
      metadata/model_parameters_and_ou_mapping.json   [only source of alpha01/alpha10]
      metadata/latest_state.json
      metadata/pde_model_contract.json
      metadata/covariate_scaling.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    def _p(self, name: str) -> Path:
        return self.root / name

    @property
    def parameter_estimates(self) -> Path: return self._p("parameter_estimates.csv")
    @property
    def transition_coefficients(self) -> Path: return self._p("transition_coefficients.csv")
    @property
    def model_comparison(self) -> Path: return self._p("model_comparison.csv")
    @property
    def pde_export(self) -> Path: return self._p("pde_export.json")
    @property
    def prepared_meta(self) -> Path: return self._p("prepared_meta.json")
    @property
    def run_summary(self) -> Path: return self._p("run_summary.json")
    @property
    def timeseries(self) -> Path: return self._p("pde_timeseries.parquet")
    @property
    def transition_series(self) -> Path:
        return self._p("transition_probabilities_and_generator.parquet")
    @property
    def smoothed(self) -> Path: return self._p("smoothed_probabilities.parquet")
    @property
    def filtered(self) -> Path: return self._p("filtered_probabilities.parquet")
    @property
    def ou_mapping(self) -> Path:
        return self._p("metadata/model_parameters_and_ou_mapping.json")
    @property
    def latest_state(self) -> Path: return self._p("metadata/latest_state.json")
    @property
    def model_contract(self) -> Path: return self._p("metadata/pde_model_contract.json")
    @property
    def covariate_scaling(self) -> Path: return self._p("metadata/covariate_scaling.json")


@dataclass
class RawArtifacts:
    parameter_estimates: pd.DataFrame
    transition_coefficients: pd.DataFrame
    model_comparison: pd.DataFrame
    pde_export: dict[str, Any]
    prepared_meta: dict[str, Any]
    run_summary: dict[str, Any]
    ou_mapping: dict[str, Any]
    latest_state: dict[str, Any]
    model_contract: dict[str, Any]
    timeseries: pd.DataFrame
    transition_series: Optional[pd.DataFrame] = None
    smoothed: Optional[pd.DataFrame] = None
    filtered: Optional[pd.DataFrame] = None
    missing: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_ts_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "datetime" in df.columns:
        df = df.set_index(pd.to_datetime(df["datetime"], utc=True)).drop(columns=["datetime"])
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{path.name}: expected a datetime index")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{path.name}: index is not monotonic increasing")
    return df


def load_artifacts(input_root: str | Path) -> RawArtifacts:
    """Load every artifact, failing loudly on required files."""
    paths = ArtifactPaths(Path(input_root))
    required = {
        "parameter_estimates": paths.parameter_estimates,
        "transition_coefficients": paths.transition_coefficients,
        "pde_export": paths.pde_export,
        "prepared_meta": paths.prepared_meta,
        "ou_mapping": paths.ou_mapping,
        "timeseries": paths.timeseries,
    }
    for name, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"required artifact '{name}' missing at {p}")

    missing: list[str] = []

    def _opt_df(p: Path) -> Optional[pd.DataFrame]:
        if p.exists():
            return _read_ts_frame(p)
        missing.append(str(p))
        return None

    def _opt_json(p: Path) -> dict[str, Any]:
        if p.exists():
            return _read_json(p)
        missing.append(str(p))
        return {}

    def _opt_csv(p: Path) -> pd.DataFrame:
        if p.exists():
            return pd.read_csv(p)
        missing.append(str(p))
        return pd.DataFrame()

    raw = RawArtifacts(
        parameter_estimates=pd.read_csv(paths.parameter_estimates),
        transition_coefficients=pd.read_csv(paths.transition_coefficients),
        model_comparison=_opt_csv(paths.model_comparison),
        pde_export=_read_json(paths.pde_export),
        prepared_meta=_read_json(paths.prepared_meta),
        run_summary=_opt_json(paths.run_summary),
        ou_mapping=_read_json(paths.ou_mapping),
        latest_state=_opt_json(paths.latest_state),
        model_contract=_opt_json(paths.model_contract),
        timeseries=_read_ts_frame(paths.timeseries),
        transition_series=_opt_df(paths.transition_series),
        smoothed=_opt_df(paths.smoothed),
        filtered=_opt_df(paths.filtered),
        missing=missing,
    )
    if missing:
        logger.warning("optional artifacts missing: %s", missing)

    # structural validation of the master time series
    ts = raw.timeseries
    needed_cols = {"PTF_TRY_MWh", "Demand_MWh", "Wind_MWh", "Solar_MWh",
                   "filter_p_state0", "filter_p_state1",
                   "smooth_p_state0", "smooth_p_state1"}
    absent = needed_cols - set(ts.columns)
    if absent:
        raise ValueError(f"pde_timeseries is missing required columns: {sorted(absent)}")
    if ts.index.has_duplicates:
        raise ValueError("pde_timeseries has duplicate timestamps")
    logger.info("loaded %d hourly rows spanning %s .. %s",
                len(ts), ts.index[0], ts.index[-1])
    return raw
