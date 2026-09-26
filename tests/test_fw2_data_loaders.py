"""FW2 §0.4 data loader tests: TUIK CPI + extended realized PTF."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.load_cpi import (PROVENANCE, deflator, load_cpi,   # noqa: E402
                                   yoy_december_inflation)


# ---------------------------------------------------------------------------
# CPI
# ---------------------------------------------------------------------------
def test_cpi_panel_has_128_monthly_observations_2016_01_to_2026_08():
    df = load_cpi()
    assert len(df) == 128
    assert df["date"].iloc[0] == pd.Timestamp("2016-01-01")
    assert df["date"].iloc[-1] == pd.Timestamp("2026-08-01")
    # dtype and integrity
    assert df["cpi"].dtype == float
    assert not df["cpi"].isna().any()


def test_cpi_endpoints_match_the_recorded_values():
    df = load_cpi()
    assert df["cpi"].iloc[0] == pytest.approx(274.44, abs=1e-6)
    assert df["cpi"].iloc[-1] == pytest.approx(4289.23, abs=1e-6)


def test_cpi_footer_block_is_dropped():
    """The EVDS 'Seri Aciklamalari' + 'Notlar' block is preserved in the
    workbook (source metadata for the paper appendix) but MUST NOT
    appear as data rows in the loaded panel."""
    df = load_cpi()
    for stray in ("Seri", "Notlar", "TP.GENENDEKS"):
        assert not df["date"].astype(str).str.contains(stray).any()


def test_yoy_december_matches_official_tuik_inflation():
    """Structural guard against a base-year drift or a series mismatch:
    2017-2024 official TUIK Dec/Dec inflation percentages must reproduce
    to two decimal places from the loaded index."""
    df = load_cpi()
    yoy = yoy_december_inflation(df).set_index("year")["yoy_dec_dec_pct"]
    official = {2017: 11.92, 2018: 20.30, 2019: 11.84, 2020: 14.60,
                2021: 36.08, 2022: 64.27, 2023: 64.77, 2024: 44.38}
    for year, expected in official.items():
        assert round(float(yoy.loc[year]), 2) == expected, (
            f"{year}: got {float(yoy.loc[year]):.4f}, expected {expected}")


def test_deflator_at_base_month_is_one():
    df = load_cpi()
    d = deflator(df, base_month=pd.Timestamp("2025-12-01"))
    row = d.loc[d["date"] == pd.Timestamp("2025-12-01")]
    assert float(row["deflator"].iloc[0]) == pytest.approx(1.0, abs=1e-12)


def test_provenance_metadata_present():
    """Series code, base year, and source are needed by the manuscript
    data appendix.  Fail if any of the three is missing."""
    for key in ("series_code", "base_year", "source"):
        assert key in PROVENANCE and PROVENANCE[key]
    assert PROVENANCE["provenance_tag"] == "Inherited"


# ---------------------------------------------------------------------------
# Extended realized PTF: file plumbing only, no numerical claim
# ---------------------------------------------------------------------------
EXTENDED_CSV = REPO_ROOT / "inputs" / "market" / "realized_ptf_2025-12-31_2026-09-27.csv"
FROZEN_CSV = REPO_ROOT / "inputs" / "market" / "realized_ptf_2026.csv"


def _load_ptf(csv_path: Path) -> pd.DataFrame:
    """Read one realized-PTF CSV.  Dates are LEFT as strings before the
    controlled parse to avoid pandas' thousands-separator eating dots
    in ``dd.mm.YYYY``.  Timestamps are localized to Europe/Istanbul and
    converted to UTC to match the tracked convention."""
    df = pd.read_csv(csv_path, sep=";", decimal=",", thousands=".",
                     dtype={"Tarih": str, "Saat": str})
    ts_local = pd.to_datetime(df["Tarih"] + " " + df["Saat"],
                              format="%d.%m.%Y %H:%M", errors="coerce")
    ts_utc = (ts_local.dt.tz_localize("Europe/Istanbul", ambiguous="infer",
                                      nonexistent="shift_forward")
              .dt.tz_convert("UTC"))
    return pd.DataFrame({"ts_utc": ts_utc,
                         "ptf_TRY_MWh": df["PTF (TL/MWh)"].astype(float)})


def test_extended_realized_ptf_is_6504_hours_no_gaps_no_dupes_no_nan():
    df = _load_ptf(EXTENDED_CSV)
    assert len(df) == 6504
    assert df["ts_utc"].is_monotonic_increasing
    assert not df["ts_utc"].duplicated().any()
    assert not df["ptf_TRY_MWh"].isna().any()
    dt = df["ts_utc"].diff().dropna()
    assert (dt == pd.Timedelta(hours=1)).all()


def test_extended_realized_ptf_timezone_window():
    """First row = 2025-12-31 00:00 Turkey local; last = 2026-09-27
    23:00 Turkey local.  Converted to UTC that is
    2025-12-30 21:00 UTC to 2026-09-27 20:00 UTC."""
    df = _load_ptf(EXTENDED_CSV)
    assert df["ts_utc"].iloc[0] == pd.Timestamp("2025-12-30 21:00", tz="UTC")
    assert df["ts_utc"].iloc[-1] == pd.Timestamp("2026-09-27 20:00", tz="UTC")


def test_frozen_realized_ptf_untouched_and_extended_matches_on_overlap():
    """The FW3 backtest numbers depend on the frozen realized_ptf_2026
    file being byte-stable; the extended file is a superset for FW6.
    Overlapping hours must therefore be identical to the last decimal.
    """
    frozen = _load_ptf(FROZEN_CSV)
    extended = _load_ptf(EXTENDED_CSV)
    m = frozen.merge(extended, on="ts_utc", suffixes=("_frozen", "_ext"))
    assert len(m) == len(frozen)
    assert (m["ptf_TRY_MWh_frozen"] == m["ptf_TRY_MWh_ext"]).all()
