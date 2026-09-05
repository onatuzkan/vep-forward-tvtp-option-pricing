"""Turkish delivery-calendar utilities (UTC+3, fixed, no daylight saving).

Türkiye has used a permanent UTC+3 offset since September 2016; no DST rule is
applied anywhere in this package.  Every EPİAŞ monthly baseload contract is
defined on *local* Turkish delivery hours:

    delivery window of month m = [ m/01 00:00 TRT , (m+1)/01 00:00 TRT )

which in UTC is

    [ (m-1 day) 21:00 UTC , (last day of m) 21:00 UTC ).

The number of delivery hours is therefore exactly ``days_in_month * 24``
(no 23/25-hour DST days).  February 2026 has 28 days -> 672 hours.

Units: all timestamps are tz-aware; all durations are in hours.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

TURKEY_UTC_OFFSET_HOURS: int = 3
TURKEY_TZ: str = "Etc/GMT-3"          # POSIX sign convention: GMT-3 == UTC+3
HOURS_PER_DAY: int = 24

_MONTH_CODES = {
    1: "01", 2: "02", 3: "03", 4: "04", 5: "05", 6: "06",
    7: "07", 8: "08", 9: "09", 10: "10", 11: "11", 12: "12",
}


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------
def to_turkey(ts_utc: pd.Timestamp | pd.DatetimeIndex):
    """UTC -> Turkish local time (UTC+3, fixed)."""
    if isinstance(ts_utc, pd.DatetimeIndex):
        if ts_utc.tz is None:
            raise ValueError("expected a tz-aware UTC DatetimeIndex")
        return ts_utc.tz_convert(TURKEY_TZ)
    if ts_utc.tzinfo is None:
        raise ValueError("expected a tz-aware UTC Timestamp")
    return ts_utc.tz_convert(TURKEY_TZ)


def to_utc(ts_local: pd.Timestamp | pd.DatetimeIndex):
    """Turkish local time -> UTC."""
    if isinstance(ts_local, pd.DatetimeIndex):
        idx = ts_local.tz_localize(TURKEY_TZ) if ts_local.tz is None else ts_local
        return idx.tz_convert("UTC")
    ts = ts_local.tz_localize(TURKEY_TZ) if ts_local.tzinfo is None else ts_local
    return ts.tz_convert("UTC")


def turkey_local(year: int, month: int, day: int = 1, hour: int = 0) -> pd.Timestamp:
    """Construct a Turkish-local tz-aware timestamp."""
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, tz=TURKEY_TZ)


# ---------------------------------------------------------------------------
# month arithmetic
# ---------------------------------------------------------------------------
def days_in_month(year: int, month: int) -> int:
    """Calendar days, leap-year aware (2026 is NOT a leap year)."""
    return calendar.monthrange(int(year), int(month))[1]


def hours_in_month(year: int, month: int) -> int:
    """Delivery hours of a Turkish baseload month = days * 24 (no DST)."""
    return days_in_month(year, month) * HOURS_PER_DAY


def month_start_utc(year: int, month: int) -> pd.Timestamp:
    """First delivery hour of the month, in UTC."""
    return to_utc(turkey_local(year, month, 1, 0))


def month_end_utc(year: int, month: int) -> pd.Timestamp:
    """Exclusive end of the delivery window (= start of the next month), in UTC."""
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return to_utc(turkey_local(ny, nm, 1, 0))


def month_delivery_hours_utc(year: int, month: int) -> pd.DatetimeIndex:
    """All hourly delivery timestamps of the month, left-closed, in UTC."""
    start, end = month_start_utc(year, month), month_end_utc(year, month)
    idx = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    n = hours_in_month(year, month)
    if len(idx) != n:
        raise ValueError(
            f"delivery-hour count mismatch for {year}-{month:02d}: "
            f"generated {len(idx)}, expected {n}")
    return idx


def month_label(year: int, month: int) -> str:
    """Canonical 'YYYY-MM' delivery-month key."""
    return f"{int(year):04d}-{int(month):02d}"


def parse_month_label(label: str) -> Tuple[int, int]:
    y, m = str(label).split("-")[:2]
    return int(y), int(m)


def month_of(ts_utc: pd.Timestamp) -> Tuple[int, int]:
    """(year, month) of the Turkish *local* delivery month containing ts."""
    loc = to_turkey(ts_utc)
    return int(loc.year), int(loc.month)


def iter_months(start_year: int, start_month: int,
                end_year: int, end_month: int) -> Iterator[Tuple[int, int]]:
    """Inclusive month iterator."""
    y, m = int(start_year), int(start_month)
    while (y, m) <= (int(end_year), int(end_month)):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def add_months(year: int, month: int, k: int) -> Tuple[int, int]:
    idx = (int(year) * 12 + (int(month) - 1)) + int(k)
    return idx // 12, idx % 12 + 1


# ---------------------------------------------------------------------------
# EPİAŞ VEP contract codes
# ---------------------------------------------------------------------------
def vep_contract_code(year: int, month: int) -> str:
    """EPİAŞ monthly baseload code, e.g. (2026, 2) -> 'EBM0226'."""
    return f"EBM{_MONTH_CODES[int(month)]}{int(year) % 100:02d}"


def parse_vep_contract_code(code: str) -> Tuple[int, int]:
    """'EBM0226' -> (2026, 2).  Century pinned to 2000-2099."""
    code = str(code).strip().upper()
    if not code.startswith("EBM") or len(code) != 7 or not code[3:].isdigit():
        raise ValueError(f"not a monthly baseload VEP code: {code!r} "
                         "(expected EBM<MM><YY>, e.g. EBM0226)")
    month, yy = int(code[3:5]), int(code[5:7])
    if not 1 <= month <= 12:
        raise ValueError(f"invalid month in VEP code {code!r}")
    return 2000 + yy, month


# ---------------------------------------------------------------------------
# hourly grids
# ---------------------------------------------------------------------------
def hourly_index_utc(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DatetimeIndex:
    """Left-closed hourly UTC index [start, end)."""
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("start and end must be tz-aware")
    if end_utc <= start_utc:
        raise ValueError("end must be strictly after start")
    return pd.date_range(start_utc, end_utc, freq="h", inclusive="left", tz="UTC")


@dataclass(frozen=True)
class DeliveryMonth:
    """A single monthly baseload delivery window."""

    year: int
    month: int

    @property
    def label(self) -> str:
        return month_label(self.year, self.month)

    @property
    def contract_code(self) -> str:
        return vep_contract_code(self.year, self.month)

    @property
    def start_utc(self) -> pd.Timestamp:
        return month_start_utc(self.year, self.month)

    @property
    def end_utc(self) -> pd.Timestamp:
        return month_end_utc(self.year, self.month)

    @property
    def n_hours(self) -> int:
        return hours_in_month(self.year, self.month)

    def hours_utc(self) -> pd.DatetimeIndex:
        return month_delivery_hours_utc(self.year, self.month)

    def contains(self, ts_utc: pd.Timestamp) -> bool:
        return self.start_utc <= ts_utc < self.end_utc

    def mask(self, index_utc: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index_utc >= self.start_utc) &
                          (index_utc < self.end_utc), dtype=bool)


def delivery_months_between(start_utc: pd.Timestamp,
                            end_utc: pd.Timestamp) -> List[DeliveryMonth]:
    """Every delivery month overlapping [start, end)."""
    y0, m0 = month_of(start_utc)
    y1, m1 = month_of(end_utc - pd.Timedelta(hours=1))
    return [DeliveryMonth(y, m) for y, m in iter_months(y0, m0, y1, m1)]
