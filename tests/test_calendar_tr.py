"""Turkish delivery calendar: UTC+3 fixed, month boundaries, delivery hours."""
from __future__ import annotations

import pandas as pd
import pytest

from pde_option_model import calendar_tr as cal


def test_february_2026_has_672_delivery_hours():
    """2026 is NOT a leap year: February has 28 days = 672 delivery hours."""
    assert cal.days_in_month(2026, 2) == 28
    assert cal.hours_in_month(2026, 2) == 672
    assert len(cal.month_delivery_hours_utc(2026, 2)) == 672


def test_february_2024_leap_year_contrast():
    assert cal.hours_in_month(2024, 2) == 696
    assert cal.hours_in_month(2100, 2) == 672        # century non-leap


@pytest.mark.parametrize("month,hours", [(1, 744), (2, 672), (3, 744), (4, 720),
                                         (5, 744), (6, 720), (7, 744), (8, 744),
                                         (9, 720), (10, 744), (11, 720), (12, 744)])
def test_all_2026_month_hour_counts(month, hours):
    assert cal.hours_in_month(2026, month) == hours
    assert len(cal.month_delivery_hours_utc(2026, month)) == hours


def test_utc_to_turkey_conversion_is_fixed_plus_three():
    for m in range(1, 13):
        ts = pd.Timestamp(f"2026-{m:02d}-15T12:00:00Z")
        assert cal.to_turkey(ts).hour == 15, "Türkiye must be UTC+3 in every month"


def test_no_daylight_saving_shift_across_european_dst_dates():
    """European DST switches in late March / late October must not move Türkiye."""
    before = cal.to_turkey(pd.Timestamp("2026-03-28T12:00:00Z"))
    after = cal.to_turkey(pd.Timestamp("2026-03-30T12:00:00Z"))
    assert before.hour == after.hour == 15
    before = cal.to_turkey(pd.Timestamp("2026-10-24T12:00:00Z"))
    after = cal.to_turkey(pd.Timestamp("2026-10-26T12:00:00Z"))
    assert before.hour == after.hour == 15


def test_roundtrip_utc_turkey():
    ts = pd.Timestamp("2026-04-17T08:00:00Z")
    assert cal.to_utc(cal.to_turkey(ts)) == ts


def test_month_start_and_end_are_local_midnight():
    start = cal.month_start_utc(2026, 2)
    end = cal.month_end_utc(2026, 2)
    assert start == pd.Timestamp("2026-01-31T21:00:00Z")
    assert end == pd.Timestamp("2026-02-28T21:00:00Z")
    assert cal.to_turkey(start).hour == 0 and cal.to_turkey(start).day == 1
    assert cal.to_turkey(end).hour == 0 and cal.to_turkey(end).day == 1


def test_month_end_equals_next_month_start():
    for m in range(1, 12):
        assert cal.month_end_utc(2026, m) == cal.month_start_utc(2026, m + 1)
    assert cal.month_end_utc(2026, 12) == cal.month_start_utc(2027, 1)


def test_delivery_window_is_left_closed_right_open():
    hrs = cal.month_delivery_hours_utc(2026, 2)
    assert hrs[0] == cal.month_start_utc(2026, 2)
    assert hrs[-1] == cal.month_end_utc(2026, 2) - pd.Timedelta(hours=1)


def test_vep_contract_code_roundtrip():
    assert cal.vep_contract_code(2026, 2) == "EBM0226"
    assert cal.vep_contract_code(2026, 12) == "EBM1226"
    assert cal.parse_vep_contract_code("EBM0226") == (2026, 2)
    assert cal.parse_vep_contract_code("EBM0726") == (2026, 7)


@pytest.mark.parametrize("bad", ["EBM026", "XYZ0226", "EBM1326", "EBM0026"])
def test_bad_contract_codes_rejected(bad):
    with pytest.raises(ValueError):
        cal.parse_vep_contract_code(bad)


def test_delivery_month_dataclass_and_mask():
    dm = cal.DeliveryMonth(2026, 2)
    assert (dm.label, dm.contract_code, dm.n_hours) == ("2026-02", "EBM0226", 672)
    idx = cal.hourly_index_utc(pd.Timestamp("2026-01-25T00:00:00Z"),
                               pd.Timestamp("2026-03-05T00:00:00Z"))
    assert dm.mask(idx).sum() == 672


def test_delivery_months_between_covers_the_valuation_stub():
    months = cal.delivery_months_between(pd.Timestamp("2025-12-31T20:00:00Z"),
                                         pd.Timestamp("2026-07-31T21:00:00Z"))
    assert [m.label for m in months][:3] == ["2025-12", "2026-01", "2026-02"]


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError):
        cal.to_turkey(pd.Timestamp("2026-01-01T00:00:00"))
