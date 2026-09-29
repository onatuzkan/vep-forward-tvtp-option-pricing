"""FW10b -- production-style hourly forward curve builder for any
Turkish business day d.

The helper builds the exact same forward curve the shipped pricer
uses (smooth constrained QP, near-term anchor spot_to_next_linear,
HPFC shape applied inside every quoted month) but with the FW10b
corrected timing rule:

  * Valuation timestamp   d 11:00 TRT.
  * Last known PTF hour   d 23:00 TRT (published on d-1 at ~14:00 TRT).
  * Spot                  PTF at last_known.
  * VEP quote day used    latest VEP GGF publication STRICTLY before
                          d 11:00 TRT.  d's own GGF publishes at
                          end-of-day, so this is d-1 or earlier.

The HPFC shape model is fit ONCE from PTF data up to the frozen
2025-12-31 20:00 UTC boundary using the production settings
(``half_life_years = 0.5``, ``n_harmonics = 2``, matching the shipped
``outputs/hpfc/hpfc_selection.json`` selection).  It is then applied
inside every quoted month of every daily curve.

Only monthly baseload EBM contracts are used as constraints.  If the
target-hour falls inside a delivery month that is already IN PROGRESS
at the valuation date, that partial month has no observed baseload
quote; the target hour is then anchored by ``spot_to_next_linear``
between the spot and the first quoted future month.  This is the
exact same rule the production curve uses for the partial current
month of the shipped 2025-12-31 valuation.
"""
from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pde_option_model.calendar_tr import TURKEY_TZ
from pde_option_model.forward_curve import (ForwardCurve, NearTermAnchor,
                                              build_forward_curve)
from pde_option_model.hpfc import ShapeModel, apply_shape_to_curve, fit_shape
from pde_option_model.market_data import (MarketQuoteSet,
                                            MonthlyBaseloadQuote)

from scripts.fw10._data import (day_end_utc, load_realized_ptf,
                                 load_vep_quotes_daily, valuation_utc,
                                 FREEZE_UTC)

HPFC_HALF_LIFE_YEARS = 0.5
HPFC_N_HARMONICS = 2
HPFC_CUTOFF_UTC = FREEZE_UTC  # 2025-12-31 20:00 UTC; matches the frozen
                              # calibration boundary.  Shipped
                              # hpfc_selection.json used 21:00 UTC (one
                              # additional hour of 2025 data), so this
                              # is even MORE conservative.


@functools.lru_cache(maxsize=1)
def _load_hpfc_shape() -> ShapeModel:
    """Fit the production HPFC shape once, from data before FREEZE_UTC.

    Uses the 2019-2025 historical PTF archive; the fit is strictly on
    hours before ``HPFC_CUTOFF_UTC`` = 2025-12-31 20:00 UTC.  No 2026
    data enters the shape.
    """
    ptf = load_realized_ptf(include_history=True)
    ptf_pre = ptf[ptf.index <= HPFC_CUTOFF_UTC]
    return fit_shape(ptf_pre, cutoff_utc=HPFC_CUTOFF_UTC,
                     half_life_years=HPFC_HALF_LIFE_YEARS,
                     n_harmonics=HPFC_N_HARMONICS)


def _last_vep_quote_day_before(vep: pd.DataFrame,
                                 target_utc: pd.Timestamp) -> Optional[str]:
    """Return the YYYY-MM-DD VEP quotation day strictly before
    ``target_utc``.  VEP GGF publishes on trading days at end-of-day,
    so a valuation at day d 11:00 TRT can only see d-1's GGF or
    earlier.
    """
    target_local_date = target_utc.tz_convert(TURKEY_TZ).strftime("%Y-%m-%d")
    days = sorted(vep["valuation_date"].unique())
    earlier = [x for x in days if x < target_local_date]
    return earlier[-1] if earlier else None


def _horizon_end_utc(quotes_slice: pd.DataFrame) -> pd.Timestamp:
    """End-of-latest-delivery-month, TRT-local, in UTC."""
    last = quotes_slice.sort_values(
        ["delivery_year", "delivery_month"]).iloc[-1]
    y, m = int(last["delivery_year"]), int(last["delivery_month"])
    if m == 12:
        end_local = pd.Timestamp(f"{y+1}-01-01 00:00", tz=TURKEY_TZ)
    else:
        end_local = pd.Timestamp(f"{y}-{m+1:02d}-01 00:00", tz=TURKEY_TZ)
    return end_local.tz_convert("UTC")


@dataclass(frozen=True)
class DailyCurveResult:
    valuation_utc: pd.Timestamp
    last_known_utc: pd.Timestamp
    vep_quote_day: str
    spot_TRY_MWh: float
    hourly_forward_TRY_MWh: pd.Series
    quote_set: MarketQuoteSet
    hpfc_applied: bool


def build_daily_production_curve(d_local: pd.Timestamp,
                                  ptf: Optional[pd.Series] = None,
                                  vep: Optional[pd.DataFrame] = None,
                                  hpfc: bool = True
                                  ) -> Optional[DailyCurveResult]:
    """Return the production hourly forward curve for day ``d_local``.

    ``d_local`` is a tz-aware Europe/Istanbul timestamp naming the
    valuation day.  Returns ``None`` if either the spot price or the
    VEP quotation day is unavailable.
    """
    if ptf is None:
        ptf = load_realized_ptf()
    if vep is None:
        vep = load_vep_quotes_daily()

    val_utc = valuation_utc(d_local)
    last_known = day_end_utc(d_local)
    if last_known not in ptf.index:
        return None
    spot = float(ptf.loc[last_known])
    qday = _last_vep_quote_day_before(vep, val_utc)
    if qday is None:
        return None
    slice_ = vep[vep["valuation_date"] == qday].reset_index(drop=True)

    # Drop contracts whose delivery window has already ended by
    # valuation (MarketQuoteSet forbids stale contracts)
    keep_rows = []
    for _, r in slice_.iterrows():
        y, m = int(r["delivery_year"]), int(r["delivery_month"])
        if m == 12:
            end_local = pd.Timestamp(f"{y+1}-01-01 00:00", tz=TURKEY_TZ)
        else:
            end_local = pd.Timestamp(f"{y}-{m+1:02d}-01 00:00", tz=TURKEY_TZ)
        end_utc = end_local.tz_convert("UTC")
        if end_utc > val_utc:
            keep_rows.append(r)
    if not keep_rows:
        return None
    slice_ = pd.DataFrame(keep_rows).reset_index(drop=True)

    quotes: List[MonthlyBaseloadQuote] = []
    for _, r in slice_.iterrows():
        y, m = int(r["delivery_year"]), int(r["delivery_month"])
        quotes.append(MonthlyBaseloadQuote(
            contract_name=str(r["contract_name"]),
            year=y, month=m,
            price_TRY_MWh=float(r["price_TRY_MWh"]),
            source="EPIAS_VEP_daily_reference_price",
            quote_type="monthly_baseload", weight=1.0))
    quote_set = MarketQuoteSet(
        valuation_utc=val_utc, quotes=quotes,
        spot_price_TRY_MWh=spot,
        notes={"vep_quote_day": qday, "d_local": str(d_local)})

    anchor = NearTermAnchor(mode="spot_to_next_linear")
    horizon_end = _horizon_end_utc(slice_)
    curve = build_forward_curve(quote_set, mode="smooth_constrained",
                                 anchor=anchor,
                                 spot_price_TRY_MWh=spot,
                                 horizon_end_utc=horizon_end)
    if hpfc:
        shape = _load_hpfc_shape()
        cframe = curve.frame.copy()
        cframe = cframe.rename(columns={"time_utc": "time_utc"})
        # apply_shape_to_curve expects the "time_utc" and
        # "hourly_forward_TRY_MWh" columns; the curve frame already
        # has them.
        hpfc_df = apply_shape_to_curve(cframe, shape,
                                        time_col="time_utc",
                                        fwd_col="hourly_forward_TRY_MWh")
        hpfc_series = pd.Series(
            hpfc_df["hpfc_TRY_MWh"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(
                hpfc_df["time_utc"], utc=True)),
            name="hourly_forward_TRY_MWh_hpfc")
        forward_series = hpfc_series
    else:
        forward_series = curve.values

    return DailyCurveResult(
        valuation_utc=val_utc, last_known_utc=last_known,
        vep_quote_day=qday, spot_TRY_MWh=spot,
        hourly_forward_TRY_MWh=forward_series,
        quote_set=quote_set, hpfc_applied=bool(hpfc))


def F_at_hours(result: DailyCurveResult, hours: List[int]) -> Dict[int, float]:
    """Return F(last_known + h) for h in hours (in whole hours)."""
    out = {}
    for h in hours:
        t = result.last_known_utc + pd.Timedelta(hours=int(h))
        # nearest hourly node (curve is hourly)
        if t in result.hourly_forward_TRY_MWh.index:
            out[h] = float(result.hourly_forward_TRY_MWh.loc[t])
        else:
            # exact hour must exist in an hourly curve; fall back to
            # interpolation from the nearest neighbours
            s = result.hourly_forward_TRY_MWh
            i = s.index.searchsorted(t)
            if 0 < i < len(s):
                t0, t1 = s.index[i - 1], s.index[i]
                v0, v1 = float(s.iloc[i - 1]), float(s.iloc[i])
                w = (t - t0).total_seconds() / (t1 - t0).total_seconds()
                out[h] = float(v0 + w * (v1 - v0))
            else:
                out[h] = float("nan")
    return out


def dF_target_dF_month_bump(d_local: pd.Timestamp, target_utc: pd.Timestamp,
                              delivery_year_month: Tuple[int, int],
                              bump_TRY_MWh: float = 10.0,
                              ptf: Optional[pd.Series] = None,
                              vep: Optional[pd.DataFrame] = None,
                              hpfc: bool = True) -> Optional[float]:
    """Sensitivity dF(target)/dF_month for the given delivery month.

    Positive-bump the delivery-month monthly quote by
    ``bump_TRY_MWh``, rebuild the curve, and read the shift in
    ``F(target)``.  A value near 1 means the monthly move is fully
    passed through to the delivery hour; a value near 0 means the
    delivery hour is dominated by the spot anchor.
    """
    if ptf is None:
        ptf = load_realized_ptf()
    if vep is None:
        vep = load_vep_quotes_daily()
    base = build_daily_production_curve(d_local, ptf, vep, hpfc=hpfc)
    if base is None:
        return None
    # Locate the base target F(target) via nearest hour
    def F_at(res: DailyCurveResult) -> float:
        s = res.hourly_forward_TRY_MWh
        if target_utc in s.index:
            return float(s.loc[target_utc])
        i = s.index.searchsorted(target_utc)
        if 0 < i < len(s):
            t0, t1 = s.index[i - 1], s.index[i]
            v0, v1 = float(s.iloc[i - 1]), float(s.iloc[i])
            w = (target_utc - t0).total_seconds() / (t1 - t0).total_seconds()
            return float(v0 + w * (v1 - v0))
        return float("nan")
    F0 = F_at(base)

    # Bump: build a modified VEP frame with the delivery-month price
    # shifted by +bump_TRY_MWh
    vep_bumped = vep.copy()
    y, m = delivery_year_month
    mask = ((vep_bumped["valuation_date"] == base.vep_quote_day)
            & (vep_bumped["delivery_year"] == y)
            & (vep_bumped["delivery_month"] == m))
    if not mask.any():
        return None
    vep_bumped.loc[mask, "price_TRY_MWh"] = (
        vep_bumped.loc[mask, "price_TRY_MWh"] + bump_TRY_MWh)
    bumped = build_daily_production_curve(d_local, ptf, vep_bumped, hpfc=hpfc)
    if bumped is None:
        return None
    F1 = F_at(bumped)
    return (F1 - F0) / bump_TRY_MWh
