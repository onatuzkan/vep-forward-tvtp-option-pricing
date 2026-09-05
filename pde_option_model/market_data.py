"""EPİAŞ VEP monthly baseload quotes: schema, loading and validation.

A VEP monthly contract is **not** a point-maturity forward.  Its price is the
arithmetic baseload average of the hourly prices over the whole delivery month:

    F_m = (1 / N_m) * sum_{h in month m} E^Q[P_h]

where N_m is the true number of Turkish local delivery hours (days * 24).
Treating such a quote as E^Q[P_T] at a single maturity T (month start, middle
or end) is a modelling error; :class:`MonthlyQuoteMisuseError` is raised
whenever the code is asked to do that.

Supported input formats
-----------------------
CSV  ``inputs/market/vep_monthly_quotes.csv`` with columns

    contract_name, delivery_year, delivery_month, price_TRY_MWh,
    quote_type, source, valuation_date [, weight]

JSON ``inputs/market/vep_2025-12-31.json`` with the same information plus
metadata.  Both formats reserve an optional ``JANUARY``-style near-term row
(``quote_type: monthly_baseload``) so that a January 2026 or day-ahead
baseload quote can be dropped in later without any code change.

Units: every price is TRY/MWh, baseload (unweighted mean over delivery hours).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .calendar_tr import (DeliveryMonth, month_label, parse_vep_contract_code,
                          vep_contract_code)

logger = logging.getLogger(__name__)

QUOTE_TYPE_MONTHLY = "monthly_baseload"
REQUIRED_CSV_COLUMNS = (
    "contract_name", "delivery_year", "delivery_month",
    "price_TRY_MWh", "quote_type", "source",
)


class MonthlyQuoteMisuseError(ValueError):
    """Raised when a monthly baseload quote is used as a point-maturity forward."""


class MarketDataError(ValueError):
    """Raised on malformed or internally inconsistent market input."""


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MonthlyBaseloadQuote:
    """One observed monthly baseload contract (TRY/MWh)."""

    contract_name: str
    year: int
    month: int
    price_TRY_MWh: float
    source: str = "EPIAS_VEP"
    quote_type: str = QUOTE_TYPE_MONTHLY
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.quote_type != QUOTE_TYPE_MONTHLY:
            raise MarketDataError(
                f"{self.contract_name}: unsupported quote_type {self.quote_type!r}; "
                f"only {QUOTE_TYPE_MONTHLY!r} is implemented")
        if not (self.price_TRY_MWh == self.price_TRY_MWh):     # NaN guard
            raise MarketDataError(f"{self.contract_name}: price is NaN")
        if self.price_TRY_MWh <= 0:
            raise MarketDataError(
                f"{self.contract_name}: non-positive price {self.price_TRY_MWh}")
        if self.weight <= 0:
            raise MarketDataError(f"{self.contract_name}: weight must be positive")
        if not 1 <= int(self.month) <= 12:
            raise MarketDataError(f"{self.contract_name}: bad month {self.month}")

    # -- delivery window ---------------------------------------------------
    @property
    def delivery(self) -> DeliveryMonth:
        return DeliveryMonth(int(self.year), int(self.month))

    @property
    def label(self) -> str:
        return month_label(self.year, self.month)

    @property
    def n_delivery_hours(self) -> int:
        return self.delivery.n_hours

    @property
    def delivery_start_utc(self) -> pd.Timestamp:
        return self.delivery.start_utc

    @property
    def delivery_end_utc(self) -> pd.Timestamp:
        return self.delivery.end_utc

    # -- explicit misuse guard --------------------------------------------
    @property
    def maturity_utc(self) -> pd.Timestamp:                     # pragma: no cover
        raise MonthlyQuoteMisuseError(
            f"{self.contract_name} is a monthly baseload contract covering "
            f"{self.n_delivery_hours} delivery hours "
            f"[{self.delivery_start_utc} .. {self.delivery_end_utc}); it has no "
            "single maturity. Use delivery_start_utc/delivery_end_utc and the "
            "delivery-average constraint F_m = mean_h E^Q[P_h] instead.")

    def as_point_forward(self, *_: Any, **__: Any):              # pragma: no cover
        raise MonthlyQuoteMisuseError(
            f"{self.contract_name}: refusing to collapse a monthly baseload "
            "average into a point-maturity forward quote.")

    def delivery_average(self, curve: pd.Series) -> float:
        """Baseload average of an hourly UTC-indexed curve over this month."""
        idx = self.delivery.hours_utc()
        missing = idx.difference(curve.index)
        if len(missing):
            raise MarketDataError(
                f"{self.contract_name}: hourly curve does not cover the whole "
                f"delivery month ({len(missing)} of {len(idx)} hours missing, "
                f"first {missing[0]})")
        return float(curve.reindex(idx).to_numpy().mean())

    def to_row(self) -> Dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "delivery_year": int(self.year),
            "delivery_month": int(self.month),
            "delivery_month_label": self.label,
            "price_TRY_MWh": float(self.price_TRY_MWh),
            "quote_type": self.quote_type,
            "source": self.source,
            "weight": float(self.weight),
            "n_delivery_hours": self.n_delivery_hours,
            "delivery_start_utc": self.delivery_start_utc.isoformat(),
            "delivery_end_utc": self.delivery_end_utc.isoformat(),
        }


# ---------------------------------------------------------------------------
@dataclass
class MarketQuoteSet:
    """Validated set of monthly baseload quotes for one valuation date."""

    valuation_utc: pd.Timestamp
    quotes: List[MonthlyBaseloadQuote]
    quote_source: str = "EPIAS_VEP_daily_reference_prices"
    spot_price_TRY_MWh: Optional[float] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valuation_utc.tzinfo is None:
            raise MarketDataError("valuation_utc must be tz-aware")
        if not self.quotes:
            raise MarketDataError("no monthly quotes supplied")
        labels = [q.label for q in self.quotes]
        dup = {l for l in labels if labels.count(l) > 1}
        if dup:
            raise MarketDataError(f"duplicate delivery months: {sorted(dup)}")
        self.quotes = sorted(self.quotes, key=lambda q: (q.year, q.month))
        for q in self.quotes:
            if q.delivery_end_utc <= self.valuation_utc:
                raise MarketDataError(
                    f"{q.contract_name}: delivery window ends at "
                    f"{q.delivery_end_utc}, before valuation {self.valuation_utc}")

    # -- accessors ---------------------------------------------------------
    @property
    def labels(self) -> List[str]:
        return [q.label for q in self.quotes]

    @property
    def first_delivery_utc(self) -> pd.Timestamp:
        return self.quotes[0].delivery_start_utc

    @property
    def last_delivery_utc(self) -> pd.Timestamp:
        return self.quotes[-1].delivery_end_utc

    def get(self, year: int, month: int) -> Optional[MonthlyBaseloadQuote]:
        for q in self.quotes:
            if (q.year, q.month) == (int(year), int(month)):
                return q
        return None

    def has(self, year: int, month: int) -> bool:
        return self.get(year, month) is not None

    # -- validation --------------------------------------------------------
    def partial_months(self) -> List[str]:
        """Months the valuation timestamp falls strictly inside.

        Valuing at 2025-12-31 20:00 UTC (= 23:00 Turkish time) leaves a
        one-hour sliver of the 2025-12 delivery month inside the horizon.  Such
        a stub is never a calibration target: only the hours from valuation
        onward exist, so no monthly baseload average can be formed for it.
        """
        from .calendar_tr import delivery_months_between
        out: List[str] = []
        for dm in delivery_months_between(self.valuation_utc, self.last_delivery_utc):
            if dm.start_utc < self.valuation_utc < dm.end_utc:
                out.append(dm.label)
        return out

    def missing_months(self, include_partial: bool = False) -> List[str]:
        """Full delivery months inside the horizon WITHOUT an observed quote.

        For the 2025-12-31 VEP set this returns ``['2026-01']``: January 2026 is
        inside the pricing horizon but has no observed monthly contract, so any
        January value is an anchored extrapolation, never a market constraint.
        """
        from .calendar_tr import delivery_months_between
        partial = set(self.partial_months())
        out: List[str] = []
        for dm in delivery_months_between(self.valuation_utc, self.last_delivery_utc):
            if self.has(dm.year, dm.month):
                continue
            if dm.label in partial and not include_partial:
                continue
            out.append(dm.label)
        return out

    def gap_months(self) -> List[str]:
        """Months missing *between* two quoted months (interior holes)."""
        from .calendar_tr import iter_months
        first, last = self.quotes[0], self.quotes[-1]
        return [month_label(y, m)
                for y, m in iter_months(first.year, first.month, last.year, last.month)
                if not self.has(y, m)]

    def validate(self, strict_interior: bool = True) -> List[str]:
        """Return warnings; raise on interior gaps when ``strict_interior``."""
        warns: List[str] = []
        part = self.partial_months()
        if part:
            warns.append(f"valuation falls inside delivery month(s) {part}; only the "
                         "hours from valuation onward are priced and no monthly "
                         "baseload average is formed for them")
        gaps = self.gap_months()
        if gaps:
            msg = (f"interior delivery months without a quote: {gaps}; a monthly "
                   "curve cannot be built across an unquoted interior month "
                   "without an explicit interpolation rule")
            if strict_interior:
                raise MarketDataError(msg)
            warns.append(msg)
        missing = self.missing_months()
        if missing:
            warns.append(
                f"delivery months inside the pricing horizon with NO observed "
                f"quote: {missing} -> near-term anchored, not directly "
                "constrained by an observed VEP quote")
        for q in self.quotes:
            code = vep_contract_code(q.year, q.month)
            if q.contract_name.upper() != code:
                warns.append(f"{q.contract_name}: expected contract code {code} "
                             f"for delivery {q.label}")
        for w in warns:
            logger.warning("market data: %s", w)
        return warns

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([q.to_row() for q in self.quotes])


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def _coerce_year_month(rec: Dict[str, Any]) -> tuple[int, int]:
    if rec.get("delivery_year") is not None and rec.get("delivery_month") is not None:
        return int(rec["delivery_year"]), int(rec["delivery_month"])
    name = rec.get("contract_name")
    if name:
        return parse_vep_contract_code(str(name))
    raise MarketDataError(f"cannot determine delivery month from record {rec!r}")


def load_quotes_csv(path: str | Path,
                    valuation_utc: Optional[pd.Timestamp] = None,
                    spot_price_TRY_MWh: Optional[float] = None) -> MarketQuoteSet:
    """Load monthly baseload quotes from CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"quotes file not found: {p}")
    df = pd.read_csv(p)
    missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing_cols:
        raise MarketDataError(f"{p.name}: missing columns {missing_cols}")
    if valuation_utc is None:
        if "valuation_date" not in df.columns:
            raise MarketDataError(
                f"{p.name}: no valuation_date column and no valuation_utc argument")
        vals = sorted(set(df["valuation_date"].astype(str)))
        if len(vals) != 1:
            raise MarketDataError(f"{p.name}: mixed valuation dates {vals}")
        valuation_utc = _parse_valuation(vals[0])
    quotes = [
        MonthlyBaseloadQuote(
            contract_name=str(r["contract_name"]).strip(),
            year=y, month=m,
            price_TRY_MWh=float(r["price_TRY_MWh"]),
            source=str(r["source"]),
            quote_type=str(r["quote_type"]),
            weight=float(r.get("weight", 1.0) or 1.0),
        )
        for r, (y, m) in ((r, _coerce_year_month(dict(r)))
                          for _, r in df.iterrows())
    ]
    if spot_price_TRY_MWh is None and "spot_price_TRY_MWh" in df.columns:
        s = df["spot_price_TRY_MWh"].dropna()
        if len(s):
            spot_price_TRY_MWh = float(s.iloc[0])
    qs = MarketQuoteSet(valuation_utc=valuation_utc, quotes=quotes,
                        spot_price_TRY_MWh=spot_price_TRY_MWh,
                        notes={"file": str(p)})
    logger.info("loaded %d monthly quotes from %s (valuation %s)",
                len(quotes), p.name, valuation_utc)
    return qs


def load_quotes_json(path: str | Path) -> MarketQuoteSet:
    """Load monthly baseload quotes from the richer JSON format."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"quotes file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    for key in ("valuation_date", "quotes"):
        if key not in blob:
            raise MarketDataError(f"{p.name}: missing top-level key {key!r}")
    valuation = _parse_valuation(blob.get("valuation_utc") or blob["valuation_date"])
    quotes: List[MonthlyBaseloadQuote] = []
    for rec in blob["quotes"]:
        y, m = _coerce_year_month(rec)
        quotes.append(MonthlyBaseloadQuote(
            contract_name=str(rec["contract_name"]).strip(), year=y, month=m,
            price_TRY_MWh=float(rec["price_TRY_MWh"]),
            source=str(rec.get("source", blob.get("quote_source", "EPIAS_VEP"))),
            quote_type=str(rec.get("quote_type", QUOTE_TYPE_MONTHLY)),
            weight=float(rec.get("weight", 1.0)),
        ))
    qs = MarketQuoteSet(
        valuation_utc=valuation, quotes=quotes,
        quote_source=str(blob.get("quote_source", "EPIAS_VEP_daily_reference_prices")),
        spot_price_TRY_MWh=(None if blob.get("spot_price_TRY_MWh") is None
                            else float(blob["spot_price_TRY_MWh"])),
        notes={k: v for k, v in blob.items() if k not in ("quotes",)},
    )
    logger.info("loaded %d monthly quotes from %s (valuation %s)",
                len(quotes), p.name, valuation)
    return qs


def load_quotes(path: str | Path, **kwargs: Any) -> MarketQuoteSet:
    """Dispatch on file suffix (.csv or .json)."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return load_quotes_json(p)
    if p.suffix.lower() in (".csv", ".txt"):
        return load_quotes_csv(p, **kwargs)
    raise MarketDataError(f"unsupported quotes file type: {p.suffix!r}")


def _parse_valuation(value: str) -> pd.Timestamp:
    """Parse a valuation stamp; bare dates are read as the Turkish day end.

    A bare '2025-12-31' means the *end* of that Turkish trading day, i.e.
    2025-12-31 23:00 TRT = 2025-12-31 20:00 UTC, which is the last hourly
    observation of the historical sample.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and len(str(value)) <= 10:
            from .calendar_tr import turkey_local
            return turkey_local(ts.year, ts.month, ts.day, 23).tz_convert("UTC")
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def assert_not_point_forward(obj: Any) -> None:
    """Guard used by pricing code paths that expect point-maturity forwards."""
    if isinstance(obj, MonthlyBaseloadQuote):
        raise MonthlyQuoteMisuseError(
            f"{obj.contract_name} is a monthly baseload average over "
            f"{obj.n_delivery_hours} hours and must not be used as a "
            "point-maturity forward quote.")
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for item in obj:
            assert_not_point_forward(item)
