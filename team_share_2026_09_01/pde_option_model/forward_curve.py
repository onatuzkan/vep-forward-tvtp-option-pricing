"""Deterministic hourly market forward curve F(t) built from monthly VEP quotes.

The curve is the market-observed level around which the risk-neutral residual
process is centred.  Its defining property is the *delivery-average*
constraint, one per observed monthly baseload contract:

    (1 / N_m) * sum_{h in delivery month m} F(h)  =  VEP_m          (exact)

Two construction modes
----------------------
``piecewise_constant``
    F(h) = VEP_m for every hour of month m.  The constraint holds identically
    (residual = 0 by construction).  Deliberately blunt: no intra-month shape.

``smooth_constrained``
    F minimises a roughness + level-anchoring functional subject to the SAME
    equality constraints, solved as one sparse KKT system:

        min_F   w_smooth * || D2 F ||^2  +  w_level * || F - F_pwc ||^2
        s.t.    A F = b

    where D2 is the second-difference operator, F_pwc the piecewise-constant
    baseline, and each row of A is the delivery-hour averaging operator of one
    contract (plus the near-term anchor row).  Because the constraints enter as
    exact linear equalities via Lagrange multipliers, the monthly averages are
    reproduced to solver precision (~1e-9 TRY/MWh), NOT approximately.
    Smoothing therefore can never break the monthly constraints.

Near-term (January 2026) treatment
----------------------------------
The 2025-12-31 VEP strip starts at EBM0226, so no observed contract constrains
January 2026.  Every hour before the first quoted delivery month is flagged

    extrapolation_flag = True, near_term_anchor_flag = True,
    source = "near_term_anchor:<mode>"

and is produced by an explicit, configurable rule (:class:`NearTermAnchor`)
rather than by silent extrapolation.  Consistency requirement: because the
spot at the valuation date is *known*, E^Q[P_0] = P_spot, so the curve must
satisfy F(t_0) = P_spot whenever the residual starts at zero.  All anchor modes
except ``flat_next_month``/``explicit_level`` enforce this exactly.

Units: F is TRY/MWh throughout.  Time is UTC; delivery months are Turkish local.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .calendar_tr import (DeliveryMonth, delivery_months_between, hourly_index_utc,
                          month_label, to_turkey)
from .market_data import MarketQuoteSet, MonthlyBaseloadQuote

logger = logging.getLogger(__name__)

CurveMode = Literal["piecewise_constant", "smooth_constrained"]
AnchorMode = Literal["spot_flat", "spot_to_next_linear", "flat_next_month",
                     "explicit_level"]

CURVE_COLUMNS: Tuple[str, ...] = (
    "time_utc", "time_turkey", "delivery_month", "contract_name",
    "hourly_forward_TRY_MWh", "source", "interpolation_flag",
    "extrapolation_flag", "near_term_anchor_flag",
)

MAX_PLAUSIBLE_FORWARD_TRY_MWh: float = 1.0e5


class ForwardCurveError(ValueError):
    """Raised when a curve cannot be built or violates its own constraints."""


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NearTermAnchor:
    """Rule for the unquoted near-term window (January 2026 here).

    mode
      ``spot_flat``            F(t) = spot for the whole unquoted window
      ``spot_to_next_linear``  linear ramp from spot at t0 to the first quoted
                               monthly level at the start of that month
      ``flat_next_month``      F(t) = first quoted monthly level (ignores spot)
      ``explicit_level``       F(t) = ``level_TRY_MWh`` (a user-supplied
                               January baseload assumption)
    """

    mode: AnchorMode = "spot_to_next_linear"
    level_TRY_MWh: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode == "explicit_level" and self.level_TRY_MWh is None:
            raise ForwardCurveError("anchor mode 'explicit_level' requires level_TRY_MWh")
        if self.level_TRY_MWh is not None and not np.isfinite(self.level_TRY_MWh):
            raise ForwardCurveError("anchor level must be finite")

    @property
    def pins_spot(self) -> bool:
        """True if the rule forces F(t_0) = spot."""
        return self.mode in ("spot_flat", "spot_to_next_linear")

    def describe(self) -> str:
        base = f"near_term_anchor:{self.mode}"
        return base if self.level_TRY_MWh is None else f"{base}@{self.level_TRY_MWh:.2f}"


@dataclass
class ForwardCurve:
    """Hourly deterministic market forward curve with full provenance flags."""

    values: pd.Series                       # TRY/MWh, hourly UTC index
    frame: pd.DataFrame                     # the CURVE_COLUMNS schema
    mode: CurveMode
    anchor: NearTermAnchor
    quotes: MarketQuoteSet
    constrained_months: List[str]
    extrapolated_months: List[str]
    diagnostics: Dict[str, float] = field(default_factory=dict)

    # -- basics ------------------------------------------------------------
    @property
    def index(self) -> pd.DatetimeIndex:
        return self.values.index

    @property
    def start_utc(self) -> pd.Timestamp:
        return self.values.index[0]

    @property
    def end_utc(self) -> pd.Timestamp:
        return self.values.index[-1]

    @staticmethod
    def _epoch_ns(idx: pd.DatetimeIndex) -> np.ndarray:
        """Nanoseconds since epoch, unit-normalised.

        pandas resolves datetime64 to different units ('us' from date_range,
        'ns' after adding a Timedelta), so both sides must be pinned to the
        same unit before any integer comparison.
        """
        return np.asarray(idx.as_unit("ns").astype("int64"), dtype=float)

    def at(self, ts_utc: pd.Timestamp | pd.DatetimeIndex) -> np.ndarray | float:
        """F(t) with linear interpolation between hourly nodes; no extrapolation."""
        x = self._epoch_ns(pd.DatetimeIndex(self.values.index))
        single = not isinstance(ts_utc, pd.DatetimeIndex)
        target = pd.DatetimeIndex([ts_utc] if single else ts_utc)
        if target.tz is None:
            raise ForwardCurveError("query timestamps must be tz-aware")
        xt = self._epoch_ns(target)
        tol_ns = 1.0e3          # 1 microsecond of float rounding slack
        if xt.min() < x.min() - tol_ns or xt.max() > x.max() + tol_ns:
            raise ForwardCurveError(
                f"requested time outside the curve horizon "
                f"[{self.start_utc} .. {self.end_utc}]")
        out = np.interp(xt, x, self.values.to_numpy(dtype=float))
        return float(out[0]) if single else out

    def at_hours(self, valuation_utc: pd.Timestamp,
                 hours: Sequence[float] | np.ndarray) -> np.ndarray:
        """F(valuation + h) for solver times expressed in hours."""
        h = np.asarray(hours, dtype=float)
        target = pd.DatetimeIndex(
            [valuation_utc + pd.Timedelta(hours=float(v)) for v in h], tz="UTC")
        return np.atleast_1d(self.at(target))

    # -- constraint check --------------------------------------------------
    def monthly_averages(self) -> Dict[str, float]:
        """Baseload average of F over every FULL delivery month in the horizon."""
        out: Dict[str, float] = {}
        for dm in delivery_months_between(self.start_utc, self.end_utc + pd.Timedelta(hours=1)):
            if dm.start_utc < self.start_utc or dm.end_utc > self.end_utc + pd.Timedelta(hours=1):
                continue
            hrs = dm.hours_utc()
            miss = hrs.difference(self.values.index)
            if len(miss):
                continue
            out[dm.label] = float(self.values.reindex(hrs).to_numpy().mean())
        return out

    def fit_table(self) -> pd.DataFrame:
        """monthly_forward_fit.csv content (quoted months only)."""
        avgs = self.monthly_averages()
        rows = []
        for q in self.quotes.quotes:
            model = avgs.get(q.label, float("nan"))
            resid = model - q.price_TRY_MWh
            rows.append({
                "contract_name": q.contract_name,
                "delivery_start_utc": q.delivery_start_utc.isoformat(),
                "delivery_end_utc": q.delivery_end_utc.isoformat(),
                "number_of_delivery_hours": q.n_delivery_hours,
                "market_forward_TRY_MWh": round(float(q.price_TRY_MWh), 6),
                "model_average_TRY_MWh": float(model),
                "residual_TRY_MWh": float(resid),
                "relative_error_pct": float(100.0 * resid / q.price_TRY_MWh),
            })
        return pd.DataFrame(rows)

    def max_abs_monthly_error(self) -> float:
        t = self.fit_table()
        return float(np.abs(t["residual_TRY_MWh"].to_numpy()).max()) if len(t) else float("nan")


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def _horizon(quotes: MarketQuoteSet,
             horizon_end_utc: Optional[pd.Timestamp]) -> pd.DatetimeIndex:
    end = horizon_end_utc or quotes.last_delivery_utc
    if end <= quotes.valuation_utc:
        raise ForwardCurveError("curve horizon must extend beyond the valuation date")
    return hourly_index_utc(quotes.valuation_utc, end)


def _piecewise_baseline(index: pd.DatetimeIndex, quotes: MarketQuoteSet,
                        anchor: NearTermAnchor,
                        spot: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Baseline levels, a per-hour 'is anchored' mask, and the anchored labels."""
    F = np.full(len(index), np.nan)
    anchored = np.zeros(len(index), dtype=bool)
    labels: List[str] = []
    first = quotes.quotes[0]
    first_start = first.delivery_start_utc

    for q in quotes.quotes:
        m = q.delivery.mask(index)
        F[m] = q.price_TRY_MWh

    pre = np.asarray(index < first_start, dtype=bool)
    if pre.any():
        anchored |= pre
        for dm in delivery_months_between(index[0], first_start):
            if dm.label not in labels:
                labels.append(dm.label)
        if anchor.mode == "spot_flat":
            F[pre] = spot
        elif anchor.mode == "flat_next_month":
            F[pre] = first.price_TRY_MWh
        elif anchor.mode == "explicit_level":
            F[pre] = float(anchor.level_TRY_MWh)          # type: ignore[arg-type]
        elif anchor.mode == "spot_to_next_linear":
            n = int(pre.sum())
            w = np.arange(n, dtype=float) / max(n, 1)     # 0 at t0, ->1 at month start
            F[pre] = spot + w * (first.price_TRY_MWh - spot)
        else:                                             # pragma: no cover
            raise ForwardCurveError(f"unknown anchor mode {anchor.mode!r}")

    post = np.asarray(index >= quotes.last_delivery_utc, dtype=bool)
    if post.any():
        F[post] = quotes.quotes[-1].price_TRY_MWh
    if np.isnan(F).any():
        bad = index[np.isnan(F)]
        raise ForwardCurveError(
            f"{len(bad)} hours have no quote and no anchor rule (first {bad[0]}); "
            "an interior delivery month is unquoted")
    return F, anchored, labels


def _averaging_rows(index: pd.DatetimeIndex,
                    quotes: MarketQuoteSet) -> Tuple[sp.csr_matrix, np.ndarray, List[str]]:
    """One exact delivery-average equality row per quoted month."""
    n = len(index)
    rows, cols, vals, rhs, labels = [], [], [], [], []
    for r, q in enumerate(quotes.quotes):
        m = np.flatnonzero(q.delivery.mask(index))
        if m.size != q.n_delivery_hours:
            raise ForwardCurveError(
                f"{q.contract_name}: horizon covers {m.size} of {q.n_delivery_hours} "
                "delivery hours; extend the curve horizon")
        rows.extend([r] * m.size)
        cols.extend(m.tolist())
        vals.extend([1.0 / q.n_delivery_hours] * m.size)
        rhs.append(q.price_TRY_MWh)
        labels.append(q.label)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(quotes.quotes), n))
    return A, np.asarray(rhs, dtype=float), labels


def build_forward_curve(
    quotes: MarketQuoteSet,
    mode: CurveMode = "smooth_constrained",
    anchor: Optional[NearTermAnchor] = None,
    spot_price_TRY_MWh: Optional[float] = None,
    horizon_end_utc: Optional[pd.Timestamp] = None,
    smoothness_weight: float = 1.0,
    level_weight: float = 1.0e-4,
    shape_profile: Optional[pd.Series] = None,
    constraint_tolerance_TRY_MWh: float = 1.0e-6,
) -> ForwardCurve:
    """Build the hourly market forward curve.

    Parameters
    ----------
    smoothness_weight, level_weight
        Relative weights of the second-difference roughness penalty and of the
        pull towards the piecewise-constant baseline.  Neither can violate the
        monthly constraints, which are imposed as hard equalities.
    shape_profile
        Optional hourly multiplicative shape (e.g. an hour-of-day baseload
        profile).  It is renormalised inside every delivery month so that the
        monthly average is preserved exactly.
    """
    anchor = anchor or NearTermAnchor()
    spot = spot_price_TRY_MWh if spot_price_TRY_MWh is not None else quotes.spot_price_TRY_MWh
    if spot is None and anchor.pins_spot:
        raise ForwardCurveError(
            f"anchor mode {anchor.mode!r} needs a spot price; supply "
            "spot_price_TRY_MWh or put it in the quote file")
    index = _horizon(quotes, horizon_end_utc)
    baseline, anchored_mask, anchor_labels = _piecewise_baseline(
        index, quotes, anchor, float(spot) if spot is not None else float("nan"))

    if mode == "piecewise_constant":
        F = baseline.copy()
    elif mode == "smooth_constrained":
        F = _solve_smooth(index, baseline, quotes, anchor, spot,
                          smoothness_weight, level_weight)
    else:                                                  # pragma: no cover
        raise ForwardCurveError(f"unknown curve mode {mode!r}")

    if shape_profile is not None:
        F = _apply_shape(index, F, shape_profile, quotes)

    if not np.all(np.isfinite(F)):
        raise ForwardCurveError("forward curve contains non-finite values")
    if np.max(np.abs(F)) > MAX_PLAUSIBLE_FORWARD_TRY_MWh:
        raise ForwardCurveError(
            f"forward curve reached {np.max(np.abs(F)):.6g} TRY/MWh, beyond the "
            f"plausibility cap {MAX_PLAUSIBLE_FORWARD_TRY_MWh:.0g}")

    values = pd.Series(F, index=index, name="hourly_forward_TRY_MWh")
    frame = _build_frame(index, F, quotes, anchored_mask, anchor, mode)
    curve = ForwardCurve(
        values=values, frame=frame, mode=mode, anchor=anchor, quotes=quotes,
        constrained_months=[q.label for q in quotes.quotes],
        extrapolated_months=anchor_labels,
        diagnostics={
            "n_hours": float(len(index)),
            "min_TRY_MWh": float(F.min()),
            "max_TRY_MWh": float(F.max()),
            "mean_TRY_MWh": float(F.mean()),
            "first_value_TRY_MWh": float(F[0]),
            "spot_used_TRY_MWh": float(spot) if spot is not None else float("nan"),
        },
    )
    err = curve.max_abs_monthly_error()
    curve.diagnostics["max_abs_monthly_error_TRY_MWh"] = err
    if not np.isfinite(err) or err > constraint_tolerance_TRY_MWh:
        raise ForwardCurveError(
            f"monthly delivery-average constraints violated: max abs error "
            f"{err:.6g} TRY/MWh > tolerance {constraint_tolerance_TRY_MWh:g}")
    logger.info("forward curve '%s': %d hours, F in [%.2f, %.2f], max monthly "
                "constraint error %.3e TRY/MWh", mode, len(index), F.min(), F.max(), err)
    return curve


def _solve_smooth(index: pd.DatetimeIndex, baseline: np.ndarray,
                  quotes: MarketQuoteSet, anchor: NearTermAnchor,
                  spot: Optional[float], w_smooth: float,
                  w_level: float) -> np.ndarray:
    """Equality-constrained least squares via one sparse KKT solve."""
    n = len(index)
    if w_smooth <= 0 or w_level < 0:
        raise ForwardCurveError("smoothness_weight must be > 0 and level_weight >= 0")
    D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csr")
    scale = float(np.mean(np.abs(baseline))) or 1.0
    H = (w_smooth / scale**2) * (D2.T @ D2) + (w_level / scale**2) * sp.identity(n, format="csr")
    g = -(w_level / scale**2) * baseline                     # from ||F - baseline||^2

    A, b, _ = _averaging_rows(index, quotes)
    if anchor.pins_spot:
        if spot is None:                                     # pragma: no cover
            raise ForwardCurveError("spot anchor requested without a spot price")
        e0 = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n))
        A = sp.vstack([A, e0], format="csr")
        b = np.concatenate([b, [float(spot)]])

    m = A.shape[0]
    KKT = sp.bmat([[H, A.T], [A, None]], format="csc")
    rhs = np.concatenate([-g, b])
    sol = spla.spsolve(KKT, rhs)
    if not np.all(np.isfinite(sol)):
        raise ForwardCurveError(
            "smooth forward-curve KKT system produced non-finite values; the "
            "constraint set is likely rank-deficient")
    return np.asarray(sol[:n], dtype=float)


def _apply_shape(index: pd.DatetimeIndex, F: np.ndarray,
                 shape: pd.Series, quotes: MarketQuoteSet) -> np.ndarray:
    """Multiply by an hourly shape, renormalised per month to keep the mean."""
    s = shape.reindex(index)
    if s.isna().any():
        raise ForwardCurveError("shape profile does not cover the curve horizon")
    sv = s.to_numpy(dtype=float)
    if np.any(sv <= 0):
        raise ForwardCurveError("shape profile must be strictly positive")
    out = F.copy()
    for dm in delivery_months_between(index[0], index[-1] + pd.Timedelta(hours=1)):
        m = dm.mask(index)
        if not m.any():
            continue
        target = float(F[m].mean())
        prod = F[m] * sv[m]
        out[m] = prod * (target / float(prod.mean()))
    return out


def _build_frame(index: pd.DatetimeIndex, F: np.ndarray, quotes: MarketQuoteSet,
                 anchored: np.ndarray, anchor: NearTermAnchor,
                 mode: CurveMode) -> pd.DataFrame:
    loc = to_turkey(index)
    month_lab = [month_label(t.year, t.month) for t in loc]
    contract = []
    source = []
    for lab, is_anchor in zip(month_lab, anchored):
        q = None
        for cand in quotes.quotes:
            if cand.label == lab:
                q = cand
                break
        if is_anchor or q is None:
            contract.append("")
            source.append(anchor.describe() if is_anchor else "beyond_quoted_strip")
        else:
            contract.append(q.contract_name)
            source.append(f"EPIAS_VEP:{q.contract_name}")
    interp = np.full(len(index), mode == "smooth_constrained")
    interp |= anchored
    return pd.DataFrame({
        "time_utc": [t.isoformat() for t in index],
        "time_turkey": [t.isoformat() for t in loc],
        "delivery_month": month_lab,
        "contract_name": contract,
        "hourly_forward_TRY_MWh": F,
        "source": source,
        "interpolation_flag": interp,
        "extrapolation_flag": anchored,
        "near_term_anchor_flag": anchored,
    })[list(CURVE_COLUMNS)]


# ---------------------------------------------------------------------------
def load_forward_curve_csv(path: str | Path,
                           quotes: Optional[MarketQuoteSet] = None) -> pd.Series:
    """Read a previously written hourly_forward_curve.csv back into a Series."""
    p = Path(path)
    if p.is_dir():
        p = p / "hourly_forward_curve.csv"
    if not p.exists():
        raise FileNotFoundError(f"forward curve not found: {p}")
    df = pd.read_csv(p)
    missing = [c for c in CURVE_COLUMNS if c not in df.columns]
    if missing:
        raise ForwardCurveError(f"{p.name}: missing curve columns {missing}")
    idx = pd.DatetimeIndex(pd.to_datetime(df["time_utc"], utc=True))
    vals = df["hourly_forward_TRY_MWh"].to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ForwardCurveError(f"{p.name}: non-finite forward values")
    return pd.Series(vals, index=idx, name="hourly_forward_TRY_MWh")
