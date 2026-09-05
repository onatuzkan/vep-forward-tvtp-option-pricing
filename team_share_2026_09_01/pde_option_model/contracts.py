"""European option contracts on the electricity spot price.

Terminal payoffs are evaluated in the original price space using the exact
inverse transform P(y) = scale_P * sinh(y) identified from the preprocessing
metadata (report.md section 3), then expressed on the y-grid:

    call: V_i(T, y) = max(scale_P * sinh(y) - K, 0)
    put:  V_i(T, y) = max(K - scale_P * sinh(y), 0)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .transformations import PriceTransform

OptionType = Literal["call", "put"]

HOURS_PER_YEAR = 8760.0   # rate convention: r_per_hour = r_annual / 8760 (ACT/365-style)


@dataclass(frozen=True)
class EuropeanOption:
    option_type: OptionType
    strike: float
    valuation_utc: pd.Timestamp
    maturity_utc: pd.Timestamp
    r_annual: float

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity_utc <= self.valuation_utc:
            raise ValueError("maturity must be after valuation")

    @property
    def tau_hours(self) -> float:
        return (self.maturity_utc - self.valuation_utc).total_seconds() / 3600.0

    @property
    def r_per_hour(self) -> float:
        return self.r_annual / HOURS_PER_YEAR

    def payoff_from_price(self, price: np.ndarray) -> np.ndarray:
        price = np.asarray(price, dtype=float)
        if self.option_type == "call":
            return np.maximum(price - self.strike, 0.0)
        return np.maximum(self.strike - price, 0.0)

    def payoff_on_grid(self, y: np.ndarray, transform: PriceTransform) -> np.ndarray:
        return self.payoff_from_price(transform.price_from_y(y))

    def describe(self) -> str:
        return (f"European {self.option_type}, K={self.strike:.2f}, "
                f"valuation {self.valuation_utc.isoformat()}, "
                f"maturity {self.maturity_utc.isoformat()} "
                f"(tau={self.tau_hours:.1f} h), r_annual={self.r_annual:.4f}")
