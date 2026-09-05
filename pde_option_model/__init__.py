"""Regime-switching TVTP PDE valuation for European electricity options.

Turkish day-ahead market (EPIAS PTF), built on the uploaded Markov
regime-switching outputs (primary model: M2_tvtp_TVTP-1).

Regime convention used throughout this package (M2 / TRY-run convention):
    regime 0 = "normal"  (low volatility)
    regime 1 = "stress"  (high volatility)
"""

__version__ = "0.1.0"
