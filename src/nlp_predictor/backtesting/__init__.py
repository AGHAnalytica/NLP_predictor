"""
Backtesting and historical evaluation module for NLP Predictor.

Provides data acquisition for historical news and prices, simulation engines,
and performance attribution metrics.
"""

from .data import (
    DEFAULT_MACRO_TICKERS,
    AlpacaHistoricalFetcher,
    AlpacaNewsFetcher,
    YahooPriceFetcher,
    clean_historical_news,
    fetch_alpaca_historical_news,
    process_historical_data,
)

__all__ = [
    "AlpacaNewsFetcher",
    "AlpacaHistoricalFetcher",
    "fetch_alpaca_historical_news",
    "clean_historical_news",
    "process_historical_data",
    "YahooPriceFetcher",
    "DEFAULT_MACRO_TICKERS",
]

