"""
Backtesting and historical evaluation module for NLP Predictor.

Provides data acquisition for historical news and prices, simulation engines,
and performance attribution metrics.
"""

from .data import (
    DEFAULT_MACRO_TICKERS,
    AlpacaHistoricalFetcher,
    AlpacaNewsFetcher,
    XfinlinkClient,
    YahooPriceFetcher,
    clean_historical_news,
    extract_explicit_tickers,
    fetch_alpaca_historical_news,
    filter_dataset,
    filter_historical_news,
    is_macro_event,
    is_spam_title,
    load_universe_tickers,
    process_historical_data,
)
from .data_loader import (
    CUSTOM_PYBROKER_COLUMNS,
    PYBROKER_PANEL_COLUMNS,
    BacktestDataLoader,
)
from .engine import BacktestRunOutput, PyBrokerBacktester
from .replayer import HistoricalReplayer, build_universe_instruments_csv

__all__ = [
    "AlpacaNewsFetcher",
    "AlpacaHistoricalFetcher",
    "fetch_alpaca_historical_news",
    "clean_historical_news",
    "process_historical_data",
    "filter_historical_news",
    "filter_dataset",
    "is_spam_title",
    "is_macro_event",
    "extract_explicit_tickers",
    "load_universe_tickers",
    "YahooPriceFetcher",
    "DEFAULT_MACRO_TICKERS",
    "XfinlinkClient",
    "CUSTOM_PYBROKER_COLUMNS",
    "PYBROKER_PANEL_COLUMNS",
    "BacktestDataLoader",
    "BacktestRunOutput",
    "PyBrokerBacktester",
    "HistoricalReplayer",
    "build_universe_instruments_csv",
]
