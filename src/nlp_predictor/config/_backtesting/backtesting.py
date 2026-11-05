"""Backtesting configuration schemas for date ranges and historical market data."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_repo_path(v: Any) -> Path:
    """Resolve relative paths against ``REPO_ROOT`` into absolute ``Path`` objects."""
    p = Path(v)
    if not p.is_absolute():
        return (REPO_ROOT / p).resolve()
    return p.resolve()


class TimeframeConfig(BaseModel):
    """Synchronized ISO date range (YYYY-MM-DD) for historical news and market
    prices.
    """

    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"


class BacktestingDataConfig(BaseModel):
    """Storage location and macroeconomic ETF universe for historical price
    downloads.
    """

    market_prices_dir: Path = Path("data/historical/market_prices")
    price_fetch_delay: float = 0.2
    macro_tickers: list[str] = []

    @field_validator("market_prices_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        """Resolve ``market_prices_dir`` relative to the repository root."""
        return _resolve_repo_path(v)


class AlpacaConfig(BaseModel):
    """Alpaca Historical News API connection and pagination parameters."""

    api_url: str = "https://data.alpaca.markets/v1beta1/news"
    max_articles: int = 1_000_000
    batch_flush_size: int = 5_000
    page_limit: int = 50
    max_retries: int = 5
    request_timeout: int = 30
    request_delay: float = 0.1


class XfinlinkConfig(BaseModel):
    """Xfinlink point-in-time dataset download parameters."""

    indices: list[str] = ["sp500", "ndx100", "djia"]
    start_date: str = "2015-01-01"
    snapshots_frequency: str = "yearly"
    fundamentals_periods: list[str] = ["annual", "quarterly"]
    fundamentals_version: str = "original"
    metrics_periods: list[str] = ["annual", "quarterly", "ttm"]
    request_delay: float = 0.1
    events_page_limit: int = 1000


class NewsFilteringConfig(BaseModel):
    """Parameters for pre-LLM historical news filtering and deduplication."""

    min_article_length: int = 200
    macro_lead_chars: int = 600
    sp500_constituents_file: str = "constituents_sp500_latest.parquet"


class BacktestingConfig(BaseModel):
    """Top-level backtesting configuration combining timeframe and data settings."""

    timeframe: TimeframeConfig = TimeframeConfig()
    data: BacktestingDataConfig = BacktestingDataConfig()
    alpaca: AlpacaConfig = AlpacaConfig()
    xfinlink: XfinlinkConfig = XfinlinkConfig()
    filtering: NewsFilteringConfig = NewsFilteringConfig()
