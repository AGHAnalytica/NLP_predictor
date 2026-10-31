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
    macro_tickers: list[str] = []

    @field_validator("market_prices_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        """Resolve ``market_prices_dir`` relative to the repository root."""
        return _resolve_repo_path(v)


class BacktestingConfig(BaseModel):
    """Top-level backtesting configuration combining timeframe and data settings."""

    timeframe: TimeframeConfig = TimeframeConfig()
    data: BacktestingDataConfig = BacktestingDataConfig()
