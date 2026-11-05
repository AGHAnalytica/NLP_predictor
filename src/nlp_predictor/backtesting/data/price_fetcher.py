"""
Module for fetching and storing historical OHLCV market prices from Yahoo Finance.

Provides resilient downloading, ticker validation, and Parquet caching for backtesting.
"""

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from nlp_predictor.config import config

logger = logging.getLogger(__name__)

# Standard macroeconomic ETF proxies loaded from configuration
DEFAULT_MACRO_TICKERS: list[str] = list(config.backtesting.data.macro_tickers)


def _shift_exclusive_end_date(end_date: str | None) -> str | None:
    """
    Shift end_date by +1 day for yfinance because its end parameter is exclusive.

    This ensures that when a user specifies end_date='2025-12-31', market data for
    2025-12-31 is included in the download.
    """
    if not end_date:
        return None
    try:
        from datetime import datetime, timedelta

        dt = datetime.strptime(end_date, "%Y-%m-%d")
        return (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        return end_date


class YahooPriceFetcher:
    """Fetcher for historical daily price data from Yahoo Finance via yfinance."""

    def __init__(self, output_dir: str | Path | None = None):
        if output_dir is None:
            output_dir = config.backtesting.data.market_prices_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ticker(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price history for a single ticker and persist to Parquet.

        Args:
            ticker: Ticker symbol (e.g. 'AAPL', 'SPY').
            start_date: Start date string (YYYY-MM-DD, defaults to config).
            end_date: End date string (YYYY-MM-DD, defaults to config).
            overwrite: If False and the parquet cache already exists, load from disk.

        Returns:
            pd.DataFrame: OHLCV price history or empty DataFrame on failure.
        """
        start_date = start_date or config.backtesting.timeframe.start_date
        end_date = end_date or config.backtesting.timeframe.end_date
        yf_end_date = _shift_exclusive_end_date(end_date)

        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for price fetching. Install it via "
                "'uv pip install yfinance' or add the backtest optional dependency."
            ) from e

        file_path = self.output_dir / f"{ticker}_daily.parquet"
        if not overwrite and file_path.exists():
            logger.debug(f"Loading cached prices for {ticker} from {file_path}")
            return pd.read_parquet(file_path)

        logger.info(f"Downloading prices for {ticker} from Yahoo Finance...")
        try:
            data = yf.download(
                ticker, start=start_date, end=yf_end_date, progress=False
            )
            if data.empty:
                logger.warning(f"No price data returned for ticker '{ticker}'")
                return pd.DataFrame()

            data = data.reset_index()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            expected_cols = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]
            cols = [col for col in expected_cols if col in data.columns]
            data = data[cols]

            data.to_parquet(file_path, engine="pyarrow", index=False)
            logger.info(f"Saved {len(data)} bars for {ticker} to {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()

    def fetch_tickers(
        self,
        tickers: Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        delay_seconds: float | None = None,
        overwrite: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV price history for multiple tickers sequentially with pacing.

        Args:
            tickers: Sequence of ticker symbols (defaults to config macro_tickers).
            start_date: Start date string (defaults to config).
            end_date: End date string (defaults to config).
            delay_seconds: Delay between requests to avoid rate limits.
            overwrite: Whether to force re-download.

        Returns:
            dict[str, pd.DataFrame]: Mapping of ticker -> DataFrame.
        """
        if tickers is None:
            tickers = config.backtesting.data.macro_tickers or DEFAULT_MACRO_TICKERS
        start_date = start_date or config.backtesting.timeframe.start_date
        end_date = end_date or config.backtesting.timeframe.end_date
        delay = (
            float(delay_seconds)
            if delay_seconds is not None
            else float(config.backtesting.data.price_fetch_delay)
        )

        results: dict[str, pd.DataFrame] = {}
        for idx, ticker in enumerate(tickers):
            if not isinstance(ticker, str) or not ticker.strip():
                continue
            cleaned_ticker = ticker.strip().upper()
            df = self.fetch_ticker(
                cleaned_ticker,
                start_date=start_date,
                end_date=end_date,
                overwrite=overwrite,
            )
            if not df.empty:
                results[cleaned_ticker] = df
            if delay > 0 and idx < len(tickers) - 1:
                time.sleep(delay)
        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    fetcher = YahooPriceFetcher()
    fetcher.fetch_tickers()
