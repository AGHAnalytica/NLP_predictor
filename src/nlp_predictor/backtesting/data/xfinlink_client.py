"""High-level client for Xfinlink financial data with automatic Parquet persistence.

Uses the official ``xfinlink`` SDK under the hood for clean pandas DataFrame returns,
built-in pagination, and type conversion, while enforcing local Parquet caching
for reproducible, offline backtesting without look-ahead or survivorship bias.
"""

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import xfinlink as xfl

logger = logging.getLogger(__name__)


class XfinlinkClient:
    """Convenience wrapper around official ``xfinlink`` SDK with Parquet persistence.

    Capabilities:
        - ``index(index_name, as_of)``: Survivorship-free index roster.
        - ``index_events(index_name, start, end)``: Additions & removals log.
        - ``prices(ticker, start, end)``: Daily OHLCV bars (supports delisted stocks).
        - ``fundamentals(ticker, start, end)``: As-reported SEC XBRL statements.
        - ``metrics(ticker, period_type)``: Pre-computed valuation multiples.
        - ``insiders(ticker)``: SEC Form 4 insider transactions.
        - ``holdings(ticker)``: Institutional 13F positions.
        - ``search(query)``: Search securities and company profiles.
        - ``resolve(ticker)``: Entity resolution and ticker recycling history.
    """

    def __init__(
        self,
        api_key: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Initialize Xfinlink client and set active API key.

        Args:
            api_key: Xfinlink API key. Defaults to ``XFINLINK_API_KEY`` env var.
            output_dir: Cache directory for Parquet files.
        """
        self.api_key = api_key or os.getenv("XFINLINK_API_KEY", "").strip()
        if self.api_key:
            xfl.set_api_key(self.api_key)

        if output_dir is None:
            output_dir = Path("data/historical/xfinlink")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_status() -> dict[str, Any]:
        """Check API service health and dataset coverage timestamps."""
        return xfl.status()

    @staticmethod
    def get_usage() -> dict[str, Any]:
        """Check plan tier and remaining API quota usage."""
        return xfl.usage()

    def fetch_index_constituents(
        self,
        index_name: str = "sp500",
        as_of: str | None = None,
        save: bool = True,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """Fetch index constituents as of a specific date (survivorship-bias free).

        Args:
            index_name: Index symbol (``sp500``, ``ndx100``, or ``djia``).
            as_of: Point-in-time date (``YYYY-MM-DD``). Defaults to current roster.
            save: Whether to cache result as Parquet.
            skip_existing: Whether to skip download if local file exists.

        Returns:
            DataFrame of index constituents.
        """
        date_tag = as_of or "latest"
        dest = self.output_dir / f"constituents_{index_name}_{date_tag}.parquet"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info(f"Skipping constituents for {index_name} ({dest.name} exists)")
            return pd.read_parquet(dest)

        tag = as_of or "today"
        logger.info(f"Fetching {index_name.upper()} constituents (as_of={tag})...")
        df: pd.DataFrame = xfl.index(index_name=index_name, as_of=as_of)

        if df is None or df.empty:
            logger.warning(f"No constituents returned for index {index_name}")
            return pd.DataFrame()

        if save:
            df.to_parquet(dest, index=False)
            logger.info(f"Saved {len(df)} constituents to {dest}")

        return df

    def fetch_index_events(
        self,
        index_name: str = "sp500",
        start_date: str | None = None,
        end_date: str | None = None,
        save: bool = True,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """Fetch all historical index additions and removals.

        Args:
            index_name: Index name (``sp500``, ``ndx100``, ``djia``).
            start_date: Earliest effective date (``YYYY-MM-DD``).
            end_date: Latest effective date (``YYYY-MM-DD``).
            save: Whether to cache result as Parquet.
            skip_existing: Whether to skip download if local file exists.

        Returns:
            DataFrame of additions and removals log.
        """
        dest = self.output_dir / f"index_events_{index_name}.parquet"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info(f"Skipping events for {index_name} ({dest.name} exists)")
            return pd.read_parquet(dest)

        logger.info(
            f"Fetching {index_name.upper()} index events "
            f"(start={start_date}, end={end_date})..."
        )
        all_events: list[pd.DataFrame] = []
        offset = 0
        limit = 1000

        while True:
            batch = xfl.index_events(
                index_name=index_name,
                start=start_date,
                end=end_date,
                limit=limit,
                offset=offset,
            )
            if batch is None or batch.empty:
                break
            all_events.append(batch)
            if len(batch) < limit:
                break
            offset += len(batch)

        if not all_events:
            logger.warning(f"No events returned for index {index_name}")
            return pd.DataFrame()

        df = pd.concat(all_events, ignore_index=True)
        if save:
            df.to_parquet(dest, index=False)
            logger.info(f"Saved {len(df)} index events to {dest}")

        return df

    def fetch_index_history_snapshots(
        self,
        index_name: str = "sp500",
        dates: Sequence[str] | None = None,
        delay_seconds: float = 0.2,
        skip_existing: bool = True,
    ) -> pd.DataFrame:
        """Fetch Point-in-Time constituent rosters across multiple historical dates.

        Args:
            index_name: Target index name (e.g., ``sp500``).
            dates: Sequence of ``YYYY-MM-DD`` snapshot dates.
            delay_seconds: Delay between requests.
            skip_existing: Skip downloading if target Parquet already exists.

        Returns:
            Consolidated DataFrame of all snapshot rosters.
        """
        dest = self.output_dir / f"index_history_{index_name}.parquet"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info(
                f"Skipping history snapshots for {index_name} ({dest.name} exists)"
            )
            return pd.read_parquet(dest)

        if not dates:
            return pd.DataFrame()

        dfs: list[pd.DataFrame] = []
        for idx, date_str in enumerate(dates):
            try:
                df = self.fetch_index_constituents(
                    index_name=index_name, as_of=date_str, save=False
                )
                if not df.empty:
                    df["as_of_date"] = date_str
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Error fetching {index_name} for date {date_str}: {e}")
            if delay_seconds > 0 and idx < len(dates) - 1:
                time.sleep(delay_seconds)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        dest = self.output_dir / f"index_history_{index_name}.parquet"
        combined.to_parquet(dest, index=False)
        logger.info(f"Saved {len(combined)} rows across {len(dfs)} dates to {dest}")
        return combined

    def fetch_prices(
        self,
        ticker: str | list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        save: bool = True,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars (supports delisted stocks).

        Args:
            ticker: Single ticker symbol or list of symbols.
            start_date: Start date (``YYYY-MM-DD``).
            end_date: End date (``YYYY-MM-DD``).
            period: Shorthand period (e.g. ``max``, ``10y``).
            save: Whether to save to Parquet.
            skip_existing: Whether to skip download if local file exists.

        Returns:
            DataFrame of daily OHLCV price bars.
        """
        sym_tag = ticker if isinstance(ticker, str) else "multi"
        dest = self.output_dir / f"prices_{sym_tag}_daily.parquet"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info(f"Skipping prices for {sym_tag} ({dest.name} exists)")
            return pd.read_parquet(dest)

        if period is not None:
            df: pd.DataFrame = xfl.prices(ticker=ticker, period=period)
        else:
            df = xfl.prices(ticker=ticker, start=start_date, end=end_date)

        if df is None or df.empty:
            logger.warning(f"No prices returned for {ticker}")
            return pd.DataFrame()

        if save:
            df.to_parquet(dest, index=False)
            logger.info(f"Saved {len(df)} price rows for {ticker} to {dest}")

        return df

    def fetch_fundamentals(
        self,
        ticker: str | list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        period_type: str = "annual",
        version: str = "original",
        as_of: str | None = None,
        save: bool = True,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """Fetch standardized SEC XBRL statement data.

        Args:
            ticker: Single ticker symbol or list of symbols.
            start_date: Earliest period end date (``YYYY-MM-DD``).
            end_date: Latest period end date (``YYYY-MM-DD``).
            period: Shorthand period (e.g. ``max``, ``10y``).
            period_type: ``annual``, ``quarterly``, or ``all``.
            version: ``original`` for point-in-time as-reported filings,
                ``restated``, or ``all``.
            as_of: Point-in-time filter date.
            save: Whether to cache result as Parquet.
            skip_existing: Whether to skip download if local file exists.

        Returns:
            DataFrame of financial statement lines.
        """
        sym_tag = ticker if isinstance(ticker, str) else "multi"
        dest = (
            self.output_dir / f"fundamentals_{sym_tag}_{period_type}_{version}.parquet"
        )
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info(f"Skipping fundamentals for {sym_tag} ({dest.name} exists)")
            return pd.read_parquet(dest)

        if period is not None:
            df: pd.DataFrame = xfl.fundamentals(
                ticker=ticker,
                period=period,
                period_type=period_type,
                version=version,
                as_of=as_of,
            )
        else:
            df = xfl.fundamentals(
                ticker=ticker,
                start=start_date,
                end=end_date,
                period_type=period_type,
                version=version,
                as_of=as_of,
            )

        if df is None or df.empty:
            logger.warning(f"No fundamentals returned for {ticker}")
            return pd.DataFrame()

        if save:
            df.to_parquet(dest, index=False)
            logger.info(f"Saved {len(df)} fundamental rows for {ticker} to {dest}")

        return df

    def fetch_metrics(
        self,
        ticker: str | list[str],
        period_type: str = "annual",
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        as_of: str | None = None,
        save: bool = True,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """Fetch pre-computed valuation multiples, margins, and ratios.

        Args:
            ticker: Single ticker symbol or list of symbols.
            period_type: ``annual``, ``quarterly``, or ``ttm``.
            start_date: Start date (``YYYY-MM-DD``).
            end_date: End date (``YYYY-MM-DD``).
            period: Shorthand period (e.g. ``max``, ``10y``).
            as_of: Point-in-time filter date.
            save: Whether to cache result as Parquet.
            skip_existing: Whether to skip download if local file exists.

        Returns:
            DataFrame of valuation metrics.
        """
        sym_tag = ticker if isinstance(ticker, str) else "multi"
        dest = self.output_dir / f"metrics_{sym_tag}_{period_type}.parquet"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info(f"Skipping metrics for {sym_tag} ({dest.name} exists)")
            return pd.read_parquet(dest)

        if period is not None:
            df: pd.DataFrame = xfl.metrics(
                ticker=ticker,
                period_type=period_type,
                period=period,
                as_of=as_of,
            )
        else:
            df = xfl.metrics(
                ticker=ticker,
                period_type=period_type,
                start=start_date,
                end=end_date,
                as_of=as_of,
            )

        if df is None or df.empty:
            logger.warning(f"No metrics returned for {ticker}")
            return pd.DataFrame()

        if save:
            df.to_parquet(dest, index=False)
            logger.info(f"Saved {len(df)} metrics rows for {ticker} to {dest}")

        return df

    def fetch_insiders(
        self,
        ticker: str | list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        save: bool = True,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """Fetch SEC Form 4 insider transactions.

        Args:
            ticker: Single ticker symbol or list of symbols.
            start_date: Start date (``YYYY-MM-DD``).
            end_date: End date (``YYYY-MM-DD``).
            period: Shorthand period (e.g. ``max``, ``5y``).
            save: Whether to cache result as Parquet.
            skip_existing: Whether to skip download if local file exists.

        Returns:
            DataFrame of insider transactions.
        """
        sym_tag = ticker if isinstance(ticker, str) else "multi"
        dest = self.output_dir / f"insiders_{sym_tag}.parquet"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info(f"Skipping insiders for {sym_tag} ({dest.name} exists)")
            return pd.read_parquet(dest)

        if period is not None:
            df: pd.DataFrame = xfl.insiders(ticker=ticker, period=period)
        else:
            df = xfl.insiders(ticker=ticker, start=start_date, end=end_date)

        if df is None or df.empty:
            return pd.DataFrame()

        if save:
            df.to_parquet(dest, index=False)
            logger.info(f"Saved {len(df)} insider rows for {ticker} to {dest}")

        return df

    def fetch_holdings(
        self,
        ticker: str | list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        save: bool = True,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """Fetch institutional 13F holdings.

        Args:
            ticker: Single ticker symbol or list of symbols.
            start_date: Start date (``YYYY-MM-DD``).
            end_date: End date (``YYYY-MM-DD``).
            period: Shorthand period (e.g. ``max``, ``5y``).
            save: Whether to cache result as Parquet.
            skip_existing: Whether to skip download if local file exists.

        Returns:
            DataFrame of institutional holdings.
        """
        sym_tag = ticker if isinstance(ticker, str) else "multi"
        dest = self.output_dir / f"holdings_{sym_tag}.parquet"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info(f"Skipping holdings for {sym_tag} ({dest.name} exists)")
            return pd.read_parquet(dest)

        if period is not None:
            df: pd.DataFrame = xfl.holdings(ticker=ticker, period=period)
        else:
            df = xfl.holdings(ticker=ticker, start=start_date, end=end_date)

        if df is None or df.empty:
            return pd.DataFrame()

        if save:
            df.to_parquet(dest, index=False)
            logger.info(f"Saved {len(df)} holdings rows for {ticker} to {dest}")

        return df

    @staticmethod
    def resolve_entity(ticker: str) -> dict[str, Any]:
        """Resolve ticker symbol to persistent entity_id, CIK, and name history."""
        return xfl.resolve(ticker=ticker)

    def dump_universe_data(
        self,
        tickers: Sequence[str],
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        fetch_fundamentals_data: bool = True,
        fundamentals_periods: Sequence[str] | str = ("annual",),
        fundamentals_version: str = "original",
        fetch_metrics_data: bool = True,
        metrics_periods: Sequence[str] | str = ("annual",),
        fetch_prices_data: bool = True,
        fetch_insiders_data: bool = False,
        fetch_holdings_data: bool = False,
        skip_existing: bool = True,
        delay_seconds: float = 0.1,
    ) -> dict[str, int]:
        """Dump comprehensive data (fundamentals, metrics, prices) for tickers.

        Args:
            tickers: Sequence of ticker symbols.
            start_date: Earliest date (``YYYY-MM-DD``).
            end_date: Latest date (``YYYY-MM-DD``).
            period: Shorthand period string (e.g. ``max``).
            fetch_fundamentals_data: Whether to fetch SEC fundamentals.
            fundamentals_periods: Tuple/list of periods (e.g. ``annual``,
                ``quarterly``).
            fundamentals_version: Accounting version (``original`` or
                ``restated``).
            fetch_metrics_data: Whether to fetch valuation metrics.
            metrics_periods: Tuple/list of periods (e.g. ``annual``,
                ``quarterly``, ``ttm``).
            fetch_prices_data: Whether to fetch daily price bars.
            fetch_insiders_data: Whether to fetch Form 4 insider trades.
            fetch_holdings_data: Whether to fetch 13F institutional holdings.
            skip_existing: Skip requests if destination parquet already exists.
            delay_seconds: Politeness interval between requests.

        Returns:
            Dictionary with download counts for each data category.
        """
        stats = {
            "fundamentals": 0,
            "metrics": 0,
            "prices": 0,
            "insiders": 0,
            "holdings": 0,
        }
        total = len(tickers)

        if isinstance(fundamentals_periods, str):
            f_periods = [fundamentals_periods]
        else:
            f_periods = list(fundamentals_periods)

        if isinstance(metrics_periods, str):
            m_periods = [metrics_periods]
        else:
            m_periods = list(metrics_periods)

        for i, ticker in enumerate(tickers, start=1):
            sym = ticker.strip().upper()

            if fetch_fundamentals_data:
                for fp in f_periods:
                    dest = (
                        self.output_dir
                        / f"fundamentals_{sym}_{fp}_{fundamentals_version}.parquet"
                    )
                    if skip_existing and dest.exists() and dest.stat().st_size > 0:
                        stats["fundamentals"] += 1
                        continue
                    logger.info(
                        f"[{i}/{total}] Fetching fundamentals ({fp}) for {sym}..."
                    )
                    try:
                        df = self.fetch_fundamentals(
                            sym,
                            start_date=start_date,
                            end_date=end_date,
                            period=period,
                            period_type=fp,
                            version=fundamentals_version,
                            save=True,
                            skip_existing=skip_existing,
                        )
                        if not df.empty:
                            stats["fundamentals"] += 1
                    except Exception as e:
                        logger.error(
                            f"Error fetching fundamentals ({fp}) for {sym}: {e}"
                        )
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

            if fetch_metrics_data:
                for mp in m_periods:
                    dest = self.output_dir / f"metrics_{sym}_{mp}.parquet"
                    if skip_existing and dest.exists() and dest.stat().st_size > 0:
                        stats["metrics"] += 1
                        continue
                    logger.info(f"[{i}/{total}] Fetching metrics ({mp}) for {sym}...")
                    try:
                        df = self.fetch_metrics(
                            sym,
                            period_type=mp,
                            start_date=start_date,
                            end_date=end_date,
                            period=period,
                            save=True,
                            skip_existing=skip_existing,
                        )
                        if not df.empty:
                            stats["metrics"] += 1
                    except Exception as e:
                        logger.error(f"Error fetching metrics ({mp}) for {sym}: {e}")
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

            if fetch_prices_data:
                dest = self.output_dir / f"prices_{sym}_daily.parquet"
                if skip_existing and dest.exists() and dest.stat().st_size > 0:
                    stats["prices"] += 1
                else:
                    logger.info(f"[{i}/{total}] Fetching prices for {sym}...")
                    try:
                        df = self.fetch_prices(
                            sym,
                            start_date=start_date,
                            end_date=end_date,
                            period=period,
                            save=True,
                            skip_existing=skip_existing,
                        )
                        if not df.empty:
                            stats["prices"] += 1
                    except Exception as e:
                        logger.error(f"Error fetching prices for {sym}: {e}")
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

            if fetch_insiders_data:
                dest = self.output_dir / f"insiders_{sym}.parquet"
                if skip_existing and dest.exists() and dest.stat().st_size > 0:
                    stats["insiders"] += 1
                else:
                    logger.info(f"[{i}/{total}] Fetching insiders for {sym}...")
                    try:
                        df = self.fetch_insiders(
                            sym,
                            start_date=start_date,
                            end_date=end_date,
                            period=period,
                            save=True,
                            skip_existing=skip_existing,
                        )
                        if not df.empty:
                            stats["insiders"] += 1
                    except Exception as e:
                        logger.error(f"Error fetching insiders for {sym}: {e}")
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

            if fetch_holdings_data:
                dest = self.output_dir / f"holdings_{sym}.parquet"
                if skip_existing and dest.exists() and dest.stat().st_size > 0:
                    stats["holdings"] += 1
                else:
                    logger.info(f"[{i}/{total}] Fetching holdings for {sym}...")
                    try:
                        df = self.fetch_holdings(
                            sym,
                            start_date=start_date,
                            end_date=end_date,
                            period=period,
                            save=True,
                            skip_existing=skip_existing,
                        )
                        if not df.empty:
                            stats["holdings"] += 1
                    except Exception as e:
                        logger.error(f"Error fetching holdings for {sym}: {e}")
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

        logger.info(f"Xfinlink universe dump finished: {stats}")
        return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    client = XfinlinkClient()
    logger.info("Checking Xfinlink status via SDK...")
    try:
        logger.info(f"Status: {client.get_status()}")
        if client.api_key:
            logger.info(f"Usage: {client.get_usage()}")
    except Exception as err:
        logger.warning(f"Status check failed: {err}")
