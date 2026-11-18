"""Historical price, point-in-time index membership, and signal panel assembly.

Merges local Parquet OHLCV histories with point-in-time index membership intervals
and daily NLP signals into the tabular schema expected by :mod:`pybroker`.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.signal.signal_log import SignalLog

logger = logging.getLogger(__name__)

CUSTOM_PYBROKER_COLUMNS: tuple[str, ...] = (
    "signal_dir",
    "confidence",
    "instrument_severity",
    "n_topics",
    "in_index",
)

PYBROKER_PANEL_COLUMNS: list[str] = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    *CUSTOM_PYBROKER_COLUMNS,
]


def _encode_signal_direction(signal_series: pd.Series) -> pd.Series:
    """Map string signal labels ('UP', 'DOWN', 'NEUTRAL') to integers (1, -1, 0)."""
    normalized = signal_series.astype(str).str.strip().str.upper()
    direction = np.where(
        normalized == "UP",
        1,
        np.where(normalized == "DOWN", -1, 0),
    )
    return pd.Series(direction, index=signal_series.index, dtype=int)


class BacktestDataLoader:
    """Builds survivorship-free daily OHLCV + NLP signal panels for PyBroker.

    Reads daily price bars from local Xfinlink and Yahoo Finance Parquet caches,
    evaluates point-in-time constituent intervals (`added_date` to `removed_date`),
    and left-joins aggregated daily signals per instrument and severity variant.
    """

    def __init__(
        self,
        *,
        xfinlink_dir: Path | None = None,
        market_prices_dir: Path | None = None,
        signal_log: SignalLog | None = None,
        universe_index: str | None = None,
        use_adjusted_close: bool | None = None,
    ) -> None:
        """Initialize the backtest data loader with storage paths and dependencies.

        Args:
            xfinlink_dir: Directory containing Xfinlink Parquet price and index files.
            market_prices_dir: Directory containing fallback Yahoo Finance price files.
            signal_log: Optional :class:`SignalLog` instance for loading signals.
            universe_index: Target index name (e.g., ``'sp500'``, ``'ndx100'``).
            use_adjusted_close: Whether to scale OHLC proportionally by ``adj_close``.
        """
        strat_cfg = config.backtesting.strategy
        self.xfinlink_dir = Path(xfinlink_dir or config.paths.historical.xfinlink)
        self.market_prices_dir = Path(
            market_prices_dir or config.backtesting.data.market_prices_dir
        )
        self.signal_log = signal_log or SignalLog()
        self.universe_index = (universe_index or strat_cfg.universe_index).lower()
        self.use_adjusted_close = (
            bool(use_adjusted_close)
            if use_adjusted_close is not None
            else bool(strat_cfg.use_adjusted_close)
        )

    @classmethod
    def build_default(cls) -> "BacktestDataLoader":
        """Construct a production :class:`BacktestDataLoader` from configuration."""
        return cls()

    def load_membership_intervals(self) -> pd.DataFrame:
        """Load deduplicated point-in-time index membership intervals.

        Combines ``index_history_<index>.parquet`` and
        ``constituents_<index>_latest.parquet`` into ``(ticker, added_date,
        removed_date)`` intervals.

        Returns:
            DataFrame with ``ticker``, ``added_date``, and ``removed_date`` columns.
        """
        frames: list[pd.DataFrame] = []
        history_path = (
            self.xfinlink_dir / f"index_history_{self.universe_index}.parquet"
        )
        latest_path = (
            self.xfinlink_dir / f"constituents_{self.universe_index}_latest.parquet"
        )

        for path in (history_path, latest_path):
            if path.exists() and path.stat().st_size > 0:
                df = pd.read_parquet(path)
                required = {"ticker", "added_date", "removed_date"}
                if required.issubset(df.columns):
                    frames.append(df[["ticker", "added_date", "removed_date"]])

        if not frames:
            return pd.DataFrame(columns=["ticker", "added_date", "removed_date"])

        combined = pd.concat(frames, ignore_index=True)
        combined["ticker"] = combined["ticker"].astype(str).str.strip().str.upper()
        combined["added_date"] = pd.to_datetime(
            combined["added_date"], errors="coerce"
        ).dt.tz_localize(None)
        combined["removed_date"] = pd.to_datetime(
            combined["removed_date"], errors="coerce"
        ).dt.tz_localize(None)

        combined["added_date"] = combined["added_date"].fillna(
            pd.Timestamp("1900-01-01")
        )
        combined["removed_date"] = combined["removed_date"].fillna(
            pd.Timestamp("2099-12-31")
        )
        combined = combined.drop_duplicates(
            subset=["ticker", "added_date", "removed_date"]
        ).reset_index(drop=True)
        return combined

    def load_ticker_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load and normalize daily OHLCV bars for a single ticker from local Parquet.

        Checks Xfinlink ``prices_<TICKER>_daily.parquet`` first and falls back to
        ``market_prices/<TICKER>_daily.parquet``.

        Args:
            ticker: Uppercase ticker symbol.
            start_date: Optional inclusive start date (``YYYY-MM-DD``).
            end_date: Optional inclusive end date (``YYYY-MM-DD``).

        Returns:
            DataFrame with ``symbol``, ``date``, ``open``, ``high``, ``low``,
            ``close``, and ``volume`` columns, or an empty DataFrame if unavailable.
        """
        sym = ticker.strip().upper()
        xfl_path = self.xfinlink_dir / f"prices_{sym}_daily.parquet"
        yf_path = self.market_prices_dir / f"{sym}_daily.parquet"

        if xfl_path.exists() and xfl_path.stat().st_size > 0:
            raw = pd.read_parquet(xfl_path)
        elif yf_path.exists() and yf_path.stat().st_size > 0:
            raw = pd.read_parquet(yf_path)
        else:
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        rename_map = {col: col.lower().replace(" ", "_") for col in raw.columns}
        df = raw.rename(columns=rename_map).copy()

        required_ohlc = {"date", "open", "high", "low", "close"}
        if not required_ohlc.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = (
            pd.to_datetime(df["date"], utc=True, errors="coerce")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        df = df.dropna(subset=["date", "open", "high", "low", "close"])

        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[df["close"] > 0.0].copy()

        if self.use_adjusted_close and "adj_close" in df.columns:
            adj_close = pd.to_numeric(df["adj_close"], errors="coerce")
            valid_adj = adj_close.notna() & (adj_close > 0.0) & (df["close"] > 0.0)
            factor = np.where(valid_adj, adj_close / df["close"], 1.0)
            df["open"] = df["open"] * factor
            df["high"] = df["high"] * factor
            df["low"] = df["low"] * factor
            df["close"] = np.where(valid_adj, adj_close, df["close"])

        if "volume" in df.columns:
            df["volume"] = (
                pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)
            )
        else:
            df["volume"] = 0.0

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        if df.empty:
            return pd.DataFrame()

        df["symbol"] = sym
        df = (
            df[["symbol", "date", "open", "high", "low", "close", "volume"]]
            .sort_values("date")
            .drop_duplicates(subset=["symbol", "date"], keep="last")
            .reset_index(drop=True)
        )
        return df

    def load_price_panel(
        self,
        tickers: Sequence[str],
        start_date: str | None = None,
        end_date: str | None = None,
        enforce_point_in_time: bool | None = None,
    ) -> pd.DataFrame:
        """Load consolidated daily OHLCV panel across multiple tickers with PIT flags.

        Args:
            tickers: Sequence of ticker symbols to load.
            start_date: Earliest date (``YYYY-MM-DD``).
            end_date: Latest date (``YYYY-MM-DD``).
            enforce_point_in_time: Whether to compute ``in_index`` from constituent
                history.

        Returns:
            Consolidated DataFrame of price bars with ``in_index`` (int: 1 or 0).
        """
        enforce_pit = (
            bool(enforce_point_in_time)
            if enforce_point_in_time is not None
            else bool(config.backtesting.strategy.enforce_point_in_time_universe)
        )
        intervals = (
            self.load_membership_intervals()
            if enforce_pit
            else pd.DataFrame(columns=["ticker", "added_date", "removed_date"])
        )

        unique_tickers = sorted({t.strip().upper() for t in tickers if t and t.strip()})
        frames: list[pd.DataFrame] = []
        for sym in unique_tickers:
            ticker_df = self.load_ticker_prices(
                sym, start_date=start_date, end_date=end_date
            )
            if not ticker_df.empty:
                frames.append(ticker_df)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "in_index",
                ]
            )

        panel = pd.concat(frames, ignore_index=True)
        if not enforce_pit or intervals.empty:
            panel["in_index"] = 1
            return panel

        merged = panel.merge(
            intervals,
            left_on="symbol",
            right_on="ticker",
            how="left",
        )
        in_window = (
            merged["added_date"].notna()
            & (merged["date"] >= merged["added_date"])
            & (merged["date"] <= merged["removed_date"])
        )
        merged["in_index_flag"] = in_window.astype(int)
        pit_flags = (
            merged.groupby(["symbol", "date"], as_index=False)["in_index_flag"]
            .max()
            .rename(columns={"in_index_flag": "in_index"})
        )
        panel = panel.merge(pit_flags, on=["symbol", "date"], how="left")
        panel["in_index"] = panel["in_index"].fillna(0).astype(int)
        return panel

    @staticmethod
    def aggregate_daily_signals(
        signals_df: pd.DataFrame,
        variant: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate intra-day tick signals into one daily signal per instrument.

        When multiple ticks occur on the same calendar UTC date for an instrument,
        the latest tick on that date is selected.

        Args:
            signals_df: Raw signal DataFrame from :class:`SignalLog` or
                :class:`SignalGenerator`.
            variant: Optional severity variant name to filter on.

        Returns:
            Daily signal DataFrame with ``symbol``, ``date``, ``signal_dir``,
            ``confidence``, ``instrument_severity``, and ``n_topics``.
        """
        out_cols = [
            "symbol",
            "date",
            "signal_dir",
            "confidence",
            "instrument_severity",
            "n_topics",
        ]
        if signals_df is None or signals_df.empty:
            return pd.DataFrame(columns=out_cols)

        df = signals_df.copy()
        if variant is not None and "variant" in df.columns:
            df = df[df["variant"] == variant].copy()
        if df.empty:
            return pd.DataFrame(columns=out_cols)

        time_col = "tick_time" if "tick_time" in df.columns else "current_time"
        if time_col not in df.columns:
            raise ValueError(
                "signals_df must contain either 'tick_time' or 'current_time'."
            )

        timestamps = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df["timestamp"] = timestamps
        df = df.dropna(subset=["timestamp", "instrument"]).copy()
        if df.empty:
            return pd.DataFrame(columns=out_cols)

        df["symbol"] = df["instrument"].astype(str).str.strip().str.upper()
        df["date"] = df["timestamp"].dt.tz_localize(None).dt.normalize()
        df["signal_dir"] = _encode_signal_direction(df["signal"])
        df["confidence"] = (
            pd.to_numeric(df.get("confidence", 0.0), errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        df["instrument_severity"] = pd.to_numeric(
            df.get("instrument_severity", 0.0), errors="coerce"
        ).fillna(0.0)
        df["n_topics"] = (
            pd.to_numeric(df.get("n_topics", 0), errors="coerce").fillna(0).astype(int)
        )

        daily = (
            df.sort_values("timestamp")
            .drop_duplicates(subset=["symbol", "date"], keep="last")
            .reset_index(drop=True)
        )
        return daily[out_cols]

    def build_pybroker_dataframe(
        self,
        signals_df: pd.DataFrame | None = None,
        *,
        variant: str | None = None,
        tickers: Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        enforce_point_in_time: bool | None = None,
    ) -> pd.DataFrame:
        """Assemble the unified daily OHLCV + signal DataFrame for PyBroker.

        Args:
            signals_df: Optional DataFrame of signals (loads from :attr:`signal_log`
                when ``None``).
            variant: Severity variant name to evaluate.
            tickers: Explicit ticker list (defaults to all instruments present in
                ``signals_df``).
            start_date: Backtest start date (``YYYY-MM-DD``).
            end_date: Backtest end date (``YYYY-MM-DD``).
            enforce_point_in_time: Whether to populate ``in_index`` from constituent
                history.

        Returns:
            DataFrame conforming to :data:`PYBROKER_PANEL_COLUMNS`.
        """
        raw_signals = signals_df if signals_df is not None else self.signal_log.load()
        daily_signals = self.aggregate_daily_signals(raw_signals, variant=variant)

        if tickers is None:
            if not daily_signals.empty:
                target_tickers = sorted(daily_signals["symbol"].unique().tolist())
            else:
                target_tickers = []
        else:
            target_tickers = sorted(
                {t.strip().upper() for t in tickers if t and t.strip()}
            )

        if not target_tickers:
            return pd.DataFrame(columns=PYBROKER_PANEL_COLUMNS)

        start_dt = start_date or config.backtesting.timeframe.start_date
        end_dt = end_date or config.backtesting.timeframe.end_date

        price_panel = self.load_price_panel(
            target_tickers,
            start_date=start_dt,
            end_date=end_dt,
            enforce_point_in_time=enforce_point_in_time,
        )
        if price_panel.empty:
            return pd.DataFrame(columns=PYBROKER_PANEL_COLUMNS)

        merged = price_panel.merge(
            daily_signals,
            on=["symbol", "date"],
            how="left",
        )
        merged["signal_dir"] = merged["signal_dir"].fillna(0).astype(int)
        merged["confidence"] = merged["confidence"].fillna(0.0).astype(float)
        merged["instrument_severity"] = (
            merged["instrument_severity"].fillna(0.0).astype(float)
        )
        merged["n_topics"] = merged["n_topics"].fillna(0).astype(int)
        merged["in_index"] = merged["in_index"].fillna(1).astype(int)

        merged = (
            merged[PYBROKER_PANEL_COLUMNS]
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
        )
        return merged
