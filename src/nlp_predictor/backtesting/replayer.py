"""Historical news feature ingestion and chronological tick replay orchestrator.

Bridges filtered historical news archives (``alpaca_news_filtered.parquet``) and the
PyBroker simulation engine by ingesting article features in checkpointed batches into
an unpruned historical :class:`FeatureStore` and replaying :class:`TickRunner`
chronologically across historical timestamps without future look-ahead bias.
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.config._backtesting import HistoricalReplayConfig
from nlp_predictor.lower_path.lower_path_analyzer import LowerPathAnalyzer
from nlp_predictor.lower_path.macro_overlay import MacroOverlayAnalyzer
from nlp_predictor.preprocessing.feature_ingestor import FeatureIngestor
from nlp_predictor.preprocessing.feature_store import FeatureStore
from nlp_predictor.preprocessing.topic_extractor import TopicExtractor
from nlp_predictor.signal.signal_generator import SignalGenerator
from nlp_predictor.signal.signal_log import SignalLog
from nlp_predictor.tick.tick_runner import TickRunner
from nlp_predictor.upper_path.upper_path_analyzer import UpperPathAnalyzer

logger = logging.getLogger(__name__)


def build_universe_instruments_csv(
    output_csv_path: Path | None = None,
    xfinlink_dir: Path | None = None,
    indices: Sequence[str] | None = None,
) -> Path:
    """Export a consolidated ``(ticker, full_name)`` CSV from local Xfinlink rosters.

    Combines ``constituents_<index>_latest.parquet`` and
    ``index_history_<index>.parquet`` so that :class:`UpperPathAnalyzer` and
    :class:`MacroOverlayAnalyzer` can map historical news across the full survivorship-
    free index universe.

    Args:
        output_csv_path: Target CSV destination path.
        xfinlink_dir: Local Xfinlink Parquet directory.
        indices: Target index names (defaults to
            ``config.backtesting.xfinlink.indices``).

    Returns:
        Path to the written instruments CSV file.
    """
    dest = Path(output_csv_path or config.paths.input.instruments_large)
    xfl_dir = Path(xfinlink_dir or config.paths.historical.xfinlink)
    target_indices = list(indices or config.backtesting.xfinlink.indices)

    frames: list[pd.DataFrame] = []
    for idx_name in target_indices:
        for prefix in ("constituents", "index_history"):
            pattern = (
                f"{prefix}_{idx_name}_latest.parquet"
                if prefix == "constituents"
                else f"{prefix}_{idx_name}.parquet"
            )
            file_path = xfl_dir / pattern
            if file_path.exists() and file_path.stat().st_size > 0:
                df = pd.read_parquet(file_path)
                if {"ticker", "entity_name"}.issubset(df.columns):
                    sub = df[["ticker", "entity_name"]].rename(
                        columns={"entity_name": "full_name"}
                    )
                    frames.append(sub)

    if dest.exists() and dest.stat().st_size > 0:
        existing = pd.read_csv(dest)
        if {"ticker", "full_name"}.issubset(existing.columns):
            frames.append(existing[["ticker", "full_name"]])

    if not frames:
        fallback = pd.read_csv(config.paths.input.instruments)
        return Path(config.paths.input.instruments) if fallback.empty else dest

    combined = pd.concat(frames, ignore_index=True)
    combined["ticker"] = combined["ticker"].astype(str).str.strip().str.upper()
    combined["full_name"] = combined["full_name"].fillna("").astype(str).str.strip()
    combined = (
        combined[combined["ticker"] != ""]
        .drop_duplicates(subset=["ticker"], keep="first")
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest, index=False)
    logger.info(f"Saved {len(combined)} universe instruments to {dest}")
    return dest


class HistoricalReplayer:
    """Orchestrates checkpointed historical feature ingestion and chronological replay.

    Uses a dedicated unpruned :class:`FeatureStore` and :class:`SignalLog` so live
    caches remain untouched and historical articles are never pruned by the 36-hour
    live retention window.
    """

    def __init__(
        self,
        *,
        store: FeatureStore,
        ingestor: FeatureIngestor,
        tick_runner: TickRunner,
        signal_log: SignalLog,
        replay_config: HistoricalReplayConfig | None = None,
    ) -> None:
        """Initialize the historical replayer with injected pipeline collaborators."""
        self.store = store
        self.ingestor = ingestor
        self.tick_runner = tick_runner
        self.signal_log = signal_log
        self.replay_config = replay_config or config.backtesting.replay

    @classmethod
    def build_default(
        cls,
        *,
        instruments_path: str | Path | None = None,
        feature_store_path: str | Path | None = None,
        signal_log_path: str | Path | None = None,
    ) -> "HistoricalReplayer":
        """Construct a production :class:`HistoricalReplayer` wired to historical paths.

        Args:
            instruments_path: Optional instruments CSV path (defaults to
                ``config.paths.input.instruments_large`` if present, else
                ``config.paths.input.instruments``).
            feature_store_path: Path to the unpruned historical feature store Parquet.
            signal_log_path: Path to the historical signals log Parquet.

        Returns:
            Configured :class:`HistoricalReplayer` instance.
        """
        replay_cfg = config.backtesting.replay
        default_inst = (
            config.paths.input.instruments_large
            if config.paths.input.instruments_large.exists()
            else config.paths.input.instruments
        )
        inst_str = str(instruments_path or default_inst)
        store_path = str(feature_store_path or config.paths.historical.feature_store)
        log_path = str(signal_log_path or config.paths.historical.signal_log)

        store = FeatureStore(
            path=store_path,
            max_window_hours=float(replay_cfg.retain_window_hours),
        )
        upper = UpperPathAnalyzer(instruments_path=inst_str)
        lower = LowerPathAnalyzer(instruments_path=inst_str)
        macro = MacroOverlayAnalyzer(instruments_path=inst_str)
        extractor = TopicExtractor.build_default()
        signal_gen = SignalGenerator.build_default()
        signal_log = SignalLog(path=log_path)

        ingestor = FeatureIngestor(
            store=store,
            extractor=extractor,
            upper=upper,
            lower=lower,
        )
        tick_runner = TickRunner(
            store=store,
            upper=upper,
            lower=lower,
            macro=macro,
            signal=signal_gen,
            log=signal_log,
        )
        return cls(
            store=store,
            ingestor=ingestor,
            tick_runner=tick_runner,
            signal_log=signal_log,
            replay_config=replay_cfg,
        )

    def ingest_historical_articles(
        self,
        articles_df: pd.DataFrame | None = None,
        *,
        articles_path: str | Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        batch_size: int | None = None,
    ) -> int:
        """Ingest historical articles in checkpointed batches without 36h pruning.

        Args:
            articles_df: Optional pre-loaded articles DataFrame.
            articles_path: Parquet path to load when ``articles_df`` is ``None``
                (defaults to ``config.paths.historical.news_filtered``).
            start_date: Optional earliest article date filter (``YYYY-MM-DD``).
            end_date: Optional latest article date filter (``YYYY-MM-DD``).
            batch_size: Number of unseen articles per LLM checkpoint batch.

        Returns:
            Total number of newly cached topic feature rows.
        """
        if articles_df is None:
            src_path = Path(articles_path or config.paths.historical.news_filtered)
            if not src_path.exists():
                logger.error(f"Historical articles file not found: {src_path}")
                return 0
            articles_df = pd.read_parquet(src_path)

        if articles_df.empty:
            logger.info("Historical ingest: input articles DataFrame is empty.")
            return 0

        df = articles_df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(
                hours=23, minutes=59, seconds=59
            )
            df = df[df["date"] <= end_ts]

        known_hashes = self.store.known_doc_hashes()
        unseen = df[~df["doc_hash"].astype(str).isin(known_hashes)].reset_index(
            drop=True
        )
        if unseen.empty:
            logger.info(
                f"Historical ingest: all {len(df)} article(s) in range are already "
                "cached."
            )
            return 0

        chunk_size = max(1, int(batch_size or self.replay_config.ingest_batch_size))
        total_unseen = len(unseen)
        total_topics = 0

        for start_idx in range(0, total_unseen, chunk_size):
            end_idx = min(start_idx + chunk_size, total_unseen)
            batch = unseen.iloc[start_idx:end_idx].copy()
            batch_max_dt = batch["date"].max().to_pydatetime()
            logger.info(
                f"Historical ingest batch [{start_idx + 1}..{end_idx}/{total_unseen}] "
                f"(up to {batch_max_dt.strftime('%Y-%m-%d %H:%M UTC')})..."
            )
            cached_count = self.ingestor.ingest(batch, now=batch_max_dt)
            total_topics += cached_count

        logger.info(
            f"Historical ingest finished: cached {total_topics} topic(s) across "
            f"{total_unseen} unseen article(s)."
        )
        return total_topics

    def generate_replay_schedule(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        cached_topics_df: pd.DataFrame | None = None,
    ) -> list[datetime]:
        """Build chronological UTC decision timestamps for historical tick simulation.

        When ``skip_empty_windows`` is enabled, only timestamps that have at least one
        cached topic within the maximum lookback window are included.

        Args:
            start_date: Start date (``YYYY-MM-DD``).
            end_date: End date (``YYYY-MM-DD``).
            cached_topics_df: Optional pre-loaded FeatureStore DataFrame.

        Returns:
            Sorted list of timezone-aware UTC :class:`datetime` tick timestamps.
        """
        topics = cached_topics_df if cached_topics_df is not None else self.store.load()
        if topics.empty:
            return []

        min_topic_dt = topics["date"].min()
        max_topic_dt = topics["date"].max()

        start_str = start_date or min_topic_dt.strftime("%Y-%m-%d")
        end_str = end_date or max_topic_dt.strftime("%Y-%m-%d")

        base_range = pd.date_range(
            start=start_str,
            end=end_str,
            freq=self.replay_config.tick_frequency,
            tz="UTC",
        )
        hour_offset = pd.Timedelta(
            hours=int(self.replay_config.decision_hour_utc),
            minutes=int(self.replay_config.decision_minute_utc),
        )
        if self.replay_config.tick_frequency.upper().endswith(("D", "B")):
            tick_index = base_range.normalize() + hour_offset
        else:
            tick_index = base_range

        max_lookback = max(
            self.tick_runner.upper_window_hours,
            self.tick_runner.climate_window_hours,
        )
        topic_dates = topics["date"].sort_values()

        schedule: list[datetime] = []
        for ts in tick_index:
            if self.replay_config.skip_empty_windows:
                cutoff = ts - pd.Timedelta(hours=max_lookback)
                in_win = (topic_dates >= cutoff) & (topic_dates <= ts)
                if not in_win.any():
                    continue
            schedule.append(ts.to_pydatetime().astimezone(UTC))
        return schedule

    def replay_ticks(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Run :class:`TickRunner` chronologically across the historical schedule.

        Skips tick timestamps that have already been recorded in :attr:`signal_log` so
        interrupted replay jobs can resume cleanly.

        Args:
            start_date: Earliest replay date (``YYYY-MM-DD``).
            end_date: Latest replay date (``YYYY-MM-DD``).

        Returns:
            Consolidated DataFrame of all signals in :attr:`signal_log`.
        """
        cached_topics = self.store.load()
        if cached_topics.empty:
            logger.warning(
                "Historical replay: feature store is empty; run ingest first."
            )
            return pd.DataFrame()

        schedule = self.generate_replay_schedule(
            start_date=start_date,
            end_date=end_date,
            cached_topics_df=cached_topics,
        )
        if not schedule:
            logger.info("Historical replay: no active tick dates found in timeframe.")
            return self.signal_log.load()

        existing_log = self.signal_log.load()
        completed_times: set[str] = (
            set(existing_log["tick_time"].dropna().astype(str))
            if not existing_log.empty and "tick_time" in existing_log.columns
            else set()
        )

        total_ticks = len(schedule)
        for idx, sim_now in enumerate(schedule, start=1):
            now_str = sim_now.strftime("%Y-%m-%d %H:%M:%S")
            if now_str in completed_times:
                logger.debug(
                    f"[{idx}/{total_ticks}] Skipping already-logged tick {now_str}"
                )
                continue
            logger.info(f"[{idx}/{total_ticks}] Replaying historical tick @ {now_str}")
            self.tick_runner.run(now=sim_now)

        return self.signal_log.load()

    def run_full_pipeline(
        self,
        *,
        articles_df: pd.DataFrame | None = None,
        articles_path: str | Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Execute historical feature ingestion followed by chronological tick replay.

        Args:
            articles_df: Optional pre-loaded historical articles DataFrame.
            articles_path: Optional path to filtered historical news Parquet.
            start_date: Earliest date (``YYYY-MM-DD``).
            end_date: Latest date (``YYYY-MM-DD``).

        Returns:
            Dictionary with ``cached_topics`` count and ``signals_df``.
        """
        cached_topics = self.ingest_historical_articles(
            articles_df=articles_df,
            articles_path=articles_path,
            start_date=start_date,
            end_date=end_date,
        )
        signals_df = self.replay_ticks(start_date=start_date, end_date=end_date)
        return {
            "cached_topics": cached_topics,
            "signals_df": signals_df,
        }
