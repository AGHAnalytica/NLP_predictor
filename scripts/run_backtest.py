"""CLI runner executing the complete historical backtesting pipeline end to end.

Stages:
1. Build or refresh the survivorship-free instrument universe CSV from local Xfinlink
   index snapshots (`--refresh-universe`).
2. Ingest unseen historical articles from `alpaca_news_filtered.parquet` into the
   unpruned historical FeatureStore in checkpointed batches (`--skip-ingest` to skip).
3. Replay `TickRunner` chronologically across historical dates into the historical
   SignalLog (`--skip-replay` to skip).
4. Evaluate all configured A/B/n severity variants in `PyBrokerBacktester` and persist
   metrics, equity curves, trades, and orders to `data/backtest_results/`.
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from nlp_predictor.backtesting import (
    BacktestDataLoader,
    PyBrokerBacktester,
)
from nlp_predictor.backtesting.replayer import (
    HistoricalReplayer,
    build_universe_instruments_csv,
)
from nlp_predictor.config import config
from nlp_predictor.core.logging_config import setup_logging
from nlp_predictor.signal.signal_log import SignalLog

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the historical backtest orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run historical NLP feature replay and PyBroker backtesting."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=config.backtesting.timeframe.start_date,
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=config.backtesting.timeframe.end_date,
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--articles-path",
        type=Path,
        default=config.paths.historical.news_filtered,
        help="Path to filtered historical news Parquet file.",
    )
    parser.add_argument(
        "--feature-store-path",
        type=Path,
        default=config.paths.historical.feature_store,
        help="Path to historical unpruned FeatureStore Parquet file.",
    )
    parser.add_argument(
        "--signal-log-path",
        type=Path,
        default=config.paths.historical.signal_log,
        help="Path to historical SignalLog Parquet file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.backtesting.replay.ingest_batch_size,
        help="Number of unseen articles per LLM checkpoint batch.",
    )
    parser.add_argument(
        "--refresh-universe",
        action="store_true",
        help="Rebuild instruments_sample_large.csv from Xfinlink index rosters.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip LLM feature ingestion and use existing historical FeatureStore.",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip chronological TickRunner replay and use existing SignalLog.",
    )
    parser.add_argument(
        "--skip-pybroker",
        action="store_true",
        help="Skip PyBroker portfolio simulation after generating historical signals.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the configured historical replay and PyBroker simulation stages."""
    load_dotenv(override=True)
    if not logging.getLogger().hasHandlers():
        setup_logging()

    args = parse_args()

    if args.refresh_universe:
        logger.info("====== [0/3] Refreshing Universe Instruments CSV ======")
        build_universe_instruments_csv()

    if not args.skip_ingest or not args.skip_replay:
        replayer = HistoricalReplayer.build_default(
            feature_store_path=args.feature_store_path,
            signal_log_path=args.signal_log_path,
        )

        if not args.skip_ingest:
            logger.info("====== [1/3] Ingesting Historical Article Features ======")
            replayer.ingest_historical_articles(
                articles_path=args.articles_path,
                start_date=args.start_date,
                end_date=args.end_date,
                batch_size=args.batch_size,
            )

        if not args.skip_replay:
            logger.info("====== [2/3] Replaying Historical Evaluation Ticks ======")
            replayer.replay_ticks(
                start_date=args.start_date,
                end_date=args.end_date,
            )

    if not args.skip_pybroker:
        logger.info("====== [3/3] Running PyBroker Multi-Variant Backtest ======")
        hist_signal_log = SignalLog(path=str(args.signal_log_path))
        loader = BacktestDataLoader(signal_log=hist_signal_log)
        backtester = PyBrokerBacktester(
            data_loader=loader,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        summary_df = backtester.run_all_variants(
            start_date=args.start_date,
            end_date=args.end_date,
            save_outputs=True,
        )
        if summary_df.empty:
            logger.warning(
                "PyBroker simulation produced no variant metrics "
                "(check that historical signals and local price files overlap)."
            )
        else:
            key_cols = [
                col
                for col in (
                    "variant",
                    "trade_count",
                    "total_return_pct",
                    "annual_return_pct",
                    "max_drawdown_pct",
                    "sharpe",
                    "sortino",
                    "win_rate",
                    "profit_factor",
                )
                if col in summary_df.columns
            ]
            logger.info(
                "PyBroker Multi-Variant Summary:\n"
                f"{summary_df[key_cols].to_string(index=False)}"
            )


if __name__ == "__main__":
    main()
