"""CLI entry point to filter and clean historical Alpaca news for LLM backtesting."""

import argparse
import logging
from pathlib import Path

from nlp_predictor.backtesting.data.clean_news import filter_historical_news
from nlp_predictor.config import config


def main() -> None:
    """CLI orchestrator for historical news filtering."""
    parser = argparse.ArgumentParser(
        description="Filter Alpaca news articles for efficient LLM backtesting."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(config.paths.historical.partitioned_dir),
        help="Path to partitioned Parquet directory.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(config.paths.historical.news_filtered),
        help="Output filtered Parquet file destination.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date YYYY-MM-DD (e.g. 2020-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date YYYY-MM-DD.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=int(config.backtesting.filtering.min_article_length),
        help="Minimum character count for article content.",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=str(config.paths.input.instruments_large),
        help="Path to instruments universe CSV.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )

    filter_historical_news(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output_path),
        start_date=args.start_date,
        end_date=args.end_date,
        min_length=args.min_length,
        instruments_path=Path(args.instruments),
    )


if __name__ == "__main__":
    main()
