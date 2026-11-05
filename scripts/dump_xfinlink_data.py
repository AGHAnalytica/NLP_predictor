"""Comprehensive data dumping script to cache all Xfinlink datasets locally.

Downloads survivorship-bias-free index rosters, additions/removals event logs,
point-in-time constituent snapshots, split-adjusted daily OHLCV bars, SEC Form 3/4/5
insider transactions, Form 13F institutional holdings, SEC as-reported
fundamentals, and valuation metrics for all target tickers.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from nlp_predictor.backtesting.data.xfinlink_client import XfinlinkClient
from nlp_predictor.config import config

logger = logging.getLogger("dump_xfinlink")


def format_bytes(size: int) -> str:
    """Format byte size into human readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(size)
    for u in units:
        if val < 1024.0:
            return f"{val:.2f} {u}"
        val /= 1024.0
    return f"{val:.2f} PB"


def main() -> None:
    """CLI orchestrator to dump all available Xfinlink data for research."""
    xfl_cfg = config.backtesting.xfinlink
    default_indices = ",".join(xfl_cfg.indices)

    parser = argparse.ArgumentParser(
        description="Comprehensive data dump script for Xfinlink."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("XFINLINK_API_KEY", ""),
        help="Xfinlink API key (or set XFINLINK_API_KEY env var).",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=default_indices,
        help="Comma-separated index list (sp500, ndx100, djia).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=str(xfl_cfg.start_date),
        help="Start date YYYY-MM-DD for fundamentals/prices.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="Shorthand period (e.g. 'max', '10y', '5y'). Overrides start/end.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.paths.historical.xfinlink),
        help="Output directory for cached Parquet files.",
    )
    parser.add_argument(
        "--snapshots-frequency",
        type=str,
        choices=["yearly", "monthly", "none"],
        default=str(xfl_cfg.snapshots_frequency),
        help="Constituent snapshot cadence (yearly, monthly, none).",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated ticker list (uses index constituents if empty).",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Limit number of tickers to download (0 = all).",
    )
    parser.add_argument(
        "--include-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch historical index additions and removals event logs.",
    )
    parser.add_argument(
        "--include-historical-tickers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include all historical added/removed tickers (eliminates bias).",
    )
    parser.add_argument(
        "--include-quarterly",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch quarterly fundamentals and metrics in addition to annual.",
    )
    parser.add_argument(
        "--include-ttm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch TTM metrics in addition to annual and quarterly.",
    )
    parser.add_argument(
        "--include-insiders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch SEC Form 4 insider transactions.",
    )
    parser.add_argument(
        "--include-holdings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch institutional Form 13F quarterly holdings.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip downloading if local Parquet file already exists.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(xfl_cfg.request_delay),
        help="Politeness delay in seconds between requests.",
    )
    parser.add_argument(
        "--dump-everything",
        action="store_true",
        help="Master switch to download everything available across all indices.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )

    if args.dump_everything:
        args.indices = default_indices
        args.include_events = True
        args.include_historical_tickers = True
        args.include_quarterly = True
        args.include_ttm = True
        args.include_insiders = True
        args.include_holdings = True
        args.snapshots_frequency = str(xfl_cfg.snapshots_frequency)
        args.skip_existing = True
        if not args.period:
            args.period = "max"

    out_path = Path(args.output_dir)
    client = XfinlinkClient(api_key=args.api_key, output_dir=out_path)

    # 1. API Status check
    try:
        status = client.get_status()
        logger.info(f"Connected to Xfinlink. Status: {status}")
        if args.api_key:
            usage = client.get_usage()
            logger.info(f"Account usage & plan: {usage}")
    except Exception as e:
        logger.error(f"Failed to query Xfinlink status: {e}")

    # 2. Parse target indices
    index_list = [idx.strip().lower() for idx in args.indices.split(",") if idx.strip()]
    logger.info(f"Target indices for processing: {index_list}")

    all_tickers: set[str] = set()

    for idx_name in index_list:
        logger.info(f"--- Processing Index: {idx_name.upper()} ---")
        # 2a. Current constituents
        try:
            constituents_df = client.fetch_index_constituents(
                index_name=idx_name,
                save=True,
                skip_existing=args.skip_existing,
            )
            if not constituents_df.empty and "ticker" in constituents_df.columns:
                tickers = constituents_df["ticker"].dropna().astype(str).tolist()
                all_tickers.update(t.strip().upper() for t in tickers if t.strip())
        except Exception as e:
            logger.error(f"Error fetching constituents for {idx_name}: {e}")

        # 2b. Index events (additions and removals)
        if args.include_events:
            try:
                events_df = client.fetch_index_events(
                    index_name=idx_name,
                    save=True,
                    skip_existing=args.skip_existing,
                )
                if (
                    args.include_historical_tickers
                    and not events_df.empty
                    and "ticker" in events_df.columns
                ):
                    hist_tickers = events_df["ticker"].dropna().astype(str).tolist()
                    all_tickers.update(
                        t.strip().upper() for t in hist_tickers if t.strip()
                    )
            except Exception as e:
                logger.error(f"Error fetching events for {idx_name}: {e}")

        # 2c. Point-in-time constituent snapshots
        if args.snapshots_frequency != "none":
            start_year = (
                int(args.start_date.split("-")[0])
                if args.start_date
                else datetime.now().year - 10
            )
            end_year = (
                int(args.end_date.split("-")[0])
                if args.end_date
                else datetime.now().year
            )

            snapshot_dates: list[str] = []
            if args.snapshots_frequency == "yearly":
                snapshot_dates = [f"{y}-01-15" for y in range(start_year, end_year + 1)]
            elif args.snapshots_frequency == "monthly":
                for y in range(start_year, end_year + 1):
                    for m in range(1, 13):
                        snapshot_dates.append(f"{y}-{m:02d}-15")

            logger.info(
                f"Generating {len(snapshot_dates)} {args.snapshots_frequency} "
                f"snapshots for {idx_name.upper()}..."
            )
            try:
                snap_df = client.fetch_index_history_snapshots(
                    index_name=idx_name,
                    dates=snapshot_dates,
                    delay_seconds=args.delay,
                    skip_existing=args.skip_existing,
                )
                if (
                    args.include_historical_tickers
                    and not snap_df.empty
                    and "ticker" in snap_df.columns
                ):
                    snap_tickers = snap_df["ticker"].dropna().astype(str).tolist()
                    all_tickers.update(
                        t.strip().upper() for t in snap_tickers if t.strip()
                    )
            except Exception as e:
                logger.error(f"Error fetching snapshots for {idx_name}: {e}")

    # 3. Resolve target tickers list
    if args.tickers.strip():
        target_tickers = [
            t.strip().upper() for t in args.tickers.split(",") if t.strip()
        ]
    else:
        target_tickers = sorted(all_tickers)

    if args.max_tickers > 0 and len(target_tickers) > args.max_tickers:
        logger.info(
            f"Limiting tickers list from {len(target_tickers)} to {args.max_tickers}"
        )
        target_tickers = target_tickers[: args.max_tickers]

    logger.info(f"Total resolved tickers for full dump: {len(target_tickers)}")

    # 4. Configure periods
    fundamentals_periods = (
        ["annual", "quarterly"] if args.include_quarterly else ["annual"]
    )
    metrics_periods = ["annual"]
    if args.include_quarterly:
        metrics_periods.append("quarterly")
    if args.include_ttm:
        metrics_periods.append("ttm")

    # 5. Dump universe data
    if target_tickers:
        stats = client.dump_universe_data(
            tickers=target_tickers,
            start_date=args.start_date if not args.period else None,
            end_date=args.end_date if not args.period else None,
            period=args.period,
            fetch_fundamentals_data=True,
            fundamentals_periods=fundamentals_periods,
            fundamentals_version=str(xfl_cfg.fundamentals_version),
            fetch_metrics_data=True,
            metrics_periods=metrics_periods,
            fetch_prices_data=True,
            fetch_insiders_data=args.include_insiders,
            fetch_holdings_data=args.include_holdings,
            skip_existing=args.skip_existing,
            delay_seconds=args.delay,
        )
        logger.info(f"Dump complete! Statistics: {stats}")

    # 6. Disk usage summary
    parquet_files = list(out_path.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    logger.info(
        f"Storage Summary: {len(parquet_files)} Parquet files cached in "
        f"{out_path} ({format_bytes(total_size)})"
    )


if __name__ == "__main__":
    main()
