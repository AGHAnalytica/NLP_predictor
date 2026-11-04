"""Filter and clean historical Alpaca news for cost-efficient LLM backtesting.

Removes automated spam digests, short/empty bodies, duplicate syndications, and
filters articles to only those relevant to the target instruments universe or key
macroeconomic market-moving events.
"""

import argparse
import glob
import logging
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nlp_predictor.preprocessing.clean_data import clean_html_content, generate_doc_hash

logger = logging.getLogger("filter_news")

SPAM_PATTERNS = [
    re.compile(
        r"^\d+\s+(stocks|biggest movers|gainers|losers|price target changes|"
        r"movers|penny stocks)",
        re.I,
    ),
    re.compile(r"earnings scheduled for", re.I),
    re.compile(r"economic data scheduled for", re.I),
    re.compile(r"stocks moving in.*pre-market", re.I),
    re.compile(r"stocks moving in.*after-hours", re.I),
    re.compile(r"jim cramer", re.I),
    re.compile(r"top \d+ price prediction", re.I),
    re.compile(r"options alert", re.I),
    re.compile(r"a peek into the markets", re.I),
    re.compile(r"mid-day market update", re.I),
    re.compile(r"recap:", re.I),
    re.compile(r"wrap-up:", re.I),
    re.compile(r"daily biotech pulse", re.I),
    re.compile(r"cryptocurrency market update", re.I),
    re.compile(r"unusual options activity", re.I),
    re.compile(r"premarket movers", re.I),
]

MACRO_PATTERNS = [
    re.compile(
        r"\b(federal reserve|fed\b|jerome powell|interest rate[s]?|rate cut[s]?|"
        r"rate hike[s]?|inflation|cpi\b|ppi\b|pce\b|nonfarm payroll|"
        r"unemployment rate|recession|crude oil|opec|tariff[s]?|gdp\b|"
        r"yield curve|treasury yield[s]?|debt ceiling|trade war)\b",
        re.I,
    )
]

TAG_PATTERN = re.compile(r'href="/stock/([A-Za-z0-9\.\-]+)#', re.I)
CASHTAG_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
EXCHANGE_PATTERN = re.compile(r"\b(NYSE|NASDAQ|AMEX):\s*([A-Z]{1,5})\b")


def is_spam_title(title: str | None) -> bool:
    """Detect whether article title matches automated spam/scanner digests."""
    if not title or not isinstance(title, str):
        return True
    return any(p.search(title) for p in SPAM_PATTERNS)


def is_macro_event(title: str, content: str) -> bool:
    """Detect if title or lead content mentions key macroeconomic indicators."""
    text = f"{title} {content[:600]}"
    return any(p.search(text) for p in MACRO_PATTERNS)


def extract_explicit_tickers(text: str) -> set[str]:
    """Extract explicit ticker mentions from HTML links, cashtags, or exchanges."""
    if not text or not isinstance(text, str):
        return set()
    t1 = set(m.upper() for m in TAG_PATTERN.findall(text))
    t2 = set(m.upper() for m in CASHTAG_PATTERN.findall(text))
    t3 = set(m[1].upper() for m in EXCHANGE_PATTERN.findall(text))
    return t1 | t2 | t3


def load_universe_tickers(instruments_path: Path | None = None) -> set[str]:
    """Load universe tickers from CSV and Xfinlink constituents."""
    tickers: set[str] = set()

    # Load S&P 500 constituents if available
    sp_path = Path("data/historical/xfinlink/constituents_sp500_latest.parquet")
    if sp_path.exists():
        try:
            sp_df = pd.read_parquet(sp_path)
            tickers.update(sp_df["ticker"].dropna().str.upper().tolist())
        except Exception as e:
            logger.warning(f"Could not load S&P 500 constituents: {e}")

    # Load instruments CSV
    inst_csv = (
        instruments_path
        if instruments_path and instruments_path.exists()
        else Path("data/instruments/instruments_sample_large.csv")
    )
    if inst_csv.exists():
        try:
            inst_df = pd.read_csv(inst_csv)
            if "ticker" in inst_df.columns:
                tickers.update(inst_df["ticker"].dropna().str.upper().tolist())
        except Exception as e:
            logger.warning(f"Could not load instruments from {inst_csv}: {e}")

    # Fallback to key ETFs if nothing found
    if not tickers:
        tickers = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"}

    return tickers


def print_cost_summary(num_articles: int) -> None:
    """Print estimated LLM inference costs for processed articles."""
    # Assumptions: ~400 input tokens, ~120 output tokens per article
    in_tokens = num_articles * 400
    out_tokens = num_articles * 120

    # Pricing per 1M tokens (Nov 2024 rates)
    # GPT-4o: $2.50 in, $10.00 out
    cost_4o = (in_tokens / 1_000_000 * 2.50) + (out_tokens / 1_000_000 * 10.00)
    # GPT-4o-mini: $0.15 in, $0.60 out
    cost_mini = (in_tokens / 1_000_000 * 0.15) + (out_tokens / 1_000_000 * 0.60)
    # Batch API is exactly 50% discount
    cost_batch = cost_mini * 0.50

    logger.info("=" * 60)
    logger.info(f"ESTIMATED LLM INFERENCE COSTS ({num_articles:,} articles):")
    logger.info(f"  • GPT-4o (Standard API):       ${cost_4o:,.2f}")
    logger.info(f"  • GPT-4o-mini (Standard API):  ${cost_mini:,.2f}")
    logger.info(
        f"  • GPT-4o-mini (Batch API -50%): ${cost_batch:,.2f} "
        f"(~{cost_batch * 4.0:.1f} PLN)"
    )
    logger.info("=" * 60)


def filter_dataset(
    input_dir: Path,
    output_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    min_length: int = 200,
    instruments_path: Path | None = None,
) -> pd.DataFrame:
    """Filter historical news files and save high-signal articles as Parquet.

    Args:
        input_dir: Directory containing partitioned Parquet files.
        output_path: Target output Parquet file path.
        start_date: Earliest date string (``YYYY-MM-DD``).
        end_date: Latest date string (``YYYY-MM-DD``).
        min_length: Minimum clean content length threshold.
        instruments_path: Optional path to instruments CSV.

    Returns:
        Filtered DataFrame of high-signal articles.
    """
    parquet_files = sorted(glob.glob(f"{input_dir}/**/*.parquet", recursive=True))
    if not parquet_files:
        logger.error(f"No parquet files found in {input_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(parquet_files)} partitioned files in {input_dir}")
    target_tickers = load_universe_tickers(instruments_path)
    logger.info(f"Loaded {len(target_tickers)} universe tickers for matching.")

    # Read all parquet files
    logger.info("Loading parquet tables into memory...")
    tables = [pq.read_table(f) for f in parquet_files]
    df = pa.concat_tables(tables).to_pandas()
    initial_count = len(df)
    logger.info(f"Initial raw articles count: {initial_count:,}")

    # Stage 1: Date filtering
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    if start_date:
        start_dt = pd.to_datetime(start_date, utc=True)
        df = df[df["date"] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date, utc=True)
        df = df[df["date"] <= end_dt]
    after_date_count = len(df)
    logger.info(
        f"Articles in timeframe: {after_date_count:,} "
        f"({initial_count - after_date_count:,} dropped)"
    )

    # Stage 2: Spam & Automated Scanner Filter
    logger.info("Filtering out automated spam, calendars, and scanner lists...")
    mask_spam = df["title"].apply(is_spam_title)
    spam_dropped = mask_spam.sum()
    df = df[~mask_spam].copy()
    logger.info(f"Spam digests dropped: {spam_dropped:,}")

    # Stage 3: Clean HTML and Length check
    logger.info("Checking clean content length...")
    # Fast HTML tag stripping / clean
    df["content_clean"] = df["content"].astype(str).apply(clean_html_content)
    mask_short = df["content_clean"].str.len() < min_length
    short_dropped = mask_short.sum()
    df = df[~mask_short].copy()
    logger.info(f"Short articles (<{min_length} chars) dropped: {short_dropped:,}")

    # Stage 4: Universe & Macro Event Relevance
    logger.info("Matching articles against universe tickers and macro topics...")
    explicit_tickers = df["content"].apply(extract_explicit_tickers)
    has_target = explicit_tickers.apply(lambda s: bool(s & target_tickers))
    is_macro = [
        is_macro_event(t, c)
        for t, c in zip(df["title"], df["content_clean"], strict=False)
    ]
    mask_relevant = has_target | is_macro
    irrelevant_dropped = (~mask_relevant).sum()
    df = df[mask_relevant].copy()
    logger.info(f"Irrelevant non-target articles dropped: {irrelevant_dropped:,}")

    # Stage 5: Hash & Deduplication
    logger.info("Deduplicating identical articles...")
    df["doc_hash"] = df["content_clean"].apply(generate_doc_hash)
    dup_before = len(df)
    df = df.drop_duplicates(subset=["doc_hash"]).copy()
    duplicates_dropped = dup_before - len(df)
    logger.info(f"Duplicate articles dropped: {duplicates_dropped:,}")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop temp column and keep standard schema
    df["content"] = df["content_clean"]
    df = df.drop(columns=["content_clean"])
    df.to_parquet(output_path, index=False)

    final_count = len(df)
    reduction = (1 - final_count / initial_count) * 100
    logger.info("=" * 60)
    logger.info(
        f"FILTERING COMPLETE! Final high-signal dataset: {final_count:,} articles"
    )
    logger.info(
        f"Total volume reduced by {reduction:.1f}% "
        f"({initial_count - final_count:,} articles removed)"
    )
    logger.info(f"Saved clean filtered Parquet to: {output_path}")

    print_cost_summary(final_count)
    return df


def main() -> None:
    """CLI orchestrator for historical news filtering."""
    parser = argparse.ArgumentParser(
        description="Filter Alpaca news articles for efficient LLM backtesting."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/historical/alpaca_news_partitioned",
        help="Path to partitioned Parquet directory.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/historical/alpaca_news_filtered.parquet",
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
        default=200,
        help="Minimum character count for article content.",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default="data/instruments/instruments_sample_large.csv",
        help="Path to instruments universe CSV.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )

    filter_dataset(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output_path),
        start_date=args.start_date,
        end_date=args.end_date,
        min_length=args.min_length,
        instruments_path=Path(args.instruments),
    )


if __name__ == "__main__":
    main()
