"""Historical news cleaning, spam filtering, and deduplication pipeline."""

import glob
import logging
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nlp_predictor.config import config
from nlp_predictor.preprocessing.clean_data import (
    clean_html_content,
    generate_doc_hash,
    is_valid_article,
    normalize_date_to_utc,
    strip_html_fragments,
)

logger = logging.getLogger(__name__)

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
    """Detect whether an article title matches automated spam or scanner digests."""
    if not title or not isinstance(title, str):
        return True
    return any(pattern.search(title) for pattern in SPAM_PATTERNS)


def is_macro_event(title: str, content: str) -> bool:
    """Detect if title or lead content mentions key macroeconomic indicators."""
    lead_chars = int(config.backtesting.filtering.macro_lead_chars)
    text = f"{title} {content[:lead_chars]}"
    return any(pattern.search(text) for pattern in MACRO_PATTERNS)


def extract_explicit_tickers(text: str) -> set[str]:
    """Extract explicit ticker mentions from HTML links, cashtags, or exchanges."""
    if not text or not isinstance(text, str):
        return set()
    stock_links = {match.upper() for match in TAG_PATTERN.findall(text)}
    cashtags = {match.upper() for match in CASHTAG_PATTERN.findall(text)}
    exchanges = {match[1].upper() for match in EXCHANGE_PATTERN.findall(text)}
    return stock_links | cashtags | exchanges


def load_universe_tickers(instruments_path: Path | None = None) -> set[str]:
    """Load universe tickers from CSV and Xfinlink S&P 500 constituents."""
    tickers: set[str] = set()

    sp_file = str(config.backtesting.filtering.sp500_constituents_file)
    sp_path = config.paths.historical.xfinlink / sp_file
    if sp_path.exists():
        try:
            sp_df = pd.read_parquet(sp_path)
            tickers.update(sp_df["ticker"].dropna().str.upper().tolist())
        except Exception as e:
            logger.warning(f"Could not load S&P 500 constituents: {e}")

    inst_csv = (
        instruments_path
        if instruments_path and instruments_path.exists()
        else config.paths.input.instruments_large
    )
    if inst_csv.exists():
        try:
            inst_df = pd.read_csv(inst_csv)
            if "ticker" in inst_df.columns:
                tickers.update(inst_df["ticker"].dropna().str.upper().tolist())
        except Exception as e:
            logger.warning(f"Could not load instruments from {inst_csv}: {e}")

    if not tickers:
        tickers = set(config.backtesting.data.macro_tickers)

    return tickers


def filter_historical_news(
    input_dir: Path | None = None,
    output_path: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_length: int | None = None,
    instruments_path: Path | None = None,
) -> pd.DataFrame:
    """Filter historical news files and save high-signal articles as Parquet.

    Args:
        input_dir: Directory containing partitioned Parquet files (or single file).
        output_path: Target output Parquet file path.
        start_date: Earliest date string (``YYYY-MM-DD``).
        end_date: Latest date string (``YYYY-MM-DD``).
        min_length: Minimum clean content length threshold.
        instruments_path: Optional path to instruments CSV.

    Returns:
        Filtered DataFrame of high-signal articles.
    """
    input_dir = input_dir or config.paths.historical.partitioned_dir
    output_path = output_path or config.paths.historical.news_filtered
    min_len = (
        int(min_length)
        if min_length is not None
        else int(config.backtesting.filtering.min_article_length)
    )

    if input_dir.is_file():
        parquet_files = [str(input_dir)]
    else:
        parquet_files = sorted(glob.glob(f"{input_dir}/**/*.parquet", recursive=True))

    if not parquet_files:
        logger.error(f"No parquet files found in {input_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(parquet_files)} parquet file(s) in {input_dir}")
    target_tickers = load_universe_tickers(instruments_path)
    logger.info(f"Loaded {len(target_tickers)} universe tickers for matching.")

    tables = [pq.read_table(file_path) for file_path in parquet_files]
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
    mask_spam = df["title"].apply(is_spam_title)
    spam_dropped = int(mask_spam.sum())
    df = df[~mask_spam].copy()
    logger.info(f"Spam digests dropped: {spam_dropped:,}")

    # Stage 3: Clean HTML and Length check
    df["content_clean"] = df["content"].astype(str).apply(strip_html_fragments)
    mask_short = df["content_clean"].str.len() < min_len
    short_dropped = int(mask_short.sum())
    df = df[~mask_short].copy()
    logger.info(f"Short articles (<{min_len} chars) dropped: {short_dropped:,}")

    # Stage 4: Universe & Macro Event Relevance
    explicit_tickers = df["content"].apply(extract_explicit_tickers)
    has_target = explicit_tickers.apply(lambda s: bool(s & target_tickers))
    is_macro = pd.Series(
        [
            is_macro_event(title, content)
            for title, content in zip(df["title"], df["content_clean"], strict=False)
        ],
        index=df.index,
        dtype=bool,
    )
    mask_relevant = has_target | is_macro
    irrelevant_dropped = int((~mask_relevant).sum())
    df = df[mask_relevant].copy()
    logger.info(f"Irrelevant non-target articles dropped: {irrelevant_dropped:,}")

    # Stage 5: Hash & Deduplication
    df["doc_hash"] = df["content_clean"].apply(generate_doc_hash)
    dup_before = len(df)
    df = df.drop_duplicates(subset=["doc_hash"]).copy()
    duplicates_dropped = dup_before - len(df)
    logger.info(f"Duplicate articles dropped: {duplicates_dropped:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df["content"] = df["content_clean"]
    df = df.drop(columns=["content_clean"])
    df.to_parquet(output_path, index=False, engine="pyarrow")

    final_count = len(df)
    reduction = (1 - final_count / initial_count) * 100 if initial_count else 0.0
    logger.info(
        f"FILTERING COMPLETE! Final high-signal dataset: {final_count:,} articles "
        f"({reduction:.1f}% reduction). Saved to: {output_path}"
    )
    return df


filter_dataset = filter_historical_news


def _parse_dates_vectorized(date_series: pd.Series) -> pd.Series:
    """Parse timestamps via vectorized ``pd.to_datetime`` with ``dateparser``
    fallback.
    """
    parsed = pd.to_datetime(date_series, utc=True, errors="coerce")
    missing_mask = parsed.isna()
    if missing_mask.any():
        fallback = date_series[missing_mask].astype(str).apply(normalize_date_to_utc)
        parsed.loc[missing_mask] = pd.to_datetime(fallback, utc=True, errors="coerce")
    return parsed


def clean_historical_news(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    """Clean, validate, and hash raw historical news Parquet data.

    Args:
        input_path: Raw Parquet file or partitioned directory path. Defaults to
            ``config.paths.historical.partitioned_dir`` if non-empty, else
            ``config.paths.historical.news_parquet``.
        output_path: Output Parquet file path. Defaults to
            ``config.paths.historical.news_clean``.
    """
    if input_path is None:
        partitioned = config.paths.historical.partitioned_dir
        if partitioned.exists() and any(partitioned.iterdir()):
            input_path = partitioned
        else:
            input_path = config.paths.historical.news_parquet

    if output_path is None:
        output_path = config.paths.historical.news_clean

    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Loading raw parquet data from {input_path}")
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    df = pd.read_parquet(input_path)
    initial_count = len(df)
    logger.info(f"Number of articles loaded before cleaning: {initial_count}")

    logger.info("Cleaning in progress...")
    df["content"] = df["content"].astype(str).apply(clean_html_content)
    df = df[df["content"].apply(is_valid_article)].copy()

    df["date"] = _parse_dates_vectorized(df["date"])
    df["doc_hash"] = df["content"].apply(generate_doc_hash)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    final_count = len(df)
    logger.info(f"Success! Clean dataset saved to: {output_path}")
    logger.info(
        f"Number of articles after cleaning: {final_count} "
        f"({initial_count - final_count} removed)"
    )


process_historical_data = clean_historical_news


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    clean_historical_news()
