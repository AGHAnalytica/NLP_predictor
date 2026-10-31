"""Historical news cleaning and deduplication pipeline for backtesting datasets."""

import logging
from pathlib import Path

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.preprocessing.clean_data import (
    clean_html_content,
    generate_doc_hash,
    is_valid_article,
    normalize_date_to_utc,
)

logger = logging.getLogger(__name__)


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
