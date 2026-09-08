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


def clean_historical_news(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    """
    Reads a raw historical Parquet file or directory, applies standard text cleaning
    and validation, and saves the production-ready dataset.

    Args:
        input_path: Path to the raw input Parquet file or partitioned directory.
            Defaults to partitioned_dir if it exists, otherwise news_parquet.
        output_path: Path where the cleaned Parquet file will be saved.
            Defaults to news_clean.
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

    df["is_valid"] = df["content"].apply(is_valid_article)
    df = df[df["is_valid"]].copy()
    df.drop(columns=["is_valid"], inplace=True)

    df["date"] = df["date"].astype(str).apply(normalize_date_to_utc)
    df["doc_hash"] = df["content"].apply(generate_doc_hash)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    final_count = len(df)
    logger.info(f"Success! Clean dataset saved to: {output_path}")
    logger.info(
        f"Number of articles after cleaning: {final_count} "
        f"({initial_count - final_count} removed)"
    )


# Backwards compatibility alias
process_historical_data = clean_historical_news


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    clean_historical_news()
