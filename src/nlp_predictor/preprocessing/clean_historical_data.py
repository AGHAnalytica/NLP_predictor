import logging
import os

import pandas as pd
from clean_data import (
    clean_html_content,
    generate_doc_hash,
    is_valid_article,
    normalize_date_to_utc,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_historical_data(input_path: str, output_path: str):
    """
    Reads a raw historical Parquet file, applies standard text cleaning
    and validation, and saves the production-ready dataset.

    Args:
        input_path (str): The absolute or relative path to the raw input Parquet file.
        output_path (str): The absolute or relative path where the cleaned Parquet file
        will be saved.
    """

    logger.info(f"Loading raw parquet data from {input_path}")

    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    final_count = len(df)
    logger.info(f"Success! Clean dataset saved to: {output_path}")
    logger.info(
        "Number of articles after cleaning: "
        + f"{final_count} ({initial_count - final_count} removed)"
    )


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # testing, testing !!!
    RAW_FILE = os.path.join(CURRENT_DIR, "data", "historical", "alpaca_news.parquet")
    CLEANED_FILE = os.path.join(
        CURRENT_DIR, "data", "historical", "alpaca_news_clean.parquet"
    )

    process_historical_data(input_path=RAW_FILE, output_path=CLEANED_FILE)
