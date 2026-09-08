"""
Module for fetching, hashing, and storing historical news data from Alpaca API.

Provides resilient pagination, rate-limit backoff, and chunked Parquet export
using Hive partitioning schemes.
"""

import logging
import os
import time
import uuid

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout

from nlp_predictor.config import config
from nlp_predictor.preprocessing.clean_data import generate_doc_hash

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

DEFAULT_OUTPUT_DIR = str(config.paths.historical.partitioned_dir)


def _flush_batch_to_parquet(records: list[dict], output_dir: str) -> None:
    """
    Write a batch of collected records into Hive-partitioned Parquet files.

    Generates unique filenames to prevent accidental overwrite of existing
    partitions.

    Args:
        records: List of article dictionaries.
        output_dir: Root directory for the partitioned Parquet files.
    """
    if not records:
        return

    df = pd.DataFrame(records)
    expected_cols = ["doc_hash", "source", "url", "date", "title", "content"]
    df = df[expected_cols]

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Group by partition to avoid overwriting existing data with delete_matching
    for (year, month), group in df.groupby(["year", "month"]):
        partition_dir = os.path.join(output_dir, f"year={year}", f"month={month}")
        os.makedirs(partition_dir, exist_ok=True)
        file_path = os.path.join(partition_dir, f"batch_{uuid.uuid4().hex[:8]}.parquet")
        group.to_parquet(file_path, index=False, engine="pyarrow")

    logger.info(f"Flushed {len(df)} records to disk across partitions.")


class AlpacaHistoricalFetcher:
    """Fetcher for historical news articles from the Alpaca News API."""

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir

    def fetch(
        self,
        start_date: str,
        end_date: str,
        max_articles: int = 1_000_000,
        batch_flush_size: int = 5_000,
    ) -> int:
        """
        Fetch historical news articles from the Alpaca API and stream to Parquet files.

        Implements token-based pagination, exponential backoff for rate limits,
        network failure handling, and periodic disk flushing to prevent OOM errors.

        Args:
            start_date: RFC3339 formatted start timestamp
                (e.g., '2024-01-01T00:00:00Z').
            end_date: RFC3339 formatted end timestamp (e.g., '2024-12-31T23:59:59Z').
            max_articles: Upper bound of articles to retrieve.
            batch_flush_size: Number of records to accumulate before flushing to disk.

        Returns:
            Total number of articles successfully downloaded and persisted.

        Raises:
            ValueError: If Alpaca API credentials are not found in the environment.
            RequestException: If API requests fail persistently after maximum retries.
        """
        api_key = os.getenv("ALPACA_API_KEY_ID")
        api_secret = os.getenv("ALPACA_API_SECRET_KEY")

        if not api_key or not api_secret:
            raise ValueError(
                "Missing ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY in environment."
            )

        url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        params: dict[str, object] = {
            "start": start_date,
            "end": end_date,
            "sort": "asc",
            "limit": 50,
            "include_content": True,
            "exclude_contentless": True,
        }

        collected_records: list[dict] = []
        page_token: str | None = None
        total_fetched = 0
        max_retries = 5

        logger.info(f"Starting historical fetch: {start_date} to {end_date}")

        while total_fetched < max_articles:
            if page_token:
                params["page_token"] = page_token

            data = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.get(
                        url, headers=headers, params=params, timeout=30
                    )

                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 10))
                        logger.warning(
                            f"Rate limited (429). Sleeping for {retry_after}s "
                            f"(Attempt {attempt}/{max_retries})..."
                        )
                        time.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    break

                except (Timeout, RequestException) as err:
                    sleep_time = attempt * 3
                    logger.warning(
                        f"Request issue on attempt {attempt}/{max_retries}: {err}. "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                    if attempt == max_retries:
                        # Flush remaining items before failing
                        _flush_batch_to_parquet(collected_records, self.output_dir)
                        logger.error(
                            "Max retries reached. Persisted buffer and aborting."
                        )
                        raise

            if not data:
                break

            articles = data.get("news", [])
            if not articles:
                logger.info("No articles returned for the current page query.")
                break

            for item in articles:
                article_id = str(item.get("id"))
                article_url = (
                    item.get("url") or f"https://alpaca.markets/news/{article_id}"
                )
                content = item.get("content", item.get("summary", ""))
                doc_hash = generate_doc_hash(
                    content or item.get("headline", "") or article_id
                )

                record = {
                    "doc_hash": doc_hash,
                    "source": item.get("source", "alpaca"),
                    "url": article_url,
                    "date": item.get("created_at"),
                    "title": item.get("headline", ""),
                    "content": content,
                }
                collected_records.append(record)
                total_fetched += 1

                if len(collected_records) >= batch_flush_size:
                    _flush_batch_to_parquet(collected_records, self.output_dir)
                    collected_records.clear()

                if total_fetched >= max_articles:
                    break

            logger.info(f"Progress: {total_fetched}/{max_articles} articles fetched.")

            page_token = data.get("next_page_token")
            if not page_token:
                logger.info("Pagination complete: next_page_token is null.")
                break

            time.sleep(0.1)

        # Flush any remaining items in the buffer
        if collected_records:
            _flush_batch_to_parquet(collected_records, self.output_dir)
            collected_records.clear()

        return total_fetched


def fetch_alpaca_historical_news(
    start_date: str,
    end_date: str,
    max_articles: int = 1_000_000,
    batch_flush_size: int = 5_000,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> int:
    """
    Procedural wrapper for AlpacaHistoricalFetcher.
    """
    fetcher = AlpacaHistoricalFetcher(output_dir=output_dir)
    return fetcher.fetch(
        start_date=start_date,
        end_date=end_date,
        max_articles=max_articles,
        batch_flush_size=batch_flush_size,
    )


if __name__ == "__main__":
    years_range = [str(year) for year in range(2015, 2026)]

    for year in years_range:
        try:
            fetch_alpaca_historical_news(
                start_date=f"{year}-01-01T00:00:00Z",
                end_date=f"{year}-12-31T23:59:59Z",
                max_articles=1_000_000,
            )
        except Exception as e:
            logger.error(f"Failed to process year {year}. Error: {e}")
