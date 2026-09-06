import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from nlp_predictor.preprocessing.clean_data import (
    clean_html_content,
    generate_doc_hash,
    is_valid_article,
    normalize_date_to_utc,
)

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Base class for all news scrapers.
    Provides a shared HTTP session, request limits, headers, and Parquet-based
    deduplication. Child classes must implement the run() method to fetch and p
    rocess articles.
    """

    def __init__(self, history_file: str = "data/scraping/scraped_history.parquet"):
        self.session = requests.Session()
        self.session.headers.update(
            {
                # change the email later
                "User-Agent": (
                    "AGH_Analityca_scraper/1.0 (contact: papiezpolak2137@gmail.com)",
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.history_file = history_file
        self.history_df = self._load_history()
        self._cleanup_old_history()

    def _load_history(self) -> pd.DataFrame:
        """Loads historical scraped data from Parquet into a Pandas DataFrame."""
        if os.path.exists(self.history_file):
            try:
                df = pd.read_parquet(self.history_file)
                df["scraped_at"] = pd.to_datetime(df["scraped_at"], utc=True)
                return df
            except Exception as e:
                logger.error(
                    f"Error reading Parquet file: {e}. Creating new DataFrame."
                )

        return pd.DataFrame(columns=["doc_hash", "url", "scraped_at"])

    def _cleanup_old_history(self) -> None:
        """R
        emoves rows older than 4 hours from the DataFrame in memory.
        This prevents the historical cache from growing indefinitely and ensures
        that articles can be re-scraped after a reasonable time window.

        After cleanup, the updated DataFrame is not automatically saved to disk; it will
        be saved when save_history() is called at the end of the run() method in child
        classes.

        Returns:
            None
        """
        if self.history_df.empty:
            return

        now = datetime.now(UTC)
        cutoff_time = now - timedelta(hours=4)

        initial_len = len(self.history_df)
        self.history_df = self.history_df[self.history_df["scraped_at"] >= cutoff_time]

        if len(self.history_df) < initial_len:
            logger.info("Successfully cleaned up old historical duplicates from cache.")

    def is_duplicate(self, doc_hash: str, url: str) -> bool:
        """
        Checks if an article with the given hash or URL is in the current DataFrame.
        Returns True if either the hash or URL is found, indicating a duplicate.

        Args:
            doc_hash (str): The unique hash of the article content.
            url (str): The URL of the article.

        Returns:
            bool: True if the article is a duplicate (hash or URL exists),
            False otherwise.
        """
        if self.history_df.empty:
            return False

        is_hash_dup = (
            (self.history_df["doc_hash"] == doc_hash).any() if doc_hash else False
        )
        is_url_dup = (self.history_df["url"] == url).any() if url else False

        return is_hash_dup or is_url_dup

    def mark_as_scraped(self, doc_hash: str, url: str) -> None:
        """
        Appends a new article's metadata to the DataFrame in memory.
        MUST be called for each new article before saving history.

        Args:
            doc_hash (str): The unique hash of the article content.
            url (str): The URL of the article.

        Returns:
            None
        """
        new_row = pd.DataFrame(
            [{"doc_hash": doc_hash, "url": url, "scraped_at": datetime.now(UTC)}]
        )
        self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

    def save_history(self) -> None:
        """
        Saves the updated in-memory DataFrame back to the Parquet file.
        MUST be called at the end of the run() method in child classes.
        """
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            self.history_df.to_parquet(self.history_file, index=False)
            logger.info(f"Successfully saved history cache to {self.history_file}.")
        except Exception as e:
            logger.error(f"Error saving historical database to Parquet: {e}")

    def fetch_html(self, url: str) -> str | None:
        """
        Fetches the HTML content of a given URL.
        Enforces a hard delay to avoid overloading publisher servers.
        If the request is blocked (403 or 429), logs a warning and returns None.
        If any network error occurs, logs the error and returns None.
        Access blocking is handled gracefully by logging a warning and skipping the URL,
        allowing the scraper to continue functioning without crashing.

        Args:
            url (str): The URL to fetch.

        Returns:
            str | None: The HTML content as a string if successful, or None if blocked
        """
        time.sleep(2.5)

        try:
            response = self.session.get(url, timeout=15)

            if response.status_code in (403, 429):
                logger.warning(
                    f"Access blocked for {url} (Status: {response.status_code})."
                    + " Pausing scraper."
                )
                return None

            response.raise_for_status()
            return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while fetching {url}: {e}")
            return None

    def process_article(
        self,
        source: str,
        url: str,
        raw_title: str,
        raw_date: str,
        raw_html: str,
    ) -> dict[str, Any] | None:
        """
        Passes raw data through the cleaner pipeline.
        Returns a standardized article dictionary or None if it's a
        duplicate or invalid.

        Args:
            source (str): The source name of the article (e.g., "AP News").
            url (str): The URL of the article.
            raw_title (str): The raw title extracted from the article metadata.
            raw_date (str): The raw publication date extracted from the article
            metadata.
            raw_html (str): The raw HTML content of the article.

        Returns:
            dict[str, Any] | None: A dictionary with cleaned article data if valid and
            not a duplicate, or None if the article is invalid or a duplicate.
        """
        content = clean_html_content(raw_html)

        if not is_valid_article(content):
            logger.debug(f"Article too short, discarding: {url}")
            return None

        doc_hash = generate_doc_hash(content)

        if self.is_duplicate(doc_hash, url):
            logger.debug(f"Duplicate skipped: {url}")
            return None

        formatted_date = normalize_date_to_utc(raw_date)
        self.mark_as_scraped(doc_hash, url)

        return {
            "doc_hash": doc_hash,
            "source": source,
            "url": url,
            "date": formatted_date,
            "title": raw_title.strip(),
            "content": content,
        }

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """
        Main execution method to be triggered by main.py.
        Each child scraper must return a Pandas DataFrame with columns:
        doc_hash, source, url, date, title, content. The DataFrame should only include
        new articles that are not duplicates of previously scraped content,
        as determined by the is_duplicate() method.

        The run() method should handle the entire scraping and processing workflow
        for the specific news source, including fetching article URLs, extracting
        metadata, cleaning content, and utilizing the process_article() method to
        standardize the output. At the end of the run() method, save_history() must be
        called to persist any new entries in the historical cache.

        Returns:
            pd.DataFrame: A DataFrame containing the new articles scraped during this
            run, with the specified columns.
        """
        pass
