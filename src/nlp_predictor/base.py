import time
import sqlite3
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests

from src.nlp_predictor.data_prep.cleaner import (
    clean_html_content,
    is_valid_article,
    generate_doc_hash,
    normalize_date_to_utc
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Base class for all news scrapers.
    Provides a shared HTTP session, request limits, headers, and database deduplication.
    """

    def __init__(self, db_path: str = "data/scraped_history.db"):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ANALScraper/1.0 (contact: wojtczakkf@gmail.com)",
            "Accept-Language": "en-US,en;q=0.9"
        })
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """
        Initializes an SQLite table to track previously scraped articles
        and prevent processing duplicates in subsequent runs.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                             CREATE TABLE IF NOT EXISTS history
                             (
                                 doc_hash
                                 TEXT
                                 PRIMARY
                                 KEY,
                                 url
                                 TEXT
                                 UNIQUE,
                                 scraped_at
                                 TIMESTAMP
                                 DEFAULT
                                 CURRENT_TIMESTAMP
                             )
                             ''')
        except sqlite3.Error as e:
            logger.error(f"SQLite DB initialization error: {e}")

    def _cleanup_old_history(self) -> None:
        """
        Removes articles older than 4 hours from the history table
        to keep the database size optimized.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Wymiatamy wszystko, co ma timestamp starszy niż 4 godziny
                conn.execute("DELETE FROM history WHERE scraped_at < datetime('now', '-4 hours')")
                logger.info("Successfully cleaned up old historical duplicates from SQLite.")
        except sqlite3.Error as e:
            logger.error(f"Error during historical database cleanup: {e}")

    def is_duplicate(self, doc_hash: str, url: str) -> bool:
        """
        Checks if an article with the given hash or URL has already been scraped.
        """
        if not doc_hash and not url:
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM history WHERE doc_hash = ? OR url = ?",
                    (doc_hash, url)
                )
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Error checking for duplicates: {e}")
            return False

    def mark_as_scraped(self, doc_hash: str, url: str) -> None:
        """
        Saves the article's hash and URL to the historical database
        to bypass it during the next execution cycle.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO history (doc_hash, url) VALUES (?, ?)",
                    (doc_hash, url)
                )
        except sqlite3.Error as e:
            logger.error(f"Error saving to historical database: {e}")

    def fetch_html(self, url: str) -> Optional[str]:
        """
        Fetches the HTML content of a given URL.
        Enforces a hard delay to avoid overloading publisher servers.
        """
        time.sleep(2.5)

        try:
            response = self.session.get(url, timeout=15)

            if response.status_code in (403, 429):
                logger.warning(f"Access blocked for {url} (Status: {response.status_code}). Pausing scraper.")
                return None

            response.raise_for_status()
            return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while fetching {url}: {e}")
            return None

    def process_article(self, source: str, url: str, raw_title: str, raw_date: str, raw_html: str) -> Optional[
        Dict[str, Any]]:
        """
        Passes raw data through the cleaner pipeline.
        Returns a standardized article dictionary or None if it's a duplicate or invalid.
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
            "content": content
        }

    @abstractmethod
    def run(self) -> List[Dict[str, Any]]:
        """
        Main execution method to be triggered by main.py.
        Each child scraper must implement its specific extraction logic here.
        """
        pass