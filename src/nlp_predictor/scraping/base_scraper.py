"""Extensible base class for web, RSS, and text/API news scrapers."""

import asyncio
import logging
import os
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp
import pandas as pd
import requests

from nlp_predictor.config import config
from nlp_predictor.preprocessing.clean_data import (
    MIN_ARTICLE_LENGTH,
    clean_html_content,
    generate_doc_hash,
    is_valid_article,
    normalize_date_to_utc,
    normalize_plain_text,
)

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Abstract base scraper providing HTTP/async fetching, deduplication, and cleaning.

    Supports two ingestion modes for subclasses:

    1. **HTML/Web & RSS sources**: Extract text via ``process_article`` (or run
       ``_run_rss_pipeline`` for standard RSS feeds).
    2. **API / Social Media sources (e.g., Twitter/X, wire feeds)**: Pass raw text
       directly to ``process_text_item`` and customize ``min_content_length`` for
       short-form posts.

    Output schema (``ARTICLE_COLUMNS``):
        ``[doc_hash, source, url, date, title, content]``
    """

    ARTICLE_COLUMNS = ["doc_hash", "source", "url", "date", "title", "content"]
    HISTORY_COLUMNS = ["doc_hash", "url", "scraped_at"]
    HISTORY_RETENTION_HOURS = float(config.hyperparams.scraping.history_retention_hours)

    def __init__(
        self,
        history_file: str | None = None,
        min_content_length: int = MIN_ARTICLE_LENGTH,
        fetch_delay: float | None = None,
    ) -> None:
        """Initialize the scraper session and load deduplication history.

        Args:
            history_file: Parquet path for scraped URL/hash history.
            min_content_length: Minimum cleaned character count to accept an item.
            fetch_delay: Polite delay in seconds before synchronous HTTP requests.
        """
        scraping_cfg = config.hyperparams.scraping
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": str(scraping_cfg.user_agent),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.history_file = history_file or str(config.paths.cache.scraped_history)
        self.min_content_length = min_content_length
        self.fetch_delay = (
            float(fetch_delay)
            if fetch_delay is not None
            else float(scraping_cfg.fetch_delay)
        )
        self.request_timeout = int(scraping_cfg.request_timeout)
        self.max_concurrent = int(scraping_cfg.max_concurrent)

        self.history_df = self._load_history()
        self._cleanup_old_history()
        self._known_hashes: set[str] = set(
            self.history_df["doc_hash"].dropna().astype(str)
        )
        self._known_urls: set[str] = set(self.history_df["url"].dropna().astype(str))
        self._new_history_rows: list[dict[str, Any]] = []

    def _load_history(self) -> pd.DataFrame:
        """Load historical deduplication records from Parquet."""
        if os.path.exists(self.history_file):
            try:
                df = pd.read_parquet(self.history_file)
                df["scraped_at"] = pd.to_datetime(df["scraped_at"], utc=True)
                return df
            except Exception as e:
                logger.error(
                    f"Error reading Parquet file: {e}. Creating new DataFrame."
                )

        return pd.DataFrame(columns=self.HISTORY_COLUMNS)

    def _cleanup_old_history(self) -> None:
        """Prune history records older than :attr:`HISTORY_RETENTION_HOURS` in
        memory.
        """
        if self.history_df.empty:
            return

        cutoff_time = datetime.now(UTC) - timedelta(hours=self.HISTORY_RETENTION_HOURS)
        initial_len = len(self.history_df)
        self.history_df = self.history_df[
            self.history_df["scraped_at"] >= cutoff_time
        ].reset_index(drop=True)

        if len(self.history_df) < initial_len:
            logger.info("Successfully cleaned up old historical duplicates from cache.")

    def is_duplicate(self, doc_hash: str, url: str) -> bool:
        """Check in O(1) time whether ``doc_hash`` or ``url`` was already scraped."""
        is_hash_dup = bool(doc_hash and doc_hash in self._known_hashes)
        is_url_dup = bool(url and url in self._known_urls)
        return is_hash_dup or is_url_dup

    def mark_as_scraped(self, doc_hash: str, url: str) -> None:
        """Record a newly scraped item in the in-memory deduplication sets and
        buffer.
        """
        if doc_hash:
            self._known_hashes.add(doc_hash)
        if url:
            self._known_urls.add(url)
        self._new_history_rows.append(
            {"doc_hash": doc_hash, "url": url, "scraped_at": datetime.now(UTC)}
        )

    def _flush_history_buffer(self) -> None:
        """Merge buffered history records into :attr:`history_df` in a single batch."""
        if not self._new_history_rows:
            return
        new_df = pd.DataFrame(self._new_history_rows, columns=self.HISTORY_COLUMNS)
        self.history_df = (
            new_df
            if self.history_df.empty
            else pd.concat([self.history_df, new_df], ignore_index=True)
        )
        self._new_history_rows.clear()

    def save_history(self) -> None:
        """Flush buffered items and persist :attr:`history_df` to Parquet."""
        self._flush_history_buffer()
        try:
            parent = os.path.dirname(self.history_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self.history_df.to_parquet(self.history_file, index=False)
            logger.info(f"Successfully saved history cache to {self.history_file}.")
        except Exception as e:
            logger.error(f"Error saving historical database to Parquet: {e}")

    def fetch_html(self, url: str) -> str | None:
        """Fetch URL text synchronously with polite rate limiting.

        Args:
            url: Target HTTP/HTTPS URL.

        Returns:
            Response body text, or ``None`` if rate-limited/blocked or on network error.
        """
        if self.fetch_delay > 0:
            time.sleep(self.fetch_delay)

        try:
            response = self.session.get(url, timeout=self.request_timeout)
            if response.status_code in (403, 429):
                logger.warning(
                    f"Access blocked for {url} (Status: {response.status_code}). "
                    "Pausing scraper."
                )
                return None

            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while fetching {url}: {e}")
            return None

    async def fetch_urls_concurrently(
        self, urls: Sequence[str], max_concurrent: int | None = None
    ) -> list[tuple[str, str]]:
        """Fetch multiple URLs concurrently using ``aiohttp`` and a semaphore.

        Args:
            urls: Sequence of URLs to fetch.
            max_concurrent: Maximum simultaneous HTTP requests.

        Returns:
            List of ``(url, response_text)`` tuples (empty string on failure).
        """
        limit = (
            int(max_concurrent) if max_concurrent is not None else self.max_concurrent
        )
        semaphore = asyncio.Semaphore(limit)
        async_headers = dict(self.session.headers)

        async with aiohttp.ClientSession(headers=async_headers) as async_session:

            async def fetch(url: str) -> tuple[str, str]:
                async with semaphore:
                    try:
                        async with async_session.get(
                            url, timeout=self.request_timeout
                        ) as response:
                            if response.status in (403, 429):
                                logger.warning(
                                    f"Async access blocked for {url}. "
                                    f"(Status: {response.status}.)"
                                )
                                return url, ""

                            response.raise_for_status()
                            html = await response.text()
                            return url, html
                    except Exception as e:
                        logger.error(f"Async network error while fetching {url}: {e}")
                        return url, ""

            tasks = [fetch(url) for url in urls]
            return await asyncio.gather(*tasks)

    def process_text_item(
        self,
        source: str,
        url: str,
        raw_title: str,
        raw_date: str,
        raw_text: str,
    ) -> dict[str, Any] | None:
        """Clean, validate, deduplicate, and standardize a plain-text item.

        Suitable directly for non-HTML sources (e.g., social media posts, API feeds) and
        used as the second stage of :meth:`process_article`.

        Args:
            source: Source identifier (e.g., ``"BBC"``, ``"Twitter"``).
            url: Canonical item URL or URI.
            raw_title: Item title or headline.
            raw_date: Raw publication timestamp string.
            raw_text: Plain text content.

        Returns:
            Standardized record dictionary conforming to :attr:`ARTICLE_COLUMNS`, or
            ``None`` if invalid or duplicate.
        """
        content = normalize_plain_text(raw_text)
        if not is_valid_article(content, min_length=self.min_content_length):
            logger.debug(f"Item too short, discarding: {url}")
            return None

        doc_hash = generate_doc_hash(content)
        if not doc_hash or self.is_duplicate(doc_hash, url):
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

    def process_article(
        self,
        source: str,
        url: str,
        raw_title: str,
        raw_date: str,
        raw_html: str,
    ) -> dict[str, Any] | None:
        """Extract article body from raw HTML and delegate to :meth:`process_text_item`.

        Args:
            source: Publisher name (e.g., ``"AP News"``).
            url: Article URL.
            raw_title: Article headline.
            raw_date: Raw publication timestamp string.
            raw_html: Full HTML page markup.

        Returns:
            Standardized article dictionary, or ``None`` if invalid or duplicate.
        """
        content = clean_html_content(raw_html)
        if not content:
            logger.debug(f"Empty HTML extraction, discarding: {url}")
            return None
        return self.process_text_item(
            source=source,
            url=url,
            raw_title=raw_title,
            raw_date=raw_date,
            raw_text=content,
        )

    def _run_rss_pipeline(
        self,
        rss_feeds: Sequence[str],
        source_name: str,
        skip_url: Callable[[str], bool] | None = None,
        max_concurrent: int | None = None,
    ) -> pd.DataFrame:
        """Fetch RSS feeds, filter unseen article links, and scrape full HTML
        concurrently.

        Args:
            rss_feeds: List of RSS XML feed URLs.
            source_name: Publisher name stored in the ``source`` column.
            skip_url: Optional predicate returning ``True`` for URLs to ignore.
            max_concurrent: Maximum concurrent article HTML downloads.

        Returns:
            DataFrame of newly scraped articles with :attr:`ARTICLE_COLUMNS`.
        """
        pending_articles: dict[str, dict[str, str]] = {}

        for rss_url in rss_feeds:
            logger.info(f"Fetching RSS feed: {rss_url}")
            rss_xml = self.fetch_html(rss_url)
            if not rss_xml:
                continue

            try:
                root = ET.fromstring(rss_xml)
                for item in root.findall(".//item"):
                    raw_title = item.findtext("title")
                    article_url = item.findtext("link")
                    raw_date = item.findtext("pubDate")

                    if not raw_title or not article_url or not raw_date:
                        continue

                    raw_title = raw_title.strip()
                    article_url = article_url.strip()
                    raw_date = raw_date.strip()

                    if skip_url and skip_url(article_url):
                        logger.debug(f"Skipped filtered URL: {article_url}")
                        continue

                    if self.is_duplicate(doc_hash="", url=article_url):
                        logger.debug(
                            f"Article already exists in DB (URL): {article_url}"
                        )
                        continue

                    pending_articles[article_url] = {
                        "title": raw_title,
                        "date": raw_date,
                    }
            except ET.ParseError as e:
                logger.error(f"XML parsing error for {rss_url}: {e}")

        if not pending_articles:
            logger.info(f"No new {source_name} articles to fetch.")
            return pd.DataFrame(columns=self.ARTICLE_COLUMNS)

        urls_to_fetch = list(pending_articles.keys())
        logger.info(
            f"Starting concurrent fetching for {len(urls_to_fetch)} "
            f"{source_name} URLs..."
        )
        html_results = asyncio.run(
            self.fetch_urls_concurrently(urls_to_fetch, max_concurrent=max_concurrent)
        )

        articles_data: list[dict[str, Any]] = []
        for article_url, article_html in html_results:
            if not article_html:
                continue
            meta = pending_articles[article_url]
            processed = self.process_article(
                source=source_name,
                url=article_url,
                raw_title=meta["title"],
                raw_date=meta["date"],
                raw_html=article_html,
            )
            if processed:
                articles_data.append(processed)

        self.save_history()
        logger.info(
            f"{self.__class__.__name__} finished. "
            f"Fetched {len(articles_data)} new articles."
        )

        if not articles_data:
            return pd.DataFrame(columns=self.ARTICLE_COLUMNS)

        return pd.DataFrame(articles_data)[self.ARTICLE_COLUMNS]

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Execute the scraper and return newly collected items with
        :attr:`ARTICLE_COLUMNS`.
        """
