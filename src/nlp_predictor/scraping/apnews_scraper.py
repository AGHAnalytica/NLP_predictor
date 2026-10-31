"""AP News hub scraper extracting article links from HTML section pages."""

import asyncio
import logging
from typing import Any
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup  # type: ignore

from nlp_predictor.scraping.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class APNewsScraper(BaseScraper):
    """Scraper for AP News Business and Climate HTML hub pages."""

    def __init__(self, history_file: str | None = None) -> None:
        """Initialize AP News hub URLs and base scraper state."""
        super().__init__(history_file=history_file)
        self.base_url = "https://apnews.com"
        self.hub_urls = [
            "https://apnews.com/hub/business",
            "https://apnews.com/hub/climate-and-environment",
        ]

    def run(self) -> pd.DataFrame:
        """Discover article links across AP hubs, fetch HTML concurrently, and clean
        articles.

        Returns:
            DataFrame of newly scraped AP News articles with :attr:`ARTICLE_COLUMNS`.
        """
        articles_data: list[dict[str, Any]] = []
        found_links: set[str] = set()

        for hub_url in self.hub_urls:
            logger.info(f"Fetching AP News hub: {hub_url}")
            hub_html = self.fetch_html(hub_url)
            if not hub_html:
                continue

            soup = BeautifulSoup(hub_html, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = str(a_tag["href"])
                if "/article/" in href:
                    found_links.add(urljoin(self.base_url, href))

        logger.info(f"Found {len(found_links)} unique article links on AP News hubs.")

        urls_to_fetch = [
            url for url in found_links if not self.is_duplicate(doc_hash="", url=url)
        ]

        if not urls_to_fetch:
            logger.info("No new AP News articles found. Skipping.")
            return pd.DataFrame(columns=self.ARTICLE_COLUMNS)

        logger.info(f"Starting concurrent fetching for {len(urls_to_fetch)} URLs...")
        html_results = asyncio.run(
            self.fetch_urls_concurrently(urls_to_fetch, max_concurrent=15)
        )

        for article_url, article_html in html_results:
            if not article_html:
                continue

            article_soup = BeautifulSoup(article_html, "html.parser")
            title_tag = article_soup.find("meta", property="og:title")
            raw_title = (
                str(title_tag["content"])
                if title_tag and title_tag.has_attr("content")
                else ""
            )

            date_tag = article_soup.find("meta", property="article:published_time")
            raw_date = (
                str(date_tag["content"])
                if date_tag and date_tag.has_attr("content")
                else ""
            )

            if not raw_title or not raw_date:
                logger.warning(
                    f"Could not extract metadata for {article_url}. Skipping."
                )
                continue

            processed = self.process_article(
                source="AP News",
                url=article_url,
                raw_title=raw_title,
                raw_date=raw_date,
                raw_html=article_html,
            )
            if processed:
                articles_data.append(processed)

        self.save_history()
        logger.info(
            f"APNewsScraper finished. Fetched {len(articles_data)} new articles."
        )

        if not articles_data:
            return pd.DataFrame(columns=self.ARTICLE_COLUMNS)

        return pd.DataFrame(articles_data)[self.ARTICLE_COLUMNS]
