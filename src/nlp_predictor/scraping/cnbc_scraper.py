"""CNBC RSS scraper filtering out paywalled CNBC Pro articles."""

import pandas as pd

from nlp_predictor.scraping.base_scraper import BaseScraper


class CNBCScraper(BaseScraper):
    """Scraper for CNBC RSS feeds, excluding paywalled CNBC Pro links."""

    def __init__(self, history_file: str | None = None) -> None:
        """Initialize CNBC RSS feed URLs and base scraper state."""
        super().__init__(history_file=history_file)
        self.rss_feeds = [
            "https://www.cnbc.com/id/19836768/device/rss/rss.html",
            "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        ]

    @staticmethod
    def _is_pro_url(url: str) -> bool:
        """Return ``True`` if the URL belongs to paywalled CNBC Pro content."""
        return "cnbc.com/pro/" in url

    def run(self) -> pd.DataFrame:
        """Fetch CNBC RSS feeds and scrape new non-Pro articles concurrently."""
        return self._run_rss_pipeline(
            rss_feeds=self.rss_feeds,
            source_name="CNBC",
            skip_url=self._is_pro_url,
        )
