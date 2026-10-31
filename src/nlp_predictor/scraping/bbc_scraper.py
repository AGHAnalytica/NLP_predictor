"""BBC News RSS scraper for Business and Science/Environment feeds."""

import pandas as pd

from nlp_predictor.scraping.base_scraper import BaseScraper


class BBCScraper(BaseScraper):
    """Scraper for BBC News RSS feeds, excluding audio/video endpoints."""

    def __init__(self, history_file: str | None = None) -> None:
        """Initialize BBC RSS feed URLs and base scraper state."""
        super().__init__(history_file=history_file)
        self.rss_feeds = [
            "http://feeds.bbci.co.uk/news/business/rss.xml",
            "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        ]

    @staticmethod
    def _is_media_url(url: str) -> bool:
        """Return ``True`` if the BBC URL points to audio/video content."""
        return "/av/" in url or "bbc.co.uk/sounds" in url

    def run(self) -> pd.DataFrame:
        """Fetch BBC RSS feeds and scrape new articles concurrently."""
        return self._run_rss_pipeline(
            rss_feeds=self.rss_feeds,
            source_name="BBC",
            skip_url=self._is_media_url,
        )
