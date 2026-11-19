"""Bankier.pl RSS scraper for Polish market and GPW news."""

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.scraping.base_scraper import BaseScraper


class BankierScraper(BaseScraper):
    """Scraper for Bankier.pl news and stock market (Giełda) RSS feeds."""

    def __init__(self, history_file: str | None = None) -> None:
        """Initialize Bankier.pl RSS feed URLs and base scraper state."""
        super().__init__(history_file=history_file)
        self.rss_feeds = list(config.hyperparams.scraping.bankier_rss_feeds)

    def run(self) -> pd.DataFrame:
        """Fetch Bankier.pl RSS feeds and scrape new full articles concurrently."""
        return self._run_rss_pipeline(
            rss_feeds=self.rss_feeds,
            source_name="Bankier.pl",
            canonicalize_url=self.strip_query_string,
        )
