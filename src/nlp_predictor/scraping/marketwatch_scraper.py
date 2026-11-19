"""MarketWatch feed-only RSS scraper for US market news."""

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.scraping.base_scraper import BaseScraper


class MarketWatchScraper(BaseScraper):
    """Scraper for MarketWatch (Dow Jones) RSS feeds using headline and lead only.

    Article pages are bot-protected (HTTP 401), so content is built from the RSS
    title and ``description`` without fetching the article HTML.
    """

    def __init__(self, history_file: str | None = None) -> None:
        """Initialize MarketWatch RSS feed URLs and a short-text length threshold."""
        super().__init__(
            history_file=history_file,
            min_content_length=config.hyperparams.scraping.feed_only_min_length,
        )
        self.rss_feeds = list(config.hyperparams.scraping.marketwatch_rss_feeds)

    def run(self) -> pd.DataFrame:
        """Fetch MarketWatch RSS feeds and build articles from feed text only."""
        return self._run_rss_pipeline(
            rss_feeds=self.rss_feeds,
            source_name="MarketWatch",
            canonicalize_url=self.strip_query_string,
            fetch_full_article=False,
        )
