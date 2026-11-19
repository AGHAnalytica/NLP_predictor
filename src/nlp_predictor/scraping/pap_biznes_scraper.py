"""PAP Biznes feed-only RSS scraper for Polish GPW company and market news."""

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.scraping.base_scraper import BaseScraper


class PapBiznesScraper(BaseScraper):
    """Scraper for the PAP Biznes RSS feed (GPW company reports and market news).

    Article pages are blocked by Incapsula bot protection, so content is built
    from the RSS feed only. The feed ``description`` carries no lead (only the
    title, author and date), so items are effectively headlines.
    """

    def __init__(self, history_file: str | None = None) -> None:
        """Initialize PAP Biznes RSS feed URLs and a short-text length threshold."""
        super().__init__(
            history_file=history_file,
            min_content_length=config.hyperparams.scraping.feed_only_min_length,
        )
        self.rss_feeds = list(config.hyperparams.scraping.pap_biznes_rss_feeds)

    def run(self) -> pd.DataFrame:
        """Fetch PAP Biznes RSS feeds and build articles from feed text only."""
        return self._run_rss_pipeline(
            rss_feeds=self.rss_feeds,
            source_name="PAP Biznes",
            fetch_full_article=False,
        )
