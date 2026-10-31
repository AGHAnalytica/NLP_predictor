"""Orchestrator executing registered news scrapers and concatenating results."""

import logging
from collections.abc import Sequence

import pandas as pd

from nlp_predictor.scraping.apnews_scraper import APNewsScraper
from nlp_predictor.scraping.base_scraper import BaseScraper
from nlp_predictor.scraping.bbc_scraper import BBCScraper
from nlp_predictor.scraping.cnbc_scraper import CNBCScraper

logger = logging.getLogger(__name__)


def run_real_scrapers(scrapers: Sequence[BaseScraper] | None = None) -> pd.DataFrame:
    """Run all configured scrapers and concatenate newly discovered items.

    Args:
        scrapers: Optional custom sequence of :class:`BaseScraper` instances (e.g., web,
            RSS, or social media scrapers). Defaults to CNBC, AP News, and BBC scrapers.

    Returns:
        Combined DataFrame conforming to :attr:`BaseScraper.ARTICLE_COLUMNS`.
    """
    logger.info("Starting real article extraction (Group 1)...")

    active_scrapers = (
        list(scrapers)
        if scrapers is not None
        else [CNBCScraper(), APNewsScraper(), BBCScraper()]
    )
    collected_dfs: list[pd.DataFrame] = []

    for scraper in active_scrapers:
        try:
            df = scraper.run()
            if not df.empty:
                collected_dfs.append(df)
        except Exception as e:
            logger.error(f"Critical error in {scraper.__class__.__name__}: {e}")

    if collected_dfs:
        return pd.concat(collected_dfs, ignore_index=True)

    return pd.DataFrame(columns=BaseScraper.ARTICLE_COLUMNS)
