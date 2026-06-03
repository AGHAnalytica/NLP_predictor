import logging

import pandas as pd

from nlp_predictor.apnews import APNewsScraper
from nlp_predictor.bbcb import BBCScraper
from nlp_predictor.cnbc import CNBCScraper

logger = logging.getLogger(__name__)


def run_real_scrapers() -> pd.DataFrame:
    """
    Executes all scrapers, collects the data, and concatenates it into a single
    DataFrame. Each scraper is responsible for its own deduplication and media
    filtering, so the final concatenation simply combines the cleaned results.
    If no new articles are found, an empty DataFrame is returned.

    The returned DataFrame has the following columns:
    - doc_hash: A unique hash of the article content for deduplication.
    - source: The name of the news source (e.g., "AP News", "BBC", "CNBC").
    - url: The URL of the article.
    - date: The publication date of the article, normalized to UTC.
    - title: The cleaned title of the article.
    - content: The cleaned main text content of the article.

    Returns:
        pd.DataFrame: A DataFrame containing all new articles from the scrapers,
        or an empty DataFrame with the correct columns if no new articles are found.
    """
    logger.info("Starting real article extraction (Group 1)...")

    scrapers = [CNBCScraper(), APNewsScraper(), BBCScraper()]
    collected_dfs = []

    for scraper in scrapers:
        try:
            df = scraper.run()
            if not df.empty:
                collected_dfs.append(df)
        except Exception as e:
            logger.error(f"Critical error in {scraper.__class__.__name__}: {e}")

    if collected_dfs:
        final_df = pd.concat(collected_dfs, ignore_index=True)
        return final_df

    return pd.DataFrame(
        columns=["doc_hash", "source", "url", "date", "title", "content"]
    )
