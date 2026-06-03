import logging
import pandas as pd

from nlp_predictor.cnbc import CNBCScraper
from nlp_predictor.apnews import APNewsScraper
from nlp_predictor.bbcb import BBCScraper

logger = logging.getLogger(__name__)


def run_real_scrapers() -> pd.DataFrame:
    """
    Executes all scrapers, collects the data, and concatenates it into a single DataFrame.
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


    return pd.DataFrame(columns=["doc_hash", "source", "url", "date", "title", "content"])