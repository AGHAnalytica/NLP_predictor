"""Application entry point executing one 15-minute scrape, ingest, and evaluation
tick.
"""

import logging

import pandas as pd
from dotenv import load_dotenv

from nlp_predictor.config import config
from nlp_predictor.core.logging_config import setup_logging
from nlp_predictor.lower_path.lower_path_analyzer import LowerPathAnalyzer
from nlp_predictor.lower_path.macro_overlay import MacroOverlayAnalyzer
from nlp_predictor.preprocessing.feature_ingestor import FeatureIngestor
from nlp_predictor.preprocessing.feature_store import FeatureStore
from nlp_predictor.preprocessing.topic_extractor import TopicExtractor
from nlp_predictor.scraping.scrape_runner import run_real_scrapers
from nlp_predictor.signal.signal_generator import SignalGenerator
from nlp_predictor.signal.signal_log import SignalLog
from nlp_predictor.tick.tick_runner import TickRunner
from nlp_predictor.upper_path.upper_path_analyzer import UpperPathAnalyzer

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def main() -> None:
    """Execute one complete pipeline cycle: scrape articles, cache features, and log
    signals.
    """
    if not logging.getLogger().hasHandlers():
        setup_logging()

    logger.info("====== [1/3] Scraping real articles (Group 1) ======")
    articles_df = run_real_scrapers()
    if articles_df.empty:
        logger.info("No new articles scraped; running the tick on cached features.")
    else:
        if "source" in articles_df.columns:
            articles_df = articles_df.rename(columns={"source": "source_name"})
        logger.info(f"Scraped {len(articles_df)} article(s).")
        articles_df.to_csv(config.paths.input.articles, index=False)

    store = FeatureStore()
    upper = UpperPathAnalyzer(str(config.paths.input.instruments))
    lower = LowerPathAnalyzer(str(config.paths.input.instruments))

    logger.info("====== [2/3] Ingesting new article features (Group 2 -> cache) ======")
    ingestor = FeatureIngestor(
        store=store, extractor=TopicExtractor(), upper=upper, lower=lower
    )
    ingestor.ingest(articles_df if not articles_df.empty else pd.DataFrame())

    logger.info("====== [3/3] Scoring variants -> signals (Group 3/4 + fusion) ======")
    runner = TickRunner(
        store=store,
        upper=upper,
        lower=lower,
        macro=MacroOverlayAnalyzer(str(config.paths.input.instruments)),
        signal=SignalGenerator(),
        log=SignalLog(),
    )
    signals = runner.run()

    if signals.empty:
        logger.info(
            "Tick produced no signals (feature store window empty). "
            "This is expected until a few ticks have populated the store."
        )
    else:
        actionable = signals[signals["signal"] != "NEUTRAL"]
        cols = ["variant", "instrument", "signal", "confidence", "macro_context"]
        top_calls = (
            signals.sort_values("confidence", ascending=False)[cols]
            .head(10)
            .to_string(index=False)
        )
        logger.info(
            f"{len(actionable)} actionable of {len(signals)} signal(s) "
            f"across {signals['variant'].nunique()} variant(s). Top calls:\n{top_calls}"
        )

    logger.info("====== Tick complete ======")


if __name__ == "__main__":
    setup_logging()
    main()
