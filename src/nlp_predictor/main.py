"""
Application entry point — ONE 15-minute tick of the live A/B/n loop.

This is the process Azure Container Apps invokes on its schedule (every ~15 min): each
call is a single, self-contained tick, so there is no in-process scheduler here — the
container runtime owns the cadence.

A tick has two halves, split exactly along the feature store:

    WRITE (ingestion)   scrape new articles -> compute their features once -> cache them
                        (:class:`ingestion.FeatureIngestor`; Group 1 + Group 2)
    READ  (tick)        score every severity variant on the cached window, fuse with one
                        shared macro climate, append each signal to the eval log
                        (:class:`tick.TickRunner`; Group 3 + Group 4 + fusion)

Because the expensive, stochastic work (extraction, embeddings, topic->instrument
mapping) is cached on first sight of an article and reused for every later tick it stays
in-window, a steady-state tick mostly does cheap, deterministic severity recomputation
plus one shared macro LLM call. The durable output is the append-only signal log
(``config.paths.logs.signal_log``) — the dataset each variant's demo-account PnL is
scored against.
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

# Load environment variables (OPENAI_API_KEY) from the project-root .env
load_dotenv(override=True)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run exactly one tick: scrape, ingest new article features, emit per-variant signals.

    Steps:
      1. Scrape fresh articles (Group 1). An empty scrape is fine — the tick still runs
         on whatever is already cached and in-window.
      2. Ingest: cache features for genuinely new articles only (Group 1->2, step 4.5).
      3. Tick: cached features -> per-variant signals -> appended to the eval log.

    The analyzers and the feature store are built once and SHARED between the ingest
    (write) and tick (read) halves, so the instrument table and OpenAI client are loaded
    a single time per tick.

    Returns:
        None
    """
    if not logging.getLogger().hasHandlers():
        setup_logging()

    logger.info("====== [1/3] Scraping real articles (Group 1) ======")
    articles_df = run_real_scrapers()
    if articles_df.empty:
        logger.info("No new articles scraped; running the tick on cached features.")
    else:
        if "source" in articles_df.columns:
            articles_df.rename(columns={"source": "source_name"}, inplace=True)
        logger.info(f"Scraped {len(articles_df)} article(s).")
        # Persist the latest scrape for audit/debugging (not the cache of record).
        articles_df.to_csv(config.paths.input.articles, index=False)

    # One shared stack for both halves of the tick.
    store = FeatureStore()
    upper = UpperPathAnalyzer(str(config.paths.input.instruments))
    lower = LowerPathAnalyzer(str(config.paths.input.instruments))

    # ------------------------------------------------------------- WRITE half
    logger.info("====== [2/3] Ingesting new article features (Group 2 -> cache) ======")
    ingestor = FeatureIngestor(
        store=store, extractor=TopicExtractor(), upper=upper, lower=lower
    )
    ingestor.ingest(articles_df if not articles_df.empty else pd.DataFrame())

    # -------------------------------------------------------------- READ half
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
