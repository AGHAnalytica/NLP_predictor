"""
End-to-end pipeline::

    Grupa 1 (dummy articles) -> Grupa 2 (topic extraction)  -> Grupa 3 (upper path)
                                                            -> Grupa 4 (lower path)
    Grupa 3 + Grupa 4 -> final signal (UP / DOWN / NEUTRAL + confidence)

Grupa 1 (scraping & preprocessing) is mocked here with the sample article table; the
remaining stages run for real. The final stage fuses both paths into one actionable
per-instrument trading signal.
"""

import pandas as pd
from dotenv import load_dotenv

from nlp_predictor.base_analyzer import BaseAnalyzer
from nlp_predictor.config import Config
from nlp_predictor.global_trends import LowerPathAnalyzer
from nlp_predictor.instrument_trends import UpperPathAnalyzer
from nlp_predictor.scraper_runner import run_real_scrapers
from nlp_predictor.signal_generator import SignalGenerator
from nlp_predictor.topic_extractor import TopicExtractor

# Load environment variables (OPENAI_API_KEY) from the project-root .env
load_dotenv(override=True)


def _enrich_topics_for_upper_path(topics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the columns the upper path's DCIM severity model needs.

    The upper path scores each instrument using ``source_name`` and ``source_weight``
    (credibility of the publisher) and ``date`` (recency). Topics only carry
    ``doc_hash``, so we join back to the articles for the source, then to the source
    weights table, exactly as the upper-path module test does.

    Args:
        topics_df (pd.DataFrame): The DataFrame containing extracted topics with at
        least a 'doc_hash' column.
    Returns:
        pd.DataFrame: An enriched DataFrame with the necessary columns for the
        upper path.
    """
    articles_df = pd.read_csv(Config["ARTICLES_PATH"])[["doc_hash", "source_name"]]
    weights_df = pd.read_csv(Config["WEIGHTS_PATH"])[["source_name", "source_weight"]]

    enriched = topics_df.merge(articles_df, on="doc_hash", how="left")
    enriched = enriched.merge(weights_df, on="source_name", how="left")

    # Unknown sources default to full weight so they are not silently dropped.
    enriched["source_weight"] = enriched["source_weight"].fillna(1.0)
    # The DCIM time-decay needs a timezone-aware datetime.
    enriched["date"] = pd.to_datetime(enriched["date"], utc=True)
    return enriched


def main() -> None:
    """
    Run the real articles -> topics -> upper/lower trends pipeline.
    Each stage saves its output to disk, both as a checkpoint and for later analysis.
    The final signal generation stage fuses the upper and lower path outputs into a
    single actionable signal per instrument, which is also saved to disk.
    Logs key steps and outputs summary statistics at each stage for monitoring.
    If any critical stage fails (e.g. no articles scraped, no topics extracted), the
    pipeline aborts with an error message to prevent downstream errors.

    Returns:
        None
    """
    # ------------------------------------------------------------------ Grupa 1
    print("====== [1/5] Scraping real articles (Grupa 1) ======")


    articles_df = run_real_scrapers()

    if articles_df.empty:
        print("[FATAL] No articles scraped today; cannot run the pipeline.")
        return

    if "source" in articles_df.columns:
        articles_df.rename(columns={"source": "source_name"}, inplace=True)

    print(f"[INFO] Scraped {len(articles_df)} real articles.")

    articles_df.to_csv(Config["ARTICLES_PATH"], index=False)

    # ------------------------------------------------------------------ Grupa 2
    print("\n====== [2/5] Extracting topics (Grupa 2) ======")
    extractor = TopicExtractor()
    topics_df = extractor.extract_topics(articles_df)
    topics_df.to_csv(Config["TOPICS_SAVE_PATH_CSV"], index=False)
    print(f"[INFO] Saved {len(topics_df)} topics to {Config['TOPICS_SAVE_PATH_CSV']}")

    if topics_df.empty:
        print("[FATAL] No topics extracted; cannot run the trend paths.")
        return

    # ------------------------------------------------------------- Grupa 3 & 4
    # Both paths receive the SAME topics table and run independently.
    print("\n====== [3/5] Upper path - instrument trends (Grupa 3) ======")
    upper = UpperPathAnalyzer(Config["INSTRUMENT_PATH"])
    instrument_trends = upper.process_data(_enrich_topics_for_upper_path(topics_df))
    UpperPathAnalyzer.save_trends(
        instrument_trends, Config["INSTRUMENT_TRENDS_SAVE_PATH_PARQUET"]
    )
    instrument_trends.to_csv(Config["INSTRUMENT_TRENDS_SAVE_PATH_CSV"], index=False)
    print(f"[INFO] Generated {len(instrument_trends)} instrument trends.")

    print("\n====== [4/5] Lower path - global trends (Grupa 4) ======")
    lower = LowerPathAnalyzer(Config["INSTRUMENT_PATH"])
    global_trends = lower.process_data(topics_df.copy())
    LowerPathAnalyzer.save_trends(
        global_trends, Config["GLOBAL_TRENDS_SAVE_PATH_PARQUET"]
    )
    global_trends.to_csv(Config["GLOBAL_TRENDS_SAVE_PATH_CSV"], index=False)
    print(f"[INFO] Generated {len(global_trends)} global trends.")

    # ------------------------------------------------------- Final signal layer
    # Fuse the two paths: instrument severity sets direction, macro narrative
    # modulates confidence. Produces the system's actionable output.
    print("\n====== [5/5] Final signal generation ======")
    signals = SignalGenerator().generate(instrument_trends, global_trends)
    BaseAnalyzer.save_trends(signals, Config["SIGNALS_SAVE_PATH_PARQUET"])
    signals.to_csv(Config["SIGNALS_SAVE_PATH_CSV"], index=False)

    if not signals.empty:
        actionable = signals[signals["signal"] != "NEUTRAL"]
        print(
            f"[INFO] {len(actionable)} actionable signal(s) "
            f"out of {len(signals)} instruments. Top calls:"
        )
        preview_cols = ["instrument", "signal", "confidence", "macro_context"]
        print(signals[preview_cols].head(10).to_string(index=False))

    print("\n====== Pipeline complete ======")


if __name__ == "__main__":
    main()
