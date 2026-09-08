"""
Ad-hoc end-to-end test on the larger article corpus (articles_sample_large.csv).

Runs the full pipeline (topics -> upper path -> lower path -> signals) and writes
outputs to ``*_large.csv`` so the committed demo reports stay untouched. Prints the
macro trends, instrument trends and the final signal table with rationales so the
two-path fusion can be inspected end to end.

Usage (from the repo root):  PYTHONPATH=src python scripts/run_large_test.py
"""

import pathlib

import pandas as pd
from dotenv import load_dotenv

from nlp_predictor.config import REPO_ROOT, config
from nlp_predictor.lower_path import LowerPathAnalyzer
from nlp_predictor.preprocessing import TopicExtractor
from nlp_predictor.signal import SignalGenerator
from nlp_predictor.upper_path import UpperPathAnalyzer


def _enrich_topics_for_upper_path(
    topics_df: pd.DataFrame, articles_path: str | None = None
) -> pd.DataFrame:
    path = articles_path or str(config.paths.input.articles)
    articles_df = pd.read_csv(path)[["doc_hash", "source_name"]]
    weights_df = pd.read_csv(config.paths.input.weights)[
        ["source_name", "source_weight"]
    ]

    enriched = topics_df.merge(articles_df, on="doc_hash", how="left")
    enriched = enriched.merge(weights_df, on="source_name", how="left")
    enriched["source_weight"] = enriched["source_weight"].fillna(1.0)
    enriched["date"] = pd.to_datetime(enriched["date"], utc=True)
    return enriched


load_dotenv(override=True)


# Treat the sample batch as freshly scraped: shift timestamps so the newest article
# is ~now, preserving relative spacing. Without this the static 2026-06-01 timestamps
# get heavily time-decayed by the DCIM and single-article signals vanish.
REFRESH_DATES = True


def _out(name: str) -> str:
    return str(REPO_ROOT / "data" / "test_results" / name)


def _refresh_dates(topics: pd.DataFrame) -> pd.DataFrame:
    """Shift topic timestamps so the newest is ~now, keeping their relative spacing."""
    topics = topics.copy()
    parsed = pd.to_datetime(topics["date"], utc=True)
    offset = pd.Timestamp.now(tz="UTC") - parsed.max()
    topics["date"] = (parsed + offset).astype(str)
    return topics


def main() -> None:
    large_articles_path = str(
        REPO_ROOT / "data" / "scraping" / "articles_sample_large.csv"
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)

    articles = pd.read_csv(large_articles_path)
    print(f"====== [1/5] {len(articles)} articles ======")

    # Reuse cached topics if present so a re-run isolates downstream changes
    # (e.g. the instrument-mapping prompt) instead of re-extracting from scratch.
    topics_path = _out("topics_report_large.csv")
    if pathlib.Path(topics_path).exists():
        topics = pd.read_csv(topics_path)
        print(f"====== [2/5] reusing {len(topics)} cached topics ======")
    else:
        topics = TopicExtractor().extract_topics(articles)
        topics.to_csv(topics_path, index=False)
        print(f"====== [2/5] {len(topics)} topics ======")

    if REFRESH_DATES:
        topics = _refresh_dates(topics)
        print("[INFO] Refreshed topic timestamps so the batch looks freshly scraped.")

    inst_path = str(config.paths.input.instruments)
    enriched = _enrich_topics_for_upper_path(topics, articles_path=large_articles_path)
    instruments = UpperPathAnalyzer(inst_path).process_data(enriched)
    instruments.to_csv(_out("instrument_trends_report_large.csv"), index=False)
    print(f"====== [3/5] {len(instruments)} instrument trends ======")

    macro = LowerPathAnalyzer(inst_path).process_data(topics.copy())
    macro.to_csv(_out("global_trends_report_large.csv"), index=False)
    print(f"====== [4/5] {len(macro)} macro trends ======")

    signals = SignalGenerator().generate(instruments, macro)
    signals.to_csv(_out("signals_report_large.csv"), index=False)
    print(f"====== [5/5] {len(signals)} signals ======\n")

    print("===== MACRO TRENDS (lower path) =====")
    if macro.empty:
        print("(no clusters formed)")
    else:
        print(
            macro[["cluster_id", "title", "direction", "severity"]].to_string(
                index=False
            )
        )

    print("\n===== INSTRUMENT TRENDS (upper path) =====")
    print(instruments[["instrument", "direction", "severity"]].to_string(index=False))

    print("\n===== FINAL SIGNALS =====")
    print(
        signals[
            ["instrument", "signal", "confidence", "macro_context", "n_topics"]
        ].to_string(index=False)
    )
    print("\n----- rationales -----")
    for _, row in signals.iterrows():
        print(f"- {row['instrument']}: {row['rationale']}")


if __name__ == "__main__":
    main()
