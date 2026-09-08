"""
Tests for FeatureIngestor — the WRITE half of the feature store loop (step 4.5).

The three LLM-bound steps (topic extraction, embeddings, topic->instrument mapping) are
faked, so these exercise the real orchestration: article-level dedup (the cost saver),
enrichment, and that genuinely cached per-topic rows land in a real FeatureStore.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd

from nlp_predictor.preprocessing.feature_ingestor import FeatureIngestor
from nlp_predictor.preprocessing.feature_store import FeatureStore

NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)


class _FakeExtractor:
    """One topic per article; records which doc_hashes it was asked to process."""

    def __init__(self):
        self.seen_doc_hashes: list[str] = []

    def extract_topics(self, articles_df):
        self.seen_doc_hashes = list(articles_df["doc_hash"].astype(str))
        rows = [
            {
                "topic_id": f"top-{a['doc_hash']}",
                "doc_hash": a["doc_hash"],
                "topic_name": "AI capex",
                "topic_description": "desc",
                "date": a["date"],
            }
            for _, a in articles_df.iterrows()
        ]
        return pd.DataFrame(rows)


class _EmptyExtractor:
    def extract_topics(self, articles_df):
        return pd.DataFrame()


class _FakeUpper:
    def _parallel_map(self, fn, items):
        return [fn(x) for x in items]

    def _map_topic_to_instrument(self, row):
        return [{"instrument": "AAPL", "sentiment": 0.5}]


class _FakeLower:
    def _generate_embeddings_batch(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _articles(n=2, start=0):
    rows = [
        {
            "doc_hash": f"h{start + i}",
            "source_name": "Bloomberg",
            "date": NOW - timedelta(hours=i + 1),
            "title": "t",
            "content": "c",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def _ingestor(tmp_path, store=None, extractor=None):
    weights = tmp_path / "weights.csv"
    pd.DataFrame([{"source_name": "Bloomberg", "source_weight": 0.9}]).to_csv(
        weights, index=False
    )
    store = store or FeatureStore(
        path=str(tmp_path / "fs" / "feat.parquet"), max_window_hours=36.0
    )
    return FeatureIngestor(
        store=store,
        extractor=extractor or _FakeExtractor(),
        upper=_FakeUpper(),
        lower=_FakeLower(),
        weights_path=str(weights),
    ), store


def test_ingest_caches_new_article_features(tmp_path):
    ing, store = _ingestor(tmp_path)
    n = ing.ingest(_articles(2), now=NOW)

    assert n == 2
    df = store.load()
    assert set(df["doc_hash"]) == {"h0", "h1"}
    # Both cached features are present and correctly typed.
    assert isinstance(df.iloc[0]["embedding"], list)
    assert df.iloc[0]["impacts"] == [{"instrument": "AAPL", "sentiment": 0.5}]
    # Enrichment attached publisher credibility from the weights table.
    assert df.iloc[0]["source_weight"] == 0.9
    # The cached impacts explode into the severity-model frame.
    inst = FeatureStore.to_instrument_frame(store.window(36.0, NOW))
    assert set(inst["instrument"]) == {"AAPL"}


def test_already_cached_articles_are_skipped(tmp_path):
    ing, store = _ingestor(tmp_path)
    ing.ingest(_articles(2), now=NOW)

    # Second pass over the SAME articles must do no work and add nothing.
    ext = _FakeExtractor()
    ing2, _ = _ingestor(tmp_path, store=store, extractor=ext)
    assert ing2.ingest(_articles(2), now=NOW) == 0
    assert ext.seen_doc_hashes == []  # dedup happens BEFORE extraction
    assert len(store.load()) == 2  # unchanged


def test_only_new_articles_are_extracted(tmp_path):
    ing, store = _ingestor(tmp_path)
    ing.ingest(_articles(2), now=NOW)  # caches h0, h1

    ext = _FakeExtractor()
    ing2, _ = _ingestor(tmp_path, store=store, extractor=ext)
    mixed = pd.concat([_articles(1, start=1), _articles(1, start=2)])  # h1 (known) + h2
    n = ing2.ingest(mixed, now=NOW)

    assert n == 1
    assert ext.seen_doc_hashes == ["h2"]  # only the genuinely new one
    assert set(store.load()["doc_hash"]) == {"h0", "h1", "h2"}


def test_source_column_is_normalized(tmp_path):
    ing, store = _ingestor(tmp_path)
    arts = _articles(1).rename(columns={"source_name": "source"})  # scraper's name
    assert ing.ingest(arts, now=NOW) == 1
    assert store.load().iloc[0]["source_name"] == "Bloomberg"


def test_empty_or_topicless_inputs_are_noops(tmp_path):
    ing, store = _ingestor(tmp_path)
    assert ing.ingest(pd.DataFrame(), now=NOW) == 0
    assert ing.ingest(None, now=NOW) == 0

    empty_topics, store2 = _ingestor(tmp_path, extractor=_EmptyExtractor())
    assert empty_topics.ingest(_articles(2), now=NOW) == 0
    assert store2.load().empty
