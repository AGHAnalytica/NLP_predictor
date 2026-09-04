"""Unit tests for the rolling FeatureStore (persistence layer of the A/B loop)."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from nlp_predictor.preprocessing.feature_store import FeatureStore

NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)


def _store(tmp_path, max_window_hours=24.0):
    return FeatureStore(
        path=str(tmp_path / "topic_features.parquet"),
        max_window_hours=max_window_hours,
    )


def _record(doc_hash, topic_id, hours_ago, *, embedding, impacts, source="Bloomberg",
            weight=1.0):
    """One topic row as the orchestration layer would hand it to the store."""
    return {
        "doc_hash": doc_hash,
        "topic_id": topic_id,
        "topic_name": f"topic-{topic_id}",
        "topic_description": "desc",
        "date": NOW - timedelta(hours=hours_ago),
        "source_name": source,
        "source_weight": weight,
        "embedding": embedding,
        "impacts": impacts,
    }


def test_empty_store_reads_back_typed(tmp_path):
    store = _store(tmp_path)
    df = store.load()
    assert df.empty
    assert list(df.columns) == FeatureStore.COLUMNS
    assert store.known_doc_hashes() == set()
    assert store.window(4.0, now=NOW).empty


def test_add_load_roundtrip_preserves_objects(tmp_path):
    store = _store(tmp_path)
    rec = _record("a", "t1", 1.0,
                  embedding=[0.1, 0.2, 0.3],
                  impacts=[{"instrument": "AAPL", "sentiment": 0.5}])
    store.add(pd.DataFrame([rec]), now=NOW)

    df = store.load()
    assert len(df) == 1
    row = df.iloc[0]
    # JSON columns must come back as real Python objects, not strings.
    assert row["embedding"] == [0.1, 0.2, 0.3]
    assert row["impacts"] == [{"instrument": "AAPL", "sentiment": 0.5}]
    assert row["date"].tzinfo is not None
    assert row["cached_at"].tzinfo is not None


def test_add_is_idempotent_on_doc_hash(tmp_path):
    store = _store(tmp_path)
    rec = _record("a", "t1", 1.0, embedding=[0.1], impacts=[])
    store.add(pd.DataFrame([rec]), now=NOW)
    # Re-submitting the same article (even with a new topic_id) must not duplicate it.
    rec2 = _record("a", "t1-again", 1.0, embedding=[0.9], impacts=[])
    store.add(pd.DataFrame([rec2]), now=NOW)

    df = store.load()
    assert len(df) == 1
    assert df.iloc[0]["embedding"] == [0.1]  # original kept, not overwritten


def test_known_doc_hashes(tmp_path):
    store = _store(tmp_path)
    store.add(pd.DataFrame([
        _record("a", "t1", 1.0, embedding=[0.1], impacts=[]),
        _record("b", "t2", 1.0, embedding=[0.2], impacts=[]),
    ]), now=NOW)
    assert store.known_doc_hashes() == {"a", "b"}


def test_window_filters_by_article_date(tmp_path):
    store = _store(tmp_path)
    store.add(pd.DataFrame([
        _record("recent", "t1", 1.0, embedding=[0.1], impacts=[]),   # in 4h
        _record("edge", "t2", 3.9, embedding=[0.2], impacts=[]),     # in 4h
        _record("old", "t3", 6.0, embedding=[0.3], impacts=[]),      # outside 4h
    ]), now=NOW)

    win = store.window(4.0, now=NOW)
    assert set(win["doc_hash"]) == {"recent", "edge"}

    win_wide = store.window(8.0, now=NOW)
    assert set(win_wide["doc_hash"]) == {"recent", "edge", "old"}


def test_prune_on_add_drops_beyond_max_window(tmp_path):
    store = _store(tmp_path, max_window_hours=4.0)
    store.add(pd.DataFrame([
        _record("keep", "t1", 1.0, embedding=[0.1], impacts=[]),
        _record("stale", "t2", 10.0, embedding=[0.2], impacts=[]),  # beyond 4h max
    ]), now=NOW)

    df = store.load()
    assert set(df["doc_hash"]) == {"keep"}


def test_to_instrument_frame_explodes_impacts(tmp_path):
    store = _store(tmp_path)
    store.add(pd.DataFrame([
        _record("a", "t1", 1.0, embedding=[0.1],
                impacts=[{"instrument": "AAPL", "sentiment": 0.5},
                         {"instrument": "SPY", "sentiment": -0.5}]),
        _record("b", "t2", 1.0, embedding=[0.2],
                impacts=[{"instrument": "AAPL", "sentiment": 1.0}]),
    ]), now=NOW)

    long = FeatureStore.to_instrument_frame(store.window(4.0, now=NOW))
    assert len(long) == 3
    aapl = long[long["instrument"] == "AAPL"]
    assert set(aapl["sentiment"]) == {0.5, 1.0}
    assert set(long.columns) == {
        "instrument", "sentiment", "date", "source_name", "source_weight", "topic_id",
    }


def test_to_instrument_frame_empty_is_typed():
    out = FeatureStore.to_instrument_frame(pd.DataFrame())
    assert out.empty
    assert "instrument" in out.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
