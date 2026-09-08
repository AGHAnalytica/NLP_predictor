"""
Tests for LowerPathAnalyzer's feature-store integration: it must REUSE embeddings cached
on each topic row (the live-tick path) and only hit the embedding API for an uncached
batch. The per-cluster summary LLM call is stubbed, so whatever HDBSCAN does with the
points, no network is touched and the assertions are purely about embedding behaviour.
"""

from datetime import UTC, datetime

import pandas as pd

from nlp_predictor.lower_path.lower_path_analyzer import LowerPathAnalyzer

NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)


def _analyzer(monkeypatch):
    # Bypass __init__ (no OpenAI client); stub the only other LLM touchpoint.
    a = LowerPathAnalyzer.__new__(LowerPathAnalyzer)
    monkeypatch.setattr(
        a,
        "_generate_global_trend_summary",
        lambda cid, group, sev: {
            "title": "t",
            "direction": "up",
            "severity": sev,
            "topic_ids": group["topic_id"].tolist(),
        },
    )
    return a


def _topics(with_embedding, n=4):
    rows = []
    for i in range(n):  # >= MIN_CLUSTER_SIZE so HDBSCAN runs without raising
        row = {
            "topic_id": f"t{i}",
            "topic_name": "n",
            "topic_description": "d",
            "date": NOW,
        }
        if with_embedding:
            row["embedding"] = [0.01 * i, 0.02 * i]
        rows.append(row)
    return pd.DataFrame(rows)


def test_reuses_cached_embeddings(monkeypatch):
    a = _analyzer(monkeypatch)

    def _boom(texts):
        raise AssertionError("must not call the embedding API when vectors are cached")

    monkeypatch.setattr(a, "_generate_embeddings_batch", _boom)
    # Cached vectors present -> reuse path -> no embedding API call (and no crash).
    a.process_data(_topics(with_embedding=True))


def test_computes_embeddings_when_absent(monkeypatch):
    a = _analyzer(monkeypatch)
    calls = {"n": 0}

    def _emb(texts):
        calls["n"] += 1
        return [[0.01 * i, 0.02 * i] for i in range(len(texts))]

    monkeypatch.setattr(a, "_generate_embeddings_batch", _emb)
    a.process_data(_topics(with_embedding=False))
    assert calls["n"] == 1  # no cached column -> exactly one batch embed call


def test_too_few_topics_skips_clustering(monkeypatch):
    # Below MIN_CLUSTER_SIZE: short-circuit to empty rather than letting HDBSCAN raise,
    # and without paying for an embedding call.
    a = _analyzer(monkeypatch)

    def _boom(texts):
        raise AssertionError("must not embed when there are too few topics to cluster")

    monkeypatch.setattr(a, "_generate_embeddings_batch", _boom)
    assert a.process_data(_topics(with_embedding=False, n=2)).empty
