"""
Tests for UpperPathAnalyzer.score_variants — the live tick's LLM-free scoring path.

Unlike generate_variant_trends, this takes an ALREADY-MAPPED frame (the feature store's
to_instrument_frame output) and makes no API calls, so the whole thing is deterministic:
no stubbing needed. We check per-variant windowing, the severity-sign direction, the
lean output schema, and that no LLM helper is ever touched.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd

from nlp_predictor.instrument_trends import UpperPathAnalyzer

NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

VARIANTS = [
    {"name": "baseline_16h", "model": "baseline", "window_hours": 16.0,
     "params": {"half_life_hours": 12.0, "kappa_scale": 0.8}},
    {"name": "dcim_12h", "model": "dcim", "window_hours": 12.0,
     "params": {"half_life_hours": 8.0, "kappa_scale": 0.8}},
]


def _analyzer():
    # Bypass __init__ (no API key / instrument CSV needed for the deterministic path).
    return UpperPathAnalyzer.__new__(UpperPathAnalyzer)


def _mapped_df():
    """Pre-mapped impacts (what FeatureStore.to_instrument_frame yields)."""
    rows = [
        ("AAPL", 0.5, 2.0, "Bloomberg", 1.0, "t1"),   # in both 12h and 16h windows
        ("AAPL", 0.5, 14.0, "Reuters", 0.95, "t2"),   # only in the 16h window
        ("SPY", -0.5, 1.0, "CNBC", 0.8, "t3"),
    ]
    return pd.DataFrame({
        "instrument": [r[0] for r in rows],
        "sentiment": [r[1] for r in rows],
        "date": [NOW - timedelta(hours=r[2]) for r in rows],
        "source_name": [r[3] for r in rows],
        "source_weight": [r[4] for r in rows],
        "topic_id": [r[5] for r in rows],
    })


def test_row_per_variant_instrument_with_lean_schema():
    out = _analyzer().score_variants(_mapped_df(), VARIANTS, now=NOW)

    # 2 variants x 2 instruments = 4 rows.
    assert len(out) == 4
    assert set(out["variant"]) == {"baseline_16h", "dcim_12h"}
    assert set(out["instrument"]) == {"AAPL", "SPY"}
    for col in ("variant", "model", "window_hours", "instrument", "severity",
                "direction", "topic_ids"):
        assert col in out.columns
    # Lean path carries NO LLM rationale columns.
    for col in ("title", "description", "llm_direction"):
        assert col not in out.columns


def test_direction_is_sign_of_severity():
    out = _analyzer().score_variants(_mapped_df(), VARIANTS, now=NOW)
    assert (out[out["instrument"] == "SPY"]["direction"] == "down").all()
    assert (out[out["instrument"] == "AAPL"]["direction"] == "up").all()


def test_window_changes_which_articles_count():
    # AAPL's 14h-old article is inside 16h but outside 12h, so topic counts differ.
    out = _analyzer().score_variants(_mapped_df(), VARIANTS, now=NOW)
    n16 = len(out[(out["variant"] == "baseline_16h")
                  & (out["instrument"] == "AAPL")].iloc[0]["topic_ids"])
    n12 = len(out[(out["variant"] == "dcim_12h")
                  & (out["instrument"] == "AAPL")].iloc[0]["topic_ids"])
    assert n16 == 2
    assert n12 == 1


def test_no_llm_call_is_made(monkeypatch):
    # The whole point of score_variants: zero API calls. Booby-trap the LLM helpers.
    a = _analyzer()

    def _boom(*args, **kwargs):
        raise AssertionError("score_variants must not call the LLM")

    monkeypatch.setattr(a, "_map_and_explode", _boom)
    monkeypatch.setattr(a, "_summaries_by_instrument", _boom)
    out = a.score_variants(_mapped_df(), VARIANTS, now=NOW)
    assert len(out) == 4


def test_empty_input_returns_empty():
    assert _analyzer().score_variants(pd.DataFrame(), VARIANTS, now=NOW).empty
