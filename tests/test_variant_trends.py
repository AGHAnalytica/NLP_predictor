"""
Tests for UpperPathAnalyzer.generate_variant_trends (step 3 orchestration).

The two LLM-touching steps (topic->instrument mapping and the per-instrument summary)
are stubbed, so the test exercises the deterministic part: per-variant windowing,
context, severity dispatch, and long-form row assembly.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd

from nlp_predictor.upper_path.upper_path_analyzer import UpperPathAnalyzer

NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

VARIANTS = [
    {
        "name": "baseline_16h",
        "model": "baseline",
        "window_hours": 16.0,
        "params": {"half_life_hours": 12.0, "kappa_scale": 0.8},
    },
    {
        "name": "dcim_12h",
        "model": "dcim",
        "window_hours": 12.0,
        "params": {"half_life_hours": 8.0, "kappa_scale": 0.8},
    },
]


def _analyzer():
    # Bypass __init__ (no API key / instrument CSV needed for the deterministic path).
    return UpperPathAnalyzer.__new__(UpperPathAnalyzer)


def _mapped_df():
    """Two instruments; AAPL has a fresh + an old article, SPY one fresh article."""
    rows = [
        ("AAPL", 0.5, 2.0, "Bloomberg", 1.0, "t1"),  # in both 12h and 16h windows
        ("AAPL", 0.5, 14.0, "Reuters", 0.95, "t2"),  # only in the 16h window
        ("SPY", -0.5, 1.0, "CNBC", 0.8, "t3"),
    ]
    return pd.DataFrame(
        {
            "instrument": [r[0] for r in rows],
            "sentiment": [r[1] for r in rows],
            "date": [NOW - timedelta(hours=r[2]) for r in rows],
            "source_name": [r[3] for r in rows],
            "source_weight": [r[4] for r in rows],
            "topic_id": [r[5] for r in rows],
            "topic_name": ["n"] * len(rows),
            "topic_description": ["d"] * len(rows),
        }
    )


_STUB_SUMMARY = {"title": "T", "description": "D", "llm_direction": "neutral"}


def _patch(monkeypatch, analyzer, mapped):
    monkeypatch.setattr(analyzer, "_map_and_explode", lambda df: mapped)
    monkeypatch.setattr(
        analyzer,
        "_summaries_by_instrument",
        lambda df: {inst: dict(_STUB_SUMMARY) for inst in df["instrument"].unique()},
    )


def test_long_form_has_row_per_variant_instrument(monkeypatch):
    a = _analyzer()
    _patch(monkeypatch, a, _mapped_df())

    out = a.generate_variant_trends(_mapped_df(), VARIANTS, now=NOW)

    # 2 variants x 2 instruments = 4 rows.
    assert len(out) == 4
    assert set(out["variant"]) == {"baseline_16h", "dcim_12h"}
    assert set(out["instrument"]) == {"AAPL", "SPY"}
    # Long-form schema carries the A/B label + provenance.
    for col in (
        "variant",
        "model",
        "window_hours",
        "severity",
        "direction",
        "llm_direction",
    ):
        assert col in out.columns


def test_direction_is_sign_of_severity(monkeypatch):
    a = _analyzer()
    _patch(monkeypatch, a, _mapped_df())
    out = a.generate_variant_trends(_mapped_df(), VARIANTS, now=NOW)

    spy = out[out["instrument"] == "SPY"]
    assert (spy["direction"] == "down").all()  # negative sentiment -> DOWN
    aapl = out[out["instrument"] == "AAPL"]
    assert (aapl["direction"] == "up").all()


def test_window_changes_which_articles_count(monkeypatch):
    # AAPL's 14h-old article is inside the 16h window but outside the 12h window, so the
    # two variants must see a different number of topics for AAPL.
    a = _analyzer()
    _patch(monkeypatch, a, _mapped_df())
    out = a.generate_variant_trends(_mapped_df(), VARIANTS, now=NOW)

    n16 = len(
        out[(out["variant"] == "baseline_16h") & (out["instrument"] == "AAPL")].iloc[0][
            "topic_ids"
        ]
    )
    n12 = len(
        out[(out["variant"] == "dcim_12h") & (out["instrument"] == "AAPL")].iloc[0][
            "topic_ids"
        ]
    )
    assert n16 == 2
    assert n12 == 1


def test_empty_input_returns_empty(monkeypatch):
    a = _analyzer()
    out = a.generate_variant_trends(pd.DataFrame(), VARIANTS, now=NOW)
    assert out.empty
