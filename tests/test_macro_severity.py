"""
Tests for the macro (lower-path) cluster severity models — the unsigned narrative
INTENSITY that feeds climate conviction. The point of the upgrade: two clusters of the
same raw size must score differently once recency and publisher credibility are folded
in (a fresh, reputable narrative > a stale, low-credibility one).
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from nlp_predictor.core.severity_models import (
    MACRO_SEVERITY_REGISTRY,
    macro_severity_count,
    macro_severity_decayed,
)

NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)
HL = {"half_life_hours": 18.0}


def _cluster(rows):
    """rows: list of (age_hours, source_weight)."""
    return pd.DataFrame(
        {
            "date": [NOW - timedelta(hours=a) for a, _ in rows],
            "source_weight": [w for _, w in rows],
        }
    )


def test_count_is_raw_size():
    assert macro_severity_count(_cluster([(1, 1.0), (2, 0.5), (3, 0.8)])) == 3.0


def test_fresh_full_weight_topic_scores_its_weight():
    # decay ~1 at age 0 -> intensity ~= source_weight.
    assert macro_severity_decayed(_cluster([(0.0, 1.0)]), HL, NOW) == pytest.approx(1.0)


def test_one_half_life_old_halves_the_weight():
    out = macro_severity_decayed(_cluster([(18.0, 1.0)]), HL, NOW)
    assert out == pytest.approx(0.5)


def test_source_weight_scales_intensity():
    low = macro_severity_decayed(_cluster([(0.0, 0.3)]), HL, NOW)
    high = macro_severity_decayed(_cluster([(0.0, 1.0)]), HL, NOW)
    assert high > low


def test_fresh_beats_stale_for_same_raw_size():
    fresh = _cluster([(1, 1.0), (1, 1.0), (1, 1.0)])
    stale = _cluster([(30, 1.0), (30, 1.0), (30, 1.0)])
    # Same count...
    assert macro_severity_count(fresh) == macro_severity_count(stale) == 3.0
    # ...different intensity. That divergence is the whole reason for the upgrade.
    s_fresh = macro_severity_decayed(fresh, HL, NOW)
    s_stale = macro_severity_decayed(stale, HL, NOW)
    assert s_fresh > s_stale


def test_longer_half_life_keeps_more_weight_on_old_news():
    stale = _cluster([(24.0, 1.0)])
    fast = macro_severity_decayed(stale, {"half_life_hours": 8.0}, NOW)
    slow = macro_severity_decayed(stale, {"half_life_hours": 24.0}, NOW)
    assert slow > fast  # the "macro persists longer" knob


def test_falls_back_to_count_without_columns():
    bare = pd.DataFrame({"topic_id": ["a", "b"]})  # no date / source_weight
    assert macro_severity_decayed(bare, HL, NOW) == 2.0


def test_empty_is_zero():
    assert macro_severity_decayed(pd.DataFrame(), HL, NOW) == 0.0


def test_registry_exposes_both_models():
    assert set(MACRO_SEVERITY_REGISTRY) == {"count", "decayed_weighted"}
