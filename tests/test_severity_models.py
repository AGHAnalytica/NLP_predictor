"""
Tests for the pure severity registry (step 3).

These lock in the properties the A/B/n relies on: each model is a deterministic,
bounded function of its inputs; different params produce different severities on the
same group; direction follows sentiment; and abnormal_volume keys magnitude off volume.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from nlp_predictor.lower_path.severity_models import (
    REQUIRED_COLUMNS,
    SEVERITY_REGISTRY,
    severity_abnormal_volume,
    severity_baseline,
    severity_dcim,
)

NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

DCIM_PARAMS = {
    "half_life_hours": 12.0,
    "beta_penalty": 0.2,
    "gamma_consensus": 0.3,
    "delta_velocity": 0.3,
    "kappa_scale": 0.8,
    "recent_window_hours": 1.5,
}


def _group(rows):
    """rows: list of (hours_ago, source, sentiment, weight)."""
    return pd.DataFrame(
        {
            "date": [NOW - timedelta(hours=h) for h, _, _, _ in rows],
            "source_name": [s for _, s, _, _ in rows],
            "sentiment": [sent for _, _, sent, _ in rows],
            "source_weight": [w for _, _, _, w in rows],
            "topic_id": [f"t{i}" for i in range(len(rows))],
        }
    )


def test_registry_has_expected_models():
    assert set(SEVERITY_REGISTRY) == {"baseline", "dcim", "abnormal_volume"}
    assert REQUIRED_COLUMNS == ("date", "source_name", "sentiment", "source_weight")


def test_empty_group_is_zero():
    empty = _group([])
    for fn in SEVERITY_REGISTRY.values():
        assert fn(empty, DCIM_PARAMS, NOW, None) == 0.0


def test_severity_is_bounded():
    # A pile of strong, fresh, agreeing articles must still clip into [-1, 1].
    g = _group([(0.1, f"src{i}", 1.0, 1.0) for i in range(50)])
    for name, fn in SEVERITY_REGISTRY.items():
        s = fn(g, {**DCIM_PARAMS, "baseline_count": 1.0}, NOW, {"baseline_count": 5.0})
        assert -1.0 <= s <= 1.0, name


def test_direction_follows_sentiment_sign():
    bullish = _group([(1, "Bloomberg", 1.0, 1.0), (2, "Reuters", 0.5, 0.95)])
    bearish = _group([(1, "Bloomberg", -1.0, 1.0), (2, "Reuters", -0.5, 0.95)])
    for fn in (severity_dcim, severity_baseline):
        assert fn(bullish, DCIM_PARAMS, NOW, None) > 0
        assert fn(bearish, DCIM_PARAMS, NOW, None) < 0


def test_zero_sentiment_gives_zero_severity():
    # Lots of articles, but no directional sentiment -> no signal, for every model.
    g = _group([(1, f"src{i}", 0.0, 1.0) for i in range(10)])
    assert severity_dcim(g, DCIM_PARAMS, NOW, None) == 0.0
    assert severity_baseline(g, DCIM_PARAMS, NOW, None) == 0.0
    assert severity_abnormal_volume(g, DCIM_PARAMS, NOW, {"baseline_count": 1.0}) == 0.0


def test_different_params_change_dcim_severity():
    # Same group, two DCIM tunings must be distinguishable (the basis of the A/B).
    g = _group(
        [(1, "Bloomberg", 0.5, 1.0), (3, "Reuters", 0.5, 0.95), (5, "CNBC", 0.5, 0.8)]
    )
    fast = {**DCIM_PARAMS, "half_life_hours": 2.0, "kappa_scale": 1.0}
    slow = {**DCIM_PARAMS, "half_life_hours": 24.0, "kappa_scale": 0.5}
    assert severity_dcim(g, fast, NOW, None) != severity_dcim(g, slow, NOW, None)


def test_baseline_and_dcim_differ_on_multisource():
    # The DCIM's consensus/echo terms should move it off the plain baseline sum.
    g = _group(
        [(1, "Bloomberg", 0.5, 1.0), (1, "Reuters", 0.5, 0.95), (1, "CNBC", 0.5, 0.8)]
    )
    assert severity_dcim(g, DCIM_PARAMS, NOW, None) != severity_baseline(
        g, DCIM_PARAMS, NOW, None
    )


def test_abnormal_volume_scales_with_coverage():
    params = {"half_life_hours": 12.0, "kappa_scale": 1.0}
    quiet = _group([(1, "Bloomberg", 0.5, 1.0)])  # 1 article
    loud = _group([(1, f"src{i}", 0.5, 1.0) for i in range(8)])  # 8 articles
    ctx = {"baseline_count": 2.0}
    s_quiet = severity_abnormal_volume(quiet, params, NOW, ctx)
    s_loud = severity_abnormal_volume(loud, params, NOW, ctx)
    # Below baseline coverage -> ~0; well above baseline -> a real positive signal.
    assert s_quiet == 0.0
    assert s_loud > 0.5


def test_abnormal_volume_direction_from_sentiment_not_count():
    params = {"half_life_hours": 12.0, "kappa_scale": 1.0}
    ctx = {"baseline_count": 2.0}
    bearish_loud = _group([(1, f"src{i}", -1.0, 1.0) for i in range(8)])
    s = severity_abnormal_volume(bearish_loud, params, NOW, ctx)
    assert s < 0  # heavy coverage, but negative sentiment -> DOWN


def test_dcim_matches_legacy_numbers():
    # Guard against drift: the extracted pure DCIM must reproduce the in-class result
    # the quantify script measured, given identical legacy params.
    legacy = {
        "half_life_hours": 4.0,
        "beta_penalty": 0.2,
        "gamma_consensus": 0.3,
        "delta_velocity": 0.3,
        "kappa_scale": 1.0,
        "recent_window_hours": 0.5,
    }
    g = _group([(2, f"src{i}", 0.5, 0.8) for i in range(4)])
    # Reproduce the analytic value: 4 distinct sources, decay over 2h, half-life 4h.
    decay = np.exp(-np.log(2) / 4.0 * 2.0)
    total = 4 * (0.8 * 0.5 * decay)  # 4 sources, no echo penalty (1 each)
    C = 1.0 + 0.3 * 1.0  # all 4 agree
    V = 0.0  # none within 0.5h
    expected = float(np.tanh(1.0 * C * (1.0 + 0.3 * V) * total))
    assert (
        severity_dcim(g, legacy, NOW, None) == round(expected, 12)
        or abs(severity_dcim(g, legacy, NOW, None) - expected) < 1e-9
    )


def test_dcim_mixed_sentiments_same_source():
    # 1. A negative article from the same source MUST reduce the severity compared to
    # positive only.
    pos_only = _group([(1.0, "Bloomberg", 0.8, 1.0)])
    mixed = _group([(1.0, "Bloomberg", 0.8, 1.0), (2.0, "Bloomberg", -0.4, 1.0)])
    sev_pos = severity_dcim(pos_only, DCIM_PARAMS, NOW, None)
    sev_mixed = severity_dcim(mixed, DCIM_PARAMS, NOW, None)
    assert 0.0 < sev_mixed < sev_pos, "Negative article should reduce bullish severity"

    # 2. Equal and opposite articles published simultaneously must cancel out to 0.0.
    symmetric = _group([(1.0, "Bloomberg", 0.8, 1.0), (1.0, "Bloomberg", -0.8, 1.0)])
    sev_sym = severity_dcim(symmetric, DCIM_PARAMS, NOW, None)
    assert abs(sev_sym) < 1e-9, "Equal and opposite articles should cancel out to 0.0"

    # 3. Echo-chamber penalty still applies to same-sign articles (sublinear growth).
    two_pos = _group([(1.0, "Bloomberg", 0.8, 1.0), (2.0, "Bloomberg", 0.8, 1.0)])
    sev_two = severity_dcim(two_pos, DCIM_PARAMS, NOW, None)
    assert sev_two > sev_pos
    # Diminishing returns: second article has lower weight than first
    assert sev_two < 2.0 * sev_pos
