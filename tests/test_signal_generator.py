"""
Tests for SignalGenerator after step 2 — direction/confidence are a deterministic
function of severity ONLY. The per-instrument LLM ``direction`` field must not influence
the signal or confidence (otherwise the A/B/n across severity methods is confounded).
"""

import pandas as pd

from nlp_predictor.signal_generator import SignalGenerator


def _trend(instrument, severity, direction, topic_ids=("t1",)):
    return {
        "instrument": instrument,
        "severity": severity,
        "direction": direction,  # the (now decorative) LLM opinion
        "topic_ids": list(topic_ids),
    }


def test_confidence_independent_of_llm_direction():
    # Same severity; only the LLM ``direction`` field differs (one agrees with the
    # severity sign, one contradicts it). Confidence and signal must be identical.
    df = pd.DataFrame([
        _trend("AAPL", 0.5, "up"),
        _trend("MSFT", 0.5, "down"),  # LLM says down, severity says up
    ])
    out = SignalGenerator().generate(df, None).set_index("instrument")

    assert out.loc["AAPL", "confidence"] == out.loc["MSFT", "confidence"]
    assert out.loc["AAPL", "signal"] == "UP"
    assert out.loc["MSFT", "signal"] == "UP"  # severity sign wins, not the LLM


def test_direction_follows_severity_sign():
    df = pd.DataFrame([
        _trend("UPPER", 0.4, "down"),
        _trend("LOWER", -0.4, "up"),
        _trend("FLAT", 0.0, "up"),
    ])
    out = SignalGenerator().generate(df, None).set_index("instrument")

    assert out.loc["UPPER", "signal"] == "UP"
    assert out.loc["LOWER", "signal"] == "DOWN"
    assert out.loc["FLAT", "signal"] == "NEUTRAL"  # zero severity -> below threshold


def test_rationale_has_no_internal_conflict_note():
    df = pd.DataFrame([_trend("AAPL", 0.5, "down")])
    out = SignalGenerator().generate(df, None)
    assert "disagreed" not in out.iloc[0]["rationale"].lower()
