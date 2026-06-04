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


# ----------------------------------------------------- macro overlay (step 3b)
def _conf(severity, overlay=None, conviction=0.0):
    df = pd.DataFrame([_trend("AAPL", severity, "neutral")])
    out = SignalGenerator().generate(
        df, None, macro_overlay=overlay, macro_conviction=conviction
    )
    return out.iloc[0]


def _ov(effect, strength=1.0):
    return {"AAPL": {"effect": effect, "strength": strength, "rationale": "climate"}}


def test_overlay_agree_boosts_confidence():
    base = _conf(0.5)["confidence"]                          # no overlay
    agree = _conf(0.5, _ov("tailwind"), conviction=1.0)["confidence"]  # up + tailwind
    assert agree > base


def test_overlay_conflict_trims_confidence():
    base = _conf(0.5)["confidence"]
    row = _conf(0.5, _ov("headwind"), conviction=1.0)  # up signal vs down climate
    assert row["confidence"] < base
    assert row["signal"] == "UP"  # direction unchanged by macro


def test_overlay_neutral_and_missing_do_nothing():
    base = _conf(0.5)["confidence"]
    assert _conf(0.5, _ov("neutral"), conviction=1.0)["confidence"] == base
    assert _conf(0.5, {}, conviction=1.0)["confidence"] == base  # instrument not mapped


def test_conviction_scales_the_effect():
    weak = _conf(0.5, _ov("tailwind"), conviction=0.3)["confidence"]
    strong = _conf(0.5, _ov("tailwind"), conviction=1.0)["confidence"]
    assert strong > weak  # a more pervasive climate moves confidence more


def test_overlay_strength_scales_the_effect():
    soft = _conf(0.5, _ov("tailwind", strength=0.3), conviction=1.0)["confidence"]
    hard = _conf(0.5, _ov("tailwind", strength=1.0), conviction=1.0)["confidence"]
    assert hard > soft


def test_strong_opposing_climate_vetoes_weak_call():
    # Weak up call (|sev| 0.3 < veto cap 0.4) + strong opposing climate -> NEUTRAL.
    row = _conf(0.3, _ov("headwind"), conviction=1.0)
    assert row["signal"] == "NEUTRAL"
    assert row["macro_context"] == "CONFLICT"


def test_overlay_does_not_flip_direction():
    # Even a maximal tailwind cannot turn a DOWN severity into an UP signal.
    df = pd.DataFrame([_trend("AAPL", -0.5, "neutral")])
    out = SignalGenerator().generate(
        df, None, macro_overlay=_ov("tailwind"), macro_conviction=1.0
    )
    assert out.iloc[0]["signal"] == "DOWN"
