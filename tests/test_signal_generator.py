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
# AAPL's trend carries topic_ids ['t1']; a cluster sharing 't1' is a DIRECT driver, so
# its severity sets AAPL's conviction = tanh(severity / MACRO_CONVICTION_SCALE).
def _cluster(topic_ids, severity, direction="up", title="climate"):
    return {"topic_ids": list(topic_ids), "severity": severity,
            "direction": direction, "title": title}


def _gt(*clusters):
    return pd.DataFrame(list(clusters))


def _conf(severity, overlay=None, gt=None):
    df = pd.DataFrame([_trend("AAPL", severity, "neutral")])  # topic_ids ['t1']
    out = SignalGenerator().generate(df, gt, macro_overlay=overlay)
    return out.iloc[0]


def _ov(effect, strength=1.0):
    return {"AAPL": {"effect": effect, "strength": strength, "rationale": "climate"}}


# A strong direct driver for AAPL (conviction ~ tanh(30/10) ~ 0.995).
def _strong():
    return _gt(_cluster(["t1"], 30))


def test_overlay_agree_boosts_confidence():
    base = _conf(0.5)["confidence"]                                 # no overlay
    agree = _conf(0.5, _ov("tailwind"), _strong())["confidence"]    # up + tailwind
    assert agree > base


def test_overlay_conflict_trims_confidence():
    base = _conf(0.5)["confidence"]
    row = _conf(0.5, _ov("headwind"), _strong())  # up signal vs down climate
    assert row["confidence"] < base
    assert row["signal"] == "UP"  # direction unchanged by macro


def test_overlay_neutral_and_missing_do_nothing():
    base = _conf(0.5)["confidence"]
    assert _conf(0.5, _ov("neutral"), _strong())["confidence"] == base
    assert _conf(0.5, {}, _strong())["confidence"] == base  # instrument not mapped


def test_overlay_strength_scales_the_effect():
    soft = _conf(0.5, _ov("tailwind", strength=0.3), _strong())["confidence"]
    hard = _conf(0.5, _ov("tailwind", strength=1.0), _strong())["confidence"]
    assert hard > soft


def test_strong_opposing_climate_vetoes_weak_call():
    # Weak up call (|sev| 0.3 < veto cap 0.4) + strong opposing climate -> NEUTRAL.
    row = _conf(0.3, _ov("headwind"), _strong())
    assert row["signal"] == "NEUTRAL"
    assert row["macro_context"] == "CONFLICT"


def test_overlay_does_not_flip_direction():
    # Even a maximal tailwind cannot turn a DOWN severity into an UP signal.
    df = pd.DataFrame([_trend("AAPL", -0.5, "neutral")])
    out = SignalGenerator().generate(df, _strong(), macro_overlay=_ov("tailwind"))
    assert out.iloc[0]["signal"] == "DOWN"


def test_per_instrument_conviction_reflects_driver_not_total():
    # The fix: AAPL's conviction comes from ITS driving cluster (shares 't1'), not the
    # climate-wide total. Same total volume (23), but a 3-article driver must modulate
    # far less than a 20-article driver — a global sum would make these identical.
    small_driver = _gt(_cluster(["t1"], 3), _cluster(["zzz"], 20))   # AAPL driver = 3
    big_driver = _gt(_cluster(["t1"], 20), _cluster(["zzz"], 3))     # AAPL driver = 20
    c_small = _conf(0.5, _ov("tailwind"), small_driver)["confidence"]
    c_big = _conf(0.5, _ov("tailwind"), big_driver)["confidence"]
    assert c_big > c_small


def test_diffuse_fallback_uses_whole_climate():
    # AAPL shares NO topics with any cluster -> conviction = the whole climate's volume
    # (the broad-mood case), so a tailwind still meaningfully boosts the up call.
    no_overlap = _gt(_cluster(["x"], 10, "down"), _cluster(["y"], 10, "down"))
    boosted = _conf(0.5, _ov("tailwind"), no_overlap)["confidence"]
    assert boosted > _conf(0.5)["confidence"]
