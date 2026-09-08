"""
Tests for the macro overlay (step 3b). The LLM call is stubbed, so these exercise the
deterministic parts: climate-summary building, conviction, ticker validation /
clamping / default-neutral, and the empty-climate path.
"""

import math

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.lower_path.macro_overlay import (
    MacroOverlay,
    MacroOverlayAnalyzer,
    _InstrumentEffect,
    _MacroMapping,
)


def _analyzer(tickers=("AAPL", "SPY", "GLD")):
    """Build the analyzer without __init__ (no API key / CSV needed)."""
    a = MacroOverlayAnalyzer.__new__(MacroOverlayAnalyzer)
    a.valid_tickers = set(tickers)
    a.available_instruments = {t: f"{t} Full Name" for t in tickers}
    return a


def _global_trends(rows):
    """rows: list of (title, direction, severity)."""
    return pd.DataFrame(
        [
            {"title": t, "direction": d, "severity": s, "description": "desc"}
            for t, d, s in rows
        ]
    )


def test_overlay_for_instrument_defaults_neutral():
    ov = MacroOverlay(
        effects={"AAPL": {"effect": "tailwind", "strength": 0.8, "rationale": "r"}}
    )
    assert ov.for_instrument("AAPL")["effect"] == "tailwind"
    miss = ov.for_instrument("UNKNOWN")
    assert miss["effect"] == "neutral" and miss["strength"] == 0.0


def test_build_empty_when_no_trends():
    a = _analyzer()
    assert a.build(None).effects == {}
    assert a.build(pd.DataFrame()).conviction == 0.0


def test_conviction_is_tanh_of_total_severity():
    a = _analyzer()
    gt = _global_trends([("A", "up", 6), ("B", "down", 4)])  # total 10
    scale = float(config.hyperparams.signal.macro_conviction_scale)
    assert a._conviction(gt) == math.tanh(10.0 / scale)


def test_conviction_grows_with_volume():
    a = _analyzer()
    weak = a._conviction(_global_trends([("A", "up", 2)]))
    strong = a._conviction(_global_trends([("A", "up", 20), ("B", "up", 15)]))
    assert strong > weak


def test_climate_summary_lists_clusters():
    a = _analyzer()
    gt = _global_trends([("Hawkish Fed", "down", 5)])
    summary = a._climate_summary(gt)
    assert "Hawkish Fed" in summary
    assert "down" in summary
    assert "5 topics" in summary


def test_map_climate_validates_and_clamps(monkeypatch):
    a = _analyzer(tickers=("AAPL", "SPY"))

    def _e(ticker, effect, strength):
        return _InstrumentEffect(
            ticker=ticker, effect=effect, strength=strength, rationale="r"
        )

    fake = _MacroMapping(
        effects=[
            _e("AAPL", "Tailwind", 1.5),  # mixed case + over-range strength
            _e("SPY", "headwind", 0.4),
            _e("FAKE", "tailwind", 0.9),  # unknown ticker -> dropped
            _e("AAPL", "sideways", 0.2),  # invalid effect -> dropped
        ]
    )
    monkeypatch.setattr(a, "_call_openai_api_with_retry", lambda prompt, schema: fake)

    effects = a._map_climate("some climate")
    # AAPL kept (effect lowercased, strength clamped to 1.0); SPY kept; FAKE dropped
    # (unknown ticker); the 'sideways' AAPL entry dropped (invalid effect) — but the
    # first valid AAPL stays.
    assert effects["AAPL"]["effect"] == "tailwind"
    assert effects["AAPL"]["strength"] == 1.0
    assert effects["SPY"]["effect"] == "headwind"
    assert "FAKE" not in effects


def test_map_climate_returns_empty_on_llm_error(monkeypatch):
    a = _analyzer()

    def _boom(prompt, schema):
        raise RuntimeError("api down")

    monkeypatch.setattr(a, "_call_openai_api_with_retry", _boom)
    assert a._map_climate("climate") == {}  # failure -> no climate, not a crash
