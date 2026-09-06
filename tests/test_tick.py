"""
Tests for TickRunner — the one-tick orchestration that produces the eval dataset.

Strategy: wire the REAL deterministic stack (FeatureStore, score_variants,
SignalGenerator, SignalLog) on tmp paths, and stub ONLY the two LLM-bound collaborators
(lower path clustering + macro overlay). That exercises the actual wiring end to end —
cached features -> per-variant signals -> persisted log — with no network.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
from nlp_predictor.instrument_trends import UpperPathAnalyzer

from nlp_predictor.lower_path.macro_overlay import MacroOverlay
from nlp_predictor.preprocessing.feature_store import FeatureStore
from nlp_predictor.signal.signal_generator import SignalGenerator
from nlp_predictor.signal.signal_log import SignalLog
from nlp_predictor.tick import TickRunner

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


class _FakeLower:
    """Stub clustering: returns a fixed macro-climate frame, counts its calls."""

    def __init__(self, global_trends):
        self._gt = global_trends
        self.calls = 0

    def process_data(self, topics_df, now=None):
        self.calls += 1
        return self._gt


class _FakeMacro:
    """Stub overlay: returns a fixed MacroOverlay, counts its calls."""

    def __init__(self, overlay):
        self._ov = overlay
        self.calls = 0

    def build(self, global_trends_df):
        self.calls += 1
        return self._ov


def _populate_store(store):
    """Cache three topics (AAPL x2, SPY x1) as if already ingested this/last tick."""
    records = pd.DataFrame(
        [
            {
                "doc_hash": "h1",
                "topic_id": "t1",
                "topic_name": "AI capex",
                "topic_description": "d",
                "date": NOW - timedelta(hours=2),
                "source_name": "Bloomberg",
                "source_weight": 1.0,
                "embedding": [0.1, 0.2],
                "impacts": [{"instrument": "AAPL", "sentiment": 0.5}],
            },
            {
                "doc_hash": "h2",
                "topic_id": "t2",
                "topic_name": "Cloud demand",
                "topic_description": "d",
                "date": NOW - timedelta(hours=3),
                "source_name": "Reuters",
                "source_weight": 0.95,
                "embedding": [0.1, 0.3],
                "impacts": [{"instrument": "AAPL", "sentiment": 0.5}],
            },
            {
                "doc_hash": "h3",
                "topic_id": "t3",
                "topic_name": "Risk off",
                "topic_description": "d",
                "date": NOW - timedelta(hours=1),
                "source_name": "CNBC",
                "source_weight": 0.8,
                "embedding": [-0.4, 0.1],
                "impacts": [{"instrument": "SPY", "sentiment": -0.5}],
            },
        ]
    )
    store.add(records, now=NOW)


def _global_trends():
    """A macro climate where AAPL's topics (t1,t2) are a strong direct driver."""
    return pd.DataFrame(
        [
            {
                "title": "AI boom",
                "direction": "up",
                "severity": 20,
                "topic_ids": ["t1", "t2"],
                "description": "d",
            },
            {
                "title": "Risk off",
                "direction": "down",
                "severity": 5,
                "topic_ids": ["t3"],
                "description": "d",
            },
        ]
    )


def _overlay():
    return MacroOverlay(
        effects={
            "AAPL": {"effect": "tailwind", "strength": 1.0, "rationale": "AI boom"},
            "SPY": {"effect": "headwind", "strength": 1.0, "rationale": "risk off"},
        },
        conviction=0.9,
    )


def _fakes():
    return _FakeLower(_global_trends()), _FakeMacro(_overlay())


def _runner(tmp_path, lower, macro):
    store = FeatureStore(
        path=str(tmp_path / "fs" / "feat.parquet"), max_window_hours=36.0
    )
    _populate_store(store)
    log = SignalLog(path=str(tmp_path / "log" / "sig.parquet"))
    runner = TickRunner(
        store=store,
        upper=UpperPathAnalyzer.__new__(UpperPathAnalyzer),  # no init needed
        lower=lower,
        macro=macro,
        signal=SignalGenerator(),
        log=log,
        variants=VARIANTS,
        climate_window_hours=16.0,
    )
    return runner, log


def test_tick_logs_row_per_variant_instrument(tmp_path):
    runner, log = _runner(tmp_path, *_fakes())
    result = runner.run(now=NOW)

    # 2 variants x 2 instruments = 4 signals, persisted to the log.
    assert len(result) == 4
    assert set(result["variant"]) == {"baseline_16h", "dcim_12h"}
    assert set(result["instrument"]) == {"AAPL", "SPY"}

    logged = log.load()
    assert len(logged) == 4
    assert set(logged["variant"]) == {"baseline_16h", "dcim_12h"}


def test_tick_id_is_shared_across_the_tick(tmp_path):
    runner, log = _runner(tmp_path, *_fakes())
    result = runner.run(now=NOW)
    assert result["tick_id"].nunique() == 1
    assert result["tick_time"].iloc[0] == "2026-06-04 12:00:00"


def test_climate_built_once_per_tick_not_per_variant(tmp_path):
    lower, macro = _FakeLower(_global_trends()), _FakeMacro(_overlay())
    runner, _ = _runner(tmp_path, lower, macro)
    runner.run(now=NOW)
    # Shared across the 2 variants -> exactly one lower-path + one overlay call.
    assert lower.calls == 1
    assert macro.calls == 1


def test_direction_follows_severity_sign(tmp_path):
    runner, _ = _runner(tmp_path, _FakeLower(_global_trends()), _FakeMacro(_overlay()))
    result = runner.run(now=NOW).set_index(["variant", "instrument"])
    # AAPL fed by +0.5 sentiment -> UP; SPY by -0.5 -> DOWN (both variants).
    assert result.loc[("baseline_16h", "AAPL"), "signal"] == "UP"
    assert result.loc[("dcim_12h", "SPY"), "signal"] == "DOWN"


def test_empty_store_yields_no_signals(tmp_path):
    store = FeatureStore(
        path=str(tmp_path / "fs" / "feat.parquet"), max_window_hours=36.0
    )  # never populated
    log = SignalLog(path=str(tmp_path / "log" / "sig.parquet"))
    lower, macro = _FakeLower(_global_trends()), _FakeMacro(_overlay())
    runner = TickRunner(
        store=store,
        upper=UpperPathAnalyzer.__new__(UpperPathAnalyzer),
        lower=lower,
        macro=macro,
        signal=SignalGenerator(),
        log=log,
        variants=VARIANTS,
    )
    assert runner.run(now=NOW).empty
    assert log.load().empty
    # Short-circuited before touching the (LLM-bound) climate.
    assert lower.calls == 0
    assert macro.calls == 0
