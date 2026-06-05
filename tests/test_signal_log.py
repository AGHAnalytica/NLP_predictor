"""
Tests for SignalLog — the append-only eval dataset.

Exercise the round trip (JSON list columns survive), the fixed-schema coercion (extra
columns dropped, missing ones filled), append accumulation across ticks, and the
empty/no-op guard. Uses a tmp parquet path so nothing touches the real log.
"""

import pandas as pd

from nlp_predictor.signal_log import SignalLog


def _log(tmp_path):
    return SignalLog(path=str(tmp_path / "log" / "signals.parquet"))


def _rows(tick_id, variant="dcim_16h", instrument="AAPL", signal="UP"):
    # Includes extra columns (signal_id/current_time) that the fixed schema should drop.
    return pd.DataFrame([{
        "signal_id": "x",
        "current_time": "2026-06-04 12:00:00",
        "tick_id": tick_id,
        "tick_time": "2026-06-04 12:00:00",
        "variant": variant,
        "instrument": instrument,
        "signal": signal,
        "confidence": 0.6,
        "instrument_severity": 0.5,
        "macro_context": "AGREE",
        "n_topics": 2,
        "topic_ids": ["t1", "t2"],
        "rationale": "because",
    }])


def test_append_then_load_round_trip(tmp_path):
    log = _log(tmp_path)
    n = log.append(_rows("tick-1"))
    assert n == 1

    out = log.load()
    assert len(out) == 1
    row = out.iloc[0]
    assert row["tick_id"] == "tick-1"
    assert row["variant"] == "dcim_16h"
    # topic_ids round-trips back into a real list, not a JSON string.
    assert row["topic_ids"] == ["t1", "t2"]


def test_schema_is_fixed(tmp_path):
    log = _log(tmp_path)
    log.append(_rows("tick-1"))
    out = log.load()
    assert list(out.columns) == SignalLog.COLUMNS
    # Extra source columns were dropped by the schema coercion.
    assert "signal_id" not in out.columns


def test_append_accumulates_across_ticks(tmp_path):
    log = _log(tmp_path)
    log.append(_rows("tick-1", instrument="AAPL"))
    log.append(_rows("tick-2", instrument="MSFT"))

    out = log.load()
    assert len(out) == 2
    assert set(out["tick_id"]) == {"tick-1", "tick-2"}
    assert set(out["instrument"]) == {"AAPL", "MSFT"}


def test_missing_column_is_filled(tmp_path):
    log = _log(tmp_path)
    rows = _rows("tick-1").drop(columns=["rationale"])  # caller omitted a field
    log.append(rows)
    out = log.load()
    assert "rationale" in out.columns
    assert pd.isna(out.iloc[0]["rationale"])


def test_empty_append_is_noop(tmp_path):
    log = _log(tmp_path)
    assert log.append(pd.DataFrame()) == 0
    assert log.append(None) == 0
    assert log.load().empty


def test_load_missing_file_is_empty(tmp_path):
    log = _log(tmp_path)
    out = log.load()
    assert out.empty
    assert list(out.columns) == SignalLog.COLUMNS
