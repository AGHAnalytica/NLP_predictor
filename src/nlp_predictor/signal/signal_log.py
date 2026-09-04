"""
Signal log — the append-only EVAL DATASET for the live A/B/n.

Every 15-min tick scores each severity variant and turns it into a per-instrument
signal (see :class:`tick.TickRunner`). This module persists those signals, one row per
``(tick, variant, instrument)``, so they can later be joined to each variant's realized
demo-account PnL. That join is the whole point of the experiment: the log is the X (what
each model decided, when, with what confidence and macro context) and the broker fills
are the y (what actually happened). Without a durable, append-only record of every
decision at the moment it was made, there is nothing to score a model against.

It is deliberately append-only and immutable: a tick's rows are written once and never
edited, so the dataset is an honest, point-in-time record (no look-ahead, no silent
rewrites of past decisions). All rows from one tick share a ``tick_id``, so a tick is an
atomic unit you can group by.

On-disk schema (one row per (tick, variant, instrument), Parquet):
    - tick_id (str)              : groups every row produced in one 15-min tick
    - tick_time (str)            : tick timestamp (UTC, 'YYYY-MM-DD HH:MM:SS')
    - variant (str)              : the severity strategy that produced this signal
    - instrument (str)           : ticker
    - signal (str)               : 'UP' | 'DOWN' | 'NEUTRAL' (the decision)
    - confidence (float)         : [0, 1]
    - instrument_severity (float): signed severity that set the direction
    - macro_context (str)        : 'AGREE' | 'CONFLICT' | 'NONE'
    - n_topics (int)             : number of topics behind the signal
    - topic_ids (JSON str)       : the topics that drove it (audit trail)
    - rationale (str)            : human-readable explanation
"""

import json
import os

import pandas as pd

from nlp_predictor.config import Config  # type: ignore

# Columns serialized to JSON on disk and parsed back into Python objects in memory.
_JSON_COLUMNS = ("topic_ids",)


class SignalLog:
    """
    Parquet-backed, append-only log of per-tick signals — the A/B/n eval dataset.

    A thin, dependency-free persistence layer: :meth:`append` (write a tick's rows) and
    :meth:`load` (read the whole history back). It computes nothing — the
    :class:`tick.TickRunner` owns producing the signals; this only records them — which
    keeps the eval dataset trivially auditable and the writer easy to unit-test.
    """

    COLUMNS = [
        "tick_id",
        "tick_time",
        "variant",
        "instrument",
        "signal",
        "confidence",
        "instrument_severity",
        "macro_context",
        "n_topics",
        "topic_ids",
        "rationale",
    ]

    def __init__(self, path: str = Config["SIGNAL_LOG_PATH"]):
        """
        Args:
            path (str): Parquet file backing the log. Created lazily on first
                :meth:`append`; a missing file reads back as an empty log.
        """
        self.path = path

    # ------------------------------------------------------------------ read
    def load(self) -> pd.DataFrame:
        """
        Read the entire signal history with JSON columns parsed back into objects.

        Returns:
            pd.DataFrame: All logged rows with :attr:`COLUMNS` (``topic_ids`` as a
            ``list``). An empty, correctly typed frame if the log does not exist yet.
        """
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.COLUMNS)

        df = pd.read_parquet(self.path)
        if df.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        for col in _JSON_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(self._from_json)
        return df

    # ----------------------------------------------------------------- write
    def append(self, rows: pd.DataFrame) -> int:
        """
        Append one tick's signals to the log (append-only — never edits past rows).

        Coerces the incoming frame to the fixed :attr:`COLUMNS` schema (dropping extra
        columns such as ``signal_id``/``current_time`` and filling any missing one), so
        callers can hand it raw :meth:`SignalGenerator.generate` output augmented with
        ``tick_id``/``tick_time``/``variant`` without bookkeeping here.

        Args:
            rows (pd.DataFrame): Signal rows to record. Should carry the identity
                columns ``tick_id``, ``tick_time`` and ``variant`` plus the
                per-instrument signal fields; anything else in :attr:`COLUMNS` is
                filled with a null/default.

        Returns:
            int: Number of rows appended (0 for an empty/None input).
        """
        if rows is None or rows.empty:
            return 0

        incoming = rows.reindex(columns=self.COLUMNS)
        for col in _JSON_COLUMNS:
            incoming[col] = incoming[col].apply(self._to_json)

        current = self._load_raw()
        combined = pd.concat([current, incoming], ignore_index=True)
        self._write_raw(combined)
        return len(incoming)

    # ------------------------------------------------------------- internals
    def _load_raw(self) -> pd.DataFrame:
        """Read the log without parsing JSON columns (write-path helper)."""
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.COLUMNS)
        df = pd.read_parquet(self.path)
        if df.empty:
            return pd.DataFrame(columns=self.COLUMNS)
        return df

    def _write_raw(self, df: pd.DataFrame) -> None:
        """Persist the log, creating the parent directory on first write."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        df.reindex(columns=self.COLUMNS).to_parquet(self.path, index=False)

    @staticmethod
    def _to_json(value: object) -> str:
        """Serialize a list cell to JSON (idempotent on strings)."""
        if isinstance(value, str):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "[]"
        if isinstance(value, (list, tuple, set)):
            value = list(value)
        return json.dumps(value)

    @staticmethod
    def _from_json(value: object) -> object:
        """Parse a JSON string cell into a Python object; pass non-strings through."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return []
        return value


if __name__ == "__main__":
    print("====== SignalLog Module Test ======")
    log = SignalLog()
    print(f"[INFO] Log path: {log.path}")
    df = log.load()
    print(f"[INFO] Logged rows: {len(df)}")
    if not df.empty:
        by_variant = df.groupby("variant")["signal"].value_counts()
        print("[INFO] Signal counts by variant:")
        print(by_variant.to_string())
