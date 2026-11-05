"""Append-only Parquet log storing per-tick variant signals for A/B/n evaluation."""

import logging

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.core.parquet_io import (
    deserialize_json_cell,
    read_raw_parquet,
    serialize_json_cell,
    write_raw_parquet,
)

logger = logging.getLogger(__name__)

_JSON_COLUMNS = ("topic_ids",)


class SignalLog:
    """Append-only Parquet persistence layer for per-(tick, variant, instrument)
    signals.

    Schema (:attr:`COLUMNS`):
        - ``tick_id`` (str): Tick batch UUID.
        - ``tick_time`` (str): Tick UTC timestamp (``YYYY-MM-DD HH:MM:SS``).
        - ``variant`` (str): Severity strategy name.
        - ``instrument`` (str): Ticker symbol.
        - ``signal`` (str): ``'UP'``, ``'DOWN'``, or ``'NEUTRAL'``.
        - ``confidence`` (float): Confidence score in $[0, 1]$.
        - ``instrument_severity`` (float): Signed severity score.
        - ``macro_context`` (str): ``'AGREE'``, ``'CONFLICT'``, or ``'NONE'``.
        - ``n_topics`` (int): Supporting topic count.
        - ``topic_ids`` (list[str] / JSON str on disk): Contributing topic UUIDs.
        - ``rationale`` (str): Human-readable explanation.
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

    def __init__(self, path: str = str(config.paths.logs.signal_log)) -> None:
        """Initialize the signal log backed by the Parquet file at ``path``."""
        self.path = path

    def load(self) -> pd.DataFrame:
        """Load the full signal history with ``topic_ids`` parsed into Python lists."""
        df = self._load_raw()
        if df.empty:
            return df

        for col in _JSON_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(self._from_json)
        return df

    def append(self, rows: pd.DataFrame) -> int:
        """Append new signal rows to the Parquet log.

        Args:
            rows: Signal DataFrame containing :attr:`COLUMNS`.

        Returns:
            Number of rows appended.
        """
        if rows is None or rows.empty:
            return 0

        incoming = rows.reindex(columns=self.COLUMNS)
        for col in _JSON_COLUMNS:
            incoming[col] = incoming[col].apply(self._to_json)

        current = self._load_raw()
        combined = (
            incoming
            if current.empty
            else pd.concat([current, incoming], ignore_index=True)
        )
        self._write_raw(combined)
        return len(incoming)

    def _load_raw(self) -> pd.DataFrame:
        """Read the Parquet file without parsing JSON string columns."""
        return read_raw_parquet(self.path, self.COLUMNS)

    def _write_raw(self, df: pd.DataFrame) -> None:
        """Write raw DataFrame to Parquet, creating parent directories if needed."""
        write_raw_parquet(df, self.path, self.COLUMNS)

    @staticmethod
    def _to_json(value: object) -> str:
        """Serialize a list-like ``topic_ids`` cell to a JSON string."""
        return serialize_json_cell(value, empty_default="[]")

    @staticmethod
    def _from_json(value: object) -> object:
        """Deserialize a JSON string cell into a Python list."""
        return deserialize_json_cell(value, error_default=[])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    logger.info("====== SignalLog Module Test ======")
    log = SignalLog()
    logger.info(f"Log path: {log.path}")
    df = log.load()
    logger.info(f"Logged rows: {len(df)}")
    if not df.empty:
        by_variant = df.groupby("variant")["signal"].value_counts()
        logger.info(f"Signal counts by variant:\n{by_variant.to_string()}")
