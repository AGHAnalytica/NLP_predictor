"""Parquet-backed rolling cache for per-topic embeddings and instrument impacts."""

import logging
import os
from datetime import UTC, datetime

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.core.parquet_io import (
    deserialize_json_cell,
    read_raw_parquet,
    serialize_json_cell,
    write_raw_parquet,
)

logger = logging.getLogger(__name__)

_JSON_COLUMNS = ("embedding", "impacts")


class FeatureStore:
    """Rolling Parquet cache of per-topic features keyed by article ``doc_hash``.

    Caches the expensive LLM/embedding outputs (``embedding`` and ``impacts``) once per
    article so subsequent 15-minute ticks can slice any lookback window and recompute
    deterministic severity scores without repeating API calls.

    Schema (:attr:`COLUMNS`):
        - ``doc_hash`` (str): Source article hash (deduplication key).
        - ``topic_id`` (str): Unique topic UUID.
        - ``topic_name`` (str): Short topic title.
        - ``topic_description`` (str): Topic explanation.
        - ``date`` (datetime, UTC): Article publication timestamp.
        - ``source_name`` (str): Publisher identifier.
        - ``source_weight`` (float): Publisher credibility weight.
        - ``embedding`` (list[float] / JSON str on disk): Topic embedding vector.
        - ``impacts`` (list[dict] / JSON str on disk): ``{instrument, sentiment}``
          mappings.
        - ``cached_at`` (datetime, UTC): Cache insertion timestamp.
    """

    COLUMNS = [
        "doc_hash",
        "topic_id",
        "topic_name",
        "topic_description",
        "date",
        "source_name",
        "source_weight",
        "embedding",
        "impacts",
        "cached_at",
    ]

    def __init__(
        self,
        path: str = str(config.paths.cache.feature_store),
        max_window_hours: float = config.hyperparams.windows.feature_store_max_hours,
    ) -> None:
        """Initialize the feature store.

        Args:
            path: Parquet file path backing the store.
            max_window_hours: Maximum retention horizon in hours based on article
                ``date``.
        """
        self.path = path
        self.max_window_hours = float(max_window_hours)

    def load(self) -> pd.DataFrame:
        """Load all cached rows from disk with JSON columns deserialized into Python
        lists.
        """
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.COLUMNS)

        df = pd.read_parquet(self.path)
        if df.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        for col in _JSON_COLUMNS:
            df[col] = df[col].apply(self._from_json)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["cached_at"] = pd.to_datetime(df["cached_at"], utc=True)
        return df

    def known_doc_hashes(self) -> set[str]:
        """Return distinct cached ``doc_hash`` values without deserializing JSON
        columns.
        """
        if not os.path.exists(self.path):
            return set()
        try:
            df = pd.read_parquet(self.path, columns=["doc_hash"])
        except Exception:
            df = self._load_raw()
        if df.empty or "doc_hash" not in df.columns:
            return set()
        return set(df["doc_hash"].dropna().astype(str))

    def window(self, hours: float, now: datetime | None = None) -> pd.DataFrame:
        """Return cached topic rows whose article ``date`` is within the last ``hours``.

        Args:
            hours: Lookback horizon in hours.
            now: Reference UTC timestamp (defaults to ``datetime.now(UTC)``).

        Returns:
            Filtered DataFrame with deserialized ``embedding`` and ``impacts`` columns.
        """
        df = self.load()
        if df.empty:
            return df
        now = now or datetime.now(UTC)
        cutoff = now - pd.Timedelta(hours=float(hours))
        return df[df["date"] >= cutoff].reset_index(drop=True)

    def add(self, records: pd.DataFrame, now: datetime | None = None) -> None:
        """Append new topic rows, deduplicate by ``doc_hash``, and prune expired rows.

        Args:
            records: New topic feature rows.
            now: Reference UTC timestamp for pruning and ``cached_at``.
        """
        if records is None or records.empty:
            return

        now = now or datetime.now(UTC)
        incoming = records.copy()

        if "cached_at" not in incoming.columns:
            incoming["cached_at"] = now
        incoming["date"] = pd.to_datetime(incoming["date"], utc=True)
        incoming["cached_at"] = pd.to_datetime(incoming["cached_at"], utc=True)

        current = self._load_raw()
        if not current.empty:
            existing = set(current["doc_hash"].dropna().astype(str))
            incoming = incoming[~incoming["doc_hash"].astype(str).isin(existing)]
        if incoming.empty:
            return

        incoming = incoming.reindex(columns=self.COLUMNS)
        for col in _JSON_COLUMNS:
            incoming[col] = incoming[col].apply(self._to_json)

        combined = (
            incoming
            if current.empty
            else pd.concat([current, incoming], ignore_index=True)
        )
        combined = self._prune_raw(combined, now)
        self._write_raw(combined)

    def prune(self, now: datetime | None = None) -> None:
        """Remove rows older than :attr:`max_window_hours` and persist."""
        now = now or datetime.now(UTC)
        self._write_raw(self._prune_raw(self._load_raw(), now))

    @staticmethod
    def to_instrument_frame(window_df: pd.DataFrame) -> pd.DataFrame:
        """Explode topic ``impacts`` into long-form per-(topic, instrument) rows.

        Args:
            window_df: Topic DataFrame from :meth:`window` or :meth:`load`.

        Returns:
            DataFrame with columns ``[instrument, sentiment, date, source_name,
            source_weight, topic_id]``.
        """
        out_cols = [
            "instrument",
            "sentiment",
            "date",
            "source_name",
            "source_weight",
            "topic_id",
        ]
        if window_df is None or window_df.empty:
            return pd.DataFrame(columns=out_cols)

        rows: list[dict] = []
        for r in window_df.to_dict(orient="records"):
            impacts = r.get("impacts")
            if not isinstance(impacts, list):
                continue
            for impact in impacts:
                rows.append(
                    {
                        "instrument": str(impact["instrument"]),
                        "sentiment": float(impact["sentiment"]),
                        "date": r["date"],
                        "source_name": r["source_name"],
                        "source_weight": float(r["source_weight"]),
                        "topic_id": r["topic_id"],
                    }
                )
        return pd.DataFrame(rows, columns=out_cols)

    def _load_raw(self) -> pd.DataFrame:
        """Read the Parquet store without parsing JSON string columns."""
        df = read_raw_parquet(self.path, self.COLUMNS)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["cached_at"] = pd.to_datetime(df["cached_at"], utc=True)
        return df

    def _prune_raw(self, df: pd.DataFrame, now: datetime) -> pd.DataFrame:
        """Filter out rows whose ``date`` is older than :attr:`max_window_hours`."""
        if df.empty:
            return df
        cutoff = now - pd.Timedelta(hours=self.max_window_hours)
        return df[df["date"] >= cutoff].reset_index(drop=True)

    def _write_raw(self, df: pd.DataFrame) -> None:
        """Write raw DataFrame to Parquet, creating parent directories if needed."""
        write_raw_parquet(df, self.path, self.COLUMNS)

    @staticmethod
    def _to_json(value: object) -> str:
        """Serialize a list/dict cell to a JSON string."""
        return serialize_json_cell(value, empty_default="null")

    @staticmethod
    def _from_json(value: object) -> object:
        """Deserialize a JSON string cell into a Python object."""
        return deserialize_json_cell(value, error_default=None)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    logger.info("====== FeatureStore Module Test ======")
    store = FeatureStore()
    logger.info(f"Store path: {store.path}")
    logger.info(f"Max retention: {store.max_window_hours}h")
    df = store.load()
    logger.info(f"Cached topic rows: {len(df)}")
    if not df.empty:
        win = store.window(config.hyperparams.windows.signal_hours)
        logger.info(
            f"In last {config.hyperparams.windows.signal_hours}h: {len(win)} topics -> "
            f"{len(FeatureStore.to_instrument_frame(win))} instrument impacts."
        )
