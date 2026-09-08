"""
Rolling feature store — the persistence layer that makes the 15-min A/B/n loop cheap.

The expensive parts of the pipeline are the per-article LLM calls (topic extraction,
topic->instrument mapping) and the per-topic embedding call. In a stateless run those
are recomputed from scratch every time. Once we move to a 15-min cadence over a rolling
window, the SAME article sits inside the window for ~16 consecutive ticks — recomputing
its features 16 times would be pure waste (and cost).

This store fixes that. Each article is processed exactly ONCE: its topics, their
embeddings and their instrument impacts are cached here, keyed by the article's stable
``doc_hash``. Every tick then just:

    1. asks :meth:`known_doc_hashes` which scraped articles are already cached (skip),
    2. computes features only for the genuinely new ones and :meth:`add`-s them,
    3. reads back the last N hours with :meth:`window` and recomputes severity.

Severity recomputation is cheap and deterministic, so the same cached window can be fed
through any number of severity models / window lengths and compared fairly — only the
model differs, the inputs are identical. That is the whole point of the store.

Windowing and pruning are by the article's **publication date** (news recency), the
same clock the DCIM time-decay uses — not by when we happened to scrape it.
``SIGNAL_WINDOW_HOURS`` is the per-tick lookback (an A/B knob);
``FEATURE_STORE_MAX_WINDOW_HOURS`` is the longest window we retain so no A/B variant is
ever starved of data.

On-disk schema (one row per topic, Parquet):
    - doc_hash (str)            : FK to the source article — the cache/dedup key
    - topic_id (str)            : unique topic id
    - topic_name (str)          : short topic title
    - topic_description (str)   : 2-3 sentence explanation
    - date (datetime, UTC)      : article publication time — drives window + decay
    - source_name (str)         : publisher
    - source_weight (float)     : publisher credibility
    - embedding (JSON str)      : the topic embedding vector (lower path)
    - impacts (JSON str)        : list of {instrument, sentiment} (upper path)
    - cached_at (datetime, UTC) : when this row was computed
"""

import json
import logging
import os
from datetime import UTC, datetime

import pandas as pd

from nlp_predictor.config import config

logger = logging.getLogger(__name__)

# Columns serialized to JSON on disk and parsed back into Python objects in memory.
_JSON_COLUMNS = ("embedding", "impacts")


class FeatureStore:
    """
    Parquet-backed rolling cache of per-topic features (embedding + instrument impacts).

    Deliberately a thin, dependency-free data layer: load / add / window / prune plus a
    dedup guard. It does NOT call the LLM or compute severity — orchestration owns that.
    Keeping it pure makes it trivial to unit-test against synthetic frames and keeps the
    expensive, stochastic work out of the part of the system that has to be reliable.
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
    ):
        """
        Args:
            path (str): Parquet file backing the store. Created lazily on first
                :meth:`add`; a missing file simply reads back as an empty store.
            max_window_hours (float): Longest retention window. Rows whose article
                ``date`` is older than this (relative to now) are dropped on every
                :meth:`add`, since they can never re-enter any active signal window.
        """
        self.path = path
        self.max_window_hours = float(max_window_hours)

    # ============================== READ ==============================
    def load(self) -> pd.DataFrame:
        """
        Read the whole store into memory with JSON columns parsed back into objects.

        Returns:
            pd.DataFrame: All cached rows with :attr:`COLUMNS` (``embedding`` as a
            ``list[float]`` and ``impacts`` as a ``list[dict]``). An empty, correctly
            typed frame if the store does not exist yet.
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
        """
        Return the set of article hashes already cached, so the caller can skip the
        expensive feature computation for them. This is the primary dedup guard — an
        article is processed once and reused across every tick it remains in-window for.

        Returns:
            set[str]: Distinct ``doc_hash`` values currently in the store.
        """
        df = self.load()
        if df.empty:
            return set()
        return set(df["doc_hash"].dropna().astype(str))

    def window(self, hours: float, now: datetime | None = None) -> pd.DataFrame:
        """
        Return the cached rows whose article ``date`` falls within the last ``hours``.

        This is the per-tick read that feeds the severity models. Filtering by article
        publication time (not scrape time) keeps it consistent with the DCIM time-decay,
        and lets different A/B variants ask for different horizons off the same store.

        Args:
            hours (float): Lookback window in hours
                (e.g. ``config.hyperparams.windows.signal_hours``).
            now (datetime | None): Reference time; defaults to ``datetime.now(UTC)``.

        Returns:
            pd.DataFrame: The in-window subset of :meth:`load`, index-reset.
        """
        df = self.load()
        if df.empty:
            return df
        now = now or datetime.now(UTC)
        cutoff = now - pd.Timedelta(hours=float(hours))
        return df[df["date"] >= cutoff].reset_index(drop=True)

    # ============================== WRITE ==============================
    def add(self, records: pd.DataFrame, now: datetime | None = None) -> None:
        """
        Append newly computed topic features, then prune anything past the max window.

        Idempotent at the article level: any incoming row whose ``doc_hash`` is already
        cached is dropped, so a re-scraped or double-submitted article never inflates
        the store (and never gets re-counted in a severity sum). The caller should still
        use :meth:`known_doc_hashes` upstream to avoid paying for the LLM call at all;
        this is the defensive backstop.

        Args:
            records (pd.DataFrame): New rows with (at least) :attr:`COLUMNS` minus
                ``cached_at`` (added here if absent). ``embedding``/``impacts`` may be
                Python objects or already-JSON strings.
            now (datetime | None): Reference time for the prune step; defaults to
                ``datetime.now(UTC)``.
        """
        if records is None or records.empty:
            return

        now = now or datetime.now(UTC)
        incoming = records.copy()

        if "cached_at" not in incoming.columns:
            incoming["cached_at"] = now
        incoming["date"] = pd.to_datetime(incoming["date"], utc=True)
        incoming["cached_at"] = pd.to_datetime(incoming["cached_at"], utc=True)

        # Drop articles we already hold (idempotent re-runs).
        existing = self.known_doc_hashes()
        if existing:
            incoming = incoming[~incoming["doc_hash"].astype(str).isin(existing)]
        if incoming.empty:
            return

        incoming = incoming.reindex(columns=self.COLUMNS)
        for col in _JSON_COLUMNS:
            incoming[col] = incoming[col].apply(self._to_json)

        current = self._load_raw()
        combined = pd.concat([current, incoming], ignore_index=True)
        combined = self._prune_raw(combined, now)
        self._write_raw(combined)

    def prune(self, now: datetime | None = None) -> None:
        """
        Drop every row older than ``max_window_hours`` and persist the result.

        Normally :meth:`add` prunes for you; this exists for a standalone sweep (e.g. a
        tick with no new articles that still wants to age out stale ones).

        Args:
            now (datetime | None): Reference time; defaults to ``datetime.now(UTC)``.
        """
        now = now or datetime.now(UTC)
        self._write_raw(self._prune_raw(self._load_raw(), now))

    # ============================== TRANSFORM ==============================
    @staticmethod
    def to_instrument_frame(window_df: pd.DataFrame) -> pd.DataFrame:
        """
        Explode cached topics into the per-(topic, instrument) rows the severity models
        consume. Each topic's ``impacts`` list becomes one row per impacted instrument,
        carrying the columns the severity models need (``date``, ``source_name``,
        ``sentiment``, ``source_weight``) plus ``instrument`` and ``topic_id`` for
        grouping and topic-id aggregation downstream.

        Args:
            window_df (pd.DataFrame): Output of :meth:`window` (or :meth:`load`).

        Returns:
            pd.DataFrame: Long-form rows with columns ``instrument``, ``sentiment``,
            ``date``, ``source_name``, ``source_weight``, ``topic_id``. Empty (with that
            schema) if there are no impacts in the window.
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
        for _, r in window_df.iterrows():
            impacts = r["impacts"] if isinstance(r["impacts"], list) else []
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

    # ============================== INTERNALS ==============================
    def _load_raw(self) -> pd.DataFrame:
        """Read the store without parsing JSON columns (write-path helper)."""
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.COLUMNS)
        df = pd.read_parquet(self.path)
        if df.empty:
            return pd.DataFrame(columns=self.COLUMNS)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["cached_at"] = pd.to_datetime(df["cached_at"], utc=True)
        return df

    def _prune_raw(self, df: pd.DataFrame, now: datetime) -> pd.DataFrame:
        """Drop rows whose article date is older than the max retention window."""
        if df.empty:
            return df
        cutoff = now - pd.Timedelta(hours=self.max_window_hours)
        return df[df["date"] >= cutoff].reset_index(drop=True)

    def _write_raw(self, df: pd.DataFrame) -> None:
        """Persist the store, creating the parent directory on first write."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        df.reindex(columns=self.COLUMNS).to_parquet(self.path, index=False)

    @staticmethod
    def _to_json(value: object) -> str:
        """Serialize an embedding/impacts cell to JSON (idempotent on strings)."""
        if isinstance(value, str):
            return value
        if value is None:
            return "null"
        return json.dumps(value)

    @staticmethod
    def _from_json(value: object) -> object:
        """Parse a JSON string cell into a Python object; pass non-strings through."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return None
        return value


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )  # noqa: E501
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
