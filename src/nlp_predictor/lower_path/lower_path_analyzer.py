"""Lower-path analysis: batch topic embeddings, HDBSCAN semantic clustering, and macro
narrative synthesis.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from nlp_predictor.config import config
from nlp_predictor.core.base_analyzer import BaseAnalyzer
from nlp_predictor.core.severity_models import MACRO_SEVERITY_REGISTRY
from nlp_predictor.lower_path.global_trend import GlobalTrend
from nlp_predictor.prompts import PROMPTS

logger = logging.getLogger(__name__)


class LowerPathAnalyzer(BaseAnalyzer):
    """Identifies macroeconomic narratives by clustering topic embeddings with HDBSCAN
    and
    summarizing clusters via LLM.
    """

    def __init__(self, instrument_path: str | None = None):
        """Initialize the lower-path analyzer.

        Args:
            instrument_path: Optional path retained for backward compatibility with
                callers.
        """
        super().__init__()
        self.instrument_path = instrument_path or str(config.paths.input.instruments)

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for ``texts`` in chunks of 100 via the OpenAI
        Embeddings API.

        Args:
            texts: Strings to embed.

        Returns:
            List of embedding vectors aligned 1:1 with ``texts``.
        """
        if not texts:
            return []

        chunk_size = 100
        embeddings: list[list[float]] = []
        model_name = str(config.hyperparams.models.embedding)
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            response = self.client.embeddings.create(input=chunk, model=model_name)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    # Keep private alias for backward compatibility with existing tests/mocks
    _generate_embeddings_batch = generate_embeddings_batch

    def _calculate_severity(self, group: pd.DataFrame) -> float:
        """Compute unsigned narrative intensity for one topic cluster via
        ``MACRO_SEVERITY_REGISTRY``.
        """
        macro_model_name = str(config.hyperparams.lower_path.macro_severity_model)
        model = MACRO_SEVERITY_REGISTRY[macro_model_name]
        params = {"half_life_hours": config.hyperparams.lower_path.half_life_hours}
        return float(model(group, params, self.current_time))

    def process_data(
        self, topics_df: pd.DataFrame, now: datetime | None = None
    ) -> pd.DataFrame:
        """Cluster topics into global macroeconomic trends and synthesize each cluster
        via LLM.

        Reuses cached ``embedding`` vectors when present on ``topics_df`` (live-tick
        path)
        and only calls the OpenAI Embeddings API for uncached batches.

        Args:
            topics_df: Topics DataFrame with ``topic_id``, ``topic_name``,
                ``topic_description``,
                ``date``, and optionally ``embedding`` and ``source_weight``.
            now: Reference UTC timestamp for macro severity time decay.

        Returns:
            DataFrame with columns ``trend_id``, ``current_time``,
            ``first_article_time``,
            ``newest_article_time``, ``cluster_id``, ``title``, ``description``,
            ``severity``,
            ``n_topics``, ``direction``, and ``topic_ids``.
        """
        if topics_df.empty:
            return pd.DataFrame()

        self.current_time = now or datetime.now(UTC)
        topics_df = topics_df.copy()

        min_cluster_size = int(config.hyperparams.lower_path.min_cluster_size)
        if len(topics_df) < min_cluster_size:
            logger.info(
                f"Only {len(topics_df)} topic(s) "
                f"(< min_cluster_size={min_cluster_size}); skipping clustering."
            )
            return pd.DataFrame()

        has_cached = "embedding" in topics_df.columns and all(
            isinstance(v, list) for v in topics_df["embedding"]
        )
        if has_cached:
            logger.info("Reusing cached topic embeddings from the feature store.")
        else:
            logger.info("Requesting embeddings from OpenAI (batch mode)...")
            texts_list = (
                topics_df["topic_name"].astype(str)
                + ". "
                + topics_df["topic_description"].astype(str)
            ).tolist()
            try:
                topics_df["embedding"] = self._generate_embeddings_batch(texts_list)
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return pd.DataFrame()

        logger.info("Clustering global topics using HDBSCAN...")
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            copy=True,
        )
        embeddings_array = np.asarray(topics_df["embedding"].tolist(), dtype=float)
        topics_df["cluster_id"] = clusterer.fit_predict(embeddings_array)

        clustered_df = topics_df[topics_df["cluster_id"] != -1]
        if clustered_df.empty:
            logger.warning("No significant clusters found (all noise).")
            return pd.DataFrame()

        size_threshold = int(config.hyperparams.lower_path.severity_threshold)
        candidates: list[tuple[int, pd.DataFrame, float]] = []
        for cluster_id, group in clustered_df.groupby("cluster_id"):
            group = cast(pd.DataFrame, group)
            if len(group) < size_threshold:
                continue
            severity = self._calculate_severity(group)
            candidates.append((int(cluster_id), group, severity))

        results = self._parallel_map(
            lambda item: self._generate_global_trend_summary(*item), candidates
        )
        return pd.DataFrame(results)

    def _generate_global_trend_summary(
        self, cluster_id: int, group: pd.DataFrame, severity: float
    ) -> dict[str, Any]:
        """Summarize a single topic cluster via the LLM into a global trend record."""
        topics_context = group[["topic_name", "topic_description"]].to_dict(
            orient="records"
        )
        prompt = PROMPTS["global_trends"].format(topics_context=topics_context)

        try:
            parsed_result = self._call_openai_api_with_retry(prompt, GlobalTrend)
            result = parsed_result.model_dump()
        except Exception as e:
            logger.error(f"Trend generation failed for cluster {cluster_id}: {e}")
            result = {
                "title": "Macro aggregation error",
                "description": "Failed to generate summary.",
                "direction": "neutral",
            }

        return {
            "trend_id": str(uuid.uuid4()),
            "current_time": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "first_article_time": str(group["date"].min())[:19],
            "newest_article_time": str(group["date"].max())[:19],
            "cluster_id": int(cluster_id),
            "title": str(result.get("title", "Unknown")),
            "description": str(result.get("description", "Unknown")),
            "severity": round(severity, 2),
            "n_topics": int(len(group)),
            "direction": str(result.get("direction", "neutral")).lower(),
            "topic_ids": group["topic_id"].tolist(),
        }
