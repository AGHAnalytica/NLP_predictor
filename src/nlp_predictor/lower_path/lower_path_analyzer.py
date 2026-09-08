import logging
import uuid
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from nlp_predictor.base_analyzer import BaseAnalyzer
from nlp_predictor.config import config
from nlp_predictor.lower_path.global_trend import GlobalTrend
from nlp_predictor.lower_path.severity_models import MACRO_SEVERITY_REGISTRY
from nlp_predictor.prompts import PROMPTS

logger = logging.getLogger(__name__)


class LowerPathAnalyzer(BaseAnalyzer):
    """
    Analyzer for identifying global macroeconomic trends by clustering semantically
    similar topics together and generating a macro trend summary for each cluster.
    """

    def __init__(self, instrument_path: str = str(config.paths.input.instruments)):
        super().__init__()
        self.instrument_path = instrument_path

    def _generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of texts using OpenAI's API in batch mode.
        Automatically chunks requests into batches of 100 to stay within API limits.

        Args:
            texts (list[str]): A list of strings for which to generate embeddings.

        Returns:
            list[list[float]]: A list of embedding vectors corresponding to the input
            texts, in the same order as the input list.
        """
        if not texts:
            return []

        chunk_size = 100
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            response = self.client.embeddings.create(
                input=chunk,
                model=str(config.hyperparams.models.embedding),
            )
            embeddings.extend([embedding.embedding for embedding in response.data])
        return embeddings

    def _calculate_severity(self, group: pd.DataFrame) -> float:
        """
        Score a cluster's macro severity — its UNSIGNED narrative intensity (the weight
        the climate conviction leans on), not its direction.

        Dispatches to the model named by
        ``config.hyperparams.lower_path.macro_severity_model``
        from :data:`severity_models.MACRO_SEVERITY_REGISTRY` (``count`` or the default
        ``decayed_weighted`` = source-weighted, time-decayed volume), so the macro
        severity is an A/B knob just like the upper-path severity. Recency uses
        ``self.current_time`` (the tick clock), keeping it consistent with the article
        dates the cluster is built from.

        Args:
            group (pd.DataFrame): The topics in one cluster (carrying ``date`` and
                ``source_weight`` for the weighted models).

        Returns:
            float: A non-negative intensity; larger = a bigger / fresher / more credible
            narrative.
        """
        macro_model_name = str(config.hyperparams.lower_path.macro_severity_model)
        model = MACRO_SEVERITY_REGISTRY[macro_model_name]
        params = {"half_life_hours": config.hyperparams.lower_path.half_life_hours}
        return model(group, params, self.current_time)

    def process_data(
        self, topics_df: pd.DataFrame, now: datetime | None = None
    ) -> pd.DataFrame:
        """
        Processes the input topics DataFrame to identify global macroeconomic trends.
        This involves generating embeddings, clustering topics, and summarizing trends.

        Args:
            topics_df (pd.DataFrame): DataFrame containing the topics with columns
                ``['topic_id', 'topic_name', 'topic_description', 'date']`` (plus
                ``'source_weight'`` for the weighted macro severity).
            now (datetime | None): Reference time for the macro-severity time decay;
                defaults to ``datetime.now(UTC)``. The tick passes its own clock so the
                decay is consistent with the article dates the clusters are built from.

        Returns:
            pd.DataFrame: A DataFrame containing the identified global trends with
                columns ``['trend_id', 'current_time', 'first_article_time',
                'newest_article_time', 'cluster_id', 'title', 'description',
                'severity', 'n_topics', 'direction', 'topic_ids']``.
        """
        if topics_df.empty:
            return pd.DataFrame()

        self.current_time = now or datetime.now(UTC)
        topics_df = topics_df.copy()

        # HDBSCAN needs at least MIN_CLUSTER_SIZE samples; fewer cannot form a cluster
        # anyway. A live tick's window can be that thin, so short-circuit instead of
        # letting the clusterer raise (and skip a pointless embedding call).
        min_cluster_size = int(config.hyperparams.lower_path.min_cluster_size)
        if len(topics_df) < min_cluster_size:
            logger.info(
                f"Only {len(topics_df)} topic(s) (< MIN_CLUSTER_SIZE "
                f"{min_cluster_size}); no macro clusters."
            )
            return pd.DataFrame()

        # Reuse the embeddings cached in the feature store when present: every cached
        # topic row carries its vector, so on a live tick the lower path must NOT re-pay
        # the embedding API for articles it has already embedded. Only an uncached batch
        # (e.g. the legacy single-pass run) hits the API.
        has_cached = (
            "embedding" in topics_df.columns
            and topics_df["embedding"].apply(lambda v: isinstance(v, list)).all()
        )
        if has_cached:
            logger.info("Reusing cached topic embeddings from the feature store.")
        else:
            logger.info("Requesting embeddings from OpenAI (Batch Mode)...")
            # Concatenate name + description for rich semantic meaning, then embed in
            # one single API call.
            topics_df["text_for_embedding"] = (
                topics_df["topic_name"] + ". " + topics_df["topic_description"]
            )
            texts_list = topics_df["text_for_embedding"].tolist()
            try:
                topics_df["embedding"] = self._generate_embeddings_batch(texts_list)
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return pd.DataFrame()

        logger.info("Clustering global topics using HDBSCAN...")
        # 3. Apply HDBSCAN Clustering
        # min_cluster_size dictates the minimum number of articles to form a trend
        # TODO - Deftermine optimal min_cluster_size through experimentation
        clusterer = HDBSCAN(
            min_cluster_size=config.hyperparams.lower_path.min_cluster_size,
            metric="euclidean",
            copy=True,
        )

        # Convert list of lists to a 2D numpy array for scikit-learn
        embeddings_array = np.array(topics_df["embedding"].tolist())
        topics_df["cluster_id"] = clusterer.fit_predict(embeddings_array)

        # 4. Filter out noise (HDBSCAN inherently assigns noise points a label of -1)
        clustered_df = topics_df[topics_df["cluster_id"] != -1]

        if clustered_df.empty:
            logger.warning(
                "No significant clusters found (all noise). Result is empty."
            )
            return pd.DataFrame()

        logger.info("Generating macro trends per cluster...")

        # 5. Severity is local/cheap. Gate clusters on SIZE (raw topic count) — a
        # membership test kept separate from the weighted macro severity, so a small but
        # fresh/credible cluster is not filtered out by a low conviction weight. Then
        # generate each surviving macro summary concurrently (one LLM call per cluster).
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
    ) -> dict:
        """
        Generates a summary for a given cluster of topics by calling the OpenAI API.
        The summary includes a title, description, and direction of the trend.

        Args:
            cluster_id (int): The ID of the cluster being summarized.
            group (pd.DataFrame): A DataFrame containing the topics in the cluster.
            severity (float): The calculated severity score for the cluster.

        Returns:
            dict: A dictionary containing the trend summary and related metadata.
        """
        topics_context = group[["topic_name", "topic_description"]].to_dict(
            orient="records"
        )
        topic_ids = group["topic_id"].tolist()

        first_article_time = str(group["date"].min())[:19]
        newest_article_time = str(group["date"].max())[:19]

        prompt = PROMPTS["global_trends"].format(
            topics_context=topics_context,
        )

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
            "first_article_time": first_article_time,
            "newest_article_time": newest_article_time,
            "cluster_id": int(cluster_id),
            "title": result.get("title", "Unknown"),
            "description": result.get("description", "Unknown"),
            "severity": round(severity, 2),
            "n_topics": int(len(group)),
            "direction": result.get("direction", "neutral").lower(),
            "topic_ids": topic_ids,
        }
