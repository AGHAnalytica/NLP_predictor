# Core imports
import os
import uuid
from datetime import UTC, datetime
from typing import cast

import fastparquet  # noqa: F401
import numpy as np
import pandas as pd

# Handle API and LLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sklearn.cluster import HDBSCAN

from nlp_predictor.base_analyzer import BaseAnalyzer  # type: ignore
from nlp_predictor.config import Config  # type: ignore
from nlp_predictor.prompts import PROMPTS  # type: ignore

# Load environment variables from the .env file located in the project root
load_dotenv(override=True)


class GlobalTrend(BaseModel):
    """
    Pydantic model for parsing the output of the global trend summary from OpenAI.
    """

    title: str = Field(description="Short summary of the global macroeconomic trend.")
    description: str = Field(
        description="Detailed explanation of the trend and its global impact."
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")


class LowerPathAnalyzer(BaseAnalyzer):
    """
    Analyzer for identifying global macroeconomic trends by clustering semantically
    similar topics together and generating a macro trend summary for each cluster.
    """

    def __init__(self, instrument_path: str = Config["INSTRUMENT_PATH"]):
        super().__init__()
        self.instrument_path = instrument_path

    def _generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of texts using OpenAI's API in batch mode.
        This is more efficient than making individual API calls for each text.

        Args:
            texts (list[str]): A list of strings for which to generate embeddings.

        Returns:
            list[list[float]]: A list of embedding vectors corresponding to the input
            texts, in the same order as the input list.
        """
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",  # NOTE: Using text-embedding-3-small
            # for cost efficiency; can switch to text-embedding-3-large
            # for better performance if needed
        )
        # Extract the embedding vectors in the exact order of the input texts
        return [embedding.embedding for embedding in response.data]

    def _calculate_severity(self, group: pd.DataFrame) -> float:
        """
        Calculates the severity of a group of topics based on their embeddings.
        For simplicity, this implementation uses the number of topics in the cluster as
        a severity indicator, but this can be enhanced with more complex logic
        (e.g., average distance)

        Args:
            group (pd.DataFrame): A DataFrame containing the topics in a cluster.

        Returns:
            float: A severity score for the cluster, where higher values indicate more
            significant trends.
        """
        return float(len(group))

    def process_data(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the input topics DataFrame to identify global macroeconomic trends.
        This involves generating embeddings, clustering topics, and summarizing trends.

        Args:
            topics_df (pd.DataFrame): DataFrame containing the topics with columns
            ['topic_id', 'topic_name', 'topic_description', 'date'].

        Returns:
            pd.DataFrame: A DataFrame containing the identified global trends with
            columns ['trend_id', 'current_time', 'first_article_time',
            'newest_article_time', 'cluster_id', 'title', 'description', 'severity',
            'direction', 'topic_ids'].
        """
        if topics_df.empty:
            return pd.DataFrame()

        print("[INFO] Preparing texts for embeddings...")

        # 1. Concatenate name and description for rich semantic meaning
        topics_df["text_for_embedding"] = (
            topics_df["topic_name"] + ". " + topics_df["topic_description"]
        )
        print("[INFO] Requesting embeddings from OpenAI (Batch Mode)...")

        # 2. Get embeddings in one single API call
        texts_list = topics_df["text_for_embedding"].tolist()
        try:
            embeddings_list = self._generate_embeddings_batch(texts_list)
            topics_df["embedding"] = embeddings_list
        except Exception as e:
            print(f"[ERROR] Failed to generate embeddings: {e}")
            return pd.DataFrame()

        print("[INFO] Clustering global topics using HDBSCAN...")
        # 3. Apply HDBSCAN Clustering
        # min_cluster_size dictates the minimum number of articles to form a trend
        # TODO - Deftermine optimal min_cluster_size through experimentation
        clusterer = HDBSCAN(
            min_cluster_size=Config["MIN_CLUSTER_SIZE"], metric="euclidean", copy=True
        )

        # Convert list of lists to a 2D numpy array for scikit-learn
        embeddings_array = np.array(topics_df["embedding"].tolist())
        topics_df["cluster_id"] = clusterer.fit_predict(embeddings_array)

        # 4. Filter out noise (HDBSCAN inherently assigns noise points a label of -1)
        clustered_df = topics_df[topics_df["cluster_id"] != -1]

        if clustered_df.empty:
            print(
                "[WARNING] No significant clusters found (all noise)."
                + " Result is empty."
            )
            return pd.DataFrame()

        print("[INFO] Generating macro trends per cluster...")

        # 5. Severity is local/cheap, so filter clusters above the threshold first,
        # then generate each macro summary concurrently (one LLM call per cluster).
        candidates: list[tuple[int, pd.DataFrame, float]] = []
        for cluster_id, group in clustered_df.groupby("cluster_id"):
            group = cast(pd.DataFrame, group)
            severity = self._calculate_severity(group)
            if severity >= Config["SEVERITY_THRESHOLD"]:
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
            print(f"[ERROR] Trend generation failed for cluster {cluster_id}: {e}")
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
            "direction": result.get("direction", "neutral").lower(),
            "topic_ids": topic_ids,
        }


if __name__ == "__main__":
    print("====== LowerPathAnalyzer Module Test ======")
    try:
        lower_path_analyzer = LowerPathAnalyzer(Config["INSTRUMENT_PATH"])
        df_topics = pd.read_csv(Config["TOPICS_PATH"])

        result = lower_path_analyzer.process_data(df_topics)
        print("====== Processing Lower Path Completed ======")

        print("====== Saving Results To File ======")
        saving_result = LowerPathAnalyzer.save_trends(
            result, Config["GLOBAL_TRENDS_SAVE_PATH_PARQUET"]
        )
        if saving_result:
            print("====== Saving Completed ======")
        else:
            print("====== Saving Failed ======")

        # For debugging: save the result to a CSV as well
        # (not required, but useful for quick checks)
        result.to_csv(Config["GLOBAL_TRENDS_SAVE_PATH_CSV"], index=False)
        print(
            "[DEBUG] Also saved results to CSV for"
            + f" quick inspection at {Config['GLOBAL_TRENDS_SAVE_PATH_CSV']}"
        )
    except Exception as e:
        print(f"[FATAL ERROR] An error occurred during the LowerPathAnalyzer test: {e}")
