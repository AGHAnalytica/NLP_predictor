# Core imports
import uuid
from datetime import UTC, datetime
from pathlib import Path

import fastparquet  # noqa: F401
import numpy as np
import pandas as pd
from base_analyzer import BaseAnalyzer  # type: ignore

# Handle API and LLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sklearn.cluster import HDBSCAN

# Load environment variables from the .env file located in the project root
load_dotenv()

# Default instrument path (resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parents[2]
INSTRUMENT_PATH = str(BASE_DIR / "data" / "instruments" / "instruments_sample.csv")
TOPICS_PATH = str(BASE_DIR / "data" / "topics" / "topics_sample_large.csv")
SAVE_PATH_CSV = str(BASE_DIR / "data" / "test_results" / "global_trends_report.csv")
SAVE_PATH_PARQUET = str(BASE_DIR / "data" / "test_results" / "global_trends_parquet")

# Minimum number of topics in a cluster to be considered a valid trend
SEVERITY_THRESHOLD = 2


class GlobalTrend(BaseModel):
    """ """

    title: str = Field(description="Short summary of the global macroeconomic trend.")
    description: str = Field(
        description="Detailed explanation of the trend and its global impact."
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")


class LowerPathAnalyzer(BaseAnalyzer):
    """_summary_"""

    def __init__(self, instrument_path: str = INSTRUMENT_PATH):
        super().__init__()
        self.instrument_path = instrument_path

    def _generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """ """
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
        """
        return float(len(group))

    def process_data(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """ """
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
        clusterer = HDBSCAN(min_cluster_size=3, metric="euclidean", copy=True)

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

        results = []
        print("[INFO] Generating macro trends per cluster...")

        # 5. Group by clusters and generate a macro trend summary for each cluster group
        for cluster_id, group in clustered_df.groupby("cluster_id"):
            severity = self._calculate_severity(group)
            if severity >= SEVERITY_THRESHOLD:  # Skip clusters that are too small
                trend_data = self._generate_global_trend_summary(
                    cluster_id,
                    group,
                    severity,
                )
                results.append(trend_data)
        return pd.DataFrame(results)

    def _generate_global_trend_summary(
        self, cluster_id: int, group: pd.DataFrame, severity: float
    ) -> dict:
        """ """
        topics_context = group[["topic_name", "topic_description"]].to_dict(
            orient="records"
        )
        topic_ids = group["topic_id"].tolist()

        first_article_time = str(group["date"].min())[:19]
        newest_article_time = str(group["date"].max())[:19]

        prompt = f"""
        You are a Chief Economist. Analyze the following cluster of news topics.
        These topics were mathematically clustered together due to their semantic
        similarity. Determine the overarching global macroeconomic narrative and its
        overall direction ('up', 'down', or 'neutral'). Provide a concise title
        and a 1-2 sentence description explaining the global impact.

        Cluster Topics Context:
        {topics_context}
        """

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
        lower_path_analyzer = LowerPathAnalyzer(INSTRUMENT_PATH)
        df_topics = pd.read_csv(TOPICS_PATH)

        result = lower_path_analyzer.process_data(df_topics)
        print("====== Processing Lower Path Completed ======")

        print("====== Saving Results To File ======")
        saving_result = LowerPathAnalyzer.save_trends(result, SAVE_PATH_PARQUET)
        if saving_result:
            print("====== Saving Completed ======")
        else:
            print("====== Saving Failed ======")

        # For debugging: save the result to a CSV as well
        # (not required, but useful for quick checks)
        result.to_csv(SAVE_PATH_CSV, index=False)
        print(
            "[DEBUG] Also saved results to CSV for"
            + f" quick inspection at {SAVE_PATH_CSV}"
        )
    except Exception as e:
        print(f"[FATAL ERROR] An error occurred during the LowerPathAnalyzer test: {e}")
