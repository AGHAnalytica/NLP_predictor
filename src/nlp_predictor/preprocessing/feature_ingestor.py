"""Feature ingestion orchestrator populating the rolling FeatureStore for new
articles.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.lower_path.lower_path_analyzer import LowerPathAnalyzer
from nlp_predictor.preprocessing.clean_data import normalize_source_column
from nlp_predictor.preprocessing.feature_store import FeatureStore
from nlp_predictor.preprocessing.topic_extractor import TopicExtractor
from nlp_predictor.upper_path.upper_path_analyzer import UpperPathAnalyzer

logger = logging.getLogger(__name__)


class FeatureIngestor:
    """Computes and caches per-topic features once per article.

    Filters incoming scraped articles against ``FeatureStore.known_doc_hashes``,
    runs topic extraction, batch embedding, and instrument mapping only for unseen
    documents, and appends the resulting records to ``FeatureStore``.
    """

    def __init__(
        self,
        store: FeatureStore,
        extractor: TopicExtractor,
        upper: UpperPathAnalyzer,
        lower: LowerPathAnalyzer,
        weights_path: str = str(config.paths.input.weights),
    ) -> None:
        """Initialize the feature ingestor with pipeline collaborators."""
        self.store = store
        self.extractor = extractor
        self.upper = upper
        self.lower = lower
        self.weights_path = weights_path

    @classmethod
    def build_default(cls) -> "FeatureIngestor":
        """Construct a production :class:`FeatureIngestor` with default dependencies."""
        return cls(
            store=FeatureStore(),
            extractor=TopicExtractor.build_default(),
            upper=UpperPathAnalyzer.build_default(),
            lower=LowerPathAnalyzer.build_default(),
        )

    def ingest(self, articles_df: pd.DataFrame, now: datetime | None = None) -> int:
        """Extract, embed, map, and cache topics for all unseen articles in
        ``articles_df``.

        Args:
            articles_df: Cleaned articles DataFrame with ``doc_hash``, ``source`` or
                ``source_name``, ``date``, ``title``, and ``content``.
            now: Reference UTC timestamp for cache retention pruning.

        Returns:
            Number of newly cached topic rows.
        """
        if articles_df is None or articles_df.empty:
            logger.info("Ingest: no articles scraped; feature store unchanged.")
            return 0
        now = now or datetime.now(UTC)

        articles = normalize_source_column(articles_df)

        known = self.store.known_doc_hashes()
        fresh = articles[~articles["doc_hash"].astype(str).isin(known)]
        if fresh.empty:
            logger.info(
                f"Ingest: all {len(articles)} scraped article(s) already "
                "cached; nothing to compute."
            )
            return 0
        logger.info(
            f"Ingest: {len(fresh)} new of {len(articles)} scraped article(s) "
            "need features."
        )

        topics = self.extractor.extract_topics(fresh)
        if topics.empty:
            logger.info("Ingest: new articles produced no topics.")
            return 0

        enriched = self._enrich(topics, fresh)
        enriched["embedding"] = self._embed(enriched)
        enriched["impacts"] = self._map(enriched)

        self.store.add(enriched, now=now)
        logger.info(
            f"Ingest: cached {len(enriched)} topic row(s) from "
            f"{len(fresh)} new article(s)."
        )
        return len(enriched)

    def _enrich(self, topics: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
        """Join ``source_name`` and ``source_weight`` onto extracted topics."""
        arts = articles[["doc_hash", "source_name"]].copy()
        out = topics.merge(arts, on="doc_hash", how="left")
        out = out.merge(self._weights(), on="source_name", how="left")
        out["source_weight"] = out["source_weight"].fillna(1.0)
        out["date"] = pd.to_datetime(out["date"], utc=True)
        return out

    def _weights(self) -> pd.DataFrame:
        """Load publisher credibility weights CSV, or return an empty DataFrame on
        missing
        file.
        """
        try:
            return pd.read_csv(self.weights_path)[["source_name", "source_weight"]]
        except (FileNotFoundError, KeyError, ValueError):
            logger.warning("Ingest: source weights unavailable; defaulting to 1.0.")
            return pd.DataFrame(columns=["source_name", "source_weight"])

    def _embed(self, enriched: pd.DataFrame) -> list[list[float]]:
        """Batch-embed topics via :class:`LowerPathAnalyzer`."""
        texts = (
            enriched["topic_name"].astype(str)
            + ". "
            + enriched["topic_description"].astype(str)
        ).tolist()
        if hasattr(self.lower, "generate_embeddings_batch"):
            return self.lower.generate_embeddings_batch(texts)
        return self.lower._generate_embeddings_batch(texts)

    def _map(self, enriched: pd.DataFrame) -> list[list[dict[str, Any]]]:
        """Map topics to impacted instruments via :class:`UpperPathAnalyzer`."""
        if hasattr(self.upper, "map_topics"):
            return self.upper.map_topics(enriched)
        rows = enriched.to_dict(orient="records")
        return self.upper._parallel_map(self.upper._map_topic_to_instrument, rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    logger.info("====== FeatureIngestor Module Test ======")
    from dotenv import load_dotenv

    from nlp_predictor.scraping.scrape_runner import run_real_scrapers

    load_dotenv(override=True)
    try:
        scraped = run_real_scrapers()
        n = FeatureIngestor.build_default().ingest(scraped)
        logger.info(f"Cached {n} new topic row(s).")
    except Exception as e:
        logger.error(f"FeatureIngestor test failed: {e}", exc_info=True)
