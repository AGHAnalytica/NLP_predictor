"""
Feature ingestion — the WRITE half of the rolling feature store loop (step 4.5).

The :class:`tick.TickRunner` only ever READS the feature store; this is the piece that
FILLS it. Each tick, fresh articles are scraped, and the expensive, stochastic features
they need downstream — the topic embeddings (lower path) and the topic->instrument
impacts (upper path) — are computed ONCE here and cached, keyed by the article's stable
``doc_hash``. Every later tick the article stays in-window then reuses those cached
features for free, which is the entire reason the store exists.

The cost saver is the dedup guard: :meth:`FeatureStore.known_doc_hashes` tells us which
scraped articles are already cached, so topic extraction / embedding / mapping run only
for genuinely NEW articles. An article seen on 16 consecutive ticks is processed on
exactly one of them.

Pipeline (Group 1 -> Group 2 -> cache)::

    scrape -> drop already-cached doc_hashes -> extract topics (new only)
           -> enrich (publisher + credibility) -> embed + map (the two cached features)
           -> FeatureStore.add(per-topic records)

This module owns only orchestration: it reuses the canonical embedding
(:meth:`LowerPathAnalyzer._generate_embeddings_batch`) and mapping
(:meth:`UpperPathAnalyzer._map_topic_to_instrument`) implementations rather than
re-deriving them, so ingestion and the analytical paths can never drift apart.
"""

from datetime import UTC, datetime

import pandas as pd

from nlp_predictor.config import Config  # type: ignore
from nlp_predictor.preprocessing.feature_store import FeatureStore
from nlp_predictor.global_trends import LowerPathAnalyzer
from nlp_predictor.instrument_trends import UpperPathAnalyzer
from nlp_predictor.topic.topic_extractor import TopicExtractor


class FeatureIngestor:
    """
    Turns freshly scraped articles into cached per-topic feature rows.

    Collaborators are injected so the ingestor stays thin orchestration:
    - ``store``     : the rolling :class:`FeatureStore` it writes into.
    - ``extractor`` : :class:`TopicExtractor`, run only on NEW articles.
    - ``upper``     : :class:`UpperPathAnalyzer`, reused for topic->instrument mapping.
    - ``lower``     : :class:`LowerPathAnalyzer`, reused for batch embeddings.

    Use :meth:`build_default` for the real stack; construct with fakes in tests.
    """

    def __init__(
        self,
        store: FeatureStore,
        extractor: TopicExtractor,
        upper: UpperPathAnalyzer,
        lower: LowerPathAnalyzer,
        weights_path: str = Config["WEIGHTS_PATH"],
    ):
        self.store = store
        self.extractor = extractor
        self.upper = upper
        self.lower = lower
        self.weights_path = weights_path

    @classmethod
    def build_default(cls) -> "FeatureIngestor":
        """Wire the real production stack (needs OPENAI_API_KEY for the LLM pieces)."""
        return cls(
            store=FeatureStore(),
            extractor=TopicExtractor(),
            upper=UpperPathAnalyzer(),
            lower=LowerPathAnalyzer(),
        )

    def ingest(self, articles_df: pd.DataFrame, now: datetime | None = None) -> int:
        """
        Cache the features for every NEW scraped article, skipping the already-cached.

        Args:
            articles_df (pd.DataFrame): Scraped articles (Group 1). Needs ``doc_hash``,
                a publisher column (``source_name`` or ``source``), ``date``, ``title``
                and ``content``.
            now (datetime | None): Reference time for the store's prune step;
                defaults to ``datetime.now(UTC)``.

        Returns:
            int: Number of per-topic rows newly cached (0 if nothing new / no topics).
        """
        if articles_df is None or articles_df.empty:
            print("[INFO] Ingest: no articles scraped; feature store unchanged.")
            return 0
        now = now or datetime.now(UTC)

        articles = articles_df.copy()
        if "source_name" not in articles.columns and "source" in articles.columns:
            articles = articles.rename(columns={"source": "source_name"})

        # 1) Dedup at the article level — the cost saver. Already-cached articles never
        #    re-pay for extraction / embedding / mapping.
        known = self.store.known_doc_hashes()
        fresh = articles[~articles["doc_hash"].astype(str).isin(known)]
        if fresh.empty:
            print(
                f"[INFO] Ingest: all {len(articles)} scraped article(s) already "
                "cached; nothing to compute."
            )
            return 0
        print(
            f"[INFO] Ingest: {len(fresh)} new of {len(articles)} scraped article(s) "
            "need features."
        )

        # 2) Extract topics ONLY for the new articles (the per-article LLM call).
        topics = self.extractor.extract_topics(fresh)
        if topics.empty:
            print("[INFO] Ingest: new articles produced no topics.")
            return 0

        # 3) Attach publisher + credibility (the severity models need both).
        enriched = self._enrich(topics, fresh)

        # 4) Compute the two features the store exists to cache, each once per topic.
        enriched["embedding"] = self._embed(enriched)
        enriched["impacts"] = self._map(enriched)

        # 5) Persist. store.add reindexes to its schema, dedups defensively and prunes.
        self.store.add(enriched, now=now)
        print(
            f"[INFO] Ingest: cached {len(enriched)} topic row(s) from "
            f"{len(fresh)} new article(s)."
        )
        return len(enriched)

    def _enrich(self, topics: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
        """
        Join publisher (``source_name``) and credibility (``source_weight``) on.

        ``source_name`` comes from the source article (by ``doc_hash``);
        ``source_weight`` from the weights table. Unknown sources default to full weight
        so they are never silently dropped, mirroring the upper path's enrichment.
        """
        arts = articles[["doc_hash", "source_name"]].copy()
        out = topics.merge(arts, on="doc_hash", how="left")
        out = out.merge(self._weights(), on="source_name", how="left")
        out["source_weight"] = out["source_weight"].fillna(1.0)
        # The store windows and the DCIM decay both need tz-aware datetimes.
        out["date"] = pd.to_datetime(out["date"], utc=True)
        return out

    def _weights(self) -> pd.DataFrame:
        """Source credibility table, or an empty frame if it is unavailable."""
        try:
            return pd.read_csv(self.weights_path)[["source_name", "source_weight"]]
        except (FileNotFoundError, KeyError, ValueError):
            print("[WARNING] Ingest: source weights unavailable; defaulting to 1.0.")
            return pd.DataFrame(columns=["source_name", "source_weight"])

    def _embed(self, enriched: pd.DataFrame) -> list[list[float]]:
        """Batch-embed each topic (name + description) via the lower path, once."""
        texts = (
            enriched["topic_name"].astype(str)
            + ". "
            + enriched["topic_description"].astype(str)
        ).tolist()
        return self.lower._generate_embeddings_batch(texts)

    def _map(self, enriched: pd.DataFrame) -> list[list[dict]]:
        """Map each topic to its impacted instruments via the upper path."""
        rows = [row for _, row in enriched.iterrows()]
        return self.upper._parallel_map(self.upper._map_topic_to_instrument, rows)


if __name__ == "__main__":
    print("====== FeatureIngestor Module Test ======")
    from dotenv import load_dotenv

    from nlp_predictor.scraping.scrape_runner import run_real_scrapers

    load_dotenv(override=True)
    try:
        scraped = run_real_scrapers()
        n = FeatureIngestor.build_default().ingest(scraped)
        print(f"[INFO] Cached {n} new topic row(s).")
    except Exception as e:
        print(f"[FATAL ERROR] FeatureIngestor test failed: {e}")
