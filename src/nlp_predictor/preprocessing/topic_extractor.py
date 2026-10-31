"""LLM-powered extraction of market-relevant topics from cleaned articles."""

import logging
import uuid
from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.core.base_analyzer import BaseAnalyzer
from nlp_predictor.preprocessing.topic import ExtractedTopic, ExtractedTopics
from nlp_predictor.prompts import PROMPTS

logger = logging.getLogger(__name__)


class TopicExtractor(BaseAnalyzer):
    """Extracts structured market topics from cleaned articles concurrently via LLM.

    Output schema (:attr:`OUTPUT_COLUMNS`):
        - ``topic_id`` (str): UUID4 identifier.
        - ``doc_hash`` (str): Source article SHA-256 hash.
        - ``topic_name`` (str): Concise event title.
        - ``topic_description`` (str): 2–3 sentence factual explanation.
        - ``date`` (Any): Publication timestamp copied from the source article.
    """

    OUTPUT_COLUMNS = [
        "topic_id",
        "doc_hash",
        "topic_name",
        "topic_description",
        "date",
    ]

    _DATE_COLUMNS = ("date", "scrape_date", "published_at")

    def process_data(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """BaseAnalyzer pipeline entry point; delegates to :meth:`extract_topics`."""
        return self.extract_topics(articles_df)

    def extract_topics(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """Extract market topics from every article in parallel.

        Args:
            articles_df: Cleaned articles DataFrame containing ``doc_hash``, ``title``,
                ``content``, and a publication timestamp column.

        Returns:
            DataFrame of extracted topics conforming to :attr:`OUTPUT_COLUMNS`.
        """
        if articles_df.empty:
            logger.info("Articles DataFrame is empty. Nothing to extract.")
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        date_col = self._resolve_date_column(articles_df)
        articles = articles_df.to_dict(orient="records")

        per_article_rows = self._parallel_map(
            lambda article: self._article_to_rows(article, date_col), articles
        )
        rows = [row for article_rows in per_article_rows for row in article_rows]

        logger.info(f"Extracted {len(rows)} topics from {len(articles)} articles.")
        return pd.DataFrame(rows, columns=self.OUTPUT_COLUMNS)

    def _article_to_rows(
        self, article: Mapping[str, Any], date_col: str | None
    ) -> list[dict[str, Any]]:
        """Extract topics for a single article and format as output rows."""
        return [
            {
                "topic_id": str(uuid.uuid4()),
                "doc_hash": article.get("doc_hash"),
                "topic_name": topic.name,
                "topic_description": topic.description,
                "date": article.get(date_col) if date_col else None,
            }
            for topic in self._extract_from_article(article)
        ]

    def _resolve_date_column(self, articles_df: pd.DataFrame) -> str | None:
        """Return the first matching timestamp column from :attr:`_DATE_COLUMNS`."""
        for candidate in self._DATE_COLUMNS:
            if candidate in articles_df.columns:
                return candidate
        logger.warning(
            "No date column found in articles "
            f"(looked for {self._DATE_COLUMNS}). 'date' will be empty."
        )
        return None

    def _extract_from_article(self, article: Mapping[str, Any]) -> list[ExtractedTopic]:
        """Query the LLM for a single article and return validated topic models."""
        title = str(article.get("title", "")).strip()
        content = str(article.get("content", "")).strip()

        if not title and not content:
            return []

        logger.info(f"Extracting topics from: {title[:60]}")
        prompt = PROMPTS["topic_extraction"].format(title=title, content=content)

        try:
            parsed = self._call_openai_api_with_retry(prompt, ExtractedTopics)
            return cast(ExtractedTopics, parsed).topics
        except Exception as e:  # noqa: BLE001
            logger.error(f"Topic extraction failed for '{title[:50]}': {e}")
            return []


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    logger.info("====== TopicExtractor Module Test ======")
    try:
        extractor = TopicExtractor()
        df_articles = pd.read_csv(config.paths.input.articles)

        topics = extractor.extract_topics(df_articles)
        logger.info("====== Topic Extraction Completed ======")
        logger.info(f"\n{topics.to_string(index=False)}")

        topics.to_csv(config.paths.output.topics_csv, index=False)
        logger.debug(f"Saved topics to {config.paths.output.topics_csv}")
    except Exception as e:
        logger.error(f"TopicExtractor test failed: {e}", exc_info=True)
