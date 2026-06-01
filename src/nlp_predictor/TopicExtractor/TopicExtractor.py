import uuid
from typing import cast

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from nlp_predictor.base_analyzer import BaseAnalyzer
from nlp_predictor.config import Config
from nlp_predictor.prompts import PROMPTS

# Load environment variables (OPENAI_API_KEY) from the project-root .env
load_dotenv(override=True)


# Pydantic schemas for strict, parseable LLM output
class ExtractedTopic(BaseModel):
    """A single market-relevant topic extracted from an article."""

    name: str = Field(
        description="Short, specific event/phenomenon name "
        "(e.g. 'Strait of Hormuz Reopening'). NOT a person or a news outlet."
    )
    description: str = Field(
        description="2-3 sentences describing what is happening, based on the article. "
        "May include a brief, general note on market relevance or likely direction, "
        "but must NOT name specific instruments/tickers or give buy/sell advice."
    )


class ExtractedTopics(BaseModel):
    """Container for the list of topics extracted from one article."""

    topics: list[ExtractedTopic] = Field(
        description="Market-relevant topics found in the article. Empty if none."
    )


class TopicExtractor(BaseAnalyzer):
    """
    Grupa 2 — Topic extraction.

    Reads the cleaned article table produced by Grupa 1 (scraping & preprocessing)
    and, for every article, asks the LLM for a list of market-relevant topics.
    The result is the canonical ``topics`` DataFrame that BOTH the upper path
    (instrument trends) and the lower path (global trends) consume.

    Output contract (one row per topic):
        - topic_id (str)          : unique uuid
        - doc_hash (str)          : FK to the source article (Grupa 1)
        - topic_name (str)        : short topic title
        - topic_description (str) : 2-3 sentence explanation
        - date (datetime/str)     : copied from the source article
    """

    OUTPUT_COLUMNS = [
        "topic_id",
        "doc_hash",
        "topic_name",
        "topic_description",
        "date",
    ]

    # Article columns that carry the publication timestamp, in priority order.
    _DATE_COLUMNS = ("date", "scrape_date", "published_at")

    def process_data(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline entry point (BaseAnalyzer interface). Delegates to
        :meth:`extract_topics`. ``articles_df`` is the Grupa 1 output, not a topics
        table, hence the dedicated alias for readability.
        """
        return self.extract_topics(articles_df)

    def extract_topics(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate over every article and extract its topics.

        Args:
            articles_df (pd.DataFrame): Grupa 1 output. Expected columns include
                ``doc_hash``, ``title``, ``content`` and a date column
                (``date`` or ``scrape_date``).

        Returns:
            pd.DataFrame: The ``topics`` table with :attr:`OUTPUT_COLUMNS`.
        """
        if articles_df.empty:
            print("[INFO] Articles DataFrame is empty. Nothing to extract.")
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        date_col = self._resolve_date_column(articles_df)

        rows: list[dict] = []
        total = len(articles_df)
        for position, (_, article) in enumerate(articles_df.iterrows(), start=1):
            title = str(article.get("title", "")).strip()
            print(f"[INFO] ({position}/{total}) Extracting topics from: {title[:60]}")

            for topic in self._extract_from_article(article):
                rows.append(
                    {
                        "topic_id": str(uuid.uuid4()),
                        "doc_hash": article.get("doc_hash"),
                        "topic_name": topic.name,
                        "topic_description": topic.description,
                        "date": article.get(date_col) if date_col else None,
                    }
                )

        print(f"[INFO] Extracted {len(rows)} topics from {total} articles.")
        return pd.DataFrame(rows, columns=self.OUTPUT_COLUMNS)

    def _resolve_date_column(self, articles_df: pd.DataFrame) -> str | None:
        """Pick the first available date-like column from the article schema."""
        for candidate in self._DATE_COLUMNS:
            if candidate in articles_df.columns:
                return candidate
        print(
            "[WARNING] No date column found in articles "
            f"(looked for {self._DATE_COLUMNS}). 'date' will be empty."
        )
        return None

    def _extract_from_article(self, article: pd.Series) -> list[ExtractedTopic]:
        """
        Call the LLM for a single article and return its parsed topics.

        Uses the whole article (title + full content) as context. Failures for an
        individual article are logged and skipped so one bad article does not abort
        the whole batch.
        """
        title = str(article.get("title", "")).strip()
        content = str(article.get("content", "")).strip()

        if not title and not content:
            return []

        prompt = PROMPTS["topic_extraction"].format(title=title, content=content)

        try:
            parsed = self._call_openai_api_with_retry(prompt, ExtractedTopics)
            return cast(ExtractedTopics, parsed).topics
        except Exception as e:  # noqa: BLE001 - never let one article kill the batch
            print(f"[ERROR] Topic extraction failed for '{title[:50]}': {e}")
            return []


if __name__ == "__main__":
    print("====== TopicExtractor Module Test ======")
    try:
        extractor = TopicExtractor()
        df_articles = pd.read_csv(Config["ARTICLES_PATH"])

        topics = extractor.extract_topics(df_articles)
        print("====== Topic Extraction Completed ======")
        print(topics.to_string(index=False))

        topics.to_csv(Config["TOPICS_SAVE_PATH_CSV"], index=False)
        print(f"[DEBUG] Saved topics to {Config['TOPICS_SAVE_PATH_CSV']}")
    except Exception as e:
        print(f"[FATAL ERROR] TopicExtractor test failed: {e}")
