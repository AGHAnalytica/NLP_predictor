import uuid
from typing import cast

import pandas as pd
from dotenv import load_dotenv

from nlp_predictor.base_analyzer import BaseAnalyzer
from nlp_predictor.config import Config
from nlp_predictor.preprocessing.topic.topic import ExtractedTopic, ExtractedTopics
from nlp_predictor.prompts import PROMPTS

# Load environment variables (OPENAI_API_KEY) from the project-root .env
load_dotenv(override=True)


class TopicExtractor(BaseAnalyzer):
    """
    Group 2 — Topic extraction.

    Reads the cleaned article table produced by Group 1 (scraping & preprocessing)
    and, for every article, asks the LLM for a list of market-relevant topics.
    The result is the canonical ``topics`` DataFrame that BOTH the upper path
    (instrument trends) and the lower path (global trends) consume.

    Output contract (one row per topic):
        - topic_id (str)          : unique uuid
        - doc_hash (str)          : FK to the source article (Group 1)
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
        :meth:`extract_topics`. ``articles_df`` is the Group 1 output, not a topics
        table, hence the dedicated alias for readability. The main logic is in
        ``extract_topics()``, which iterates over the articles and calls
        ``_extract_from_article()`` for each one to get its topics. The resulting
        list of topics is then flattened and converted into a DataFrame with the
        specified output columns. If the input DataFrame is empty, it
        returns an empty topics DataFrame with the correct schema.

        Args:
            articles_df (pd.DataFrame): The DataFrame containing cleaned articles with
            columns like 'doc_hash', 'title', 'content', and a date column.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted topics with columns:
            'topic_id', 'doc_hash', 'topic_name', 'topic_description', 'date
            If the input DataFrame is empty, returns an empty DataFrame with these
            columns.

        """
        return self.extract_topics(articles_df)

    def extract_topics(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate over every article and extract its topics. The main logic is in
        ``_extract_from_article()``, which calls the LLM for each article to get its
        topics. The resulting list of topics is then flattened and converted into a
        DataFrame with the specified output columns. If the input DataFrame is empty, it
        returns an empty topics DataFrame with the correct schema. This method is
        separate from the BaseAnalyzer interface to allow for more specific type
        annotations and to keep the code organized and readable. The topic extraction
        is done sequentially for simplicity, but it could be parallelized if needed,
        since each article is independent. The LLM calls are wrapped in a retry
        mechanism to handle transient failures, and any errors for individual articles
        are logged and skipped to ensure that one bad article does not abort the whole
        batch.

        Args:
            articles_df (pd.DataFrame): Group 1 output. Expected columns include
                ``doc_hash``, ``title``, ``content`` and a date column
                (``date`` or ``scrape_date``).

        Returns:
            pd.DataFrame: The ``topics`` table with :attr:`OUTPUT_COLUMNS`.
        """
        if articles_df.empty:
            print("[INFO] Articles DataFrame is empty. Nothing to extract.")
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        date_col = self._resolve_date_column(articles_df)
        articles = [article for _, article in articles_df.iterrows()]

        # Each article is independent, so fan the LLM calls out concurrently.
        per_article_rows = self._parallel_map(
            lambda article: self._article_to_rows(article, date_col), articles
        )
        rows = [row for article_rows in per_article_rows for row in article_rows]

        print(f"[INFO] Extracted {len(rows)} topics from {len(articles)} articles.")
        return pd.DataFrame(rows, columns=self.OUTPUT_COLUMNS)

    def _article_to_rows(self, article: pd.Series, date_col: str | None) -> list[dict]:
        """
        Extract one article's topics and shape them into topics-table rows.
        The LLM is called to extract topics from the article, and the resulting list of
        topics is transformed into a list of dictionaries, each representing a row in
        the output topics DataFrame. Each row includes a unique topic_id, the doc_hash
        to link back to the source article, the topic name and description from the LLM
        output, and the date copied from the article using the specified date column.
        If the date column is not found in the article, the date field in the output
        will be None. This method is called for each article, and the results are
        aggregated in the main extraction method. The LLM call is wrapped in a
        retry mechanism, and any errors are logged and result in an empty list of
        topics for that article, so one bad article does not abort the whole batch.
        The output of this method is a list of dictionaries, which are then converted
        into a DataFrame in the main extraction method. The doc_hash serves as a
        foreign key to join back to the source article, and the date is included
        for temporal context in later stages of the pipeline.

        Args:
            article (pd.Series): A single row from the articles DataFrame, containing
            the article's metadata and content.
            date_col (str | None): The name of the column in the article that contains
            the publication date. If None, the date field in the output will be None.

        Returns:
            list[dict]: A list of dictionaries, each representing a topic extracted from
            the article, with keys corresponding to the output columns of the topics
            DataFrame.
        """
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
        """
        Pick the first available date-like column from the article schema.
        The topic extractor needs a date for each topic to provide temporal context for
        later stages of the pipeline. Since different scrapers might use different
        column names for the publication date, this method checks for a predefined list
        of common date column names (e.g. 'date', 'scrape_date', 'published_at')
        and returns the first one it finds in the DataFrame. If none of the expected
        date columns are found, it logs a warning and returns None, which will result
        in the date field in the output topics being None. This is a fallback mechanism
        to handle inconsistencies in the input data, but ideally, the scrapers should be
        standardized to always include a date column with a consistent name.
        The presence of a date is important for the temporal analysis of trends in later
        stages, so it's recommended to ensure that the scrapers produce a date column
        with one of the expected names.

        Args:
            articles_df (pd.DataFrame): The DataFrame containing the articles, expected
            to have one of the date columns.

        Returns:
            str | None: The name of the date column found, or None if no date column
            is found.
        """
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

        Args:
            article (pd.Series): A single row from the articles DataFrame, containing
            the article's metadata and content.

        Returns:
            list[ExtractedTopic]: A list of topics extracted from the article. If the
            article contains no market-relevant events or if the LLM call fails, this
            willbe an empty list.
        """
        title = str(article.get("title", "")).strip()
        content = str(article.get("content", "")).strip()

        if not title and not content:
            return []

        print(f"[INFO] Extracting topics from: {title[:60]}")
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
