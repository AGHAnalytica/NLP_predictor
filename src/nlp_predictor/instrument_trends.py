# Core imports
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Handle API and LLM
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load environment variables from the .env file located in the project root
load_dotenv()

# Default instrument path (resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parents[2]
INSTRUMENT_PATH = str(BASE_DIR / "data" / "instruments" / "instruments_sample.csv")
SAVE_PATH = str(BASE_DIR / "data" / "test_results" / "instrument_trends_report.csv")


# Define Pydantic schemas for strict LLM JSON outputs
class InstrumentMapping(BaseModel):
    tickers: list[str] = Field(
        description="""List of ticker symbols of the financial instruments impacted.
        Return an empty list if none."""
    )


class InstrumentTrend(BaseModel):
    title: str = Field(description="Short summary of the trend.")
    description: str = Field(
        description="Detailed explanation of the trend and its impact."
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")


class UpperPathAnalyzer:
    """
    This class encapsulates the logic for processing the upper path of the data
    pipeline. It includes methods for mapping news topics to financial instruments,
    generating trend summaries, and saving the results to a partitioned Parquet dataset.
    The class is designed to be robust against API rate limits and potential
    hallucinations by the LLM, with strict validation against known instruments.
    It also includes detailed logging for monitoring the processing steps and handling a
    ny issues that arise.

    The main method, `process_upper_path`, takes a DataFrame of topics, maps each
    topic to relevant instruments, and generates a trend summary for each instrument
    based on the associated topics. The final output is a DataFrame containing the
    trends for each instrument, ready to be saved as a partitioned Parquet dataset.

    """

    def __init__(self, instruments_path: str = INSTRUMENT_PATH):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Critical Security Error: OPENAI_API_KEY is not set in the environment."
            )
        self.client = OpenAI(api_key=api_key)

        # Load available instruments and create a context string for the LLM
        self.instruments_df = pd.read_csv(instruments_path)
        self.valid_tickers = set(self.instruments_df["ticker"].dropna().tolist())

        # Create clean dictionary for the LLM (Ticker: Full Name)
        self.available_instruments = dict(
            zip(
                self.instruments_df["ticker"],
                self.instruments_df["full_name"],
                strict=False,
            )
        )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=lambda retry_state: print(
            f"[WAIT] Rate limit hit. Sleeping... (Attempt {retry_state.attempt_number})"
        ),
    )
    def _call_openai_api_with_retry(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        """
        Calls the OpenAI API with structured output and returns a validated
        Pydantic object. Uses exponential backoff to handle rate limits.

        Args:
            prompt (str): The prompt to send to the LLM.
            schema (type[BaseModel]): The Pydantic model class to validate the output.

        Returns:
            BaseModel: An instance of the provided Pydantic model containing the parsed
            response.

        """
        response = self.client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format=schema,
        )
        return response.choices[0].message.parsed  # type: ignore

    def process_upper_path(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing function for the upper path. It takes a DataFrame of topics,
        maps each topic to relevant financial instruments, and then generates a trend
        summary for each instrument based on the associated topics. The final output
        is a DataFrame containing the trends for each instrument, ready to be saved as a
        partitioned Parquet dataset.

        Args:
            topics_df (pd.DataFrame): A DataFrame containing the topics with columns
            such as 'topic_id', 'topic_name', 'topic_description', and 'date'.

        Returns:
            pd.DataFrame: A DataFrame containing the generated trends for each
            instrument, including columns like 'trend_id', 'current_time',
            'first_article_time', 'newest_article_time', 'instrument', 'title',
            'description', 'severity', 'direction', and 'topic_ids'.

        """
        if topics_df.empty:
            return pd.DataFrame()

        # Map topics to multiple instruments
        print("[INFO] Starting topic to instrument mapping...")
        topics_df["instruments"] = topics_df.apply(
            self._map_topic_to_instrument, axis=1
        )

        # Explode the list column into separate rows
        exploded_df = topics_df.explode("instruments")

        # Filter out topics that don't map to any specific instrument
        mapped_df = exploded_df.dropna(subset=["instruments"]).rename(
            columns={"instruments": "instrument"}
        )

        if mapped_df.empty:
            print(
                "[WARNING] All topics were dropped after mapping. "
                "The result will be empty."
            )
            return pd.DataFrame()

        results = []

        # Group by instrument
        print("[INFO] Generating trends per instrument...")
        for instrument, group in mapped_df.groupby("instrument"):
            severity = self._calculate_instrument_severity(group)
            trend_data = self._generate_instrument_trend(instrument, group, severity)
            results.append(trend_data)

        return pd.DataFrame(results)

    def _map_topic_to_instrument(self, row: pd.Series) -> list[str]:
        """Maps a single topic to relevant financial instruments using the LLM.
        The prompt provides the topic name and description, along with a list of
        available instruments (tickers and full names). The LLM is asked to return a
        list of ticker symbols that are directly impacted by the topic. If no
        specific instruments are impacted, it should return an empty list.
        The output is strictly validated against the known tickers to filter
        out any hallucinations.

        Args:
            row (pd.Series): A row from the topics DataFrame containing 'topic_name'
            and 'topic_description'.

        Returns:
            list[str]: A list of ticker symbols that are relevant to the topic.

        """
        prompt = f"""
        You are a quantitative financial analyst. Map the following news topic to ALL
        applicable instruments from the list. If the topic does not directly impact any
        of these specific instruments, return an empty list. IMPORTANT: Return ONLY the
        exact 'ticker' symbols from the provided list, never full company names.

        Available Instruments: {self.available_instruments}

        Topic Name: {row["topic_name"]}
        Topic Description: {row["topic_description"]}
        """

        try:
            print(f"[INFO] Processing topic: {row['topic_name'][:30]}...")

            parsed_result = self._call_openai_api_with_retry(prompt, InstrumentMapping)

            tickers = parsed_result.tickers
            tickers = [str(t).strip() for t in tickers]

            # Validate against known tickers to filter out hallucinations
            valid_extracted_tickers = [
                ticker for ticker in tickers if ticker in self.valid_tickers
            ]

            if not valid_extracted_tickers and tickers:
                print(
                    "[DEBUG] Validated out hallucinatory tickers for '"
                    f"{row['topic_name']}' : {tickers}"
                )

            return valid_extracted_tickers

        except Exception as e:
            print(f"[ERROR] Mapping failed for '{row['topic_name']}': {e}")
            return []

    def _calculate_instrument_severity(self, group: pd.DataFrame) -> float:
        """Calculates a severity score for the instrument based on the number of
        related topics. This is a simple heuristic where more related topics
        indicate higher severity.

        Args:
            group (pd.DataFrame): A DataFrame containing the topics related to the
            instrument.

        Returns:
            float: A severity score (e.g., the count of related topics).

        """
        return float(len(group))

    def _generate_instrument_trend(
        self, instrument: str, group: pd.DataFrame, severity: float
    ) -> dict:
        """
        Generates a trend summary for a given instrument based on the associated topics.
        The prompt includes the instrument name, a list of related topics with
        their descriptions, and the time range of the articles.
        The LLM is asked to determine the overall price trend direction
        ('up', 'down', 'neutral') and provide a concise title and description explaining
        the aggregate impact.

        Args:
            instrument (str): The ticker symbol of the financial instrument.
            group (pd.DataFrame): A DataFrame containing the topics related to
            the instrument.
            severity (float): A calculated severity score based on the number of
            related topics.

        Returns:
            dict: A dictionary containing the trend information, including a unique
            trend ID, timestamps, instrument name, title, description, severity score,
            direction, and associated topic IDs.

        """
        topics_context = group[["topic_name", "topic_description"]].to_dict(
            orient="records"
        )  # type: ignore[call-overload]
        topic_ids = group["topic_id"].tolist()

        # Derive time range from the articles' posted timestamps
        # (ISO strings sort correctly)
        first_article_time = str(group["date"].min())[:19]
        newest_article_time = str(group["date"].max())[:19]

        prompt = f"""
        You are a senior financial analyst. Analyze the following news topics related to
        the instrument '{instrument}'. Determine the overall price trend direction
        ('up', 'down', or 'neutral'). Provide a concise title and a 1-2 sentence
        description explaining the aggregate impact.

        Topics Context:
        {topics_context}
        """

        try:
            parsed_result = self._call_openai_api_with_retry(prompt, InstrumentTrend)
            result = parsed_result.model_dump()

        except Exception as e:
            print(f"[ERROR] Trend generation failed for '{instrument}': {e}")
            result = {
                "title": "Data aggregation error",
                "description": "Failed to generate summary.",
                "direction": "neutral",
            }

        return {
            "trend_id": str(uuid.uuid4()),
            "current_time": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "first_article_time": first_article_time,
            "newest_article_time": newest_article_time,
            "instrument": instrument,
            "title": result.get("title", "Unknown"),
            "description": result.get("description", "Unknown"),
            "severity": round(severity, 2),
            "direction": result.get("direction", "neutral").lower(),
            "topic_ids": topic_ids,
        }

    @staticmethod
    def save_trends(result_df: pd.DataFrame, output_path: str):
        """
        Saves the resulting DataFrame to a partitioned Parquet dataset.
        - Partitions by date (extracted from 'current_time' or 'date' column).
        - Uses fastparquet engine for efficient writing and native
        Hive-style partitioning.
        - Appends new data without overwriting existing partitions, creating new
        parquet files with unique names.
        - Handles empty DataFrames gracefully by skipping the save operation.
        - Ensures the target directory exists before saving.
        - Logs key steps and any issues encountered during the save process.
        Note: The 'current_time' column is expected to be in a datetime format.
        If it's not present, it falls back to using the 'date' column for partitioning.

        Args:
            result_df (pd.DataFrame): The DataFrame containing the trends to save.
            output_path (str): The path to the directory where the Parquet files will be
            saved.

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if result_df.empty:
            print("[INFO] DataFrame is empty. Skipping Parquet save operation.")
            return False

        try:
            # Ensure the target base directory exists
            os.makedirs(output_path, exist_ok=True)

            # Identify the time column ('current_time' based on latest implementation)
            time_col = "current_time" if "current_time" in result_df.columns else "date"

            # Create a new column strictly for partitioning (extracting just the date)
            # Converting to string ensures clean folder names in the OS
            result_df["date_partition"] = pd.to_datetime(
                result_df[time_col]
            ).dt.date.astype(str)

            # Check SPECIFICALLY for the fastparquet metadata file to enable appending
            metadata_path = os.path.join(output_path, "_metadata")
            dataset_exists = os.path.isfile(metadata_path)

            # Write to Parquet using the fastparquet engine.
            # fastparquet handles Hive partitioning natively.
            # When called multiple times, it generates new, unique parquet files (UUIDs)
            # inside the partition folders, effectively appending data without
            # overwriting historical batches.
            result_df.to_parquet(
                output_path,
                engine="fastparquet",
                partition_cols=["date_partition"],
                index=False,
                append=dataset_exists,
            )

            mode_str = "Appended to" if dataset_exists else "Created new"
            print(f"[INFO] {mode_str} partitioned Parquet dataset at {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save results to Parquet: {e}")
            return False


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    INSTRUMENT_PATH = str(BASE_DIR / "data" / "instruments" / "instruments_sample.csv")
    TOPICS_PATH = str(BASE_DIR / "data" / "topics" / "topics_sample.csv")
    SAVE_PATH = str(BASE_DIR / "data" / "test_results" / "instrument_trends_parquet")

    print("====== UpperPathAnalyzer Module Test ======")

    upper_path_analyzer = UpperPathAnalyzer(INSTRUMENT_PATH)
    df_topics = pd.read_csv(TOPICS_PATH)

    result = upper_path_analyzer.process_upper_path(df_topics)
    print("====== Processing Upper Path Completed ======")

    print("====== Saving Results To File ======")
    saving_result = UpperPathAnalyzer.save_trends(result, SAVE_PATH)
    if saving_result:
        print("====== Saving Completed ======")
    else:
        print("====== Saving Failed ======")

    # For debugging: save the result to a CSV as well
    # (not required, but useful for quick checks)
    debug_csv_path = str(
        BASE_DIR / "data" / "test_results" / "instrument_trends_report.csv"
    )
    result.to_csv(debug_csv_path, index=False)
    print(f"[DEBUG] Also saved results to CSV for quick inspection at {debug_csv_path}")
