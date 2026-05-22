# Core imports
import os
import uuid
from datetime import datetime, timezone
import pandas as pd
from pydantic import BaseModel, Field

# Handle API and LLM
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Load environment variables from the .env file located in the project root
load_dotenv()

# Default instrument path
INSTRUMENT_PATH = "../../data/instruments/instruments_sample.csv"
SAVE_PATH = "../../data/test_results/instrument_trends_report.csv"


# Define Pydantic schemas for strict LLM JSON outputs
class InstrumentMapping(BaseModel):
    tickers: list[str] = Field(
        description="List of ticker symbols of the financial instruments impacted. Return an empty list if none."
    )


class InstrumentTrend(BaseModel):
    title: str = Field(description="Short summary of the trend.")
    description: str = Field(
        description="Detailed explanation of the trend and its impact."
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")


class UpperPathAnalyzer:
    """
    Handles Group 3 tasks: mapping topics to specific financial instruments
    and generating per-instrument trends based on the project specification.
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
            zip(self.instruments_df["ticker"], self.instruments_df["full_name"])
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
        Calls the OpenAI API with structured output and returns a validated Pydantic object.
        Uses exponential backoff to handle rate limits.
        """
        response = self.client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
        )
        return response.choices[0].message.parsed  # type: ignore

    def process_upper_path(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline for the upper path.
        Expects a DataFrame with 'topic_id', 'doc_hash', 'topic_name', 'topic_description', 'date'.
        The 'date' column must contain the article post timestamp.
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
                "[WARNING] All topics were dropped after mapping. The result will be empty."
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
        prompt = f"""
        You are a quantitative financial analyst. Map the following news topic to ALL applicable instruments from the list.
        If the topic does not directly impact any of these specific instruments, return an empty list.
        IMPORTANT: Return ONLY the exact 'ticker' symbols from the provided list, never full company names.

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
                    f"[DEBUG] Validated out hallucinatory tickers for '{row['topic_name']}': {tickers}"
                )

            return valid_extracted_tickers

        except Exception as e:
            print(f"[ERROR] Mapping failed for '{row['topic_name']}': {e}")
            return []

    def _calculate_instrument_severity(self, group: pd.DataFrame) -> float:
        return float(len(group))

    def _generate_instrument_trend(
        self, instrument: str, group: pd.DataFrame, severity: float
    ) -> dict:
        topics_context = group[["topic_name", "topic_description"]].to_dict(
            orient="records"
        )  # type: ignore[call-overload]
        topic_ids = group["topic_id"].tolist()

        # Derive time range from the articles' posted timestamps (ISO strings sort correctly)
        first_article_time = str(group["date"].min())[:19]
        newest_article_time = str(group["date"].max())[:19]

        prompt = f"""
        You are a senior financial analyst. Analyze the following news topics related to the instrument '{instrument}'.
        Determine the overall price trend direction ('up', 'down', or 'neutral').
        Provide a concise title and a 1-2 sentence description explaining the aggregate impact.

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
            "current_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "first_article_time": first_article_time,
            "newest_article_time": newest_article_time,
            "instrument": instrument,
            "title": result.get("title", "Unknown"),
            "description": result.get("description", "Unknown"),
            "severity": round(severity, 2),
            "direction": result.get("direction", "neutral").lower(),
            "topic_ids": topic_ids,
        }


if __name__ == "__main__":
    INSTRUMENT_PATH = "../../data/instruments/instruments_sample.csv"
    TOPICS_PATH = "../../data/topics/topics_sample.csv"
    print("====== UpperPathAnalyzer Module Test ======")

    upper_path_analyzer = UpperPathAnalyzer(INSTRUMENT_PATH)
    df_topics = pd.read_csv(TOPICS_PATH)

    result = upper_path_analyzer.process_upper_path(df_topics)
    print("====== Processing Upper Path Completed ======")

    print("====== Saving Results To File ======")
    result.to_csv(SAVE_PATH, index=False)
    print("====== Saving Completed ======")

    result.head()
