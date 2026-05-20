# Core imports
import os
import uuid
from datetime import datetime, timezone
import pandas as pd
from pydantic import BaseModel, Field
import time

# Handle API and LLM
from dotenv import load_dotenv
from google.genai.errors import ClientError
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Load environment variables from the .env file located in the project root
load_dotenv()

# Default instrument path
INSTRUMENT_PATH = "../../data/instrument_trends/instruments.csv"


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
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Critical Security Error: GEMINI_API_KEY is not set in the environment."
            )
        self.client = genai.Client(api_key=api_key)

        # Load available instruments and create a context string for the LLM
        self.instruments_df = pd.read_csv(instruments_path)
        self.available_instruments = self.instruments_df.to_dict("records")
        self.valid_tickers = set(self.instruments_df["ticker"].dropna().tolist())

        # Create clean dictionary for the LLM (Ticker: Full Name)
        self.available_instruments = dict(
            zip(self.instruments_df["ticker"], self.instruments_df["full_name"])
        )

    @retry(
        stop=stop_after_attempt(20),  # Allow more retries for small quota
        wait=wait_exponential(multiplier=2, min=15, max=120),
        retry=retry_if_exception_type(ClientError),
        before_sleep=lambda retry_state: print(
            f"[WAIT] Rate limit hit. Sleeping... (Attempt {retry_state.attempt_number})")
    )
    def _call_gemini_api_with_retry(
            self, prompt: str, schema: type[BaseModel], temperature: float
    ) -> BaseModel:
        """
        Calls the Gemini API and returns a validated Pydantic object directly.
        Uses exponential backoff to handle rate limits.
        """
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            ),
        )
        # Directly return the parsed Pydantic object
        return response.parsed

    def process_upper_path(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline for the upper path.
        Expects a DataFrame with 'topic_id', 'doc_hash', 'topic_name', 'topic_description', 'date'.
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
            # Artificially delay to prevent aggressive hitting of the 5 RPM limit
            print(f"[INFO] Processing topic: {row['topic_name'][:30]}...")
            time.sleep(5)

            # The returned object is a validated InstrumentMapping Pydantic instance
            parsed_result = self._call_gemini_api_with_retry(
                prompt, InstrumentMapping, 0.0
            )

            tickers = parsed_result.tickers
            tickers = [str(t).strip() for t in tickers]

            # Validate against known tickers to filter out hallucinations
            valid_extracted_tickers = [
                ticker for ticker in tickers if ticker in self.valid_tickers
            ]

            if not valid_extracted_tickers and tickers:
                print(f"[DEBUG] Validated out hallucinatory tickers for '{row['topic_name']}': {tickers}")

            return valid_extracted_tickers

        except Exception as e:
            print(f"[ERROR] Mapping failed for '{row['topic_name']}': {e}")
            return []

    def _calculate_instrument_severity(self, group: pd.DataFrame) -> float:
        """
        Calculates severity based on the number of grouped topics.
        """
        # Baseline severity: number of related topics
        return float(len(group))

    def _generate_instrument_trend(
            self, instrument: str, group: pd.DataFrame, severity: float
    ) -> dict:
        topics_context = group[["topic_name", "topic_description"]].to_dict("records")
        topic_ids = group["topic_id"].tolist()

        prompt = f"""
        You are a senior financial analyst. Analyze the following news topics related to the instrument '{instrument}'.
        Determine the overall price trend direction ('up', 'down', or 'neutral').
        Provide a concise title and a 1-2 sentence description explaining the aggregate impact.

        Topics Context:
        {topics_context}
        """

        try:
            time.sleep(5)
            # The returned object is a validated InstrumentTrend Pydantic instance
            parsed_result = self._call_gemini_api_with_retry(
                prompt, InstrumentTrend, 0.2
            )

            # Convert Pydantic model to a standard dictionary
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
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "instrument": instrument,
            "title": result.get("title", "Unknown"),
            "description": result.get("description", "Unknown"),
            "severity": round(severity, 2),
            "direction": result.get("direction", "neutral").lower(),
            "topic_ids": topic_ids,
        }


if __name__ == "__main__":
    INSTRUMENT_PATH = "../../data/instrument_trends/instruments.csv"
    TOPICS_PATH = "../../data/instrument_trends/topics_sample.csv"
    print("====== UpperPathAnalyzer Module Test ======")

    upper_path_analyzer = UpperPathAnalyzer(INSTRUMENT_PATH)
    df_topics = pd.read_csv(TOPICS_PATH)

    result = upper_path_analyzer.process_upper_path(df_topics)
    print("====== Processing Upper Path Completed ======")

    print("====== Saving Results To File ======")
    result.to_csv("../../data/instrument_trends/result.csv")
    print("====== Saving Completed ======")

    result.head()
