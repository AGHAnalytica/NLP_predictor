import os
import json
import uuid
from datetime import datetime, timezone
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

INSTRUMENT_PATH = "../data/instrument_trands/instruments.csv"


# Define Pydantic schemas for strict LLM JSON outputs
class InstrumentMapping(BaseModel):
    tickers: list[str] = Field(
        description="List of ticker symbols of the financial instruments impacted. Return an empty list if none."
    )


class InstrumentTrend(BaseModel):
    title: str = Field(description="Short summary of the trend.")
    description: str = Field(description="Detailed explanation of the trend and its impact.")
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")


class UpperPathAnalyzer:
    """
    Handles Group 3 tasks: mapping topics to specific financial instruments
    and generating per-instrument trends based on the project specification.
    """

    def __init__(self, instruments_path: str = INSTRUMENT_PATH):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Load available instruments and create a context string for the LLM
        self.instruments_df = pd.read_csv(instruments_path)
        self.available_instruments = self.instruments_df.to_dict('records')
        self.valid_tickers = set(self.instruments_df['ticker'].dropna().tolist())

    def process_upper_path(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline for the upper path.
        Expects a DataFrame with 'topic_id', 'doc_hash', 'topic_name', 'topic_description', 'date'.
        """
        if topics_df.empty:
            return pd.DataFrame()

        # Map topics to multiple instruments
        topics_df['instruments'] = topics_df.apply(self._map_topic_to_instrument, axis=1)

        # Explode the list column into separate rows
        # E.g., row with ['AAPL', 'GOOGL'] becomes two identical rows, one for each ticker
        exploded_df = topics_df.explode('instruments')

        # Filter out topics that don't map to any specific instrument
        mapped_df = exploded_df.dropna(subset=['instruments']).rename(columns={'instruments': 'instrument'})

        results = []

        # Group by instrument
        for instrument, group in mapped_df.groupby('instrument'):
            # Calculate severity per instrument
            severity = self._calculate_instrument_severity(group)

            # Generate instrument trend
            trend_data = self._generate_instrument_trend(instrument, group, severity)
            results.append(trend_data)

        return pd.DataFrame(results)

    def _map_topic_to_instrument(self, row: pd.Series) -> list[str]:
        """
        Uses LLM to map a single topic to MULTIPLE available instrument tickers.
        """
        prompt = f"""
        You are a quantitative financial analyst. Map the following news topic to ALL applicable instruments from the list.
        If the topic does not directly impact any of these specific instruments, return an empty list.

        Available Instruments: {self.available_instruments}

        Topic Name: {row['topic_name']}
        Topic Description: {row['topic_description']}
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=InstrumentMapping,
                    temperature=0.0  # Low temperature for deterministic mapping
                )
            )

            result = json.loads(response.text)
            tickers = result.get('tickers', [])

            # Validate against known tickers to prevent hallucinations and filter out invalid ones
            valid_extracted_tickers = [ticker for ticker in tickers if ticker in self.valid_tickers]

            return valid_extracted_tickers

        except Exception as e:
            # Fallback in case of API failure or rate limit
            return []

    def _calculate_instrument_severity(self, group: pd.DataFrame) -> float:
        """
        Calculates severity based on the number of grouped topics.
        """
        # Baseline severity: number of related topics
        return float(len(group))

    def _generate_instrument_trend(self, instrument: str, group: pd.DataFrame, severity: float) -> dict:
        """
        Generates the final trend summary and establishes direction using LLM.
        """
        topics_context = group[['topic_name', 'topic_description']].to_dict('records')
        topic_ids = group['topic_id'].tolist()

        prompt = f"""
        You are a senior financial analyst. Analyze the following news topics related to the instrument '{instrument}'.
        Determine the overall price trend direction ('up', 'down', or 'neutral').
        Provide a concise title and a 1-2 sentence description explaining the aggregate impact.

        Topics Context:
        {topics_context}
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=InstrumentTrend,
                    temperature=0.2  # Slight temperature allowance for text generation
                )
            )

            result = json.loads(response.text)

        except Exception as e:
            # Fallback structure if LLM fails
            result = {
                "title": "Data aggregation error",
                "description": "Failed to generate summary.",
                "direction": "neutral"
            }

        return {
            "trend_id": str(uuid.uuid4()),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "instrument": instrument,
            "title": result.get("title", "Unknown"),
            "description": result.get("description", "Unknown"),
            "severity": round(severity, 2),
            "direction": result.get("direction", "neutral").lower(),
            "topic_ids": topic_ids
        }
