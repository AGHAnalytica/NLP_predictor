# Core imports

import fastparquet  # noqa: F401

# Handle API and LLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from the .env file located in the project root
load_dotenv(override=True)

# Fixed sentiment scale the LLM is instructed to use (see the instrument_trends prompt).
# Snapping the model's free-form score onto these anchors removes run-to-run decimal
# drift and keeps the value calibrated the same way across independent calls.
_SENTIMENT_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _quantize_sentiment(value: float) -> float:
    """Snap a raw sentiment score to the nearest fixed-scale level."""
    return min(_SENTIMENT_LEVELS, key=lambda level: abs(level - value))


# Define Pydantic schemas for strict LLM JSON outputs
class TickerImpact(BaseModel):
    """
    Pydantic model for parsing the mapping of a topic to impacted tickers and their
    sentiment.
    """

    ticker: str = Field(description="Exact ticker symbol from the available list.")
    sentiment: float = Field(
        description="Directional impact of the news on THIS SPECIFIC ticker, on a "
        "fixed 5-point scale: -1.0 / -0.5 / 0.0 / 0.5 / 1.0 (negative = bearish, "
        "0.0 = neutral, positive = bullish). See the instrument_trends prompt for the "
        "anchors. Also snapped to the nearest level by _quantize_sentiment."
    )


class InstrumentMapping(BaseModel):
    """
    Pydantic model for parsing the mapping of a topic to impacted tickers and their
    sentiment.
    """

    impacts: list[TickerImpact] = Field(
        description="List of impacted tickers and their sentiment."
        + " Return empty if none."
    )


class InstrumentTrend(BaseModel):
    """
    Pydantic model for parsing the output of the instrument trend summary from OpenAI.
    """

    title: str = Field(description="Short summary of the trend.")
    description: str = Field(
        description="Detailed explanation of the trend and its impact."
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")
