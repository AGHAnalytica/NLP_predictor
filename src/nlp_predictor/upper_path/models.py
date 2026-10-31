"""Pydantic schemas and quantization helpers for upper-path LLM outputs."""

from pydantic import BaseModel, Field

_SENTIMENT_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _quantize_sentiment(value: float) -> float:
    """Snap a raw sentiment float to the nearest discrete 5-point scale anchor in
    ``{-1.0,
    -0.5, 0.0, 0.5, 1.0}``.
    """
    return min(_SENTIMENT_LEVELS, key=lambda level: abs(level - value))


class TickerImpact(BaseModel):
    """Single instrument impact pair (ticker symbol and discrete sentiment score)."""

    ticker: str = Field(description="Exact ticker symbol from the available list.")
    sentiment: float = Field(
        description=(
            "Directional impact on this specific ticker on the 5-point scale: "
            "-1.0, -0.5, 0.0, 0.5, 1.0."
        )
    )


class InstrumentMapping(BaseModel):
    """Structured LLM response mapping a single news topic to its impacted
    instruments.
    """

    impacts: list[TickerImpact] = Field(
        description="List of impacted tickers and their sentiment (empty if none)."
    )


class InstrumentTrend(BaseModel):
    """Structured LLM response summarizing aggregate news topics for a single
    instrument.
    """

    title: str = Field(description="Short summary of the instrument trend.")
    description: str = Field(
        description="1-2 sentence explanation of the aggregate market impact."
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")
