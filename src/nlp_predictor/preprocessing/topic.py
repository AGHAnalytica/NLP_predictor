"""Pydantic schemas for structured LLM topic extraction."""

from pydantic import BaseModel, Field


class ExtractedTopic(BaseModel):
    """Single market-relevant event or phenomenon extracted from an article."""

    name: str = Field(
        ...,
        description=(
            "Short, specific event/phenomenon name (e.g. 'Strait of Hormuz Reopening')."
            "NOT a person, ticker, or news outlet."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "2-3 sentences describing what is happening, based on the article. "
            "May include a brief, general note on market relevance, but must NOT "
            "name specific instruments/tickers or give buy/sell advice."
        ),
    )


class ExtractedTopics(BaseModel):
    """Top-level response container holding all topics extracted from a single
    article.
    """

    topics: list[ExtractedTopic] = Field(
        default_factory=list,
        description="List of market-relevant topics identified in the article.",
    )
