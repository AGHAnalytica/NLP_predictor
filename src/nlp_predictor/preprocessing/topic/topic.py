from pydantic import BaseModel, Field


class ExtractedTopic(BaseModel):
    """
    A single market-relevant topic extracted from an article.

    The topic name should represent a concise event or phenomenon, while the
    description provides contextual explanation based on the text without
    naming specific tickers or trading recommendations.
    """

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
    """Container schema holding a list of topics extracted by the LLM."""

    topics: list[ExtractedTopic] = Field(
        default_factory=list,
        description="List of market-relevant topics identified in the article.",
    )
