"""Pydantic schema for lower-path macroeconomic cluster summaries."""

from pydantic import BaseModel, Field


class GlobalTrend(BaseModel):
    """Structured LLM response summarizing a semantic cluster of macroeconomic
    topics.
    """

    title: str = Field(description="Concise title of the global macroeconomic trend.")
    description: str = Field(
        description=(
            "1-2 sentence explanation of the narrative and its global market impact."
        )
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")
