from pydantic import BaseModel, Field


class GlobalTrend(BaseModel):
    """
    Pydantic model for parsing the output of the global trend summary from OpenAI.
    """

    title: str = Field(description="Short summary of the global macroeconomic trend.")
    description: str = Field(
        description="Detailed explanation of the trend and its global impact."
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")
