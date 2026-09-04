from pydantic import BaseModel, Field


class Topic(BaseModel):
    """
    A single market-relevant topic extracted from an article. The LLM is prompted to
    find topics that are relevant for financial markets, so the ``topic_name``
    should be a concise label for a specific event, phenomenon, or trend
    (e.g. "Strait of Hormuz Reopening"), and the ``topic_description`` should be a brief
    explanation of what the topic is about, based solely on the article's content.
    The description may include a general note on market relevance or likely direction,
    but it must NOT name specific financial instruments or give buy/sell advice, to
    keep the topic extraction stage focused on identifying events rather than making
    trading recommendations. The output contract for the topic extractor is a list of
    these topics per article, which are then fed into the upper and lower paths for
    further analysis and ultimately contribute to the final trading signal.
    """

    name: str = Field(
        description="Short, specific event/phenomenon name "
        "(e.g. 'Strait of Hormuz Reopening'). NOT a person or a news outlet."
    )
    description: str = Field(
        description="2-3 sentences describing what is happening, based on the article. "
        "May include a brief, general note on market relevance or likely direction, "
        "but must NOT name specific instruments/tickers or give buy/sell advice."
    )
