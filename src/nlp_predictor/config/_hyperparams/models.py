"""OpenAI model identifiers, concurrency limits, and retry policy parameters."""

from pydantic import BaseModel


class ModelsHyperparams(BaseModel):
    """LLM and embedding model names alongside concurrency and exponential-backoff
    bounds.
    """

    llm: str
    embedding: str
    severity_model: str
    max_workers: int = 2
    request_delay: float = 0.2
    retry_attempts: int = 12
    retry_min_wait: float = 4.0
    retry_max_wait: float = 60.0
