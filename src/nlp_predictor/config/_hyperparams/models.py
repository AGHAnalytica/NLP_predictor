from pydantic import BaseModel


class ModelsHyperparams(BaseModel):
    llm: str
    embedding: str
    severity_model: str
