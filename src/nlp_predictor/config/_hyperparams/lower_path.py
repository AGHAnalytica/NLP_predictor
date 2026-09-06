from pydantic import BaseModel


class LowerPathHyperparams(BaseModel):
    severity_threshold: float
    min_cluster_size: int
    macro_severity_model: str
    half_life_hours: float
    climate_window_hours: float
