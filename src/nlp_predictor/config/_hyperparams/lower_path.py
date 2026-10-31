"""Hyperparameters for the lower path (HDBSCAN clustering and macro severity)."""

from pydantic import BaseModel


class LowerPathHyperparams(BaseModel):
    """Clustering thresholds, decay half-life, and lookback window for macro trends."""

    severity_threshold: float
    min_cluster_size: int
    macro_severity_model: str
    half_life_hours: float
    climate_window_hours: float
