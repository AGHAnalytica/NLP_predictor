"""Lookback and retention time windows (in hours) for live signal scoring."""

from pydantic import BaseModel


class WindowsHyperparams(BaseModel):
    """Default signal lookback horizon and maximum feature-store retention window."""

    signal_hours: float
    feature_store_max_hours: float
