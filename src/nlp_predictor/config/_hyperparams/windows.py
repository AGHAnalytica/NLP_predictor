from pydantic import BaseModel


class WindowsHyperparams(BaseModel):
    signal_hours: float
    feature_store_hours: float
