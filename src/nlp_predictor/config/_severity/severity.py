from typing import Literal

from pydantic import BaseModel, Field


class BaseParams(BaseModel):
    """Shared parameters across all models."""
    half_life_hours: float
    kappa_scale: float

class DCIMParams(BaseParams):
    """Specific parameters only allowed in DCIM."""
    beta_penalty: float
    gamma_consensus: float
    delta_velocity: float
    recent_window_hours: float

class BaselineVariant(BaseModel):
    model: Literal["baseline"]
    window_hours: float
    params: BaseParams

class AbnormalVolumeVariant(BaseModel):
    model: Literal["abnormal_volume"]
    window_hours: float
    params: BaseParams

class DCIMVariant(BaseModel):
    model: Literal["dcim"]
    window_hours: float
    params: DCIMParams


VariantType = BaselineVariant | DCIMVariant | AbnormalVolumeVariant

class Severity(BaseModel):
    variants: dict[str, VariantType] = Field(discriminator="model")
