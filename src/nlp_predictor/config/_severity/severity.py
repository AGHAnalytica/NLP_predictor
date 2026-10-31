"""Pydantic schemas for A/B/n severity model variants."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class BaseParams(BaseModel):
    """Core time-decay and tanh-scaling parameters shared by all severity models."""

    half_life_hours: float
    kappa_scale: float


class DCIMParams(BaseParams):
    """Extended parameters specific to the 5-phase DCIM severity model."""

    beta_penalty: float
    gamma_consensus: float
    delta_velocity: float
    recent_window_hours: float


class BaselineVariant(BaseModel):
    """Configuration for a baseline sentiment-sum variant."""

    model: Literal["baseline"]
    window_hours: float
    params: BaseParams


class AbnormalVolumeVariant(BaseModel):
    """Configuration for a volume-anomaly variant."""

    model: Literal["abnormal_volume"]
    window_hours: float
    params: BaseParams


class DCIMVariant(BaseModel):
    """Configuration for a full DCIM variant."""

    model: Literal["dcim"]
    window_hours: float
    params: DCIMParams


VariantType = Annotated[
    BaselineVariant | DCIMVariant | AbnormalVolumeVariant,
    Field(discriminator="model"),
]


class Severity(BaseModel):
    """Registry of configured severity variants evaluated on every live tick."""

    variants: dict[str, VariantType]

    @property
    def variants_list(self) -> list[dict[str, Any]]:
        """Serialize variants into flat dictionaries with keys ``name``, ``model``,
        ``window_hours``, and ``params``.
        """
        return [
            {
                "name": name,
                "model": v.model,
                "window_hours": v.window_hours,
                "params": v.params.model_dump(),
            }
            for name, v in self.variants.items()
        ]
