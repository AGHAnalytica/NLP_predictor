"""Hyperparameters for the upper path (DCIM single-model parameters)."""

from pydantic import BaseModel


class DCIMHyperparams(BaseModel):
    """Parameters for the 5-phase Directional Confirmed Impact Model (DCIM)."""

    half_life_hours: float
    beta_penalty: float
    gamma_consensus: float
    delta_velocity: float
    kappa_scale: float
    recent_window_hours: float


class UpperPathHyperparams(BaseModel):
    """Upper-path scoring hyperparameters."""

    dcim: DCIMHyperparams
