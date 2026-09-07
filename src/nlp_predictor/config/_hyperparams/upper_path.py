from pydantic import BaseModel


class DCIMHyperparams(BaseModel):
    half_life_hours: float
    beta_penalty: float
    gamma_consensus: float
    delta_velocity: float
    kappa_scale: float
    recent_window_hours: float


class UpperPathHyperparams(BaseModel):
    dcim: DCIMHyperparams
