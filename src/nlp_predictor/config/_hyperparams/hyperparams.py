"""Aggregated hyperparameter schema for models, paths, windows, and signal fusion."""

from pydantic import BaseModel

from .lower_path import LowerPathHyperparams
from .models import ModelsHyperparams
from .signal import SignalHyperparams
from .upper_path import UpperPathHyperparams
from .windows import WindowsHyperparams


class Hyperparams(BaseModel):
    """Composite hyperparameter configuration across all analytical stages."""

    models: ModelsHyperparams
    upper_path: UpperPathHyperparams
    lower_path: LowerPathHyperparams
    windows: WindowsHyperparams
    signal: SignalHyperparams
