from pydantic import BaseModel

from .lower_path import LowerPathHyperparams
from .models import ModelsHyperparams
from .signal import SignalHyperparams
from .upper_path import UpperPathHyperparams
from .windows import WindowsHyperparams


class Hyperparams(BaseModel):
    models: ModelsHyperparams
    upper_path: UpperPathHyperparams
    lower_path: LowerPathHyperparams
    windows: WindowsHyperparams
    signal: SignalHyperparams
