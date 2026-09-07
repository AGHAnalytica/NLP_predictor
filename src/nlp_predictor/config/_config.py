from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ._hyperparams import Hyperparams
from ._paths import Paths
from ._severity import Severity

REPO_ROOT = Path(__file__).resolve().parents[2]
_YAML_DIR = REPO_ROOT / "config"


class Config(BaseModel):
    paths: Paths
    hyperparams: Hyperparams
    severity: Severity


def _load_config() -> Config:
    with open(_YAML_DIR / "paths.yaml") as f:
        paths_data = yaml.safe_load(f) or {}

    with open(_YAML_DIR / "hyperparameters.yaml") as f:
        hyperparams_data = yaml.safe_load(f) or {}

    with open(_YAML_DIR / "severity.yaml") as f:
        severity_data = yaml.safe_load(f) or {}

    combined_config: dict[str, Any] = {
        "paths": paths_data,
        "hyperparams": hyperparams_data,
        "severity": severity_data,
    }

    return Config(**combined_config)


config = _load_config()
