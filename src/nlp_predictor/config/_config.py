"""Centralized configuration loader composing typed YAML schemas."""

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, Field

from ._backtesting import BacktestingConfig
from ._hyperparams import Hyperparams
from ._paths import Paths
from ._severity import Severity

REPO_ROOT = Path(__file__).resolve().parents[3]
_YAML_DIR = REPO_ROOT / "configs"


class Config(BaseModel):
    """Root application configuration combining paths, hyperparams, severity,
    backtesting, and sources.
    """

    paths: Paths
    hyperparams: Hyperparams
    severity: Severity
    backtesting: BacktestingConfig = BacktestingConfig()
    sources: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = {}


def _load_yaml_section(filename: str, section_key: str) -> dict[str, Any]:
    """Load a YAML configuration file and extract its top-level section dictionary."""
    file_path = _YAML_DIR / filename
    if not file_path.exists():
        return {}
    with open(file_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get(section_key, raw)


def _load_config() -> Config:
    """Instantiate the global Config object from the repository YAML files."""
    return Config(
        paths=_load_yaml_section("paths.yaml", "paths"),
        hyperparams=_load_yaml_section("hyperparameters.yaml", "hyperparams"),
        severity=_load_yaml_section("severity.yaml", "severity"),
        backtesting=_load_yaml_section("backtesting.yaml", "backtesting"),
        sources=_load_yaml_section("sources.yaml", "sources"),
    )


config = _load_config()
