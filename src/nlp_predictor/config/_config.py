from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass

from ._hyperparams import Hyperparams
from ._paths import Paths
from ._severity import Severity

REPO_ROOT = Path(__file__).resolve().parents[3]
_YAML_DIR = REPO_ROOT / "configs"

_overrides: dict[str, Any] = {}


def _build_legacy_resolvers(cfg: "Config") -> dict[str, Callable[[], Any]]:
    return {
        "BASE_DIR": lambda: REPO_ROOT,
        "INSTRUMENT_PATH": lambda: str(cfg.paths.input.instruments),
        "TOPICS_PATH": lambda: str(cfg.paths.input.topics),
        "ARTICLES_PATH": lambda: str(cfg.paths.input.articles),
        "WEIGHTS_PATH": lambda: str(cfg.paths.input.weights),
        "TOPICS_SAVE_PATH_CSV": lambda: str(cfg.paths.output.topics_csv),
        "TOPICS_SAVE_PATH_PARQUET": lambda: str(cfg.paths.output.topics_parquet),
        "GLOBAL_TRENDS_SAVE_PATH_CSV": lambda: str(cfg.paths.output.global_trends_csv),
        "GLOBAL_TRENDS_SAVE_PATH_PARQUET": lambda: str(
            cfg.paths.output.global_trends_parquet
        ),
        "INSTRUMENT_TRENDS_SAVE_PATH_CSV": lambda: str(
            cfg.paths.output.instrument_trends_csv
        ),
        "INSTRUMENT_TRENDS_SAVE_PATH_PARQUET": lambda: str(
            cfg.paths.output.instrument_trends_parquet
        ),
        "SIGNALS_SAVE_PATH_CSV": lambda: str(cfg.paths.output.signals_csv),
        "SIGNALS_SAVE_PATH_PARQUET": lambda: str(cfg.paths.output.signals_parquet),
        "SIGNAL_LOG_PATH": lambda: str(cfg.paths.logs.signal_log),
        "LOG_PATH": lambda: str(cfg.paths.logs.log),
        "FEATURE_STORE_PATH": lambda: str(cfg.paths.cache.feature_store),
        "SCRAPED_HISTORY_PATH": lambda: str(cfg.paths.cache.scraped_history),
        "ALPACA_NEWS_CSV_PATH": lambda: str(cfg.paths.historical.news_csv),
        "ALPACA_NEWS_PARQUET_PATH": lambda: str(cfg.paths.historical.news_parquet),
        "ALPACA_NEWS_CLEAN_PATH": lambda: str(cfg.paths.historical.news_clean),
        "ALPACA_NEWS_PARTITIONED_DIR": lambda: str(
            cfg.paths.historical.partitioned_dir
        ),
        "LLM_MODEL": lambda: cfg.hyperparams.models.llm,
        "EMBEDDING_MODEL": lambda: cfg.hyperparams.models.embedding,
        "SEVERITY_MODEL": lambda: cfg.hyperparams.models.severity_model,
        "OPENAI_MAX_WORKERS": lambda: cfg.hyperparams.models.max_workers,
        "SEVERITY_THRESHOLD": lambda: cfg.hyperparams.lower_path.severity_threshold,
        "MIN_CLUSTER_SIZE": lambda: cfg.hyperparams.lower_path.min_cluster_size,
        "MACRO_SEVERITY_MODEL": lambda: cfg.hyperparams.lower_path.macro_severity_model,
        "MACRO_SEVERITY_PARAMS": lambda: {
            "half_life_hours": cfg.hyperparams.lower_path.half_life_hours
        },
        "CLIMATE_WINDOW_HOURS": lambda: cfg.hyperparams.lower_path.climate_window_hours,
        "DCIM_HALF_LIFE_HOURS": lambda: cfg.hyperparams.upper_path.dcim.half_life_hours,
        "DCIM_BETA_PENALTY": lambda: cfg.hyperparams.upper_path.dcim.beta_penalty,
        "DCIM_GAMMA_CONSENSUS": lambda: cfg.hyperparams.upper_path.dcim.gamma_consensus,
        "DCIM_DELTA_VELOCITY": lambda: cfg.hyperparams.upper_path.dcim.delta_velocity,
        "DCIM_KAPPA_SCALE": lambda: cfg.hyperparams.upper_path.dcim.kappa_scale,
        "DCIM_RECENT_WINDOW_HOURS": lambda: (
            cfg.hyperparams.upper_path.dcim.recent_window_hours
        ),  # noqa: E501
        "SIGNAL_WINDOW_HOURS": lambda: cfg.hyperparams.windows.signal_hours,
        "FEATURE_STORE_MAX_WINDOW_HOURS": lambda: (
            cfg.hyperparams.windows.feature_store_max_hours
        ),  # noqa: E501
        "SIGNAL_NEUTRAL_THRESHOLD": lambda: cfg.hyperparams.signal.neutral_threshold,
        "SIGNAL_MACRO_ALIGN_BONUS": lambda: cfg.hyperparams.signal.macro_align_bonus,
        "SIGNAL_MACRO_CONFLICT_PENALTY": lambda: (
            cfg.hyperparams.signal.macro_conflict_penalty
        ),  # noqa: E501
        "SIGNAL_MACRO_VETO": lambda: cfg.hyperparams.signal.macro_veto,
        "SIGNAL_VETO_COVERAGE": lambda: cfg.hyperparams.signal.veto_coverage,
        "SIGNAL_VETO_STRENGTH_CAP": lambda: cfg.hyperparams.signal.veto_strength_cap,
        "MACRO_CONVICTION_SCALE": lambda: cfg.hyperparams.signal.macro_conviction_scale,
        "SEVERITY_VARIANTS": lambda: cfg.severity.variants_list,
    }


def _get_legacy_value(key: str) -> Any:
    if key in _overrides:
        return _overrides[key]
    resolvers = _build_legacy_resolvers(config)
    if key in resolvers:
        return resolvers[key]()
    raise KeyError(f"Configuration key '{key}' not found.")


class _ConfigMeta(ModelMetaclass):
    def __getitem__(cls, key: str) -> Any:
        return _get_legacy_value(key)

    def __setitem__(cls, key: str, value: Any) -> None:
        _overrides[key] = value

    def __contains__(cls, key: str) -> bool:
        if key in _overrides:
            return True
        return key in _build_legacy_resolvers(config)

    def get(cls, key: str, default: Any = None) -> Any:
        try:
            return cls[key]
        except KeyError:
            return default


class Config(BaseModel, metaclass=_ConfigMeta):
    paths: Paths
    hyperparams: Hyperparams
    severity: Severity

    def __getitem__(self, key: str) -> Any:
        return _get_legacy_value(key)

    def __setitem__(self, key: str, value: Any) -> None:
        _overrides[key] = value

    def __contains__(self, key: str) -> bool:
        if key in _overrides:
            return True
        return key in _build_legacy_resolvers(self)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _load_config() -> Config:
    with open(_YAML_DIR / "paths.yaml") as f:
        paths_raw = yaml.safe_load(f) or {}
    paths_data = paths_raw.get("paths", paths_raw)

    with open(_YAML_DIR / "hyperparameters.yaml") as f:
        hyperparams_raw = yaml.safe_load(f) or {}
    hyperparams_data = hyperparams_raw.get("hyperparams", hyperparams_raw)

    with open(_YAML_DIR / "severity.yaml") as f:
        severity_raw = yaml.safe_load(f) or {}
    severity_data = severity_raw.get("severity", severity_raw)

    combined_config: dict[str, Any] = {
        "paths": paths_data,
        "hyperparams": hyperparams_data,
        "severity": severity_data,
    }

    return Config(**combined_config)


config = _load_config()
