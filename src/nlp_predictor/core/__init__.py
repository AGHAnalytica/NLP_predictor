"""Core infrastructure: base LLM analyzer, severity models, and logging setup."""

from .base_analyzer import (
    BaseAnalyzer,
    _is_transient_rate_limit,
    load_instrument_universe,
)
from .logging_config import setup_logging
from .parquet_io import (
    deserialize_json_cell,
    read_raw_parquet,
    serialize_json_cell,
    write_raw_parquet,
)
from .severity_models import (
    MACRO_SEVERITY_REGISTRY,
    REQUIRED_COLUMNS,
    SEVERITY_REGISTRY,
    macro_severity_count,
    macro_severity_decayed,
    severity_abnormal_volume,
    severity_baseline,
    severity_dcim,
)

__all__ = [
    "BaseAnalyzer",
    "_is_transient_rate_limit",
    "load_instrument_universe",
    "setup_logging",
    "serialize_json_cell",
    "deserialize_json_cell",
    "read_raw_parquet",
    "write_raw_parquet",
    "SEVERITY_REGISTRY",
    "MACRO_SEVERITY_REGISTRY",
    "REQUIRED_COLUMNS",
    "severity_baseline",
    "severity_dcim",
    "severity_abnormal_volume",
    "macro_severity_count",
    "macro_severity_decayed",
]
