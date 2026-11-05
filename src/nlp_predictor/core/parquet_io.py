"""Shared Parquet I/O and JSON-column serialization helpers for cache and log stores."""

import json
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd


def serialize_json_cell(value: object, empty_default: str = "null") -> str:
    """Serialize a list or dict cell into a JSON string for Parquet storage.

    Args:
        value: Python object (list, dict, tuple, set, ndarray, or JSON string).
        empty_default: Fallback JSON literal when ``value`` is ``None`` or ``NaN``.

    Returns:
        JSON-encoded string representation.
    """
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return empty_default
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, (tuple, set)):
        value = list(value)
    return json.dumps(value)


def deserialize_json_cell(value: object, error_default: object = None) -> object:
    """Deserialize a JSON string or PyArrow array cell back into Python lists/dicts.

    Args:
        value: Stored cell value (JSON string, ndarray, list, or dict).
        error_default: Fallback value returned when JSON parsing fails.

    Returns:
        Deserialized Python object.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return error_default
    return value


def read_raw_parquet(path: str, columns: Sequence[str]) -> pd.DataFrame:
    """Read a Parquet file if it exists, or return an empty DataFrame with ``columns``.

    Args:
        path: Filesystem path to the Parquet file.
        columns: Expected schema column names.

    Returns:
        Loaded DataFrame or empty DataFrame conforming to ``columns``.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(columns))
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=list(columns))
    return df


def write_raw_parquet(df: pd.DataFrame, path: str, columns: Sequence[str]) -> None:
    """Write ``df`` reindexed to ``columns`` into a Parquet file, creating directories.

    Args:
        df: DataFrame to persist.
        path: Destination Parquet file path.
        columns: Schema column ordering.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.reindex(columns=list(columns)).to_parquet(path, index=False, engine="pyarrow")
