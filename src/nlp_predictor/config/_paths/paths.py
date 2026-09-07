from pathlib import Path

from pydantic import BaseModel


class InputPaths(BaseModel):
    instruments: Path
    topics: Path
    articles: Path
    weights: Path


class OutputPaths(BaseModel):
    topics_csv: Path
    topics_parquet: Path
    global_trends_csv: Path
    global_trends_parquet: Path
    instrument_trends_csv: Path
    instrument_trends_parquet: Path
    signals_csv: Path
    signals_parquet: Path


class LogsPaths(BaseModel):
    log: Path
    signal_log: Path


class CachePaths(BaseModel):
    feature_store: Path


class Paths(BaseModel):
    input: InputPaths
    output: OutputPaths
    logs: LogsPaths
    cache: CachePaths
