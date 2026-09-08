from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_repo_path(v: Any) -> Path:
    p = Path(v)
    if not p.is_absolute():
        return (REPO_ROOT / p).resolve()
    return p.resolve()


class InputPaths(BaseModel):
    instruments: Path
    topics: Path
    articles: Path
    weights: Path

    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        return _resolve_repo_path(v)


class OutputPaths(BaseModel):
    topics_csv: Path
    topics_parquet: Path
    global_trends_csv: Path
    global_trends_parquet: Path
    instrument_trends_csv: Path
    instrument_trends_parquet: Path
    signals_csv: Path
    signals_parquet: Path

    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        return _resolve_repo_path(v)


class LogsPaths(BaseModel):
    log: Path
    signal_log: Path

    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        return _resolve_repo_path(v)


class CachePaths(BaseModel):
    feature_store: Path
    scraped_history: Path = Path("data/scraped_history.parquet")

    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        return _resolve_repo_path(v)


class HistoricalPaths(BaseModel):
    news_csv: Path = Path("data/historical/alpaca_news.csv")
    news_parquet: Path = Path("data/historical/alpaca_news.parquet")
    news_clean: Path = Path("data/historical/alpaca_news_clean.parquet")
    partitioned_dir: Path = Path("data/historical/alpaca_news_partitioned")

    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        return _resolve_repo_path(v)


class Paths(BaseModel):
    input: InputPaths
    output: OutputPaths
    logs: LogsPaths
    cache: CachePaths
    historical: HistoricalPaths = HistoricalPaths()
