"""Filesystem path schemas resolved relative to the repository root."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_repo_path(v: Any) -> Path:
    """Resolve relative paths against ``REPO_ROOT`` into absolute ``Path`` objects."""
    p = Path(v)
    if not p.is_absolute():
        return (REPO_ROOT / p).resolve()
    return p.resolve()


class InputPaths(BaseModel):
    """Static reference tables and latest scrape inputs."""

    instruments: Path
    topics: Path
    articles: Path
    weights: Path

    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        """Resolve all input paths relative to the repository root."""
        return _resolve_repo_path(v)


class OutputPaths(BaseModel):
    """CSV and Parquet destinations for pipeline analytical outputs."""

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
        """Resolve all output paths relative to the repository root."""
        return _resolve_repo_path(v)


class LogsPaths(BaseModel):
    """Application log file template and append-only signal evaluation dataset."""

    log: Path
    signal_log: Path

    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        """Resolve all log paths relative to the repository root."""
        return _resolve_repo_path(v)


class CachePaths(BaseModel):
    """Rolling Parquet caches for topic features and scraper deduplication."""

    feature_store: Path
    scraped_history: Path = Path("data/scraping/scraped_history.parquet")

    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        """Resolve all cache paths relative to the repository root."""
        return _resolve_repo_path(v)


class HistoricalPaths(BaseModel):
    """Storage paths for raw, partitioned, and cleaned historical news archives."""

    news_csv: Path = Path("data/historical/alpaca_news.csv")
    news_parquet: Path = Path("data/historical/alpaca_news.parquet")
    news_clean: Path = Path("data/historical/alpaca_news_clean.parquet")
    partitioned_dir: Path = Path("data/historical/alpaca_news_partitioned")

    @field_validator("*", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        """Resolve all historical data paths relative to the repository root."""
        return _resolve_repo_path(v)


class Paths(BaseModel):
    """Aggregated filesystem path configuration across all pipeline stages."""

    input: InputPaths
    output: OutputPaths
    logs: LogsPaths
    cache: CachePaths
    historical: HistoricalPaths = HistoricalPaths()
