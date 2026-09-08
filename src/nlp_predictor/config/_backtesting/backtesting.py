from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_repo_path(v: Any) -> Path:
    p = Path(v)
    if not p.is_absolute():
        return (REPO_ROOT / p).resolve()
    return p.resolve()


class TimeframeConfig(BaseModel):
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"


class BacktestingDataConfig(BaseModel):
    market_prices_dir: Path = Path("data/historical/market_prices")
    macro_tickers: list[str] = []

    @field_validator("market_prices_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        return _resolve_repo_path(v)


class BacktestingConfig(BaseModel):
    timeframe: TimeframeConfig = TimeframeConfig()
    data: BacktestingDataConfig = BacktestingDataConfig()
