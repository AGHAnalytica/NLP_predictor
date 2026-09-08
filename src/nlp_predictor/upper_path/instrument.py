"""Backwards-compatibility alias for models.py."""

from nlp_predictor.upper_path.models import (
    InstrumentMapping,
    InstrumentTrend,
    TickerImpact,
)

__all__ = ["TickerImpact", "InstrumentMapping", "InstrumentTrend"]

