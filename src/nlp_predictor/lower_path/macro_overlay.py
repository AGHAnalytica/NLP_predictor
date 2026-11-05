"""Macroeconomic climate overlay mapping lower-path narratives to per-instrument
tailwind/headwind effects.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from nlp_predictor.config import config
from nlp_predictor.core.base_analyzer import BaseAnalyzer, load_instrument_universe
from nlp_predictor.prompts import PROMPTS

logger = logging.getLogger(__name__)

_VALID_EFFECTS = ("tailwind", "headwind", "neutral")


class _InstrumentEffect(BaseModel):
    """Single instrument's reaction to the macroeconomic climate."""

    ticker: str = Field(description="Exact ticker from the provided list.")
    effect: str = Field(description="One of: 'tailwind', 'headwind', 'neutral'.")
    strength: float = Field(description="Conviction of the effect, 0.0-1.0.")
    rationale: str = Field(description="One short sentence.")


class _MacroMapping(BaseModel):
    """Structured LLM output containing climate effects for all instruments."""

    effects: list[_InstrumentEffect] = Field(description="One entry per instrument.")


@dataclass
class MacroOverlay:
    """Resolved macro climate containing per-instrument effects and overall climate
    conviction.
    """

    effects: dict[str, dict[str, Any]] = field(default_factory=dict)
    conviction: float = 0.0
    climate_summary: str = ""

    def for_instrument(self, instrument: str) -> dict[str, Any]:
        """Return ``{'effect', 'strength', 'rationale'}`` for ``instrument`` (defaults
        to
        neutral).
        """
        return self.effects.get(
            instrument, {"effect": "neutral", "strength": 0.0, "rationale": ""}
        )


class MacroOverlayAnalyzer(BaseAnalyzer):
    """Resolves lower-path macro clusters into a shared per-instrument ``MacroOverlay``
    via
    one LLM call.
    """

    def __init__(self, instruments_path: str = str(config.paths.input.instruments)):
        """Initialize the analyzer and load the tradable instrument universe from CSV.

        Args:
            instruments_path: Path to the instruments CSV file.
        """
        super().__init__()
        (
            self.instruments_df,
            self.valid_tickers,
            self.available_instruments,
        ) = load_instrument_universe(instruments_path)

    @classmethod
    def build_default(cls) -> "MacroOverlayAnalyzer":
        """Construct a default :class:`MacroOverlayAnalyzer` using configured
        instruments.
        """
        return cls(str(config.paths.input.instruments))

    def build(self, global_trends_df: pd.DataFrame | None) -> MacroOverlay:
        """Build a ``MacroOverlay`` from the lower-path ``global_trends_df``.

        Args:
            global_trends_df: Lower-path clusters with ``title``, ``direction``, and
                ``severity`` columns.

        Returns:
            Populated ``MacroOverlay`` (or an empty overlay with 0.0 conviction if input
            is empty).
        """
        if global_trends_df is None or global_trends_df.empty:
            logger.info("No macro trends; macro overlay is empty.")
            return MacroOverlay()

        climate = self._climate_summary(global_trends_df)
        conviction = self._conviction(global_trends_df)
        effects = self._map_climate(climate)
        logger.info(
            f"Macro overlay: {len(effects)} instruments mapped, "
            f"climate conviction {conviction:.2f}."
        )
        return MacroOverlay(
            effects=effects, conviction=conviction, climate_summary=climate
        )

    def _climate_summary(self, global_trends_df: pd.DataFrame) -> str:
        """Format macro clusters into a bulleted text summary for the LLM prompt."""
        lines: list[str] = []
        for row in global_trends_df.to_dict(orient="records"):
            title = str(row.get("title", "macro trend"))
            direction = str(row.get("direction", "neutral")).lower()
            count = row.get("n_topics", row.get("severity", 0))
            description = str(row.get("description", "")).strip()
            suffix = f": {description}" if description else ""
            lines.append(f"- {title} (direction: {direction}, {count} topics){suffix}")
        return "\n".join(lines)

    def _conviction(self, global_trends_df: pd.DataFrame) -> float:
        """Compute aggregate climate conviction in ``[0.0, 1.0]`` as $\\tanh(\\sum
        \\text{severity} / \\text{scale})$.
        """
        total = float(
            pd.to_numeric(global_trends_df["severity"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        scale = float(config.hyperparams.signal.macro_conviction_scale)
        return float(math.tanh(total / scale)) if scale > 0 else 0.0

    def _map_climate(self, climate: str) -> dict[str, dict[str, Any]]:
        """Query the LLM once to classify each instrument's climate effect and validate
        tickers.
        """
        prompt = PROMPTS["macro_overlay"].format(
            climate=climate, instruments=self._instrument_block()
        )
        try:
            parsed = self._call_openai_api_with_retry(prompt, _MacroMapping)
        except Exception as e:
            logger.error(f"Macro overlay mapping failed: {e}")
            return {}

        effects: dict[str, dict[str, Any]] = {}
        for item in parsed.effects:  # type: ignore[attr-defined]
            ticker = str(item.ticker).strip()
            effect = str(item.effect).strip().lower()
            if ticker in self.valid_tickers and effect in _VALID_EFFECTS:
                effects[ticker] = {
                    "effect": effect,
                    "strength": max(0.0, min(1.0, float(item.strength))),
                    "rationale": str(item.rationale).strip(),
                }
        return effects

    def _instrument_block(self) -> str:
        """Format available instruments as ``- TICKER — Full Name`` lines."""
        return "\n".join(
            f"- {ticker} — {name}"
            for ticker, name in self.available_instruments.items()
        )
