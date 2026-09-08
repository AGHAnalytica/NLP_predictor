"""
Macro overlay — the bridge between the lower path's broad CLIMATE and each instrument.

The lower path (Group 4) answers "w jakim otoczeniu" — is the world risk-on or fleeing
to safety. The job here is to turn that one climate into a per-instrument effect so the
final signal can be modulated with context: a call that rides the climate is more
trustworthy, one that fights it is less (a good long still falls in a panic).

Why an LLM here, not a hand-coded rule: the same climate pushes different instruments in
*opposite* directions (risk-on lifts equities but sinks gold), and the sign is
regime-dependent (gold can fall in a dash-for-cash). A static ``risk_class`` sign-flip
is brittle and, if wrong, inverts the signal in exactly the tail cases that matter. One
structured LLM call reads the narrative and returns ``tailwind | headwind | neutral`` +
strength per instrument — the same judgement the upper path already makes per topic,
applied one level up. A probe (scripts/probe_macro_mapping.py) confirmed the configured
model handles the canonical mappings, including the haven inversions.

Two deliberate properties keep this safe and clean:
- It is computed ONCE per tick, shared across every A/B severity variant — so it is a
  constant across the comparison and cannot confound which severity model wins.
- It only ever modulates CONFIDENCE downstream, never direction (direction stays the
  upper-path severity sign), so a bad tag mis-sizes a bet but never flips it. We bias
  the model toward ``neutral`` so its errors fail toward "no opinion", the safe default.

Output (:class:`MacroOverlay`):
    - effects (dict)     : ``{instrument: {'effect', 'strength', 'rationale'}}``
    - conviction (float) : [0, 1] overall climate strength (the quantitative macro
                           severity) — scales how hard the overlay moves confidence.
    - climate_summary    : the text fed to the model, for audit/logging.
"""

import logging
import math
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, Field

from nlp_predictor.base_analyzer import BaseAnalyzer
from nlp_predictor.config import Config

logger = logging.getLogger(__name__)

_VALID_EFFECTS = ("tailwind", "headwind", "neutral")


class _InstrumentEffect(BaseModel):
    """One instrument's response to the climate (strict LLM output)."""

    ticker: str = Field(description="Exact ticker from the provided list.")
    effect: str = Field(description="One of: 'tailwind', 'headwind', 'neutral'.")
    strength: float = Field(description="Conviction of the effect, 0.0-1.0.")
    rationale: str = Field(description="One short sentence.")


class _MacroMapping(BaseModel):
    """Container for the per-instrument effects."""

    effects: list[_InstrumentEffect] = Field(description="One entry per instrument.")


@dataclass
class MacroOverlay:
    """The macro climate resolved to per-instrument effects + an overall conviction."""

    effects: dict[str, dict] = field(default_factory=dict)
    conviction: float = 0.0
    climate_summary: str = ""

    def for_instrument(self, instrument: str) -> dict:
        """Effect for one instrument, defaulting to neutral/no-effect if unmapped."""
        return self.effects.get(
            instrument, {"effect": "neutral", "strength": 0.0, "rationale": ""}
        )

# TODO: move prompt to config
_PROMPT = """You are a macro strategist. Below is the current market CLIMATE, then a
list of instruments. For EACH instrument, decide how this climate affects its price over
the next ~12-16 hours:
- "tailwind" = the climate pushes its price UP
- "headwind" = the climate pushes its price DOWN
- "neutral"  = no clear, direct effect
Return an effect, a strength (0.0-1.0) and a one-line rationale for every instrument.
Guidance: risk assets (equities, oil, crypto) tend to RISE in optimistic/risk-on
climates and FALL in fearful/risk-off ones. Safe havens (gold, government bonds, the US
dollar) tend to move the OPPOSITE way, rising when fearful. Weigh what the climate is
specifically about, not just whether it is positive or negative. Only choose
tailwind/headwind when the effect is clear and direct; otherwise neutral.

CLIMATE:
{climate}

INSTRUMENTS:
{instruments}
"""


class MacroOverlayAnalyzer(BaseAnalyzer):
    """
    Builds the :class:`MacroOverlay` from the lower-path macro trends in one LLM call.

    Pure orchestration around a single structured, validated call: it does not touch
    the upper path or compute any severity itself. ``process_data`` exists only for the
    BaseAnalyzer interface; the real entry point is :meth:`build`.
    """

    def __init__(self, instruments_path: str = Config["INSTRUMENT_PATH"]):
        super().__init__()
        self.instruments_df = pd.read_csv(instruments_path)
        self.valid_tickers = set(self.instruments_df["ticker"].dropna().astype(str))
        self.available_instruments = dict(
            zip(
                self.instruments_df["ticker"].astype(str),
                self.instruments_df["full_name"].astype(str),
                strict=False,
            )
        )

    def process_data(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """Unused BaseAnalyzer hook; the overlay is produced by :meth:`build`."""
        raise NotImplementedError("Use build(global_trends_df) instead.")

    def build(self, global_trends_df: pd.DataFrame | None) -> MacroOverlay:
        """
        Resolve the macro climate to a per-instrument overlay.

        Args:
            global_trends_df (pd.DataFrame | None): Lower-path output, with at least
                'title', 'direction' and 'severity' columns. Empty/None -> an empty
                overlay (no macro modulation), which downstream treats as "no climate".

        Returns:
            MacroOverlay: effects + conviction + the climate text used.
        """
        if global_trends_df is None or global_trends_df.empty:
            logger.info("No macro trends; macro overlay is empty (no climate).")
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
        """Render the macro clusters into the climate text the model reads."""
        lines = []
        for _, row in global_trends_df.iterrows():
            title = str(row.get("title", "macro trend"))
            direction = str(row.get("direction", "neutral")).lower()
            # Human/LLM-facing size is the raw topic count; the weighted 'severity' is
            # for the conviction math, not for display (a "5.3 topics" reads wrong).
            count = row.get("n_topics", row.get("severity", 0))
            description = str(row.get("description", "")).strip()
            lines.append(
                f"- {title} (direction: {direction}, {count} topics)"
                + (f": {description}" if description else "")
            )
        return "\n".join(lines)

    def _conviction(self, global_trends_df: pd.DataFrame) -> float:
        """
        Overall climate strength in [0, 1] — the quantitative macro severity.

        v1 proxy: tanh of the total clustered topic volume (sum of per-cluster
        severities). Bigger, broader clusters -> a more pervasive, more trustworthy
        climate. This is the simple, defensible starting point; a richer macro-severity
        model (source diversity, embedding cohesion, recency) can replace it later
        without touching the fusion, since only this scalar is consumed downstream.
        """
        total = float(
            pd.to_numeric(global_trends_df["severity"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        scale = float(Config["MACRO_CONVICTION_SCALE"])
        return float(math.tanh(total / scale)) if scale > 0 else 0.0

    def _map_climate(self, climate: str) -> dict[str, dict]:
        """One structured LLM call: climate -> per-instrument effect, validated."""
        prompt = _PROMPT.format(climate=climate, instruments=self._instrument_block())
        try:
            parsed = self._call_openai_api_with_retry(prompt, _MacroMapping)
        except Exception as e:  # noqa: BLE001 - a failed overlay just means no climate
            logger.error(f"Macro overlay mapping failed: {e}")
            return {}

        effects: dict[str, dict] = {}
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
        """The 'TICKER — Full Name' lines the model classifies."""
        return "\n".join(
            f"- {ticker} — {name}"
            for ticker, name in self.available_instruments.items()
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("====== MacroOverlayAnalyzer Module Test ======")
    try:
        gt = pd.read_csv(Config["GLOBAL_TRENDS_SAVE_PATH_CSV"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        gt = pd.DataFrame()
    overlay = MacroOverlayAnalyzer().build(gt)
    logger.info(f"Conviction: {overlay.conviction:.2f}")
    for inst, eff in overlay.effects.items():
        logger.info(
            f"  {inst:8s} {eff['effect']:9s} {eff['strength']:.2f}  {eff['rationale']}"
        )
