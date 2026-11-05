"""Single-tick orchestrator scoring all severity variants over cached topic features."""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.lower_path.lower_path_analyzer import LowerPathAnalyzer
from nlp_predictor.lower_path.macro_overlay import MacroOverlayAnalyzer
from nlp_predictor.preprocessing.feature_store import FeatureStore
from nlp_predictor.signal.signal_generator import SignalGenerator
from nlp_predictor.signal.signal_log import SignalLog
from nlp_predictor.upper_path.upper_path_analyzer import UpperPathAnalyzer

logger = logging.getLogger(__name__)


class TickRunner:
    """Executes one 15-minute evaluation tick across all configured severity variants.

    Reads cached topic features from :class:`FeatureStore` once per tick, scores each
    upper-path variant deterministically via :meth:`UpperPathAnalyzer.score_variants`,
    computes a single shared macro climate and overlay via :class:`LowerPathAnalyzer`
    and :class:`MacroOverlayAnalyzer`, fuses each variant via :class:`SignalGenerator`,
    and appends all generated signals to :class:`SignalLog`.
    """

    def __init__(
        self,
        *,
        store: FeatureStore,
        upper: UpperPathAnalyzer,
        lower: LowerPathAnalyzer,
        macro: MacroOverlayAnalyzer,
        signal: SignalGenerator,
        log: SignalLog,
        variants: list[dict[str, Any]] | None = None,
        climate_window_hours: float | None = None,
    ) -> None:
        """Initialize the tick runner with injected pipeline collaborators."""
        self.store = store
        self.upper = upper
        self.lower = lower
        self.macro = macro
        self.signal = signal
        self.log = log
        self.variants = (
            variants if variants is not None else config.severity.variants_list
        )
        self.climate_window_hours = (
            float(climate_window_hours)
            if climate_window_hours is not None
            else float(config.hyperparams.lower_path.climate_window_hours)
        )

    @classmethod
    def build_default(cls) -> "TickRunner":
        """Construct a production :class:`TickRunner` with default dependencies."""
        return cls(
            store=FeatureStore(),
            upper=UpperPathAnalyzer.build_default(),
            lower=LowerPathAnalyzer.build_default(),
            macro=MacroOverlayAnalyzer.build_default(),
            signal=SignalGenerator.build_default(),
            log=SignalLog(),
        )

    def run(self, now: datetime | None = None) -> pd.DataFrame:
        """Execute one tick and persist all generated variant signals to :attr:`log`.

        Args:
            now: Reference UTC timestamp for windowing and time decay.

        Returns:
            DataFrame of all signals emitted during this tick, or an empty DataFrame
            if the feature store has no in-window data.
        """
        now = now or datetime.now(UTC)
        tick_id = str(uuid.uuid4())
        tick_time = now.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"====== Tick {tick_id[:8]} @ {tick_time} ======")

        max_lookback = max(self.store.max_window_hours, self.climate_window_hours)
        full_window = self.store.window(max_lookback, now)
        if full_window.empty:
            logger.info("Tick: feature store window is empty; no signals.")
            return pd.DataFrame()

        store_cutoff = now - pd.Timedelta(hours=self.store.max_window_hours)
        window = full_window[full_window["date"] >= store_cutoff].reset_index(drop=True)
        mapped = FeatureStore.to_instrument_frame(window)
        if mapped.empty:
            logger.info("Tick: feature store window has no impacts; no signals.")
            return pd.DataFrame()

        variant_trends = self.upper.score_variants(mapped, self.variants, now)
        if variant_trends.empty:
            logger.info("Tick: no variant produced in-window severity; no signals.")
            return pd.DataFrame()

        climate_cutoff = now - pd.Timedelta(hours=self.climate_window_hours)
        climate_topics = full_window[full_window["date"] >= climate_cutoff].reset_index(
            drop=True
        )
        global_trends = self.lower.process_data(climate_topics, now)
        overlay = self.macro.build(global_trends)

        parts: list[pd.DataFrame] = []
        for variant in self.variants:
            name = variant["name"]
            rows = variant_trends[variant_trends["variant"] == name]
            if rows.empty:
                continue
            signals = self.signal.generate(
                rows, global_trends, overlay.effects, now=now
            )
            if signals.empty:
                continue
            signals = signals.copy()
            signals["variant"] = name
            signals["tick_id"] = tick_id
            signals["tick_time"] = tick_time
            parts.append(signals)

        if not parts:
            logger.info("Tick: variants produced no signals.")
            return pd.DataFrame()

        all_signals = pd.concat(parts, ignore_index=True)
        n = self.log.append(all_signals)
        actionable = int((all_signals["signal"] != "NEUTRAL").sum())
        logger.info(
            f"Tick {tick_id[:8]}: logged {n} signal(s) across "
            f"{len(parts)} variant(s); {actionable} actionable."
        )
        return all_signals


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    logger.info("====== TickRunner Single-Tick Test ======")
    from dotenv import load_dotenv

    load_dotenv(override=True)
    try:
        result = TickRunner.build_default().run()
        if result.empty:
            logger.info(
                "No signals produced. Populate the feature store first "
                "(the tick consumes already-cached article features)."
            )
        else:
            cols = ["variant", "instrument", "signal", "confidence", "macro_context"]
            logger.info(f"\n{result[cols].to_string(index=False)}")
    except Exception as e:
        logger.error(f"TickRunner test failed: {e}", exc_info=True)
