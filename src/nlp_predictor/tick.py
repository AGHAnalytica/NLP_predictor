"""
Tick runner — one 15-min cycle of the live A/B/n loop.

A "tick" is the unit the experiment runs on: read the rolling feature store, score every
severity VARIANT on the SAME cached window, fuse each with one SHARED macro climate, and
append every resulting signal to the eval log. Routing each variant's signal to its own
demo account (step 5) then makes realized PnL the ground truth — and because all
variants saw byte-identical inputs this tick, PnL attributes cleanly to the
(model, params, window) triple and nothing else.

What makes a tick cheap (the whole reason the feature store exists): the expensive,
stochastic work — topic extraction, embeddings, topic->instrument mapping — was paid for
ONCE when each article entered the store, and is reused for every tick the article stays
in-window (~16 of them). So a tick recomputes only:
  1. severity per (variant, window)             — deterministic, no LLM
  2. the macro climate + per-instrument overlay — ONE shared LLM call, not per-variant
Keeping the macro overlay shared (not per-variant) is what stops it confounding the
comparison: it is a constant across the variants being scored this tick.

Flow::

    feature store window ─┬─> score_variants ─────────────> per-(variant,instrument) sev
    (cached features)     └─> lower path -> macro overlay ─> shared climate (1 LLM call)
                                            │
        for each variant: SignalGenerator.generate(variant rows, climate) -> signals
                                            │
                                  SignalLog.append(all signals, tagged tick_id)

The runner is pure orchestration with every collaborator injected, so it unit-tests with
fakes (no network); the only LLM-bound pieces (lower path + macro overlay) are the two
stubbed in tests.
"""

import uuid
from datetime import UTC, datetime

import pandas as pd

from nlp_predictor.config import Config  # type: ignore
from nlp_predictor.feature_store import FeatureStore
from nlp_predictor.global_trends import LowerPathAnalyzer
from nlp_predictor.instrument_trends import UpperPathAnalyzer
from nlp_predictor.macro_overlay import MacroOverlayAnalyzer
from nlp_predictor.signal_generator import SignalGenerator
from nlp_predictor.signal_log import SignalLog


class TickRunner:
    """
    Runs one tick: cached features -> per-variant signals -> eval log.

    Collaborators are injected so the runner stays a thin, testable orchestrator:
    - ``store``  : the rolling :class:`FeatureStore` (already-cached topic features).
    - ``upper``  : :class:`UpperPathAnalyzer`, used ONLY via the LLM-free
      :meth:`UpperPathAnalyzer.score_variants`.
    - ``lower``  : :class:`LowerPathAnalyzer`, clusters the window into the macro
      climate.
    - ``macro``  : :class:`MacroOverlayAnalyzer`, resolves the climate to per-instrument
      effects (one shared LLM call).
    - ``signal`` : :class:`SignalGenerator`, fuses each variant with the climate.
    - ``log``    : the append-only :class:`SignalLog` (the eval dataset).

    Use :meth:`build_default` for the real, wired stack; construct directly with fakes
    in tests.
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
        variants: list[dict] | None = None,
        climate_window_hours: float | None = None,
    ):
        self.store = store
        self.upper = upper
        self.lower = lower
        self.macro = macro
        self.signal = signal
        self.log = log
        self.variants = (
            variants if variants is not None else Config["SEVERITY_VARIANTS"]
        )
        self.climate_window_hours = (
            float(climate_window_hours)
            if climate_window_hours is not None
            else float(Config["SIGNAL_WINDOW_HOURS"])
        )

    @classmethod
    def build_default(cls) -> "TickRunner":
        """Wire the real production stack (needs OPENAI_API_KEY for the LLM pieces)."""
        return cls(
            store=FeatureStore(),
            upper=UpperPathAnalyzer(),
            lower=LowerPathAnalyzer(),
            macro=MacroOverlayAnalyzer(),
            signal=SignalGenerator(),
            log=SignalLog(),
        )

    def run(self, now: datetime | None = None) -> pd.DataFrame:
        """
        Execute one tick and append its signals to the log.

        Args:
            now (datetime | None): Reference time for windowing/decay; defaults to
                ``datetime.now(UTC)``. Injectable so ticks are reproducible in tests.

        Returns:
            pd.DataFrame: Every signal produced this tick (one row per
            (variant, instrument)), already tagged with ``tick_id``/``tick_time``/
            ``variant`` and persisted. Empty if the store had no in-window data.
        """
        now = now or datetime.now(UTC)
        tick_id = str(uuid.uuid4())
        tick_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n====== Tick {tick_id[:8]} @ {tick_time} ======")

        # 1) Read the widest retained window once; each variant slices its own
        #    sub-window inside score_variants, so the store is read once per tick.
        window = self.store.window(self.store.max_window_hours, now)
        mapped = FeatureStore.to_instrument_frame(window)
        if mapped.empty:
            print("[INFO] Tick: feature store window is empty; no signals.")
            return pd.DataFrame()

        # 2) Cheap, deterministic per-variant severity over the cached impacts (no LLM).
        variant_trends = self.upper.score_variants(mapped, self.variants, now)
        if variant_trends.empty:
            print("[INFO] Tick: no variant produced in-window severity; no signals.")
            return pd.DataFrame()

        # 3) ONE shared climate for the whole tick: cluster the standard signal window,
        #    then resolve it to a per-instrument overlay (the single LLM call here).
        #    Being shared across variants, it cannot bias which variant wins the A/B.
        climate_topics = self.store.window(self.climate_window_hours, now)
        global_trends = self.lower.process_data(climate_topics)
        overlay = self.macro.build(global_trends)

        # 4) Fuse each variant against that shared climate; tag with the tick identity.
        parts: list[pd.DataFrame] = []
        for variant in self.variants:
            name = variant["name"]
            rows = variant_trends[variant_trends["variant"] == name]
            if rows.empty:
                continue
            signals = self.signal.generate(rows, global_trends, overlay.effects)
            if signals.empty:
                continue
            signals = signals.copy()
            signals["variant"] = name
            signals["tick_id"] = tick_id
            signals["tick_time"] = tick_time
            parts.append(signals)

        if not parts:
            print("[INFO] Tick: variants produced no signals.")
            return pd.DataFrame()

        all_signals = pd.concat(parts, ignore_index=True)
        n = self.log.append(all_signals)
        actionable = int((all_signals["signal"] != "NEUTRAL").sum())
        print(
            f"[INFO] Tick {tick_id[:8]}: logged {n} signal(s) across "
            f"{len(parts)} variant(s); {actionable} actionable."
        )
        return all_signals


if __name__ == "__main__":
    print("====== TickRunner Single-Tick Test ======")
    from dotenv import load_dotenv

    load_dotenv(override=True)
    try:
        result = TickRunner.build_default().run()
        if result.empty:
            print(
                "[INFO] No signals produced. Populate the feature store first "
                "(the tick consumes already-cached article features)."
            )
        else:
            cols = ["variant", "instrument", "signal", "confidence", "macro_context"]
            print(result[cols].to_string(index=False))
    except Exception as e:
        print(f"[FATAL ERROR] TickRunner test failed: {e}")
