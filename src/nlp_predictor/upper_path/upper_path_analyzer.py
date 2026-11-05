"""Upper-path analysis: topic-to-instrument mapping, sentiment quantization, and per-
variant severity scoring.
"""

import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.core.base_analyzer import BaseAnalyzer, load_instrument_universe
from nlp_predictor.core.severity_models import (
    REQUIRED_COLUMNS,
    SEVERITY_REGISTRY,
    severity_baseline,
    severity_dcim,
)
from nlp_predictor.prompts import PROMPTS
from nlp_predictor.upper_path.models import (
    InstrumentMapping,
    InstrumentTrend,
    _quantize_sentiment,
)

logger = logging.getLogger(__name__)


class UpperPathAnalyzer(BaseAnalyzer):
    """Maps news topics to financial instruments and computes directional severity
    scores.

    Supports three execution modes:
        - ``score_variants``: LLM-free live-tick hot path scoring pre-mapped
          feature-store rows across A/B/n variants.
        - ``generate_variant_trends``: Offline multi-variant scoring with shared LLM
          instrument summaries.
        - ``process_data``: Single-model end-to-end mapping, DCIM/baseline scoring, and
          LLM summary generation.
    """

    def __init__(self, instruments_path: str = str(config.paths.input.instruments)):
        """Initialize the analyzer and load the valid instrument universe from CSV.

        Args:
            instruments_path: Path to the instruments CSV file (``ticker``,
                ``full_name``).
        """
        super().__init__()
        (
            self.instruments_df,
            self.valid_tickers,
            self.available_instruments,
        ) = load_instrument_universe(instruments_path)
        self._mapping_failures: list[str] = []

    @classmethod
    def build_default(cls) -> "UpperPathAnalyzer":
        """Construct a default :class:`UpperPathAnalyzer` from config."""
        return cls(str(config.paths.input.instruments))

    def map_topics(self, topics_df: pd.DataFrame) -> list[list[dict[str, Any]]]:
        """Map each topic row in ``topics_df`` to its impacted instruments and quantized
        sentiments concurrently.

        Args:
            topics_df: DataFrame with ``topic_name`` and ``topic_description`` columns.

        Returns:
            List aligned 1:1 with ``topics_df`` rows, where each element is a list of
            ``{'instrument': str, 'sentiment': float}`` dicts.
        """
        if topics_df.empty:
            return []

        self._mapping_failures = []
        topic_rows = topics_df.to_dict(orient="records")
        impacts = self._parallel_map(self._map_topic_to_instrument, topic_rows)

        if self._mapping_failures:
            logger.warning(
                f"{len(self._mapping_failures)}/{len(topic_rows)} topic mappings "
                "failed; instrument signals may be incomplete."
            )
        return impacts

    def process_data(
        self, topics_df: pd.DataFrame, now: datetime | None = None
    ) -> pd.DataFrame:
        """Map topics to instruments, compute single-model severity, and generate LLM
        summaries.

        Args:
            topics_df: Enriched topics DataFrame with ``topic_id``, ``topic_name``,
                ``topic_description``, ``date``, ``source_name``, and ``source_weight``.
            now: Reference UTC timestamp for age decay and trend timestamping.

        Returns:
            DataFrame of per-instrument trends with columns ``trend_id``,
            ``current_time``,
            ``first_article_time``, ``newest_article_time``, ``instrument``, ``title``,
            ``description``, ``severity``, ``direction``, ``llm_direction``, and
            ``topic_ids``.
        """
        if topics_df.empty:
            return pd.DataFrame()

        self.current_time = now or datetime.now(UTC)
        mapped_df = self._map_and_explode(topics_df)
        if mapped_df.empty:
            return pd.DataFrame()

        logger.info("Generating trends per instrument...")

        def _trend_for_instrument(item: tuple[Any, pd.DataFrame]) -> dict[str, Any]:
            instrument, group = item
            severity = self._calculate_instrument_severity(group)
            return self._generate_instrument_trend(str(instrument), group, severity)

        results = self._parallel_map(
            _trend_for_instrument, list(mapped_df.groupby("instrument"))
        )
        return pd.DataFrame(results)

    def _map_and_explode(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """Map topics to instruments and explode into long-form ``(topic, instrument)``
        rows.
        """
        logger.info("Mapping topics to financial instruments...")
        topics_copy = topics_df.copy()
        topics_copy["impacts"] = self.map_topics(topics_copy)

        exploded_df = topics_copy.explode("impacts").dropna(subset=["impacts"])
        if exploded_df.empty:
            logger.warning("No topics mapped to valid instruments.")
            return pd.DataFrame()

        impacts_expanded = pd.DataFrame.from_records(
            exploded_df["impacts"].tolist(), index=exploded_df.index
        )
        mapped_df = pd.concat(
            [exploded_df.drop(columns=["impacts"]), impacts_expanded], axis=1
        )
        mapped_df["instrument"] = mapped_df["instrument"].astype(str)
        mapped_df["sentiment"] = mapped_df["sentiment"].astype(float)
        mapped_df["date"] = pd.to_datetime(mapped_df["date"], utc=True)
        return mapped_df

    def generate_variant_trends(
        self,
        topics_df: pd.DataFrame,
        variants: list[dict[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        """Score all severity variants over ``topics_df`` with shared LLM mapping and
        summaries.

        Args:
            topics_df: Enriched topics DataFrame.
            variants: Variant definitions (defaults to
                ``config.severity.variants_list``).
            now: Reference UTC timestamp for windowing and time decay.

        Returns:
            Long-form DataFrame with one row per ``(variant, instrument)``.
        """
        variants = variants if variants is not None else config.severity.variants_list
        if topics_df.empty:
            return pd.DataFrame()

        self.current_time = now or datetime.now(UTC)
        mapped_df = self._map_and_explode(topics_df)
        if mapped_df.empty:
            return pd.DataFrame()

        summaries = self._summaries_by_instrument(mapped_df)

        rows: list[dict[str, Any]] = []
        for variant in variants:
            window_df = self._window_frame(mapped_df, variant["window_hours"])
            if window_df.empty:
                continue
            context = self._variant_context(window_df)
            model_fn = SEVERITY_REGISTRY[variant["model"]]
            params = variant.get("params", {})
            for instrument, group in window_df.groupby("instrument"):
                severity = self._severity_for(model_fn, group, params, context)
                rows.append(
                    self._variant_trend_row(
                        variant, str(instrument), group, severity, summaries
                    )
                )

        logger.info(
            f"Generated {len(rows)} variant trend rows "
            f"across {len(variants)} variant(s)."
        )
        return pd.DataFrame(rows)

    def score_variants(
        self,
        mapped_df: pd.DataFrame,
        variants: list[dict[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        """Compute deterministic per-variant severity over pre-mapped feature-store
        impacts (no LLM calls).

        Args:
            mapped_df: Pre-mapped impacts from ``FeatureStore.to_instrument_frame`` with
                columns
                ``instrument``, ``sentiment``, ``date``, ``source_name``,
                ``source_weight``, ``topic_id``.
            variants: Variant definitions (defaults to
                ``config.severity.variants_list``).
            now: Reference UTC timestamp for windowing and time decay.

        Returns:
            DataFrame with columns ``variant``, ``model``, ``window_hours``,
            ``current_time``,
            ``instrument``, ``severity``, ``direction``, and ``topic_ids``.
        """
        variants = variants if variants is not None else config.severity.variants_list
        if mapped_df is None or mapped_df.empty:
            return pd.DataFrame()

        self.current_time = now or datetime.now(UTC)
        mapped_df = mapped_df.copy()
        mapped_df["date"] = pd.to_datetime(mapped_df["date"], utc=True)

        rows: list[dict[str, Any]] = []
        for variant in variants:
            window_df = self._window_frame(mapped_df, variant["window_hours"])
            if window_df.empty:
                continue
            context = self._variant_context(window_df)
            model_fn = SEVERITY_REGISTRY[variant["model"]]
            params = variant.get("params", {})
            for instrument, group in window_df.groupby("instrument"):
                severity = self._severity_for(model_fn, group, params, context)
                sev_direction = (
                    "up" if severity > 0 else "down" if severity < 0 else "neutral"
                )
                rows.append(
                    {
                        "variant": variant["name"],
                        "model": variant["model"],
                        "window_hours": variant["window_hours"],
                        "current_time": self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "instrument": str(instrument),
                        "severity": round(severity, 4),
                        "direction": sev_direction,
                        "topic_ids": group["topic_id"].tolist(),
                    }
                )

        logger.info(
            f"Scored {len(rows)} (variant, instrument) pairs "
            f"across {len(variants)} variant(s)."
        )
        return pd.DataFrame(rows)

    def _window_frame(self, mapped_df: pd.DataFrame, hours: float) -> pd.DataFrame:
        """Filter ``mapped_df`` to rows within ``hours`` of ``self.current_time``."""
        cutoff = self.current_time - pd.Timedelta(hours=float(hours))
        return mapped_df[mapped_df["date"] >= cutoff]

    @staticmethod
    def _variant_context(window_df: pd.DataFrame) -> dict[str, float]:
        """Compute cross-instrument context (mean article count per instrument) for
        ``abnormal_volume``.
        """
        counts = window_df.groupby("instrument").size()
        baseline = float(counts.mean()) if len(counts) else 1.0
        return {"baseline_count": baseline}

    def _severity_for(
        self,
        model_fn: Any,
        group: pd.DataFrame,
        params: dict[str, Any],
        context: dict[str, float],
    ) -> float:
        """Evaluate ``model_fn`` on ``group`` if all ``REQUIRED_COLUMNS`` are present,
        else
        return 0.0.
        """
        if group.empty:
            return 0.0
        if not all(col in group.columns for col in REQUIRED_COLUMNS):
            logger.warning("Missing severity columns for variant; scoring 0.0.")
            return 0.0
        return float(model_fn(group, params, self.current_time, context))

    def _summaries_by_instrument(
        self, mapped_df: pd.DataFrame
    ) -> dict[str, dict[str, str]]:
        """Generate LLM trend summaries concurrently once per unique instrument."""
        groups = list(mapped_df.groupby("instrument"))
        pairs = self._parallel_map(
            lambda item: (
                str(item[0]),
                self._summarize_instrument(str(item[0]), item[1]),
            ),
            groups,
        )
        return dict(pairs)

    def _variant_trend_row(
        self,
        variant: dict[str, Any],
        instrument: str,
        group: pd.DataFrame,
        severity: float,
        summaries: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        """Assemble a complete trend record for one ``(variant, instrument)`` pair."""
        summary = summaries.get(
            instrument,
            {"title": "Unknown", "description": "Unknown", "llm_direction": "neutral"},
        )
        sev_direction = "up" if severity > 0 else "down" if severity < 0 else "neutral"
        return {
            "variant": variant["name"],
            "model": variant["model"],
            "window_hours": variant["window_hours"],
            "trend_id": str(uuid.uuid4()),
            "current_time": self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "first_article_time": str(group["date"].min())[:19],
            "newest_article_time": str(group["date"].max())[:19],
            "instrument": instrument,
            "title": summary["title"],
            "description": summary["description"],
            "severity": round(severity, 2),
            "direction": sev_direction,
            "llm_direction": summary["llm_direction"],
            "topic_ids": group["topic_id"].tolist(),
        }

    def _map_topic_to_instrument(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Map one topic row via the LLM, quantize sentiment, and filter out unknown
        tickers.
        """
        topic_name = str(row.get("topic_name", ""))
        topic_description = str(row.get("topic_description", ""))
        prompt = PROMPTS["instrument_trends"].format(
            available_instruments=self.available_instruments,
            topic_name=topic_name,
            topic_description=topic_description,
        )

        try:
            logger.info(f"Processing topic: {topic_name[:30]}...")
            parsed_result = self._call_openai_api_with_retry(prompt, InstrumentMapping)
            impacts = parsed_result.impacts  # type: ignore[attr-defined]

            valid_impacts: list[dict[str, Any]] = []
            extracted_tickers: list[str] = []

            for item in impacts:
                ticker = str(item.ticker).strip()
                extracted_tickers.append(ticker)
                if ticker in self.valid_tickers:
                    valid_impacts.append(
                        {
                            "instrument": ticker,
                            "sentiment": _quantize_sentiment(float(item.sentiment)),
                        }
                    )

            if not valid_impacts and extracted_tickers:
                logger.debug(
                    f"Filtered out unknown tickers for '{topic_name}': "
                    f"{extracted_tickers}"
                )

            return valid_impacts

        except Exception as e:
            logger.error(f"Mapping failed for '{topic_name}': {e}")
            self._mapping_failures.append(topic_name or "?")
            return []

    def _calculate_instrument_severity(self, group: pd.DataFrame) -> float:
        """Compute single-model severity in ``[-1.0, 1.0]`` using
        ``config.hyperparams.models.severity_model``.
        """
        if group.empty:
            return 0.0
        if not all(col in group.columns for col in REQUIRED_COLUMNS):
            logger.warning("Missing severity columns; falling back to article count.")
            return float(len(group))

        if config.hyperparams.models.severity_model == "baseline":
            return self._severity_baseline(group)
        return self._severity_dcim(group)

    def _severity_baseline(self, group: pd.DataFrame) -> float:
        """Delegate to ``severity_baseline`` with global DCIM config parameters."""
        return severity_baseline(
            group, self._dcim_params_from_config(), self.current_time
        )

    def _severity_dcim(self, group: pd.DataFrame) -> float:
        """Delegate to ``severity_dcim`` with global DCIM config parameters."""
        return severity_dcim(group, self._dcim_params_from_config(), self.current_time)

    @staticmethod
    def _dcim_params_from_config() -> dict[str, float]:
        """Extract DCIM hyperparameters from ``config.hyperparams.upper_path.dcim``."""
        dcim = config.hyperparams.upper_path.dcim
        return {
            "half_life_hours": dcim.half_life_hours,
            "beta_penalty": dcim.beta_penalty,
            "gamma_consensus": dcim.gamma_consensus,
            "delta_velocity": dcim.delta_velocity,
            "kappa_scale": dcim.kappa_scale,
            "recent_window_hours": dcim.recent_window_hours,
        }

    def _generate_instrument_trend(
        self, instrument: str, group: pd.DataFrame, severity: float
    ) -> dict[str, Any]:
        """Combine deterministic severity with an LLM summary into a single instrument
        trend dict.
        """
        summary = self._summarize_instrument(instrument, group)
        sev_direction = "up" if severity > 0 else "down" if severity < 0 else "neutral"

        return {
            "trend_id": str(uuid.uuid4()),
            "current_time": self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "first_article_time": str(group["date"].min())[:19],
            "newest_article_time": str(group["date"].max())[:19],
            "instrument": instrument,
            "title": summary["title"],
            "description": summary["description"],
            "severity": round(severity, 2),
            "direction": sev_direction,
            "llm_direction": summary["llm_direction"],
            "topic_ids": group["topic_id"].tolist(),
        }

    def _summarize_instrument(
        self, instrument: str, group: pd.DataFrame
    ) -> dict[str, str]:
        """Request an LLM narrative summary (``title``, ``description``,
        ``llm_direction``)
        for one instrument.
        """
        topics_context = group[["topic_name", "topic_description"]].to_dict(
            orient="records"
        )
        prompt = PROMPTS["instrument_summary"].format(
            instrument=instrument, topics_context=topics_context
        )

        try:
            parsed_result = self._call_openai_api_with_retry(prompt, InstrumentTrend)
            result = parsed_result.model_dump()
        except Exception as e:
            logger.error(f"Trend summary failed for '{instrument}': {e}")
            result = {
                "title": "Data aggregation error",
                "description": "Failed to generate summary.",
                "direction": "neutral",
            }

        return {
            "title": str(result.get("title", "Unknown")),
            "description": str(result.get("description", "Unknown")),
            "llm_direction": str(result.get("direction", "neutral")).lower(),
        }
