"""Deterministic signal fusion combining upper-path severity with lower-path macro
context.
"""

import ast
import logging
import math
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from nlp_predictor.config import config

logger = logging.getLogger(__name__)


def _as_id_list(value: object) -> list[str]:
    """Coerce a ``topic_ids`` cell (list or serialized list string) into a Python
    list.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return list(parsed) if isinstance(parsed, (list, tuple, set)) else []
    return []


class SignalGenerator:
    r"""Fuses upper-path severity and lower-path macro climate into trading signals.

    Direction is determined by the sign of instrument severity ``S`` in ``[-1, 1]``.
    Base confidence ``abs(S)`` is modulated by macro agreement (``c_agree``) and
    conflict (``c_conflict``):

    .. math::

        \text{conf} = \operatorname{clip}\!\Big(\lvert S \rvert \cdot
        \max\!\big(0,\, 1 + \alpha_{\text{align}} c_{\text{agree}}
        - \alpha_{\text{conflict}} c_{\text{conflict}}\big),\, 0,\, 1\Big)

    Weak signals below ``neutral_threshold`` or contrarian calls triggering
    ``macro_veto`` are gated to ``NEUTRAL``.

    Output schema (:attr:`OUTPUT_COLUMNS`):
        - ``signal_id`` (str): UUID4 identifier.
        - ``current_time`` (str): Generation UTC timestamp (``YYYY-MM-DD HH:MM:SS``).
        - ``instrument`` (str): Ticker symbol.
        - ``signal`` (str): ``'UP'``, ``'DOWN'``, or ``'NEUTRAL'``.
        - ``confidence`` (float): Confidence score in $[0, 1]$.
        - ``instrument_severity`` (float): Signed severity score from the upper path.
        - ``macro_context`` (str): ``'AGREE'``, ``'CONFLICT'``, or ``'NONE'``.
        - ``n_topics`` (int): Number of supporting topics.
        - ``rationale`` (str): Human-readable audit trail.
        - ``topic_ids`` (list[str]): Contributing topic UUIDs.
    """

    OUTPUT_COLUMNS = [
        "signal_id",
        "current_time",
        "instrument",
        "signal",
        "confidence",
        "instrument_severity",
        "macro_context",
        "n_topics",
        "rationale",
        "topic_ids",
    ]

    def generate(
        self,
        instrument_trends_df: pd.DataFrame,
        global_trends_df: pd.DataFrame | None = None,
        macro_overlay: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate per-instrument signals sorted by descending confidence.

        Args:
            instrument_trends_df: Upper-path DataFrame with ``instrument``,
                ``severity``,
                ``direction``, and ``topic_ids``.
            global_trends_df: Optional lower-path macro cluster DataFrame with
                ``topic_ids``, ``severity``, ``direction``, and ``title``.
            macro_overlay: Optional per-instrument effect map from :class:`MacroOverlay`
                (``{instrument: {'effect', 'strength', 'rationale'}}``). When provided,
                serves as the primary macro alignment channel.

        Returns:
            DataFrame conforming to :attr:`OUTPUT_COLUMNS`, sorted by ``confidence``
                desc.
        """
        if instrument_trends_df is None or instrument_trends_df.empty:
            logger.info("No instrument trends; nothing to turn into signals.")
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        macro_index = self._build_macro_index(global_trends_df)
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        signals = [
            self._signal_for_instrument(row, macro_index, now, macro_overlay)
            for row in instrument_trends_df.to_dict(orient="records")
        ]

        result = pd.DataFrame(signals, columns=self.OUTPUT_COLUMNS)
        result = result.sort_values("confidence", ascending=False).reset_index(
            drop=True
        )
        logger.info(f"Generated {len(result)} instrument signals.")
        return result

    def _build_macro_index(
        self, global_trends_df: pd.DataFrame | None
    ) -> list[dict[str, Any]]:
        """Convert ``global_trends_df`` into a list of cluster dicts with set
        ``topic_ids``.
        """
        if global_trends_df is None or global_trends_df.empty:
            return []

        index: list[dict[str, Any]] = []
        for row in global_trends_df.to_dict(orient="records"):
            topic_ids = set(_as_id_list(row.get("topic_ids")))
            if not topic_ids:
                continue
            try:
                severity = float(row.get("severity", 0) or 0)
            except (TypeError, ValueError):
                severity = 0.0
            index.append(
                {
                    "topic_ids": topic_ids,
                    "direction": str(row.get("direction", "neutral")).lower(),
                    "title": str(row.get("title", "macro trend")),
                    "severity": severity,
                }
            )
        return index

    def _macro_conviction(
        self, instr_topics: set[str], macro_index: list[dict[str, Any]]
    ) -> float:
        r"""Compute per-instrument macro conviction $\tanh(\text{volume} / \text{scale})
        \in [0, 1]$.

        Uses the summed severity of clusters overlapping ``instr_topics`` when direct
        topic links exist, or falls back to total macro climate severity for diffuse
        mood.
        """
        scale = float(config.hyperparams.signal.macro_conviction_scale)
        if scale <= 0.0 or not macro_index:
            return 0.0

        drivers = [m for m in macro_index if instr_topics & m["topic_ids"]]
        if drivers:
            volume = sum(m["severity"] for m in drivers)
        else:
            volume = sum(m["severity"] for m in macro_index)
        return math.tanh(volume / scale)

    def _macro_alignment(
        self,
        instr_topics: set[str],
        direction: str,
        macro_index: list[dict[str, Any]],
    ) -> tuple[float, float, str]:
        """Compute topic-overlap agreement and conflict coverage against macro
        clusters.
        """
        agree_cov, agree_title = 0.0, ""
        conflict_cov, conflict_title = 0.0, ""

        if instr_topics and macro_index and direction in ("up", "down"):
            for macro in macro_index:
                shared = instr_topics & macro["topic_ids"]
                if not shared:
                    continue
                coverage = len(shared) / len(instr_topics)
                m_dir = macro["direction"]
                if m_dir == direction and coverage > agree_cov:
                    agree_cov, agree_title = coverage, macro["title"]
                elif (
                    m_dir in ("up", "down")
                    and m_dir != direction
                    and coverage > conflict_cov
                ):
                    conflict_cov, conflict_title = coverage, macro["title"]

        dominant_title = agree_title if agree_cov >= conflict_cov else conflict_title
        return agree_cov, conflict_cov, dominant_title

    @staticmethod
    def _overlay_alignment(
        instrument: str,
        sev_direction: str,
        macro_overlay: dict[str, Any],
        macro_conviction: float,
    ) -> tuple[float, float, str]:
        """Compute effective agreement/conflict from the LLM macro overlay and
        conviction.
        """
        if sev_direction not in ("up", "down") or macro_conviction <= 0.0:
            return 0.0, 0.0, ""

        entry = macro_overlay.get(instrument)
        if not entry:
            return 0.0, 0.0, ""

        effect = str(entry.get("effect", "neutral")).lower()
        strength = max(0.0, min(1.0, float(entry.get("strength", 0.0))))
        title = str(entry.get("rationale", "macro climate"))
        effect_dir = (
            "up"
            if effect == "tailwind"
            else "down"
            if effect == "headwind"
            else "neutral"
        )

        if effect_dir == "neutral" or strength == 0.0:
            return 0.0, 0.0, ""

        effective = strength * macro_conviction
        if effect_dir == sev_direction:
            return effective, 0.0, title
        return 0.0, effective, title

    def _signal_for_instrument(
        self,
        row: Mapping[str, Any],
        macro_index: list[dict[str, Any]],
        now: str,
        macro_overlay: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute direction, modulated confidence, gating, and rationale for one
        instrument.
        """
        instrument = str(row["instrument"])
        severity = float(row.get("severity", 0.0))
        strength = abs(severity)
        instr_topics = set(_as_id_list(row.get("topic_ids")))
        n_topics = len(instr_topics)

        sev_direction = "up" if severity > 0 else "down" if severity < 0 else "neutral"
        base_conf = strength

        if macro_overlay is not None:
            macro_conviction = self._macro_conviction(instr_topics, macro_index)
            agree_cov, conflict_cov, macro_title = self._overlay_alignment(
                instrument, sev_direction, macro_overlay, macro_conviction
            )
        else:
            agree_cov, conflict_cov, macro_title = self._macro_alignment(
                instr_topics, sev_direction, macro_index
            )

        macro_factor = (
            1.0
            + float(config.hyperparams.signal.macro_align_bonus) * agree_cov
            - float(config.hyperparams.signal.macro_conflict_penalty) * conflict_cov
        )
        macro_factor = max(0.0, macro_factor)
        confidence = base_conf * macro_factor

        signal = sev_direction.upper()
        gate_reason = ""
        if strength < float(config.hyperparams.signal.neutral_threshold):
            signal = "NEUTRAL"
            gate_reason = f"|severity| {strength:.2f} below trade threshold"
        elif (
            bool(config.hyperparams.signal.macro_veto)
            and conflict_cov >= float(config.hyperparams.signal.veto_coverage)
            and strength < float(config.hyperparams.signal.veto_strength_cap)
        ):
            signal = "NEUTRAL"
            gate_reason = (
                f"macro '{macro_title}' opposes a weak {sev_direction} call (veto)"
            )

        confidence = round(min(max(confidence, 0.0), 1.0), 3)

        if agree_cov > conflict_cov and agree_cov > 0:
            macro_context = "AGREE"
        elif conflict_cov > 0:
            macro_context = "CONFLICT"
        else:
            macro_context = "NONE"

        rationale = self._build_rationale(
            severity=severity,
            sev_direction=sev_direction,
            n_topics=n_topics,
            macro_context=macro_context,
            macro_title=macro_title,
            agree_cov=agree_cov,
            conflict_cov=conflict_cov,
            signal=signal,
            confidence=confidence,
            gate_reason=gate_reason,
        )

        return {
            "signal_id": str(uuid.uuid4()),
            "current_time": now,
            "instrument": instrument,
            "signal": signal,
            "confidence": confidence,
            "instrument_severity": round(severity, 3),
            "macro_context": macro_context,
            "n_topics": n_topics,
            "rationale": rationale,
            "topic_ids": sorted(instr_topics),
        }

    def _build_rationale(
        self,
        *,
        severity: float,
        sev_direction: str,
        n_topics: int,
        macro_context: str,
        macro_title: str,
        agree_cov: float,
        conflict_cov: float,
        signal: str,
        confidence: float,
        gate_reason: str,
    ) -> str:
        """Format a concise human-readable audit string explaining the signal
        decision.
        """
        strength = abs(severity)
        strength_word = (
            "strong" if strength >= 0.6 else "moderate" if strength >= 0.3 else "weak"
        )
        parts = [
            f"Instrument severity {severity:+.2f} "
            f"({strength_word} {sev_direction}) over {n_topics} topic(s)."
        ]

        if macro_context == "AGREE":
            parts.append(
                f"Macro climate '{macro_title}' aligns "
                f"(strength {agree_cov:.0%}), reinforcing confidence."
            )
        elif macro_context == "CONFLICT":
            parts.append(
                f"Macro climate '{macro_title}' opposes it "
                f"(strength {conflict_cov:.0%}), trimming confidence."
            )
        else:
            parts.append("No overlapping macro narrative.")

        if gate_reason:
            parts.append(f"Gated to NEUTRAL: {gate_reason}.")

        parts.append(f"Final: {signal} @ {confidence:.2f} confidence.")
        return " ".join(parts)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    logger.info("====== SignalGenerator Module Test ======")
    try:
        instrument_trends = pd.read_csv(config.paths.output.instrument_trends_csv)
        try:
            global_trends = pd.read_csv(config.paths.output.global_trends_csv)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            global_trends = pd.DataFrame()

        signals = SignalGenerator().generate(instrument_trends, global_trends)
        logger.info("====== Signal Generation Completed ======")
        logger.info(f"\n{signals.to_string(index=False)}")

        signals.to_csv(config.paths.output.signals_csv, index=False)
        logger.debug(f"Saved signals to {config.paths.output.signals_csv}")
    except Exception as e:
        logger.error(f"SignalGenerator test failed: {e}", exc_info=True)
