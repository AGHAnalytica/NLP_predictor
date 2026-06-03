"""
Final signal generation — the layer that turns analysis into a decision.

Fuses the two analysis paths into one actionable, per-instrument trading signal
(UP / DOWN / NEUTRAL + a [0, 1] confidence). The upper path sets the direction
(signed severity); the lower path modulates the confidence (macro context).

The design idea (why there are two paths at all):
- The UPPER path says how this *specific* instrument's news nets out — a signed
severity in [-1, 1]. That is the source of truth for *direction*.
- The LOWER path says what the *macro* narrative is doing. It does not set the
direction; it *modulates confidence*. A single-instrument call that rides a matching
macro narrative is more trustworthy; one that fights the macro tape is less, and a
weak contrarian call can be vetoed down to NEUTRAL.
- The two paths are linked by the ``topic_ids`` they share: a macro trend is relevant
to an instrument only insofar as it is built from the same underlying news.

This layer is deliberately DETERMINISTIC (no LLM). The semantic work was already done
upstream (sentiment, directions, summaries), so the final fusion should be transparent,
fast and auditable — you can read a signal's ``rationale`` and reconstruct exactly why
it fired. All tunables live in Config under the ``SIGNAL_*`` keys.

Output contract (one row per instrument, sorted by confidence):
    - signal_id (str)            : unique uuid
    - current_time (str)         : generation timestamp (UTC)
    - instrument (str)           : ticker
    - signal (str)               : 'UP' | 'DOWN' | 'NEUTRAL' (the decision)
    - confidence (float)         : [0, 1]
    - instrument_severity (float): signed DCIM severity from the upper path
    - macro_context (str)        : 'AGREE' | 'CONFLICT' | 'NONE'
    - n_topics (int)             : number of topics behind the instrument
    - rationale (str)            : human-readable explanation of the decision
    - topic_ids (list)           : the topics that drove this signal
"""

import ast
import uuid
from datetime import UTC, datetime

import pandas as pd

from nlp_predictor.config import Config  # type: ignore


def _as_id_list(value: object) -> list:
    """
    Coerce a ``topic_ids`` cell into a list.
    In-memory the column holds real lists; round-tripped through CSV it becomes a
    string like ``"['a', 'b']"``. This handles both, plus missing/NaN values.

    Args:
        value (object): The value to coerce, which can be a list, a string
        representation of a list, or something else (e.g., NaN).
    Returns:
        list: A list of topic IDs. If the input is already a list, it is returned as is.
        If the input is a string that can be parsed as a list, the parsed list is
        returned. For any other input (including NaN), an empty list is returned.

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
    """
    Combine the upper-path instrument trends and the lower-path macro trends into a
    final per-instrument signal. Pure aggregation — no external calls — so it is cheap
    to run and easy to unit-test against synthetic frames. The rationale field is a v
    erbose, human-readable explanation of how the signal was reached, for auditability.
    The macro modulation is designed to be interpretable and transparent, not a
    black-box formula, so that the final signal can be easily traced back to its inputs.
    The main method is ``generate()``, which takes the two input DataFrames and produces
    the final signal DataFrame according to the contract described at the
    top of this file. The other methods are helpers to keep the code organized and
    readable.
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
    ) -> pd.DataFrame:
        """
        Produce the final signal table. The main entry point for this class.

        Args:
            instrument_trends_df (pd.DataFrame): Upper-path output. Needs
                'instrument', 'severity', 'direction' and 'topic_ids' columns.
            global_trends_df (pd.DataFrame | None): Lower-path output (may be empty or
                None — macro modulation is simply skipped in that case).

        Returns:
            pd.DataFrame: Signals with :attr:`OUTPUT_COLUMNS`, sorted by descending
            confidence.
        """
        if instrument_trends_df is None or instrument_trends_df.empty:
            print("[INFO] No instrument trends; nothing to turn into signals.")
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        macro_index = self._build_macro_index(global_trends_df)
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        signals = [
            self._signal_for_instrument(row, macro_index, now)
            for _, row in instrument_trends_df.iterrows()
        ]

        result = pd.DataFrame(signals, columns=self.OUTPUT_COLUMNS)
        result = result.sort_values("confidence", ascending=False)
        result = result.reset_index(drop=True)
        print(f"[INFO] Generated {len(result)} instrument signals.")
        return result

    def _build_macro_index(self, global_trends_df: pd.DataFrame | None) -> list[dict]:
        """
        Pre-digest the macro trends into ``{topic_ids: set, direction, title}`` records
        so each instrument can be matched against them by shared topics. This also
        gracefully handles the case where there are no macro trends (e.g., first-ever
        run, or a failure in the lower path) by returning an empty index, which causes
        the macro modulation step to be effectively skipped without errors. The index is
        a list of dicts rather than a DataFrame for simplicity of use in the matching
        step; we just need to iterate over the macro trends and check set intersections
        with the instrument topics, so a list of dicts is straightforward and efficient
        for that.

        Args:
            global_trends_df (pd.DataFrame | None): The DataFrame containing the macro
            trends, which should have at least 'topic_ids', 'direction' and 'title'
            columns. Can be None or empty, in which case an empty index is returned.

        Returns:
            list[dict]: A list of dictionaries, each representing a macro trend
            with keys:
            - 'topic_ids': a set of topic IDs associated with the macro trend
            - 'direction': the direction of the macro trend ('up', 'down', 'neutral
                etc.)
            - 'title': the title of the macro trend for context
        """
        if global_trends_df is None or global_trends_df.empty:
            return []

        index: list[dict] = []
        for _, row in global_trends_df.iterrows():
            topic_ids = set(_as_id_list(row.get("topic_ids")))
            if not topic_ids:
                continue
            index.append(
                {
                    "topic_ids": topic_ids,
                    "direction": str(row.get("direction", "neutral")).lower(),
                    "title": str(row.get("title", "macro trend")),
                }
            )
        return index

    def _macro_alignment(
        self, instr_topics: set, direction: str, macro_index: list[dict]
    ) -> tuple[float, float, str]:
        """
        Measure how the macro narratives relate to this instrument's direction.

        For every macro trend sharing topics, ``coverage`` is the fraction of the
        instrument's own topics that fall inside that macro narrative. We keep the
        strongest agreeing coverage and the strongest conflicting coverage (using the
        single dominant narrative on each side rather than summing, to avoid
        double-counting overlapping clusters).

        Args:
            instr_topics (set): The set of topic IDs associated with
            the instrument trend.
            direction (str): The direction of the instrument trend
            ('up', 'down', 'neutral').
            macro_index (list[dict]): The pre-digested macro index, where each dict has
            'topic_ids', 'direction', and 'title' keys.

        Returns:
            - agree_coverage (float): The highest coverage of any macro narrative that
            agrees with the instrument's direction.
            - conflict_coverage (float): The highest coverage of any macro narrative
            that conflicts with the instrument's direction.
            - dominant_macro_title (str): The title of the dominant macro narrative (the
            one with the highest coverage, whether agreeing or conflicting),
            for context.
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
                    and (coverage > conflict_cov)
                ):
                    conflict_cov, conflict_title = coverage, macro["title"]

        dominant_title = agree_title if agree_cov >= conflict_cov else conflict_title
        return agree_cov, conflict_cov, dominant_title

    def _signal_for_instrument(
        self, row: pd.Series, macro_index: list[dict], now: str
    ) -> dict:
        """
        Turn one upper-path instrument trend into a final signal row.
        The logic is as follows:
        - Start with the severity sign for direction and magnitude for base confidence.
        - Modulate confidence based on the macro alignment (coverage of shared topics
        with macro narratives, and whether they agree or conflict).
        - If the per-instrument summary's direction disagrees with the severity sign, it
        is likely a noisy/conflicted case, so discount confidence and flag it in the
        rationale.
        - Apply gating rules: if severity is too weak, or if there is a strong macro
        veto, set the signal to NEUTRAL and explain why in the rationale.
        - Compose a human-readable rationale that explains the key factors behind the
        signal, for auditability.

        Args:
            row (pd.Series): A row from the instrument trends DataFrame, which should
            contain at least 'instrument', 'severity', 'direction' and 'topic_ids'
            fields. macro_index (list[dict]): The pre-digested macro index
            for alignment.
            now (str): The current timestamp as a string, to be used in the signal
            output for consistency.

        Returns:
            dict: A dictionary representing the signal for this instrument, with keys
            corresponding to the OUTPUT_COLUMNS of the final signal DataFrame.
        """
        instrument = str(row["instrument"])
        severity = float(row.get("severity", 0.0))
        strength = abs(severity)
        instr_topics = set(_as_id_list(row.get("topic_ids")))
        n_topics = len(instr_topics)
        llm_direction = str(row.get("direction", "neutral")).lower()

        # Direction comes from the quantitative severity sign (the source of truth).
        sev_direction = "up" if severity > 0 else "down" if severity < 0 else "neutral"

        # 1) Base confidence is just |severity|. The DCIM already folds article
        #    volume, source weight, consensus and recency into that magnitude, so
        #    there is nothing extra to add here.
        base_conf = strength

        # 2) Macro modulation via shared topics.
        agree_cov, conflict_cov, macro_title = self._macro_alignment(
            instr_topics, sev_direction, macro_index
        )
        macro_factor = (
            1.0
            + float(Config["SIGNAL_MACRO_ALIGN_BONUS"]) * agree_cov
            - float(Config["SIGNAL_MACRO_CONFLICT_PENALTY"]) * conflict_cov
        )
        macro_factor = max(0.0, macro_factor)
        confidence = base_conf * macro_factor

        # 3) Internal consistency check: the per-instrument LLM summary's stated
        #    direction should match the severity sign. If it doesn't, trust the
        #    number but discount confidence and flag it.
        internal_conflict = (
            sev_direction != "neutral"
            and llm_direction in ("up", "down")
            and llm_direction != sev_direction
        )
        if internal_conflict:
            confidence *= 0.7

        # 4) Gating + macro veto -> the final discrete decision.
        signal = sev_direction.upper()
        gate_reason = ""
        if strength < float(Config["SIGNAL_NEUTRAL_THRESHOLD"]):
            signal = "NEUTRAL"
            gate_reason = f"|severity| {strength:.2f} below trade threshold"
        elif (
            bool(Config["SIGNAL_MACRO_VETO"])
            and conflict_cov >= float(Config["SIGNAL_VETO_COVERAGE"])
            and strength < float(Config["SIGNAL_VETO_STRENGTH_CAP"])
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
            internal_conflict=internal_conflict,
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
        internal_conflict: bool,
        signal: str,
        confidence: float,
        gate_reason: str,
    ) -> str:
        """
        Compose a short, human-readable explanation of how the signal was reached.
        This is crucial for auditability and trust in the system, allowing a human
        to understand the key factors behind each signal and to reconstruct the logic if
        needed. The rationale includes the severity and its strength, the number of
        topics (as a proxy for signal robustness), the macro context (agreement or
        conflict with macro narratives), and any internal conflicts or gating reasons.
        The macro modulation is explained in terms of coverage of shared topics with
        macro narratives, and the presence of a strong macro veto is explicitly called
        out when it applies. The rationale is designed to be concise but informative,
        giving a clear picture of why the system made the decision it did for each
        instrument.

        Args:
            severity (float): The signed severity score from the upper path.
            sev_direction (str): The direction implied by the severity ('up', 'down',
            'neutral').
            n_topics (int): The number of topics associated with this instrument trend.
            macro_context (str): The macro context classification ('AGREE', 'CONFLICT',
            'NONE').
            macro_title (str): The title of the dominant macro narrative for context.
            agree_cov (float): The coverage of agreeing macro narratives.
            conflict_cov (float): The coverage of conflicting macro narratives.
            internal_conflict (bool): Whether there was a conflict between the severity
            sign and the LLM summary direction.
            signal (str): The final discrete signal ('UP', 'DOWN', 'NEUTRAL').
            confidence (float): The final confidence score in [0, 1].
            gate_reason (str): If the signal was gated to NEUTRAL, the reason for the
            gating.

        Returns:
            str: A human-readable rationale explaining how the signal was derived from
            the inputs.

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
                f"Macro narrative '{macro_title}' aligns "
                f"(coverage {agree_cov:.0%}), reinforcing confidence."
            )
        elif macro_context == "CONFLICT":
            parts.append(
                f"Macro narrative '{macro_title}' opposes it "
                f"(coverage {conflict_cov:.0%}), trimming confidence."
            )
        else:
            parts.append("No overlapping macro narrative.")

        if internal_conflict:
            parts.append(
                "Note: the per-instrument summary's direction disagreed with the "
                "severity sign, so confidence was discounted."
            )
        if gate_reason:
            parts.append(f"Gated to NEUTRAL: {gate_reason}.")

        parts.append(f"Final: {signal} @ {confidence:.2f} confidence.")
        return " ".join(parts)


if __name__ == "__main__":
    print("====== SignalGenerator Module Test ======")
    try:
        instrument_trends = pd.read_csv(Config["INSTRUMENT_TRENDS_SAVE_PATH_CSV"])
        try:
            global_trends = pd.read_csv(Config["GLOBAL_TRENDS_SAVE_PATH_CSV"])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            global_trends = pd.DataFrame()

        signals = SignalGenerator().generate(instrument_trends, global_trends)
        print("====== Signal Generation Completed ======")
        print(signals.to_string(index=False))

        signals.to_csv(Config["SIGNALS_SAVE_PATH_CSV"], index=False)
        print(f"[DEBUG] Saved signals to {Config['SIGNALS_SAVE_PATH_CSV']}")
    except Exception as e:
        print(f"[FATAL ERROR] SignalGenerator test failed: {e}")
