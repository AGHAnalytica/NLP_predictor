# Core imports
import uuid
from datetime import UTC, datetime

import fastparquet  # noqa: F401
import pandas as pd

# Handle API and LLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from nlp_predictor.base_analyzer import BaseAnalyzer  # type: ignore
from nlp_predictor.config import Config  # type: ignore
from nlp_predictor.prompts import PROMPTS  # type: ignore
from nlp_predictor.severity_models import (  # type: ignore
    REQUIRED_COLUMNS,
    SEVERITY_REGISTRY,
    severity_baseline,
    severity_dcim,
)

# Load environment variables from the .env file located in the project root
load_dotenv(override=True)

# Fixed sentiment scale the LLM is instructed to use (see the instrument_trends prompt).
# Snapping the model's free-form score onto these anchors removes run-to-run decimal
# drift and keeps the value calibrated the same way across independent calls.
_SENTIMENT_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _quantize_sentiment(value: float) -> float:
    """Snap a raw sentiment score to the nearest fixed-scale level."""
    return min(_SENTIMENT_LEVELS, key=lambda level: abs(level - value))


# Define Pydantic schemas for strict LLM JSON outputs
class TickerImpact(BaseModel):
    """
    Pydantic model for parsing the mapping of a topic to impacted tickers and their
    sentiment.
    """

    ticker: str = Field(description="Exact ticker symbol from the available list.")
    sentiment: float = Field(
        description="Directional impact of the news on THIS SPECIFIC ticker, on a "
        "fixed 5-point scale: -1.0 / -0.5 / 0.0 / 0.5 / 1.0 (negative = bearish, "
        "0.0 = neutral, positive = bullish). See the instrument_trends prompt for the "
        "anchors. Also snapped to the nearest level by _quantize_sentiment."
    )


class InstrumentMapping(BaseModel):
    """
    Pydantic model for parsing the mapping of a topic to impacted tickers and their
    sentiment.
    """

    impacts: list[TickerImpact] = Field(
        description="List of impacted tickers and their sentiment."
        + " Return empty if none."
    )


class InstrumentTrend(BaseModel):
    """
    Pydantic model for parsing the output of the instrument trend summary from
    OpenAI."""

    title: str = Field(description="Short summary of the trend.")
    description: str = Field(
        description="Detailed explanation of the trend and its impact."
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")


class UpperPathAnalyzer(BaseAnalyzer):
    """
    This class encapsulates the logic for processing the upper path of the data
    pipeline. It includes methods for mapping news topics to financial instruments,
    generating trend summaries, and saving the results to a partitioned Parquet dataset.
    The class is designed to be robust against API rate limits and potential
    hallucinations by the LLM, with strict validation against known instruments.
    It also includes detailed logging for monitoring the processing steps and handling a
    ny issues that arise.

    The main method, `process_data`, takes a DataFrame of topics, maps each
    topic to relevant instruments, and generates a trend summary for each instrument
    based on the associated topics. The final output is a DataFrame containing the
    trends for each instrument, ready to be saved as a partitioned Parquet dataset.
    """

    def __init__(self, instruments_path: str = Config["INSTRUMENT_PATH"]):
        super().__init__()

        # Load available instruments and create a context string for the LLM
        self.instruments_df = pd.read_csv(instruments_path)
        self.valid_tickers = set(self.instruments_df["ticker"].dropna().tolist())

        # Create clean dictionary for the LLM (Ticker: Full Name)
        self.available_instruments = dict(
            zip(
                self.instruments_df["ticker"],
                self.instruments_df["full_name"],
                strict=False,
            )
        )

        # Topic names whose instrument mapping raised (distinct from "no instruments").
        # Reset at the start of every process_data run.
        self._mapping_failures: list[str] = []

    def process_data(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing function for the upper path. It takes a DataFrame of topics,
        maps each topic to relevant financial instruments along with their
        entity-specific sentiment, and then generates a trend summary for each
        instrument based on the associated topics using the DCIM severity model.

        Args:
            topics_df (pd.DataFrame): A DataFrame containing the topics with columns
                such as 'topic_id', 'topic_name', 'topic_description', 'date',
                'source_name', and 'source_weight'.

        Returns:
            pd.DataFrame: A DataFrame containing the generated trends for each
                instrument, including columns like 'trend_id', 'current_time',
                'first_article_time', 'newest_article_time', 'instrument', 'title',
                'description', 'severity', 'direction', and 'topic_ids'.
        """
        if topics_df.empty:
            return pd.DataFrame()

        self.current_time = datetime.now(UTC)
        mapped_df = self._map_and_explode(topics_df)
        if mapped_df.empty:
            return pd.DataFrame()

        print("[INFO] Generating trends per instrument using the DCIM model...")

        def _trend_for_instrument(item: tuple[str, pd.DataFrame]) -> dict:
            instrument, group = item
            severity = self._calculate_instrument_severity(group)
            return self._generate_instrument_trend(instrument, group, severity)

        # Severity is local; only the trend-summary LLM call benefits from threading.
        results = self._parallel_map(
            _trend_for_instrument, list(mapped_df.groupby("instrument"))
        )

        return pd.DataFrame(results)

    def _map_and_explode(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map every topic to its impacted instruments (the expensive LLM step) and explode
        into one long row per (topic, instrument) with the columns the severity models
        need. Shared by the single-model :meth:`process_data` and the multi-variant
        :meth:`generate_variant_trends`, so the costly mapping runs once per tick no
        matter how many severity variants are being compared.

        Args:
            topics_df (pd.DataFrame): Enriched topics with 'topic_name',
                'topic_description', 'date', 'source_name' and 'source_weight'.

        Returns:
            pd.DataFrame: Long-form rows with 'instrument', 'sentiment' and the carried
            topic/source/date columns. Empty if nothing mapped.
        """
        print("[INFO] Starting topic to instrument mapping and sentiment analysis...")

        # Each topic maps independently, so fan the LLM calls out concurrently.
        self._mapping_failures = []
        topic_rows = [row for _, row in topics_df.iterrows()]
        topics_df = topics_df.copy()
        topics_df["impacts"] = self._parallel_map(
            self._map_topic_to_instrument, topic_rows
        )

        # Surface failed mappings: an errored topic returns [] just like a genuinely
        # irrelevant one, so without this a partial outage looks like "no signal".
        if self._mapping_failures:
            print(
                f"[WARNING] {len(self._mapping_failures)}/{len(topic_rows)} topic "
                "mappings FAILED (not the same as 'no instruments impacted'); "
                "instrument signals may be incomplete."
            )

        exploded_df = topics_df.explode("impacts")
        exploded_df = exploded_df.dropna(subset=["impacts"])

        if exploded_df.empty:
            print(
                "[WARNING] All topics were dropped after mapping. "
                "The result will be empty."
            )
            return pd.DataFrame()

        # Expand the dictionaries inside the 'impacts' column into separate columns
        impacts_expanded = exploded_df["impacts"].apply(pd.Series)

        mapped_df = pd.concat(
            [exploded_df.drop(columns=["impacts"]), impacts_expanded], axis=1
        )

        mapped_df["instrument"] = mapped_df["instrument"].astype(str)
        mapped_df["sentiment"] = mapped_df["sentiment"].astype(float)
        # Severity time-decay and per-variant windowing both need tz-aware datetimes.
        mapped_df["date"] = pd.to_datetime(mapped_df["date"], utc=True)
        return mapped_df

    def generate_variant_trends(
        self,
        topics_df: pd.DataFrame,
        variants: list[dict] | None = None,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Score every configured severity VARIANT over the same mapped articles and return
        one long-form row per (variant, instrument). This is the upper-path output the
        A/B/n consumes: each variant is a self-contained ``(model, params, window)``
        strategy routed to its own demo account.

        The expensive work is shared, not repeated per variant: topic->instrument
        mapping runs once (:meth:`_map_and_explode`) and the per-instrument LLM summary
        runs once (:meth:`_summaries_by_instrument`). Only the cheap, deterministic
        severity is recomputed per variant — over that variant's own window slice — so
        the comparison isolates the model as the single variable.

        Args:
            topics_df (pd.DataFrame): Enriched topics (same input as
                :meth:`process_data`).
            variants (list[dict] | None): Variant specs; defaults to
                ``Config["SEVERITY_VARIANTS"]``. Each has 'name', 'model',
                'window_hours' and 'params'.
            now (datetime | None): Reference time for windowing/decay; defaults to
                ``datetime.now(UTC)``. Injectable for deterministic tests.

        Returns:
            pd.DataFrame: Long-form trends with a 'variant' column. Empty if nothing
            mapped or no variant had in-window articles.
        """
        variants = variants if variants is not None else Config["SEVERITY_VARIANTS"]
        if topics_df.empty:
            return pd.DataFrame()

        self.current_time = now or datetime.now(UTC)
        mapped_df = self._map_and_explode(topics_df)
        if mapped_df.empty:
            return pd.DataFrame()

        # One LLM summary per instrument, shared across every variant (rationale only).
        summaries = self._summaries_by_instrument(mapped_df)

        rows: list[dict] = []
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

        print(
            f"[INFO] Generated {len(rows)} variant trend rows across "
            f"{len(variants)} variant(s)."
        )
        return pd.DataFrame(rows)

    def _window_frame(self, mapped_df: pd.DataFrame, hours: float) -> pd.DataFrame:
        """Rows whose article date is within the last ``hours`` of ``current_time``."""
        cutoff = self.current_time - pd.Timedelta(hours=float(hours))
        return mapped_df[mapped_df["date"] >= cutoff]

    @staticmethod
    def _variant_context(window_df: pd.DataFrame) -> dict:
        """
        Cross-instrument context a model may need. Currently the baseline article count
        for ``abnormal_volume``: the mean number of articles per instrument in this
        window. This is the interim, cross-sectional stand-in ("abnormal vs the average
        instrument right now") until per-instrument temporal history is collected.
        """
        counts = window_df.groupby("instrument").size()
        baseline = float(counts.mean()) if len(counts) else 1.0
        return {"baseline_count": baseline}

    def _severity_for(
        self, model_fn, group: pd.DataFrame, params: dict, context: dict
    ) -> float:
        """
        Run one severity model with the shared input guard. If the columns the models
        need are missing the signal is undefined, so return 0.0 (no trade) rather than
        the legacy raw-count fallback, which would break the bounded [-1, 1] contract
        the variants are compared on.
        """
        if group.empty:
            return 0.0
        if not all(col in group.columns for col in REQUIRED_COLUMNS):
            print("[WARNING] Missing severity columns for variant; scoring 0.0.")
            return 0.0
        return model_fn(group, params, self.current_time, context)

    def _summaries_by_instrument(self, mapped_df: pd.DataFrame) -> dict[str, dict]:
        """LLM title/description/llm_direction per instrument, once per tick."""
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
        variant: dict,
        instrument: str,
        group: pd.DataFrame,
        severity: float,
        summaries: dict[str, dict],
    ) -> dict:
        """Assemble one long-form trend row for a (variant, instrument) pair."""
        summary = summaries.get(
            instrument,
            {"title": "Unknown", "description": "Unknown", "llm_direction": "neutral"},
        )
        sev_direction = (
            "up" if severity > 0 else "down" if severity < 0 else "neutral"
        )
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

    def _map_topic_to_instrument(self, row: pd.Series) -> list[dict]:
        """
        Maps a single topic to relevant financial instruments using the LLM and
        calculates sentiment. The prompt provides the topic name and description, along
        with a list of available instruments (tickers and full names). The LLM is asked
        to return a list of ticker symbols that are directly impacted by the topic,
        along with a sentiment score for each on a fixed 5-point scale
        (-1.0 / -0.5 / 0.0 / 0.5 / 1.0).

        Each returned score is snapped to the nearest scale level via
        :func:`_quantize_sentiment`, so run-to-run decimal drift cannot leak into the
        DCIM. The output is also strictly validated against the known tickers to filter
        out any hallucinations.

        Args:
            row (pd.Series): A row from the topics DataFrame containing 'topic_name'
            and 'topic_description'.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains:
                - 'instrument': The valid ticker symbol.
                - 'sentiment': The quantized sentiment score, one of
                  {-1.0, -0.5, 0.0, 0.5, 1.0}.
        """
        prompt = PROMPTS["instrument_trends"].format(
            available_instruments=self.available_instruments,
            topic_name=row["topic_name"],
            topic_description=row["topic_description"],
        )

        try:
            print(f"[INFO] Processing topic: {row['topic_name'][:30]}...")

            parsed_result = self._call_openai_api_with_retry(prompt, InstrumentMapping)

            impacts = parsed_result.impacts

            valid_impacts = []
            extracted_tickers = []

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
                print(
                    "[DEBUG] Validated out hallucinatory tickers for '"
                    f"{row['topic_name']}' : {extracted_tickers}"
                )

            return valid_impacts

        except Exception as e:
            print(f"[ERROR] Mapping failed for '{row['topic_name']}': {e}")
            self._mapping_failures.append(str(row.get("topic_name", "?")))
            return []

    def _calculate_instrument_severity(self, group: pd.DataFrame) -> float:
        """
        Score a financial instrument's directional severity in [-1.0, 1.0].

        Dispatches to the model selected by ``Config["SEVERITY_MODEL"]``:
        - ``"dcim"``     : the full Directional Confirmed Impact Model.
        - ``"baseline"`` : a minimal time-decayed, source-weighted sentiment sum.
        Both share these guards and output range, so they can be A/B'd against each
        other without changing anything downstream. Falls back to a raw article count
        if the columns the models need are absent.

        Args:
            group (pd.DataFrame): The articles/topics mapped to one instrument. Needs
            'date', 'source_name', 'sentiment' and 'source_weight' columns.

        Returns:
            float: Severity in [-1.0, 1.0] (negative = bearish, ~0 = neutral/conflict,
            positive = bullish).
        """
        if group.empty:
            return 0.0

        required_cols = ["date", "source_name", "sentiment", "source_weight"]
        if not all(col in group.columns for col in required_cols):
            print(
                "[WARNING] Missing severity columns. "
                "Using simple article counter instead"
            )
            return float(len(group))

        if Config.get("SEVERITY_MODEL", "dcim") == "baseline":
            return self._severity_baseline(group)
        return self._severity_dcim(group)

    def _severity_baseline(self, group: pd.DataFrame) -> float:
        """
        Minimal control severity (legacy single-model path).

        Thin wrapper delegating to :func:`severity_models.severity_baseline` with the
        hyperparameters read from the global ``DCIM_*`` Config keys. The A/B/n path
        calls the pure function directly with per-variant params instead, so one
        implementation backs both. Kept so existing callers (and the quantify script)
        keep working.

        Args:
            group (pd.DataFrame): The articles/topics mapped to one instrument.

        Returns:
            float: Severity in [-1.0, 1.0].
        """
        return severity_baseline(
            group, self._dcim_params_from_config(), self.current_time
        )

    def _severity_dcim(self, group: pd.DataFrame) -> float:
        """
        Directional Confirmed Impact Model severity (legacy single-model path).

        Thin wrapper delegating to :func:`severity_models.severity_dcim` with the global
        ``DCIM_*`` Config hyperparameters. The five-phase math now lives in
        ``severity_models`` so one implementation backs both this path and every A/B
        variant.

        Args:
            group (pd.DataFrame): The articles/topics mapped to one instrument.

        Returns:
            float: Severity in [-1.0, 1.0].
        """
        return severity_dcim(
            group, self._dcim_params_from_config(), self.current_time
        )

    @staticmethod
    def _dcim_params_from_config() -> dict:
        """Build a severity-model params dict from the global ``DCIM_*`` Config keys."""
        return {
            "half_life_hours": Config["DCIM_HALF_LIFE_HOURS"],
            "beta_penalty": Config["DCIM_BETA_PENALTY"],
            "gamma_consensus": Config["DCIM_GAMMA_CONSENSUS"],
            "delta_velocity": Config["DCIM_DELTA_VELOCITY"],
            "kappa_scale": Config["DCIM_KAPPA_SCALE"],
            "recent_window_hours": Config["DCIM_RECENT_WINDOW_HOURS"],
        }

    def _generate_instrument_trend(
        self, instrument: str, group: pd.DataFrame, severity: float
    ) -> dict:
        """
        Generates a trend summary for a given instrument based on the associated topics.
        The prompt includes the instrument name, a list of related topics with
        their descriptions, and the time range of the articles.
        The LLM is asked to determine the overall price trend direction
        ('up', 'down', 'neutral') and provide a concise title and description explaining
        the aggregate impact.

        Args:
            instrument (str): The ticker symbol of the financial instrument.
            group (pd.DataFrame): A DataFrame containing the topics related to
            the instrument.
            severity (float): The signed DCIM severity in [-1.0, 1.0] computed by
            :meth:`_calculate_instrument_severity` (passed through to the output).

        Returns:
            dict: A dictionary containing the trend information, including a unique
            trend ID, timestamps, instrument name, title, description, severity score,
            direction, and associated topic IDs.

        """
        summary = self._summarize_instrument(instrument, group)
        topic_ids = group["topic_id"].tolist()

        # Derive time range from the articles' posted timestamps
        # (ISO strings sort correctly)
        first_article_time = str(group["date"].min())[:19]
        newest_article_time = str(group["date"].max())[:19]

        # Direction is the deterministic sign of THIS method's own severity — the source
        # of truth the signal trades on. The LLM summary's stated direction is retained
        # only as a side column (``llm_direction``) for analysis/audit; it never enters
        # the decision path, so each severity method stays self-contained and the A/B/n
        # comparison isolates the method as the only variable.
        sev_direction = "up" if severity > 0 else "down" if severity < 0 else "neutral"

        return {
            "trend_id": str(uuid.uuid4()),
            "current_time": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "first_article_time": first_article_time,
            "newest_article_time": newest_article_time,
            "instrument": instrument,
            "title": summary["title"],
            "description": summary["description"],
            "severity": round(severity, 2),
            "direction": sev_direction,
            "llm_direction": summary["llm_direction"],
            "topic_ids": topic_ids,
        }

    def _summarize_instrument(self, instrument: str, group: pd.DataFrame) -> dict:
        """
        Run the per-instrument LLM summary (title / description / stated direction).

        Pure rationale generation, decoupled from severity so it can be computed ONCE
        per instrument and shared across every A/B variant (see
        :meth:`_summaries_by_instrument`) rather than re-billed per variant. Its stated
        direction is returned as ``llm_direction``, never used in the decision path.

        Args:
            instrument (str): The ticker symbol.
            group (pd.DataFrame): The topics mapped to this instrument.

        Returns:
            dict: ``{'title', 'description', 'llm_direction'}``.
        """
        topics_context = group[["topic_name", "topic_description"]].to_dict(
            orient="records"
        )  # type: ignore[call-overload]

        prompt = f"""
        You are a senior financial analyst. Analyze the following news topics related to
        the instrument '{instrument}'. Determine the overall price trend direction
        ('up', 'down', or 'neutral'). Provide a concise title and a 1-2 sentence
        description explaining the aggregate impact.

        Topics Context:
        {topics_context}
        """

        try:
            parsed_result = self._call_openai_api_with_retry(prompt, InstrumentTrend)
            result = parsed_result.model_dump()
        except Exception as e:
            print(f"[ERROR] Trend generation failed for '{instrument}': {e}")
            result = {
                "title": "Data aggregation error",
                "description": "Failed to generate summary.",
                "direction": "neutral",
            }

        return {
            "title": result.get("title", "Unknown"),
            "description": result.get("description", "Unknown"),
            "llm_direction": result.get("direction", "neutral").lower(),
        }


if __name__ == "__main__":
    print("====== UpperPathAnalyzer Module Test ======")

    try:
        upper_path_analyzer = UpperPathAnalyzer(Config["INSTRUMENT_PATH"])
        df_topics = pd.read_csv(Config["TOPICS_PATH"])
        df_articles = pd.read_csv(Config["ARTICLES_PATH"])
        df_weights = pd.read_csv(Config["WEIGHTS_PATH"])

        df_article_mapping = pd.merge(
            df_articles[["doc_hash", "source_name"]],
            df_weights[["source_name", "source_weight"]],
            on="source_name",
            how="left",
        )

        df_topics_enriched = pd.merge(
            df_topics,
            df_article_mapping[["doc_hash", "source_weight", "source_name"]],
            on="doc_hash",
            how="left",
        )

        df_topics_enriched["source_weight"] = df_topics_enriched[
            "source_weight"
        ].fillna(1.0)

        df_topics_enriched["date"] = pd.to_datetime(
            df_topics_enriched["date"], utc=True
        )

        print("[INFO] Successfully merged source weights into topics.")

        result = upper_path_analyzer.process_data(df_topics_enriched)
        print("====== Processing Upper Path Completed ======")

        print("====== Saving Results To File ======")
        saving_result = UpperPathAnalyzer.save_trends(
            result, Config["INSTRUMENT_TRENDS_SAVE_PATH_PARQUET"]
        )
        if saving_result:
            print("====== Saving Completed ======")
        else:
            print("====== Saving Failed ======")

        # For debugging: save the result to a CSV as well
        result.to_csv(Config["INSTRUMENT_TRENDS_SAVE_PATH_CSV"], index=False)
        print(
            "[DEBUG] Also saved results to CSV for quick inspection at  + "
            f"{Config['INSTRUMENT_TRENDS_SAVE_PATH_CSV']}"
        )
    except Exception as e:
        print(f"[FATAL ERROR] An error occurred during the UpperPathAnalyzer test: {e}")
