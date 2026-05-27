# Core imports
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path

import fastparquet  # noqa: F401
import numpy as np
import pandas as pd
from base_analyzer import BaseAnalyzer  # type: ignore

# Handle API and LLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from the .env file located in the project root
load_dotenv()

# Default instrument path (resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parents[2]
INSTRUMENT_PATH = str(BASE_DIR / "data" / "instruments" / "instruments_sample.csv")
TOPICS_PATH = str(BASE_DIR / "data" / "topics" / "topics_sample.csv")
SAVE_PATH_CSV = str(BASE_DIR / "data" / "test_results" / "instrument_trends_report.csv")
SAVE_PATH_PARQUET = str(
    BASE_DIR / "data" / "test_results" / "instrument_trends_parquet"
)
ARTICLES_PATH = str(BASE_DIR / "data" / "scraping" / "articles_sample.csv")
WEIGHTS_PATH = str(BASE_DIR / "data" / "scraping" / "source_weights_sample.csv")


# Define Pydantic schemas for strict LLM JSON outputs
class TickerImpact(BaseModel):
    ticker: str = Field(description="Exact ticker symbol from the available list.")
    sentiment: float = Field(
        description="Directional impact of the news on THIS SPECIFIC ticker. "
        "From -1.0 (highly negative) to 1.0 (highly positive). 0.0 is neutral."
    )


class InstrumentMapping(BaseModel):
    impacts: list[TickerImpact] = Field(
        description="List of impacted tickers and their sentiment."
        + " Return empty if none."
    )


class InstrumentTrend(BaseModel):
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

    # =========================================================================
    # DCIM HYPERPARAMETERS (Directional Confirmed Impact Model)
    # =========================================================================
    DCIM_HALF_LIFE_HOURS = 4.0
    DCIM_BETA_PENALTY = 0.5
    DCIM_GAMMA_CONSENSUS = 0.3
    DCIM_DELTA_VELOCITY = 0.3
    DCIM_KAPPA_SCALE = 2.0
    DCIM_RECENT_WINDOW_HOURS = 0.5

    def __init__(self, instruments_path: str = INSTRUMENT_PATH):
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

        print("[INFO] Starting topic to instrument mapping and sentiment analysis...")

        topics_df["impacts"] = topics_df.apply(self._map_topic_to_instrument, axis=1)

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

        results = []
        print("[INFO] Generating trends per instrument using the DCIM model...")
        for instrument, group in mapped_df.groupby("instrument"):
            severity = self._calculate_instrument_severity(group)
            trend_data = self._generate_instrument_trend(instrument, group, severity)
            results.append(trend_data)

        return pd.DataFrame(results)

    def _map_topic_to_instrument(self, row: pd.Series) -> list[dict]:
        """
        Maps a single topic to relevant financial instruments using the LLM and
        calculates sentiment. The prompt provides the topic name and description, along
        with a list of available instruments (tickers and full names). The LLM is asked
        to return a list of ticker symbols that are directly impacted by the topic,
        along with a sentiment score for each.
        The output is strictly validated against the known tickers to filter
        out any hallucinations.

        Args:
            row (pd.Series): A row from the topics DataFrame containing 'topic_name'
            and 'topic_description'.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains:
                - 'instrument': The valid ticker symbol.
                - 'sentiment': The calculated sentiment score (-1.0 to 1.0).
        """
        prompt = f"""
        You are a quantitative financial analyst. Map the following news topic to ALL
        applicable instruments from the list.

        For each impacted instrument, calculate the 'sentiment' score
        between -1.0 and 1.0,
        representing how this specific news affects that specific company's stock price.
        IMPORTANT: Return ONLY exact 'ticker' symbols from the provided list.

        Available Instruments: {self.available_instruments}

        Topic Name: {row["topic_name"]}
        Topic Description: {row["topic_description"]}
        """

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
                        {"instrument": ticker, "sentiment": float(item.sentiment)}
                    )

            if not valid_impacts and extracted_tickers:
                print(
                    "[DEBUG] Validated out hallucinatory tickers for '"
                    f"{row['topic_name']}' : {extracted_tickers}"
                )

            return valid_impacts

        except Exception as e:
            print(f"[ERROR] Mapping failed for '{row['topic_name']}': {e}")
            return []

    def _calculate_instrument_severity(self, group: pd.DataFrame) -> float:
        """
        Calculates the directional severity score for a financial instrument using the
        Directional Confirmed Impact Model (DCIM).

        This method replaces simple heuristic counting with a robust quantitative model.
        It evaluates a collection of news articles (or topics) related to a specific
        instrument by processing them through five distinct stages:
        1. Time Decay: Applies an exponential decay based on the article's
        age (half-life).
        2. Echo Chamber Penalty: Penalizes redundant publications from the same source
        using a harmonic decay function.
        3. Consensus Bonus: Amplifies the signal if multiple independent sources agree
        on the sentiment direction.
        4. Velocity Multiplier: Increases the impact if there is a sudden influx of
        recent articles (breaking news).
        5. Market Saturation: Flattens the final score into a [-1.0, 1.0] range using a
        hyperbolic tangent (tanh) function.

        Args:
            group (pd.DataFrame): A DataFrame containing the articles/topics related to
            the instrument.
                Must contain the following columns:
                - 'date' (datetime): The publication timestamp of the article.
                - 'source_name' (str): The name or identifier of the publisher
                (e.g., 'Bloomberg', 'Reuters').
                - 'sentiment' (float): The directional sentiment of the article,
                typically ranging from -1.0 (highly negative) to 1.0 (highly positive).
                - 'source_weight' (float): The credibility or influence weight of the
                source, strictly in the range (0.0, 1.0].

        Returns:
            float: The final instrument severity score, bounded between -1.0 and 1.0.
                - Values near -1.0 indicate a strong negative (bearish) consensus.
                - Values near  0.0 indicate neutral sentiment, conflicting signals,
                or market noise.
                - Values near +1.0 indicate a strong positive (bullish) consensus.

        Raises:
            ValueError: If the input `group` DataFrame is missing any of the
            required columns.

        Notes:
            The method relies on several class-level hyperparameters prefixed with
            `DCIM_` (e.g., `DCIM_HALF_LIFE_HOURS`,
            `DCIM_BETA_PENALTY`) to calibrate the model's sensitivity. It also attempts
            to dynamically resolve the reference
            time via `self.current_time` to calculate the exact age of the articles.
        """
        if group.empty:
            return 0.0

        required_cols = ["date", "source_name", "sentiment", "source_weight"]
        if not all(col in group.columns for col in required_cols):
            print(
                "[WARNING] Missing DCIM required columns."
                + "Using simple article counter instead"
            )
            return float(len(group))

        # PHASE 1 - influence of a single article
        time_diffs = (self.current_time - group["date"]).dt.total_seconds() / 3600.0
        time_diffs = np.maximum(time_diffs, 0.0)

        decay_constant = np.log(2) / self.DCIM_HALF_LIFE_HOURS
        decay_factors = np.exp(-decay_constant * time_diffs)

        raw_impacts = group["source_weight"] * group["sentiment"] * decay_factors

        temp_df = group.copy()
        temp_df["raw_impact"] = raw_impacts
        temp_df = temp_df.sort_values(by="date", ascending=False)

        # PHASE 2 - echo chamber penalty
        source_impacts = []
        for _, s_group in temp_df.groupby("source_name"):
            n_s = len(s_group)
            j_indices = np.arange(n_s)

            penalties = 1.0 + self.DCIM_BETA_PENALTY * j_indices
            abs_impacts = np.abs(s_group["raw_impact"].values)
            penalized_sum = np.sum(abs_impacts / penalties)

            mean_impact = s_group["raw_impact"].mean()
            sign_s = np.sign(mean_impact) if mean_impact != 0 else 1.0
            source_impacts.append(sign_s * penalized_sum)

        source_impacts = np.array(source_impacts)
        total_impact_sum = np.sum(source_impacts)

        if total_impact_sum == 0:
            return 0.0

        # PHASE 3 - different sources bonus (consensus)
        U = len(source_impacts)
        if U >= 2:
            global_sign = np.sign(total_impact_sum)
            U_agree = np.sum(np.sign(source_impacts) == global_sign)
            C = 1.0 + self.DCIM_GAMMA_CONSENSUS * (U_agree / U)
        else:
            C = 1.0

        # PHASE 4 - velocity
        N_total = len(group)
        N_recent = np.sum(time_diffs <= self.DCIM_RECENT_WINDOW_HOURS)
        V = N_recent / (N_total + 1.0)

        # PHASE 5 - final severity
        inside_tanh = self.DCIM_KAPPA_SCALE * C * (1.0 + self.DCIM_DELTA_VELOCITY * V)
        inside_tanh *= total_impact_sum
        severity = math.tanh(inside_tanh)

        return float(np.clip(severity, -1.0, 1.0))

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
            severity (float): A calculated severity score based on the number of
            related topics.

        Returns:
            dict: A dictionary containing the trend information, including a unique
            trend ID, timestamps, instrument name, title, description, severity score,
            direction, and associated topic IDs.

        """
        topics_context = group[["topic_name", "topic_description"]].to_dict(
            orient="records"
        )  # type: ignore[call-overload]
        topic_ids = group["topic_id"].tolist()

        # Derive time range from the articles' posted timestamps
        # (ISO strings sort correctly)
        first_article_time = str(group["date"].min())[:19]
        newest_article_time = str(group["date"].max())[:19]

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
            "trend_id": str(uuid.uuid4()),
            "current_time": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "first_article_time": first_article_time,
            "newest_article_time": newest_article_time,
            "instrument": instrument,
            "title": result.get("title", "Unknown"),
            "description": result.get("description", "Unknown"),
            "severity": round(severity, 2),
            "direction": result.get("direction", "neutral").lower(),
            "topic_ids": topic_ids,
        }


if __name__ == "__main__":
    print("====== UpperPathAnalyzer Module Test ======")

    try:
        upper_path_analyzer = UpperPathAnalyzer(INSTRUMENT_PATH)
        df_topics = pd.read_csv(TOPICS_PATH)
        df_articles = pd.read_csv(ARTICLES_PATH)
        df_weights = pd.read_csv(WEIGHTS_PATH)

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
        saving_result = UpperPathAnalyzer.save_trends(result, SAVE_PATH_PARQUET)
        if saving_result:
            print("====== Saving Completed ======")
        else:
            print("====== Saving Failed ======")

        # For debugging: save the result to a CSV as well
        result.to_csv(SAVE_PATH_CSV, index=False)
        print(
            "[DEBUG] Also saved results to CSV for quick inspection at  + "
            f"{SAVE_PATH_CSV}"
        )
    except Exception as e:
        print(f"[FATAL ERROR] An error occurred during the UpperPathAnalyzer test: {e}")
