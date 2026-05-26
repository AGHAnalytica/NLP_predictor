# Core imports
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import math

# Handle API and LLM
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load environment variables from the .env file located in the project root
load_dotenv()

# Default instrument path (resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parents[2]
INSTRUMENT_PATH = str(BASE_DIR / "data" / "instruments" / "instruments_sample.csv")
SAVE_PATH = str(BASE_DIR / "data" / "test_results" / "instrument_trends_report.csv")


# Define Pydantic schemas for strict LLM JSON outputs
class TickerImpact(BaseModel):
    ticker: str = Field(description="Exact ticker symbol from the available list.")
    sentiment: float = Field(
        description="Directional impact of the news on THIS SPECIFIC ticker. "
                    "From -1.0 (highly negative) to 1.0 (highly positive). 0.0 is neutral."
    )

class InstrumentMapping(BaseModel):
    impacts: list[TickerImpact] = Field(
        description="List of impacted tickers and their sentiment. Return empty if none."
    )


class InstrumentTrend(BaseModel):
    title: str = Field(description="Short summary of the trend.")
    description: str = Field(
        description="Detailed explanation of the trend and its impact."
    )
    direction: str = Field(description="One of: 'up', 'down', 'neutral'.")


class UpperPathAnalyzer:
    """
    This class encapsulates the logic for processing the upper path of the data
    pipeline. It includes methods for mapping news topics to financial instruments,
    generating trend summaries, and saving the results to a partitioned Parquet dataset.
    The class is designed to be robust against API rate limits and potential
    hallucinations by the LLM, with strict validation against known instruments.
    It also includes detailed logging for monitoring the processing steps and handling a
    ny issues that arise.

    The main method, `process_upper_path`, takes a DataFrame of topics, maps each
    topic to relevant instruments, and generates a trend summary for each instrument
    based on the associated topics. The final output is a DataFrame containing the
    trends for each instrument, ready to be saved as a partitioned Parquet dataset.

    """

    def __init__(self, instruments_path: str = INSTRUMENT_PATH):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Critical Security Error: OPENAI_API_KEY is not set in the environment."
            )
        self.client = OpenAI(api_key=api_key)

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

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=lambda retry_state: print(
            f"[WAIT] Rate limit hit. Sleeping... (Attempt {retry_state.attempt_number})"
        ),
    )
    def _call_openai_api_with_retry(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        """
        Calls the OpenAI API with structured output and returns a validated
        Pydantic object. Uses exponential backoff to handle rate limits.

        Args:
            prompt (str): The prompt to send to the LLM.
            schema (type[BaseModel]): The Pydantic model class to validate the output.

        Returns:
            BaseModel: An instance of the provided Pydantic model containing the parsed
            response.

        """
        response = self.client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format=schema,
        )
        return response.choices[0].message.parsed  # type: ignore

    def process_upper_path(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing function for the upper path. It takes a DataFrame of topics,
        maps each topic to relevant financial instruments along with their entity-specific
        sentiment, and then generates a trend summary for each instrument based on 
        the associated topics using the DCIM severity model.

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

        topics_df["impacts"] = topics_df.apply(
            self._map_topic_to_instrument, axis=1
        )

        exploded_df = topics_df.explode("impacts")

        exploded_df = exploded_df.dropna(subset=["impacts"])

        if exploded_df.empty:
            print(
                "[WARNING] All topics were dropped after mapping. "
                "The result will be empty."
            )
            return pd.DataFrame()

        impacts_expanded = exploded_df["impacts"].apply(pd.Series)
        
        mapped_df = pd.concat([exploded_df.drop(columns=["impacts"]), impacts_expanded], axis=1)

        mapped_df["instrument"] = mapped_df["instrument"].astype(str)
        mapped_df["sentiment"] = mapped_df["sentiment"].astype(float)

        results = []
        print("[INFO] Generating trends per instrument using the DCIM model...")
        for instrument, group in mapped_df.groupby("instrument"):
            severity = self._calculate_instrument_severity(group)
            trend_data = self._generate_instrument_trend(instrument, group, severity)
            results.append(trend_data)

        return pd.DataFrame(results)

    def _map_topic_to_instrument(self, row: pd.Series) -> list[str]:
        """Maps a single topic to relevant financial instruments using the LLM and calculates sentiment.
        The prompt provides the topic name and description, along with a list of
        available instruments (tickers and full names). The LLM is asked to return a
        list of ticker symbols that are directly impacted by the topic, along with
        a sentiment score for each.
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
        
        For each impacted instrument, calculate the 'sentiment' score between -1.0 and 1.0, 
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
                    valid_impacts.append({
                        "instrument": ticker,
                        "sentiment": float(item.sentiment)
                    })

            if not valid_impacts and extracted_tickers:
                print(
                    "[DEBUG] Validated out hallucinatory tickers for '"
                    f"{row['topic_name']}' : {extracted_tickers}"
                )

            return valid_impacts

        except Exception as e:
            print(f"[ERROR] Mapping failed for '{row['topic_name']}': {e}")
            return []

    # =========================================================================
    # DCIM HIPERPARAMETERS (Directional Confirmed Impact Model)
    # =========================================================================
    DCIM_HALF_LIFE_HOURS = 4.0
    DCIM_BETA_PENALTY = 0.5
    DCIM_GAMMA_CONSENSUS = 0.3
    DCIM_DELTA_VELOCITY = 0.3
    DCIM_KAPPA_SCALE = 2.0
    DCIM_RECENT_WINDOW_HOURS = 0.5

    def _calculate_instrument_severity(self, group: pd.DataFrame) -> float:
        """
        Calculates the directional severity score for a financial instrument using the Directional Confirmed Impact Model (DCIM).

        This method replaces simple heuristic counting with a robust quantitative model. It evaluates a collection
        of news articles (or topics) related to a specific instrument by processing them through five distinct stages:
        1. Time Decay: Applies an exponential decay based on the article's age (half-life).
        2. Echo Chamber Penalty: Penalizes redundant publications from the same source using a harmonic decay function.
        3. Consensus Bonus: Amplifies the signal if multiple independent sources agree on the sentiment direction.
        4. Velocity Multiplier: Increases the impact if there is a sudden influx of recent articles (breaking news).
        5. Market Saturation: Flattens the final score into a [-1.0, 1.0] range using a hyperbolic tangent (tanh) function.

        Args:
            group (pd.DataFrame): A DataFrame containing the articles/topics related to the instrument.
                Must contain the following columns:
                - 'date' (datetime): The publication timestamp of the article.
                - 'source_name' (str): The name or identifier of the publisher (e.g., 'Bloomberg', 'Reuters').
                - 'sentiment' (float): The directional sentiment of the article, typically ranging from -1.0 (highly negative) to 1.0 (highly positive).
                - 'source_weight' (float): The credibility or influence weight of the source, strictly in the range (0.0, 1.0].

        Returns:
            float: The final instrument severity score, bounded between -1.0 and 1.0.
                - Values near -1.0 indicate a strong negative (bearish) consensus.
                - Values near  0.0 indicate neutral sentiment, conflicting signals, or market noise.
                - Values near +1.0 indicate a strong positive (bullish) consensus.

        Raises:
            ValueError: If the input `group` DataFrame is missing any of the required columns.

        Notes:
            The method relies on several class-level hyperparameters prefixed with `DCIM_` (e.g., `DCIM_HALF_LIFE_HOURS`,
            `DCIM_BETA_PENALTY`) to calibrate the model's sensitivity. It also attempts to dynamically resolve the reference
            time via `self.current_time` to calculate the exact age of the articles.
        """
        if group.empty:
            return 0.0

        required_cols = ['date', 'source_name', 'sentiment', 'source_weight']
        if not all(col in group.columns for col in required_cols):
            print(f"[WARNING] Missing DCIM required columns. Using simple article counter instead")
            return float(len(group))

        # PHASE 1 - influence of a single article
        time_diffs = (self.current_time - group['date']).dt.total_seconds() / 3600.0
        time_diffs = np.maximum(time_diffs, 0.0)

        decay_constant = np.log(2) / self.DCIM_HALF_LIFE_HOURS
        decay_factors = np.exp(-decay_constant * time_diffs)

        raw_impacts = group['source_weight'] * group['sentiment'] * decay_factors

        temp_df = group.copy()
        temp_df['raw_impact'] = raw_impacts
        temp_df = temp_df.sort_values(by='date', ascending=False)

        print(f"[DEBUG] temp_df: {temp_df}")
        # PHASE 2 - echo chamber penalty
        source_impacts = []
        for source, s_group in temp_df.groupby('source_name'):
            n_s = len(s_group)
            j_indices = np.arange(n_s)

            penalties = 1.0 + self.DCIM_BETA_PENALTY * j_indices
            abs_impacts = np.abs(s_group['raw_impact'].values)
            penalized_sum = np.sum(abs_impacts / penalties)

            mean_impact = s_group['raw_impact'].mean()
            sign_s = np.sign(mean_impact) if mean_impact != 0 else 1.0
            source_impacts.append(sign_s * penalized_sum)

        source_impacts = np.array(source_impacts)
        total_impact_sum = np.sum(source_impacts)
        
        if total_impact_sum == 0:
            return 0.0

        print(f"[DEBUG] total_impact_sum: {total_impact_sum}")
        print(f"[DEBUG] source_impacts: {source_impacts}")
        # PHASE 3 - different sources bonus (consensus)
        U = len(source_impacts)
        if U >= 2:
            global_sign = np.sign(total_impact_sum)
            U_agree = np.sum(np.sign(source_impacts) == global_sign)
            C = 1.0 + self.DCIM_GAMMA_CONSENSUS * (U_agree / U)
        else:
            C = 1.0

        print(f"[DEBUG] C: {C}")
        # PHASE 4 - velocity
        N_total = len(group)
        N_recent = np.sum(time_diffs <= self.DCIM_RECENT_WINDOW_HOURS)
        V = N_recent / (N_total + 1.0)

        print(f"[DEBUG] N_total: {N_total}")
        print(f"[DEBUG] N_recent: {N_recent}")
        print(f"[DEBUG] V: {V}")

        #PHASE 5 - final severity
        inside_tanh = self.DCIM_KAPPA_SCALE * C * (1.0 + self.DCIM_DELTA_VELOCITY * V) * total_impact_sum
        severity = math.tanh(inside_tanh)

        print(f"[DEBUG] severity: {severity}")
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

    @staticmethod
    def save_trends(result_df: pd.DataFrame, output_path: str):
        """
        Saves the resulting DataFrame to a partitioned Parquet dataset.
        - Partitions by date (extracted from 'current_time' or 'date' column).
        - Uses fastparquet engine for efficient writing and native
        Hive-style partitioning.
        - Appends new data without overwriting existing partitions, creating new
        parquet files with unique names.
        - Handles empty DataFrames gracefully by skipping the save operation.
        - Ensures the target directory exists before saving.
        - Logs key steps and any issues encountered during the save process.
        Note: The 'current_time' column is expected to be in a datetime format.
        If it's not present, it falls back to using the 'date' column for partitioning.

        Args:
            result_df (pd.DataFrame): The DataFrame containing the trends to save.
            output_path (str): The path to the directory where the Parquet files will be
            saved.

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if result_df.empty:
            print("[INFO] DataFrame is empty. Skipping Parquet save operation.")
            return False

        try:
            # Ensure the target base directory exists
            os.makedirs(output_path, exist_ok=True)

            # Identify the time column ('current_time' based on latest implementation)
            time_col = "current_time" if "current_time" in result_df.columns else "date"

            # Create a new column strictly for partitioning (extracting just the date)
            # Converting to string ensures clean folder names in the OS
            result_df["date_partition"] = pd.to_datetime(
                result_df[time_col]
            ).dt.date.astype(str)

            # Check SPECIFICALLY for the fastparquet metadata file to enable appending
            metadata_path = os.path.join(output_path, "_metadata")
            dataset_exists = os.path.isfile(metadata_path)

            # Write to Parquet using the fastparquet engine.
            # fastparquet handles Hive partitioning natively.
            # When called multiple times, it generates new, unique parquet files (UUIDs)
            # inside the partition folders, effectively appending data without
            # overwriting historical batches.
            result_df.to_parquet(
                output_path,
                engine="fastparquet",
                partition_cols=["date_partition"],
                index=False,
                append=dataset_exists,
            )

            mode_str = "Appended to" if dataset_exists else "Created new"
            print(f"[INFO] {mode_str} partitioned Parquet dataset at {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save results to Parquet: {e}")
            return False


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    INSTRUMENT_PATH = str(BASE_DIR / "data" / "instruments" / "instruments_sample.csv")
    TOPICS_PATH = str(BASE_DIR / "data" / "topics" / "topics_sample.csv")
    SAVE_PATH = str(BASE_DIR / "data" / "test_results" / "instrument_trends_parquet")
    ARTICLES_PATH = str(BASE_DIR / "data" / "scraping" / "articles_sample.csv")
    WEIGHTS_PATH = str(BASE_DIR / "data" / "scraping" / "source_weights_sample.csv")

    print("====== UpperPathAnalyzer Module Test ======")

    upper_path_analyzer = UpperPathAnalyzer(INSTRUMENT_PATH)
    df_topics = pd.read_csv(TOPICS_PATH)
    df_articles = pd.read_csv(ARTICLES_PATH)
    df_weights = pd.read_csv(WEIGHTS_PATH)

    df_article_mapping = pd.merge(
        df_articles[["doc_hash", "source_name"]], 
        df_weights[["source_name", "source_weight"]], 
        on="source_name", 
        how="left"
    )

    df_topics_enriched = pd.merge(
        df_topics, 
        df_article_mapping[["doc_hash", "source_weight", "source_name"]], 
        on="doc_hash", 
        how="left"
    )

    df_topics_enriched["source_weight"] = df_topics_enriched["source_weight"].fillna(1.0)
    
    df_topics_enriched["date"] = pd.to_datetime(df_topics_enriched["date"], utc=True)
    
    print("[INFO] Successfully merged source weights into topics.")

    result = upper_path_analyzer.process_upper_path(df_topics_enriched)
    print("====== Processing Upper Path Completed ======")

    print("====== Saving Results To File ======")
    saving_result = UpperPathAnalyzer.save_trends(result, SAVE_PATH)
    if saving_result:
        print("====== Saving Completed ======")
    else:
        print("====== Saving Failed ======")

    # For debugging: save the result to a CSV as well
    # (not required, but useful for quick checks)
    debug_csv_path = str(
        BASE_DIR / "data" / "test_results" / "instrument_trends_report.csv"
    )
    result.to_csv(debug_csv_path, index=False)
    print(f"[DEBUG] Also saved results to CSV for quick inspection at {debug_csv_path}")
