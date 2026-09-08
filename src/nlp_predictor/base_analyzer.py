import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import pandas as pd
from openai import OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from nlp_predictor.config import config

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BaseAnalyzer(ABC):
    """
    Abstract base class for analyzing topic data and generating trends reports.
    Provides common functionality for calling the OpenAI API with structured
    output and saving results to a partitioned Parquet dataset. Subclasses
    must implement the process_data method to define specific analysis logic.
    Key features:
    - OpenAI API integration with structured output parsing using Pydantic models.
    - Exponential backoff retry mechanism to handle API rate limits gracefully.
    - Robust Parquet saving functionality that appends to existing datasets without
    overwriting, using the fastparquet engine for efficient partitioning and
    writing.

    Note: The class assumes that the OPENAI_API_KEY environment variable is set for
    authentication with the OpenAI API. It also expects the input DataFrame to have
    specific columns (e.g., 'embedding') for processing, which should be handled by the
    subclasses.
    """

    # Max concurrent in-flight OpenAI requests for independent per-item calls.
    # These calls are I/O-bound, so threads (which release the GIL during network
    # I/O) give near-linear speedup; the sync OpenAI client is thread-safe. Higher
    # values are faster but more likely to hit rate limits (handled by the retry).
    MAX_WORKERS = 2

    def __init__(
        self,
        max_workers: int | None = None,
        request_delay: float | None = None,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Critical Security Error: OPENAI_API_KEY is not set in the environment."
            )
        self.client = OpenAI(api_key=api_key)
        self.max_workers = (
            max_workers
            if max_workers is not None
            else int(config.hyperparams.models.max_workers)
        )
        self.request_delay = (
            request_delay
            if request_delay is not None
            else float(config.hyperparams.models.request_delay)
        )

    @retry(
        stop=stop_after_attempt(12),
        wait=wait_random_exponential(min=4, max=60),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limit hit. Sleeping for {retry_state.upcoming_sleep:.1f}s... "
            f"(Attempt {retry_state.attempt_number})"
        ),
    )
    def _call_openai_api_with_retry(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        """
        Calls the OpenAI API with structured output and returns a validated
        Pydantic object. Uses exponential backoff with random jitter to handle
        rate limits and avoid synchronized retries.

        Args:
            prompt (str): The prompt to send to the LLM.
            schema (type[BaseModel]): The Pydantic model class to validate the output.

        Returns:
            BaseModel: An instance of the provided Pydantic model containing the parsed
            response.

        """
        response = self.client.beta.chat.completions.parse(
            model=str(config.hyperparams.models.llm),
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format=schema,
        )
        return response.choices[0].message.parsed  # type: ignore

    def _parallel_map(self, fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """
        Run ``fn`` over ``items`` concurrently, returning results in input order.

        Used to fan out independent LLM calls (e.g. one per article, topic or
        cluster) instead of issuing them serially. Concurrency is capped at
        :attr:`max_workers`; the per-call rate-limit retry still applies inside
        ``fn``. If :attr:`request_delay` > 0, task submissions are paced to
        prevent bursting.

        Args:
            fn (Callable): Function applied to each item (typically wraps one API call).
            items (Sequence): The items to process.

        Returns:
            list: Results aligned 1:1 with ``items``.
        """
        if not items:
            return []

        workers = min(self.max_workers, len(items))
        if workers <= 1:
            return [fn(item) for item in items]

        def paced_fn(index_and_item: tuple[int, T]) -> tuple[int, R]:
            idx, item = index_and_item
            if self.request_delay > 0:
                if idx < workers:
                    time.sleep(idx * self.request_delay)
                else:
                    time.sleep(self.request_delay)
            return idx, fn(item)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = ex.map(paced_fn, enumerate(items))
            return [r for _, r in results]

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
            logger.info("DataFrame is empty. Skipping Parquet save operation.")
            return False

        try:
            # 1. Ensure the target base directory exists
            os.makedirs(output_path, exist_ok=True)

            # 2. Identify the time column
            # ('current_time' based on latest implementation)
            time_col = "current_time" if "current_time" in result_df.columns else "date"

            # 3. Create a new column strictly for partitioning
            # (extracting just the date) Converting to string
            # ensures clean folder names in the OS
            result_df["date_partition"] = pd.to_datetime(
                result_df[time_col]
            ).dt.date.astype(str)

            # 4. Check SPECIFICALLY for the fastparquet metadata file
            # to enable appending
            metadata_path = os.path.join(output_path, "_metadata")
            dataset_exists = os.path.isfile(metadata_path)

            # 5. Write to Parquet using the fastparquet engine.
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
            logger.info(f"{mode_str} partitioned Parquet dataset at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save results to Parquet: {e}")
            return False

    @abstractmethod
    def process_data(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to process the input DataFrame and generate a trends
        report. Must be implemented by subclasses.

        Args:
            topics_df (pd.DataFrame): The input DataFrame containing topic data.

        Returns:
            pd.DataFrame: A DataFrame containing the generated trends report.
        """
        pass
