"""Base infrastructure for LLM-backed pipeline analyzers.

Provides thread-pooled OpenAI structured-output calls with exponential backoff,
instrument universe loading, and Hive-partitioned Parquet persistence via PyArrow.
"""

import logging
import os
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import pandas as pd
from openai import OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from nlp_predictor.config import config

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def _is_transient_rate_limit(exception: BaseException) -> bool:
    """Return True for transient TPM/RPM rate limits and False for permanent quota
    exhaustion.
    """
    if not isinstance(exception, RateLimitError):
        return False

    code = getattr(exception, "code", None)
    err_type = getattr(exception, "type", None)
    body = getattr(exception, "body", None)

    is_exhausted = (
        code in ("credit_balance_exhausted", "insufficient_quota")
        or err_type == "insufficient_quota"
        or (
            isinstance(body, dict)
            and isinstance(body.get("error"), dict)
            and (
                body["error"].get("code") == "credit_balance_exhausted"
                or body["error"].get("type") == "insufficient_quota"
            )
        )
    )
    if is_exhausted:
        logger.critical(
            "OpenAI billing error: insufficient quota / credit_balance_exhausted. "
            "Add credits at https://platform.openai.com/settings/organization/billing/."
        )
        return False
    return True


def load_instrument_universe(
    instruments_path: str | Path,
) -> tuple[pd.DataFrame, set[str], dict[str, str]]:
    """Load the tradable instrument table and build lookup structures for validation and
    prompts.

    Args:
        instruments_path: Path to the CSV file with ``ticker`` and ``full_name``
            columns.

    Returns:
        Tuple of ``(instruments_df, valid_tickers_set, ticker_to_full_name_dict)``.
    """
    df = pd.read_csv(instruments_path)
    tickers = df["ticker"].dropna().astype(str).str.strip()
    names = df["full_name"].fillna("").astype(str).str.strip()
    valid_tickers = set(tickers)
    available_instruments = dict(zip(tickers, names, strict=False))
    return df, valid_tickers, available_instruments


class BaseAnalyzer:
    """Base class providing OpenAI structured-output retries, paced thread parallelism,
    and
    Parquet export.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        request_delay: float | None = None,
    ):
        """Initialize the OpenAI client and concurrency settings from config or
        overrides.

        Args:
            max_workers: Maximum concurrent LLM threads (defaults to
                ``config.hyperparams.models.max_workers``).
            request_delay: Pacing delay in seconds between thread submissions.

        Raises:
            ValueError: If ``OPENAI_API_KEY`` is not set in the environment.
        """
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

    def _call_openai_api_with_retry(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        """Call OpenAI structured outputs with exponential backoff on transient rate
        limits.

        Args:
            prompt: User prompt sent to the configured LLM.
            schema: Pydantic model class enforcing the response JSON schema.

        Returns:
            Parsed and validated instance of ``schema``.
        """
        models_cfg = config.hyperparams.models
        retrying = Retrying(
            stop=stop_after_attempt(int(models_cfg.retry_attempts)),
            wait=wait_exponential(
                multiplier=1.5,
                min=float(models_cfg.retry_min_wait),
                max=float(models_cfg.retry_max_wait),
            )
            + wait_random(0.5, 3.0),
            retry=retry_if_exception(_is_transient_rate_limit),
            before_sleep=lambda retry_state: logger.warning(
                f"Rate limit hit. Sleeping for {retry_state.upcoming_sleep:.1f}s "
                f"(Attempt {retry_state.attempt_number})..."
            ),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                response = self.client.beta.chat.completions.parse(
                    model=str(models_cfg.llm),
                    messages=[{"role": "user", "content": prompt}],
                    response_format=schema,
                )
                return response.choices[0].message.parsed  # type: ignore[return-value]
        raise RuntimeError("Exhausted OpenAI retry loop.")

    def _parallel_map(self, fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """Execute ``fn`` across ``items`` using a paced thread pool while preserving
        input order.

        Args:
            fn: Callable applied to each element.
            items: Sequence of inputs.

        Returns:
            List of results in the exact order of ``items``.
        """
        if not items:
            return []

        workers = min(self.max_workers, len(items))
        if workers <= 1:
            return [fn(item) for item in items]

        def paced_fn(index_and_item: tuple[int, T]) -> tuple[int, R]:
            idx, item = index_and_item
            if self.request_delay > 0:
                delay = (
                    idx * self.request_delay if idx < workers else self.request_delay
                )
                time.sleep(delay)
            return idx, fn(item)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = ex.map(paced_fn, enumerate(items))
            return [r for _, r in results]

    @staticmethod
    def save_trends(result_df: pd.DataFrame, output_path: str | Path) -> bool:
        """Append ``result_df`` to a date-partitioned Parquet dataset using PyArrow.

        Partitions by ``date_partition`` derived from ``current_time`` (or ``date``).
        Writes unique batch files inside each Hive partition directory without
        overwriting history.

        Args:
            result_df: DataFrame of trends to persist.
            output_path: Root directory for the partitioned Parquet dataset.

        Returns:
            True if written successfully, False if ``result_df`` is empty or an error
            occurred.
        """
        if result_df.empty:
            logger.info("DataFrame is empty. Skipping Parquet save operation.")
            return False

        try:
            out_dir = Path(output_path)
            time_col = "current_time" if "current_time" in result_df.columns else "date"
            df_to_save = result_df.copy()
            df_to_save["date_partition"] = pd.to_datetime(
                df_to_save[time_col]
            ).dt.date.astype(str)

            for date_part, group in df_to_save.groupby("date_partition", sort=False):
                part_dir = out_dir / f"date_partition={date_part}"
                part_dir.mkdir(parents=True, exist_ok=True)
                file_path = part_dir / f"part_{uuid.uuid4().hex[:12]}.parquet"
                group.drop(columns=["date_partition"]).to_parquet(
                    file_path, engine="pyarrow", index=False
                )

            logger.info(f"Saved partitioned Parquet dataset to {out_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to save results to Parquet: {e}")
            return False
