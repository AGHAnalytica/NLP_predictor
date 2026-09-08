"""
Application logging configuration.

Sets up structured logging to both console (stdout/stderr) and daily log files.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from nlp_predictor.config import Config


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | str | None = None,
) -> Path:
    """
    Configure application logging to both console and a daily log file.

    Log file path is derived from Config["LOG_PATH"] (default: logs/{date}.log)
    or custom log_dir if provided.

    Args:
        level (int): Logging level (default: logging.INFO).
        log_dir (Path | str | None): Optional custom directory for log files.

    Returns:
        Path: Absolute path to the active log file.
    """
    now = datetime.now(UTC)
    date_str = now.strftime("%Y-%m-%d")

    if log_dir is not None:
        log_file = (Path(log_dir) / f"{date_str}.log").resolve()
    else:
        log_path_str = str(Config["LOG_PATH"]).format(date=date_str)
        log_file = Path(log_path_str).resolve()

    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
    formatter = logging.Formatter(log_format)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    return log_file
