"""Application logging setup for console and daily rotating file output."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from nlp_predictor.config import config


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | str | None = None,
) -> Path:
    """Configure root logging with a stream handler and a UTC-dated file handler.

    Args:
        level: Logging verbosity level (defaults to ``logging.INFO``).
        log_dir: Optional directory override; defaults to ``config.paths.logs.log``.

    Returns:
        Absolute ``Path`` to the active log file.
    """
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    if log_dir is not None:
        log_file = (Path(log_dir) / f"{date_str}.log").resolve()
    else:
        log_path_str = str(config.paths.logs.log).format(date=date_str)
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
