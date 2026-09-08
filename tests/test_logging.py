import logging
from pathlib import Path

from nlp_predictor.logging_config import setup_logging


def test_setup_logging_creates_handlers_and_writes_file(tmp_path: Path) -> None:
    original_handlers = list(logging.getLogger().handlers)
    try:
        log_file = setup_logging(level=logging.DEBUG, log_dir=tmp_path)

        assert log_file.exists()
        assert log_file.parent == tmp_path

        root_logger = logging.getLogger()
        has_stream = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in root_logger.handlers
        )
        has_file = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file
            for h in root_logger.handlers
        )
        assert has_stream, "Root logger should have a StreamHandler"
        assert has_file, "Root logger should have a FileHandler"

        test_msg = "Unique test log message for verification"
        test_logger = logging.getLogger("test_module")
        test_logger.info(test_msg)

        content = log_file.read_text(encoding="utf-8")
        assert test_msg in content
        assert "[INFO]" in content
        assert "test_module" in content
    finally:
        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            h.close()
            root_logger.removeHandler(h)
        for h in original_handlers:
            root_logger.addHandler(h)


def test_setup_logging_default_path(tmp_path: Path) -> None:
    original_handlers = list(logging.getLogger().handlers)
    mock_log_path = str(tmp_path / "mock_logs" / "{date}.log")
    from nlp_predictor.config import Config

    Config["LOG_PATH"] = mock_log_path
    try:
        log_file = setup_logging()
        assert log_file.exists()
        assert log_file.parent.name == "mock_logs"
    finally:
        from nlp_predictor.config._config import _overrides

        _overrides.pop("LOG_PATH", None)
        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            h.close()
            root_logger.removeHandler(h)
        for h in original_handlers:
            root_logger.addHandler(h)


