"""Public configuration API exposing the singleton ``config`` and ``REPO_ROOT``."""

from ._config import REPO_ROOT, Config, config

__all__ = ["REPO_ROOT", "config", "Config"]
