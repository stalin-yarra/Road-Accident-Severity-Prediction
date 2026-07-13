"""
src/utils/logger.py
====================
Reusable logging configuration for the Road Accident Severity Prediction
project. Every module in `src/` and every page in `app/` should obtain its
logger via `get_logger(__name__)` rather than configuring logging itself, so
log formatting and destinations stay consistent project-wide.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config.settings import LOG_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_loggers: set[str] = set()


def get_logger(name: str, log_to_file: bool = True, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a configured logger for a given module name.

    Args:
        name: Usually `__name__` of the calling module, so log lines are
            traceable back to their source.
        log_to_file: If True, also write log records to `logs/app.log`, in
            addition to the console. Set to False for lightweight, ephemeral
            scripts that should not persist logs.
        level: Logging level for this logger (default `logging.INFO`).

    Returns:
        A configured `logging.Logger` instance. Safe to call multiple times
        for the same `name` — handlers are only attached once.
    """
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # If the log directory can't be created/written (e.g. read-only
            # deployment environment), fall back to console-only logging
            # rather than crashing the application over a logging concern.
            logger.warning("Could not attach file log handler; logging to console only.")

    _configured_loggers.add(name)
    return logger
