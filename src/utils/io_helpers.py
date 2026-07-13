"""
src/utils/io_helpers.py
========================
Reusable, exception-safe I/O helpers shared by the prediction pipeline and
every Streamlit page. Centralizing these avoids duplicating the same
try/except-and-log pattern in six different `app/pages/*.py` files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model_safe(model_path: Path):
    """
    Load a joblib-serialized model, returning None (rather than raising) if
    the file is missing or corrupt, so calling UI code can render a friendly
    message instead of crashing the app.

    Args:
        model_path: Path to the `.pkl` model file.

    Returns:
        The deserialized model object, or None if it could not be loaded.
    """
    if not model_path.exists():
        logger.warning("Model file not found: %s", model_path)
        return None
    try:
        model = joblib.load(model_path)
        logger.info("Loaded model from %s (%s)", model_path, type(model).__name__)
        return model
    except (OSError, EOFError, ValueError) as error:
        logger.error("Failed to load model from %s: %s", model_path, error)
        return None


def load_csv_safe(csv_path: Path, **read_csv_kwargs) -> Optional[pd.DataFrame]:
    """
    Load a CSV file, returning None (rather than raising) if it is missing
    or unreadable.

    Args:
        csv_path: Path to the CSV file.
        **read_csv_kwargs: Extra keyword arguments forwarded to `pd.read_csv`
            (e.g. `nrows=1000` to preview a large file cheaply).

    Returns:
        The loaded DataFrame, or None if it could not be loaded.
    """
    if not csv_path.exists():
        logger.warning("CSV file not found: %s", csv_path)
        return None
    try:
        return pd.read_csv(csv_path, **read_csv_kwargs)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as error:
        logger.error("Failed to load CSV from %s: %s", csv_path, error)
        return None


def load_markdown_safe(markdown_path: Path) -> Optional[str]:
    """
    Load the raw text content of a markdown file, returning None if it is
    missing or unreadable.

    Args:
        markdown_path: Path to the `.md` file.

    Returns:
        The file's text content, or None if it could not be loaded.
    """
    if not markdown_path.exists():
        logger.warning("Markdown file not found: %s", markdown_path)
        return None
    try:
        return markdown_path.read_text(encoding="utf-8")
    except OSError as error:
        logger.error("Failed to read markdown file %s: %s", markdown_path, error)
        return None


def image_exists(image_path: Path) -> bool:
    """
    Check whether an image file exists, as a lightweight guard before
    calling `st.image()` on a path that may not have been generated yet
    (e.g., a notebook phase the user hasn't run on their own machine).

    Args:
        image_path: Path to the image file.

    Returns:
        True if the file exists, False otherwise.
    """
    return image_path.exists() and image_path.is_file()
