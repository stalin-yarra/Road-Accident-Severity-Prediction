"""
src/visualization/report_loader.py
=====================================
Loads already-generated reports, figures, and tables produced by the project
notebooks (Phases 2-9). This module never recomputes a metric or regenerates
a figure — its entire purpose is to *reuse* what the notebooks already saved
to `reports/`, so the Streamlit app stays fast and stays a single source of
truth with the notebooks that produced its content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import (
    BASELINE_MODEL_RESULTS_PATH,
    DATA_DICTIONARY_PATH,
    DATASET_REPORT_PATH,
    FINAL_METRICS_PATH,
    FINAL_SUMMARY_MARKDOWN_PATH,
    FINAL_VALIDATION_FIGURES_DIR,
    HYPERPARAMETER_TUNING_FIGURES_DIR,
    MODEL_COMPARISON_FIGURES_DIR,
    TUNED_MODEL_RESULTS_PATH,
)
from src.utils.io_helpers import image_exists, load_csv_safe, load_markdown_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_baseline_model_results() -> Optional[pd.DataFrame]:
    """Load the Phase 7 baseline model comparison table, if it exists."""
    return load_csv_safe(BASELINE_MODEL_RESULTS_PATH)


def get_tuned_model_results() -> Optional[pd.DataFrame]:
    """Load the Phase 8 baseline-vs-tuned comparison table, if it exists."""
    return load_csv_safe(TUNED_MODEL_RESULTS_PATH)


def get_final_metrics() -> Optional[pd.DataFrame]:
    """Load the Phase 9 final production-model metrics table, if it exists."""
    return load_csv_safe(FINAL_METRICS_PATH)


def get_final_summary_markdown() -> Optional[str]:
    """Load the Phase 9 auto-generated final research summary, if it exists."""
    return load_markdown_safe(FINAL_SUMMARY_MARKDOWN_PATH)


def get_dataset_report_markdown() -> Optional[str]:
    """Load the Phase 2 dataset report, if it exists."""
    return load_markdown_safe(DATASET_REPORT_PATH)


def get_data_dictionary() -> Optional[pd.DataFrame]:
    """Load the Phase 2 data dictionary table, if it exists."""
    return load_csv_safe(DATA_DICTIONARY_PATH)


def find_figure(figures_dir: Path, filename: str) -> Optional[Path]:
    """
    Resolve the path to a specific saved figure, returning None (rather
    than a path to a nonexistent file) if it has not been generated yet.

    Args:
        figures_dir: The directory the figure is expected in (see
            `src/config/settings.py` for the standard figure directories).
        filename: The exact filename to look up.

    Returns:
        The full Path if the figure exists, otherwise None.
    """
    candidate = figures_dir / filename
    return candidate if image_exists(candidate) else None


def get_model_comparison_figure(filename: str) -> Optional[Path]:
    """Resolve a Phase 7 model-comparison figure by filename, if it exists."""
    return find_figure(MODEL_COMPARISON_FIGURES_DIR, filename)


def get_hyperparameter_tuning_figure(filename: str) -> Optional[Path]:
    """Resolve a Phase 8 hyperparameter-tuning figure by filename, if it exists."""
    return find_figure(HYPERPARAMETER_TUNING_FIGURES_DIR, filename)


def get_final_validation_figure(filename: str) -> Optional[Path]:
    """Resolve a Phase 9 final-validation figure by filename, if it exists."""
    return find_figure(FINAL_VALIDATION_FIGURES_DIR, filename)


def list_figures(figures_dir: Path) -> list[Path]:
    """
    List every PNG figure available in a given figures directory, for pages
    that want to display "whatever was generated" rather than a fixed,
    hardcoded filename list.

    Args:
        figures_dir: The directory to scan.

    Returns:
        A sorted list of PNG file paths (empty if the directory doesn't exist).
    """
    if not figures_dir.exists():
        return []
    return sorted(figures_dir.glob("*.png"))
