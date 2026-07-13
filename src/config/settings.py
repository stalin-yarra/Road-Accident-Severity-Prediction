"""
src/config/settings.py
=======================
Central configuration for the Road Accident Severity Prediction project.

Every path, filename, and shared constant used across the `src/` package and
the Streamlit application is defined here — nowhere else in the codebase
should a raw path string be hardcoded. This keeps the project portable
(moving the repository, renaming the production model file, etc. requires
changing exactly one file) and is the single source of truth referenced by
`app/streamlit_app.py` and every page under `app/pages/`.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# PROJECT ROOT
# =============================================================================
# This file lives at <project_root>/src/config/settings.py, so the project
# root is three levels up from this file's location.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# =============================================================================
# DATASET PATHS
# =============================================================================
DATASET_DIR: Path = PROJECT_ROOT / "Dataset"
RAW_DATA_DIR: Path = DATASET_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATASET_DIR / "processed"
ML_READY_DIR: Path = PROCESSED_DATA_DIR / "ml_ready"

CLEANED_DATA_PATH: Path = PROCESSED_DATA_DIR / "cleaned_accident_data.csv"
FEATURED_DATA_PATH: Path = PROCESSED_DATA_DIR / "featured_accident_data.csv"
X_TRAIN_PATH: Path = ML_READY_DIR / "X_train.csv"
X_TEST_PATH: Path = ML_READY_DIR / "X_test.csv"
Y_TRAIN_PATH: Path = ML_READY_DIR / "y_train.csv"
Y_TEST_PATH: Path = ML_READY_DIR / "y_test.csv"

# =============================================================================
# MODEL PATHS
# =============================================================================
MODELS_DIR: Path = PROJECT_ROOT / "models" / "trained"
PRODUCTION_MODEL_FILENAME: str = "best_gradient_boosting.pkl"
PRODUCTION_MODEL_PATH: Path = MODELS_DIR / PRODUCTION_MODEL_FILENAME
PRODUCTION_MODEL_DISPLAY_NAME: str = "Gradient Boosting"

# =============================================================================
# REPORT / FIGURE PATHS (all reused by the Streamlit app — never recomputed)
# =============================================================================
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

DATASET_REPORT_DIR: Path = REPORTS_DIR / "dataset_report"
DATASET_REPORT_PATH: Path = DATASET_REPORT_DIR / "Dataset_Report.md"
DATA_DICTIONARY_PATH: Path = DATASET_REPORT_DIR / "Data_Dictionary.csv"

EDA_FIGURES_DIR: Path = FIGURES_DIR  # Phase 4 EDA figures were saved directly here
MODEL_COMPARISON_FIGURES_DIR: Path = FIGURES_DIR / "model_comparison"
HYPERPARAMETER_TUNING_FIGURES_DIR: Path = FIGURES_DIR / "hyperparameter_tuning"
FINAL_VALIDATION_FIGURES_DIR: Path = FIGURES_DIR / "final_validation"

BASELINE_MODEL_RESULTS_PATH: Path = REPORTS_DIR / "model_results.csv"
TUNED_MODEL_RESULTS_PATH: Path = REPORTS_DIR / "tuned_model_results.csv"
FINAL_METRICS_PATH: Path = REPORTS_DIR / "final_metrics.csv"
FINAL_SUMMARY_MARKDOWN_PATH: Path = REPORTS_DIR / "final_model_summary.md"

# =============================================================================
# APP PATHS
# =============================================================================
APP_DIR: Path = PROJECT_ROOT / "app"
ASSETS_DIR: Path = APP_DIR / "assets"
LOGO_PATH: Path = ASSETS_DIR / "logo.png"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# =============================================================================
# SHARED CONSTANTS
# =============================================================================
RANDOM_STATE: int = 42
TARGET_COLUMN: str = "Accident_Severity"
CLASS_LABELS: list[str] = ["Slight", "Serious", "Fatal"]
METRIC_AVERAGE: str = "weighted"

# Severity -> display color, reused everywhere a severity badge/chart is shown
# so color-coding is consistent across every page of the application.
SEVERITY_COLORS: dict[str, str] = {
    "Slight": "#4C956C",
    "Serious": "#F2A541",
    "Fatal": "#D7263D",
}

# =============================================================================
# FORM VOCABULARIES FOR THE PREDICTION PAGE
# =============================================================================
# These mirror the raw UK STATS19 category vocabularies used throughout the
# feature engineering notebook (04_Feature_Engineering.ipynb), so a user's
# selection can be mapped through the exact same categorize_* logic that
# produced the model's training features. See src/prediction/preprocessing.py.
WEATHER_CONDITIONS_OPTIONS: list[str] = [
    "Fine no high winds",
    "Raining no high winds",
    "Fine + high winds",
    "Raining + high winds",
    "Snowing no high winds",
    "Snowing + high winds",
    "Fog or mist",
    "Other",
]

ROAD_SURFACE_CONDITIONS_OPTIONS: list[str] = [
    "Dry",
    "Wet or damp",
    "Frost or ice",
    "Snow",
    "Flood over 3cm. deep",
]

LIGHT_CONDITIONS_OPTIONS: list[str] = [
    "Daylight",
    "Darkness - lights lit",
    "Darkness - lighting unknown",
    "Darkness - no lighting",
]

ROAD_TYPE_OPTIONS: list[str] = [
    "Single carriageway",
    "Dual carriageway",
    "Roundabout",
    "One way street",
    "Slip road",
]

FIRST_ROAD_CLASS_OPTIONS: list[str] = ["A", "B", "C", "Unclassified", "Motorway"]

JUNCTION_DETAIL_OPTIONS: list[str] = [
    "Not at junction or within 20 metres",
    "Roundabout",
    "Crossroads",
    "T or staggered junction",
    "Slip road",
    "Other junction",
]

JUNCTION_CONTROL_OPTIONS: list[str] = [
    "Give way or uncontrolled",
    "Auto traffic signal",
    "Stop sign",
    "Authorised person",
]

URBAN_RURAL_OPTIONS: list[str] = ["Urban", "Rural"]

VEHICLE_TYPE_OPTIONS: list[str] = [
    "Car",
    "Taxi/Private hire car",
    "Motorcycle over 500cc",
    "Motorcycle 125cc and under",
    "Van / Goods 3.5 tonnes mgw or under",
    "Bus or coach",
    "Pedal cycle",
    "Goods vehicle over 3.5 tonnes",
]

DAY_OF_WEEK_OPTIONS: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

SPEED_LIMIT_OPTIONS: list[int] = [20, 30, 40, 50, 60, 70]
