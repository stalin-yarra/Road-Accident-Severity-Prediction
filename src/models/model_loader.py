"""
src/models/model_loader.py
============================
Loads the production model exactly as saved by
`07_Hyperparameter_Tuning.ipynb` — this module never fits, tunes, or
modifies the model in any way. It exists purely to centralize model-loading
logic (with logging and graceful error handling) so both the prediction
pipeline and any Streamlit page needing direct model access share one
implementation.
"""

from __future__ import annotations

from typing import Optional

from src.config.settings import PRODUCTION_MODEL_DISPLAY_NAME, PRODUCTION_MODEL_PATH
from src.utils.io_helpers import load_model_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when the production model cannot be loaded from disk."""


def load_production_model():
    """
    Load the saved production model (`best_gradient_boosting.pkl`).

    Returns:
        The deserialized, already-trained scikit-learn estimator.

    Raises:
        ModelLoadError: If the model file is missing or cannot be
            deserialized. Callers in the Streamlit app should catch this
            and render a friendly instruction to run the training
            notebooks, rather than letting a raw traceback surface.
    """
    model = load_model_safe(PRODUCTION_MODEL_PATH)
    if model is None:
        raise ModelLoadError(
            f"Could not load '{PRODUCTION_MODEL_DISPLAY_NAME}' from "
            f"{PRODUCTION_MODEL_PATH}. Please run "
            f"'notebooks/07_Hyperparameter_Tuning.ipynb' to produce this file."
        )
    return model


def get_model_feature_names(model) -> Optional[list[str]]:
    """
    Retrieve the exact, ordered feature names the model was trained on, if
    available. Scikit-learn estimators fit on a pandas DataFrame expose this
    via `feature_names_in_` (available since scikit-learn 1.0).

    Args:
        model: A fitted scikit-learn-compatible estimator.

    Returns:
        A list of feature names in training order, or None if the estimator
        does not expose this attribute (e.g., it was fit on a plain numpy
        array rather than a DataFrame).
    """
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        logger.warning(
            "Model does not expose 'feature_names_in_'; a fallback feature "
            "schema (e.g., from X_train.csv) will be required for prediction."
        )
        return None
    return list(feature_names)
