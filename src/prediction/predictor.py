"""
src/prediction/predictor.py
=============================
Public prediction interface used by the Streamlit "Predict Severity" page.
Wraps model loading and feature preprocessing behind a single, simple
`AccidentSeverityPredictor.predict(raw_input)` call. The underlying model is
never retrained or modified here — this module only loads and calls
`.predict()` / `.predict_proba()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.model_loader import load_production_model
from src.prediction.preprocessing import RawAccidentInput, build_feature_vector
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """
    The result of a single severity prediction.

    Attributes:
        predicted_class: The predicted severity label (e.g., "Serious").
        probabilities: Mapping of every class label to its predicted probability.
        confidence: The probability of the predicted class (i.e., max(probabilities)).
    """

    predicted_class: str
    probabilities: dict[str, float]
    confidence: float


class AccidentSeverityPredictor:
    """
    Loads the production Gradient Boosting model once and exposes a simple
    `predict()` method for the Streamlit prediction page. Instantiate this
    once per session (e.g., cached via `st.cache_resource`) rather than
    reloading the model on every prediction.
    """

    def __init__(self) -> None:
        self._model = load_production_model()
        self._class_labels = list(self._model.classes_)
        logger.info(
            "AccidentSeverityPredictor ready (model=%s, classes=%s)",
            type(self._model).__name__, self._class_labels,
        )

    def predict(self, raw_input: RawAccidentInput) -> PredictionResult:
        """
        Predict accident severity for a single raw input record.

        Args:
            raw_input: The raw, human-friendly form input.

        Returns:
            PredictionResult: predicted class, full probability distribution,
                and the confidence (probability) of the predicted class.

        Raises:
            ValueError: If the model does not support `predict_proba` (not
                expected for the Gradient Boosting production model, but
                guarded against for robustness).
        """
        feature_vector = build_feature_vector(raw_input, self._model)

        if not hasattr(self._model, "predict_proba"):
            raise ValueError(
                f"Model of type {type(self._model).__name__} does not support "
                f"'predict_proba'; cannot report class probabilities/confidence."
            )

        probability_array = self._model.predict_proba(feature_vector)[0]
        probabilities = {
            label: float(prob) for label, prob in zip(self._class_labels, probability_array)
        }

        predicted_index = int(np.argmax(probability_array))
        predicted_class = self._class_labels[predicted_index]
        confidence = float(probability_array[predicted_index])

        logger.info(
            "Prediction complete: class=%s confidence=%.4f", predicted_class, confidence
        )

        return PredictionResult(
            predicted_class=predicted_class,
            probabilities=probabilities,
            confidence=confidence,
        )

    @property
    def class_labels(self) -> list[str]:
        """The model's class labels, in the order used internally by scikit-learn."""
        return self._class_labels
