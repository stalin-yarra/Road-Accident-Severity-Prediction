"""
src/visualization/charts.py
==============================
Small, live-computed chart builders for the Streamlit app. Everything that
*can* be reused from a notebook-generated PNG is loaded via
`report_loader.py` instead — this module covers only the handful of visuals
that are inherently interactive/request-specific (the prediction probability
bar chart) or genuinely optional/on-demand (ROC and Precision-Recall curves
on Page 4, computed only if the test set and model are both available).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from src.config.settings import SEVERITY_COLORS


def build_probability_bar_chart(probabilities: dict[str, float]) -> go.Figure:
    """
    Build an interactive horizontal bar chart of predicted class
    probabilities, colored using the shared severity color palette.

    Args:
        probabilities: Mapping of class label -> predicted probability.

    Returns:
        A Plotly Figure ready for `st.plotly_chart()`.
    """
    labels = list(probabilities.keys())
    values = [probabilities[label] * 100 for label in labels]
    colors = [SEVERITY_COLORS.get(label, "#6b7280") for label in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Predicted Severity Probability",
        xaxis_title="Probability (%)",
        yaxis_title="",
        xaxis_range=[0, 100],
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
    )
    return fig


def build_roc_curves(model, X_test: pd.DataFrame, y_test: pd.Series, class_labels: list[str]) -> Optional[go.Figure]:
    """
    Compute one-vs-rest ROC curves for a multi-class model, on demand.

    Args:
        model: A fitted classifier exposing `predict_proba`.
        X_test: Test feature set.
        y_test: True test labels.
        class_labels: Ordered class labels matching `model.classes_`.

    Returns:
        A Plotly Figure with one ROC curve per class, or None if the model
        does not support `predict_proba`.
    """
    if not hasattr(model, "predict_proba"):
        return None

    y_test_binarized = label_binarize(y_test, classes=class_labels)
    y_scores = model.predict_proba(X_test)

    fig = go.Figure()
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{label} (AUC = {roc_auc:.3f})",
                line=dict(color=SEVERITY_COLORS.get(label, "#6b7280")),
            )
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="gray")))
    fig.update_layout(
        title="ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=450,
    )
    return fig


def build_precision_recall_curves(
    model, X_test: pd.DataFrame, y_test: pd.Series, class_labels: list[str]
) -> Optional[go.Figure]:
    """
    Compute one-vs-rest Precision-Recall curves for a multi-class model, on
    demand — particularly informative under class imbalance, where ROC
    curves can look deceptively strong.

    Args:
        model: A fitted classifier exposing `predict_proba`.
        X_test: Test feature set.
        y_test: True test labels.
        class_labels: Ordered class labels matching `model.classes_`.

    Returns:
        A Plotly Figure with one PR curve per class, or None if the model
        does not support `predict_proba`.
    """
    if not hasattr(model, "predict_proba"):
        return None

    y_test_binarized = label_binarize(y_test, classes=class_labels)
    y_scores = model.predict_proba(X_test)

    fig = go.Figure()
    for i, label in enumerate(class_labels):
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_scores[:, i])
        pr_auc = auc(recall, precision)
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision, mode="lines",
                name=f"{label} (AUC = {pr_auc:.3f})",
                line=dict(color=SEVERITY_COLORS.get(label, "#6b7280")),
            )
        )
    fig.update_layout(
        title="Precision-Recall Curves (One-vs-Rest)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=450,
    )
    return fig


def build_target_distribution_chart(target_counts: pd.Series) -> go.Figure:
    """
    Build an interactive bar chart of target class distribution.

    Args:
        target_counts: A Series of class counts (e.g., from `value_counts()`).

    Returns:
        A Plotly Figure ready for `st.plotly_chart()`.
    """
    labels = target_counts.index.tolist()
    colors = [SEVERITY_COLORS.get(label, "#6b7280") for label in labels]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=target_counts.values,
            marker_color=colors,
            text=[f"{v:,}" for v in target_counts.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Accident Severity Distribution",
        xaxis_title="Severity",
        yaxis_title="Number of Records",
        template="plotly_white",
        height=400,
    )
    return fig
