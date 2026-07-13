"""
app/pages/5_Feature_Importance.py
====================================
Feature Importance page: reuses the built-in, permutation, and (if
generated) SHAP importance figures produced by
`08_Final_Model_Validation_and_Explainability.ipynb`. No importance values
are recomputed on this page.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config.settings import FINAL_VALIDATION_FIGURES_DIR
from src.utils.ui_helpers import configure_page, render_missing_artifact_notice, render_sidebar_branding
from src.visualization.report_loader import get_final_validation_figure, list_figures

configure_page(page_title="Feature Importance", page_icon="🧠")
render_sidebar_branding()

st.title("🧠 Feature Importance & Explainability")
st.markdown(
    "Which factors influence predicted accident severity most, according to the "
    "production model — reused from `08_Final_Model_Validation_and_Explainability.ipynb`."
)

# ---- Built-in vs. Permutation Importance, side by side ----
st.markdown("## Built-in vs. Permutation Importance")
st.markdown(
    """
    - **Built-in importance** reflects how much each feature reduced impurity
      across the Gradient Boosting ensemble during training — fast to compute,
      but known to favor high-cardinality/continuous features.
    - **Permutation importance** measures how much test-set performance
      actually degrades when a feature's values are randomly shuffled — a more
      trustworthy, model-agnostic signal of real predictive contribution.
    """
)

importance_col1, importance_col2 = st.columns(2)

with importance_col1:
    built_in_figure = get_final_validation_figure("feature_importance_built_in.png")
    if built_in_figure is not None:
        st.image(str(built_in_figure), caption="Top 20 Features — Built-in Importance", use_container_width=True)
    else:
        render_missing_artifact_notice(
            "Built-in feature importance", "08_Final_Model_Validation_and_Explainability.ipynb"
        )

with importance_col2:
    permutation_figure = get_final_validation_figure("feature_importance_permutation.png")
    if permutation_figure is not None:
        st.image(str(permutation_figure), caption="Top 20 Features — Permutation Importance", use_container_width=True)
    else:
        render_missing_artifact_notice(
            "Permutation feature importance", "08_Final_Model_Validation_and_Explainability.ipynb"
        )

# ---- SHAP Summary Plots (if generated) ----
st.markdown("## SHAP Explainability")
shap_figures = [f for f in list_figures(FINAL_VALIDATION_FIGURES_DIR) if f.stem.startswith("shap_summary")]

if shap_figures:
    st.caption(
        "SHAP summary plots show not just which features matter, but how each "
        "feature's value pushes predictions toward or away from a given severity class."
    )
    tabs = st.tabs([f.stem.replace("shap_summary_", "").replace("_", " ").title() or "Overall" for f in shap_figures])
    for tab, figure_path in zip(tabs, shap_figures):
        with tab:
            st.image(str(figure_path), use_container_width=True)
else:
    st.info(
        "ℹ️ SHAP summary plots were not generated in this run — the final "
        "validation notebook automatically falls back to permutation importance "
        "(above) whenever SHAP is unavailable or unsupported for this model, "
        "so explainability is still available even without SHAP figures."
    )

# ---- Discussion ----
st.markdown("## Which Factors Influence Severity Most?")
st.markdown(
    """
    Across this project's feature importance and EDA analysis (Phases 4, 9),
    a few factors consistently stand out:

    - **Speed limit / high-speed roads** — higher speed limits correlate with
      more severe outcomes, consistent with collision energy scaling with speed.
    - **Urban vs. rural area** — rural accidents, despite being less frequent
      overall, tend to show a disproportionately higher share of severe outcomes
      (higher speeds, longer emergency response times).
    - **Light conditions** — unlit darkness carries materially different risk
      than daylight or artificially lit darkness.
    - **Vehicle type** — motorcyclists and cyclists face substantially higher
      injury severity risk than car/van occupants in a comparable collision.
    - **Road category and junction presence** — road type and junction
      configuration shape collision dynamics and resulting severity.

    See the Model Performance page for the metrics these features ultimately
    support, and the notebooks under `notebooks/03_Exploratory_Data_Analysis.ipynb`
    and `notebooks/08_Final_Model_Validation_and_Explainability.ipynb` for the
    full, detailed analysis behind these findings.
    """
)
