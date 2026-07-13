"""
app/pages_content/home_content.py
====================================
Shared rendering logic for the Home page — imported by both
`app/streamlit_app.py` (the bare entry point) and `app/pages/1_Home.py`, so
the project overview, objectives, architecture, and workflow content exists
in exactly one place.
"""

from __future__ import annotations

import streamlit as st

from src.utils.ui_helpers import render_hero


def render_home_page() -> None:
    """Render the complete Home page content into the current Streamlit app."""
    render_hero(
        "🚦 Road Accident Severity Prediction",
        "Machine learning system for predicting UK road accident severity — "
        "Slight, Serious, or Fatal — from situational and vehicle factors.",
    )

    st.markdown("## Project Overview")
    st.markdown(
        """
        This application is the production interface for a full machine learning
        research pipeline built on the **UK Road Safety (STATS19) Dataset** —
        over 2 million recorded road traffic accidents. The goal is to predict
        the **severity of a road accident** (`Slight`, `Serious`, or `Fatal`)
        from environmental, situational, and vehicle-level factors available
        around the time of the accident.

        The trained model, evaluation results, and explainability analysis
        shown throughout this app come directly from the project's Jupyter
        notebook pipeline — nothing here is recomputed from scratch; every
        page reuses the trained model and generated reports.
        """
    )

    st.markdown("## Objectives")
    objective_cols = st.columns(3)
    objectives = [
        ("🎯", "Predict Severity", "Classify accidents into Slight, Serious, or Fatal using situational and vehicle factors."),
        ("🔍", "Explain Predictions", "Surface which factors drive severity via feature importance and SHAP explainability."),
        ("🚀", "Support Decisions", "Provide an interactive, deployable tool suitable for research and road-safety review."),
    ]
    for col, (icon, title, description) in zip(objective_cols, objectives):
        with col:
            st.markdown(f"### {icon} {title}")
            st.markdown(description)

    st.markdown("## Machine Learning Pipeline / Workflow")
    st.markdown(
        "Each stage below corresponds to a dedicated, standalone Jupyter notebook "
        "in `notebooks/`, executed in order:"
    )

    pipeline_stages = [
        "Project\nInitialization",
        "Data\nUnderstanding",
        "Data\nPreprocessing",
        "Exploratory\nData Analysis",
        "Feature\nEngineering",
        "Feature\nSelection",
        "Model\nTraining",
        "Hyperparameter\nTuning",
        "Final Validation &\nExplainability",
    ]

    stage_cols = st.columns(len(pipeline_stages))
    for i, (col, stage) in enumerate(zip(stage_cols, pipeline_stages)):
        with col:
            st.markdown(
                f"""
                <div style="text-align:center; padding:0.5rem; border-radius:0.5rem;
                            background-color:#f0f2f6; font-size:0.8rem; font-weight:600;">
                    {i + 1}. {stage}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption("→ Data flows left to right through each notebook phase, ending in the production model used by this app.")

    st.markdown("## Technology Stack")
    tech_cols = st.columns(4)
    tech_groups = [
        ("Data & ML", "pandas, numpy, scikit-learn, XGBoost, LightGBM"),
        ("Explainability", "SHAP, permutation importance"),
        ("Visualization", "matplotlib, seaborn, Plotly"),
        ("Application", "Streamlit, joblib"),
    ]
    for col, (group, tools) in zip(tech_cols, tech_groups):
        with col:
            st.markdown(f"**{group}**")
            st.caption(tools)

    st.divider()
    st.markdown(
        "Use the sidebar to explore the dataset, make a live prediction, "
        "review model performance, and see which features drive severity most."
    )
