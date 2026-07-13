"""
app/pages/3_Predict_Severity.py
==================================
Interactive prediction page. Collects road, weather, vehicle, and situational
factors via a form, runs them through `AccidentSeverityPredictor`, and
displays the predicted severity, confidence, and full probability
distribution. The production model is loaded once (cached) and is never
retrained or modified by this page.
"""

from __future__ import annotations

import sys
from datetime import date, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config.settings import (
    DAY_OF_WEEK_OPTIONS,
    FIRST_ROAD_CLASS_OPTIONS,
    JUNCTION_CONTROL_OPTIONS,
    JUNCTION_DETAIL_OPTIONS,
    LIGHT_CONDITIONS_OPTIONS,
    ROAD_SURFACE_CONDITIONS_OPTIONS,
    ROAD_TYPE_OPTIONS,
    SPEED_LIMIT_OPTIONS,
    URBAN_RURAL_OPTIONS,
    VEHICLE_TYPE_OPTIONS,
    WEATHER_CONDITIONS_OPTIONS,
)
from src.models.model_loader import ModelLoadError
from src.prediction.predictor import AccidentSeverityPredictor
from src.prediction.preprocessing import RawAccidentInput
from src.utils.logger import get_logger
from src.utils.ui_helpers import (
    configure_page,
    render_hero,
    render_metric_card,
    render_severity_badge,
    render_sidebar_branding,
)
from src.visualization.charts import build_probability_bar_chart

logger = get_logger(__name__)

configure_page(page_title="Predict Severity", page_icon="🔮")
render_sidebar_branding()
render_hero(
    "🔮 Predict Accident Severity",
    "Enter road, weather, and vehicle conditions to estimate the likely severity outcome.",
)


@st.cache_resource(show_spinner="Loading production model...")
def _get_predictor() -> AccidentSeverityPredictor:
    return AccidentSeverityPredictor()


try:
    predictor = _get_predictor()
    model_load_error: str | None = None
except ModelLoadError as error:
    predictor = None
    model_load_error = str(error)

if model_load_error:
    st.error(
        f"⚠️ The production model could not be loaded.\n\n{model_load_error}"
    )
    st.stop()

st.markdown("### Enter Accident Conditions")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Environment**")
        weather_conditions = st.selectbox("Weather Conditions", WEATHER_CONDITIONS_OPTIONS)
        light_conditions = st.selectbox("Light Conditions", LIGHT_CONDITIONS_OPTIONS)
        road_surface_conditions = st.selectbox("Road Surface Conditions", ROAD_SURFACE_CONDITIONS_OPTIONS)
        urban_or_rural_area = st.selectbox("Urban / Rural Area", URBAN_RURAL_OPTIONS)

    with col2:
        st.markdown("**Road**")
        road_type = st.selectbox("Road Type", ROAD_TYPE_OPTIONS)
        first_road_class = st.selectbox("Road Class", FIRST_ROAD_CLASS_OPTIONS)
        speed_limit = st.select_slider("Speed Limit (mph)", options=SPEED_LIMIT_OPTIONS, value=30)
        junction_detail = st.selectbox("Junction Detail", JUNCTION_DETAIL_OPTIONS)
        junction_control = st.selectbox("Junction Control", JUNCTION_CONTROL_OPTIONS)

    with col3:
        st.markdown("**Vehicle & Situation**")
        vehicle_type = st.selectbox("Vehicle Type", VEHICLE_TYPE_OPTIONS)
        vehicle_age = st.slider("Vehicle Age (years)", min_value=0, max_value=25, value=5)
        number_of_vehicles = st.slider("Number of Vehicles Involved", min_value=1, max_value=8, value=2)
        day_of_week = st.selectbox("Day of Week", DAY_OF_WEEK_OPTIONS)

    st.markdown("**Date & Time (optional — affects seasonal/time-of-day features)**")
    date_col, time_col = st.columns(2)
    with date_col:
        accident_date = st.date_input("Date", value=date.today())
    with time_col:
        accident_time = st.time_input("Time", value=time(17, 0))

    with st.expander("Advanced (optional) — driver & vehicle detail"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            engine_capacity_cc = st.number_input("Engine Capacity (cc)", min_value=50, max_value=6000, value=1600, step=50)
        with adv_col2:
            driver_imd_decile = st.slider("Driver Home Area IMD Decile", min_value=1, max_value=10, value=5)
        with adv_col3:
            sex_of_driver = st.selectbox("Sex of Driver", ["Male", "Female", "Not known"])

    submitted = st.form_submit_button("🔮 Predict Severity", use_container_width=True)

if submitted:
    raw_input = RawAccidentInput(
        weather_conditions=weather_conditions,
        road_surface_conditions=road_surface_conditions,
        light_conditions=light_conditions,
        road_type=road_type,
        first_road_class=first_road_class,
        junction_detail=junction_detail,
        junction_control=junction_control,
        urban_or_rural_area=urban_or_rural_area,
        vehicle_type=vehicle_type,
        speed_limit=speed_limit,
        vehicle_age=float(vehicle_age),
        number_of_vehicles=number_of_vehicles,
        day_of_week=day_of_week,
        accident_date=accident_date,
        accident_time=accident_time,
        engine_capacity_cc=float(engine_capacity_cc),
        driver_imd_decile=float(driver_imd_decile),
        sex_of_driver=sex_of_driver,
    )

    try:
        with st.spinner("Running prediction..."):
            result = predictor.predict(raw_input)

        st.markdown("---")
        st.markdown("## Prediction Result")

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.markdown("#### Predicted Severity")
            st.markdown(render_severity_badge(result.predicted_class), unsafe_allow_html=True)
            st.write("")
            render_metric_card("Confidence", f"{result.confidence:.1%}")

        with result_col2:
            st.plotly_chart(build_probability_bar_chart(result.probabilities), use_container_width=True)

        if result.predicted_class == "Fatal":
            st.error(
                "⚠️ This scenario is predicted as **Fatal** severity. This tool is a "
                "research/decision-support aid, not a substitute for professional judgment."
            )
        elif result.predicted_class == "Serious":
            st.warning("⚠️ This scenario is predicted as **Serious** severity.")
        else:
            st.success("✅ This scenario is predicted as **Slight** severity.")

        st.caption(
            "Prediction generated by the production Gradient Boosting model "
            "(`models/trained/best_gradient_boosting.pkl`), loaded read-only — "
            "this app never retrains or modifies the model."
        )

    except Exception as error:  # noqa: BLE001 — surface a friendly message, log the detail
        logger.exception("Prediction failed")
        st.error(
            f"⚠️ Something went wrong while generating this prediction: {error}\n\n"
            f"Please check that all inputs are valid and try again."
        )
