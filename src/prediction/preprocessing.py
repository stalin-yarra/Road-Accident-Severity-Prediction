"""
src/prediction/preprocessing.py
=================================
Transforms raw, human-friendly input (as collected on the Streamlit
"Predict Severity" page) into the exact, ordered feature vector the
production model was trained on.

This module intentionally mirrors — rather than imports — the small,
pure categorization functions from `04_Feature_Engineering.ipynb` and the
encoding convention from `05_Feature_Selection_and_Data_Preparation.ipynb`.
Notebooks in this project are frozen, standalone artifacts (per project
convention, they are never imported from application code), so the
category-grouping logic is deliberately reproduced here, verbatim in
behavior, to avoid training-serving skew. If that engineering logic is ever
revised in the notebooks, this module must be updated to match.

Known limitation
-----------------
Phase 6 (`05_Feature_Selection_and_Data_Preparation.ipynb`) did not persist
its fitted `OrdinalEncoder` objects to disk. For any categorical column that
ended up **label/ordinal-encoded** rather than one-hot-encoded (only
possible for a column with more than 15 unique categories at training
time), this module cannot reconstruct the exact category → integer mapping
at inference time, and instead maps it to `-1` (scikit-learn's
`unknown_value` convention for previously-unseen categories), with a
logged warning. In practice, every categorical field exposed on the
prediction form was low-cardinality and therefore one-hot-encoded, so this
limitation is not expected to affect the columns collected here — it is
documented for transparency and as a flagged future improvement (see
README "Future Improvements").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time as dt_time
from typing import Optional

import numpy as np
import pandas as pd

from src.config.settings import X_TRAIN_PATH
from src.utils.io_helpers import load_csv_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# CATEGORIZATION FUNCTIONS
# (mirrors 04_Feature_Engineering.ipynb — see module docstring above)
# =============================================================================

def categorize_speed_limit(speed_limit: float) -> str:
    """Map a numeric speed limit (mph) to a readable speed category band."""
    if speed_limit is None or pd.isnull(speed_limit):
        return "Unknown"
    if speed_limit <= 30:
        return "20-30"
    if speed_limit <= 50:
        return "40-50"
    return "60+"


def categorize_road(road_type: Optional[str], road_class: Optional[str] = None) -> str:
    """Map raw road type (and optionally road class) to a readable Road_Category."""
    if not road_type:
        return "Unknown"
    text = str(road_type).lower()
    if "roundabout" in text:
        return "Roundabout"
    if "one way" in text or "slip road" in text:
        return "Minor Road"
    if "dual carriageway" in text:
        return "Major Road"
    if "single carriageway" in text:
        if road_class and str(road_class).upper() in {"A", "B", "A(M)", "MOTORWAY"}:
            return "Major Road"
        return "Minor Road" if road_class else "Major Road"
    return "Other"


def categorize_junction(junction_detail: Optional[str]) -> str:
    """Map raw Junction_Detail values to a simple readable junction indicator."""
    if not junction_detail:
        return "Unknown"
    text = str(junction_detail).lower()
    if "not at junction" in text:
        return "Not at Junction"
    return "At Junction"


def categorize_weather(weather_condition: Optional[str]) -> str:
    """Map raw Weather_Conditions text to Snow > Fog > Rain > Wind > Clear > Unknown."""
    if not weather_condition:
        return "Unknown"
    text = str(weather_condition).lower()
    if "snow" in text:
        return "Snow"
    if "fog" in text or "mist" in text:
        return "Fog"
    if "rain" in text:
        return "Rain"
    if "high winds" in text and "no high winds" not in text:
        return "Wind"
    if "fine" in text:
        return "Clear"
    return "Unknown"


def categorize_light(light_condition: Optional[str]) -> str:
    """Map raw Light_Conditions text to Day / Artificial Lighting / Night / Unknown."""
    if not light_condition:
        return "Unknown"
    text = str(light_condition).lower()
    if "daylight" in text:
        return "Day"
    if "darkness" in text:
        if "lights lit" in text or ("lit" in text and "unlit" not in text and "no lighting" not in text):
            return "Artificial Lighting"
        return "Night"
    return "Unknown"


def categorize_urban_rural(value: Optional[str]) -> str:
    """Standardize an Urban/Rural value (text or legacy numeric code) to a clean label."""
    if not value:
        return "Unknown"
    text = str(value).strip().lower()
    if text in {"1", "urban"}:
        return "Urban"
    if text in {"2", "rural"}:
        return "Rural"
    if text in {"3", "unallocated"}:
        return "Unallocated"
    return "Unknown"


_VEHICLE_TYPE_GROUP_MAP_KEYWORDS = [
    ("motorcycle", "Motorcycle"),
    ("moped", "Motorcycle"),
    ("scooter", "Motorcycle"),
    ("pedal cycle", "Cycle"),
    ("bicycle", "Cycle"),
    ("bus", "Heavy Vehicle"),
    ("coach", "Heavy Vehicle"),
    ("goods", "Heavy Vehicle"),
    ("lorry", "Heavy Vehicle"),
    ("van", "Van"),
    ("car", "Car"),
    ("taxi", "Car"),
]


def categorize_vehicle_type(vehicle_type: Optional[str]) -> str:
    """Map a raw Vehicle_Type value to a broad, readable Vehicle_Type_Group."""
    if not vehicle_type:
        return "Unknown"
    text = str(vehicle_type).lower()
    for keyword, group in _VEHICLE_TYPE_GROUP_MAP_KEYWORDS:
        if keyword in text:
            return group
    return "Other"


def categorize_engine_capacity(engine_cc: Optional[float]) -> str:
    """Bin engine capacity (CC) into a readable Vehicle_Class label."""
    if engine_cc is None or pd.isnull(engine_cc):
        return "Unknown"
    if engine_cc < 1200:
        return "Small"
    if engine_cc <= 2000:
        return "Medium"
    return "Large"


def categorize_time_of_day(hour: Optional[int]) -> str:
    """Map an hour of day (0-23) to a readable time-of-day category."""
    if hour is None:
        return "Unknown"
    if 5 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 16:
        return "Afternoon"
    if 17 <= hour <= 20:
        return "Evening"
    return "Night"


# =============================================================================
# RAW INPUT CONTAINER
# =============================================================================

@dataclass
class RawAccidentInput:
    """
    Raw, human-friendly values collected from the "Predict Severity" form.
    Every field mirrors a raw STATS19-style category or a simple numeric
    quantity — none of these are yet in the model's encoded feature space.
    """

    weather_conditions: str
    road_surface_conditions: str
    light_conditions: str
    road_type: str
    first_road_class: str
    junction_detail: str
    junction_control: str
    urban_or_rural_area: str
    vehicle_type: str
    speed_limit: int
    vehicle_age: float
    number_of_vehicles: int
    day_of_week: str
    accident_date: date = field(default_factory=date.today)
    accident_time: dt_time = field(default_factory=lambda: dt_time(12, 0))
    engine_capacity_cc: float = 1600.0
    driver_imd_decile: float = 5.0
    sex_of_driver: str = "Male"
    age_band_of_driver: str = "26 - 35"
    propulsion_code: str = "Petrol"


# =============================================================================
# FEATURE ENGINEERING (raw input -> the same derived columns Phase 5 created)
# =============================================================================

def engineer_features(raw_input: RawAccidentInput) -> dict:
    """
    Apply the same derivations as `04_Feature_Engineering.ipynb` to a single
    raw input record, producing every engineered feature the model may
    expect, plus the raw passthrough columns Phase 6 retained unchanged.

    Args:
        raw_input: The raw, human-friendly form input.

    Returns:
        dict: Feature name -> value, spanning both raw passthrough columns
              and newly engineered columns (pre-encoding).
    """
    hour = raw_input.accident_time.hour
    is_weekend = 1 if raw_input.day_of_week in {"Saturday", "Sunday"} else 0
    time_of_day = categorize_time_of_day(hour)

    weather_group = categorize_weather(raw_input.weather_conditions)
    light_group = categorize_light(raw_input.light_conditions)
    road_category = categorize_road(raw_input.road_type, raw_input.first_road_class)
    junction_group = categorize_junction(raw_input.junction_detail)
    urban_rural_group = categorize_urban_rural(raw_input.urban_or_rural_area)
    speed_category = categorize_speed_limit(raw_input.speed_limit)
    vehicle_type_group = categorize_vehicle_type(raw_input.vehicle_type)
    vehicle_class = categorize_engine_capacity(raw_input.engine_capacity_cc)

    features = {
        # ---- Raw passthrough columns (retained as-is by Phase 6) ----
        "Road_Surface_Conditions": raw_input.road_surface_conditions,
        "Junction_Control": raw_input.junction_control,
        "1st_Road_Class": raw_input.first_road_class,
        "Number_of_Vehicles": raw_input.number_of_vehicles,
        "Speed_limit": raw_input.speed_limit,
        "Driver_IMD_Decile": raw_input.driver_imd_decile,
        "Sex_of_Driver": raw_input.sex_of_driver,
        "Age_Band_of_Driver": raw_input.age_band_of_driver,
        "Propulsion_Code": raw_input.propulsion_code,
        "Day_of_Week": raw_input.day_of_week,
        "Year": raw_input.accident_date.year,

        # ---- Date/time engineered features ----
        "Month": raw_input.accident_date.strftime("%B"),
        "Day": raw_input.accident_date.day,
        "Quarter": (raw_input.accident_date.month - 1) // 3 + 1,
        "Is_Weekend": is_weekend,
        "Hour": hour,
        "Minute": raw_input.accident_time.minute,
        "Time_of_Day": time_of_day,

        # ---- Grouped/categorized engineered features ----
        "Speed_Category": speed_category,
        "Road_Category": road_category,
        "Junction_Group": junction_group,
        "Weather_Group": weather_group,
        "Light_Group": light_group,
        "Urban_Rural_Group": urban_rural_group,
        "Vehicle_Age": raw_input.vehicle_age,
        "Vehicle_Type_Group": vehicle_type_group,
        "Vehicle_Class": vehicle_class,

        # ---- Binary indicators ----
        "Is_Night": 1 if time_of_day == "Night" else 0,
        "High_Speed_Road": 1 if raw_input.speed_limit >= 60 else 0,
        "Urban_Area": 1 if urban_rural_group == "Urban" else 0,
    }

    return features


# =============================================================================
# REFERENCE SCHEMA RESOLUTION
# =============================================================================

def resolve_reference_columns(model) -> list[str]:
    """
    Determine the exact, ordered list of columns the model expects, trying
    the model's own `feature_names_in_` attribute first, and falling back
    to the saved `X_train.csv` header if that attribute is unavailable.

    Args:
        model: The loaded production model.

    Returns:
        list[str]: The ordered reference feature column names.

    Raises:
        RuntimeError: If neither source is available — prediction cannot
            proceed without knowing the expected feature schema.
    """
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    logger.warning(
        "Model has no 'feature_names_in_'; falling back to the X_train.csv "
        "header as the reference feature schema."
    )
    x_train_sample = load_csv_safe(X_TRAIN_PATH, nrows=0)
    if x_train_sample is not None:
        return list(x_train_sample.columns)

    raise RuntimeError(
        "Cannot determine the model's expected feature schema: the model has "
        "no 'feature_names_in_' attribute, and "
        f"'{X_TRAIN_PATH}' is unavailable. Prediction cannot proceed."
    )


# =============================================================================
# ENCODING (raw + engineered features -> the model's exact feature vector)
# =============================================================================

def build_feature_vector(raw_input: RawAccidentInput, model) -> pd.DataFrame:
    """
    Build a single-row, fully-encoded feature DataFrame matching the
    production model's exact expected columns and order.

    This applies the same cardinality-based encoding convention as
    `05_Feature_Selection_and_Data_Preparation.ipynb`: categorical columns
    become one-hot `{column}_{value}` indicator columns (the value simply
    matched against whichever dummy columns exist in the reference schema —
    if the user's selected category was the notebook's dropped first/
    reference category, all its dummies are correctly left at 0). Any
    reference column this function cannot populate (e.g., a rare
    label-encoded high-cardinality column — see module docstring) is left
    at a safe default of 0.

    Args:
        raw_input: The raw, human-friendly form input.
        model: The loaded production model (used to resolve the expected
            feature schema).

    Returns:
        pd.DataFrame: A single-row DataFrame with columns in the exact order
            the model expects, ready to pass to `model.predict()`.
    """
    reference_columns = resolve_reference_columns(model)
    engineered = engineer_features(raw_input)

    row = {column: 0 for column in reference_columns}
    unmatched_features = []
    baseline_category_features = []

    for feature_name, value in engineered.items():
        if isinstance(value, str):
            one_hot_column = f"{feature_name}_{value}"
            if one_hot_column in row:
                row[one_hot_column] = 1
            elif feature_name in row:
                # Rare case: this categorical column was label/ordinal-encoded
                # rather than one-hot (see module docstring's known limitation).
                row[feature_name] = -1
                logger.warning(
                    "'%s' appears to be label-encoded; exact category mapping "
                    "is unavailable at inference time, using -1 (unknown).",
                    feature_name,
                )
            elif any(col.startswith(f"{feature_name}_") for col in row):
                # This feature WAS one-hot encoded during training, but the
                # user's selected value matches the notebook's dropped-first
                # reference category — correctly represented by leaving every
                # one of its dummy columns at 0. Not an error or an omission.
                baseline_category_features.append(feature_name)
            else:
                # No column or column-prefix for this feature exists at all in
                # the reference schema — it was genuinely not part of the
                # final training feature set (e.g., dropped in Phase 6).
                unmatched_features.append(feature_name)
        else:
            if feature_name in row:
                row[feature_name] = value
            else:
                unmatched_features.append(feature_name)

    if baseline_category_features:
        logger.debug(
            "%d feature(s) matched the trained model's dropped-first baseline "
            "category (correctly encoded as all-zero dummies): %s",
            len(baseline_category_features), baseline_category_features,
        )
    if unmatched_features:
        logger.info(
            "%d engineered feature(s) had no corresponding column in the "
            "model's schema (likely dropped during feature selection): %s",
            len(unmatched_features), unmatched_features,
        )

    feature_df = pd.DataFrame([row], columns=reference_columns)
    return feature_df
