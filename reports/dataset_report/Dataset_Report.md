# Dataset Report

## Road Accident Severity Prediction using Machine Learning

---

### 1. Dataset Name

**UK Road Safety Dataset** (Road Accidents and Vehicles Data)

---

### 2. Dataset Source

The dataset originates from the **UK Department for Transport (DfT)**, which publishes
detailed road safety statistics annually, covering reported personal injury road traffic
accidents across Great Britain (STATS19 data collection framework). Public mirrors of this
dataset are also commonly hosted on platforms such as Kaggle for research and educational
use.

> **Note:** The exact provenance (original source URL / Kaggle dataset link / version year)
> should be documented here once confirmed, along with citation details for academic
> reporting.

---

### 3. Dataset Description

The dataset captures granular, record-level information about road traffic accidents
reported in the UK, along with details of the vehicles involved in each accident. It spans
multiple dimensions including temporal (date/time), spatial (location coordinates, road
type), environmental (weather, light, road surface conditions), and situational (number of
casualties, vehicles involved, severity) attributes.

This makes it a rich, multi-faceted dataset suitable for classification-based severity
prediction research.

---

### 4. Available Files

| File Name                   | Description                                                                                   |
|-------------------------------|-------------------------------------------------------------------------------------------------|
| `Accident_Information.csv`  | Accident-level records: date, time, location, weather conditions, road surface conditions, light conditions, number of casualties, number of vehicles, and **accident severity**. |
| `Vehicle_Information.csv`   | Vehicle-level records: vehicle type, manoeuvre performed, point of impact, driver age band, sex of driver, engine capacity, and vehicle propulsion code, each linked to a specific accident. |

---

### 5. Purpose of Each File

- **`Accident_Information.csv`**
  Serves as the **primary/master table** describing the circumstances surrounding each
  reported accident. This file is expected to contain the **target variable** (accident
  severity) and most of the environmental/situational predictors.

- **`Vehicle_Information.csv`**
  Provides **supplementary, vehicle-level context** for each accident. Since a single
  accident can involve multiple vehicles, this file is expected to have a
  **one-to-many relationship** with `Accident_Information.csv`. It will be used in future
  phases to engineer aggregated, accident-level features (e.g., number of vehicle types
  involved, average driver age, presence of specific vehicle categories).

---

### 6. Expected Target Variable

The expected target (label) variable for this research is:

> **`Accident_Severity`**

This is anticipated to be a categorical variable with levels such as:
- `Slight`
- `Serious`
- `Fatal`

> **Note:** The exact column name and category labels will be confirmed during the data
> verification and exploratory analysis stages, as naming conventions may vary slightly
> across dataset versions (e.g., `Accident_Severity` vs. `accident_severity`).

---

### 7. Expected Merge Key

The two files are expected to be joined using a common identifier:

> **`Accident_Index`**

This key uniquely identifies each accident record in `Accident_Information.csv` and is
expected to repeat across multiple rows in `Vehicle_Information.csv` (once per vehicle
involved in that accident).

> **Note:** The exact merge key column name and data type consistency across both files
> will be validated prior to any merging activity in a future phase.

---

### 8. Research Objective

The overarching research objective is to develop a **multi-class classification model**
capable of accurately predicting road accident severity based on a combination of
environmental, situational, and vehicular attributes. The research aims to:

- Identify the most influential factors contributing to accident severity.
- Address challenges such as class imbalance (fatal accidents are typically rare relative
  to slight accidents).
- Provide interpretable, actionable insights that could inform road safety policy and
  infrastructure planning.

---

### 9. Future Workflow

The dataset will be processed and analyzed through the following planned stages:

1. **Dataset Verification** *(Current Phase)* — Confirm file existence, size, and shape.
2. **Exploratory Data Analysis (EDA)** — Understand distributions, missing values,
   correlations, and class imbalance in the target variable.
3. **Data Cleaning & Preprocessing** — Handle missing values, standardize categorical
   labels, and resolve encoding inconsistencies.
4. **Dataset Merging** — Join `Accident_Information.csv` and `Vehicle_Information.csv` on
   the `Accident_Index` key, aggregating vehicle-level data to the accident level as
   needed.
5. **Feature Engineering** — Derive new features (e.g., time-of-day buckets, weekend
   flags, vehicle count per accident) and encode categorical variables.
6. **Model Development** — Train and compare baseline and advanced ML models (Logistic
   Regression, Random Forest, XGBoost, LightGBM).
7. **Model Evaluation & Interpretability** — Assess performance using appropriate
   multi-class metrics (e.g., macro F1-score, confusion matrix) and apply SHAP for
   interpretability.
8. **Deployment** — Build a Streamlit-based interactive demo for real-time severity
   prediction.

---

**Document Status:** Phase 1 — Dataset Verification Stage. This report will be updated
with confirmed column names, data types, and summary statistics in subsequent phases.
