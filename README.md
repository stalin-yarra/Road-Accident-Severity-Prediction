# Road Accident Severity Prediction using Machine Learning

## Table of Contents
- [Project Description](#project-description)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Folder Structure](#folder-structure)
- [Technology Stack](#technology-stack)
- [Installation Instructions](#installation-instructions)
- [Future Development Phases](#future-development-phases)
- [License](#license)

---

## Project Description

This repository contains a production-quality research project developed as part of an
**On-Campus Research Internship at IIIT Vadodara**. The project aims to build a robust,
interpretable, and reproducible Machine Learning pipeline to predict the **severity of road
accidents** using the **UK Road Safety Dataset**.

The project follows a phased development approach, starting with proper project
initialization and progressively moving through data understanding, preprocessing,
feature engineering, model development, evaluation, and deployment.

This document reflects **Phase 1: Project Initialization** only. No preprocessing,
exploratory data analysis (EDA), feature engineering, or model building has been performed
at this stage.

---

## Problem Statement

Road traffic accidents are a major global public safety concern, resulting in significant
loss of life, injury, and economic cost every year. Understanding the factors that
contribute to the **severity** of an accident (e.g., Slight, Serious, or Fatal) can help
transportation authorities, urban planners, and policymakers design better interventions,
improve road infrastructure, and optimize emergency response systems.

Given historical accident and vehicle-level records, the core problem this project
addresses is:

> **Can we accurately predict the severity of a road accident using environmental,
> vehicular, and situational factors available at or near the time of the accident?**

---

## Objectives

The primary objectives of this research project are:

1. To build a clean, modular, and reproducible ML project structure suitable for
   academic research and open-source publication.
2. To thoroughly understand the UK Road Safety Dataset (`Accident_Information.csv` and
   `Vehicle_Information.csv`) through systematic verification and future exploratory
   analysis.
3. To design a robust data preprocessing and feature engineering pipeline (in later
   phases) that handles missing values, categorical encoding, and class imbalance.
4. To train, tune, and evaluate multiple Machine Learning models (e.g., Logistic
   Regression, Random Forest, XGBoost, LightGBM) for multi-class severity prediction.
5. To interpret model predictions using explainability techniques such as SHAP.
6. To build a lightweight demonstration interface (Streamlit) for showcasing model
   predictions.
7. To document the entire research process in a manner consistent with academic and
   industry best practices.

---

## Dataset Description

The project uses the **UK Road Safety Dataset**, which is a publicly available dataset
published by the UK Department for Transport, capturing detailed records of reported
road traffic accidents across the United Kingdom.

### Raw Data Files

| File Name                  | Description                                                                 |
|-----------------------------|-------------------------------------------------------------------------------|
| `Accident_Information.csv` | Contains accident-level details such as location, date, time, weather, road surface conditions, number of casualties, and accident severity. |
| `Vehicle_Information.csv`  | Contains vehicle-level details associated with each accident, such as vehicle type, manoeuvre, driver age band, sex of driver, and point of impact. |

Both files are expected to be linked using a common **Accident Index** key (exact column
name to be confirmed during the data verification stage).

> **Note:** At this stage (Phase 1), the dataset is only verified for structural
> integrity (file existence, size, shape). No cleaning, transformation, or merging is
> performed.

---

## Folder Structure

```
road-accident-severity-prediction/
├── Dataset/
│   ├── raw/                     # Original, immutable dataset files
│   │   ├── Accident_Information.csv
│   │   └── Vehicle_Information.csv
│   ├── processed/               # Cleaned/merged datasets (future phases)
│   └── external/                # Any supplementary external data (future phases)
│
├── notebooks/                   # Jupyter notebooks for EDA, prototyping (future phases)
│
├── src/                          # Source code for the ML pipeline
│   ├── data/                    # Data loading & validation scripts
│   ├── features/                # Feature engineering scripts (future phases)
│   ├── models/                  # Model training & prediction scripts (future phases)
│   ├── visualization/           # Plotting & visualization utilities (future phases)
│   └── utils/                   # Shared helper utilities
│
├── models/
│   ├── trained/                 # Serialized trained model artifacts (future phases)
│   └── evaluation/              # Model evaluation reports/metrics (future phases)
│
├── reports/
│   ├── figures/                 # Generated charts/graphs (future phases)
│   └── dataset_report/          # Dataset documentation
│       └── Dataset_Report.md
│
├── tests/                       # Unit tests (future phases)
├── config/                       # Configuration files (future phases)
├── logs/                          # Log files generated at runtime
├── docs/                          # Additional project documentation
│
├── main.py                       # Entry point to verify project initialization
├── verify_dataset.py             # Script to verify dataset integrity
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # Project documentation (this file)
└── LICENSE                       # License information
```

---

## Technology Stack

| Category                 | Tools / Libraries                                  |
|---------------------------|-----------------------------------------------------|
| Programming Language      | Python 3.10+                                       |
| Data Manipulation         | pandas, numpy                                      |
| Data Visualization        | matplotlib, seaborn, plotly                        |
| Machine Learning          | scikit-learn, xgboost, lightgbm                    |
| Scientific Computing      | scipy                                               |
| Model Interpretability    | shap                                                |
| Model Persistence         | joblib                                              |
| Interactive Prototyping   | jupyter, notebook                                  |
| Web App / Demo            | streamlit                                           |
| Version Control           | Git & GitHub                                        |

---

## Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/road-accident-severity-prediction.git
cd road-accident-severity-prediction
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

Activate the environment:

- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Place Dataset Files
Ensure the following files are placed inside `Dataset/raw/`:
```
Dataset/raw/Accident_Information.csv
Dataset/raw/Vehicle_Information.csv
```

### 5. Verify Project Initialization
```bash
python main.py
```

### 6. Verify Dataset Integrity
```bash
python verify_dataset.py
```

---

## Future Development Phases

This project will be developed in the following incremental phases:

| Phase       | Description                                                                 |
|-------------|-------------------------------------------------------------------------------|
| **Phase 1** | Project Initialization — folder structure, environment setup, dataset verification. *(Current Phase)* |
| **Phase 2** | Exploratory Data Analysis (EDA) — understanding distributions, correlations, and missing data patterns. |
| **Phase 3** | Data Preprocessing — handling missing values, encoding, merging datasets, outlier treatment. |
| **Phase 4** | Feature Engineering — creating derived features, handling class imbalance. |
| **Phase 5** | Model Development — training baseline and advanced ML models (Logistic Regression, Random Forest, XGBoost, LightGBM). |
| **Phase 6** | Model Evaluation & Interpretation — cross-validation, hyperparameter tuning, SHAP-based interpretability. |
| **Phase 7** | Deployment — building a Streamlit-based demo application for interactive predictions. |
| **Phase 8** | Documentation & Research Reporting — final research report, presentation, and publication-ready materials. |

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for
details.

---

**Maintained as part of an On-Campus Research Internship at IIIT Vadodara.**
