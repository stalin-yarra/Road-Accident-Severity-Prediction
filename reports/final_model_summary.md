# Final Model Summary — Road Accident Severity Prediction

**Generated automatically by `08_Final_Model_Validation_and_Explainability.ipynb`**

## Dataset

- **Source:** UK Road Safety Dataset (STATS19), merged accident- and vehicle-level records.
- **Target Variable:** `Accident_Severity` (Slight / Serious / Fatal).
- **Test Set Size:** 543,188 rows.

## Features Used

- **Feature Count:** 154 engineered and encoded features (see Phase 5/6 notebooks for full derivation).
- **Top 5 Features (by permutation importance):** Vehicle_Type_Group_Motorcycle, Speed_limit, X1st_Point_of_Impact_Front, Vehicle_Manoeuvre, Did_Police_Officer_Attend_Scene_of_Accident.

## Best Model

- **Model Family:** Gradient Boosting
- **Selection Basis:** best weighted F1 score among tuned Random Forest, Extra Trees, and Gradient Boosting models (Phase 8).
- **Training Time (full training set, if available):** 7205.31 seconds

## Final Test Set Performance

| Metric | Value |
|---|---|
| Accuracy | 0.8541 |
| Precision (weighted) | 0.8081 |
| Recall (weighted) | 0.8541 |
| Weighted F1 Score | 0.7940 |
| Balanced Accuracy | 0.3477 |

## Key Findings

- The model achieves a weighted F1 score of 0.7940 and balanced accuracy of 0.3477 on the held-out test set.
- Built-in and permutation feature importance agree with a Spearman rank correlation of 0.511 across their combined top features.
- The learning curve analysis shows a train-validation F1 gap of 0.0609, computed on a stratified/random sample of the training set.
- The most common misclassification type observed was **Serious → Slight**, consistent with the class-adjacency reasoning discussed in Section 6.

## Important Features

1. Vehicle_Type_Group_Motorcycle
2. Speed_limit
3. X1st_Point_of_Impact_Front
4. Vehicle_Manoeuvre
5. Did_Police_Officer_Attend_Scene_of_Accident

## Limitations

- Performance on the minority `Fatal` class is constrained by its rarity in the data, independent of model choice.
- The model can only learn from situational/environmental factors present in the dataset and cannot account for unrecorded contextual factors.
- Permutation importance and SHAP (where available) were computed on bounded samples of the test set for computational tractability, not the full test set.

## Future Improvements

- Explore targeted resampling or cost-sensitive learning strategies specifically for the `Fatal` class.
- Incorporate additional external data sources (e.g., traffic volume, weather severity indices) not present in the current STATS19 extract.
- Extend explainability analysis with SHAP interaction values (if not already available) to examine feature interaction effects, not just marginal importance.
- Re-evaluate the deployment threshold/decision policy in collaboration with domain experts, given the asymmetric real-world cost of false negatives on severe accidents.

---

*This summary was generated automatically from the notebook's computed results and should be reviewed by a domain expert before any deployment decision.*
