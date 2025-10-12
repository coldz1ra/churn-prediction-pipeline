# Customer Churn — Report

_Generated: 2025-10-12T19:54:34.102652Z_

## Overview
- End-to-end pipeline with calibrated probabilities, business lift and profit simulation.
- Models: Logistic Regression / LightGBM / XGBoost (with early stopping).

## Metrics
- **roc_auc**: 0.8469864889302231
- **pr_auc**: 0.6468009846244296
- **logloss**: 0.40706100622742925
- **brier**: 0.13320191733743725
- **lift@k**: 2.9869938884532234
- **threshold@k**: 0.6794871794871795

## Business Insights
- Top-10% contact list size: **140**.
- Baseline churn rate: **0.265**; Top-k hit rate: **0.771**; Lift@k: **2.91×**.
- Expected saves@k: **108.6**; Cost@k (CPS=2.0): **280.00**.
- LTV assumption: **100.0**; Expected profit proxy@k: **10577.69**.

## Top 10 SHAP Features (mean |SHAP|)
- Contract_Two year: 0.761536
- tenure: 0.620491
- InternetService_Fiber optic: 0.362158
- Contract_One year: 0.338670
- MonthlyCharges: 0.273007
- PaymentMethod_Electronic check: 0.175740
- PaperlessBilling_Yes: 0.114602
- StreamingMovies_Yes: 0.112745
- OnlineSecurity_Yes: 0.096586
- MultipleLines_Yes: 0.088751

## Top 10 Tree Feature Importances
- MonthlyCharges: 3147.000000
- tenure: 2672.000000
- gender_Male: 319.000000
- PaymentMethod_Electronic check: 317.000000
- OnlineBackup_Yes: 287.000000
- PaperlessBilling_Yes: 280.000000
- OnlineSecurity_Yes: 257.000000
- TechSupport_Yes: 232.000000
- SeniorCitizen: 220.000000
- Dependents_Yes: 207.000000

## Figures
![roc_curve.png](reports/figures/roc_curve.png)
![pr_curve.png](reports/figures/pr_curve.png)
![calibration_curve.png](reports/figures/calibration_curve.png)
![lift_curve.png](reports/figures/lift_curve.png)
![profit_curve.png](reports/figures/profit_curve.png)