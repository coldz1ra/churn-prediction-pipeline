# Customer Churn — Report

_Generated: 2025-10-15T14:27:12.455926Z_

## Overview
- Calibrated probabilities, lift@k and profit simulation on the Telco dataset.
- Models: Logistic Regression / LightGBM / XGBoost.

## Metrics
- **roc_auc**: 0.8213503319641428
- **pr_auc**: 0.5956216406254929
- **logloss**: 0.43637511958934777
- **brier**: 0.14391501148876382
- **lift@k**: 2.6640756302420643
- **threshold@k**: 0.5211267605633801

## Business Insights
- Top-10% contact list size: **140**.
- Baseline churn rate: **0.265**; Top-k hit rate: **0.707**.
- Lift@k: **2.66×**.
- Expected saves@k: **99.8**; Cost@k (CPS=2.0): **280.00**.
- LTV: **100**; Expected profit proxy@k: **9697.46**.

## Figures
![roc_curve.png](reports/figures/roc_curve.png)
![pr_curve.png](reports/figures/pr_curve.png)
![calibration_curve.png](reports/figures/calibration_curve.png)
![lift_curve.png](reports/figures/lift_curve.png)
![profit_curve.png](reports/figures/profit_curve.png)