# Customer Churn — Report

_Generated: 2025-10-12T21:10:52.332700Z_

## Overview
- Calibrated probabilities, lift@k and profit simulation on the Telco dataset.
- Models: Logistic Regression / LightGBM / XGBoost.

## Metrics
- **roc_auc**: 0.8247074323800666
- **pr_auc**: 0.6110669525985462
- **logloss**: 0.43213063382940986
- **brier**: 0.1423553191251967
- **lift@k**: 2.7717150496457843
- **threshold@k**: 0.6

## Business Insights
- Top-10% contact list size: **140**.
- Baseline churn rate: **0.265**; Top-k hit rate: **0.736**.
- Lift@k: **2.77×**.
- Expected saves@k: **102.4**; Cost@k (CPS=2.0): **280.00**.
- LTV: **100**; Expected profit proxy@k: **9960.00**.

## Figures
![roc_curve.png](reports/figures/roc_curve.png)
![pr_curve.png](reports/figures/pr_curve.png)
![calibration_curve.png](reports/figures/calibration_curve.png)
![lift_curve.png](reports/figures/lift_curve.png)
![profit_curve.png](reports/figures/profit_curve.png)