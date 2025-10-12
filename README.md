![CI](https://github.com/coldz1ra/churn-prediction-pipeline/actions/workflows/ci.yml/badge.svg)
# Customer Churn Prediction

**Resume-grade** end-to-end ML pipeline to predict customer churn probability and optimize retention actions under a budget.

## Highlights
- 📦 Clean, modular structure with `src/`, `configs/`, `tests/`, `reports/`, `artifacts/`
- 🤖 Models: Logistic Regression, LightGBM, XGBoost (+ early stopping)
- 🎯 Imbalanced learning, **probability calibration**, **threshold@k**, **lift@k**, **profit curve**
- 🔍 Explainability: **SHAP** summary & top features
- 🧪 CI/CD: GitHub Actions (+ pip cache), pre-commit (black, ruff), pytest
- 📊 Auto-report: `reports/churn_report.md` with metrics, figures, business insights
- 🌐 Optional REST API (FastAPI) + Dockerfile

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
make setup
pre-commit install
make test
make train            # trains and saves artifacts
make eval             # metrics + curves
make report           # generates reports/churn_report.md
make predict          # scores data/scoring.csv -> reports/predictions.csv
```

## Configs
Use one of:
- `configs/base.yaml` (LogReg baseline)
- `configs/model_xgb.yaml` (XGBoost + isotonic calibration)
- `configs/model_lgbm.yaml` (LightGBM + isotonic calibration)

Key business params:
```yaml
thresholding:
  k_fraction: 0.10   # top-10% contact rate
  cps: 2.0           # cost per save (per contacted customer)
  ltv: 100.0         # expected lifetime value per saved customer
```

## API (optional)
After `make train` (artifacts created), run:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
# or
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
Request example:
```json
POST /predict
{
  "data": [
    {"revenue": 120, "active_days": 12, "tenure_days": 180, "contract_type": "Month-to-month"}
  ]
}
```

## Figures
- ROC, PR, Calibration, Lift, Profit; **SHAP summary** for tree models.

## Stack
Python, scikit-learn, LightGBM, XGBoost, SHAP, Imbalanced-Learn, FastAPI, GitHub Actions, pytest.
