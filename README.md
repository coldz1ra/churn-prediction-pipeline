# 🧠 Churn Prediction Pipeline

End-to-end machine learning pipeline for **customer churn prediction**, built with a focus on **production readiness**, **reproducibility**, and **API deployment**.

[**Live Showcase (GitHub Pages)**](https://coldz1ra.github.io/churn-prediction-pipeline/)

---

## 🚀 Project Overview

This project simulates a real-world DS workflow — from raw data to serving model predictions through a FastAPI microservice:

1. Data ingestion
2. Preprocessing via ColumnTransformer
3. Model training (LightGBM) with probability calibration
4. Evaluation with business-oriented metrics
5. Reporting (Markdown + plots)
6. Deployment via FastAPI `/predict`
7. Publishing results via GitHub Pages

---

## 🧩 Architecture

\`\`\`
churn-prediction-pipeline/
├── src/
│   ├── app.py            # FastAPI app for serving predictions
│   ├── train.py          # Model training script
│   ├── evaluate.py       # Validation metrics computation
│   ├── inference.py      # Batch inference script
│   ├── preprocess.py     # ColumnTransformer feature pipeline
│   └── report_gen.py     # Markdown report generator
├── configs/
│   └── telco.yaml        # Main training configuration
├── scripts/
│   ├── get_telco.py      # Dataset downloader
│   ├── api_demo.sh       # Example API request
│   └── update_docs.py    # Sync reports → docs for Pages
├── artifacts/            # Trained model
├── reports/              # Validation results and figures
├── docs/                 # Published on GitHub Pages
├── Makefile
├── requirements.txt
└── README.md
\`\`\`

---

## 📊 Dataset

Source: Telco Customer Churn (IBM/Kaggle)  
Rows: ~7k  
Features: mixed categorical + numerical  
Target: \`Churn\` (binary)

---

## ⚙️ ML Pipeline

| Stage | Description |
|------|-------------|
| Preprocessing | OneHotEncoder for categoricals, SimpleImputer, StandardScaler for numericals |
| Model | LightGBMClassifier wrapped with CalibratedClassifierCV |
| Evaluation | ROC-AUC, PR-AUC, LogLoss, Brier, Lift@K; exports top-K contacts |
| Reporting | Markdown report with ROC/PR/Lift/Calibration/Profit plots |
| Serving | FastAPI \`/predict\` returns calibrated churn probabilities |

---

## 🧪 API Usage

Start API:
\`\`\`bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

Example request:
\`\`\`bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "data":[
          {
            "gender":"Male",
            "SeniorCitizen":0,
            "Partner":"Yes",
            "Dependents":"No",
            "tenure":5,
            "PhoneService":"Yes",
            "InternetService":"Fiber optic",
            "Contract":"Month-to-month",
            "PaperlessBilling":"Yes",
            "PaymentMethod":"Electronic check",
            "MonthlyCharges":79.35,
            "TotalCharges":356.65
          }
        ]
      }'
\`\`\`

---

## 📈 Model Performance

Typical validation (Telco):
- ROC-AUC ≈ 0.82
- PR-AUC ≈ 0.60
- Lift@10% ≈ 2.6

Plots:
- ROC Curve
- PR Curve
- Lift Curve
- Calibration Curve
- Profit Curve

Full report is mirrored to GitHub Pages.

---

## 🧰 Tech Stack

- ML: LightGBM, scikit-learn, pandas, numpy
- API: FastAPI, Uvicorn
- Automation: Makefile, pre-commit, pytest
- CI/CD: GitHub Actions, GitHub Pages
- Viz: matplotlib

---

## 🧑‍💻 Local Development

\`\`\`bash
git clone https://github.com/coldz1ra/churn-prediction-pipeline.git
cd churn-prediction-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data_telco
uvicorn src.app:app --host 0.0.0.0 --port 8000
bash scripts/api_demo.sh
\`\`\`

---

## 📦 Future Improvements

- Docker image and compose
- Scheduled retraining via Actions
- SHAP explainability in the report
- EDA notebook
- Extended API tests and schema

---

## 🏁 Author

[@coldz1ra](https://github.com/coldz1ra) — Data Science & MLOps
