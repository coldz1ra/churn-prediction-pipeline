from __future__ import annotations

import traceback
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

# Загружаем пайплайн-модель (внутри уже ColumnTransformer + estimator)
model = joblib.load("artifacts/model.joblib")


class Batch(BaseModel):
    data: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(batch: Batch):
    # 1) валидация входа
    if not batch.data:
        raise HTTPException(status_code=400, detail="Empty payload: 'data' is empty")

    # 2) формируем DataFrame
    df = pd.DataFrame(batch.data)
    # normalize types: known numerics → numeric; others → string
    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if c not in num_cols:
            df[c] = df[c].astype("string")

    # 3) снесём возможный id-столбец, чтобы не мешался
    for id_col in ("customerID", "customer_id", "id"):
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    # 4) инференс с отловом исключений (в лог — стэктрейс, клиенту — 400)
    try:
        proba = model.predict_proba(df)[:, 1].tolist()
        return {"churn_proba": proba}
    except Exception as e:
        # подробный лог в консоль uvicorn
        traceback.print_exc()
        # короткое сообщение — клиенту
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}") from e
