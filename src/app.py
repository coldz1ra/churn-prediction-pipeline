from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# но модель уже сама содержит препроцессор в Pipeline

app = FastAPI(title="Churn Prediction API")
model = joblib.load("artifacts/model.joblib")


class Batch(BaseModel):
    data: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(batch: Batch):
    df = pd.DataFrame(batch.data)
    proba = model.predict_proba(df)[:, 1]
    return {"churn_proba": proba.tolist()}
