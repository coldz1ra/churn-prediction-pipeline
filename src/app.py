from __future__ import annotations

import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import basic_features

app = FastAPI(title="Churn Prediction API")

# Load artifacts at startup
model = joblib.load("artifacts/model.joblib")
with open("artifacts/columns.json", encoding="utf-8") as f:
    cols = json.load(f)["columns"]


class Batch(BaseModel):
    data: list


def align_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return X.reindex(columns=cols, fill_value=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(batch: Batch):
    df = pd.DataFrame(batch.data)
    X = basic_features(df)
    X = align_columns(X, cols)
    proba = model.predict_proba(X)[:, 1]
    return {"churn_proba": proba.tolist()}
