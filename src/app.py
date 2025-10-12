from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib, json
from src.features import basic_features

app = FastAPI(title="Churn Prediction API")

# Load artifacts at startup
model = joblib.load("artifacts/model.joblib")
cols = json.load(open("artifacts/columns.json"))["columns"]

class Record(BaseModel):
    data: dict

class Batch(BaseModel):
    data: list

def align_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    return X[cols]

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
