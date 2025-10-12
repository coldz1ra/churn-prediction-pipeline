from __future__ import annotations
import argparse
import yaml
import pandas as pd
import numpy as np
import json
import joblib
from src.features import basic_features

def align_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return X.reindex(columns=cols, fill_value=0)

def main(input_path: str, output_path: str, cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = joblib.load("artifacts/model.joblib")
    cols = json.load(open("artifacts/columns.json"))["columns"]

    scoring = pd.read_csv(input_path)
    id_col = cfg["data"]["id_col"]
    X = basic_features(scoring.drop(columns=[id_col], errors="ignore"))
    X = align_columns(X, cols)

    proba = model.predict_proba(X)[:, 1]
    out = scoring.copy()
    out["churn_proba"] = proba
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.input, args.output, args.config)
