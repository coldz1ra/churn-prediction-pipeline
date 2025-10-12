from __future__ import annotations

import argparse

import joblib
import pandas as pd
import yaml


def main(input_path: str, output_path: str, cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = joblib.load("artifacts/model.joblib")
    scoring = pd.read_csv(input_path)
    id_col = cfg["data"]["id_col"]
    X = scoring.drop(columns=[id_col], errors="ignore")
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
