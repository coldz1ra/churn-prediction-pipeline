#!/usr/bin/env python3
import os, sys, pandas as pd
RAW_URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
def main(out_path="data/train.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Downloading {RAW_URL}")
    df = pd.read_csv(RAW_URL)
    df["target_churn"] = (df["Churn"].astype(str).str.strip().str.lower()=="yes").astype(int)
    if "customerID" not in df.columns:
        df["customerID"] = [f"c{i}" for i in range(len(df))]
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv)>1 else "data/train.csv"
    main(out)
