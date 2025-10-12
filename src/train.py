from __future__ import annotations
import argparse
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import joblib

from src.features import basic_features
from src.utils import ensure_binary_series
from src.plots import plot_roc, plot_pr, plot_lift

TREE_TYPES = {"xgb", "lgbm"}

def build_model(cfg):
    mtype = cfg["model"]["type"]
    if mtype == "logreg":
        from sklearn.linear_model import LogisticRegression
        params = cfg["model"]["params"]
        return LogisticRegression(**params)
    elif mtype == "xgb":
        from xgboost import XGBClassifier
        params = cfg["model"].get("params", {})
        return XGBClassifier(eval_metric="logloss", tree_method="hist", **params)
    elif mtype == "lgbm":
        from lightgbm import LGBMClassifier
        params = cfg["model"].get("params", {})
        return LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {mtype}")

def fit_with_optional_early_stopping(model, mtype: str, X_tr, y_tr, X_val, y_val):
    if mtype == "xgb":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=100)
    elif mtype == "lgbm":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, callbacks=[])
    else:
        model.fit(X_tr, y_tr)
    return model

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data"]["train_path"])
    y = ensure_binary_series(df[cfg["data"]["target"]])
    id_col = cfg["data"]["id_col"]
    X_raw = df.drop(columns=[cfg["data"]["target"], id_col], errors="ignore")
    X = basic_features(X_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg["data"]["valid_size"], stratify=y, random_state=cfg["data"]["random_state"]
    )

    # Balance train via ROS
    ros = RandomOverSampler(random_state=cfg["data"]["random_state"])
    X_tr, y_tr = ros.fit_resample(X_train, y_train)

    mtype = cfg["model"]["type"]
    model = build_model(cfg)

    calib = cfg.get("calibration", {}).get("method", None)

    if calib and mtype in TREE_TYPES:
        # Fit tree with early stopping, then calibrate with prefit (no refit)
        model_prefit = fit_with_optional_early_stopping(model, mtype, X_tr, y_tr, X_val, y_val)
        cal = CalibratedClassifierCV(base_estimator=model_prefit, method=calib, cv="prefit")
        cal.fit(X_val, y_val)
        model = cal
    else:
        # Fit with/without early stopping; if calib for linear, wrap CV=3
        model = fit_with_optional_early_stopping(model, mtype, X_tr, y_tr, X_val, y_val)
        if calib and mtype not in TREE_TYPES:
            model = CalibratedClassifierCV(base_estimator=model, method=calib, cv=3).fit(X_tr, y_tr)

    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    print(f"ROC-AUC: {auc:.4f}")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    out = X_val.copy()
    out["y_true"] = y_val.values
    out["y_proba"] = proba
    out.to_csv("reports/val_predictions.csv", index=False)

    # Plots
    plot_roc(y_val, proba, "reports/figures/roc_curve.png")
    plot_pr(y_val, proba, "reports/figures/pr_curve.png")
    plot_lift(y_val, proba, "reports/figures/lift_curve.png")

    # Save artifacts
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")
    json.dump({"columns": X.columns.tolist()}, open("artifacts/columns.json", "w"))

    # Export top-k recommendations
    k_frac = cfg.get("thresholding", {}).get("k_fraction", 0.1)
    n = len(out)
    k = max(1, int(n * k_frac))
    recs = out.sort_values("y_proba", ascending=False).head(k).copy()
    recs.to_csv("reports/topk_contacts.csv", index=False)

    # Save quick metrics json for report
    metrics = {"roc_auc": float(auc), "n_val": int(len(y_val)), "k_fraction": float(k_frac)}
    json.dump(metrics, open("reports/metrics.json", "w"))

    print("Artifacts saved: artifacts/*, reports/val_predictions.csv, reports/topk_contacts.csv, reports/metrics.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
